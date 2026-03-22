#!/usr/bin/env python3
"""Complete self-play + GRPO training loop.

Load SFT-warmed agents, then for each GRPO iteration:
  1. Generate 10K self-play episodes across all 8 games
  2. Compute rewards (payoff + strategy quality + Nash proximity)
  3. Run GRPO policy update on each agent
  4. Evaluate all agents on all games

Tracks: avg payoff per game, strategy diversity (action entropy), Nash distance.

Usage:
    python scripts/run_grpo_self_play.py --sft_dir results/sft_agents
    torchrun --nproc_per_node=4 scripts/run_grpo_self_play.py ...
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.game_environments import create_environment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

AGENT_GAME_ASSIGNMENTS = {
    "agent_0": ["prisoners_dilemma", "coordination_game"],
    "agent_1": ["battle_of_sexes", "stag_hunt"],
    "agent_2": ["public_goods", "auction"],
    "agent_3": ["ultimatum", "negotiation"],
}

NASH_EQUILIBRIA = {
    "prisoners_dilemma": {"defect": 1.0},
    "coordination_game": {"A": 0.5, "B": 0.5},
    "battle_of_sexes": {"opera": 0.5, "football": 0.5},
    "stag_hunt": {"stag": 0.5, "hare": 0.5},
    "public_goods": {"contribute": 0.5, "free_ride": 0.5},
    "ultimatum": {"accept": 0.7, "reject": 0.3},
    "auction": {},
    "negotiation": {"agree": 0.6, "propose_medium": 0.2, "propose_low": 0.1, "propose_high": 0.1},
}


def format_decision_prompt(game_prompt: str, action_space: list) -> str:
    return (
        f"<|im_start|>system\nYou are a strategic decision-making agent. "
        f"Analyze the situation carefully and choose the optimal action. "
        f"Consider both immediate and long-term consequences.<|im_end|>\n"
        f"<|im_start|>user\n{game_prompt}\n\n"
        f"Choose exactly ONE action from: {', '.join(action_space)}\n"
        f"Format: ACTION: <your_choice>\nREASONING: <brief explanation><|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_action(response: str, valid_actions: list) -> str:
    response_lower = response.lower()
    if "action:" in response_lower:
        after_action = response_lower.split("action:")[-1].strip()
        for action in valid_actions:
            if action.lower() in after_action[:50]:
                return action
    for action in valid_actions:
        if action.lower() in response_lower:
            return action
    return random.choice(valid_actions)


def get_player_ids(scenario_name: str, scenario_cfg: dict) -> list:
    game_type = scenario_cfg.get("type", "")
    if "n_player" in game_type:
        return [f"player_{i}" for i in range(scenario_cfg.get("num_players", 4))]
    elif "sequential" in game_type:
        return scenario_cfg.get("roles", ["party_A", "party_B"])
    return ["player_0", "player_1"]


def compute_strategy_diversity(action_counts: dict) -> float:
    """Shannon entropy of action distribution."""
    total = sum(action_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in action_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p + 1e-10)
    return entropy


def compute_nash_distance(action_counts: dict, scenario_name: str) -> float:
    """L2 distance between empirical action distribution and reference Nash equilibrium."""
    nash = NASH_EQUILIBRIA.get(scenario_name, {})
    if not nash:
        return 0.0
    total = sum(action_counts.values())
    if total == 0:
        return 0.0
    empirical = {a: c / total for a, c in action_counts.items()}
    dist = 0.0
    all_actions = set(list(nash.keys()) + list(empirical.keys()))
    for action in all_actions:
        dist += (empirical.get(action, 0.0) - nash.get(action, 0.0)) ** 2
    return dist ** 0.5


def load_agent(agent_name: str, sft_dir: str, model_name: str, device):
    """Load an SFT-warmed LoRA agent."""
    agent_path = Path(sft_dir) / agent_name / "final"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="flash_attention_2",
        device_map={"": device},
    )
    base_model.config.use_cache = False

    if agent_path.exists() and (agent_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base_model, str(agent_path), is_trainable=True)
        logger.info("Loaded LoRA adapter from %s", agent_path)
    else:
        from peft import LoraConfig, TaskType, get_peft_model
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM, bias="none",
        )
        model = get_peft_model(base_model, peft_config)
        logger.warning("No SFT checkpoint at %s, using fresh LoRA", agent_path)

    return model, tokenizer


def play_self_play_episode(agents, tokenizers, env, player_ids, scenario_name, device, temperature=0.7):
    """Play one episode with different agents as players."""
    state = env.reset()
    trajectories = {pid: [] for pid in player_ids}

    agent_names = list(agents.keys())
    player_to_agent = {}
    for i, pid in enumerate(player_ids):
        assigned = [aname for aname, games in AGENT_GAME_ASSIGNMENTS.items() if scenario_name in games]
        if assigned:
            player_to_agent[pid] = assigned[i % len(assigned)]
        else:
            player_to_agent[pid] = agent_names[i % len(agent_names)]

    while not state.done:
        actions = {}
        for pid in player_ids:
            agent_name = player_to_agent[pid]
            model = agents[agent_name]
            tokenizer = tokenizers[agent_name]

            action_space = env.get_action_space(pid)
            game_prompt = env.get_prompt(state, pid)
            prompt = format_decision_prompt(game_prompt, action_space)

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=80, do_sample=True,
                    temperature=temperature, top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            action = parse_action(response, action_space)
            actions[pid] = action

            action_text = f"ACTION: {action}"
            action_inputs = tokenizer(
                prompt + action_text, return_tensors="pt", truncation=True, max_length=1024
            ).to(device)
            with torch.no_grad():
                logits = model(**action_inputs).logits
            prompt_len = inputs["input_ids"].shape[1]
            action_logits = logits[0, prompt_len - 1:-1]
            action_token_ids = action_inputs["input_ids"][0, prompt_len:]
            log_probs = F.log_softmax(action_logits, dim=-1)
            token_log_probs = log_probs.gather(1, action_token_ids.unsqueeze(1)).squeeze(1)
            avg_log_prob = token_log_probs.mean().item()

            trajectories[pid].append({
                "prompt": prompt,
                "response": response,
                "action": action,
                "log_prob": avg_log_prob,
                "round": state.round_num,
                "agent_name": agent_name,
            })

        state, rewards = env.step(actions)
        for pid in player_ids:
            if trajectories[pid]:
                trajectories[pid][-1]["reward"] = rewards.get(pid, 0.0)

    return trajectories, state


def grpo_update_agent(model, tokenizer, agent_trajectories, optimizer, grpo_cfg, device):
    """Apply GRPO update to a single agent using its collected trajectories."""
    clip_range = grpo_cfg["clip_range"]
    kl_coeff = grpo_cfg["kl_coeff"]
    gamma = grpo_cfg["gamma"]

    total_loss = 0.0
    num_updates = 0

    for traj in agent_trajectories:
        if len(traj) < 2:
            continue

        rewards = torch.tensor([t["reward"] for t in traj], device=device)
        returns = torch.zeros_like(rewards)
        G = 0.0
        for i in reversed(range(len(rewards))):
            G = rewards[i] + gamma * G
            returns[i] = G

        mean_r = returns.mean()
        std_r = returns.std().clamp(min=1e-8)
        advantages = (returns - mean_r) / std_r

        for step_data, adv in zip(traj, advantages):
            prompt = step_data["prompt"]
            old_log_prob = step_data["log_prob"]

            inputs = tokenizer(
                prompt + f"ACTION: {step_data['action']}",
                return_tensors="pt", truncation=True, max_length=1024,
            ).to(device)
            labels = inputs["input_ids"]

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(**inputs, labels=labels)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                mask = (shift_labels != -100).float()
                new_log_prob = (token_log_probs * mask).sum(-1) / mask.sum(-1).clamp(min=1)

            ratio = torch.exp(new_log_prob - old_log_prob)
            clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -torch.min(ratio * adv, clipped * adv)
            kl = kl_coeff * (old_log_prob - new_log_prob)
            loss = policy_loss + kl

            loss.backward()
            total_loss += loss.item()
            num_updates += 1

    if num_updates > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    return {"loss": total_loss / max(num_updates, 1), "num_updates": num_updates}


def evaluate_agents(agents, tokenizers, config, device, num_episodes=20):
    """Evaluate all agents across all games."""
    results = {}
    scenario_names = list(config["scenarios"].keys())

    for scenario_name in scenario_names:
        scenario_cfg = config["scenarios"][scenario_name]
        player_ids = get_player_ids(scenario_name, scenario_cfg)
        action_counts = Counter()
        total_payoffs = defaultdict(float)

        for _ in range(num_episodes):
            env = create_environment(scenario_name, config)
            _, final_state = play_self_play_episode(
                agents, tokenizers, env, player_ids, scenario_name, device, temperature=0.3,
            )
            for pid in player_ids:
                total_payoffs[pid] += final_state.scores.get(pid, 0.0)
            for h in final_state.history:
                for pid, action in h.get("actions", {}).items():
                    action_counts[action] += 1

        avg_payoffs = {pid: v / num_episodes for pid, v in total_payoffs.items()}
        diversity = compute_strategy_diversity(dict(action_counts))
        nash_dist = compute_nash_distance(dict(action_counts), scenario_name)

        results[scenario_name] = {
            "avg_payoffs": avg_payoffs,
            "strategy_diversity": diversity,
            "nash_distance": nash_dist,
            "action_distribution": dict(action_counts),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="GRPO self-play training loop")
    parser.add_argument("--config", type=str, default="configs/game_scenarios.yaml")
    parser.add_argument("--sft_dir", type=str, default="results/sft_agents")
    parser.add_argument("--output_dir", type=str, default="results/grpo_self_play")
    parser.add_argument("--num_iterations", type=int, default=5)
    parser.add_argument("--episodes_per_iter", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config["model"]["base_model"]
    grpo_cfg = config["grpo"]
    scenario_names = list(config["scenarios"].keys())
    episodes_per_game = args.episodes_per_iter // len(scenario_names)

    logger.info("=== GRPO Self-Play Training ===")
    logger.info("Iterations: %d, Episodes/iter: %d, Games: %d",
                args.num_iterations, args.episodes_per_iter, len(scenario_names))

    agents = {}
    tokenizers = {}
    optimizers = {}

    for agent_name in AGENT_GAME_ASSIGNMENTS:
        model, tokenizer = load_agent(agent_name, args.sft_dir, model_name, device)
        agents[agent_name] = model
        tokenizers[agent_name] = tokenizer
        optimizers[agent_name] = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
        )

    training_log = []

    for iteration in range(args.num_iterations):
        logger.info("=== Iteration %d/%d ===", iteration + 1, args.num_iterations)
        iter_dir = output_dir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        agent_trajectories = {name: [] for name in agents}
        iter_payoffs = defaultdict(list)

        for scenario_name in scenario_names:
            logger.info("  Generating self-play episodes for %s...", scenario_name)
            scenario_cfg = config["scenarios"][scenario_name]
            player_ids = get_player_ids(scenario_name, scenario_cfg)

            for ep_idx in tqdm(range(episodes_per_game), desc=scenario_name, leave=False):
                env = create_environment(scenario_name, config)
                trajectories, final_state = play_self_play_episode(
                    agents, tokenizers, env, player_ids, scenario_name,
                    device, args.temperature,
                )

                for pid, traj in trajectories.items():
                    agent_name = traj[0]["agent_name"] if traj else None
                    if agent_name and agent_name in agent_trajectories:
                        agent_trajectories[agent_name].append(traj)

                total_payoff = sum(final_state.scores.values())
                iter_payoffs[scenario_name].append(total_payoff)

                if (ep_idx + 1) % 50 == 0:
                    torch.cuda.empty_cache()

        logger.info("  Running GRPO updates...")
        update_stats = {}
        for agent_name, model in agents.items():
            model.train()
            stats = grpo_update_agent(
                model, tokenizers[agent_name],
                agent_trajectories[agent_name],
                optimizers[agent_name], grpo_cfg, device,
            )
            update_stats[agent_name] = stats
            logger.info("    %s: loss=%.4f, updates=%d",
                         agent_name, stats["loss"], stats["num_updates"])

        logger.info("  Evaluating agents...")
        eval_results = evaluate_agents(
            agents, tokenizers, config, device, args.eval_episodes,
        )

        iter_log = {
            "iteration": iteration,
            "avg_payoffs_per_game": {
                g: sum(p) / len(p) for g, p in iter_payoffs.items()
            },
            "update_stats": update_stats,
            "eval_results": eval_results,
        }
        training_log.append(iter_log)

        with open(iter_dir / "iter_results.json", "w") as f:
            json.dump(iter_log, f, indent=2, default=str)

        for agent_name, model in agents.items():
            ckpt_path = iter_dir / agent_name
            model.save_pretrained(str(ckpt_path))
            tokenizers[agent_name].save_pretrained(str(ckpt_path))

        logger.info("  Iteration %d summary:", iteration)
        for game, payoffs in iter_payoffs.items():
            avg_p = sum(payoffs) / len(payoffs)
            div = eval_results.get(game, {}).get("strategy_diversity", 0)
            nash = eval_results.get(game, {}).get("nash_distance", 0)
            logger.info("    %s: avg_payoff=%.2f, diversity=%.3f, nash_dist=%.3f",
                         game, avg_p, div, nash)

    for agent_name, model in agents.items():
        final_path = output_dir / "final" / agent_name
        model.save_pretrained(str(final_path))
        tokenizers[agent_name].save_pretrained(str(final_path))

    with open(output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2, default=str)

    logger.info("=== GRPO Self-Play Training Complete ===")
    logger.info("Final models saved to %s/final/", output_dir)
    logger.info("Training log saved to %s/training_log.json", output_dir)


if __name__ == "__main__":
    main()
