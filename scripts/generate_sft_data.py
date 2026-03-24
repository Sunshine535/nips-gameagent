#!/usr/bin/env python3
"""Generate SFT warmup data by playing episodes with a base LLM and filtering top performers.

For each of 8 games, generate episodes using Qwen/Qwen3.5-9B, filter the top 30% by
cumulative payoff, then format surviving trajectories as chat-style training data.

Usage:
    python scripts/generate_sft_data.py --config configs/game_scenarios.yaml
    python scripts/generate_sft_data.py --episodes_per_game 5000 --top_fraction 0.3
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.game_environments import create_environment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SYSTEM_PROMPT = (
    "You are a strategic decision-making agent. "
    "Analyze the situation carefully and choose the optimal action. "
    "Consider both immediate and long-term consequences."
)


def format_decision_prompt(game_prompt: str, action_space: list) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
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
        n = scenario_cfg.get("num_players", 4)
        return [f"player_{i}" for i in range(n)]
    elif "sequential" in game_type:
        return scenario_cfg.get("roles", ["party_A", "party_B"])
    return ["player_0", "player_1"]


def play_episode_for_data(model, tokenizer, env, player_ids, device, temperature=0.8):
    """Play one episode, collecting full trajectory data for SFT."""
    state = env.reset()
    trajectory = []

    while not state.done:
        round_actions = {}
        round_data = []

        for pid in player_ids:
            action_space = env.get_action_space(pid)
            game_prompt = env.get_prompt(state, pid)
            prompt = format_decision_prompt(game_prompt, action_space)

            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=80, do_sample=True,
                    temperature=temperature, top_p=0.9,
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            action = parse_action(response, action_space)
            round_actions[pid] = action
            round_data.append({
                "player_id": pid,
                "game_prompt": game_prompt,
                "action_space": action_space,
                "response": response.strip(),
                "action": action,
            })

        state, rewards = env.step(round_actions)
        for rd in round_data:
            rd["reward"] = rewards.get(rd["player_id"], 0.0)
        trajectory.append(round_data)

    total_payoff = sum(state.scores.values())
    return trajectory, state, total_payoff


def trajectory_to_sft_samples(trajectory: list, game_name: str) -> list:
    """Convert a trajectory into chat-style SFT training samples."""
    samples = []
    for round_data in trajectory:
        for step in round_data:
            user_msg = (
                f"Game: {game_name}\n{step['game_prompt']}\n\n"
                f"Choose exactly ONE action from: {', '.join(step['action_space'])}\n"
                f"Format: ACTION: <your_choice>\nREASONING: <brief explanation>"
            )
            assistant_msg = step["response"]
            if not assistant_msg.strip():
                assistant_msg = f"ACTION: {step['action']}\nREASONING: Based on strategic analysis."

            samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                "game": game_name,
                "action": step["action"],
                "reward": step["reward"],
            })
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate SFT warmup data")
    parser.add_argument("--config", type=str, default="configs/game_scenarios.yaml")
    parser.add_argument("--episodes_per_game", type=int, default=5000)
    parser.add_argument("--top_fraction", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_episodes", type=int, default=50,
                        help="Episodes between GPU cache clears")
    parser.add_argument("--games", type=str, default=None,
                        help="Comma-separated subset of scenario names to generate")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config["model"]["base_model"]
    logger.info("Loading base model: %s on %s", model_name, device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map=device,
    )
    model.eval()
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    all_scenarios = list(config["scenarios"].keys())
    if args.games:
        scenario_names = [g.strip() for g in args.games.split(",")
                          if g.strip() in config["scenarios"]]
        if not scenario_names:
            logger.error("No valid games found in --games=%s (available: %s)",
                         args.games, all_scenarios)
            sys.exit(1)
        logger.info("Running subset: %s", scenario_names)
    else:
        scenario_names = all_scenarios
    global_stats = {}

    for scenario_name in scenario_names:
        logger.info("=== Generating data for: %s (%d episodes) ===",
                     scenario_name, args.episodes_per_game)
        scenario_cfg = config["scenarios"][scenario_name]
        player_ids = get_player_ids(scenario_name, scenario_cfg)

        episode_data = []
        for ep_idx in tqdm(range(args.episodes_per_game), desc=scenario_name):
            env = create_environment(scenario_name, config)
            traj, final_state, total_payoff = play_episode_for_data(
                model, tokenizer, env, player_ids, device, args.temperature,
            )
            episode_data.append({
                "trajectory": traj,
                "total_payoff": total_payoff,
                "final_scores": dict(final_state.scores),
            })
            if (ep_idx + 1) % args.batch_episodes == 0:
                torch.cuda.empty_cache()

        episode_data.sort(key=lambda x: x["total_payoff"], reverse=True)
        cutoff = max(1, int(len(episode_data) * args.top_fraction))
        top_episodes = episode_data[:cutoff]

        logger.info("Filtered %d → %d episodes (top %.0f%%, min payoff: %.2f)",
                     len(episode_data), len(top_episodes), args.top_fraction * 100,
                     top_episodes[-1]["total_payoff"] if top_episodes else 0)

        sft_samples = []
        for ep in top_episodes:
            sft_samples.extend(
                trajectory_to_sft_samples(ep["trajectory"], scenario_cfg["name"])
            )

        out_path = output_dir / f"sft_{scenario_name}.jsonl"
        with open(out_path, "w") as f:
            for sample in sft_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        payoffs = [ep["total_payoff"] for ep in episode_data]
        filtered_payoffs = [ep["total_payoff"] for ep in top_episodes]
        global_stats[scenario_name] = {
            "total_episodes": len(episode_data),
            "filtered_episodes": len(top_episodes),
            "sft_samples": len(sft_samples),
            "mean_payoff_all": sum(payoffs) / len(payoffs),
            "mean_payoff_filtered": sum(filtered_payoffs) / len(filtered_payoffs),
            "min_payoff_filtered": min(filtered_payoffs),
            "max_payoff_filtered": max(filtered_payoffs),
        }
        logger.info("  Saved %d SFT samples to %s", len(sft_samples), out_path)

    stats_path = output_dir / "sft_generation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(global_stats, f, indent=2)

    total_samples = sum(s["sft_samples"] for s in global_stats.values())
    logger.info("=== SFT Data Generation Complete ===")
    logger.info("Total SFT samples: %d across %d games", total_samples, len(scenario_names))
    logger.info("Stats saved to %s", stats_path)


if __name__ == "__main__":
    main()
