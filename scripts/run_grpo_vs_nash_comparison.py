#!/usr/bin/env python3
"""Cross-comparison: GRPO self-play vs Nash-DPO self-play.

Evaluates both training paradigms across ALL benchmarks (strategic reasoning
+ alignment), producing head-to-head comparison tables and analysis.

This is the key NEW experiment that unifies the two approaches and
demonstrates the complementary strengths of GRPO (strategic diversity)
and Nash-DPO (multi-objective Pareto optimality).
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.game_environments import create_environment
from src.game_environments_simple import ALL_GAMES
from src.game_protocol import compute_agent_reward, AgentRole

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REWARD_DIMS = {
    "correctness": {"correctness": 1.0, "safety": 0.0, "efficiency": 0.0, "creativity": 0.0},
    "safety": {"correctness": 0.0, "safety": 1.0, "efficiency": 0.0, "creativity": 0.0},
    "efficiency": {"correctness": 0.0, "safety": 0.0, "efficiency": 1.0, "creativity": 0.0},
    "creativity": {"correctness": 0.0, "safety": 0.0, "efficiency": 0.0, "creativity": 1.0},
}


def load_model(path, base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, path)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=1024).to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)


def evaluate_game_performance(model, tokenizer, config, num_episodes=50):
    """Evaluate on multi-round game environments (GRPO-style)."""
    results = {}
    for scenario_name in config["scenarios"]:
        env = create_environment(scenario_name, config)
        scenario_cfg = config["scenarios"][scenario_name]

        game_type = scenario_cfg.get("type", "")
        if "n_player" in game_type:
            n = scenario_cfg.get("num_players", 4)
            player_ids = [f"player_{i}" for i in range(n)]
        elif "sequential" in game_type:
            player_ids = scenario_cfg.get("roles", ["party_A", "party_B"])
        else:
            player_ids = ["player_0", "player_1"]

        total_scores = {pid: 0.0 for pid in player_ids}
        ep_count = min(num_episodes, 20)

        for _ in range(ep_count):
            state = env.reset()
            while not state.done:
                actions = {}
                for pid in player_ids:
                    action_space = env.get_action_space(pid)
                    game_prompt = env.get_prompt(state, pid)
                    prompt = (
                        f"<|im_start|>system\nYou are a strategic agent.<|im_end|>\n"
                        f"<|im_start|>user\n{game_prompt}\nChoose: {', '.join(action_space)}<|im_end|>\n"
                        f"<|im_start|>assistant\nACTION: "
                    )
                    response = generate_response(model, tokenizer, prompt, max_new_tokens=50)
                    action = response.strip().split()[0] if response.strip() else action_space[0]
                    for valid in action_space:
                        if valid.lower() in action.lower():
                            action = valid
                            break
                    else:
                        import random
                        action = random.choice(action_space)
                    actions[pid] = action
                state, _ = env.step(actions)

            for pid in player_ids:
                total_scores[pid] += state.scores.get(pid, 0)

        avg_scores = {pid: s / ep_count for pid, s in total_scores.items()}
        overall = sum(avg_scores.values()) / max(len(avg_scores), 1)
        results[scenario_name] = {"avg_payoff": overall, "per_player": avg_scores}

    return results


def evaluate_simple_games(model, tokenizer, num_episodes=50):
    """Evaluate on single-round games (Nash-DPO style)."""
    results = {}
    for game_name, game in ALL_GAMES.items():
        nash_count = 0
        total_payoff = 0.0

        for _ in range(num_episodes):
            r1 = generate_response(model, tokenizer, game.build_prompt(0))
            r2 = generate_response(model, tokenizer, game.build_prompt(1))
            result = game.play(r1, r2)
            total_payoff += result.player1_payoff + result.player2_payoff
            if result.is_nash:
                nash_count += 1

        results[game_name] = {
            "avg_payoff": total_payoff / (2 * num_episodes),
            "nash_rate": nash_count / num_episodes,
        }

    return results


def evaluate_multi_objective(model, tokenizer, prompts, num_prompts=100):
    """Evaluate response quality across correctness/safety/efficiency/creativity."""
    dim_scores = defaultdict(list)

    for prompt in prompts[:num_prompts]:
        response = generate_response(model, tokenizer, prompt)
        for dim_name, weights in REWARD_DIMS.items():
            score = compute_agent_reward(response, weights)
            dim_scores[dim_name].append(score)

    return {dim: sum(scores) / len(scores) for dim, scores in dim_scores.items()}


def main():
    parser = argparse.ArgumentParser(description="GRPO vs Nash-DPO cross-comparison")
    parser.add_argument("--game_config", type=str, default="configs/game_scenarios.yaml")
    parser.add_argument("--grpo_model", type=str, default="results/grpo_self_play/final/agent_0")
    parser.add_argument("--nash_model", type=str, default="results/nash_dpo/iter2/accuracy")
    parser.add_argument("--baseline_model", type=str, default=None,
                        help="Base model path for comparison (default: use base from config)")
    parser.add_argument("--output_dir", type=str, default="results/grpo_vs_nash")
    parser.add_argument("--game_episodes", type=int, default=50)
    parser.add_argument("--mo_prompts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    with open(args.game_config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    base_model = args.baseline_model or config["model"]["base_model"]

    mo_prompts = [
        "Explain quantum entanglement simply.", "Write Python code for binary search.",
        "What are the benefits of exercise?", "Compare renewable and fossil energy.",
        "How does the immune system work?", "Write a story about space exploration.",
        "Solve: If 2x + 5 = 15, what is x?", "Explain machine learning to a child.",
    ] * (args.mo_prompts // 8 + 1)

    model_specs = {
        "baseline": base_model,
        "grpo": args.grpo_model,
        "nash_dpo": args.nash_model,
    }

    all_results = {}

    for label, path in model_specs.items():
        logger.info("\n=== Evaluating: %s ===", label)
        model, tokenizer = load_model(path, base_model)

        logger.info("  Multi-round game evaluation...")
        game_results = evaluate_game_performance(model, tokenizer, config, args.game_episodes)

        logger.info("  Single-round game evaluation...")
        simple_results = evaluate_simple_games(model, tokenizer, args.game_episodes)

        logger.info("  Multi-objective evaluation...")
        mo_results = evaluate_multi_objective(model, tokenizer, mo_prompts, args.mo_prompts)

        all_results[label] = {
            "multi_round_games": game_results,
            "single_round_games": simple_results,
            "multi_objective": mo_results,
        }

        del model
        torch.cuda.empty_cache()

    output_path = os.path.join(args.output_dir, "grpo_vs_nash_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print comparison tables
    print("\n" + "=" * 90)
    print("GRPO vs Nash-DPO: Multi-Round Game Performance")
    print("-" * 90)
    scenarios = list(config["scenarios"].keys())
    header = f"{'Game':>20}"
    for label in all_results:
        header += f" {label:>15}"
    print(header)
    for sc in scenarios:
        row = f"{sc:>20}"
        for label in all_results:
            v = all_results[label]["multi_round_games"].get(sc, {}).get("avg_payoff", 0)
            row += f" {v:>15.3f}"
        print(row)

    print("\n" + "=" * 90)
    print("GRPO vs Nash-DPO: Multi-Objective Alignment")
    print("-" * 90)
    header = f"{'Dimension':>15}"
    for label in all_results:
        header += f" {label:>15}"
    print(header)
    for dim in ["correctness", "safety", "efficiency", "creativity"]:
        row = f"{dim:>15}"
        for label in all_results:
            v = all_results[label]["multi_objective"].get(dim, 0)
            row += f" {v:>15.4f}"
        print(row)
    print("=" * 90)

    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
