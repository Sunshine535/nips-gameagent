#!/usr/bin/env python3
"""
Game performance evaluation.

Evaluates trained agents on all 8 games (100 episodes each).
Computes: average score, Nash equilibrium distance, exploitation rate.
Compares: baseline, after SFT, after Nash-DPO iter 1/2/3.
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
from src.game_environments import ALL_GAMES, list_games

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_game_performance")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate game performance")
    p.add_argument("--config", type=str, default="configs/agent_roles.yaml")
    p.add_argument("--model_paths", nargs="+", required=True,
                   help="label:path pairs, e.g. 'baseline:Qwen/Qwen3.5-9B'")
    p.add_argument("--output_dir", type=str, default="./results/game_eval")
    p.add_argument("--episodes_per_game", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(path, base_model=None):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if base_model and base_model != path:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        try:
            model = PeftModel.from_pretrained(model, path)
        except Exception:
            pass
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
    model.eval()
    return model, tok


@torch.no_grad()
def generate_response(model, tok, prompt, max_new_tokens=128):
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=1024).to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(outputs[0, inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True)


def evaluate_on_game(model, tok, game, num_episodes, max_new_tokens):
    """Evaluate a model on a single game by self-play."""
    results = []
    total_payoff_p1 = 0.0
    total_payoff_p2 = 0.0
    nash_count = 0
    nash_distances = []

    for _ in tqdm(range(num_episodes), desc=f"  {game.name}", leave=False):
        p1_prompt = game.build_prompt(0)
        p2_prompt = game.build_prompt(1)

        r1 = generate_response(model, tok, p1_prompt, max_new_tokens)
        r2 = generate_response(model, tok, p2_prompt, max_new_tokens)

        result = game.play(r1, r2, p1_prompt)
        total_payoff_p1 += result.player1_payoff
        total_payoff_p2 += result.player2_payoff
        if result.is_nash:
            nash_count += 1

        a1 = game.parse_action(r1)
        a2 = game.parse_action(r2)
        nash_distances.append(game.nash_distance(a1, a2))

        results.append({
            "p1_action": result.player1_action,
            "p2_action": result.player2_action,
            "p1_payoff": result.player1_payoff,
            "p2_payoff": result.player2_payoff,
            "is_nash": result.is_nash,
        })

    n = max(num_episodes, 1)
    action_counts = defaultdict(int)
    for r in results:
        action_counts[r["p1_action"]] += 1
        action_counts[r["p2_action"]] += 1

    return {
        "avg_payoff_p1": total_payoff_p1 / n,
        "avg_payoff_p2": total_payoff_p2 / n,
        "avg_payoff": (total_payoff_p1 + total_payoff_p2) / (2 * n),
        "nash_rate": nash_count / n,
        "avg_nash_distance": sum(nash_distances) / n if nash_distances else 0,
        "action_distribution": dict(action_counts),
        "num_episodes": num_episodes,
    }


def compute_exploitation_rate(game_results):
    """Fraction of games where one player gets max payoff at other's expense."""
    exploited = 0
    total = 0
    for game_name, result in game_results.items():
        game = ALL_GAMES.get(game_name)
        if game is None:
            continue
        max_payoff = max(p1 for (p1, _) in game.payoff_matrix.values())
        if result["avg_payoff_p1"] > 0.8 * max_payoff or result["avg_payoff_p2"] > 0.8 * max_payoff:
            if abs(result["avg_payoff_p1"] - result["avg_payoff_p2"]) > 1.0:
                exploited += 1
        total += 1
    return exploited / max(total, 1)


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    base_model = args.base_model or cfg["model"]["base"]
    all_results = {}

    for model_spec in args.model_paths:
        if ":" in model_spec:
            label, path = model_spec.split(":", 1)
        else:
            label, path = os.path.basename(model_spec), model_spec

        logger.info("\n" + "=" * 60)
        logger.info("Evaluating: %s (%s)", label, path)
        logger.info("=" * 60)

        model, tok = load_model(path, base_model)
        model_results = {}

        for game_name, game in ALL_GAMES.items():
            logger.info("  Game: %s", game_name)
            game_result = evaluate_on_game(model, tok, game,
                                           args.episodes_per_game,
                                           args.max_new_tokens)
            model_results[game_name] = game_result
            logger.info("    Avg payoff: %.2f  Nash rate: %.2f  Nash dist: %.2f",
                        game_result["avg_payoff"],
                        game_result["nash_rate"],
                        game_result["avg_nash_distance"])

        overall_payoff = sum(r["avg_payoff"] for r in model_results.values()) / max(len(model_results), 1)
        overall_nash = sum(r["nash_rate"] for r in model_results.values()) / max(len(model_results), 1)
        exploit_rate = compute_exploitation_rate(model_results)

        model_results["_summary"] = {
            "avg_payoff": overall_payoff,
            "avg_nash_rate": overall_nash,
            "exploitation_rate": exploit_rate,
        }

        all_results[label] = model_results
        logger.info("  OVERALL: payoff=%.2f  nash_rate=%.2f  exploit=%.2f",
                    overall_payoff, overall_nash, exploit_rate)

        del model
        torch.cuda.empty_cache()

    out_path = os.path.join(args.output_dir, "game_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print comparison table
    labels = list(all_results.keys())
    print("\n" + "=" * 90)
    header = f"{'Model':>15s} {'AvgPayoff':>10s} {'NashRate':>10s} {'ExploitRate':>12s}"
    for gn in list(ALL_GAMES.keys())[:4]:
        header += f" {gn[:8]:>9s}"
    print(header)
    print("-" * 90)
    for label in labels:
        s = all_results[label].get("_summary", {})
        row = f"{label:>15s} {s.get('avg_payoff', 0):>10.3f} {s.get('avg_nash_rate', 0):>10.3f} {s.get('exploitation_rate', 0):>12.3f}"
        for gn in list(ALL_GAMES.keys())[:4]:
            v = all_results[label].get(gn, {}).get("avg_payoff", 0)
            row += f" {v:>9.2f}"
        print(row)
    print("=" * 90)
    logger.info("Results: %s", out_path)


if __name__ == "__main__":
    main()
