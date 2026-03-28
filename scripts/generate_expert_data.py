#!/usr/bin/env python3
"""Generate expert game data for Nash-DPO SFT warm-up.

Uses Qwen/Qwen3.5-9B to play 5000 episodes per game (8 simple games),
filters for top-30% reward episodes, and saves (prompt, response) JSONL.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.game_environments_simple import ALL_GAMES, GameEnvironment


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_expert_data")


def parse_args():
    p = argparse.ArgumentParser(description="Generate expert game-play data")
    p.add_argument("--config", type=str, default="configs/agent_roles.yaml")
    p.add_argument("--output_dir", type=str, default="./results/expert_data")
    p.add_argument("--episodes_per_game", type=int, default=5000)
    p.add_argument("--top_fraction", type=float, default=0.3)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--games", type=str, default=None,
                   help="Comma-separated subset of game names to generate")
    return p.parse_args()


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_new_tokens=128, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=1024).to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        temperature=temperature, do_sample=True, top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)


def play_episodes(model, tokenizer, game, num_episodes, max_new_tokens, temperature):
    episodes = []
    for _ in tqdm(range(num_episodes), desc=f"  {game.name}", leave=False):
        for player_id in [0, 1]:
            prompt = game.build_prompt(player_id)
            resp = generate_response(model, tokenizer, prompt, max_new_tokens, temperature)
            opp_prompt = game.build_prompt(1 - player_id)
            opp_resp = generate_response(model, tokenizer, opp_prompt, max_new_tokens, temperature)

            if player_id == 0:
                result = game.play(resp, opp_resp, prompt)
                payoff = result.player1_payoff
            else:
                result = game.play(opp_resp, resp, prompt)
                payoff = result.player2_payoff

            episodes.append({
                "game": game.name,
                "player_id": player_id,
                "prompt": prompt,
                "response": resp,
                "payoff": payoff,
                "action": game.parse_action(resp),
                "is_nash": result.is_nash,
            })
    return episodes


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    model_name = cfg["model"]["base"]
    logger.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True,
    )
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    if args.games:
        selected = {g.strip() for g in args.games.split(",")}
        game_items = {k: v for k, v in ALL_GAMES.items() if k in selected}
        if not game_items:
            logger.error("No valid games in --games=%s (available: %s)",
                         args.games, list(ALL_GAMES.keys()))
            sys.exit(1)
        logger.info("Running subset: %s", list(game_items.keys()))
    else:
        game_items = ALL_GAMES

    all_episodes = []
    per_game_stats = {}

    for game_name, game in game_items.items():
        logger.info("Playing %d episodes of %s", args.episodes_per_game, game_name)
        episodes = play_episodes(model, tokenizer, game, args.episodes_per_game,
                                 args.max_new_tokens, args.temperature)

        episodes.sort(key=lambda e: e["payoff"], reverse=True)
        cutoff = int(len(episodes) * args.top_fraction)
        top_episodes = episodes[:cutoff]

        per_game_stats[game_name] = {
            "total_episodes": len(episodes),
            "kept_episodes": len(top_episodes),
            "mean_payoff_all": sum(e["payoff"] for e in episodes) / max(len(episodes), 1),
            "mean_payoff_top": sum(e["payoff"] for e in top_episodes) / max(len(top_episodes), 1),
            "nash_rate": sum(1 for e in episodes if e["is_nash"]) / max(len(episodes), 1),
        }

        per_game_path = os.path.join(args.output_dir, f"expert_{game_name}.jsonl")
        with open(per_game_path, "w") as f:
            for ep in top_episodes:
                record = {"prompt": ep["prompt"], "response": ep["response"],
                          "game": ep["game"], "payoff": ep["payoff"]}
                f.write(json.dumps(record) + "\n")
        logger.info("  Saved %d records -> %s", len(top_episodes), per_game_path)

        all_episodes.extend(top_episodes)

    import random
    random.seed(args.seed)
    random.shuffle(all_episodes)

    val_size = int(len(all_episodes) * args.val_fraction)
    val_data = all_episodes[:val_size]
    train_data = all_episodes[val_size:]

    for split_name, data in [("train", train_data), ("val", val_data)]:
        path = os.path.join(args.output_dir, f"expert_{split_name}.jsonl")
        with open(path, "w") as f:
            for ep in data:
                record = {"prompt": ep["prompt"], "response": ep["response"],
                          "game": ep["game"], "payoff": ep["payoff"]}
                f.write(json.dumps(record) + "\n")
        logger.info("Saved %s: %d records -> %s", split_name, len(data), path)

    stats_path = os.path.join(args.output_dir, "generation_stats.json")
    with open(stats_path, "w") as f:
        json.dump({"total_kept": len(all_episodes), "train_size": len(train_data),
                    "val_size": len(val_data), "per_game": per_game_stats}, f, indent=2)

    logger.info("Expert data generation complete! Train: %d  Val: %d",
                len(train_data), len(val_data))


if __name__ == "__main__":
    main()
