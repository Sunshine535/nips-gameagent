#!/usr/bin/env python3
"""Cross-game transfer evaluation.

Trains agents on a subset of games and evaluates on held-out games to measure
transfer learning. Three training regimes are compared:

  1. Single-game: train on each held-out game individually (baseline)
  2. Multi-game:  train on 4 games simultaneously, evaluate on the other 4
  3. Curriculum:  train on easy games first, then harder ones, eval on held-out

Split:
  Train set: prisoners_dilemma, coordination_game, public_goods, auction
  Held-out:  battle_of_sexes, stag_hunt, ultimatum, negotiation

Usage:
    python scripts/run_cross_game_transfer.py --config configs/game_scenarios.yaml
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.game_environments import create_environment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


TRAIN_GAMES = ["prisoners_dilemma", "coordination_game", "public_goods", "auction"]
HELDOUT_GAMES = ["battle_of_sexes", "stag_hunt", "ultimatum", "negotiation"]

CURRICULUM_ORDER = [
    ["prisoners_dilemma", "coordination_game"],
    ["public_goods", "auction"],
    ["battle_of_sexes", "stag_hunt"],
    ["ultimatum", "negotiation"],
]


def format_decision_prompt(game_prompt: str, action_space: list) -> str:
    return (
        f"<|im_start|>system\nYou are a strategic decision-making agent. "
        f"Analyze the situation carefully and choose the optimal action.<|im_end|>\n"
        f"<|im_start|>user\n{game_prompt}\n\n"
        f"Choose exactly ONE action from: {', '.join(action_space)}<|im_end|>\n"
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


def load_sft_data_for_games(data_dir: str, game_names: list) -> Dataset:
    """Load SFT JSONL data for given game names."""
    texts = []
    for game in game_names:
        path = Path(data_dir) / f"sft_{game}.jsonl"
        if not path.exists():
            logger.warning("Missing SFT data: %s", path)
            continue
        with open(path) as f:
            for line in f:
                sample = json.loads(line)
                msgs = sample["messages"]
                text = ""
                for msg in msgs:
                    text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                texts.append(text)
    if not texts:
        raise FileNotFoundError(f"No SFT data for games {game_names} in {data_dir}")
    return Dataset.from_dict({"text": texts})


def train_agent_on_games(
    model_name: str,
    game_names: list,
    data_dir: str,
    output_dir: str,
    num_epochs: int = 2,
    batch_size: int = 4,
    base_model=None,
):
    """Train a LoRA agent on specified games via SFT."""
    dataset = load_sft_data_for_games(data_dir, game_names)
    logger.info("Training on %d samples from games: %s", len(dataset), game_names)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if base_model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
    else:
        model = base_model

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=20,
        save_strategy="no",
        max_length=2048,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    del trainer
    torch.cuda.empty_cache()
    return model, tokenizer


def evaluate_on_games(model, tokenizer, config, game_names, device, num_episodes=20):
    """Evaluate model across specified games. Returns per-game metrics."""
    model.eval()
    results = {}

    for scenario_name in game_names:
        scenario_cfg = config["scenarios"][scenario_name]
        player_ids = get_player_ids(scenario_name, scenario_cfg)
        action_counts = Counter()
        total_payoffs = defaultdict(float)

        for _ in range(num_episodes):
            env = create_environment(scenario_name, config)
            state = env.reset()

            while not state.done:
                actions = {}
                for pid in player_ids:
                    action_space = env.get_action_space(pid)
                    game_prompt = env.get_prompt(state, pid)
                    prompt = format_decision_prompt(game_prompt, action_space)

                    inputs = tokenizer(
                        prompt, return_tensors="pt", truncation=True, max_length=1024
                    ).to(device)
                    with torch.no_grad():
                        output = model.generate(
                            **inputs, max_new_tokens=100, do_sample=False,
                        )
                    response = tokenizer.decode(
                        output[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    action = parse_action(response, action_space)
                    actions[pid] = action
                    action_counts[action] += 1

                state, _ = env.step(actions)

            for pid in player_ids:
                total_payoffs[pid] += state.scores.get(pid, 0.0)

        avg_payoffs = {pid: v / num_episodes for pid, v in total_payoffs.items()}
        overall_avg = sum(avg_payoffs.values()) / max(len(avg_payoffs), 1)

        results[scenario_name] = {
            "avg_payoff": overall_avg,
            "per_player_payoffs": avg_payoffs,
            "action_distribution": dict(action_counts),
            "num_episodes": num_episodes,
        }
        logger.info("  %s: avg_payoff=%.3f", scenario_name, overall_avg)

    return results


def run_single_game_baseline(model_name, data_dir, output_dir, config, device, args):
    """Train and evaluate one agent per held-out game (baseline)."""
    logger.info("=== Single-Game Baseline ===")
    results = {}
    for game in HELDOUT_GAMES:
        logger.info("Training single-game agent for: %s", game)
        agent_dir = str(Path(output_dir) / "single_game" / game)
        model, tokenizer = train_agent_on_games(
            model_name, [game], data_dir, agent_dir, num_epochs=args.num_epochs,
        )
        eval_res = evaluate_on_games(model, tokenizer, config, [game], device, args.eval_episodes)
        results[game] = eval_res[game]
        del model
        torch.cuda.empty_cache()
    return results


def run_multi_game_transfer(model_name, data_dir, output_dir, config, device, args):
    """Train on TRAIN_GAMES, evaluate on HELDOUT_GAMES."""
    logger.info("=== Multi-Game Transfer ===")
    agent_dir = str(Path(output_dir) / "multi_game")
    model, tokenizer = train_agent_on_games(
        model_name, TRAIN_GAMES, data_dir, agent_dir, num_epochs=args.num_epochs,
    )

    train_results = evaluate_on_games(
        model, tokenizer, config, TRAIN_GAMES, device, args.eval_episodes,
    )
    heldout_results = evaluate_on_games(
        model, tokenizer, config, HELDOUT_GAMES, device, args.eval_episodes,
    )

    del model
    torch.cuda.empty_cache()
    return {"train_games": train_results, "heldout_games": heldout_results}


def run_curriculum_transfer(model_name, data_dir, output_dir, config, device, args):
    """Train in curriculum order, evaluating after each stage."""
    logger.info("=== Curriculum Transfer ===")
    stage_results = []
    model = None

    for stage_idx, stage_games in enumerate(CURRICULUM_ORDER):
        logger.info("Curriculum stage %d: %s", stage_idx, stage_games)
        stage_dir = str(Path(output_dir) / "curriculum" / f"stage_{stage_idx}")

        model, tokenizer = train_agent_on_games(
            model_name, stage_games, data_dir, stage_dir,
            num_epochs=args.num_epochs, base_model=model,
        )

        all_games = TRAIN_GAMES + HELDOUT_GAMES
        eval_res = evaluate_on_games(
            model, tokenizer, config, all_games, device, args.eval_episodes,
        )

        stage_results.append({
            "stage": stage_idx,
            "games_trained": stage_games,
            "eval_results": eval_res,
        })

    del model
    torch.cuda.empty_cache()
    return stage_results


def main():
    parser = argparse.ArgumentParser(description="Cross-game transfer evaluation")
    parser.add_argument("--config", type=str, default="configs/game_scenarios.yaml")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/cross_game_transfer")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_single", action="store_true", help="Skip single-game baseline")
    parser.add_argument("--skip_curriculum", action="store_true", help="Skip curriculum training")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config["model"]["base_model"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Cross-Game Transfer Experiment ===")
    logger.info("Train games: %s", TRAIN_GAMES)
    logger.info("Held-out games: %s", HELDOUT_GAMES)

    all_results = {}

    if not args.skip_single:
        single_results = run_single_game_baseline(
            model_name, args.data_dir, str(output_dir), config, device, args,
        )
        all_results["single_game"] = single_results

    multi_results = run_multi_game_transfer(
        model_name, args.data_dir, str(output_dir), config, device, args,
    )
    all_results["multi_game"] = multi_results

    if not args.skip_curriculum:
        curriculum_results = run_curriculum_transfer(
            model_name, args.data_dir, str(output_dir), config, device, args,
        )
        all_results["curriculum"] = curriculum_results

    with open(output_dir / "transfer_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("\n=== Transfer Experiment Summary ===")

    if "single_game" in all_results:
        logger.info("Single-game baselines (held-out):")
        for game, res in all_results["single_game"].items():
            logger.info("  %s: %.3f", game, res["avg_payoff"])

    logger.info("Multi-game transfer (held-out):")
    for game, res in all_results["multi_game"]["heldout_games"].items():
        logger.info("  %s: %.3f", game, res["avg_payoff"])

    if "curriculum" in all_results:
        final_stage = all_results["curriculum"][-1]
        logger.info("Curriculum final stage (held-out):")
        for game in HELDOUT_GAMES:
            res = final_stage["eval_results"].get(game, {})
            logger.info("  %s: %.3f", game, res.get("avg_payoff", 0))

    logger.info("All results saved to %s", output_dir / "transfer_results.json")


if __name__ == "__main__":
    main()
