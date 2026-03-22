#!/usr/bin/env python3
"""Nash-DPO training after self-play.

Iterates: self-play -> Nash-DPO -> self-play (configurable iterations).
Loads self-play preference data, applies NashDPOLoss with multi-agent
preference scores.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import PeftModel, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.game_environments_simple import ALL_GAMES
from src.game_protocol import (
    AgentRole, generate_candidates, cross_evaluate,
    aggregate_preferences, compute_elo_ratings,
)
from src.nash_dpo import NashDPOLoss, create_nash_dpo_dataset

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_nash_dpo")


def parse_args():
    p = argparse.ArgumentParser(description="Iterative Nash-DPO training")
    p.add_argument("--config", type=str, default="configs/agent_roles.yaml")
    p.add_argument("--agents_dir", type=str, default="./results/sft_role_agents",
                   help="Directory with SFT-trained agent LoRA checkpoints")
    p.add_argument("--output_dir", type=str, default="./results/nash_dpo")
    p.add_argument("--num_iterations", type=int, default=3)
    p.add_argument("--games_per_iter", type=int, default=500)
    p.add_argument("--dpo_epochs", type=int, default=1)
    p.add_argument("--dpo_batch_size", type=int, default=2)
    p.add_argument("--dpo_lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_agent_roles(cfg):
    agents = {}
    for aid, acfg in cfg["agents"].items():
        agents[aid] = AgentRole(
            name=acfg["name"], role_id=aid,
            description=acfg["description"],
            reward_weights=acfg["reward_weights"],
            eval_prompt_suffix=acfg["eval_prompt_suffix"],
        )
    return agents


def load_agents(agents_dir, agent_ids, base_model, agent_paths=None):
    models, tokenizers = {}, {}
    for aid in agent_ids:
        apath = (agent_paths or {}).get(aid, os.path.join(agents_dir, aid))
        logger.info("Loading agent %s from %s", aid, apath)

        tokenizer = AutoTokenizer.from_pretrained(apath, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizers[aid] = tokenizer

        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        try:
            model = PeftModel.from_pretrained(model, apath)
        except Exception as e:
            logger.warning("No LoRA for %s: %s", aid, e)
        model.eval()
        models[aid] = model

    return models, tokenizers


def generate_game_prompts(games_per_iter):
    prompts = []
    per_game = max(games_per_iter // len(ALL_GAMES), 1)
    for game_name, game in ALL_GAMES.items():
        for _ in range(per_game):
            prompts.append(game.build_prompt(0))
    import random
    random.shuffle(prompts)
    return prompts[:games_per_iter]


def run_self_play_round(models, tokenizers, agent_roles, prompts, cfg):
    sp_cfg = cfg.get("self_play", {})
    candidates = generate_candidates(
        models, tokenizers, agent_roles, prompts,
        max_new_tokens=512,
        temperature=sp_cfg.get("cross_eval_temperature", 0.7),
    )
    evaluations = cross_evaluate(candidates, agent_roles)
    pairs = aggregate_preferences(
        evaluations, candidates,
        min_margin=sp_cfg.get("min_preference_margin", 0.1),
    )
    elo = compute_elo_ratings(evaluations, agent_roles)
    return pairs, elo, evaluations


def run_nash_dpo_update(pairs, base_model, agent_id, agent_ckpt,
                        output_dir, args, iteration):
    if not pairs:
        logger.warning("No pairs for %s, skipping", agent_id)
        return agent_ckpt

    records = create_nash_dpo_dataset(pairs, [])
    if not records:
        return agent_ckpt

    dataset = Dataset.from_list(records)
    logger.info("Nash-DPO for %s: %d pairs", agent_id, len(dataset))

    tokenizer = AutoTokenizer.from_pretrained(agent_ckpt, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    try:
        model = PeftModel.from_pretrained(model, agent_ckpt)
        model = model.merge_and_unload()
    except Exception:
        pass

    iter_dir = os.path.join(output_dir, f"iter{iteration}", agent_id)
    os.makedirs(iter_dir, exist_ok=True)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    dpo_cfg = DPOConfig(
        output_dir=iter_dir,
        per_device_train_batch_size=args.dpo_batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.dpo_epochs,
        learning_rate=args.dpo_lr,
        warmup_ratio=0.1,
        bf16=True,
        beta=args.beta,
        logging_steps=10,
        save_steps=9999,
        report_to="none",
        max_length=2048,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )

    trainer.train()
    trainer.save_model(iter_dir)
    tokenizer.save_pretrained(iter_dir)
    logger.info("Nash-DPO saved to %s", iter_dir)

    del model, trainer
    torch.cuda.empty_cache()
    return iter_dir


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    base_model = cfg["model"]["base"]
    agent_roles = build_agent_roles(cfg)
    agent_ids = list(agent_roles.keys())

    current_paths = {aid: os.path.join(args.agents_dir, aid) for aid in agent_ids}
    all_stats = []

    for iteration in range(args.num_iterations):
        logger.info("\n" + "=" * 70)
        logger.info("ITERATION %d / %d", iteration + 1, args.num_iterations)
        logger.info("=" * 70)

        models, tokenizers = load_agents(
            args.agents_dir, agent_ids, base_model, current_paths,
        )

        prompts = generate_game_prompts(args.games_per_iter)
        logger.info("Self-play with %d prompts", len(prompts))
        pairs, elo, _ = run_self_play_round(
            models, tokenizers, agent_roles, prompts, cfg,
        )
        logger.info("Generated %d preference pairs", len(pairs))
        logger.info("Elo ratings: %s", elo)

        del models
        torch.cuda.empty_cache()

        for aid in agent_ids:
            agent_pairs = [p for p in pairs
                           if p.chosen_agent == aid or p.rejected_agent == aid]
            new_path = run_nash_dpo_update(
                agent_pairs, base_model, aid, current_paths[aid],
                args.output_dir, args, iteration,
            )
            current_paths[aid] = new_path

        iter_stats = {
            "iteration": iteration,
            "num_pairs": len(pairs),
            "elo": elo,
            "agent_paths": dict(current_paths),
        }
        all_stats.append(iter_stats)

        with open(os.path.join(args.output_dir, f"iter{iteration}_stats.json"), "w") as f:
            json.dump(iter_stats, f, indent=2, default=str)

    summary = {
        "num_iterations": args.num_iterations,
        "iterations": all_stats,
        "final_elo": all_stats[-1]["elo"] if all_stats else {},
        "final_paths": current_paths,
    }
    with open(os.path.join(args.output_dir, "nash_dpo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\nNash-DPO training complete!")
    logger.info("Final Elo: %s", summary["final_elo"])


if __name__ == "__main__":
    main()
