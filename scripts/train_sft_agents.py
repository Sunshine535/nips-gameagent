#!/usr/bin/env python3
"""SFT training for role-specialized agents (accuracy/safety/efficiency/creativity).

Trains 4 LoRA agents from expert game data, each agent specializing
in different game subsets using TRL's SFTTrainer.
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.game_environments_simple import list_games
from src.game_protocol import compute_agent_reward


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_sft_agents")

AGENT_GAME_SUBSETS = {
    "accuracy": ["prisoners_dilemma", "stag_hunt"],
    "safety": ["chicken", "ultimatum"],
    "efficiency": ["matching_pennies", "coordination"],
    "creativity": ["battle_of_sexes", "public_goods"],
}


def find_latest_checkpoint(output_dir):
    ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")),
                   key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
    return ckpts[-1] if ckpts else None


def parse_args():
    p = argparse.ArgumentParser(description="SFT training for game agents")
    p.add_argument("--config", type=str, default="configs/agent_roles.yaml")
    p.add_argument("--expert_data", type=str, default="./results/expert_data/expert_train.jsonl")
    p.add_argument("--output_dir", type=str, default="./results/sft_agents")
    p.add_argument("--agent", type=str, default=None,
                   choices=list(AGENT_GAME_SUBSETS.keys()))
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_from_checkpoint", type=str, default="auto",
                   help="Resume from checkpoint: 'auto', path, or 'none'")
    return p.parse_args()


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_expert_data(path, game_subset=None, reward_weights=None,
                     top_fraction=0.5):
    """Load expert data, optionally filtering by game subset and reward score.

    When *reward_weights* is given (from agent_roles.yaml), records are scored
    and only the top *top_fraction* are kept — this reward-based curriculum was
    originally in the GameRefine pipeline and improves SFT quality.
    """
    raw = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if game_subset and rec.get("game") not in game_subset:
                continue
            raw.append(rec)

    if reward_weights and raw:
        for rec in raw:
            rec["_score"] = compute_agent_reward(
                rec["response"], reward_weights, reference=None,
            )
        raw.sort(key=lambda r: r["_score"], reverse=True)
        cutoff = max(1, int(len(raw) * top_fraction))
        raw = raw[:cutoff]

    records = [
        {"text": f"### Game Prompt:\n{r['prompt']}\n\n### Response:\n{r['response']}"}
        for r in raw
    ]
    return Dataset.from_list(records)


def train_agent(agent_id, game_subset, model_name, expert_path, output_dir,
                lora_cfg, train_cfg, seed, reward_weights=None):
    logger.info("=" * 60)
    logger.info("Training agent: %s", agent_id)
    logger.info("Game subset: %s", game_subset)
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    dataset = load_expert_data(expert_path, game_subset,
                               reward_weights=reward_weights)
    logger.info("Loaded %d expert samples for %s", len(dataset), agent_id)

    if len(dataset) == 0:
        logger.warning("No data for %s, loading all games", agent_id)
        dataset = load_expert_data(expert_path, None,
                                   reward_weights=reward_weights)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
                                              padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora = LoraConfig(
        r=lora_cfg.get("lora_r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules",
                                    ["q_proj", "k_proj", "v_proj", "o_proj"]),
        task_type=TaskType.CAUSAL_LM, bias="none",
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        max_length=train_cfg.get("max_seq_length", 1024),
        dataset_text_field="text",
        report_to="none",
        seed=seed,
    )

    trainer = SFTTrainer(
        model=model_name,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )

    resume_ckpt = None
    resume_opt = train_cfg.get("resume_from_checkpoint", "auto")
    if resume_opt != "none":
        if resume_opt == "auto":
            resume_ckpt = find_latest_checkpoint(output_dir)
        else:
            resume_ckpt = resume_opt
        if resume_ckpt:
            logger.info(f"Resuming from checkpoint: {resume_ckpt}")

    result = trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    info = {
        "agent_id": agent_id,
        "game_subset": game_subset,
        "num_samples": len(dataset),
        "train_loss": result.metrics.get("train_loss"),
        "train_runtime": result.metrics.get("train_runtime"),
    }
    with open(os.path.join(output_dir, "sft_agent_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    logger.info("Agent %s trained. Loss: %.4f", agent_id,
                result.metrics.get("train_loss", -1))
    del trainer
    torch.cuda.empty_cache()
    return info


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    model_name = cfg["model"]["base"]
    lora_cfg = cfg["model"]
    train_cfg = {
        "num_epochs": args.num_epochs,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "resume_from_checkpoint": args.resume_from_checkpoint,
    }

    agents = {args.agent: AGENT_GAME_SUBSETS[args.agent]} if args.agent else AGENT_GAME_SUBSETS
    agent_cfgs = cfg.get("agents", {})
    all_info = {}

    for agent_id, game_subset in agents.items():
        out = os.path.join(args.output_dir, agent_id)
        if os.path.exists(os.path.join(out, "sft_agent_info.json")):
            logger.info("Skipping %s (already trained)", agent_id)
            continue
        rw = agent_cfgs.get(agent_id, {}).get("reward_weights")
        info = train_agent(
            agent_id, game_subset, model_name, args.expert_data, out,
            lora_cfg, train_cfg, args.seed, reward_weights=rw,
        )
        all_info[agent_id] = info

    summary_path = os.path.join(args.output_dir, "sft_agents_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_info, f, indent=2)
    logger.info("All SFT agents trained. Summary: %s", summary_path)


if __name__ == "__main__":
    main()
