#!/usr/bin/env python3
"""DEPRECATED — Replaced by scripts/train_sft_agents.py

The reward-based data filtering logic from this script has been merged into
train_sft_agents.py (via the reward_weights parameter). Use:
    python scripts/train_sft_agents.py --config configs/agent_roles.yaml

Original purpose:
Train 4 specialized agents (accuracy/safety/efficiency/creativity) via LoRA
on Qwen/Qwen3.5-9B with different reward functions.
Each agent gets a separate LoRA adapter trained with role-specific rewards.
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
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.game_protocol import (
    AgentRole, REWARD_FUNCTIONS, compute_agent_reward,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_agents")


def parse_args():
    parser = argparse.ArgumentParser(description="Train specialized RL agents")
    parser.add_argument("--config", type=str, default="configs/agent_roles.yaml")
    parser.add_argument("--expert_data", type=str, default="results/expert_data/expert_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/trained_agents")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agents", nargs="*", default=None,
                        help="Train specific agents only (e.g., accuracy safety)")
    return parser.parse_args()


def load_expert_data(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def build_agent_roles(cfg: dict) -> dict[str, AgentRole]:
    agents = {}
    for aid, acfg in cfg["agents"].items():
        agents[aid] = AgentRole(
            name=acfg["name"],
            role_id=aid,
            description=acfg["description"],
            reward_weights=acfg["reward_weights"],
            eval_prompt_suffix=acfg.get("eval_prompt_suffix", ""),
        )
    return agents


def score_and_filter_data(records: list[dict], agent: AgentRole,
                          top_fraction: float = 0.5) -> list[dict]:
    """Score expert data by agent's reward function and keep top fraction."""
    scored = []
    for r in records:
        reward = compute_agent_reward(
            r["response"], agent.reward_weights, reference=None,
        )
        scored.append({**r, "agent_reward": reward})

    scored.sort(key=lambda x: x["agent_reward"], reverse=True)
    cutoff = max(1, int(len(scored) * top_fraction))
    return scored[:cutoff]


def prepare_sft_dataset(records: list[dict], agent: AgentRole) -> Dataset:
    """Format records into chat-template texts for SFT."""
    texts = []
    for r in records:
        text = (
            f"<|im_start|>system\n{agent.description}<|im_end|>\n"
            f"<|im_start|>user\n{r['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n{r['response']}<|im_end|>\n"
        )
        texts.append(text)
    return Dataset.from_dict({"text": texts})


def train_single_agent(
    agent_id: str,
    agent: AgentRole,
    records: list[dict],
    model_name: str,
    output_dir: str,
    args,
):
    logger.info("=== Training agent: %s (%s) ===", agent_id, agent.name)

    filtered = score_and_filter_data(records, agent, top_fraction=0.5)
    dataset = prepare_sft_dataset(filtered, agent)
    logger.info("Dataset: %d samples (filtered from %d)", len(dataset), len(records))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    agent_output = Path(output_dir) / agent_id
    agent_output.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(agent_output),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=args.max_seq_length,
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
    trainer.save_model(str(agent_output / "final"))
    tokenizer.save_pretrained(str(agent_output / "final"))

    meta = {
        "agent_id": agent_id,
        "agent_name": agent.name,
        "reward_weights": agent.reward_weights,
        "base_model": model_name,
        "lora_r": args.lora_r,
        "num_epochs": args.num_epochs,
        "dataset_size": len(dataset),
        "original_records": len(records),
    }
    with open(agent_output / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved %s to %s", agent_id, agent_output / "final")

    del model, trainer
    torch.cuda.empty_cache()
    return meta


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["base"]
    agent_roles = build_agent_roles(cfg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_expert_data(args.expert_data)
    logger.info("Loaded %d expert records from %s", len(records), args.expert_data)

    agents_to_train = args.agents or list(agent_roles.keys())
    all_meta = {}

    for agent_id in agents_to_train:
        if agent_id not in agent_roles:
            logger.warning("Unknown agent: %s, skipping", agent_id)
            continue
        meta = train_single_agent(
            agent_id, agent_roles[agent_id], records,
            model_name, str(output_dir), args,
        )
        all_meta[agent_id] = meta

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    logger.info("=== Agent Training Complete ===")
    for aid, meta in all_meta.items():
        logger.info("  %s: %s, samples=%d", aid, meta["agent_name"], meta["dataset_size"])
    logger.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
