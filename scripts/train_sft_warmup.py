#!/usr/bin/env python3
"""SFT warmup: train 4 LoRA agents, each specializing in 2 of the 8 games.

Agent allocation:
  Agent 0: prisoners_dilemma + coordination_game    (symmetric matrix games)
  Agent 1: battle_of_sexes + stag_hunt              (asymmetric / trust games)
  Agent 2: public_goods + auction                   (n-player competitive)
  Agent 3: ultimatum + negotiation                  (sequential bargaining)

Uses TRL SFTTrainer with LoRA r=16, 2 epochs per agent.

Usage:
    python scripts/train_sft_warmup.py --data_dir data --output_dir results/sft_agents
    torchrun --nproc_per_node=4 scripts/train_sft_warmup.py ...
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_latest_checkpoint(output_dir):
    ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")),
                   key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
    return ckpts[-1] if ckpts else None


AGENT_GAME_ASSIGNMENTS = {
    "agent_0": ["prisoners_dilemma", "coordination_game"],
    "agent_1": ["battle_of_sexes", "stag_hunt"],
    "agent_2": ["public_goods", "auction"],
    "agent_3": ["ultimatum", "negotiation"],
}


def load_sft_data(data_dir: str, game_names: list) -> Dataset:
    """Load and merge SFT JSONL files for specified games."""
    all_samples = []
    for game in game_names:
        path = Path(data_dir) / f"sft_{game}.jsonl"
        if not path.exists():
            logger.warning("SFT data not found: %s", path)
            continue
        with open(path) as f:
            for line in f:
                sample = json.loads(line)
                all_samples.append(sample)
        logger.info("Loaded %d samples from %s", len(all_samples), path)

    if not all_samples:
        raise FileNotFoundError(f"No SFT data found in {data_dir} for games {game_names}")

    texts = []
    for sample in all_samples:
        msgs = sample["messages"]
        text = ""
        for msg in msgs:
            role = msg["role"]
            content = msg["content"]
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        texts.append(text)

    return Dataset.from_dict({"text": texts})


def train_single_agent(
    agent_name: str,
    game_names: list,
    model_name: str,
    data_dir: str,
    output_dir: str,
    args,
):
    """Train one LoRA agent on its assigned games."""
    logger.info("=== Training %s on games: %s ===", agent_name, game_names)

    dataset = load_sft_data(data_dir, game_names)
    logger.info("Dataset size: %d samples", len(dataset))

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

    agent_output = Path(output_dir) / agent_name
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

    resume_ckpt = None
    if args.resume_from_checkpoint != "none":
        if args.resume_from_checkpoint == "auto":
            resume_ckpt = find_latest_checkpoint(str(agent_output))
        else:
            resume_ckpt = args.resume_from_checkpoint
        if resume_ckpt:
            logger.info(f"Resuming from checkpoint: {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(agent_output / "final"))
    tokenizer.save_pretrained(str(agent_output / "final"))

    meta = {
        "agent_name": agent_name,
        "games": game_names,
        "base_model": model_name,
        "lora_r": args.lora_r,
        "num_epochs": args.num_epochs,
        "dataset_size": len(dataset),
    }
    with open(agent_output / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved %s to %s", agent_name, agent_output / "final")

    del model, trainer
    torch.cuda.empty_cache()

    return meta


def main():
    parser = argparse.ArgumentParser(description="SFT warmup training")
    parser.add_argument("--config", type=str, default="configs/game_scenarios.yaml")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/sft_agents")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--agents", type=str, nargs="*", default=None,
                        help="Train specific agents only (e.g., agent_0 agent_2)")
    parser.add_argument("--parallel", action="store_true",
                        help="Train all agents in parallel (one per GPU)")
    parser.add_argument("--resume_from_checkpoint", type=str, default="auto",
                        help="Resume from checkpoint: 'auto', path, or 'none'")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["base_model"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agents_to_train = args.agents or list(AGENT_GAME_ASSIGNMENTS.keys())

    if args.parallel and torch.cuda.device_count() >= len(agents_to_train):
        import subprocess
        procs = []
        for i, agent_name in enumerate(agents_to_train):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(i)
            cmd = [
                sys.executable, __file__,
                "--config", args.config,
                "--data_dir", args.data_dir,
                "--output_dir", str(output_dir),
                "--num_epochs", str(args.num_epochs),
                "--batch_size", str(args.batch_size),
                "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
                "--learning_rate", str(args.learning_rate),
                "--lora_r", str(args.lora_r),
                "--lora_alpha", str(args.lora_alpha),
                "--max_seq_length", str(args.max_seq_length),
                "--agents", agent_name,
                "--resume_from_checkpoint", args.resume_from_checkpoint,
                "--seed", str(args.seed),
            ]
            logger.info("Launching %s on GPU %d", agent_name, i)
            procs.append(subprocess.Popen(cmd, env=env))
        for p in procs:
            p.wait()
            if p.returncode != 0:
                logger.error("Agent training failed with code %d", p.returncode)
                sys.exit(p.returncode)
    else:
        all_meta = {}
        for agent_name in agents_to_train:
            if agent_name not in AGENT_GAME_ASSIGNMENTS:
                logger.warning("Unknown agent: %s, skipping", agent_name)
                continue
            game_names = AGENT_GAME_ASSIGNMENTS[agent_name]
            meta = train_single_agent(
                agent_name, game_names, model_name, args.data_dir, str(output_dir), args,
            )
            all_meta[agent_name] = meta

        summary_path = output_dir / "sft_warmup_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_meta, f, indent=2)

        logger.info("=== SFT Warmup Complete ===")
        for name, meta in all_meta.items():
            logger.info("  %s: games=%s, samples=%d", name, meta["games"], meta["dataset_size"])
        logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
