#!/usr/bin/env python3
"""
Train with NashDPOTrainer (Algorithm 1 from the paper).

Uses the custom NashDPOTrainer that subclasses TRL's DPOTrainer to apply
per-objective Nash bargaining weights. The dataset must contain
pref_correctness, pref_safety, pref_efficiency, pref_creativity columns.
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.nash_dpo_trainer import (
    NashDPOTrainer,
    NashWeightLoggingCallback,
    OBJECTIVE_NAMES,
    estimate_disagreement_from_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_formal_nash_dpo")


def parse_args():
    p = argparse.ArgumentParser(description="Formal Nash-DPO Training (Algorithm 1)")
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--preference_data", type=str, required=True,
                    help="JSONL with preference pairs + per-objective signals")
    p.add_argument("--output_dir", type=str, default="./results/formal_nash_dpo")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--ema_tau", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--method", type=str, default="nash",
                    choices=["nash", "equal", "fixed", "single_correctness"],
                    help="Weighting method for ablation comparison")
    p.add_argument("--fixed_weights", type=str, default="0.4,0.3,0.2,0.1")
    p.add_argument("--resume_from_checkpoint", type=str, default="auto",
                    help="Resume: 'auto', explicit path, or 'none'")
    p.add_argument("--model_path", type=str, default=None,
                    help="LoRA adapter to load before training (e.g. GRPO checkpoint)")
    return p.parse_args()


def find_latest_checkpoint(output_dir):
    import glob
    ckpts = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0,
    )
    return ckpts[-1] if ckpts else None


def load_preference_data(filepath):
    records = []
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            clean = {
                "prompt": rec["prompt"],
                "chosen": rec["chosen"],
                "rejected": rec["rejected"],
            }
            for name in OBJECTIVE_NAMES:
                key = f"pref_{name}"
                clean[key] = float(rec.get(key, 0.0))
            records.append(clean)
    logger.info("Loaded %d preference pairs from %s", len(records), filepath)
    return records


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    records = load_preference_data(args.preference_data)
    dataset = Dataset.from_list(records)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.model_path:
        from peft import PeftModel as PM
        logger.info("Loading LoRA adapter from %s and merging...", args.model_path)
        model = PM.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        bf16=True,
        beta=args.beta,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        max_length=args.max_length,
        seed=args.seed,
        remove_unused_columns=False,
    )

    disagreement = torch.full((len(OBJECTIVE_NAMES),), math.log(2))

    fixed_w = None
    if args.method == "fixed":
        fixed_w = [float(x) for x in args.fixed_weights.split(",")]

    trainer = NashDPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        nash_ema_tau=args.ema_tau,
        nash_warmup_steps=args.warmup_steps,
        disagreement_losses=disagreement,
        weighting_method=args.method,
        fixed_weights=fixed_w,
    )

    callback = NashWeightLoggingCallback(trainer, log_every=50)
    trainer.add_callback(callback)

    resume_ckpt = None
    if args.resume_from_checkpoint != "none":
        if args.resume_from_checkpoint == "auto":
            resume_ckpt = find_latest_checkpoint(args.output_dir)
        else:
            p = args.resume_from_checkpoint
            if os.path.isdir(p):
                resume_ckpt = p
        if resume_ckpt:
            logger.info("Resuming from checkpoint: %s", resume_ckpt)

    logger.info("Starting %s training (method=%s, beta=%.2f, ema_tau=%.2f)...",
                "Nash-DPO", args.method, args.beta, args.ema_tau)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary = {
        "method": args.method,
        "beta": args.beta,
        "final_nash_weights": trainer.nash_weights_dict,
        "total_steps": trainer._nash_step,
        "weight_history": trainer._weight_history,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training complete. Final weights: %s", trainer.nash_weights_dict)


if __name__ == "__main__":
    main()
