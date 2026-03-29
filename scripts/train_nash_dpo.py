#!/usr/bin/env python3
"""Nash-DPO training after self-play.

Iterates: self-play -> Nash-DPO -> self-play (configurable iterations).
Supports parallel DPO training across GPUs via --dpo_only subprocess mode.
"""

import argparse
import dataclasses
import glob
import json
import logging
import os
import subprocess
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
    AgentRole, PreferencePair, generate_candidates, cross_evaluate,
    aggregate_preferences, compute_elo_ratings,
)
from src.nash_dpo import NashDPOLoss, create_nash_dpo_dataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_nash_dpo")


def find_latest_checkpoint(output_dir):
    ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")),
                   key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
    return ckpts[-1] if ckpts else None


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
    p.add_argument("--nash_weights", type=str, default="nash",
                   choices=["nash", "equal"])
    # Subprocess DPO mode
    p.add_argument("--dpo_only", action="store_true",
                   help="Run DPO for a single agent (called by subprocess)")
    p.add_argument("--dpo_agent", type=str)
    p.add_argument("--pairs_file", type=str)
    p.add_argument("--agent_ckpt", type=str)
    p.add_argument("--iteration", type=int, default=0)
    p.add_argument("--resume_from_checkpoint", type=str, default="auto",
                   help="Resume from checkpoint: 'auto', path, or 'none'")
    return p.parse_args()


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Serialization ─────────────────────────────────────────────────────────

def save_pairs(pairs, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        for p in pairs:
            f.write(json.dumps(dataclasses.asdict(p)) + "\n")
    logger.info("Saved %d pairs -> %s", len(pairs), filepath)


def load_pairs(filepath):
    pairs = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                pairs.append(PreferencePair(**json.loads(line)))
    logger.info("Loaded %d pairs <- %s", len(pairs), filepath)
    return pairs


# ── Agent Management ──────────────────────────────────────────────────────

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
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
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
        max_new_tokens=256,
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

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map={"": local_rank},
    )
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    try:
        model = PeftModel.from_pretrained(model, agent_ckpt)
        model = model.merge_and_unload()
    except Exception:
        pass

    if not hasattr(model, "hf_device_map"):
        model.hf_device_map = {"": local_rank}

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
        warmup_steps=50,
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

    resume_ckpt = None
    if args.resume_from_checkpoint != "none":
        if args.resume_from_checkpoint == "auto":
            resume_ckpt = find_latest_checkpoint(iter_dir)
        else:
            resume_ckpt = args.resume_from_checkpoint
        if resume_ckpt:
            logger.info(f"Resuming from checkpoint: {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(iter_dir)
    tokenizer.save_pretrained(iter_dir)
    logger.info("Nash-DPO saved to %s", iter_dir)

    del model, trainer
    torch.cuda.empty_cache()
    return iter_dir


def run_parallel_dpo(pairs, agent_ids, current_paths, args, iteration, num_gpus=4):
    """Launch DPO for each agent as a subprocess on a separate GPU.

    Batches agents when there are more agents than GPUs to avoid OOM.
    """
    pairs_dir = os.path.join(args.output_dir, f"iter{iteration}_pairs")
    os.makedirs(pairs_dir, exist_ok=True)

    agent_pairs_files = {}
    for aid in agent_ids:
        agent_pairs = [p for p in pairs
                       if p.chosen_agent == aid or p.rejected_agent == aid]
        pf = os.path.join(pairs_dir, f"{aid}.jsonl")
        save_pairs(agent_pairs, pf)
        agent_pairs_files[aid] = pf

    failed = []
    for batch_start in range(0, len(agent_ids), num_gpus):
        batch = agent_ids[batch_start:batch_start + num_gpus]
        processes = []
        for gpu_id, aid in enumerate(batch):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--dpo_only",
                "--dpo_agent", aid,
                "--pairs_file", agent_pairs_files[aid],
                "--agent_ckpt", current_paths[aid],
                "--config", args.config,
                "--output_dir", args.output_dir,
                "--iteration", str(iteration),
                "--beta", str(args.beta),
                "--dpo_epochs", str(args.dpo_epochs),
                "--dpo_batch_size", str(args.dpo_batch_size),
                "--dpo_lr", str(args.dpo_lr),
                "--seed", str(args.seed),
                "--nash_weights", args.nash_weights,
                "--resume_from_checkpoint", args.resume_from_checkpoint,
            ]
            logger.info("Launching DPO for %s on GPU %d", aid, gpu_id)
            p = subprocess.Popen(cmd, env=env)
            processes.append((aid, p))

        for aid, p in processes:
            p.wait()
            if p.returncode != 0:
                logger.error("DPO for %s failed (exit %d)", aid, p.returncode)
                failed.append(aid)
            else:
                logger.info("DPO for %s completed OK", aid)

    if failed:
        raise RuntimeError(f"DPO failed for agents: {failed}")

    return {aid: os.path.join(args.output_dir, f"iter{iteration}", aid)
            for aid in agent_ids}


def main():
    args = parse_args()

    # ── Subprocess DPO-only mode ──────────────────────────────────────────
    if args.dpo_only:
        cfg = load_cfg(args.config)
        base_model = cfg["model"]["base"]
        pairs = load_pairs(args.pairs_file)
        run_nash_dpo_update(
            pairs, base_model, args.dpo_agent, args.agent_ckpt,
            args.output_dir, args, args.iteration,
        )
        return

    # ── Normal iterative flow ─────────────────────────────────────────────
    cfg = load_cfg(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    base_model = cfg["model"]["base"]
    agent_roles = build_agent_roles(cfg)
    agent_ids = list(agent_roles.keys())
    num_gpus = torch.cuda.device_count()

    current_paths = {aid: os.path.join(args.agents_dir, aid) for aid in agent_ids}
    all_stats = []

    for iteration in range(args.num_iterations):
        logger.info("\n" + "=" * 70)
        logger.info("ITERATION %d / %d", iteration + 1, args.num_iterations)
        logger.info("=" * 70)

        all_pairs_file = os.path.join(args.output_dir, f"iter{iteration}_all_pairs.jsonl")
        elo_file = os.path.join(args.output_dir, f"iter{iteration}_elo.json")

        if os.path.exists(all_pairs_file) and os.path.exists(elo_file):
            logger.info("Resuming from cached pairs: %s", all_pairs_file)
            pairs = load_pairs(all_pairs_file)
            with open(elo_file) as f:
                elo = json.load(f)
        else:
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

            save_pairs(pairs, all_pairs_file)
            with open(elo_file, "w") as f:
                json.dump(elo, f)

            del models
            torch.cuda.empty_cache()

        # Check if DPO already completed for this iteration
        iter_done = all(
            os.path.isdir(os.path.join(args.output_dir, f"iter{iteration}", aid))
            and os.path.exists(os.path.join(args.output_dir, f"iter{iteration}", aid, "adapter_config.json"))
            for aid in agent_ids
        )
        if iter_done:
            logger.info("Iteration %d DPO already complete, skipping", iteration)
            for aid in agent_ids:
                current_paths[aid] = os.path.join(args.output_dir, f"iter{iteration}", aid)
        else:
            new_paths = run_parallel_dpo(
                pairs, agent_ids, current_paths, args, iteration, num_gpus,
            )
            current_paths.update(new_paths)

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
