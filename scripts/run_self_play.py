#!/usr/bin/env python3
"""DEPRECATED — Replaced by scripts/train_nash_dpo.py

train_nash_dpo.py provides the same iterative self-play + Nash-DPO loop
with game-specific prompts from game_environments_simple. Use:
    python scripts/train_nash_dpo.py --config configs/agent_roles.yaml

Original purpose:
Multi-round game: each agent generates candidate -> agents cross-evaluate ->
collect preference pairs -> Nash-DPO update.
Implements iterative self-play between 4 specialized agents.
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
from src.game_protocol import (
    AgentRole, GameCandidate, PreferencePair,
    generate_candidates, cross_evaluate, aggregate_preferences,
    compute_elo_ratings,
)
from src.nash_dpo import NashDPOLoss, create_nash_dpo_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_self_play")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-agent self-play with Nash-DPO")
    parser.add_argument("--config", type=str, default="configs/agent_roles.yaml")
    parser.add_argument("--agents_dir", type=str, default=None,
                        help="Directory with trained agent checkpoints")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_rounds", type=int, default=None)
    parser.add_argument("--games_per_round", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_agent_roles(cfg: dict) -> dict[str, AgentRole]:
    agents = {}
    for agent_id, acfg in cfg["agents"].items():
        agents[agent_id] = AgentRole(
            name=acfg["name"],
            role_id=agent_id,
            description=acfg["description"],
            reward_weights=acfg["reward_weights"],
            eval_prompt_suffix=acfg["eval_prompt_suffix"],
        )
    return agents


def load_agent_models(
    agents_dir: str,
    agent_ids: list[str],
    base_model: str,
    agent_paths: dict[str, str] | None = None,
):
    """Load all agent LoRA models.

    If ``agent_paths`` is provided, each ``agent_paths[agent_id]`` is used as the
    checkpoint directory (so Nash-DPO outputs under ``output_dir/round*/`` are picked up).
    Otherwise paths default to ``os.path.join(agents_dir, agent_id)``.
    """
    models = {}
    tokenizers = {}

    for agent_id in agent_ids:
        agent_path = (
            agent_paths[agent_id]
            if agent_paths is not None and agent_id in agent_paths
            else os.path.join(agents_dir, agent_id)
        )
        logger.info(f"Loading agent: {agent_id} from {agent_path}")

        tokenizer = AutoTokenizer.from_pretrained(agent_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizers[agent_id] = tokenizer

        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        try:
            model = PeftModel.from_pretrained(model, agent_path)
            logger.info(f"Loaded LoRA adapter for {agent_id}")
        except Exception as e:
            logger.warning(f"Could not load LoRA for {agent_id}: {e}")

        model.eval()
        models[agent_id] = model

    return models, tokenizers


def load_game_prompts(num_prompts: int) -> list[str]:
    """Load diverse prompts for self-play games."""
    from datasets import load_dataset

    prompts = []
    try:
        gsm8k = load_dataset("openai/gsm8k", "main", split="train")
        for ex in gsm8k:
            prompts.append(f"Solve: {ex['question']}")
            if len(prompts) >= num_prompts // 2:
                break
    except Exception:
        pass

    try:
        alpaca = load_dataset("tatsu-lab/alpaca_eval", split="eval")
        for ex in alpaca:
            prompts.append(ex["instruction"])
            if len(prompts) >= num_prompts:
                break
    except Exception:
        pass

    if not prompts:
        prompts = [
            "Explain the theory of relativity simply.",
            "Write a short poem about technology.",
            "What causes climate change?",
            "How does a computer work?",
            "Describe the water cycle.",
        ] * (num_prompts // 5 + 1)

    import random
    random.seed(42)
    random.shuffle(prompts)
    return prompts[:num_prompts]


def run_nash_dpo_update(
    preference_pairs: list[PreferencePair],
    base_model_name: str,
    agent_id: str,
    agent_ckpt: str,
    output_dir: str,
    beta: float = 0.1,
    round_num: int = 0,
):
    """Run one round of Nash-DPO update for an agent."""
    if not preference_pairs:
        logger.warning(f"No preference pairs for {agent_id}, skipping update")
        return

    records = create_nash_dpo_dataset(preference_pairs, [])
    if not records:
        return

    dataset = Dataset.from_list(records)
    logger.info(f"Nash-DPO update for {agent_id}: {len(dataset)} pairs")

    tokenizer = AutoTokenizer.from_pretrained(agent_ckpt, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    try:
        model = PeftModel.from_pretrained(model, agent_ckpt)
        model = model.merge_and_unload()
    except Exception:
        pass

    round_output = os.path.join(output_dir, f"round{round_num}", agent_id)
    os.makedirs(round_output, exist_ok=True)

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    dpo_config = DPOConfig(
        output_dir=round_output,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        bf16=True,
        beta=beta,
        logging_steps=10,
        save_steps=9999,
        report_to="none",
        max_length=2048,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(round_output)
    tokenizer.save_pretrained(round_output)
    logger.info(f"Nash-DPO update saved to {round_output}")

    return round_output


def main():
    args = parse_args()
    cfg = load_config(args.config)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    sp_cfg = cfg["self_play"]
    agents_dir = args.agents_dir or cfg["output"]["agents_dir"]
    output_dir = args.output_dir or cfg["output"]["self_play_dir"]
    num_rounds = args.num_rounds or sp_cfg["num_rounds"]
    games_per_round = args.games_per_round or sp_cfg["games_per_round"]

    os.makedirs(output_dir, exist_ok=True)

    base_model = cfg["model"]["base"]
    agent_roles = build_agent_roles(cfg)
    agent_ids = list(agent_roles.keys())

    logger.info(f"Loading {len(agent_ids)} agents from {agents_dir}")
    models, tokenizers = load_agent_models(agents_dir, agent_ids, base_model)

    all_round_stats = []
    current_agent_paths = {aid: os.path.join(agents_dir, aid) for aid in agent_ids}

    for round_num in range(num_rounds):
        logger.info(f"\n{'='*60}")
        logger.info(f"Self-Play Round {round_num + 1}/{num_rounds}")
        logger.info(f"{'='*60}")

        prompts = load_game_prompts(games_per_round)
        logger.info(f"Loaded {len(prompts)} game prompts")

        # Phase 1: Generate
        logger.info("[Phase 1] Generating candidates...")
        candidates = generate_candidates(
            models, tokenizers, agent_roles, prompts,
            max_new_tokens=512,
            temperature=sp_cfg.get("cross_eval_temperature", 0.7),
        )

        # Phase 2: Cross-evaluate
        logger.info("[Phase 2] Cross-evaluating...")
        evaluations = cross_evaluate(candidates, agent_roles)

        # Phase 3: Aggregate preferences
        logger.info("[Phase 3] Aggregating preferences...")
        preference_pairs = aggregate_preferences(
            evaluations, candidates,
            min_margin=sp_cfg.get("min_preference_margin", 0.1),
        )
        logger.info(f"Generated {len(preference_pairs)} preference pairs")

        # Compute Elo ratings
        elo_ratings = compute_elo_ratings(evaluations, agent_roles,
                                          k_factor=sp_cfg.get("elo_k_factor", 32.0))
        logger.info(f"Elo ratings: {elo_ratings}")

        # Phase 4: Nash-DPO update for each agent
        logger.info("[Phase 4] Nash-DPO updates...")
        for agent_id in agent_ids:
            agent_pairs = [
                p for p in preference_pairs
                if p.chosen_agent == agent_id or p.rejected_agent == agent_id
            ]

            if agent_pairs:
                new_path = run_nash_dpo_update(
                    agent_pairs, base_model, agent_id,
                    current_agent_paths[agent_id], output_dir,
                    beta=sp_cfg.get("nash_dpo_beta", 0.1),
                    round_num=round_num,
                )
                if new_path:
                    current_agent_paths[agent_id] = new_path

        # Reload models with updated checkpoints
        del models
        torch.cuda.empty_cache()
        models, tokenizers = load_agent_models(
            agents_dir, agent_ids, base_model, agent_paths=current_agent_paths,
        )

        round_stats = {
            "round": round_num,
            "num_games": len(prompts),
            "num_preference_pairs": len(preference_pairs),
            "elo_ratings": elo_ratings,
            "agent_paths": dict(current_agent_paths),
        }
        all_round_stats.append(round_stats)

        round_stats_path = os.path.join(output_dir, f"round{round_num}_stats.json")
        with open(round_stats_path, "w") as f:
            json.dump(round_stats, f, indent=2)

    # Final summary
    summary = {
        "num_rounds": num_rounds,
        "games_per_round": games_per_round,
        "rounds": all_round_stats,
        "final_elo": all_round_stats[-1]["elo_ratings"] if all_round_stats else {},
        "final_agent_paths": current_agent_paths,
    }
    with open(os.path.join(output_dir, "self_play_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSelf-play complete! Summary saved to {output_dir}")
    logger.info(f"Final Elo ratings: {summary['final_elo']}")


if __name__ == "__main__":
    main()
