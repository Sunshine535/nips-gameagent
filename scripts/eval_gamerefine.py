#!/usr/bin/env python3
"""DEPRECATED — Replaced by scripts/eval_benchmarks.py

The majority_vote_select logic has been migrated to src/game_protocol.py.
Use eval_benchmarks.py for all evaluation needs:
    python scripts/eval_benchmarks.py --benchmarks gsm8k truthfulqa mt_bench

Original purpose:
Evaluate GameRefine outputs on GSM8K, TruthfulQA, and MT-Bench style tasks.
Compares: individual agents, Nash-equilibrium merged, and majority-vote selection.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.game_protocol import AgentRole, compute_agent_reward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_gamerefine")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GameRefine models")
    parser.add_argument("--config", type=str, default="configs/agent_roles.yaml")
    parser.add_argument("--agents_dir", type=str, default=None,
                        help="Directory with agent checkpoints (individual or self-play)")
    parser.add_argument("--self_play_dir", type=str, default=None,
                        help="Directory with self-play results (for latest round agents)")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--eval_gsm8k", action="store_true")
    parser.add_argument("--eval_truthfulqa", action="store_true")
    parser.add_argument("--eval_mt_bench", action="store_true")
    parser.add_argument("--eval_all", action="store_true")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_agent_model(agent_path: str, base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(agent_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    try:
        model = PeftModel.from_pretrained(model, agent_path)
        logger.info(f"Loaded LoRA from {agent_path}")
    except Exception:
        logger.info(f"No LoRA found at {agent_path}, using base model")

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_responses(model, tokenizer, prompts, max_new_tokens, batch_size):
    all_responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1024,
        ).to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        for j, output in enumerate(outputs):
            plen = inputs["input_ids"][j].shape[0]
            text = tokenizer.decode(output[plen:], skip_special_tokens=True)
            all_responses.append(text)
    return all_responses


def extract_answer(text: str) -> str:
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    return nums[-1].replace(",", "") if nums else ""


# ── GSM8K Evaluation ─────────────────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, num_samples, batch_size, max_new_tokens):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    samples = list(ds)[:num_samples]

    prompts = [
        f"Solve step by step. End with '#### <answer>'.\n\nQuestion: {s['question']}\n\nAnswer:"
        for s in samples
    ]

    responses = generate_responses(model, tokenizer, prompts, max_new_tokens, batch_size)

    correct = 0
    for sample, response in zip(samples, responses):
        gold = extract_answer(sample["answer"])
        pred = extract_answer(response)
        if gold and pred and gold.strip() == pred.strip():
            correct += 1

    accuracy = correct / max(len(samples), 1)
    return {"gsm8k_accuracy": accuracy, "gsm8k_n": len(samples)}


# ── TruthfulQA Evaluation ───────────────────────────────────────────────────

def eval_truthfulqa(model, tokenizer, num_samples, batch_size, max_new_tokens):
    ds = load_dataset("truthful_qa", "generation", split="validation")
    samples = list(ds)[:num_samples]

    prompts = [
        f"Answer truthfully:\n\nQuestion: {s['question']}\n\nAnswer:"
        for s in samples
    ]

    responses = generate_responses(model, tokenizer, prompts, max_new_tokens, batch_size)

    correct = 0
    for sample, response in zip(samples, responses):
        resp_lower = response.lower()
        is_correct = True
        for inc in sample.get("incorrect_answers", []):
            if inc.lower() in resp_lower:
                is_correct = False
                break
        if is_correct:
            correct += 1

    accuracy = correct / max(len(samples), 1)
    return {"truthfulqa_accuracy": accuracy, "truthfulqa_n": len(samples)}


# ── MT-Bench Style Evaluation ────────────────────────────────────────────────

MT_BENCH_QUESTIONS = [
    "Write a persuasive argument for renewable energy.",
    "Explain recursion to a 10 year old.",
    "What are the pros and cons of remote work?",
    "Write Python code to find the nth Fibonacci number efficiently.",
    "Compare democracy and authoritarianism objectively.",
    "How does the immune system fight infections?",
    "Create a short story about an AI discovering emotions.",
    "Solve: If 3x + 7 = 22, what is x?",
]


def eval_mt_bench_style(model, tokenizer, batch_size, max_new_tokens):
    """Evaluate on MT-Bench-style questions using heuristic quality scores."""
    prompts = [f"Question: {q}\n\nAnswer:" for q in MT_BENCH_QUESTIONS]
    responses = generate_responses(model, tokenizer, prompts, max_new_tokens, batch_size)

    scores = []
    for response in responses:
        words = response.split()
        length_score = min(len(words) / 100, 1.0) * 3
        unique_ratio = len(set(words)) / max(len(words), 1)
        diversity_score = unique_ratio * 3
        structure_score = (
            (1 if "\n" in response else 0) +
            (1 if any(c in response for c in ["1.", "2.", "-", "*"]) else 0) +
            (1 if len(response) > 50 else 0)
        )
        total = min(length_score + diversity_score + structure_score, 10.0)
        scores.append(total)

    return {
        "mt_bench_avg_score": sum(scores) / max(len(scores), 1),
        "mt_bench_scores": scores,
        "mt_bench_n": len(scores),
    }


def majority_vote_select(agent_responses: dict, agent_roles: dict) -> list[str]:
    """Select best response via multi-agent majority vote."""
    num_prompts = len(next(iter(agent_responses.values())))
    selected = []

    for i in range(num_prompts):
        best_response = None
        best_score = -float("inf")

        for agent_id in agent_responses:
            response = agent_responses[agent_id][i]
            total_score = 0
            for eval_id, eval_role in agent_roles.items():
                total_score += compute_agent_reward(response, eval_role.reward_weights)
            if total_score > best_score:
                best_score = total_score
                best_response = response

        selected.append(best_response or "")

    return selected


def main():
    args = parse_args()
    cfg = load_config(args.config)

    os.makedirs(args.output_dir, exist_ok=True)

    do_gsm = args.eval_gsm8k or args.eval_all
    do_tqa = args.eval_truthfulqa or args.eval_all
    do_mt = args.eval_mt_bench or args.eval_all
    if not any([do_gsm, do_tqa, do_mt]):
        do_gsm = do_tqa = do_mt = True

    base_model = cfg["model"]["base"]
    agents_dir = args.agents_dir or cfg["output"]["agents_dir"]
    agent_ids = list(cfg["agents"].keys())

    agent_roles = {}
    for aid, acfg in cfg["agents"].items():
        agent_roles[aid] = AgentRole(
            name=acfg["name"], role_id=aid,
            description=acfg["description"],
            reward_weights=acfg["reward_weights"],
            eval_prompt_suffix=acfg["eval_prompt_suffix"],
        )

    all_metrics = {"agents": {}, "majority_vote": {}}

    # Evaluate each agent individually
    for agent_id in agent_ids:
        agent_path = os.path.join(agents_dir, agent_id)
        if not os.path.exists(agent_path):
            logger.warning(f"Agent {agent_id} not found at {agent_path}")
            continue

        logger.info(f"\nEvaluating agent: {agent_id}")
        model, tokenizer = load_agent_model(agent_path, base_model)

        agent_metrics = {"agent_id": agent_id}

        if do_gsm:
            gsm_results = eval_gsm8k(model, tokenizer, args.num_samples,
                                      args.batch_size, args.max_new_tokens)
            agent_metrics.update(gsm_results)
            logger.info(f"  GSM8K: {gsm_results['gsm8k_accuracy']:.4f}")

        if do_tqa:
            tqa_results = eval_truthfulqa(model, tokenizer, args.num_samples,
                                           args.batch_size, args.max_new_tokens)
            agent_metrics.update(tqa_results)
            logger.info(f"  TruthfulQA: {tqa_results['truthfulqa_accuracy']:.4f}")

        if do_mt:
            mt_results = eval_mt_bench_style(model, tokenizer, args.batch_size,
                                              args.max_new_tokens)
            agent_metrics.update(mt_results)
            logger.info(f"  MT-Bench: {mt_results['mt_bench_avg_score']:.2f}")

        all_metrics["agents"][agent_id] = agent_metrics

        del model
        torch.cuda.empty_cache()

    # Save results
    output_path = os.path.join(args.output_dir, "eval_gamerefine.json")
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"{'Agent':>15} {'GSM8K':>10} {'TruthfulQA':>12} {'MT-Bench':>10}")
    print("-" * 70)
    for agent_id, metrics in all_metrics["agents"].items():
        gsm = metrics.get("gsm8k_accuracy", "-")
        tqa = metrics.get("truthfulqa_accuracy", "-")
        mt = metrics.get("mt_bench_avg_score", "-")
        gsm_str = f"{gsm:.4f}" if isinstance(gsm, float) else str(gsm)
        tqa_str = f"{tqa:.4f}" if isinstance(tqa, float) else str(tqa)
        mt_str = f"{mt:.2f}" if isinstance(mt, float) else str(mt)
        print(f"{agent_id:>15} {gsm_str:>10} {tqa_str:>12} {mt_str:>10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
