#!/usr/bin/env python3
"""Generate multi-objective preference data for Nash-DPO training.

For each prompt, generates two candidate responses from the model, evaluates
both with 4 objective reward functions, and emits a preference pair with
per-objective signals.

Output: JSONL with fields
  prompt, chosen, rejected, pref_correctness, pref_safety,
  pref_efficiency, pref_creativity, scores_chosen, scores_rejected
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.reward_models import (
    compute_correctness_reward_robust,
    compute_safety_reward_robust,
    compute_efficiency_reward_robust,
    compute_creativity_reward_robust,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_preference_data")

OBJECTIVE_FNS = {
    "correctness": compute_correctness_reward_robust,
    "safety": compute_safety_reward_robust,
    "efficiency": compute_efficiency_reward_robust,
    "creativity": compute_creativity_reward_robust,
}

PROMPT_TEMPLATES = [
    "Explain the concept of {topic} in simple terms.",
    "What are the main advantages and disadvantages of {topic}?",
    "Describe how {topic} works and give an example.",
    "Compare and contrast {topic_a} with {topic_b}.",
    "Write a brief analysis of {topic}.",
    "Summarize the key principles behind {topic}.",
    "What are common misconceptions about {topic}?",
    "How would you teach {topic} to a beginner?",
]

TOPICS = [
    "machine learning", "neural networks", "game theory",
    "Nash equilibrium", "reinforcement learning", "natural language processing",
    "computer vision", "optimization", "statistical inference",
    "decision theory", "multi-agent systems", "cooperative games",
    "mechanism design", "auction theory", "social choice theory",
    "Bayesian inference", "causal inference", "information theory",
    "graph neural networks", "attention mechanisms",
]


def generate_prompts(n_prompts: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    prompts = []
    for _ in range(n_prompts):
        template = rng.choice(PROMPT_TEMPLATES)
        if "{topic_a}" in template:
            a, b = rng.sample(TOPICS, 2)
            prompt = template.format(topic_a=a, topic_b=b)
        else:
            prompt = template.format(topic=rng.choice(TOPICS))
        prompts.append(prompt)
    return prompts


def generate_response(model, tokenizer, prompt, temperature=0.7, max_new_tokens=256):
    chat_prompt = (
        f"<|im_start|>system\nYou are a helpful, accurate, and creative assistant.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True,
                       max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True)


def evaluate_response(response: str, reference: str = None) -> dict[str, float]:
    scores = {}
    for name, fn in OBJECTIVE_FNS.items():
        try:
            if name in ("correctness", "efficiency") and reference:
                scores[name] = fn(response, reference)
            else:
                scores[name] = fn(response)
        except Exception:
            scores[name] = 0.5
    return scores


def create_preference_pair(prompt, resp_a, resp_b, reference=None):
    scores_a = evaluate_response(resp_a, reference)
    scores_b = evaluate_response(resp_b, reference)

    a_total = sum(scores_a.values())
    b_total = sum(scores_b.values())

    if a_total >= b_total:
        chosen, rejected = resp_a, resp_b
        sc, sr = scores_a, scores_b
    else:
        chosen, rejected = resp_b, resp_a
        sc, sr = scores_b, scores_a

    pref = {}
    for name in OBJECTIVE_FNS:
        diff = sc[name] - sr[name]
        if diff > 0.05:
            pref[f"pref_{name}"] = 1.0
        elif diff < -0.05:
            pref[f"pref_{name}"] = -1.0
        else:
            pref[f"pref_{name}"] = 0.0

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        **pref,
        "scores_chosen": sc,
        "scores_rejected": sr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="data/preference_pairs.jsonl")
    parser.add_argument("--n_prompts", type=int, default=1000)
    parser.add_argument("--temperature_a", type=float, default=0.7,
                        help="Temperature for candidate A")
    parser.add_argument("--temperature_b", type=float, default=1.0,
                        help="Higher temperature for candidate B (more diverse)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default=None,
                        help="LoRA adapter path (optional, for fine-tuned models)")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model: %s", args.base_model)
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
        from peft import PeftModel
        logger.info("Loading LoRA adapter: %s", args.model_path)
        model = PeftModel.from_pretrained(model, args.model_path)

    model.eval()

    prompts = generate_prompts(args.n_prompts, args.seed)
    logger.info("Generated %d prompts", len(prompts))

    records = []
    for i, prompt in enumerate(prompts):
        resp_a = generate_response(model, tokenizer, prompt,
                                   temperature=args.temperature_a,
                                   max_new_tokens=args.max_new_tokens)
        resp_b = generate_response(model, tokenizer, prompt,
                                   temperature=args.temperature_b,
                                   max_new_tokens=args.max_new_tokens)

        if not resp_a.strip() or not resp_b.strip():
            continue

        record = create_preference_pair(prompt, resp_a, resp_b)
        records.append(record)

        if (i + 1) % 50 == 0:
            logger.info("Progress: %d/%d (%d valid pairs)",
                        i + 1, len(prompts), len(records))

    with open(args.output_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    logger.info("Saved %d preference pairs to %s", len(records), args.output_file)

    obj_stats = {name: {"agree": 0, "disagree": 0, "neutral": 0}
                 for name in OBJECTIVE_FNS}
    for r in records:
        for name in OBJECTIVE_FNS:
            v = r[f"pref_{name}"]
            if v > 0:
                obj_stats[name]["agree"] += 1
            elif v < 0:
                obj_stats[name]["disagree"] += 1
            else:
                obj_stats[name]["neutral"] += 1
    logger.info("Per-objective agreement stats:")
    for name, stats in obj_stats.items():
        logger.info("  %s: agree=%d disagree=%d neutral=%d",
                    name, stats["agree"], stats["disagree"], stats["neutral"])

    stats_file = args.output_file.replace(".jsonl", "_stats.json")
    with open(stats_file, "w") as f:
        json.dump({"n_pairs": len(records), "per_objective": obj_stats}, f, indent=2)


if __name__ == "__main__":
    main()
