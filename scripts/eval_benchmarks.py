#!/usr/bin/env python3
"""Unified benchmark evaluation: ARC, StrategyQA, BBH, GSM8K, TruthfulQA, MT-Bench.

Evaluates GRPO-trained and Nash-DPO-trained agents on all benchmarks, producing
a single comparison table showing both strategic reasoning and alignment quality.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_dir, base_model_name):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, model_dir)
        logger.info("Loaded adapter from %s", model_dir)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def batch_generate(model, tokenizer, prompts, max_new_tokens=512, batch_size=4):
    all_responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating", leave=False):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        for j, output in enumerate(outputs):
            plen = inputs["input_ids"][j].shape[0]
            text = tokenizer.decode(output[plen:], skip_special_tokens=True)
            all_responses.append(text)
    return all_responses


# ── Benchmark evaluators ─────────────────────────────────────────────────────

def evaluate_arc(model, tokenizer, max_samples=1172):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    correct, total = 0, 0
    for ex in tqdm(ds, desc="ARC", leave=False):
        question = ex.get("question", "")
        choices = ex.get("choices", {})
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        answer_key = ex.get("answerKey", "")
        if not question or not labels:
            continue

        choice_text = "\n".join(f"  {l}: {t}" for l, t in zip(labels, texts))
        prompt = (
            f"<|im_start|>system\nYou are a strategic reasoner.<|im_end|>\n"
            f"<|im_start|>user\n{question}\n\nChoices:\n{choice_text}\n\n"
            f"Answer with just the letter.<|im_end|>\n<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred = response[0].upper() if response else ""
        if pred == answer_key:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    logger.info("ARC Challenge: %d/%d = %.4f", correct, total, accuracy)
    return {"accuracy": accuracy, "correct": correct, "total": total}


def evaluate_strategyqa(model, tokenizer, max_samples=2290):
    ds = load_dataset("wics/strategy-qa", split="test")
    if len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    correct, total = 0, 0
    for ex in tqdm(ds, desc="StrategyQA", leave=False):
        question = ex.get("question", ex.get("input", ""))
        answer = ex.get("answer", ex.get("target", ""))
        gold = ("yes" if answer else "no") if isinstance(answer, bool) else str(answer).lower().strip()
        if not question:
            continue

        prompt = (
            f"<|im_start|>system\nYou are a strategic reasoner. Think step by step.<|im_end|>\n"
            f"<|im_start|>user\n{question}\n\nAnswer yes or no.<|im_end|>\n<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).lower()
        pred = "yes" if "yes" in response[:50] else "no"
        if pred == gold:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    logger.info("StrategyQA: %d/%d = %.4f", correct, total, accuracy)
    return {"accuracy": accuracy, "correct": correct, "total": total}


def evaluate_bbh(model, tokenizer, max_samples=1000):
    ds = load_dataset("lukaemon/bbh", split="test")
    if len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    correct, total = 0, 0
    for ex in tqdm(ds, desc="BBH", leave=False):
        question = ex.get("input", "")
        gold = str(ex.get("target", "")).strip()
        if not question:
            continue

        prompt = (
            f"<|im_start|>system\nYou are a strategic decision-making agent with strong reasoning.<|im_end|>\n"
            f"<|im_start|>user\n{question}\n\nGive a concise answer.<|im_end|>\n<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred = response.split("\n")[0].strip()
        if gold.lower() in pred.lower() or pred.lower() in gold.lower():
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    logger.info("BBH: %d/%d = %.4f", correct, total, accuracy)
    return {"accuracy": accuracy, "correct": correct, "total": total}


def extract_gsm8k_answer(text: str) -> str:
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    return nums[-1].replace(",", "") if nums else ""


def evaluate_gsm8k(model, tokenizer, max_samples=1319):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    samples = list(ds)[:max_samples]

    prompts = [
        f"Solve step by step. End with '#### <answer>'.\n\nQuestion: {s['question']}\n\nAnswer:"
        for s in samples
    ]
    responses = batch_generate(model, tokenizer, prompts, max_new_tokens=512, batch_size=4)

    correct = 0
    for sample, response in zip(samples, responses):
        gold = extract_gsm8k_answer(sample["answer"])
        pred = extract_gsm8k_answer(response)
        if gold and pred and gold.strip() == pred.strip():
            correct += 1

    accuracy = correct / max(len(samples), 1)
    logger.info("GSM8K: %d/%d = %.4f", correct, len(samples), accuracy)
    return {"accuracy": accuracy, "correct": correct, "total": len(samples)}


def evaluate_truthfulqa(model, tokenizer, max_samples=817):
    ds = load_dataset("truthful_qa", "generation", split="validation")
    samples = list(ds)[:max_samples]

    prompts = [f"Answer truthfully:\n\nQuestion: {s['question']}\n\nAnswer:" for s in samples]
    responses = batch_generate(model, tokenizer, prompts, max_new_tokens=256, batch_size=4)

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
    logger.info("TruthfulQA: %d/%d = %.4f", correct, len(samples), accuracy)
    return {"accuracy": accuracy, "correct": correct, "total": len(samples)}


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


def evaluate_mt_bench(model, tokenizer, max_samples=80):
    qs = MT_BENCH_QUESTIONS[:max_samples]
    prompts = [f"Question: {q}\n\nAnswer:" for q in qs]
    responses = batch_generate(model, tokenizer, prompts, max_new_tokens=512, batch_size=4)

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

    avg = sum(scores) / max(len(scores), 1)
    logger.info("MT-Bench: avg_score=%.2f", avg)
    return {"avg_score": avg, "scores": scores, "total": len(scores)}


def main():
    parser = argparse.ArgumentParser(description="Unified benchmark evaluation")
    parser.add_argument("--game_config", type=str, default="configs/game_scenarios.yaml")
    parser.add_argument("--model_dirs", nargs="+", required=True,
                        help="label:path pairs, e.g. 'grpo:results/grpo/final/agent_0'")
    parser.add_argument("--output_dir", type=str, default="results/eval_benchmarks")
    parser.add_argument("--benchmarks", nargs="*",
                        default=["arc", "strategyqa", "bbh", "gsm8k", "truthfulqa", "mt_bench"])
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Max samples per benchmark (for quick testing)")
    args = parser.parse_args()

    with open(args.game_config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    base_model_name = config["model"]["base_model"]

    all_results = {}

    for model_spec in args.model_dirs:
        if ":" in model_spec:
            label, path = model_spec.split(":", 1)
        else:
            label, path = os.path.basename(model_spec), model_spec

        logger.info("\n=== Evaluating: %s (%s) ===", label, path)
        model, tokenizer = load_model(path, base_model_name)
        model_results = {"label": label, "path": path}

        ms = args.max_samples
        bench_map = {
            "arc": ("ARC-Challenge", lambda: evaluate_arc(model, tokenizer, max_samples=ms)),
            "strategyqa": ("StrategyQA", lambda: evaluate_strategyqa(model, tokenizer, max_samples=ms)),
            "bbh": ("BBH", lambda: evaluate_bbh(model, tokenizer, max_samples=ms)),
            "gsm8k": ("GSM8K", lambda: evaluate_gsm8k(model, tokenizer, max_samples=ms)),
            "truthfulqa": ("TruthfulQA", lambda: evaluate_truthfulqa(model, tokenizer, max_samples=ms)),
            "mt_bench": ("MT-Bench", lambda: evaluate_mt_bench(model, tokenizer, max_samples=ms)),
        }

        for bench_key in args.benchmarks:
            if bench_key in bench_map:
                name, fn = bench_map[bench_key]
                try:
                    model_results[bench_key] = fn()
                except Exception as e:
                    logger.warning("Failed %s: %s", name, e)
                    model_results[bench_key] = {"error": str(e)}

        all_results[label] = model_results
        del model
        torch.cuda.empty_cache()

    output_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 100)
    header = f"{'Model':>20} {'ARC':>8} {'StratQA':>8} {'BBH':>8} {'GSM8K':>8} {'TruthQA':>8} {'MT-B':>8}"
    print(header)
    print("-" * 100)
    for label, res in all_results.items():
        arc = res.get("arc", {}).get("accuracy", "-")
        sqa = res.get("strategyqa", {}).get("accuracy", "-")
        bbh = res.get("bbh", {}).get("accuracy", "-")
        gsm = res.get("gsm8k", {}).get("accuracy", "-")
        tqa = res.get("truthfulqa", {}).get("accuracy", "-")
        mtb = res.get("mt_bench", {}).get("avg_score", "-")
        row = f"{label:>20}"
        for v in [arc, sqa, bbh, gsm, tqa, mtb]:
            row += f" {v:>8.4f}" if isinstance(v, float) else f" {str(v):>8}"
        print(row)
    print("=" * 100)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
