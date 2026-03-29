#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="Qwen/Qwen3.5-9B"
OUTPUT_ROOT="./results/factorial_v2"
SEEDS=(42 123 456)
GAMES="pd bos public_goods ultimatum"

mkdir -p "$OUTPUT_ROOT/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$OUTPUT_ROOT/logs/master.log"; }

log "=== GameAgent Factorial Experiment Pipeline ==="
log "Base model: $BASE_MODEL"
log "Seeds: ${SEEDS[*]}"
log "Games: $GAMES"

# ── Phase 0: Data Generation ─────────────────────────────────────────────────

log "Phase 0: Generating game data and SFT data..."
for game in $GAMES; do
    python scripts/generate_sft_data.py \
        --game "$game" \
        --num_episodes 200 \
        --output_dir "$OUTPUT_ROOT/data/sft_${game}" \
        --base_model "$BASE_MODEL" \
        2>&1 | tee "$OUTPUT_ROOT/logs/data_${game}.log"
done

# ── Phase 1: SFT Warmup ─────────────────────────────────────────────────────

log "Phase 1: SFT warmup training..."
for seed in "${SEEDS[@]}"; do
    python scripts/train_sft_warmup.py \
        --base_model "$BASE_MODEL" \
        --data_dir "$OUTPUT_ROOT/data" \
        --output_dir "$OUTPUT_ROOT/sft/seed${seed}" \
        --seed "$seed" \
        --epochs 2 \
        2>&1 | tee "$OUTPUT_ROOT/logs/sft_seed${seed}.log"
done

# ── Phase 2: Condition A-only (GRPO Self-Play) ──────────────────────────────

log "Phase 2: GRPO self-play (Track A)..."
for seed in "${SEEDS[@]}"; do
    python scripts/run_grpo_self_play.py \
        --base_model "$BASE_MODEL" \
        --agents_dir "$OUTPUT_ROOT/sft/seed${seed}" \
        --output_dir "$OUTPUT_ROOT/grpo_only/seed${seed}" \
        --num_iterations 3 \
        --games_per_iter 500 \
        --seed "$seed" \
        2>&1 | tee "$OUTPUT_ROOT/logs/grpo_seed${seed}.log"
done

# ── Phase 3: Condition B-only (Nash-DPO) ────────────────────────────────────

log "Phase 3: Nash-DPO alignment (Track B)..."
for seed in "${SEEDS[@]}"; do

    log "  Estimating disagreement point (seed=$seed)..."
    python -c "
import json, sys
sys.path.insert(0, '.')
from scripts.train_formal_nash_dpo import estimate_disagreement_point
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained('$BASE_MODEL', torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('$BASE_MODEL', trust_remote_code=True)
prompts = ['Explain quantum computing.', 'Write a poem about nature.', 'How to stay healthy?'] * 34
d = estimate_disagreement_point(model, tokenizer, prompts, n_samples=100)
with open('$OUTPUT_ROOT/nash_dpo_only/seed${seed}/disagreement.json', 'w') as f:
    json.dump(d, f, indent=2)
" 2>&1 | tee "$OUTPUT_ROOT/logs/disagreement_seed${seed}.log"

    for method in nash equal fixed single_correctness; do
        log "  Training Nash-DPO method=$method seed=$seed..."
        python scripts/train_formal_nash_dpo.py \
            --base_model "$BASE_MODEL" \
            --preference_data "$OUTPUT_ROOT/data/preference_pairs.jsonl" \
            --output_dir "$OUTPUT_ROOT/nash_dpo_only/seed${seed}/${method}" \
            --method "$method" \
            --beta 0.1 \
            --ema_tau 0.1 \
            --warmup_steps 100 \
            --seed "$seed" \
            --disagreement_file "$OUTPUT_ROOT/nash_dpo_only/seed${seed}/disagreement.json" \
            2>&1 | tee "$OUTPUT_ROOT/logs/nash_dpo_${method}_seed${seed}.log"
    done
done

# ── Phase 4: Condition A+B (GRPO → Nash-DPO) ────────────────────────────────

log "Phase 4: Combined A+B..."
for seed in "${SEEDS[@]}"; do
    python scripts/train_formal_nash_dpo.py \
        --base_model "$OUTPUT_ROOT/grpo_only/seed${seed}/iter2" \
        --preference_data "$OUTPUT_ROOT/data/preference_pairs.jsonl" \
        --output_dir "$OUTPUT_ROOT/combined/seed${seed}" \
        --method nash \
        --seed "$seed" \
        --disagreement_file "$OUTPUT_ROOT/nash_dpo_only/seed${seed}/disagreement.json" \
        2>&1 | tee "$OUTPUT_ROOT/logs/combined_seed${seed}.log"
done

# ── Phase 5: Evaluation ─────────────────────────────────────────────────────

log "Phase 5: Evaluating all conditions..."

CONDITIONS=(
    "base:$BASE_MODEL"
    "sft:$OUTPUT_ROOT/sft/seed42"
    "a_only:$OUTPUT_ROOT/grpo_only/seed42/iter2"
    "b_only:$OUTPUT_ROOT/nash_dpo_only/seed42/nash"
    "ab:$OUTPUT_ROOT/combined/seed42"
)

for cond_path in "${CONDITIONS[@]}"; do
    IFS=: read -r cond_name model_path <<< "$cond_path"
    log "  Evaluating condition=$cond_name..."

    python scripts/eval_game_performance.py \
        --model_path "$model_path" \
        --games $GAMES \
        --output_dir "$OUTPUT_ROOT/eval/${cond_name}" \
        2>&1 | tee "$OUTPUT_ROOT/logs/eval_game_${cond_name}.log"

    python scripts/eval_benchmarks.py \
        --model_path "$model_path" \
        --benchmarks gtbench strategyqa truthfulqa mt_bench \
        --output_dir "$OUTPUT_ROOT/eval/${cond_name}" \
        2>&1 | tee "$OUTPUT_ROOT/logs/eval_bench_${cond_name}.log"
done

# ── Phase 6: Analysis ────────────────────────────────────────────────────────

log "Phase 6: Collecting results and generating figures..."
python scripts/collect_and_visualize.py \
    --results_dir "$OUTPUT_ROOT/eval" \
    --output_dir "$OUTPUT_ROOT/paper_figures" \
    2>&1 | tee "$OUTPUT_ROOT/logs/analysis.log"

log "=== All experiments complete! ==="
log "Results: $OUTPUT_ROOT/eval/"
log "Figures: $OUTPUT_ROOT/paper_figures/"
