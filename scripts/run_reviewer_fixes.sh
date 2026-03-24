#!/usr/bin/env bash
# ============================================================================
# Post-pipeline experiments addressing reviewer concerns
#
# Run AFTER run_all_experiments.sh completes.
# Addresses: W1 (factorial design), W2 (base model eval), W4 (baselines)
#
# Usage:
#   bash scripts/run_reviewer_fixes.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_utils.sh"
auto_setup

PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

GAME_CONFIG="configs/game_scenarios.yaml"
ROLE_CONFIG="configs/agent_roles.yaml"
RESULTS_DIR="results"
LOG_DIR="logs"
SEED=42
EVAL_EPISODES=20

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
run_timed() {
    local name="$1"; shift
    log "START: $name"
    local start=$SECONDS
    "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
    local elapsed=$(( SECONDS - start ))
    log "DONE:  $name (${elapsed}s)"
}

# ============================================================================
# W2 Fix: Base Model Evaluation (the "no training" baseline)
# ============================================================================

log "========== Baseline: Base Model Evaluation =========="

run_timed "E1_base_model_benchmarks" \
    python scripts/eval_benchmarks.py \
        --game_config "$GAME_CONFIG" \
        --model_dirs "base:Qwen/Qwen3.5-9B" \
        --output_dir "$RESULTS_DIR/eval_base_model" \
        --benchmarks arc strategyqa bbh gsm8k truthfulqa mt_bench

run_timed "E2_base_model_games" \
    python scripts/eval_game_performance.py \
        --config "$GAME_CONFIG" \
        --model_dir "Qwen/Qwen3.5-9B" \
        --output_dir "$RESULTS_DIR/eval_base_games" \
        --num_episodes "$EVAL_EPISODES" \
        --seed "$SEED"

# ============================================================================
# W4 Fix: SFT-Only Baseline (extra data but no self-play)
# ============================================================================

log "========== Baseline: SFT-Only Evaluation =========="

run_timed "E3_sft_only_benchmarks" \
    python scripts/eval_benchmarks.py \
        --game_config "$GAME_CONFIG" \
        --model_dirs "sft_agent0:$RESULTS_DIR/sft_agents/agent_0/final sft_agent1:$RESULTS_DIR/sft_agents/agent_1/final" \
        --output_dir "$RESULTS_DIR/eval_sft_only" \
        --benchmarks arc strategyqa bbh gsm8k truthfulqa mt_bench

# ============================================================================
# W1 Fix: Factorial Design — A+B Combined Training
# Apply Nash-DPO on top of GRPO-trained model
# ============================================================================

log "========== Factorial: A+B Combined (Nash-DPO on GRPO model) =========="

if [ -d "$RESULTS_DIR/grpo_self_play/final/agent_0" ]; then
    run_timed "E4_combined_AB_nash_dpo" \
        python scripts/train_nash_dpo.py \
            --config "$ROLE_CONFIG" \
            --agents_dir "$RESULTS_DIR/grpo_self_play/final" \
            --output_dir "$RESULTS_DIR/combined_AB" \
            --num_iterations 2 \
            --games_per_iter 200 \
            --dpo_epochs 1 \
            --beta 0.1 \
            --seed "$SEED"

    run_timed "E5_combined_AB_benchmarks" \
        python scripts/eval_benchmarks.py \
            --game_config "$GAME_CONFIG" \
            --model_dirs "combined_AB:$RESULTS_DIR/combined_AB/iter1/accuracy" \
            --output_dir "$RESULTS_DIR/eval_combined_AB" \
            --benchmarks arc strategyqa bbh gsm8k truthfulqa mt_bench
else
    log "SKIP: E4/E5 — GRPO models not yet available"
fi

# ============================================================================
# W4 Fix: Plain DPO Baseline (equal weights, no Nash bargaining)
# ============================================================================

log "========== Baseline: Plain DPO (equal weights) =========="

if [ -d "$RESULTS_DIR/sft_role_agents" ]; then
    run_timed "E6_plain_dpo" \
        python scripts/train_nash_dpo.py \
            --config "$ROLE_CONFIG" \
            --agents_dir "$RESULTS_DIR/sft_role_agents" \
            --output_dir "$RESULTS_DIR/plain_dpo_baseline" \
            --num_iterations 2 \
            --games_per_iter 200 \
            --dpo_epochs 1 \
            --beta 0.1 \
            --nash_weights "equal" \
            --seed "$SEED"

    run_timed "E7_plain_dpo_benchmarks" \
        python scripts/eval_benchmarks.py \
            --game_config "$GAME_CONFIG" \
            --model_dirs "plain_dpo:$RESULTS_DIR/plain_dpo_baseline/iter1/accuracy" \
            --output_dir "$RESULTS_DIR/eval_plain_dpo" \
            --benchmarks arc strategyqa bbh gsm8k truthfulqa mt_bench
else
    log "SKIP: E6/E7 — SFT role agents not yet available"
fi

# ============================================================================
# Summary
# ============================================================================

log "============================================="
log "Reviewer fix experiments complete."
log ""
log "Factorial Design (W1):"
log "  Base:      $RESULTS_DIR/eval_base_model/"
log "  A-only:    $RESULTS_DIR/eval_benchmarks/ (from main pipeline)"
log "  B-only:    $RESULTS_DIR/eval_benchmarks/ (from main pipeline)"
log "  A+B:       $RESULTS_DIR/eval_combined_AB/"
log ""
log "Baselines (W4):"
log "  SFT-only:  $RESULTS_DIR/eval_sft_only/"
log "  Plain DPO: $RESULTS_DIR/eval_plain_dpo/"
log "============================================="
