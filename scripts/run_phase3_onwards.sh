#!/usr/bin/env bash
# ============================================================================
# Resume pipeline from Phase 3 (Nash-DPO) onward
# Phase 1 (Expert Data) and Phase 2 (Role SFT) already completed.
# ============================================================================
set -euo pipefail

cd /gfs/space/private/wujn/Research/nips-gameagent
# HF_ENDPOINT removed (use default huggingface.co)
export TOKENIZERS_PARALLELISM=false

GAME_CFG="configs/game_scenarios.yaml"
ROLE_CFG="configs/agent_roles.yaml"
RESULTS="results"
LOG="logs"
SEED=42
mkdir -p "$RESULTS" "$LOG"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ============================================================================
# Phase 3: Nash-DPO Self-Play + Parallel DPO (B3)
# Self-play uses all GPUs via device_map=auto
# DPO launches 4 subprocesses, one per GPU per agent
# ============================================================================
log "===== Phase 3: Nash-DPO (self-play + parallel DPO) ====="

python scripts/train_nash_dpo.py \
    --config "$ROLE_CFG" \
    --agents_dir "$RESULTS/sft_role_agents" \
    --output_dir "$RESULTS/nash_dpo" \
    --num_iterations 2 \
    --games_per_iter 200 \
    --dpo_epochs 1 --dpo_batch_size 8 \
    --beta 0.1 --seed $SEED \
    > "$LOG/B3_nash_dpo.log" 2>&1

log "Phase 3 done."

# ============================================================================
# Phase 4: All evaluations in parallel — 4 GPUs
# ============================================================================
log "===== Phase 4: Evaluations (4 GPUs parallel) ====="

NASH_ITER=$((2 - 1))

CUDA_VISIBLE_DEVICES=0 python scripts/run_grpo_vs_nash_comparison.py \
    --game_config "$GAME_CFG" \
    --grpo_model "$RESULTS/grpo_self_play/final/agent_0" \
    --nash_model "$RESULTS/nash_dpo/iter${NASH_ITER}/accuracy" \
    --output_dir "$RESULTS/grpo_vs_nash" \
    --game_episodes 20 --seed $SEED \
    > "$LOG/C1_comparison.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 python scripts/eval_benchmarks.py \
    --game_config "$GAME_CFG" \
    --model_dirs \
        "grpo:$RESULTS/grpo_self_play/final/agent_0" \
        "nash_dpo:$RESULTS/nash_dpo/iter${NASH_ITER}/accuracy" \
    --output_dir "$RESULTS/eval_benchmarks" \
    --benchmarks arc strategyqa bbh gsm8k truthfulqa mt_bench \
    > "$LOG/C2_benchmarks.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 python scripts/run_cross_game_transfer.py \
    --config "$GAME_CFG" \
    --data_dir data \
    --output_dir "$RESULTS/cross_game_transfer" \
    --num_epochs 2 --eval_episodes 20 \
    --skip_single --seed $SEED \
    > "$LOG/A4_cross_transfer.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 python scripts/run_grpo_self_play.py \
    --config "$GAME_CFG" \
    --sft_dir "__nonexistent__" \
    --output_dir "$RESULTS/ablation_no_sft" \
    --num_iterations 2 \
    --episodes_per_iter 1000 \
    --eval_episodes 10 --batch_size 128 \
    --seed $SEED \
    > "$LOG/D1_ablation_no_sft.log" 2>&1 &

wait
log "Phase 4 done."

# ============================================================================
# Phase 5: Additional Ablations
# D2 (GRPO, GPU 0) runs in parallel with D3 (Nash-DPO 1-iter, GPUs 1-3)
# D4 (Nash-DPO equal weights) runs after both finish using all GPUs
# ============================================================================
log "===== Phase 5: Ablations ====="

CUDA_VISIBLE_DEVICES=0 python scripts/run_grpo_self_play.py \
    --config "$GAME_CFG" \
    --sft_dir "$RESULTS/sft_agents" \
    --output_dir "$RESULTS/ablation_2iter_grpo" \
    --num_iterations 2 \
    --episodes_per_iter 2000 \
    --eval_episodes 20 --batch_size 256 \
    --seed $SEED \
    > "$LOG/D2_ablation_2iter.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1,2,3 python scripts/train_nash_dpo.py \
    --config "$ROLE_CFG" \
    --agents_dir "$RESULTS/sft_role_agents" \
    --output_dir "$RESULTS/ablation_1iter_nash" \
    --num_iterations 1 \
    --games_per_iter 200 \
    --dpo_epochs 1 --dpo_batch_size 8 \
    --seed $SEED \
    > "$LOG/D3_ablation_1iter_nash.log" 2>&1 &

wait
log "Phase 5a done (D2 + D3)."

python scripts/train_nash_dpo.py \
    --config "$ROLE_CFG" \
    --agents_dir "$RESULTS/sft_role_agents" \
    --output_dir "$RESULTS/ablation_equal_weights" \
    --num_iterations 2 \
    --games_per_iter 200 \
    --dpo_epochs 1 --dpo_batch_size 8 \
    --nash_weights equal \
    --seed $SEED \
    > "$LOG/D4_ablation_equal_weights.log" 2>&1

log "Phase 5 done."

# ============================================================================
# Phase 6: Collect & Visualize
# ============================================================================
log "===== Phase 6: Collect & Visualize ====="

python scripts/collect_and_visualize.py \
    --results_dir "$RESULTS" --output_dir to_human \
    > "$LOG/collect_visualize.log" 2>&1 || true

log "============================================="
log "ALL EXPERIMENTS COMPLETE"
log "Results: $RESULTS/"
log "============================================="
