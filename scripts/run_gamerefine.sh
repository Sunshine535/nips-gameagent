#!/bin/bash
# =========================================================================
# DEPRECATED — Replaced by scripts/run_all_experiments.sh
#
# This was the original GameRefine pipeline. It has been superseded by the
# unified GameAgent pipeline which covers both Track A (GRPO) and
# Track B (Nash-DPO).
#
# Usage of the new pipeline:
#   bash scripts/run_all_experiments.sh              # full run
#   bash scripts/run_all_experiments.sh --skip_grpo  # Nash-DPO only (≈old GameRefine)
#   bash scripts/run_all_experiments.sh --quick       # smoke test
# =========================================================================
#
# Original purpose:
# GameRefine: Asymmetric multi-agent self-play
# Stage 1: Train 4 specialized agents (accuracy/safety/efficiency/creativity)
# Stage 2: Multi-round self-play with Nash-DPO
# Stage 3: Evaluate all agents
# 8x A100-80GB

set -euo pipefail

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=0
export NCCL_NET_MERGE_LEVEL=LOC
export NCCL_IB_DISABLE=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_DIR}/configs/agent_roles.yaml"

AGENTS_DIR="${PROJECT_DIR}/checkpoints/agents"
SELF_PLAY_DIR="${PROJECT_DIR}/checkpoints/self_play"
RESULTS_DIR="${PROJECT_DIR}/results"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "$AGENTS_DIR" "$SELF_PLAY_DIR" "$RESULTS_DIR" "$LOG_DIR"

NUM_GPUS=1
AGENTS=(accuracy safety efficiency creativity)

# =========================================
# Stage 1: Train individual agents
# =========================================
echo "========================================="
echo "[Stage 1] Training specialized agents..."
echo "========================================="

for agent in "${AGENTS[@]}"; do
    if [ -f "${AGENTS_DIR}/${agent}/agent_metrics.json" ]; then
        echo "Agent $agent already trained, skipping."
        continue
    fi

    echo "Training agent: $agent"
    accelerate launch \
        --num_processes $NUM_GPUS \
        --mixed_precision bf16 \
        "${SCRIPT_DIR}/train_agents.py" \
        --config "$CONFIG" \
        --agent "$agent" \
        --output_dir "$AGENTS_DIR" \
        --seed 42 \
        2>&1 | tee "${LOG_DIR}/train_${agent}.log"

    echo "Agent $agent training complete."
done

echo "[Stage 1] All agents trained."

# =========================================
# Stage 2: Multi-round self-play
# =========================================
echo "========================================="
echo "[Stage 2] Running self-play..."
echo "========================================="

if [ -f "${SELF_PLAY_DIR}/self_play_summary.json" ]; then
    echo "Self-play already complete, skipping."
else
    accelerate launch \
        --num_processes $NUM_GPUS \
        --mixed_precision bf16 \
        "${SCRIPT_DIR}/run_self_play.py" \
        --config "$CONFIG" \
        --agents_dir "$AGENTS_DIR" \
        --output_dir "$SELF_PLAY_DIR" \
        --num_rounds 5 \
        --games_per_round 500 \
        2>&1 | tee "${LOG_DIR}/self_play.log"
fi

echo "[Stage 2] Self-play complete."

# =========================================
# Stage 3: Evaluate
# =========================================
echo "========================================="
echo "[Stage 3] Evaluating all agents..."
echo "========================================="

# Evaluate pre-self-play agents
echo "Evaluating pre-self-play agents..."
python "${SCRIPT_DIR}/eval_gamerefine.py" \
    --config "$CONFIG" \
    --agents_dir "$AGENTS_DIR" \
    --output_dir "${RESULTS_DIR}/pre_selfplay" \
    --eval_all \
    2>&1 | tee "${LOG_DIR}/eval_pre_selfplay.log"

# Evaluate post-self-play agents (from latest round)
if [ -f "${SELF_PLAY_DIR}/self_play_summary.json" ]; then
    LAST_ROUND=$(python -c "
import json
with open('${SELF_PLAY_DIR}/self_play_summary.json') as f:
    d = json.load(f)
print(d['num_rounds'] - 1)
")
    LATEST_AGENTS="${SELF_PLAY_DIR}/round${LAST_ROUND}"

    if [ -d "$LATEST_AGENTS" ]; then
        echo "Evaluating post-self-play agents (round $LAST_ROUND)..."
        python "${SCRIPT_DIR}/eval_gamerefine.py" \
            --config "$CONFIG" \
            --agents_dir "$LATEST_AGENTS" \
            --output_dir "${RESULTS_DIR}/post_selfplay" \
            --eval_all \
            2>&1 | tee "${LOG_DIR}/eval_post_selfplay.log"
    fi
fi

echo "========================================="
echo "GameRefine pipeline complete!"
echo "Agents:    $AGENTS_DIR"
echo "Self-play: $SELF_PLAY_DIR"
echo "Results:   $RESULTS_DIR"
echo "========================================="
