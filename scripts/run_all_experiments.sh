#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_utils.sh"
auto_setup

PROJ_DIR_ROOT="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJ_DIR_ROOT/.venv/bin/activate" ]; then
    source "$PROJ_DIR_ROOT/.venv/bin/activate"
fi
export PATH="$HOME/.local/bin:$PATH"

PROJECT_DIR="$PROJ_DIR_ROOT"
cd "$PROJECT_DIR"

# --- Phase resume logic ---
PHASE_MARKER_DIR="$PROJECT_DIR/results/.phase_markers"
mkdir -p "$PHASE_MARKER_DIR"
FORCE_RERUN="${FORCE_RERUN:-0}"

phase_done() { local p="$1"; touch "$PHASE_MARKER_DIR/phase_${p}.done"; echo "[PHASE $p] Completed at $(date)"; }
is_phase_done() {
    local p="$1"
    [[ "$FORCE_RERUN" == "1" ]] && return 1
    [[ -f "$PHASE_MARKER_DIR/phase_${p}.done" ]] && echo "[PHASE $p] Already completed. Skipping." && return 0
    return 1
}

GAME_CONFIG="configs/game_scenarios.yaml"
ROLE_CONFIG="configs/agent_roles.yaml"
DATA_DIR="data"
RESULTS_DIR="results"
LOG_DIR="logs"
SEED=42
SKIP_GRPO=false
SKIP_NASH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip_grpo)   SKIP_GRPO=true; shift ;;
        --skip_nash)   SKIP_NASH=true; shift ;;
        --gpus)        NUM_GPUS="$2"; shift 2 ;;
        --seed)        SEED="$2"; shift 2 ;;
        --force)       FORCE_RERUN=1; shift ;;
        *)             echo "WARNING: Unknown arg: $1 (ignored)"; shift ;;
    esac
done

GRPO_EPISODES_PER_GAME=5000
GRPO_SFT_EPISODES=5000
GRPO_ITERS=5
GRPO_EPS_PER_ITER=10000
NASH_EXPERT_EPISODES=5000
NASH_ITERS=3
NASH_GAMES_PER_ITER=500
EVAL_EPISODES=20
EVAL_SAMPLES=500

mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
run_timed() {
    local name="$1"; shift
    log "START: $name"
    local start=$SECONDS
    "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
    local elapsed=$(( SECONDS - start ))
    log "DONE:  $name (${elapsed}s)"
}

echo "============================================"
echo " GameAgent — Full Experiment Pipeline"
echo " GPUs: $NUM_GPUS × $GPU_CLASS"
echo "============================================"

# ===== Track A: GRPO Self-Play =====
if [ "$SKIP_GRPO" = false ]; then
    log "========== Track A: GRPO Self-Play =========="

    if ! is_phase_done A1; then
        run_timed "A1_generate_sft_data" python scripts/generate_sft_data.py \
            --config "$GAME_CONFIG" --episodes_per_game "$GRPO_SFT_EPISODES" \
            --top_fraction 0.3 --output_dir "$DATA_DIR" --seed "$SEED"
        phase_done A1
    fi

    if ! is_phase_done A2; then
        run_timed "A2_train_sft_warmup" python scripts/train_sft_warmup.py \
            --config "$GAME_CONFIG" --data_dir "$DATA_DIR" \
            --output_dir "$RESULTS_DIR/sft_agents" \
            --num_epochs 2 --lora_r 16 --lora_alpha 32 --batch_size 4 --gradient_accumulation_steps 4
        phase_done A2
    fi

    if ! is_phase_done A3; then
        run_timed "A3_grpo_self_play" python scripts/run_grpo_self_play.py \
            --config "$GAME_CONFIG" --sft_dir "$RESULTS_DIR/sft_agents" \
            --output_dir "$RESULTS_DIR/grpo_self_play" \
            --num_iterations "$GRPO_ITERS" --episodes_per_iter "$GRPO_EPS_PER_ITER" \
            --eval_episodes "$EVAL_EPISODES" --learning_rate 5e-5 --seed "$SEED"
        phase_done A3
    fi

    if ! is_phase_done A4; then
        run_timed "A4_cross_game_transfer" python scripts/run_cross_game_transfer.py \
            --config "$GAME_CONFIG" --data_dir "$DATA_DIR" \
            --output_dir "$RESULTS_DIR/cross_game_transfer" \
            --num_epochs 2 --eval_episodes "$EVAL_EPISODES" --seed "$SEED"
        phase_done A4
    fi
fi

# ===== Track B: Nash-DPO Self-Play =====
if [ "$SKIP_NASH" = false ]; then
    log "========== Track B: Nash-DPO Self-Play =========="

    if ! is_phase_done B1; then
        run_timed "B1_generate_expert_data" python scripts/generate_expert_data.py \
            --config "$ROLE_CONFIG" --output_dir "$RESULTS_DIR/expert_data" \
            --episodes_per_game "$NASH_EXPERT_EPISODES" --top_fraction 0.3 --seed "$SEED"
        phase_done B1
    fi

    if ! is_phase_done B2; then
        run_timed "B2_train_sft_role_agents" python scripts/train_sft_agents.py \
            --config "$ROLE_CONFIG" --expert_data "$RESULTS_DIR/expert_data/expert_train.jsonl" \
            --output_dir "$RESULTS_DIR/sft_role_agents" --num_epochs 2 --seed "$SEED"
        phase_done B2
    fi

    if ! is_phase_done B3; then
        run_timed "B3_nash_dpo_self_play" python scripts/train_nash_dpo.py \
            --config "$ROLE_CONFIG" --agents_dir "$RESULTS_DIR/sft_role_agents" \
            --output_dir "$RESULTS_DIR/nash_dpo" \
            --num_iterations "$NASH_ITERS" --games_per_iter "$NASH_GAMES_PER_ITER" \
            --dpo_epochs 1 --beta 0.1 --seed "$SEED"
        phase_done B3
    fi
fi

# ===== Cross-Comparison & Evaluation =====
log "========== Cross-Comparison & Evaluation =========="

if [ "$SKIP_GRPO" = false ] && [ "$SKIP_NASH" = false ]; then
    if ! is_phase_done C1; then
        run_timed "C1_grpo_vs_nash_comparison" python scripts/run_grpo_vs_nash_comparison.py \
            --game_config "$GAME_CONFIG" \
            --grpo_model "$RESULTS_DIR/grpo_self_play/final/agent_0" \
            --nash_model "$RESULTS_DIR/nash_dpo/iter$((NASH_ITERS-1))/accuracy" \
            --output_dir "$RESULTS_DIR/grpo_vs_nash" \
            --game_episodes "$EVAL_EPISODES" --seed "$SEED"
        phase_done C1
    fi
fi

EVAL_MODELS=""
if [ "$SKIP_GRPO" = false ]; then
    EVAL_MODELS="grpo:$RESULTS_DIR/grpo_self_play/final/agent_0"
fi
if [ "$SKIP_NASH" = false ]; then
    [ -n "$EVAL_MODELS" ] && EVAL_MODELS="$EVAL_MODELS "
    EVAL_MODELS="${EVAL_MODELS}nash_dpo:$RESULTS_DIR/nash_dpo/iter$((NASH_ITERS-1))/accuracy"
fi

if [ -n "$EVAL_MODELS" ]; then
    if ! is_phase_done C2; then
        run_timed "C2_unified_benchmarks" python scripts/eval_benchmarks.py \
            --game_config "$GAME_CONFIG" --model_dirs $EVAL_MODELS \
            --output_dir "$RESULTS_DIR/eval_benchmarks" \
            --benchmarks arc strategyqa bbh gsm8k truthfulqa mt_bench
        phase_done C2
    fi
fi

# ===== Ablation Studies =====
log "========== Ablation Studies =========="

if [ "$SKIP_GRPO" = false ]; then
    if ! is_phase_done D1; then
        run_timed "D1_ablation_no_sft" python scripts/run_grpo_self_play.py \
            --config "$GAME_CONFIG" --sft_dir "__nonexistent__" \
            --output_dir "$RESULTS_DIR/ablation_no_sft" \
            --num_iterations "$GRPO_ITERS" --episodes_per_iter $((GRPO_EPS_PER_ITER / 2)) \
            --eval_episodes "$EVAL_EPISODES" --seed "$SEED"
        phase_done D1
    fi

    if ! is_phase_done D2; then
        run_timed "D2_ablation_fewer_grpo_iters" python scripts/run_grpo_self_play.py \
            --config "$GAME_CONFIG" --sft_dir "$RESULTS_DIR/sft_agents" \
            --output_dir "$RESULTS_DIR/ablation_2iter_grpo" \
            --num_iterations 2 --episodes_per_iter "$GRPO_EPS_PER_ITER" \
            --eval_episodes "$EVAL_EPISODES" --seed "$SEED"
        phase_done D2
    fi
fi

if [ "$SKIP_NASH" = false ]; then
    if ! is_phase_done D3; then
        run_timed "D3_ablation_fewer_nash_iters" python scripts/train_nash_dpo.py \
            --config "$ROLE_CONFIG" --agents_dir "$RESULTS_DIR/sft_role_agents" \
            --output_dir "$RESULTS_DIR/ablation_1iter_nash" \
            --num_iterations 1 --games_per_iter "$NASH_GAMES_PER_ITER" --seed "$SEED"
        phase_done D3
    fi
fi

log "============================================="
log "All experiments complete."
log "Results: $RESULTS_DIR"
log "============================================="

DONE_FILE="$PROJECT_DIR/results/.pipeline_done"
cat > "$DONE_FILE" << DONEEOF
{
  "project": "nips-gameagent",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS:-unknown}",
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo "[PIPELINE_COMPLETE] Run 'bash collect_results.sh' to package results."
