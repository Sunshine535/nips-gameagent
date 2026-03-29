#!/usr/bin/env bash
# ============================================================================
# GameAgent: Factorial Experiment Pipeline
#
# 5-condition factorial design (addressing reviewer W1 + W4):
#   Condition 1: BASE       — eval base model only
#   Condition 2: SFT-ONLY   — SFT warmup, no RL
#   Condition 3: A-ONLY     — SFT + GRPO self-play (Track A)
#   Condition 4: B-ONLY     — SFT + formal Nash-DPO (Track B)
#   Condition 5: A+B        — SFT + GRPO → Nash-DPO (combined)
#
# Each condition runs with 3 seeds (42, 123, 456) for statistical validity.
#
# Usage:
#   bash scripts/run_all_experiments.sh                  # full run
#   bash scripts/run_all_experiments.sh --quick           # smoke test
#   bash scripts/run_all_experiments.sh --condition A     # single condition
#   bash scripts/run_all_experiments.sh --seed 42         # single seed
#   bash scripts/run_all_experiments.sh --skip_eval       # skip benchmarks
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_utils.sh"
auto_setup

PROJ_DIR="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJ_DIR/.venv/bin/activate" ]; then
    source "$PROJ_DIR/.venv/bin/activate"
fi
export PATH="$HOME/.local/bin:$PATH"
cd "$PROJ_DIR"

GAME_CONFIG="configs/game_scenarios.yaml"
ROLE_CONFIG="configs/agent_roles.yaml"
DATA_DIR="data"
RESULTS_DIR="results"
LOG_DIR="logs"

SEEDS=(42 123 456)
QUICK=false
CONDITION="all"
SINGLE_SEED=""
SKIP_EVAL=false
FORCE_RERUN="${FORCE_RERUN:-0}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)       QUICK=true; shift ;;
        --condition)   CONDITION="$2"; shift 2 ;;
        --seed)        SINGLE_SEED="$2"; shift 2 ;;
        --skip_eval)   SKIP_EVAL=true; shift ;;
        --gpus)        NUM_GPUS="$2"; shift 2 ;;
        *)             echo "WARNING: Unknown arg: $1 (ignored)"; shift ;;
    esac
done

if [ -n "$SINGLE_SEED" ]; then
    SEEDS=("$SINGLE_SEED")
fi

if [ "$QUICK" = true ]; then
    GRPO_SFT_EPISODES=50
    GRPO_ITERS=1
    GRPO_EPS_PER_ITER=200
    NASH_PREF_PAIRS=100
    EVAL_EPISODES=3
    BENCH_SAMPLES=20
    SEEDS=(42)
    echo "[QUICK MODE] Reduced params for smoke test"
else
    GRPO_SFT_EPISODES=200
    GRPO_ITERS=3
    GRPO_EPS_PER_ITER=500
    NASH_PREF_PAIRS=1000
    EVAL_EPISODES=20
    BENCH_SAMPLES=200
fi

mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$LOG_DIR"

MODE_TAG="full"
if [ "$QUICK" = true ]; then MODE_TAG="quick"; fi
PHASE_MARKER_DIR="$RESULTS_DIR/.phase_markers_${MODE_TAG}"
mkdir -p "$PHASE_MARKER_DIR"

phase_done() { touch "$PHASE_MARKER_DIR/phase_${1}.done"; log "PHASE $1 DONE"; }
is_phase_done() {
    [[ "$FORCE_RERUN" == "1" ]] && return 1
    [[ -f "$PHASE_MARKER_DIR/phase_${1}.done" ]] && log "PHASE $1 already done, skipping" && return 0
    return 1
}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
run_timed() {
    local name="$1"; shift
    log "START: $name"
    local start=$SECONDS
    "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
    local elapsed=$(( SECONDS - start ))
    log "DONE:  $name (${elapsed}s)"
}

BASE_MODEL=$(python3 -c "
import yaml
with open('$GAME_CONFIG') as f:
    print(yaml.safe_load(f)['model']['base_model'])
" 2>/dev/null || echo "Qwen/Qwen3.5-9B")

log "============================================="
log " GameAgent Factorial Pipeline"
log " Base model:  $BASE_MODEL"
log " Conditions:  $CONDITION"
log " Seeds:       ${SEEDS[*]}"
log " Quick mode:  $QUICK"
log "============================================="

# ============================================================================
# Phase 0: Data Generation (shared across conditions)
# ============================================================================

if ! is_phase_done "0_data"; then
    log "========== Phase 0: Data Generation =========="

    run_timed "P0_generate_sft_data" \
        python scripts/generate_sft_data.py \
            --config "$GAME_CONFIG" \
            --episodes_per_game "$GRPO_SFT_EPISODES" \
            --output_dir "$DATA_DIR" \
            --top_fraction 0.3 --seed 42

    run_timed "P0_generate_preference_data" \
        python scripts/generate_preference_data.py \
            --base_model "$BASE_MODEL" \
            --output_file "$DATA_DIR/preference_pairs.jsonl" \
            --n_prompts "$NASH_PREF_PAIRS" \
            --seed 42

    phase_done "0_data"
fi

# ============================================================================
# Condition 1: BASE (just evaluation)
# ============================================================================

run_condition_base() {
    local seed=$1
    local tag="base_s${seed}"
    local outdir="$RESULTS_DIR/$tag"

    if is_phase_done "$tag"; then return; fi
    log "===== Condition: BASE (seed=$seed) ====="
    mkdir -p "$outdir"

    run_timed "${tag}_eval_games" \
        python scripts/eval_game_performance.py \
            --config "$GAME_CONFIG" \
            --output_dir "$outdir/game_eval" \
            --episodes "$EVAL_EPISODES" \
            --seed "$seed"

    phase_done "$tag"
}

# ============================================================================
# Condition 2: SFT-ONLY
# ============================================================================

run_condition_sft() {
    local seed=$1
    local tag="sft_s${seed}"
    local outdir="$RESULTS_DIR/$tag"

    if is_phase_done "$tag"; then return; fi
    log "===== Condition: SFT-ONLY (seed=$seed) ====="
    mkdir -p "$outdir"

    run_timed "${tag}_sft_warmup" \
        python scripts/train_sft_warmup.py \
            --config "$GAME_CONFIG" \
            --data_dir "$DATA_DIR" \
            --output_dir "$outdir/sft_agents" \
            --num_epochs 2 \
            --lora_r 16 --lora_alpha 32 \
            --batch_size 4 --gradient_accumulation_steps 4 \
            --seed "$seed"

    # Evaluate all 4 SFT agents (not just agent_0) for fair comparison
    SFT_EVAL_MODELS=""
    for i in 0 1 2 3; do
        ap="$outdir/sft_agents/agent_${i}/final"
        if [ -d "$ap" ]; then
            [ -n "$SFT_EVAL_MODELS" ] && SFT_EVAL_MODELS="$SFT_EVAL_MODELS "
            SFT_EVAL_MODELS="${SFT_EVAL_MODELS}agent_${i}:${ap}"
        fi
    done
    if [ -n "$SFT_EVAL_MODELS" ]; then
        run_timed "${tag}_eval_games" \
            python scripts/eval_game_performance.py \
                --config "$GAME_CONFIG" \
                --model_paths $SFT_EVAL_MODELS \
                --output_dir "$outdir/game_eval" \
                --episodes_per_game "$EVAL_EPISODES" \
                --seed "$seed"
    fi

    phase_done "$tag"
}

# ============================================================================
# Condition 3: A-ONLY (GRPO self-play)
# ============================================================================

run_condition_A() {
    local seed=$1
    local tag="A_s${seed}"
    local outdir="$RESULTS_DIR/$tag"

    if is_phase_done "$tag"; then return; fi
    log "===== Condition: A-ONLY / GRPO (seed=$seed) ====="
    mkdir -p "$outdir"

    run_timed "${tag}_sft_warmup" \
        python scripts/train_sft_warmup.py \
            --config "$GAME_CONFIG" \
            --data_dir "$DATA_DIR" \
            --output_dir "$outdir/sft_agents" \
            --num_epochs 2 \
            --lora_r 16 --lora_alpha 32 \
            --batch_size 4 --gradient_accumulation_steps 4 \
            --seed "$seed"

    run_timed "${tag}_grpo_self_play" \
        python scripts/run_grpo_self_play.py \
            --config "$GAME_CONFIG" \
            --sft_dir "$outdir/sft_agents" \
            --output_dir "$outdir/grpo" \
            --num_iterations "$GRPO_ITERS" \
            --episodes_per_iter "$GRPO_EPS_PER_ITER" \
            --eval_episodes "$EVAL_EPISODES" \
            --learning_rate 5e-5 \
            --seed "$seed"

    # Evaluate all 4 GRPO agents for fair comparison
    GRPO_EVAL_MODELS=""
    for i in 0 1 2 3; do
        ap="$outdir/grpo/final/agent_${i}"
        if [ -d "$ap" ]; then
            [ -n "$GRPO_EVAL_MODELS" ] && GRPO_EVAL_MODELS="$GRPO_EVAL_MODELS "
            GRPO_EVAL_MODELS="${GRPO_EVAL_MODELS}agent_${i}:${ap}"
        fi
    done
    if [ -n "$GRPO_EVAL_MODELS" ]; then
        run_timed "${tag}_eval_games" \
            python scripts/eval_game_performance.py \
                --config "$GAME_CONFIG" \
                --model_paths $GRPO_EVAL_MODELS \
                --output_dir "$outdir/game_eval" \
                --episodes_per_game "$EVAL_EPISODES" \
                --seed "$seed"
    fi

    phase_done "$tag"
}

# ============================================================================
# Condition 4: B-ONLY (formal Nash-DPO)
# ============================================================================

run_condition_B() {
    local seed=$1
    local tag="B_s${seed}"
    local outdir="$RESULTS_DIR/$tag"

    if is_phase_done "$tag"; then return; fi
    log "===== Condition: B-ONLY / Nash-DPO (seed=$seed) ====="
    mkdir -p "$outdir"

    # SFT warmup first (same as other conditions for fair comparison)
    run_timed "${tag}_sft_warmup" \
        python scripts/train_sft_warmup.py \
            --config "$GAME_CONFIG" \
            --data_dir "$DATA_DIR" \
            --output_dir "$outdir/sft_agents" \
            --num_epochs 2 \
            --lora_r 16 --lora_alpha 32 \
            --batch_size 4 --gradient_accumulation_steps 4 \
            --seed "$seed"

    SFT_CKPT_B="$outdir/sft_agents/agent_0/final"
    B_EXTRA=""
    if [ -d "$SFT_CKPT_B" ] && [ -f "$SFT_CKPT_B/adapter_config.json" ]; then
        B_EXTRA="--model_path $SFT_CKPT_B"
        log "B: Nash-DPO will load SFT LoRA from $SFT_CKPT_B"
    fi

    run_timed "${tag}_nash_dpo" \
        python scripts/train_formal_nash_dpo.py \
            --base_model "$BASE_MODEL" \
            --preference_data "$DATA_DIR/preference_pairs.jsonl" \
            --output_dir "$outdir/nash_dpo" \
            --method nash \
            --beta 0.1 \
            --ema_tau 0.1 \
            --warmup_steps 50 \
            --epochs 1 \
            --batch_size 4 \
            --gradient_accumulation 4 \
            --lr 2e-4 \
            --seed "$seed" \
            --resume_from_checkpoint auto \
            $B_EXTRA

    run_timed "${tag}_eval_games" \
        python scripts/eval_game_performance.py \
            --config "$GAME_CONFIG" \
            --model_dir "$outdir/nash_dpo" \
            --output_dir "$outdir/game_eval" \
            --episodes "$EVAL_EPISODES" \
            --seed "$seed"

    phase_done "$tag"
}

# ============================================================================
# Condition 5: A+B (GRPO → Nash-DPO)
# ============================================================================

run_condition_AB() {
    local seed=$1
    local tag="AB_s${seed}"
    local outdir="$RESULTS_DIR/$tag"

    if is_phase_done "$tag"; then return; fi
    log "===== Condition: A+B (seed=$seed) ====="
    mkdir -p "$outdir"

    run_timed "${tag}_sft_warmup" \
        python scripts/train_sft_warmup.py \
            --config "$GAME_CONFIG" \
            --data_dir "$DATA_DIR" \
            --output_dir "$outdir/sft_agents" \
            --num_epochs 2 \
            --lora_r 16 --lora_alpha 32 \
            --batch_size 4 --gradient_accumulation_steps 4 \
            --seed "$seed"

    run_timed "${tag}_grpo" \
        python scripts/run_grpo_self_play.py \
            --config "$GAME_CONFIG" \
            --sft_dir "$outdir/sft_agents" \
            --output_dir "$outdir/grpo" \
            --num_iterations "$GRPO_ITERS" \
            --episodes_per_iter "$GRPO_EPS_PER_ITER" \
            --eval_episodes "$EVAL_EPISODES" \
            --learning_rate 5e-5 \
            --seed "$seed"

    GRPO_CKPT="$outdir/grpo/final/agent_0"
    # Nash-DPO starts from GRPO checkpoint (the A+B combination)
    NASH_BASE="$BASE_MODEL"
    NASH_EXTRA_ARGS=""
    if [ -d "$GRPO_CKPT" ] && [ -f "$GRPO_CKPT/adapter_config.json" ]; then
        NASH_EXTRA_ARGS="--model_path $GRPO_CKPT"
        log "A+B: Nash-DPO will load GRPO LoRA from $GRPO_CKPT"
    fi

    run_timed "${tag}_nash_dpo" \
        python scripts/train_formal_nash_dpo.py \
            --base_model "$NASH_BASE" \
            --preference_data "$DATA_DIR/preference_pairs.jsonl" \
            --output_dir "$outdir/nash_dpo" \
            --method nash \
            --beta 0.1 \
            --ema_tau 0.1 \
            --warmup_steps 50 \
            --epochs 1 \
            --batch_size 4 \
            --gradient_accumulation 4 \
            --lr 2e-4 \
            --seed "$seed" \
            --resume_from_checkpoint auto \
            $NASH_EXTRA_ARGS

    run_timed "${tag}_eval_games" \
        python scripts/eval_game_performance.py \
            --config "$GAME_CONFIG" \
            --model_dir "$outdir/nash_dpo" \
            --output_dir "$outdir/game_eval" \
            --episodes "$EVAL_EPISODES" \
            --seed "$seed"

    phase_done "$tag"
}

# ============================================================================
# Ablation: Equal-weight DPO and Fixed-weight DPO baselines
# ============================================================================

run_ablation_baselines() {
    local seed=$1

    for method in equal fixed single_correctness; do
        local tag="ablation_${method}_s${seed}"
        local outdir="$RESULTS_DIR/$tag"

        if is_phase_done "$tag"; then continue; fi
        log "===== Ablation: $method (seed=$seed) ====="
        mkdir -p "$outdir"

        run_timed "${tag}_train" \
            python scripts/train_formal_nash_dpo.py \
                --base_model "$BASE_MODEL" \
                --preference_data "$DATA_DIR/preference_pairs.jsonl" \
                --output_dir "$outdir/dpo" \
                --method "$method" \
                --beta 0.1 \
                --epochs 1 \
                --batch_size 4 \
                --gradient_accumulation 4 \
                --lr 2e-4 \
                --seed "$seed" \
                --resume_from_checkpoint auto

        phase_done "$tag"
    done
}

# ============================================================================
# Dispatch
# ============================================================================

for seed in "${SEEDS[@]}"; do
    case "$CONDITION" in
        all)
            run_condition_base "$seed"
            run_condition_sft "$seed"
            run_condition_A "$seed"
            run_condition_B "$seed"
            run_condition_AB "$seed"
            run_ablation_baselines "$seed"
            ;;
        base)  run_condition_base "$seed" ;;
        sft)   run_condition_sft "$seed" ;;
        A)     run_condition_A "$seed" ;;
        B)     run_condition_B "$seed" ;;
        AB)    run_condition_AB "$seed" ;;
        ablation) run_ablation_baselines "$seed" ;;
        *)     echo "Unknown condition: $CONDITION"; exit 1 ;;
    esac
done

# ============================================================================
# Unified Benchmark Evaluation (all conditions)
# ============================================================================

if [ "$SKIP_EVAL" = false ] && ! is_phase_done "benchmarks"; then
    log "========== Benchmark Evaluation =========="

    EVAL_MODELS="base_s42:$BASE_MODEL"
    for seed in "${SEEDS[@]}"; do
        # B and AB produce a single model; SFT and A use agent_0 as representative for NLP benchmarks
        for cond in sft A B AB; do
            case "$cond" in
                sft)  model_path="$RESULTS_DIR/sft_s${seed}/sft_agents/agent_0/final" ;;
                A)    model_path="$RESULTS_DIR/A_s${seed}/grpo/final/agent_0" ;;
                B)    model_path="$RESULTS_DIR/B_s${seed}/nash_dpo" ;;
                AB)   model_path="$RESULTS_DIR/AB_s${seed}/nash_dpo" ;;
            esac
            if [ -d "$model_path" ]; then
                EVAL_MODELS="$EVAL_MODELS ${cond}_s${seed}:${model_path}"
            fi
        done
    done

    if [ -n "$EVAL_MODELS" ]; then
        run_timed "benchmarks" \
            python scripts/eval_benchmarks.py \
                --game_config "$GAME_CONFIG" \
                --model_dirs $EVAL_MODELS \
                --output_dir "$RESULTS_DIR/benchmarks" \
                --benchmarks strategyqa truthfulqa mt_bench \
                --max_samples "$BENCH_SAMPLES"
    fi

    phase_done "benchmarks"
fi

# ============================================================================
# Summary
# ============================================================================

log "============================================="
log " All experiments complete."
log " Results: $RESULTS_DIR/"
log ""
log " Conditions run:"
for seed in "${SEEDS[@]}"; do
    for cond in base sft A B AB; do
        tag="${cond}_s${seed}"
        if [ -f "$PHASE_MARKER_DIR/phase_${tag}.done" ]; then
            log "   ✓ $tag"
        else
            log "   ✗ $tag (not completed)"
        fi
    done
done
log "============================================="
