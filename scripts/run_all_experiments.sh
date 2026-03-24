#!/usr/bin/env bash
# ============================================================================
# GameAgent: Unified Master Experiment Pipeline
#
# Combines two training paradigms:
#   Track A (GRPO): Strategic decision-making via multi-agent game self-play
#   Track B (Nash-DPO): Multi-objective alignment via asymmetric agent roles
#
# Pipeline (default/full mode — --quick uses smaller counts):
#   1. Generate SFT data (GRPO track: 10 games × 500 episodes)
#   2. Train 4 SFT warmup agents (GRPO track, LoRA, 2 games each)
#   3. GRPO self-play (3 iterations, 2K episodes/iter)
#   4. Cross-game transfer evaluation
#   5. Generate expert data (Nash-DPO track: 8 simple games × 500 episodes)
#   6. Train 4 role-specialized SFT agents (Nash-DPO track)
#   7. Iterative Nash-DPO self-play (2 iterations)
#   8. GRPO vs Nash-DPO cross-comparison
#   9. Unified benchmark evaluation (ARC + StrategyQA + BBH + GSM8K + TruthfulQA + MT-Bench)
#   10. Ablation studies
#
# Usage:
#   bash scripts/run_all_experiments.sh
#   bash scripts/run_all_experiments.sh --skip_grpo     # only Nash-DPO track
#   bash scripts/run_all_experiments.sh --skip_nash     # only GRPO track
#   bash scripts/run_all_experiments.sh --quick         # smoke test
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_utils.sh"
auto_setup

PROJ_DIR_ROOT="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJ_DIR_ROOT/.venv/bin/activate" ]; then
    source "$PROJ_DIR_ROOT/.venv/bin/activate"
fi
export PATH="$HOME/.local/bin:$PATH"

PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

GAME_CONFIG="configs/game_scenarios.yaml"
ROLE_CONFIG="configs/agent_roles.yaml"
DATA_DIR="data"
RESULTS_DIR="results"
LOG_DIR="logs"
SEED=42
SKIP_GRPO=false
SKIP_NASH=false
QUICK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip_grpo)   SKIP_GRPO=true; shift ;;
        --skip_nash)   SKIP_NASH=true; shift ;;
        --quick)       QUICK=true; shift ;;
        --gpus)        NUM_GPUS="$2"; shift 2 ;;
        --seed)        SEED="$2"; shift 2 ;;
        *)             echo "WARNING: Unknown arg: $1 (ignored)"; shift ;;
    esac
done

if [ "$QUICK" = true ]; then
    GRPO_SFT_EPISODES=100
    GRPO_ITERS=1
    GRPO_EPS_PER_ITER=500
    NASH_EXPERT_EPISODES=100
    NASH_ITERS=1
    NASH_GAMES_PER_ITER=50
    EVAL_EPISODES=5
    BENCH_SAMPLES=50
else
    GRPO_SFT_EPISODES=500
    GRPO_ITERS=3
    GRPO_EPS_PER_ITER=2000
    NASH_EXPERT_EPISODES=500
    NASH_ITERS=2
    NASH_GAMES_PER_ITER=200
    EVAL_EPISODES=20
    BENCH_SAMPLES=500
fi

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

# ============================================================================
# Track A: GRPO Self-Play (Strategic Decision-Making)
# ============================================================================

parallel_generate() {
    local script="$1" config="$2" episodes="$3" outdir="$4"
    shift 4
    local extra_args=("$@")

    if [ "${NUM_GPUS:-4}" -ge 4 ]; then
        log "Parallel data generation on ${NUM_GPUS:-4} GPUs"
        if [ "$script" = "scripts/generate_sft_data.py" ]; then
            local G0="prisoners_dilemma,coordination_game,battle_of_sexes"
            local G1="stag_hunt,chicken,matching_pennies"
            local G2="public_goods,ultimatum"
            local G3="auction,negotiation"
        else
            local G0="prisoners_dilemma,battle_of_sexes"
            local G1="stag_hunt,chicken"
            local G2="matching_pennies,public_goods"
            local G3="ultimatum,coordination"
        fi
        CUDA_VISIBLE_DEVICES=0 python "$script" --config "$config" \
            --episodes_per_game "$episodes" --output_dir "$outdir" \
            --games "$G0" "${extra_args[@]}" 2>&1 | tee "${LOG_DIR}/${script##*/}_gpu0.log" &
        CUDA_VISIBLE_DEVICES=1 python "$script" --config "$config" \
            --episodes_per_game "$episodes" --output_dir "$outdir" \
            --games "$G1" "${extra_args[@]}" 2>&1 | tee "${LOG_DIR}/${script##*/}_gpu1.log" &
        CUDA_VISIBLE_DEVICES=2 python "$script" --config "$config" \
            --episodes_per_game "$episodes" --output_dir "$outdir" \
            --games "$G2" "${extra_args[@]}" 2>&1 | tee "${LOG_DIR}/${script##*/}_gpu2.log" &
        CUDA_VISIBLE_DEVICES=3 python "$script" --config "$config" \
            --episodes_per_game "$episodes" --output_dir "$outdir" \
            --games "$G3" "${extra_args[@]}" 2>&1 | tee "${LOG_DIR}/${script##*/}_gpu3.log" &
        wait
    else
        python "$script" --config "$config" \
            --episodes_per_game "$episodes" --output_dir "$outdir" "${extra_args[@]}"
    fi
}

if [ "$SKIP_GRPO" = false ]; then
    log "========== Track A: GRPO Self-Play =========="

    run_timed "A1_generate_sft_data" \
        parallel_generate scripts/generate_sft_data.py "$GAME_CONFIG" \
            "$GRPO_SFT_EPISODES" "$DATA_DIR" \
            --top_fraction 0.3 --seed "$SEED"

    run_timed "A2_train_sft_warmup" \
        python scripts/train_sft_warmup.py \
            --config "$GAME_CONFIG" \
            --data_dir "$DATA_DIR" \
            --output_dir "$RESULTS_DIR/sft_agents" \
            --num_epochs 2 \
            --lora_r 16 --lora_alpha 32 \
            --batch_size 4 --gradient_accumulation_steps 4 \
            --parallel

    run_timed "A3_grpo_self_play" \
        python scripts/run_grpo_self_play.py \
            --config "$GAME_CONFIG" \
            --sft_dir "$RESULTS_DIR/sft_agents" \
            --output_dir "$RESULTS_DIR/grpo_self_play" \
            --num_iterations "$GRPO_ITERS" \
            --episodes_per_iter "$GRPO_EPS_PER_ITER" \
            --eval_episodes "$EVAL_EPISODES" \
            --learning_rate 5e-5 \
            --seed "$SEED"

    run_timed "A4_cross_game_transfer" \
        python scripts/run_cross_game_transfer.py \
            --config "$GAME_CONFIG" \
            --data_dir "$DATA_DIR" \
            --output_dir "$RESULTS_DIR/cross_game_transfer" \
            --num_epochs 2 \
            --eval_episodes "$EVAL_EPISODES" \
            --seed "$SEED"
fi

# ============================================================================
# Track B: Nash-DPO Self-Play (Multi-Objective Alignment)
# ============================================================================

if [ "$SKIP_NASH" = false ]; then
    log "========== Track B: Nash-DPO Self-Play =========="

    run_timed "B1_generate_expert_data" \
        parallel_generate scripts/generate_expert_data.py "$ROLE_CONFIG" \
            "$NASH_EXPERT_EPISODES" "$RESULTS_DIR/expert_data" \
            --top_fraction 0.3 --seed "$SEED"

    if [ "${NUM_GPUS:-4}" -ge 4 ]; then
        log "Merging per-game expert data into train/val splits"
        python3 -c "
import json, os, random, sys
random.seed($SEED)
d = '$RESULTS_DIR/expert_data'
all_ep = []
for f in sorted(os.listdir(d)):
    if f.startswith('expert_') and f.endswith('.jsonl') and f not in ('expert_train.jsonl','expert_val.jsonl'):
        with open(os.path.join(d, f)) as fh:
            all_ep.extend(json.loads(l) for l in fh)
random.shuffle(all_ep)
val_n = int(len(all_ep) * 0.1)
for name, data in [('train', all_ep[val_n:]), ('val', all_ep[:val_n])]:
    with open(os.path.join(d, f'expert_{name}.jsonl'), 'w') as fh:
        for r in data: fh.write(json.dumps(r) + '\n')
print(f'Merged: {len(all_ep)} total -> train={len(all_ep)-val_n} val={val_n}')
"
    fi

    run_timed "B2_train_sft_role_agents" \
        python scripts/train_sft_agents.py \
            --config "$ROLE_CONFIG" \
            --expert_data "$RESULTS_DIR/expert_data/expert_train.jsonl" \
            --output_dir "$RESULTS_DIR/sft_role_agents" \
            --num_epochs 2 \
            --seed "$SEED"

    run_timed "B3_nash_dpo_self_play" \
        python scripts/train_nash_dpo.py \
            --config "$ROLE_CONFIG" \
            --agents_dir "$RESULTS_DIR/sft_role_agents" \
            --output_dir "$RESULTS_DIR/nash_dpo" \
            --num_iterations "$NASH_ITERS" \
            --games_per_iter "$NASH_GAMES_PER_ITER" \
            --dpo_epochs 1 \
            --beta 0.1 \
            --seed "$SEED"
fi

# ============================================================================
# Cross-Comparison & Evaluation
# ============================================================================

log "========== Cross-Comparison & Evaluation =========="

if [ "$SKIP_GRPO" = false ] && [ "$SKIP_NASH" = false ]; then
    run_timed "C1_grpo_vs_nash_comparison" \
        python scripts/run_grpo_vs_nash_comparison.py \
            --game_config "$GAME_CONFIG" \
            --grpo_model "$RESULTS_DIR/grpo_self_play/final/agent_0" \
            --nash_model "$RESULTS_DIR/nash_dpo/iter$((NASH_ITERS-1))/accuracy" \
            --output_dir "$RESULTS_DIR/grpo_vs_nash" \
            --game_episodes "$EVAL_EPISODES" \
            --seed "$SEED"
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
    run_timed "C2_unified_benchmarks" \
        python scripts/eval_benchmarks.py \
            --game_config "$GAME_CONFIG" \
            --model_dirs $EVAL_MODELS \
            --output_dir "$RESULTS_DIR/eval_benchmarks" \
            --benchmarks arc strategyqa bbh gsm8k truthfulqa mt_bench
fi

# ============================================================================
# Ablation Studies
# ============================================================================

log "========== Ablation Studies =========="

if [ "$SKIP_GRPO" = false ]; then
    run_timed "D1_ablation_no_sft" \
        python scripts/run_grpo_self_play.py \
            --config "$GAME_CONFIG" \
            --sft_dir "__nonexistent__" \
            --output_dir "$RESULTS_DIR/ablation_no_sft" \
            --num_iterations "$GRPO_ITERS" \
            --episodes_per_iter $((GRPO_EPS_PER_ITER / 2)) \
            --eval_episodes "$EVAL_EPISODES" \
            --seed "$SEED"

    run_timed "D2_ablation_fewer_grpo_iters" \
        python scripts/run_grpo_self_play.py \
            --config "$GAME_CONFIG" \
            --sft_dir "$RESULTS_DIR/sft_agents" \
            --output_dir "$RESULTS_DIR/ablation_2iter_grpo" \
            --num_iterations 2 \
            --episodes_per_iter "$GRPO_EPS_PER_ITER" \
            --eval_episodes "$EVAL_EPISODES" \
            --seed "$SEED"
fi

if [ "$SKIP_NASH" = false ]; then
    run_timed "D3_ablation_fewer_nash_iters" \
        python scripts/train_nash_dpo.py \
            --config "$ROLE_CONFIG" \
            --agents_dir "$RESULTS_DIR/sft_role_agents" \
            --output_dir "$RESULTS_DIR/ablation_1iter_nash" \
            --num_iterations 1 \
            --games_per_iter "$NASH_GAMES_PER_ITER" \
            --seed "$SEED"
fi

# ============================================================================
# Summary
# ============================================================================

log "============================================="
log "All experiments complete."
log "Results directory: $RESULTS_DIR"
log "Logs directory:    $LOG_DIR"
log ""
log "Track A (GRPO) outputs:"
log "  SFT data:       $DATA_DIR/sft_*.jsonl"
log "  SFT agents:     $RESULTS_DIR/sft_agents/"
log "  GRPO models:    $RESULTS_DIR/grpo_self_play/final/"
log "  Transfer eval:  $RESULTS_DIR/cross_game_transfer/"
log ""
log "Track B (Nash-DPO) outputs:"
log "  Expert data:    $RESULTS_DIR/expert_data/"
log "  Role agents:    $RESULTS_DIR/sft_role_agents/"
log "  Nash-DPO:       $RESULTS_DIR/nash_dpo/"
log ""
log "Comparison:"
log "  GRPO vs Nash:   $RESULTS_DIR/grpo_vs_nash/"
log "  Benchmarks:     $RESULTS_DIR/eval_benchmarks/"
log "  Ablations:      $RESULTS_DIR/ablation_*/"
log "============================================="
