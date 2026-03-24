#!/usr/bin/env bash
# Resume pipeline from A2 (SFT Warmup) after fixing max_seq_length -> max_length
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_utils.sh"
auto_setup

PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

GAME_CONFIG="configs/game_scenarios.yaml"
ROLE_CONFIG="configs/agent_roles.yaml"
DATA_DIR="data"
RESULTS_DIR="results"
LOG_DIR="logs"
SEED=42
GRPO_ITERS=3
GRPO_EPS_PER_ITER=2000
NASH_EXPERT_EPISODES=500
NASH_ITERS=2
NASH_GAMES_PER_ITER=200
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

log "========== Resuming from A2: SFT Warmup =========="

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

log "========== Track B: Nash-DPO Self-Play =========="

parallel_generate() {
    local script="$1" config="$2" episodes="$3" outdir="$4"
    shift 4
    local extra_args=("$@")
    local G0="prisoners_dilemma,battle_of_sexes"
    local G1="stag_hunt,chicken"
    local G2="matching_pennies,public_goods"
    local G3="ultimatum,coordination"
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
}

run_timed "B1_generate_expert_data" \
    parallel_generate scripts/generate_expert_data.py "$ROLE_CONFIG" \
        "$NASH_EXPERT_EPISODES" "$RESULTS_DIR/expert_data" \
        --top_fraction 0.3 --seed "$SEED"

log "Merging per-game expert data into train/val splits"
python3 << 'PYEOF'
import json, os, random
random.seed(42)
d = "results/expert_data"
all_ep = []
for f in sorted(os.listdir(d)):
    if f.startswith("expert_") and f.endswith(".jsonl") and f not in ("expert_train.jsonl", "expert_val.jsonl"):
        with open(os.path.join(d, f)) as fh:
            all_ep.extend(json.loads(l) for l in fh)
random.shuffle(all_ep)
val_n = int(len(all_ep) * 0.1)
for name, data in [("train", all_ep[val_n:]), ("val", all_ep[:val_n])]:
    with open(os.path.join(d, "expert_%s.jsonl" % name), "w") as fh:
        for r in data:
            fh.write(json.dumps(r) + "\n")
print("Merged: %d total -> train=%d val=%d" % (len(all_ep), len(all_ep) - val_n, val_n))
PYEOF

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

log "========== Cross-Comparison & Evaluation =========="

run_timed "C1_grpo_vs_nash_comparison" \
    python scripts/run_grpo_vs_nash_comparison.py \
        --game_config "$GAME_CONFIG" \
        --grpo_model "$RESULTS_DIR/grpo_self_play/final/agent_0" \
        --nash_model "$RESULTS_DIR/nash_dpo/iter$((NASH_ITERS-1))/accuracy" \
        --output_dir "$RESULTS_DIR/grpo_vs_nash" \
        --game_episodes "$EVAL_EPISODES" \
        --seed "$SEED"

EVAL_MODELS="grpo:$RESULTS_DIR/grpo_self_play/final/agent_0 nash_dpo:$RESULTS_DIR/nash_dpo/iter$((NASH_ITERS-1))/accuracy"

run_timed "C2_unified_benchmarks" \
    python scripts/eval_benchmarks.py \
        --game_config "$GAME_CONFIG" \
        --model_dirs $EVAL_MODELS \
        --output_dir "$RESULTS_DIR/eval_benchmarks" \
        --benchmarks arc strategyqa bbh gsm8k truthfulqa mt_bench

log "========== Ablation Studies =========="

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

run_timed "D3_ablation_fewer_nash_iters" \
    python scripts/train_nash_dpo.py \
        --config "$ROLE_CONFIG" \
        --agents_dir "$RESULTS_DIR/sft_role_agents" \
        --output_dir "$RESULTS_DIR/ablation_1iter_nash" \
        --num_iterations 1 \
        --games_per_iter "$NASH_GAMES_PER_ITER" \
        --seed "$SEED"

log "========== Pipeline Complete =========="
touch results/.pipeline_done
log "All experiments done. Results in $RESULTS_DIR"
