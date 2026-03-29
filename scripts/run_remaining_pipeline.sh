#!/usr/bin/env bash
# ============================================================================
# Run ALL remaining experiments — maximizing 4× H800 80GB utilization
#
# Already completed: A1 (SFT data), A2 (SFT warmup), A3 (GRPO self-play)
# Remaining: B1-B3 (Nash-DPO), A4 (cross-game), C1-C2 (comparison+bench), D (ablation)
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
# Phase 1: Expert Data Generation — 4 GPUs in parallel (B1)
# ============================================================================
log "===== Phase 1: Expert Data Generation (4 GPUs parallel) ====="

G0="prisoners_dilemma,battle_of_sexes"
G1="stag_hunt,chicken"
G2="matching_pennies,public_goods"
G3="ultimatum,coordination"

CUDA_VISIBLE_DEVICES=0 python scripts/generate_expert_data.py \
    --config "$ROLE_CFG" --output_dir "$RESULTS/expert_data" \
    --episodes_per_game 500 --top_fraction 0.3 --seed $SEED \
    --games "$G0" > "$LOG/B1_expert_gpu0.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 python scripts/generate_expert_data.py \
    --config "$ROLE_CFG" --output_dir "$RESULTS/expert_data" \
    --episodes_per_game 500 --top_fraction 0.3 --seed $SEED \
    --games "$G1" > "$LOG/B1_expert_gpu1.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 python scripts/generate_expert_data.py \
    --config "$ROLE_CFG" --output_dir "$RESULTS/expert_data" \
    --episodes_per_game 500 --top_fraction 0.3 --seed $SEED \
    --games "$G2" > "$LOG/B1_expert_gpu2.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 python scripts/generate_expert_data.py \
    --config "$ROLE_CFG" --output_dir "$RESULTS/expert_data" \
    --episodes_per_game 500 --top_fraction 0.3 --seed $SEED \
    --games "$G3" > "$LOG/B1_expert_gpu3.log" 2>&1 &

wait
log "Phase 1 done. Merging expert data..."

python3 -c "
import json, os, random
random.seed($SEED)
d = '$RESULTS/expert_data'
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

# ============================================================================
# Phase 2: Role SFT Agents — 4 GPUs, one agent each (B2)
# ============================================================================
log "===== Phase 2: Role SFT Training (4 agents × 4 GPUs) ====="

for agent in accuracy safety efficiency creativity; do
    GPU_IDX=$(($(echo "accuracy safety efficiency creativity" | tr ' ' '\n' | grep -n "^${agent}$" | cut -d: -f1) - 1))
    CUDA_VISIBLE_DEVICES=$GPU_IDX python scripts/train_sft_agents.py \
        --config "$ROLE_CFG" \
        --expert_data "$RESULTS/expert_data/expert_train.jsonl" \
        --output_dir "$RESULTS/sft_role_agents" \
        --agent "$agent" \
        --num_epochs 2 --per_device_batch_size 4 --gradient_accumulation_steps 4 \
        --max_seq_length 1024 --seed $SEED \
        > "$LOG/B2_sft_${agent}.log" 2>&1 &
done
wait
log "Phase 2 done."

# ============================================================================
# Phase 3: Nash-DPO Self-Play (B3) — needs all agents, uses device_map=auto
# ============================================================================
log "===== Phase 3: Nash-DPO Self-Play (2 iterations) ====="

python scripts/train_nash_dpo.py \
    --config "$ROLE_CFG" \
    --agents_dir "$RESULTS/sft_role_agents" \
    --output_dir "$RESULTS/nash_dpo" \
    --num_iterations 2 \
    --games_per_iter 200 \
    --dpo_epochs 1 --dpo_batch_size 4 \
    --beta 0.1 --seed $SEED \
    > "$LOG/B3_nash_dpo.log" 2>&1

log "Phase 3 done."

# ============================================================================
# Phase 4: All evaluations in parallel — 4 GPUs
# ============================================================================
log "===== Phase 4: Evaluations (4 GPUs parallel) ====="

NASH_ITER=$((2 - 1))

# GPU 0: GRPO vs Nash-DPO Comparison
CUDA_VISIBLE_DEVICES=0 python scripts/run_grpo_vs_nash_comparison.py \
    --game_config "$GAME_CFG" \
    --grpo_model "$RESULTS/grpo_self_play/final/agent_0" \
    --nash_model "$RESULTS/nash_dpo/iter${NASH_ITER}/accuracy" \
    --output_dir "$RESULTS/grpo_vs_nash" \
    --game_episodes 20 --seed $SEED \
    > "$LOG/C1_comparison.log" 2>&1 &

# GPU 1: Benchmark evaluation (GRPO + Nash-DPO)
CUDA_VISIBLE_DEVICES=1 python scripts/eval_benchmarks.py \
    --game_config "$GAME_CFG" \
    --model_dirs \
        "grpo:$RESULTS/grpo_self_play/final/agent_0" \
        "nash_dpo:$RESULTS/nash_dpo/iter${NASH_ITER}/accuracy" \
    --output_dir "$RESULTS/eval_benchmarks" \
    --benchmarks arc strategyqa bbh gsm8k truthfulqa mt_bench \
    > "$LOG/C2_benchmarks.log" 2>&1 &

# GPU 2: Cross-game transfer (multi-game + curriculum)
CUDA_VISIBLE_DEVICES=2 python scripts/run_cross_game_transfer.py \
    --config "$GAME_CFG" \
    --data_dir data \
    --output_dir "$RESULTS/cross_game_transfer" \
    --num_epochs 2 --eval_episodes 20 \
    --skip_single --seed $SEED \
    > "$LOG/A4_cross_transfer.log" 2>&1 &

# GPU 3: Ablation — no SFT warmup
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
# Phase 5: More ablations (if GPU free)
# ============================================================================
log "===== Phase 5: Additional Ablations ====="

# Ablation: fewer GRPO iterations (2 vs 3)
CUDA_VISIBLE_DEVICES=0 python scripts/run_grpo_self_play.py \
    --config "$GAME_CFG" \
    --sft_dir "$RESULTS/sft_agents" \
    --output_dir "$RESULTS/ablation_2iter_grpo" \
    --num_iterations 2 \
    --episodes_per_iter 2000 \
    --eval_episodes 20 --batch_size 256 \
    --seed $SEED \
    > "$LOG/D2_ablation_2iter.log" 2>&1 &

# Ablation: fewer Nash-DPO iterations (1 vs 2)
CUDA_VISIBLE_DEVICES=1 python scripts/train_nash_dpo.py \
    --config "$ROLE_CFG" \
    --agents_dir "$RESULTS/sft_role_agents" \
    --output_dir "$RESULTS/ablation_1iter_nash" \
    --num_iterations 1 \
    --games_per_iter 200 \
    --seed $SEED \
    > "$LOG/D3_ablation_1iter_nash.log" 2>&1 &

# Ablation: Nash-DPO with equal weights (vs Nash bargaining)
CUDA_VISIBLE_DEVICES=2 python scripts/train_nash_dpo.py \
    --config "$ROLE_CFG" \
    --agents_dir "$RESULTS/sft_role_agents" \
    --output_dir "$RESULTS/ablation_equal_weights" \
    --num_iterations 2 \
    --games_per_iter 200 \
    --nash_weights equal \
    --seed $SEED \
    > "$LOG/D4_ablation_equal_weights.log" 2>&1 &

wait
log "Phase 5 done."

# ============================================================================
# Phase 6: Collect results & Visualize
# ============================================================================
log "===== Phase 6: Collect & Visualize ====="

python scripts/collect_and_visualize.py \
    --results_dir "$RESULTS" --output_dir to_human \
    > "$LOG/collect_visualize.log" 2>&1 || true

log "============================================="
log "ALL EXPERIMENTS COMPLETE"
log "Results: $RESULTS/"
log "  GRPO:       grpo_self_play/"
log "  Nash-DPO:   nash_dpo/"
log "  Comparison: grpo_vs_nash/"
log "  Benchmarks: eval_benchmarks/"
log "  Transfer:   cross_game_transfer/"
log "  Ablations:  ablation_*/"
log "  Report:     to_human/"
log "============================================="
