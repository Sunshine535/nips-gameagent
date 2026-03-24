#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo '=============================='
echo '  GameAgent Pipeline Monitor'
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo '=============================='
echo ''

# Process check
PID=$(pgrep -f 'run_grpo_self_play.py|train_nash_dpo|eval_benchmarks|run_cross_game|generate_expert|resume_from' 2>/dev/null | head -1)
if [ -n "$PID" ]; then
    ETIME=$(ps -p $PID -o etime --no-headers 2>/dev/null | xargs)
    CMD=$(ps -p $PID -o args --no-headers 2>/dev/null | head -c 80)
    echo "[RUNNING] PID=$PID, Uptime=$ETIME"
    echo "  Cmd: $CMD"
else
    echo '[IDLE] No pipeline process detected'
fi
echo ''

# Latest progress from any log
for logf in logs/A3_grpo_self_play.log logs/A4_*.log logs/B1_*.log logs/B2_*.log logs/B3_*.log logs/C1_*.log logs/C2_*.log logs/D1_*.log logs/D2_*.log logs/D3_*.log; do
    if [ -f "$logf" ]; then
        LATEST_LOG="$logf"
    fi
done 2>/dev/null

if [ -n "${LATEST_LOG:-}" ]; then
    echo "Latest log: $LATEST_LOG"
    PROGRESS=$(tail -1 "$LATEST_LOG" 2>/dev/null | grep -oP '\w+:\s+\d+%.*?\]' | tail -1)
    echo "Current: $PROGRESS"
    echo ''
    echo 'Recent milestones:'
    grep -E '\[INFO\].*(Iteration|Generating|Running|Evaluating|DONE:|START:|Complete|Saving)' "$LATEST_LOG" 2>/dev/null | tail -5 || true
fi
echo ''

# Results summary
echo 'Results directories:'
for d in grpo_self_play nash_dpo cross_game_transfer grpo_vs_nash eval_benchmarks ablation_no_sft ablation_2iter_grpo ablation_1iter_nash; do
    if [ -d "results/$d" ]; then
        nf=$(find "results/$d" -name '*.json' 2>/dev/null | wc -l)
        echo "  [OK] $d ($nf json)"
    else
        echo "  [--] $d"
    fi
done
echo ''

if [ -f results/.pipeline_done ]; then
    echo '*** PIPELINE COMPLETE ***'
else
    echo 'Status: IN PROGRESS'
fi
echo ''

nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv 2>/dev/null || true
