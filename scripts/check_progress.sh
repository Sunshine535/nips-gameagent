#!/usr/bin/env bash
# Quick pipeline progress checker
# Usage: bash scripts/check_progress.sh
# Or remotely: sshpass -p '123456' ssh -p 30022 wujn@root@ssh-362.default@222.223.106.147 'cd /gfs/space/private/wujn/Research/nips-gameagent && bash scripts/check_progress.sh'

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

echo "======================================="
echo "  GameAgent Pipeline Progress"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================="

echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "=== Pipeline Process ==="
ps aux | grep "run_all_experiments\|run_reviewer" | grep -v grep | head -3
echo ""

echo "=== Phase Markers ==="
for phase in A1 A2 A3 A4 B1 B2 B3 C1 C2 D1 D2 D3 E1 E2 E3 E4 E5 E6 E7; do
    log="logs/${phase}_*.log"
    if ls $log &>/dev/null 2>&1; then
        lines=$(cat $log 2>/dev/null | wc -l)
        if grep -q "DONE:" logs/${phase}_*.log 2>/dev/null; then
            echo "  $phase: ✅ DONE"
        else
            echo "  $phase: 🟡 RUNNING ($lines lines)"
        fi
    fi
done

echo ""
echo "=== Data Files ==="
echo "SFT data (Track A):"
ls -lhS data/sft_*.jsonl 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
count_sft=$(ls data/sft_*.jsonl 2>/dev/null | wc -l)
echo "  Total: ${count_sft}/10 games"

echo ""
echo "Expert data (Track B):"
ls -lhS results/expert_data/expert_*.jsonl 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'

echo ""
echo "=== Model Checkpoints ==="
for d in results/sft_agents/agent_*/final results/grpo_self_play/final/agent_* results/nash_dpo/iter* results/combined_AB; do
    [ -d "$d" ] && echo "  $d ✅" || true
done

echo ""
echo "=== Results ==="
for f in results/eval_benchmarks/benchmark_results.json results/grpo_self_play/training_log.json results/nash_dpo/nash_dpo_summary.json results/cross_game_transfer/transfer_results.json results/grpo_vs_nash/comparison_results.json; do
    [ -f "$f" ] && echo "  $f ✅" || true
done

echo ""
echo "=== Pipeline Completion ==="
[ -f "results/.pipeline_done" ] && echo "PIPELINE COMPLETE ✅" || echo "Pipeline still running..."
echo "======================================="
