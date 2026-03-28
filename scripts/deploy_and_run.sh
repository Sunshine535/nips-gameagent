#!/usr/bin/env bash
# Deploy and run full experiment pipeline on server.
# Installs missing deps, then launches run_all_experiments.sh in background.
#
# Usage (from local):
#   bash scripts/deploy_and_run.sh [--quick]

set -euo pipefail

PROJ=/gfs/space/private/wujn/Research/nips-gameagent
LOG=$PROJ/logs/deploy_$(date +%Y%m%d_%H%M%S).log

cd "$PROJ"
mkdir -p logs data results

echo "$(date) - Starting deployment..." | tee "$LOG"

# Activate venv
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "$(date) - Activated .venv" | tee -a "$LOG"
else
    echo "$(date) - Setting up venv..." | tee -a "$LOG"
    if command -v uv &>/dev/null; then
        uv venv .venv --python 3.10 2>&1 | tee -a "$LOG"
        source .venv/bin/activate
        uv pip install torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tee -a "$LOG"
        uv pip install -r requirements.txt 2>&1 | tee -a "$LOG"
    else
        python3 -m venv .venv
        source .venv/bin/activate
        pip install torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tee -a "$LOG"
        pip install -r requirements.txt 2>&1 | tee -a "$LOG"
    fi
fi

# Install sentence-transformers if missing
python -c "import sentence_transformers" 2>/dev/null || {
    echo "$(date) - Installing sentence-transformers..." | tee -a "$LOG"
    pip install sentence-transformers 2>&1 | tee -a "$LOG"
}

echo "$(date) - Checking torch..." | tee -a "$LOG"
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpus={torch.cuda.device_count()}')" 2>&1 | tee -a "$LOG"

echo "$(date) - Checking key deps..." | tee -a "$LOG"
python -c "import transformers, trl, peft, sentence_transformers; print('All deps OK')" 2>&1 | tee -a "$LOG"

echo "$(date) - Environment ready. Starting experiments..." | tee -a "$LOG"

# Run the experiment pipeline
QUICK_FLAG=""
if [ "${1:-}" = "--quick" ]; then
    QUICK_FLAG="--quick"
    echo "$(date) - Quick mode enabled" | tee -a "$LOG"
fi

# HF_ENDPOINT removed (use default huggingface.co)
export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup bash scripts/run_all_experiments.sh $QUICK_FLAG > logs/full_pipeline_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID=$!
echo "$(date) - Pipeline launched as PID=$PID" | tee -a "$LOG"
echo "$PID" > logs/pipeline.pid
echo "$(date) - Deployment complete. Monitor with: tail -f logs/full_pipeline_*.log" | tee -a "$LOG"
