#!/usr/bin/env bash
# ============================================================================
# GameAgent: One-shot server bootstrap
#
# SSH into the server and run:
#   cd /gfs/space/private/wujn/Research/nips-gameagent
#   bash scripts/server_bootstrap.sh          # full run
#   bash scripts/server_bootstrap.sh --quick  # smoke test (~1 hour)
# ============================================================================
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
echo "$(date) [BOOTSTRAP] Project dir: $PROJ"
echo "$(date) [BOOTSTRAP] Mode: ${1:-full}"

# ── 1. Environment Setup ────────────────────────────────────────────────────

setup_env() {
    if [ -f .venv/bin/activate ]; then
        source .venv/bin/activate
        echo "$(date) [ENV] Activated existing .venv"
    else
        echo "$(date) [ENV] Creating new venv..."
        if command -v uv &>/dev/null; then
            uv venv .venv --python 3.10
            source .venv/bin/activate
            uv pip install torch --index-url https://download.pytorch.org/whl/cu128
            uv pip install -r requirements.txt
        else
            python3 -m venv .venv || python3.10 -m venv .venv
            source .venv/bin/activate
            pip install --upgrade pip
            pip install torch --index-url https://download.pytorch.org/whl/cu128
            pip install -r requirements.txt
        fi
    fi

    # Install any missing deps
    python -c "import sentence_transformers" 2>/dev/null || pip install sentence-transformers
    python -c "import flash_attn" 2>/dev/null || pip install flash-attn --no-build-isolation 2>/dev/null || true

    echo "$(date) [ENV] Dependency check..."
    python -c "
import torch, transformers, trl, peft
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')
print(f'transformers={transformers.__version__} trl={trl.__version__} peft={peft.__version__}')
"
}

# ── 2. Main Pipeline ────────────────────────────────────────────────────────

mkdir -p data results logs

setup_env 2>&1 | tee logs/bootstrap_env.log

# HF_ENDPOINT removed (use default huggingface.co)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$PROJ:$PYTHONPATH"

QUICK_FLAG="${1:-}"
LOGFILE="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "$(date) [BOOTSTRAP] Starting pipeline → $LOGFILE"
echo "$(date) [BOOTSTRAP] Monitor: tail -f $PROJ/$LOGFILE"

# Run in foreground so we can see output
exec bash scripts/run_all_experiments.sh $QUICK_FLAG 2>&1 | tee "$LOGFILE"
