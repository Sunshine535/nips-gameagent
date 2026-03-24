#!/bin/bash
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo " GameAgent: Environment Setup"
echo "============================================"

VENV_DIR="$PROJ_DIR/.venv"

# Strategy 1: uv (fastest)
if command -v uv &>/dev/null; then
    echo "[Setup] Using uv"
    if [ ! -d "$VENV_DIR" ]; then
        uv venv "$VENV_DIR" --python 3.10 2>/dev/null || uv venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
    uv pip install "torch>=2.1.0" --index-url https://download.pytorch.org/whl/cu128 2>/dev/null || \
        uv pip install "torch>=2.1.0" --index-url https://download.pytorch.org/whl/cu121
    uv pip install -r "$PROJ_DIR/requirements.txt" \
        --index-strategy unsafe-best-match 2>/dev/null || \
        uv pip install -r "$PROJ_DIR/requirements.txt"
    uv pip install flash-attn --no-build-isolation 2>/dev/null || true

# Strategy 2: conda (common on GPU servers)
elif [ -f "/home/nwh/anaconda3/bin/conda" ] || command -v conda &>/dev/null; then
    CONDA_BIN="${CONDA_EXE:-/home/nwh/anaconda3/bin/conda}"
    if [ ! -f "$CONDA_BIN" ]; then
        CONDA_BIN="$(which conda 2>/dev/null || echo /home/nwh/anaconda3/bin/conda)"
    fi
    echo "[Setup] Using conda at $CONDA_BIN"
    eval "$($CONDA_BIN shell.bash hook 2>/dev/null)" || export PATH="/home/nwh/anaconda3/bin:$PATH"

    if ! conda env list 2>/dev/null | grep -q gameagent; then
        echo "[Setup] Creating conda env 'gameagent'..."
        conda create -y -n gameagent python=3.10
    fi
    conda activate gameagent 2>/dev/null || source activate gameagent
    pip install "torch>=2.1.0" --index-url https://download.pytorch.org/whl/cu128 2>/dev/null || \
        pip install "torch>=2.1.0" --index-url https://download.pytorch.org/whl/cu121
    pip install -r "$PROJ_DIR/requirements.txt"
    pip install flash-attn --no-build-isolation 2>/dev/null || true
    # Create a shim so run.sh can find the env
    mkdir -p "$VENV_DIR/bin"
    ln -sf "$(which python)" "$VENV_DIR/bin/python" 2>/dev/null || true
    echo "conda activate gameagent" > "$VENV_DIR/bin/activate"

# Strategy 3: system python + venv
else
    echo "[Setup] Using system python3"
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR" || python3.10 -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install "torch>=2.1.0" --index-url https://download.pytorch.org/whl/cu128 2>/dev/null || \
        pip install "torch>=2.1.0" --index-url https://download.pytorch.org/whl/cu121
    pip install -r "$PROJ_DIR/requirements.txt"
    pip install flash-attn --no-build-isolation 2>/dev/null || true
fi

echo ""
echo "============================================"
python -c "
import torch
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA     : {torch.version.cuda}')
print(f'  GPUs     : {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>/dev/null || echo "  (torch check skipped)"
echo "============================================"
echo "Setup complete!"
