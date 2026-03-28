#!/bin/bash
# ============================================================================
# Single-session server runner for GameAgent
#
# Designed for containerized environments where:
# - Each SSH connection may hit a different container
# - /tmp is not persistent between sessions
# - Project directory at $HOME is on shared NFS
# - Must complete everything in one session
#
# Usage: 
#   cat scripts/single_session_run.sh | ssh server "bash"
# ============================================================================
P=$HOME/nips_15/github_repos/nips-gameagent
cd "$P" || { echo "FATAL: cannot cd to $P"; exit 1; }
echo "=== GameAgent Single Session Runner ==="
echo "TIME: $(date)"
echo "HOST: $(hostname)"
echo "PWD: $(pwd)"

echo "=== Step 1: Environment ==="
which python3 || { echo "FATAL: no python3"; exit 1; }
python3 --version
echo "=== Pip ==="
pip3 --version 2>/dev/null || python3 -m pip --version 2>/dev/null || echo "WARN: no pip found"
echo "=== GPU ==="
nvidia-smi -L 2>/dev/null || echo "NO GPU available"
echo "=== Torch ==="
python3 -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())' 2>&1 || echo "WARN: no torch"
echo "=== TRL ==="
python3 -c 'import transformers, trl, peft; print("deps OK")' 2>&1 || echo "WARN: missing deps, will install"

echo "=== Step 2: Install deps ==="
export PATH=$HOME/.local/bin:$PATH
pip3 install --user torch transformers trl peft datasets accelerate pyyaml scipy matplotlib tqdm pandas huggingface_hub evaluate sentence-transformers 2>&1 || python3 -m pip install --user torch transformers trl peft datasets accelerate pyyaml scipy matplotlib tqdm pandas huggingface_hub evaluate sentence-transformers 2>&1 || echo "WARN: pip install failed"

echo "=== Step 3: Verify ==="
python3 -c 'import torch; print("torch OK:", torch.__version__, "cuda:", torch.cuda.is_available())' 2>&1
python3 -c 'import transformers, trl, peft; print("ALL DEPS OK")' 2>&1

echo "=== Step 4: Run pipeline ==="
# HF_ENDPOINT removed (use default huggingface.co)
export PYTHONPATH=$P:$PYTHONPATH
mkdir -p data results logs
bash scripts/run_all_experiments.sh --quick 2>&1

echo "=== DONE $(date) ==="
ls -la results/ 2>/dev/null
