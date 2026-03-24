#!/usr/bin/env bash
cd /gfs/space/private/wujn/Research/nips-gameagent
echo "=== Python ==="
test -f .venv/bin/python && .venv/bin/python --version || echo "NO_VENV"
echo "=== PyTorch ==="
.venv/bin/python -c 'import torch; print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())' 2>&1
echo "=== TRL ==="
.venv/bin/python -c 'import transformers,trl,peft; print("transformers:", transformers.__version__, "trl:", trl.__version__, "peft:", peft.__version__)' 2>&1
echo "=== SentenceTransformers ==="
.venv/bin/python -c 'import sentence_transformers; print("sentence_transformers:", sentence_transformers.__version__)' 2>&1 || echo "MISSING - installing..."
echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv 2>/dev/null
echo "=== Running processes ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null
echo "DONE"
