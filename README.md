# GameAgent: Game-Theoretic Self-Play for Strategic Reasoning and Multi-Objective Alignment in LLMs

A unified framework combining GRPO self-play (Track A) and formal Nash-DPO (Track B) to simultaneously improve strategic decision-making and multi-objective alignment in language models.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Sunshine535/nips-gameagent.git
cd nips-gameagent

# 2. Setup (auto-detects system PyTorch, conda, uv, or venv)
bash setup.sh

# 3. Smoke test (single seed, reduced data)
bash scripts/run_all_experiments.sh --quick

# 4. Full run (3 seeds × 5 conditions + ablations)
nohup bash scripts/run_all_experiments.sh > run_full.log 2>&1 &
tail -f run_full.log
```

### Resume / Monitor

```bash
# Re-run — completed phases auto-skip via phase markers
bash scripts/run_all_experiments.sh

# Force re-run everything
FORCE_RERUN=1 bash scripts/run_all_experiments.sh

# Check progress
ls results/.phase_markers_full/

# Package results
bash collect_results.sh
```

## Method

### Track A — GRPO Self-Play

Agents play 8 classic game-theory scenarios (Prisoner's Dilemma, Stag Hunt, etc.) and optimize strategies via Group Relative Policy Optimization. Multi-agent self-play drives emergent strategic reasoning beyond what supervised training provides.

### Track B — Formal Nash-DPO

A custom `NashDPOTrainer` (subclassing TRL's `DPOTrainer`) integrates Nash bargaining directly into the DPO training loop:

- **Per-objective preference signals**: Each training pair carries `pref_correctness`, `pref_safety`, `pref_efficiency`, `pref_creativity` indicators
- **KKT-derived Nash weights**: \(w_k \propto \frac{1}{d_k - \mathcal{L}_k}\) where \(d_k = \ln 2\) (random-policy DPO loss) and \(\mathcal{L}_k\) is the running per-objective loss
- **EMA smoothing**: Weights update via exponential moving average for training stability
- Loss reverts per-objective: applies standard DPO for aligned objectives, reverse DPO for misaligned ones

### Combined (A+B)

GRPO checkpoint → Nash-DPO fine-tuning. The factorial design (5 conditions) isolates contributions of each track.

## Experiment Design

| Condition | Training Pipeline | Purpose |
|-----------|------------------|---------|
| BASE | — | Lower bound |
| SFT | SFT warmup only | Supervised baseline |
| A-only | SFT → GRPO self-play | Track A contribution |
| B-only | SFT → Nash-DPO | Track B contribution |
| A+B | SFT → GRPO → Nash-DPO | Full method |

Additional ablations: `equal` (uniform weights), `fixed` (hand-tuned weights), `single_correctness` (one-objective DPO).

Each condition runs with 3 seeds (42, 123, 456).

## Evaluation

| Benchmark | Domain | Why |
|-----------|--------|-----|
| StrategyQA | Multi-hop strategic reasoning | Core strategic capability |
| TruthfulQA | Truthfulness / safety | Safety alignment signal |
| MT-Bench | Open-ended generation quality | Generation quality |
| Game Performance | 8 game scenarios | Nash equilibrium rate, payoff, diversity |

## Project Structure

```
nips-gameagent/
├── src/
│   ├── nash_dpo_trainer.py       # NashDPOTrainer: custom DPOTrainer with Nash bargaining
│   ├── nash_dpo_formal.py        # FormalNashDPOLoss, KKT weight computation
│   ├── game_environments_simple.py  # 8 game environments
│   ├── game_environments.py      # Multi-round game environments
│   ├── reward_models.py          # Robust reward functions (NLI, diversity, safety)
│   └── visualization.py          # Publication-quality plots
├── scripts/
│   ├── run_all_experiments.sh    # Master pipeline (5 conditions + ablations)
│   ├── gpu_utils.sh              # Auto GPU detection and config
│   ├── generate_sft_data.py      # SFT training data from game play
│   ├── generate_preference_data.py  # Multi-objective preference pairs for Nash-DPO
│   ├── train_sft_warmup.py       # LoRA SFT warmup (parallel multi-agent)
│   ├── run_grpo_self_play.py     # GRPO self-play loop
│   ├── train_formal_nash_dpo.py  # Nash-DPO training with formal bargaining
│   ├── eval_game_performance.py  # Game-theoretic evaluation
│   └── eval_benchmarks.py        # NLP benchmark evaluation
├── configs/
│   ├── game_scenarios.yaml       # Game definitions + hyperparameters
│   └── agent_roles.yaml          # Agent role definitions
├── requirements.txt
└── setup.sh                      # Multi-strategy environment setup
```

## Requirements

- Python 3.10+
- PyTorch >= 2.1.0 with CUDA
- transformers >= 4.45, < 5.0
- trl >= 0.15, < 0.17
- 4+ GPUs recommended (tested on 4×H800 80GB)

## GPU Adaptation

`gpu_utils.sh` auto-detects GPU count and memory, adapting batch sizes accordingly. Supported:

| GPU | Class | Batch Size |
|-----|-------|------------|
| H800/A100 80GB | `a100_80g` | 4 |
| A100 40GB | `a100_40g` | 2 |
| A10 24GB | `a10_24g` | 1 |

## Multi-GPU / Checkpoint

- GRPO self-play partitions agents across GPUs automatically
- Nash-DPO uses `--resume_from_checkpoint auto` for automatic resumption
- SFT warmup trains 4 agents in parallel across available GPUs
- All training scripts accept `--seed` for reproducibility

## License

MIT
