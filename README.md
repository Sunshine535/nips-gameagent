# GameAgent: Game-Theoretic Self-Play for Strategic Reasoning and Multi-Objective Alignment in LLMs

---

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/Sunshine535/nips-gameagent.git
cd nips-gameagent

# 2. Install dependencies
bash setup.sh

# 3. Run all experiments
bash run.sh

# 4. (Optional) Run in background for long experiments
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

### Check Completion

```bash
cat results/.pipeline_done   # Shows PIPELINE_COMPLETE when all phases finish
ls results/.phase_markers/   # See which individual phases completed
```

### Save and Send Results

```bash
# Option A: Push to GitHub
git add results/ logs/
git commit -m "Experiment results"
git push origin main

# Option B: Package as tarball
bash collect_results.sh
# Output: results_archive/nips-gameagent_results_YYYYMMDD_HHMMSS.tar.gz
```

### Resume After Interruption

Re-run `bash run.sh` — completed phases are automatically skipped.
To force re-run all phases: `FORCE_RERUN=1 bash run.sh`

## Overview

GameAgent is a unified framework that trains LLMs through game-theoretic self-play to simultaneously improve **strategic decision-making** and **multi-objective alignment**. The framework introduces two complementary training paradigms:

- **Track A (GRPO Self-Play)**: Trains agents across 10 diverse game scenarios (Prisoner's Dilemma, Stag Hunt, Chicken, Matching Pennies, etc.) using Group Relative Policy Optimization, demonstrating emergent strategic reasoning and cross-game transfer.

- **Track B (Nash-DPO Self-Play)**: Deploys 4 asymmetric agents (accuracy, safety, efficiency, creativity) that cross-evaluate each other's outputs and train via Nash-DPO, finding Pareto-optimal policies across competing objectives.

The key insight is that game-theoretic self-play provides a unified mechanism for both strategic capability (Track A) and preference alignment (Track B), and the two tracks produce complementary improvements.

## Project Structure

```
nips-gameagent/
├── src/
│   ├── __init__.py
│   ├── game_environments.py         # 10 multi-round game environments (GRPO track)
│   ├── game_environments_simple.py  # 8 single-round games (Nash-DPO track)
│   ├── game_protocol.py             # Cross-evaluation protocol, reward functions, majority voting
│   ├── nash_dpo.py                  # Multi-objective Nash-DPO loss
│   ├── reward_models.py             # Robust reward functions (NLI, anti-hacking)
│   └── visualization.py             # Publication-quality plots (6 types)
├── scripts/
│   ├── run_all_experiments.sh       # ★ Master pipeline (both tracks)
│   ├── gpu_utils.sh                 # Auto GPU detection (4-8 A100 adaptive)
│   │
│   │  # Track A: GRPO Self-Play
│   ├── generate_sft_data.py         # SFT data generation (10 games)
│   ├── train_sft_warmup.py          # Warmup: 4 agents × 2-3 games
│   ├── run_grpo_self_play.py        # GRPO self-play loop (5 iterations)
│   ├── run_cross_game_transfer.py   # Cross-game transfer evaluation
│   │
│   │  # Track B: Nash-DPO Self-Play
│   ├── generate_expert_data.py      # Expert data generation (8 games)
│   ├── train_sft_agents.py          # Role SFT: 4 agents + reward-based curriculum
│   ├── train_nash_dpo.py            # Nash-DPO iterative training (3 iterations)
│   │
│   │  # Evaluation & Comparison
│   ├── run_grpo_vs_nash_comparison.py  # Head-to-head GRPO vs Nash-DPO
│   ├── eval_benchmarks.py           # Unified: ARC + StrategyQA + BBH + GSM8K + TruthfulQA + MT-Bench
│   ├── eval_game_performance.py     # Game-theoretic performance metrics
│   ├── collect_and_visualize.py     # Result collection + HTML report
│   │
│   │  # Deprecated (from GameRefine, superseded by above)
│   ├── train_agents.py              # → use train_sft_agents.py
│   ├── run_self_play.py             # → use train_nash_dpo.py
│   ├── run_gamerefine.sh            # → use run_all_experiments.sh --skip_grpo
│   └── eval_gamerefine.py           # → use eval_benchmarks.py
├── configs/
│   ├── game_scenarios.yaml          # 10 game definitions + GRPO hyperparams
│   └── agent_roles.yaml             # 4 agent roles + Nash-DPO hyperparams
├── paper/
│   └── outline.md                   # NeurIPS paper outline
├── requirements.txt
├── setup.sh
└── README.md
```

## Game Environments (10 Total)

| Game | Type | Players | Key Property |
|------|------|---------|-------------|
| Prisoner's Dilemma | Symmetric | 2 | Cooperation vs. Defection |
| Coordination Game | Symmetric | 2 | Focal point selection |
| Battle of the Sexes | Asymmetric | 2 | Asymmetric preferences |
| Stag Hunt | Symmetric | 2 | Trust and risk |
| Chicken (Hawk-Dove) | Asymmetric | 2 | Brinkmanship |
| Matching Pennies | Zero-sum | 2 | Mixed strategy NE |
| Public Goods | N-player | 4 | Free-riding temptation |
| Ultimatum | Sequential | 2 | Fairness norms |
| Sealed-Bid Auction | N-player | 4 | Competitive bidding |
| Multi-Issue Negotiation | Sequential | 2 | Multi-issue bargaining |

## Experiment Pipeline

### Track A: GRPO Self-Play
1. **SFT Data Generation**: Play 5000 episodes per game with base LLM, filter top 30%
2. **SFT Warmup**: Train 4 LoRA agents, each specializing in 2-3 games
3. **GRPO Self-Play**: 5 iterations of multi-agent self-play with policy optimization
4. **Cross-Game Transfer**: Train on 4 games, evaluate zero-shot on 6 held-out games

### Track B: Nash-DPO Self-Play
1. **Expert Data**: Generate game-play data across 8 simple games
2. **Role SFT**: Train 4 specialized agents (accuracy, safety, efficiency, creativity)
3. **Nash-DPO**: 3 iterations of cross-evaluation → preference aggregation → Nash-DPO update

### Cross-Comparison (Key Contribution)
- **GRPO vs Nash-DPO** on all 10 games and 6 benchmarks
- Multi-objective Pareto analysis across correctness/safety/efficiency/creativity
- Ablation: no SFT warmup, fewer iterations, single-track vs. combined

## Evaluation Benchmarks

| Benchmark | Domain | Metrics |
|-----------|--------|---------|
| ARC-Challenge | Scientific reasoning | Accuracy |
| StrategyQA | Multi-step strategic reasoning | Accuracy |
| BIG-Bench Hard | Diverse reasoning | Accuracy |
| GSM8K | Mathematical reasoning | Accuracy |
| TruthfulQA | Truthfulness / safety | Accuracy |
| MT-Bench | Open-ended quality | Score (1-10) |
| Game Performance | All 10 games | Avg payoff, Nash rate, diversity |

## Requirements

- Python 3.10+
- PyTorch 2.10+ with CUDA 12.8
- 4-8 NVIDIA A100 GPUs (80GB recommended)
- ~200GB disk for model weights and checkpoints
