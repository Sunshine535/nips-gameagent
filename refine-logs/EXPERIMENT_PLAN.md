# GameAgent: Experiment Plan

**Direction**: Game-theoretic self-play for strategic reasoning and multi-objective alignment in LLMs
**Base Model**: Qwen/Qwen3.5-9B
**Hardware**: 4× NVIDIA H800 80GB
**Estimated Total GPU-Hours**: ~188 GPU-hours (47h wall-clock with 4 GPUs)

## Run Order and Milestones

### M0: Sanity — Data Generation Validation
**Priority**: MUST-RUN | **GPU-hours**: ~24 (6h × 4 GPUs)

| Run | Description | Command | Success Criterion |
|-----|-------------|---------|-------------------|
| R001 | GRPO SFT data (10 games × 500 ep) | `parallel_generate generate_sft_data.py` | All 10 `sft_*.jsonl` files, each >100KB |
| R002 | Nash-DPO expert data (8 games × 500 ep) | `parallel_generate generate_expert_data.py` | `expert_train.jsonl` + `expert_val.jsonl` |

### M1: SFT Warmup — Baseline Agents
**Priority**: MUST-RUN | **GPU-hours**: ~8 (2h × 4 GPUs)

| Run | Description | Command | Success Criterion |
|-----|-------------|---------|-------------------|
| R003 | 4 GRPO warmup agents (parallel) | `train_sft_warmup.py --parallel` | 4 LoRA adapters saved |
| R004 | 4 Nash-DPO role agents | `train_sft_agents.py` | 4 role-specialized adapters |

### M2: Main Method — Self-Play Training
**Priority**: MUST-RUN | **GPU-hours**: ~96 (24h × 4 GPUs)

| Run | Description | Command | Success Criterion |
|-----|-------------|---------|-------------------|
| R005 | GRPO self-play (3 iter × 2K ep) | `run_grpo_self_play.py` | avg_payoff increases over iterations |
| R006 | Nash-DPO self-play (2 iter × 200 games) | `train_nash_dpo.py` | Elo ratings converge |

### M3: Evaluation — Cross-Comparison
**Priority**: MUST-RUN | **GPU-hours**: ~20 (5h × 4 GPUs)

| Run | Description | Command | Success Criterion |
|-----|-------------|---------|-------------------|
| R007 | Cross-game transfer | `run_cross_game_transfer.py` | Transfer matrix with positive entries |
| R008 | GRPO vs Nash-DPO head-to-head | `run_grpo_vs_nash_comparison.py` | Complementary strength pattern |
| R009 | Unified benchmarks (ARC, StrategyQA, BBH, GSM8K, TruthfulQA, MT-Bench) | `eval_benchmarks.py` | At least one metric improves over base |

### M4: Ablation Studies
**Priority**: MUST-RUN | **GPU-hours**: ~40 (10h × 4 GPUs)

| Run | Description | Command | Success Criterion |
|-----|-------------|---------|-------------------|
| R010 | Ablation: no SFT warmup | `run_grpo_self_play.py --sft_dir __nonexistent__` | Lower than R005 |
| R011 | Ablation: fewer GRPO iters (2) | `run_grpo_self_play.py --num_iterations 2` | Shows iteration benefit |
| R012 | Ablation: fewer Nash-DPO iters (1) | `train_nash_dpo.py --num_iterations 1` | Shows iteration benefit |

## Compute Budget

| Milestone | Experiments | Est. GPU-Hours | Est. Wall-Clock |
|-----------|------------|----------------|-----------------|
| M0: Data | 2 | 24 | 6h |
| M1: SFT | 2 | 8 | 2h |
| M2: Self-Play | 2 | 96 | 24h |
| M3: Evaluation | 3 | 20 | 5h |
| M4: Ablation | 3 | 40 | 10h |
| **Total** | **12** | **188** | **47h** |

## Claims and Hypotheses

| Claim | Experiments | Expected Evidence |
|-------|-----------|-------------------|
| H1: GRPO → strategic reasoning | R005, R007, R009 | ARC/StrategyQA/BBH improvement |
| H2: Nash-DPO → alignment | R006, R009 | GSM8K/TruthfulQA/MT-Bench improvement |
| H3: Cross-game transfer | R007 | Positive transfer matrix |
| H4: Complementarity | R008, R009 | GRPO+Nash > either alone |
| H5: Emergent behavior | R005, R006 | Strategy diversity increases |

## Method Details

See `PROPOSAL.md` Section 2 for full method description.
See `configs/game_scenarios.yaml` and `configs/agent_roles.yaml` for hyperparameters.
