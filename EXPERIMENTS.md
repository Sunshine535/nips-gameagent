# Experiment Log: GameAgent

## Status Summary (2026-03-23, post-merge)

### Project Consolidation
The **nips-gamerefine** and **nips-gameagent** projects have been merged into a single
unified codebase. GameRefine's asymmetric multi-agent self-play concept is now Track B
(Nash-DPO) of the GameAgent framework.

**Merged components:**
- `train_agents.py` reward-based data filtering → `train_sft_agents.py`
- `run_self_play.py` Nash-DPO loop → `train_nash_dpo.py`
- `eval_gamerefine.py` majority-vote selection → `game_protocol.py`
- Legacy scripts (`run_gamerefine.sh`, `eval_gamerefine.py`, `train_agents.py`, `run_self_play.py`) retained with DEPRECATED headers

### Current State
- **Track A (GRPO Self-Play)**: NOT started (new pipeline)
- **Track B (Nash-DPO Self-Play)**: NOT started (new pipeline)
- **Legacy GameRefine results**: Archived — not reusable due to reward hacking
- **Server**: 4× NVIDIA H800 80GB at `222.223.106.147:30022`
- **Code**: Unified, all known bugs fixed, reward functions overhauled

### Legacy Results (NOT Reusable)

| Experiment | Status | Why Not Reusable |
|-----------|--------|-----------------|
| Safety Agent SFT (ckpt-800) | Completed | Trained with gameable heuristic rewards |
| GRPO 670/2500 steps | Partial | Reward saturated to 1.0 at step ~300, training meaningless |

### Known Bugs Fixed
1. `eval_game_performance.py`: Import error fixed (`game_environments` → `game_environments_simple`)
2. `train_agents.py`: Truncated file completed
3. `eval_benchmarks.py`: Dead import of `create_environment` removed
4. Config key inconsistency: both `model.base` and `model.base_model` now present in both YAMLs
5. `max_new_tokens` reduced from 512 → 256 to prevent reward hacking via verbosity
6. `collect_and_visualize.py`: Hard-coded paths replaced with `--results_dir` parameter
7. `plot_transfer_matrix`: Fixed JSON schema compatibility and type annotation

### Key Improvements Over Legacy
| Area | GameRefine (old) | GameAgent (new) |
|------|-----------------|-----------------|
| Correctness reward | Word overlap | NLI entailment + semantic similarity |
| Safety reward | 12-word blacklist | Multi-level pattern detection + severity weighting |
| Efficiency reward | Word count buckets | Information density (windowed TTR + coverage) |
| Creativity reward | Unique word ratio | Sliding-window TTR + sentence variety + structure |
| Anti-hacking | None | Length penalty + repetition penalty |
| SFT data selection | Random | Reward-based curriculum (top-k by agent reward) |
| Evaluation | GSM8K + TruthfulQA + MT-Bench | + ARC + StrategyQA + BBH |
| Pipeline | 3-stage linear | 10-step dual-track with cross-comparison |

---

## Research Plan: Weeks to NeurIPS 2026

**Deadline**: July 17, 2026

### Phase 0: Critical Fixes (Week 1: Mar 23–29) ✅ DONE

- [x] Reward function overhaul (`src/reward_models.py`)
- [x] Anti-reward-hacking (length + repetition penalties)
- [x] Bug fixes (3 scripts)
- [x] Code sync setup (`scripts/sync.sh`)
- [x] Project merge (GameRefine → GameAgent)
- [x] Config consistency fixes
- [x] Visualization pipeline (`src/visualization.py`)
- [x] Paper outline (`paper/outline.md`)

### Phase 1: SFT Data Generation & Warmup (Week 2: Mar 30 – Apr 5)

#### Track A (GRPO)
- [ ] Run `generate_sft_data.py` on server: 500 episodes × 10 games
- [ ] Train 4 SFT warmup agents: ~8 hours on 4× H800
- [ ] Validate SFT quality: check agent plays reasonable game strategies

#### Track B (Nash-DPO)
- [ ] Run `generate_expert_data.py`: 500 episodes × 8 games
- [ ] Train 4 role-specialized agents (accuracy/safety/efficiency/creativity)
- [ ] Evaluate each agent on its target dimension

### Phase 2: Self-Play Training (Weeks 3–4: Apr 6–19)

#### Track A: GRPO Self-Play (3 iterations)
- [ ] Iteration 1–3: 2,000 episodes each, GRPO update, eval
- [ ] Monitor: avg payoff, strategy diversity, Nash distance
- [ ] Early stop if reward saturates

#### Track B: Nash-DPO Self-Play (2 iterations)
- [ ] Iteration 1–2: 200 games cross-eval → preference pairs → DPO update
- [ ] Monitor: Elo ratings, Nash bargaining weights

### Phase 3: Evaluation & Cross-Comparison (Week 5: Apr 20–26)

- [ ] Cross-game transfer (single-game vs multi-game vs curriculum)
- [ ] GRPO vs Nash-DPO on ALL benchmarks and ALL games
- [ ] Multi-objective Pareto analysis

### Phase 4: Ablation Studies (Week 6: Apr 27 – May 3)

| Ablation | Configurations |
|----------|---------------|
| SFT warmup | With vs without |
| GRPO iterations | 1, 2, 3 |
| Nash-DPO iterations | 1, 2 |
| Number of agents | 2, 3, 4 |
| Game diversity | 2, 4, 6, 8 games |
| LoRA rank | 16, 32, 64 |
| Nash arbitration | Nash bargaining vs majority vote vs weighted average |

### Phase 5: Analysis & Visualization (Week 7: May 4–10)

- [ ] Pareto front (4D projections)
- [ ] Convergence plots
- [ ] Emergent behavior analysis
- [ ] Game transcript examples

### Phase 6: Paper Writing (Weeks 8–10: May 11 – May 31)

- [ ] Introduction + Related Work
- [ ] Method (GameAgent framework, Nash-DPO algorithm)
- [ ] Experiments (tables, figures)
- [ ] Conclusion + Appendix

### Phase 7: Revision & Polish (Weeks 11–16: Jun 1 – Jul 13)

- [ ] Internal review
- [ ] Additional experiments
- [ ] Camera-ready

---

## Experiment Results

### New Pipeline (GameAgent) — To Be Filled

#### Track A: GRPO Self-Play
*Pending Phase 1*

#### Track B: Nash-DPO Self-Play
*Pending Phase 1*

#### Cross-Comparison
*Pending Phase 3*

#### Ablations
*Pending Phase 4*
