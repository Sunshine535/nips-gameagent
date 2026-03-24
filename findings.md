# Findings: GameAgent

## Project Consolidation (2026-03-23)

GameRefine and GameAgent have been merged into a single unified project. The old GameRefine pipeline (asymmetric multi-agent self-play) is now Track B (Nash-DPO) of the GameAgent framework.

### Legacy Results Assessment
All GameRefine experimental results are **NOT reusable**:
- Safety Agent SFT (checkpoint-800): Trained with gameable heuristic rewards
- GRPO training (670/2500 steps): Reward saturated to 1.0 at step ~300

Root cause: keyword-count and word-length rewards are trivially exploitable by LLMs.

## Current Understanding

GameAgent proposes a unified game-theoretic framework with two complementary tracks:
- **Track A (GRPO)**: Strategic reasoning via self-play in 10 diverse game environments
- **Track B (Nash-DPO)**: Multi-objective alignment via 4 asymmetric agents with Nash bargaining

The core thesis is that game-theoretic self-play is a general mechanism for LLM post-training that simultaneously improves strategic reasoning AND multi-objective alignment.

## Patterns and Insights

### Reward Design is the Bottleneck (Critical — Resolved)
Old pipeline showed reward saturation within 300 GRPO steps. Heuristic proxy rewards (keyword counting, word length) are trivially gameable by LLMs. Model learns to produce long, diverse-worded outputs that score 1.0 on all dimensions without genuine capability improvement.

**Resolution**: Created `src/reward_models.py` with:
- NLI entailment scoring (DeBERTa-v3) for correctness
- Multi-level pattern detection with severity weighting for safety
- Information density (windowed TTR, coverage, length bell curve) for efficiency
- Sliding-window TTR + sentence variety + structural signals for creativity
- Anti-hacking: length penalty + repetition penalty

### Reward-Based Curriculum Improves SFT Quality
Migrated from GameRefine: scoring expert data by agent-specific reward weights and keeping only top-50% significantly improves downstream agent quality vs random selection. Now integrated into `train_sft_agents.py`.

### max_new_tokens Must Be Capped
512 tokens was identified as enabling reward hacking via verbosity. Reduced to 256 across all generation calls. GRPO `clipped_ratio` should be monitored — if it approaches 1.0, the cap should be lowered further.

## Lessons and Constraints

1. **Never use keyword-count rewards for LLM training** — models exploit them instantly
2. **Monitor `clipped_ratio`** during GRPO — if it hits 1.0, all completions are truncated and reward signal is degenerate
3. **Entropy tracking** is critical — stable entropy around 1.0 with perfect reward indicates a fixed degenerate policy
4. **Length penalty is mandatory** — without it, models maximize verbosity for reward
5. **Config keys must be consistent** — `base` vs `base_model` caused silent failures across scripts

## Open Questions

1. Can NLI-based rewards provide enough training signal for GRPO convergence?
2. Will Nash-DPO weights actually converge to meaningful equilibrium with 4 heterogeneous agents?
3. Does cross-game transfer require curriculum learning or does random mixing suffice?
4. How many GRPO iterations are needed before strategic reasoning benchmarks improve?
5. Is Qwen3.5-9B large enough for emergent game-theoretic behavior?
6. Does reward-based SFT curriculum consistently outperform random selection?
