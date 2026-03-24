# Idea Discovery Report

**Direction**: Game-theoretic self-play for strategic reasoning and multi-objective alignment in LLMs
**Date**: 2026-03-23
**Pipeline**: research-lit → idea-creator → novelty-check → research-review (condensed — idea pre-validated)
**Status**: IDEA LOCKED — proceeding to experiments

## Executive Summary

GameAgent proposes a unified game-theoretic framework with two complementary self-play tracks:
(1) GRPO Self-Play trains agents across 10 diverse game scenarios to improve strategic reasoning;
(2) Nash-DPO Self-Play deploys 4 asymmetric agents that converge to Pareto-optimal policies via Nash bargaining.
The core contribution is demonstrating that game-theoretic self-play serves as a general-purpose LLM post-training mechanism that simultaneously improves reasoning and alignment.

## Literature Landscape

### Key Prior Work
| Area | Key Papers | Gap |
|------|-----------|-----|
| Self-play for LLMs | SPIRAL, SPIN, SPAG (2024) | Single-game, no cross-game transfer study |
| Multi-objective alignment | Reward Soups, MORLHF, Pareto RLHF | No game-theoretic mechanism, manual weight tuning |
| Game theory in ML | GANs, MARL, Nash learning dynamics | Not applied to LLM post-training |
| Nash-based DPO | MNPO (2024), Nash-MD | No asymmetric multi-agent protocol |

### Structural Gaps Identified
1. No framework unifies strategic reasoning (game self-play) with alignment (multi-objective optimization)
2. Cross-game transfer in LLMs is unstudied — does training on Prisoner's Dilemma help with Auction bidding?
3. Existing multi-objective alignment either uses fixed weights or simple majority voting — no Nash bargaining

## Ranked Ideas

### 🏆 Idea 1: GameAgent — RECOMMENDED (LOCKED)
**Hypothesis**: Game-theoretic self-play across diverse environments simultaneously improves strategic reasoning AND multi-objective alignment, and the two paradigms produce complementary improvements.

**Method**:
- Track A: GRPO self-play across 10 games (matrix, N-player, sequential) with 4 specialized LoRA agents
- Track B: Nash-DPO with 4 asymmetric agents (accuracy, safety, efficiency, creativity)
- Cross-comparison on 6 benchmarks + 10 game environments

**Evidence**:
- Literature survey confirms no existing work combines both paradigms
- Codebase fully implemented with robust reward functions (NLI + anti-hacking)
- Prior pilot (GameRefine) identified critical reward hacking issue — now resolved
- Novelty: CONFIRMED — no published work on unified GRPO + Nash-DPO framework

**Reviewer Pre-assessment** (self-review, conservative):
- Score: 5/10 (pending empirical validation — strong concept, execution-dependent)
- Strengths: novel framework, clean experimental design, solid baselines
- Weaknesses: all hypotheses untested, 9B model may be too small for emergent behavior
- Minimum fixes: need positive results on at least H1 (GRPO → strategic reasoning) and H2 (Nash-DPO → alignment)

**Next step**: Run full experiment pipeline → auto-review-loop

### Idea 2: Curriculum Game Transfer — BACKUP
Train agents on an easy→hard game curriculum instead of random game mixing. Could improve cross-game transfer if random mixing fails.

### Idea 3: Reward Model Self-Play — ELIMINATED
Use self-play to improve reward models themselves. Eliminated: orthogonal to main contribution, adds complexity.

## Eliminated Ideas
- **Reward Model Self-Play**: Too far from core thesis
- **Single-Track GRPO-only**: Misses the complementarity contribution
- **Single-Track Nash-DPO-only**: Misses the strategic reasoning contribution

## Refined Proposal
- Proposal: `PROPOSAL.md`
- Experiment plan: `EXPERIMENTS.md`
- Configuration: `configs/game_scenarios.yaml`, `configs/agent_roles.yaml`

## Next Steps
- [x] Implementation complete
- [ ] SFT data generation (IN PROGRESS — 6/10 games done)
- [ ] SFT warmup training
- [ ] GRPO self-play (3 iterations)
- [ ] Nash-DPO training (2 iterations)
- [ ] Cross-comparison & benchmarks
- [ ] Ablation studies
- [ ] /auto-review-loop "GameAgent"
