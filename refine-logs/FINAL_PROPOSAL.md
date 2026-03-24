# Final Proposal: GameAgent

## Problem Anchor

**Core Question**: Can game-theoretic self-play simultaneously improve strategic reasoning AND multi-objective alignment in LLMs?

**Motivation**: Current LLM post-training (RLHF/DPO) treats alignment as single-objective and does not train for strategic reasoning. Game theory provides a natural framework for both: competitive/cooperative dynamics for reasoning, Nash bargaining for multi-objective optimization.

**Scope Constraint**: We focus on training a single 9B-parameter model (Qwen3.5-9B) through two complementary self-play paradigms, evaluating on established benchmarks. We do NOT claim to solve general alignment — only that game-theoretic self-play is a promising training mechanism.

## Method Thesis

**GameAgent** is a unified framework with two complementary tracks:

1. **Track A (GRPO Self-Play)**: 4 LoRA agents play 10 diverse game environments (matrix games, N-player games, sequential games) and learn via Group Relative Policy Optimization. The diversity of games drives cross-game transfer — strategic reasoning learned in Prisoner's Dilemma transfers to Auction bidding.

2. **Track B (Nash-DPO Self-Play)**: 4 asymmetric agents (accuracy, safety, efficiency, creativity) cross-evaluate each other's outputs on 8 simple games, generating preference pairs that are aggregated via Nash bargaining. Nash-DPO ensures no single objective dominates.

## Dominant Contribution

The key insight is **complementarity**: GRPO improves strategic reasoning benchmarks (ARC, StrategyQA, BBH) while Nash-DPO improves alignment benchmarks (GSM8K, TruthfulQA, MT-Bench), and the two tracks produce orthogonal improvements. This suggests game-theoretic self-play as a **general-purpose mechanism** for LLM post-training.

## Technical Details

### Track A: GRPO Self-Play
- **SFT Warmup**: 500 episodes × 10 games, top 30% filtered, 4 LoRA agents (r=16, α=32)
- **Self-Play**: 3 iterations × 2000 episodes, GRPO with KL=0.05, clip=0.2, γ=0.99
- **Evaluation**: Cross-game transfer (train on 4, eval on 6), game metrics (Nash rate, diversity, payoff)

### Track B: Nash-DPO Self-Play
- **Expert Data**: 500 episodes × 8 games, top 30% filtered
- **Role SFT**: 4 agents with reward-based curriculum (top 50% by agent-specific reward)
- **Nash-DPO**: 2 iterations × 200 games, β=0.1, Nash weights via iterative best-response

### Reward Functions (Robust, Anti-Hacking)
- Correctness: NLI entailment (DeBERTa-v3) + semantic similarity
- Safety: Multi-level pattern detection + severity weighting
- Efficiency: Information density (windowed TTR + coverage + length bell curve)
- Creativity: Sliding-window TTR + sentence variety + structural signals
- Anti-hacking: Length penalty + repetition penalty

### Key Hyperparameters
| Parameter | Track A | Track B |
|-----------|---------|---------|
| Base model | Qwen/Qwen3.5-9B | Qwen/Qwen3.5-9B |
| LoRA r | 16 | 64 |
| Learning rate | 5e-5 (GRPO) / 2e-4 (SFT) | 1e-4 |
| Batch size | 4 | 4 |
| Gradient accumulation | 4 | 4 |
| max_new_tokens | 80 (games) / 256 (responses) | 128 |

## Expected Results

### Must-Demonstrate (Paper Acceptance)
1. GRPO agents improve on at least 2/3 strategic reasoning benchmarks (ARC, StrategyQA, BBH) vs base model
2. Nash-DPO agents improve on at least 2/3 alignment benchmarks (GSM8K, TruthfulQA, MT-Bench) without degrading individual objectives
3. Cross-game transfer is positive (multi-game > single-game)
4. Complementarity: GRPO and Nash-DPO improve different capabilities

### Nice-to-Show
5. Emergent strategic behaviors in game transcripts
6. Nash-DPO weight convergence to meaningful equilibrium
7. Pareto dominance over single-objective training
8. Ablation studies showing each component contributes

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| GRPO reward hacking | HIGH | NLI-based rewards + anti-hacking penalties (IMPLEMENTED) |
| Nash-DPO divergence | MEDIUM | Monitor Elo ratings, Nash weights; early stop if degenerate |
| 9B model too small | MEDIUM | Focus on relative improvements over base model |
| Cross-game transfer negative | LOW | Multi-game curriculum as fallback |
| Benchmark improvements too small | MEDIUM | Report game-theoretic metrics as primary contribution |
