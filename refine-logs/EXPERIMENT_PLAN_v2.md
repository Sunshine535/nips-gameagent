# GameAgent v2: Experiment Plan (Factorial Design)

**Revised Date**: 2026-03-29
**Base Model**: Qwen/Qwen3.5-9B + LoRA
**Budget**: ~120 GPU-hours (reduced from 188 via scope cut)

---

## Scope Reduction (Addressing Reviewer W5)

| Before | After | Rationale |
|--------|-------|-----------|
| 10 games × 6 benchmarks | 4 games × 4 benchmarks | Depth over breadth |
| Track A + B loosely connected | Factorial design with explicit synergy test | Unified paper |
| GSM8K as "alignment" | Replaced with proper metrics | Construct validity |

### Selected Games (Representative Diversity)
1. **Prisoner's Dilemma** — Classic competitive/cooperative tension
2. **Battle of the Sexes** — Coordination under conflict
3. **Public Goods Game** — N-player social dilemma
4. **Ultimatum Game** — Sequential fairness reasoning

### Selected Benchmarks (Construct-Valid)
1. **GTBench** (game-theoretic reasoning) — Primary strategic metric
2. **StrategyQA** (multi-hop strategic reasoning) — Transfer metric
3. **TruthfulQA** (truthfulness/alignment) — Primary alignment metric
4. **MT-Bench** (multi-turn quality) — Holistic quality metric

---

## Core Claims (Revised)

| ID | Claim | Strength | Experiment |
|----|-------|----------|------------|
| C1 | Nash-DPO achieves Pareto-superior multi-objective alignment vs uniform-weight DPO | Strong | Exp 2 |
| C2 | GRPO self-play improves game-theoretic reasoning (Nash rate, payoff) | Strong | Exp 1 |
| C3 | The two tracks produce complementary improvements (factorial interaction) | Key | Exp 3 |
| C4 | Nash bargaining weights converge and correlate with objective difficulty | Supporting | Exp 2 |

---

## Factorial Design (Addressing Reviewer W1 + W4)

### 5-Condition Factorial

| Condition | Track A (GRPO) | Track B (Nash-DPO) | Compute |
|-----------|:-:|:-:|---------|
| Base | ✗ | ✗ | 0 GPU-hr (just eval) |
| SFT-only | SFT warmup | SFT roles | 8 GPU-hr |
| A-only | GRPO self-play | ✗ | 32 GPU-hr |
| B-only | ✗ | Nash-DPO | 24 GPU-hr |
| A+B | GRPO → Nash-DPO | ✓ | 40 GPU-hr |

Each condition × 3 seeds.

### Equal-Compute Controls (Addressing Reviewer W4)

| Baseline | Description | Purpose |
|----------|-------------|---------|
| Base model | Qwen3.5-9B, no fine-tuning | Lower bound |
| SFT-only | SFT on same data volume | Attribute gains to RL, not data |
| Single-agent GRPO | GRPO without multi-agent self-play | Attribute gains to self-play |
| Plain DPO | Standard DPO with equal-weight objectives | Attribute gains to Nash weighting |
| Fixed-weight DPO | DPO with hand-tuned fixed weights | Attribute gains to adaptive Nash |

---

## Experiment 1: GRPO Self-Play (Track A)

### Objective
Show that multi-game GRPO self-play improves game-theoretic reasoning.

### Protocol (32 GPU-hours)
- 4 LoRA agents × 4 games × 3 self-play iterations
- SFT warmup: 200 episodes/game, top 30% filtered
- Self-play: 500 episodes/iteration, GRPO with KL=0.05, clip=0.2
- 3 seeds

### Metrics
- **Nash convergence rate**: fraction of plays reaching Nash equilibrium
- **Average payoff**: normalized per game
- **Strategy diversity**: entropy of action distribution
- **Cross-game transfer**: train on 2 games, eval on 2 held-out

### Success Criteria
- Nash rate improves by ≥15% from iteration 1→3
- Cross-game transfer is positive (held-out > random baseline)

---

## Experiment 2: Formal Nash-DPO (Track B)

### Objective
Show Nash-DPO achieves Pareto-superior multi-objective alignment.

### Protocol (24 GPU-hours)

**Step 1: Disagreement point estimation**
- Evaluate base model on all 4 objectives → d = (d_1, d_2, d_3, d_4)

**Step 2: Nash-DPO training**
- Generate preference pairs via 4-objective evaluation
- Train with FormalNashDPOLoss (β=0.1, EMA τ=0.1)
- 2 iterations × 1000 preference pairs

**Step 3: Comparison**
| Method | Description |
|--------|-------------|
| Nash-DPO | Our method (adaptive weights via Nash bargaining) |
| Equal-DPO | Uniform weights (1/4, 1/4, 1/4, 1/4) |
| Fixed-DPO | Hand-tuned weights (0.4, 0.3, 0.2, 0.1) |
| Single-DPO | Optimize correctness only |

### Metrics (per objective)
- Correctness (NLI entailment + semantic similarity)
- Safety (pattern + context detection)
- Efficiency (information density)
- Creativity (windowed TTR + variety)
- **Pareto dominance count**: # objectives where Nash-DPO ≥ baseline
- **Nash product**: Π(u_k - d_k)

### Success Criteria
- Nash-DPO Pareto-dominates Equal-DPO on ≥3/4 objectives
- Nash product of Nash-DPO > all baselines
- Weights converge (std < 0.05 in last 20% of training)

---

## Experiment 3: Factorial Interaction (A × B)

### Objective
Show complementary improvements via factorial ANOVA.

### Protocol (16 GPU-hours for A+B runs)

All 5 conditions evaluated on:
- 4 game metrics (Nash rate, payoff, diversity, transfer)
- 4 NLP benchmarks (GTBench, StrategyQA, TruthfulQA, MT-Bench)

### Analysis
- 2×2 factorial ANOVA on game metrics and NLP metrics
- Interaction term F-test: significant → complementarity confirmed
- Effect size (η²) for each factor and interaction

### Success Criteria
- Significant interaction term (p < 0.05)
- A+B > max(A-only, B-only) on at least 50% of metrics
- No metric degrades by >2% in A+B vs best single-track

---

## Experiment 4: Ablation Studies

### Protocol (included in Exp 1-3 budget via reduced configs)

| Ablation | What's Removed | Expected Impact |
|----------|---------------|-----------------|
| No SFT warmup | Skip warmup step | Slower convergence |
| Fewer games (2 vs 4) | Reduced game diversity | Lower cross-game transfer |
| Fewer self-play iterations (1 vs 3) | Less training | Lower Nash rate |
| Nash-DPO → Equal-DPO | Remove Nash weighting | Pareto inferiority |
| Single-agent (no self-play) | Remove opponent | Lower strategic reasoning |

---

## Nash-DPO Formal Definition (Addressing Reviewer W3)

### Definition: Multi-Objective Alignment Game (MOAG)
- **Players**: K objective agents (correctness, safety, efficiency, creativity)
- **Strategy space**: θ ∈ Θ (policy parameters)
- **Utility functions**: u_k(θ) = E[R_k(y|x, θ)] for k = 1,...,K
- **Feasible set**: F = {(u_1(θ),...,u_K(θ)) : θ ∈ Θ}
- **Disagreement point**: d = (u_1(θ_base),...,u_K(θ_base))

### Proposition: Nash-DPO Optimality
Under MOAG, the Nash bargaining solution maximizes:
  Π_{k=1}^K (u_k(θ) - d_k)
subject to (u_1,...,u_K) ∈ F and u_k ≥ d_k for all k.

At convergence, the optimal weights satisfy:
  w_k* = 1 / (u_k(θ*) - d_k)  (normalized)

This is equivalent to optimizing the weighted DPO loss with KKT-derived weights.

---

## Compute Budget

| Phase | Runs | GPU-Hours |
|-------|------|-----------|
| Data generation | 4 games × 2 tracks | 12 |
| SFT warmup | 4 agents × 2 tracks | 8 |
| Exp 1: GRPO self-play | 3 seeds × 3 iters | 32 |
| Exp 2: Nash-DPO | 4 methods × 3 seeds | 24 |
| Exp 3: Factorial (A+B) | 3 seeds | 16 |
| Evaluation | All conditions × all benchmarks | 12 |
| Ablations | 5 ablations × 1 seed | 16 |
| **Total** | | **120 GPU-hours** |

---

## Planned Figures

| Figure | Content |
|--------|---------|
| Fig 1 | Framework overview: GRPO self-play + Nash-DPO architecture |
| Fig 2 | Nash convergence curves across 3 self-play iterations |
| Fig 3 | Pareto front: Nash-DPO vs Equal-DPO vs Fixed-DPO |
| Fig 4 | Nash weight trajectories during training |
| Fig 5 | Factorial interaction plot (2×2 ANOVA) |
| Table 1 | Main results: 5 conditions × 8 metrics |
| Table 2 | Per-objective breakdown for Nash-DPO variants |
| Table 3 | Cross-game transfer matrix |
| Table 4 | Ablation results |
