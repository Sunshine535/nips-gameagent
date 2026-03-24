# Auto Review Log: GameAgent

## Round 0 — Pre-Experiment External Review (2026-03-24)

### Assessment (Summary)
- Score: 4/10
- Verdict: NOT READY
- Key criticisms:
  1. Track A and B appear as two separate projects glued together — no unified factorial design
  2. Evaluation doesn't measure stated constructs (GSM8K ≠ alignment, ARC ≠ strategic reasoning)
  3. Nash-DPO formulation is conceptually muddy — needs rigorous formal definition
  4. Missing baselines for causal attribution (SFT-only, single-agent GRPO, plain DPO, fixed-weight DPO)
  5. Scope too wide — 10 games + 6 benchmarks + ablations is too shallow

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Overall**
**Score: 4/10**

There is a plausible seed of a good paper here, but the current proposal is not at NeurIPS level. It reads like two different projects glued together with game-theory terminology, and the claim, method, and evaluation do not line up tightly enough.

**Top 3 Strengths**
1. The core ambition is timely and interesting: using interaction/self-play to induce capabilities that static instruction tuning may miss.
2. The proposal at least tries to measure both in-game behavior and downstream transfer, which is better than benchmark-only optimization.
3. The asymmetric-agent setup in Track B could surface real objective tradeoffs and produce useful analysis, even if the grand unified claim fails.

**Top 5 Weaknesses**

1. **This is not yet a unified paper.**
Why it matters: Track A and Track B are basically separate training pipelines, separate objectives, separate datasets, and separate benchmark suites. "Complementarity" and "orthogonality" are asserted, not operationalized.
**Minimum fix required:** Add a true joint experiment with a factorial design: base vs Track A only vs Track B only vs A+B under equal compute, with an explicit additivity/synergy hypothesis. If you cannot do that cleanly, split the paper and make one claim.

2. **The evaluation does not measure the stated constructs.**
Why it matters: GSM8K is not an alignment benchmark. ARC-Challenge, StrategyQA, and BBH are not clean measures of strategic reasoning induced by game play. MT-Bench is weak and judge-sensitive.
**Minimum fix required:** Redefine the constructs and rebuild the benchmark suite. Use interactive/planning/strategic benchmarks for "strategic reasoning," and actual alignment/safety/truthfulness/helpfulness evaluations for "multi-objective alignment," ideally with external judges or human labels.

3. **The Nash-DPO formulation is conceptually muddy.**
Why it matters: "Nash bargaining weights computed via iterative best-response" mixes different game-theoretic ideas without a clear formal object. What is the game, utility, feasible set, disagreement point, and solution concept?
**Minimum fix required:** Write the exact optimization formally. If you cannot define it rigorously, drop the "Nash" framing and present it honestly as heuristic weighted multi-objective DPO.

4. **The proposal is missing the baselines needed for causal attribution.**
Why it matters: If you see gains, you will not know whether they come from self-play, extra synthetic data, role conditioning, more optimization steps, or the game-theoretic component.
**Minimum fix required:** Add equal-compute controls: SFT-only, plain GRPO without multi-agent self-play, plain DPO, fixed-weight multi-objective DPO, single-agent critique, random-role critique, and non-game self-training baselines.

5. **The scope is too wide for a credible first paper.**
Why it matters: Two pipelines, many games, many agents, transfer claims, multi-benchmark evaluation, and ablations on a 9B model is a recipe for shallow experiments and noisy conclusions.
**Minimum fix required:** Cut aggressively. Pick 2-3 representative games per track, one primary claim per track, 1-2 primary downstream metrics, and run enough seeds/statistics to make the conclusions believable.

**Verdict**
**NOT READY**

Brutally honest version: this is currently a promising idea cluster, not a submission-ready paper plan. The biggest problem is not lack of experiments; it is lack of a clean scientific story. Tighten the thesis, fix the benchmark mismatch, formalize or de-jargon the Nash piece, and reduce scope before spending serious compute.

</details>

### Action Plan (Addressing While Experiments Run)

| # | Weakness | Fix | Status |
|---|----------|-----|--------|
| W1 | Not a unified paper | Add factorial design: base / A-only / B-only / A+B | 🔧 IMPLEMENTING |
| W2 | Evaluation mismatch | Reframe constructs; add game-theoretic primary metrics | 🔧 IMPLEMENTING |
| W3 | Nash-DPO muddy | Write formal optimization (game, utilities, feasible set) | 🔧 IMPLEMENTING |
| W4 | Missing baselines | Add SFT-only, single-agent GRPO, plain DPO baselines | 🔧 IMPLEMENTING |
| W5 | Scope too wide | Reframe: game metrics as primary, downstream as secondary | 📝 PLANNING |
