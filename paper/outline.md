# GameAgent: Paper Outline (NeurIPS 2026)

## Title
GameAgent: Game-Theoretic Self-Play Improves Both Strategic Reasoning and Multi-Objective Alignment in LLMs

## Abstract (~250 words)
- Problem: LLM post-training treats alignment as single-objective and doesn't train strategic reasoning
- Method: unified game-theoretic framework — GRPO self-play for strategic reasoning + Nash-DPO for multi-objective alignment
- Key results: TBD (fill after experiments)
- Significance: factorial design shows complementarity — game-theoretic self-play as general-purpose post-training mechanism

## 1. Introduction (1.5 pages)
- LLM alignment as single-objective → multi-objective gap
- Strategic reasoning deficit in current LLMs
- Game theory unifies both: competitive/cooperative dynamics for reasoning, Nash bargaining for multi-objective
- Contributions:
  1. **Unified framework**: Game-theoretic self-play simultaneously improves strategic reasoning and multi-objective alignment
  2. **Nash-DPO**: Formally grounded multi-objective DPO via Nash bargaining (Def. 1, Thm. 1)
  3. **Cross-game transfer**: First systematic study showing game diversity drives emergent strategic reasoning
  4. **Factorial analysis**: A-only, B-only, and A+B configurations reveal complementary improvements

## 2. Related Work (1 page)
- Self-play for LLMs: SPIRAL, SPIN, SPAG (2024)
- Multi-objective alignment: Reward Soups, MORLHF, Pareto RLHF
- Game theory in ML: GANs, MARL, Nash learning dynamics
- Nash-based policy optimization: MNPO, Nash-MD, MAgICoRe

## 3. Method (2.5 pages)
### 3.1 Game-Theoretic Self-Play Framework
- 10 game environments spanning 3 categories (matrix, N-player, sequential)
- Prompt design and action parsing
- Why game diversity matters (information-theoretic argument)

### 3.2 Track A: GRPO Self-Play for Strategic Reasoning
- SFT warmup: 4 LoRA agents × 2 games each (agent-game assignment)
- Self-play: agents play all 10 games against each other
- GRPO update: group-relative advantages with KL regularization
- Cross-game transfer protocol: train on K, evaluate zero-shot on 10−K

### 3.3 Track B: Nash-DPO for Multi-Objective Alignment

#### 3.3.1 Formal Definition

**Definition 1 (Multi-Objective Alignment Game).** We define a cooperative game G = (N, U, D, π_ref) where:
- N = {1,...,n} is the set of n objective agents (correctness, safety, efficiency, creativity)
- U = {u_i : Π → ℝ}_{i∈N} are agent utility functions, where Π is the policy space
- D = (u_1(π_ref),...,u_n(π_ref)) is the disagreement point (utilities under reference policy)
- π_ref is the reference policy (SFT model)

**Definition 2 (Nash Bargaining Solution).** The Nash-DPO objective seeks π* that maximizes:

  π* = argmax_π ∏_{i∈N} (u_i(π) − u_i(π_ref))

subject to u_i(π) ≥ u_i(π_ref) ∀i (individual rationality).

**Proposition 1.** Taking the log and using the DPO reparameterization, Nash-DPO reduces to:

  L_Nash-DPO(π) = − Σ_{i∈N} w_i · log σ(β(log π(y_w|x)/π_ref(y_w|x) − log π(y_l|x)/π_ref(y_l|x)))

where w_i are Nash bargaining weights computed via:
  w_i ∝ 1 / (u_i(π^(t)) − u_i(π_ref) + ε)
  (agents furthest from their ideal get more weight)

**Algorithm 1**: Iterative Nash-DPO
1. Initialize π^(0) = π_ref
2. For t = 1,...,T:
   a. Each agent generates candidates and cross-evaluates
   b. Compute marginal utilities: δ_i = u_i(π^(t-1)) − u_i(π_ref)
   c. Compute Nash weights: w_i^(t) ∝ 1/(δ_i + ε)
   d. Aggregate preference pairs with weights w_i^(t)
   e. DPO update: π^(t) ← argmin L_Nash-DPO(π^(t-1))

#### 3.3.2 Reward Functions (Robust, Anti-Hacking)
- Correctness: NLI entailment (DeBERTa-v3-large) + cosine similarity
- Safety: Multi-level regex + severity weighting
- Efficiency: Information density = coverage × (1 − length_penalty)
- Creativity: Windowed TTR + sentence variety
- Anti-hacking: length penalty + n-gram repetition penalty

### 3.4 Combined A+B Training (Factorial Design)
- Sequential composition: GRPO → Nash-DPO → evaluation
- Hypothesis: A and B produce orthogonal improvements (GRPO on reasoning, Nash-DPO on alignment)
- Formal test: interaction effect in factorial ANOVA

## 4. Experiments (3 pages)

### 4.1 Setup
- Base model: Qwen3.5-9B
- Hardware: 4× NVIDIA H800 80GB
- Hyperparameters table (Table 1)
- Statistical methodology: 3 seeds per condition, report mean ± std

### 4.2 Primary Evaluation: Game-Theoretic Metrics
- **Table 2**: Game performance across 8 games (from game_environments_simple) for all conditions
  - Metrics: Nash equilibrium rate, avg payoff, strategy diversity (Shannon entropy), Nash distance (L2)
  - Conditions: Base, SFT-only, A-only (GRPO), B-only (Nash-DPO), A+B
  - SFT/A conditions: report mean across all 4 agents
- **Figure 1**: Training trajectory (payoff + diversity over GRPO iterations)
- **Figure 2**: Cross-game transfer heatmap (train on K games → eval on held-out games)

### 4.3 Downstream Transfer: Standard Benchmarks
- **Table 3**: Factorial results on construct-valid benchmarks
  | | Base | SFT-only | A-only | B-only | A+B |
  |---|---|---|---|---|---|
  | StrategyQA | | | | | |
  | TruthfulQA | | | | | |
  | MT-Bench | | | | | |
- Key analysis: which benchmarks improve under which conditions?
- Note: ARC/GSM8K removed (not construct-valid for strategic reasoning / alignment)

### 4.4 Multi-Objective Analysis (Track B)
- **Figure 3**: Elo convergence across Nash-DPO iterations
- **Figure 4**: Pareto front (2D projections of 4-objective space)
- **Table 4**: Nash-DPO vs plain equal-weight DPO (ablating the Nash bargaining mechanism)

### 4.5 Ablation Studies
- **Table 5**: Component ablations
  | Ablation | Description | Expected Impact |
  |----------|-------------|-----------------|
  | No SFT warmup | GRPO from base model | Lower initial performance |
  | Fewer GRPO iters (2 vs 3) | Reduced self-play | Smaller gains |
  | Fewer Nash-DPO iters (1 vs 2) | Reduced negotiation | Less balanced |
  | Equal DPO weights | Remove Nash bargaining | One objective dominates |

## 5. Analysis (0.5 pages)
- Emergent strategic behaviors in game transcripts (qualitative examples)
- Nash-DPO weight dynamics across iterations (convergence to equilibrium)
- Complementarity decomposition: interaction effect in 2×2 factorial

## 6. Conclusion (0.5 pages)
- Summary: game-theoretic self-play is a viable general mechanism for LLM post-training
- Limitations: single model scale (9B), proxy rewards, English-only games
- Future: scale to 70B, real safety classifiers, more complex games, multi-language

## Appendix
- A: Full game descriptions and payoff matrices
- B: Prompt templates for all 10 games
- C: Reward function implementation details
- D: Nash-DPO convergence proof sketch
- E: Additional ablation results (LoRA rank, game count)
- F: Game transcript examples (3 per game type)
