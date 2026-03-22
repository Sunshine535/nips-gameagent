# GameAgent: Game-Theoretic Self-Play Improves Both Strategic Reasoning and Multi-Objective Alignment in LLMs

## Abstract

We present GameAgent, a unified framework that leverages game-theoretic self-play to simultaneously enhance strategic decision-making capabilities and multi-objective alignment in large language models. Our approach introduces two complementary training paradigms within a shared game-theoretic foundation: (1) GRPO Self-Play, where agents learn strategic reasoning through diverse matrix games, public goods dilemmas, and negotiation scenarios, demonstrating emergent cross-game transfer; and (2) Nash-DPO Self-Play, where asymmetric agents with competing objectives (correctness, safety, efficiency, creativity) converge to Pareto-optimal policies via Nash bargaining. Experiments on Qwen3.5-9B show that GRPO-trained agents achieve significant gains on strategic reasoning benchmarks (ARC-Challenge, StrategyQA, BBH) while Nash-DPO agents improve alignment metrics (GSM8K, TruthfulQA, MT-Bench) without sacrificing individual objective quality. The cross-comparison reveals that the two paradigms produce complementary improvements, suggesting game-theoretic self-play as a general-purpose mechanism for LLM post-training.

## 1. Introduction

Post-training alignment of LLMs has predominantly relied on RLHF or DPO with human preference data. However, these approaches face two fundamental limitations: (1) they treat alignment as a single-objective problem, conflating correctness, safety, helpfulness, and other desiderata; (2) they do not explicitly train for strategic reasoning—the ability to make optimal decisions in complex, multi-agent environments.

Game theory provides a natural framework to address both limitations simultaneously. In strategic games, agents must reason about opponents' strategies, anticipate consequences of actions, and balance multiple competing objectives. We hypothesize that training LLMs through diverse game-theoretic self-play:

- **Enhances strategic reasoning** via exposure to iterated game dynamics, Nash equilibrium convergence, and cross-game transfer
- **Improves multi-objective alignment** via Nash-DPO, where multiple agents with different objective functions (correctness, safety, efficiency, creativity) negotiate policy updates through Nash bargaining

## 2. Method

### 2.1 Game-Theoretic Self-Play Framework

We design 10 game environments spanning three categories:
- **Matrix games** (6): Prisoner's Dilemma, Coordination, Battle of Sexes, Stag Hunt, Chicken, Matching Pennies
- **N-player games** (2): Public Goods, Sealed-Bid Auction
- **Sequential games** (2): Ultimatum, Multi-Issue Negotiation

### 2.2 Track A: GRPO Self-Play for Strategic Reasoning

Four LoRA agents are initialized via SFT on filtered game trajectories, then trained through iterative GRPO self-play:
1. **SFT Warmup**: Each agent specializes in 2-3 games (agent_0: PD + Coordination, agent_1: BoS + Stag Hunt, agent_2: Public Goods + Auction, agent_3: Ultimatum + Negotiation)
2. **Self-Play**: Agents play all 10 games against each other, collecting trajectories with rewards
3. **GRPO Update**: Group-relative advantages drive clipped policy gradient updates with KL regularization
4. **Evaluation**: Cross-game transfer measures generalization to held-out games

### 2.3 Track B: Nash-DPO for Multi-Objective Alignment

Four role-specialized agents optimize for different objectives:
- **AccuracyAgent**: Maximizes factual correctness
- **SafetyAgent**: Minimizes harmful/unsafe content
- **EfficiencyAgent**: Optimizes conciseness
- **CreativityAgent**: Maximizes diversity and novelty

Training follows a cross-evaluation protocol:
1. **Generate**: Each agent produces candidate responses
2. **Cross-Evaluate**: Each agent scores all candidates using its reward function
3. **Aggregate**: Preference pairs are collected from cross-evaluations
4. **Nash-DPO Update**: A DPO loss weighted by Nash bargaining weights ensures no single objective dominates

### 2.4 Nash-DPO Loss

Standard DPO optimizes: L_DPO = -log σ(β(log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))

Nash-DPO extends this with agent-specific weights computed via iterative best-response:
- Each agent's weight reflects its marginal contribution to the consensus
- Agents whose preferences diverge most from the current policy receive higher weights
- This converges to a Nash bargaining solution where all agents' utilities are jointly maximized

## 3. Experiments

### 3.1 Strategic Decision-Making (Track A)
- **Benchmarks**: ARC-Challenge, StrategyQA, BIG-Bench Hard
- **Game metrics**: Average payoff, Nash equilibrium rate, strategy diversity (Shannon entropy), Nash distance
- **Cross-game transfer**: Train on 4 games → evaluate on 6 held-out games (single-game vs. multi-game vs. curriculum)

### 3.2 Multi-Objective Alignment (Track B)
- **Benchmarks**: GSM8K, TruthfulQA, MT-Bench
- **Alignment metrics**: Per-objective scores (correctness, safety, efficiency, creativity)
- **Elo ratings**: Track agent improvement across Nash-DPO iterations
- **Pareto analysis**: Show Nash-DPO achieves Pareto dominance over single-objective training

### 3.3 Cross-Comparison (Key Contribution)
- GRPO vs. Nash-DPO on ALL benchmarks and ALL games
- Multi-objective evaluation of GRPO agents (do they improve alignment?)
- Game performance of Nash-DPO agents (do they improve strategic reasoning?)
- Complementarity analysis: combined vs. individual improvements

### 3.4 Ablation Studies
- No SFT warmup (GRPO from scratch)
- Fewer GRPO iterations (2 vs. 5)
- Fewer Nash-DPO iterations (1 vs. 3)
- Single-track vs. dual-track training

## 4. Expected Contributions

1. **Unified game-theoretic framework** that simultaneously improves strategic reasoning and multi-objective alignment
2. **Nash-DPO**: A novel multi-objective DPO variant using Nash bargaining for balanced optimization
3. **Cross-game transfer**: First systematic study of how game diversity drives emergent strategic reasoning in LLMs
4. **Complementarity analysis**: Evidence that GRPO and Nash-DPO produce orthogonal improvements, suggesting a general principle for LLM post-training

## 5. Related Work

- RLHF/DPO alignment (Ouyang et al., 2022; Rafailov et al., 2023)
- Multi-objective RLHF (Rame et al., 2023; Zhou et al., 2024)
- Self-play for LLMs (Chen et al., 2024; Wu et al., 2024)
- Game-theoretic approaches to AI alignment (Conitzer et al., 2024)
- Nash learning dynamics (Gemp & Mahadevan, 2024)
