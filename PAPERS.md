# Related Papers: Asymmetric Multi-Agent Self-Play for Output Optimization

## Core References (Must-Cite)

### Multi-Agent Self-Play for LLMs

1. **SPIRAL — Self-Play in Reasoning with Language Models**
   - (ICLR 2026). Zero-sum self-play for improving LLM reasoning
   - Key: Two symmetric agents play a prove/disprove game to improve reasoning quality
   - Relevance: Most direct baseline. We extend from symmetric to asymmetric, zero-sum to mixed-motive

2. **ECON — Economic Reasoning with Bayesian Nash Equilibrium**
   - Studies LLM behavior in strategic economic settings with Bayesian Nash equilibrium
   - Key: LLMs can learn to play Nash equilibria in economic games
   - Relevance: Demonstrates LLMs can engage in game-theoretic reasoning; we use this for training

3. **MNPO — Multiplayer Nash Direct Preference Optimization**
   - Extends DPO to multi-player game settings with Nash equilibrium guarantees
   - Key: Theoretical framework for training multiple models to converge to Nash equilibrium via DPO
   - Relevance: Direct theoretical foundation for our Nash-DPO training algorithm

4. **MAgICoRe — Multi-Agent Iterative Collaborative Refinement**
   - Multiple LLM agents collaboratively refine outputs through iteration
   - Key: Shows iterative multi-agent refinement improves output quality
   - Relevance: Similar multi-agent refinement loop, but agents share same objective → we add heterogeneity

5. **FREE-MAD — Multi-Agent Debate for Factuality**
   - Agents debate to improve factual accuracy of LLM outputs
   - Key: Debate protocol with multiple rounds improves truthfulness
   - Relevance: Single-objective multi-agent system. We generalize to multi-objective

### Game Theory in ML

6. **Generative Adversarial Networks (GANs)**
   - Goodfellow et al. (NeurIPS 2014)
   - Key: Two-player min-max game between generator and discriminator
   - Relevance: Foundational game-theoretic ML framework. Our work extends to N-player cooperative-competitive games

7. **Multi-Agent Reinforcement Learning: A Selective Overview**
   - Zhang et al. (2021). Comprehensive survey of MARL
   - Key: Covers cooperative, competitive, and mixed-motive MARL
   - Relevance: Theoretical foundation for multi-agent training dynamics

8. **Nash Equilibrium in Multi-Player Games**
   - Rosen (1965). "Existence and Uniqueness of Equilibrium Points for Concave N-Person Games"
   - Key: Conditions for existence and uniqueness of Nash equilibria in concave games
   - Relevance: Theoretical basis for our convergence guarantees

### LLM Alignment & Preference Optimization

9. **DPO — Direct Preference Optimization**
   - Rafailov et al. (NeurIPS 2023)
   - Key: Trains language models from pairwise preferences without explicit reward model
   - Relevance: Foundation for our Nash-DPO extension

10. **GRPO — Group Relative Policy Optimization**
    - DeepSeek-AI (2025)
    - Key: RL algorithm using group-level relative ranking
    - Relevance: Single-agent baseline; our multi-agent system should outperform single-agent GRPO

11. **Self-Play Fine-Tuning (SPIN)**
    - Chen et al. (ICML 2024). Self-play fine-tuning with synthetic data
    - Key: Model plays against previous version to generate training data
    - Relevance: Self-play for LLM improvement, but symmetric and single-objective

### Multi-Objective Optimization

12. **Pareto Optimal Solutions in Multi-Objective Optimization**
    - Miettinen (1999). "Nonlinear Multiobjective Optimization"
    - Key: Theory of Pareto optimality and multi-objective solution concepts
    - Relevance: Our outputs should lie on the Pareto frontier of the multi-dimensional quality space

13. **Multi-Objective Reinforcement Learning**
    - Hayes et al. (JMLR 2022). Survey of multi-objective RL
    - Key: Frameworks for optimizing multiple reward signals simultaneously
    - Relevance: Our agents collectively perform multi-objective optimization through game dynamics

14. **Reward Soups: Pareto-Optimal Alignment by Mixing**
    - Rame et al. (NeurIPS 2023)
    - Key: Interpolating models trained on different rewards achieves Pareto-optimal trade-offs
    - Relevance: Alternative approach to multi-objective alignment. We use game dynamics instead of model interpolation

### Multi-Agent LLM Systems

15. **Mixture of Agents (MoA)**
    - Wang et al. (2024). "Mixture-of-Agents Enhances Large Language Model Capabilities"
    - Key: Layered multi-agent system where agents iteratively refine based on others' outputs
    - Relevance: Multi-agent refinement at inference time. We train agents to play the game

16. **LLM Debate**
    - Du et al. (2023). "Improving Factuality and Reasoning via Multi-Agent Debate"
    - Key: Multiple LLMs debate to improve factuality
    - Relevance: Multi-agent interaction for quality improvement, but debate format differs from our game structure

17. **Agents Playing Games**
    - Various 2024-2025 works on LLM agents in strategic game environments
    - Key: LLMs can learn to play complex games through experience
    - Relevance: Demonstrates feasibility of game-theoretic LLM interactions

## Evaluation References

18. **MT-Bench**
    - Zheng et al. (NeurIPS 2023). Multi-turn conversation benchmark with LLM judges
    - Relevance: Primary quality evaluation benchmark

19. **TruthfulQA**
    - Lin et al. (ACL 2022). Benchmark for measuring truthfulness
    - Relevance: Measures accuracy agent effectiveness

20. **SafetyBench**
    - Safety evaluation benchmark for LLMs
    - Relevance: Measures safety agent effectiveness

21. **AlpacaEval 2.0**
    - Dubois et al. (2024). Length-controlled evaluation
    - Relevance: Overall quality assessment

## Papers to Position Against

| Paper | Their Approach | Our Differentiation |
|-------|---------------|---------------------|
| SPIRAL | Symmetric zero-sum self-play | Asymmetric mixed-motive heterogeneous agents |
| MNPO | Nash-DPO theory | First practical application with heterogeneous LLM agents |
| MAgICoRe | Collaborative refinement, same objective | Heterogeneous objectives, game-theoretic convergence |
| FREE-MAD | Debate for single dimension (factuality) | Multi-objective optimization through game dynamics |
| MoA | Inference-time agent mixing | Training-time game-theoretic optimization |
| Reward Soups | Model weight interpolation | Dynamic game-based multi-objective optimization |

## Reading Priority

### Week 1 (Before Design)
- [1] SPIRAL, [3] MNPO, [9] DPO, [4] MAgICoRe

### Week 2 (Before Implementation)
- [6] GANs, [8] Rosen, [12] Pareto, [15] MoA

### Week 3 (Before Experiments)
- [2] ECON, [5] FREE-MAD, [14] Reward Soups

### Ongoing
- Monitor arxiv for multi-agent LLM training and game-theoretic alignment papers
