# Implementation Plan: Asymmetric Multi-Agent Self-Play for Output Optimization

## Timeline Overview

| Phase | Duration | Dates | Goal |
|-------|----------|-------|------|
| Phase 0 | 1 week | May 19–25 | Agent infrastructure & reward functions |
| Phase 1 | 2 weeks | May 26–Jun 8 | Individual agent training + game protocol |
| Phase 2 | 2 weeks | Jun 9–22 | Nash-DPO training + game data generation |
| Phase 3 | 1 week | Jun 23–29 | Ablations + analysis |
| Phase 4 | 1 week | Jun 30–Jul 6 | Paper writing |
| Buffer | 1 week | Jul 7–13 | Revisions |
| **Deadline** | | **Jul 17** | **NeurIPS 2026 submission** |

## Phase 0: Infrastructure (Week 1)

### 0.1 Agent Architecture

- [ ] Implement base agent class with shared Qwen3.5-9B backbone + LoRA adapter
  ```python
  class GameAgent:
      def __init__(self, role: str, base_model: str, lora_rank: int = 64):
          self.base = load_model(base_model)
          self.lora = LoRA(rank=lora_rank)
          self.role = role  # accuracy | safety | efficiency | creativity
          self.reward_fn = load_reward(role)
  ```
- [ ] Implement 4 reward functions:
  - Accuracy: Fact-checking via NLI model + coherence via perplexity
  - Safety: Llama-Guard score + toxicity classifier + bias detector
  - Efficiency: Information coverage (NLI entailment of key facts) / response length
  - Creativity: Embedding diversity score + engagement classifier

### 0.2 Game Protocol

- [ ] Implement round-robin game engine
  ```python
  class GameEngine:
      def play_round(self, prompt, current_output, agent):
          revision_prompt = self.format_revision_prompt(prompt, current_output, agent.role)
          revised = agent.generate(revision_prompt)
          rewards = {a.role: a.reward_fn(revised) for a in self.agents}
          return revised, rewards

      def play_game(self, prompt, num_rounds=6):
          output = self.agents[0].generate(prompt)  # accuracy starts
          history = [output]
          for round in range(num_rounds):
              agent = self.agents[round % len(self.agents)]
              output, rewards = self.play_round(prompt, output, agent)
              history.append((output, rewards))
          return self.nash_arbitrate(history)
  ```

### 0.3 Nash Arbitration

- [ ] Implement Nash bargaining solution
  ```python
  def nash_arbitrate(self, history):
      baseline_rewards = history[0][1]  # rewards of initial output
      best_nash_product = -inf
      best_output = history[0][0]
      for output, rewards in history[1:]:
          improvements = [rewards[r] - baseline_rewards[r] for r in rewards]
          if all(imp > -epsilon for imp in improvements):
              nash_product = prod(max(imp, 0) for imp in improvements)
              if nash_product > best_nash_product:
                  best_nash_product = nash_product
                  best_output = output
      return best_output
  ```

### Deliverables
- Working agent architecture with 4 reward functions
- Game engine with round-robin play and Nash arbitration
- Verified reward functions on test prompts

## Phase 1: Individual Agent Training + Game Data (Weeks 2–3)

### 1.1 Individual Agent DPO Training

Train each agent independently via DPO on its dimension-specific preferences:

| Agent | Training Data | Preference Source |
|-------|---------------|-------------------|
| Accuracy | UltraFeedback (accuracy split) | NLI fact-check score |
| Safety | SafeRLHF + BeaverTails | Safety classifier |
| Efficiency | UltraFeedback (conciseness split) | Information density score |
| Creativity | WildChat + creative writing | Diversity + engagement score |

Per agent:
- LoRA rank 64, α=128
- Learning rate: 5e-5
- Batch size: 32 (8 per GPU × 4 gradient accumulation)
- Training: 3 epochs
- Time: ~8 hours per agent on 8× A100

### 1.2 Game Data Generation

After individual training, generate game transcripts:
- Sample 10,000 prompts from UltraFeedback + MT-Bench + WildChat
- For each prompt, play a 6-round game with trained agents
- Record all intermediate outputs and rewards
- This produces training data for Nash-DPO

### 1.3 Baseline Evaluation

Run all baselines on evaluation set:
- **Single-agent GRPO**: Standard GRPO training on Qwen3.5-9B
- **Symmetric self-play (SPIRAL-style)**: Two identical agents, same objective
- **Best-of-N**: Generate N=16 responses, select best by composite reward
- **Mixture of Agents**: Standard MoA inference pipeline

### Deliverables
- 4 trained agent LoRA adapters
- 10,000 game transcripts with reward traces
- Baseline evaluation results

## Phase 2: Nash-DPO Training (Weeks 4–5)

### 2.1 Nash-DPO Implementation

```python
class NashDPOTrainer:
    def __init__(self, agents, beta=0.1):
        self.agents = agents
        self.beta = beta

    def compute_nash_preferences(self, game_data):
        """Convert game transcripts into per-agent DPO preferences."""
        preferences = {a.role: [] for a in self.agents}
        for game in game_data:
            nash_output = game.nash_selected
            for agent in self.agents:
                # Preferred: Nash-selected output
                # Rejected: agent's individual best (may not be Nash)
                agent_best = game.best_for(agent.role)
                if nash_output != agent_best:
                    preferences[agent.role].append(
                        (nash_output, agent_best)  # (chosen, rejected)
                    )
        return preferences

    def train_iteration(self, game_data):
        preferences = self.compute_nash_preferences(game_data)
        for agent in self.agents:
            agent.dpo_update(preferences[agent.role], self.beta)
```

### 2.2 Alternating Best Response Loop

```
For iteration t = 1, ..., 5:
  1. Play 5,000 games with current agents → game data
  2. Compute Nash preferences for each agent
  3. Update each agent via DPO (1 epoch on Nash preferences)
  4. Evaluate on validation set (500 prompts)
  5. Check convergence: is agent behavior stable?
```

### 2.3 Convergence Monitoring

Track per iteration:
- Nash product of game outputs (should increase)
- Per-agent reward on Nash-selected outputs (should stabilize)
- Agent agreement rate (how often agents' preferred outputs align)
- Policy divergence (KL from base model, should stay bounded)

### Deliverables
- Nash-DPO trained agents (5 iterations)
- Convergence plots showing Nash equilibrium approach
- Comparison: Nash-DPO vs. independent DPO agents

## Phase 3: Ablations & Analysis (Week 6)

### 3.1 Ablation Studies

| Ablation | Configurations | Runs |
|----------|---------------|------|
| Number of agents | 2, 3, 4 (all combinations) | 7 |
| Game rounds | 2, 4, 6, 8, 12 | 5 |
| LoRA rank | 16, 32, 64, 128 | 4 |
| Nash-DPO iterations | 1, 3, 5, 10 | 4 |
| Nash vs. majority vote arbitration | 2 methods | 2 |
| **Total** | | **22 ablation runs** |

### 3.2 Pareto Front Analysis

- Plot 4D Pareto fronts (accuracy × safety × efficiency × creativity)
- Compare Pareto fronts: GameRefine vs. each baseline
- Compute hypervolume indicator for quantitative Pareto comparison
- Identify which agent combinations give the best Pareto front

### 3.3 Emergent Behavior Analysis

Analyze game transcripts for emergent patterns:
- **Strategic yielding**: Does safety agent learn to yield on minor safety concerns to preserve accuracy?
- **Cooperative specialization**: Do agents develop complementary revision strategies?
- **Turn-order effects**: Does the order of agent play matter?
- **Communication patterns**: How do revision prompts evolve over Nash-DPO training?

### 3.4 Evaluation on Held-Out Benchmarks

| Benchmark | Dimension Tested | Metric |
|-----------|-----------------|--------|
| MT-Bench | Overall quality | Score (1-10) |
| AlpacaEval 2.0 | Helpfulness | Win rate |
| TruthfulQA | Factual accuracy | MC accuracy |
| SafetyBench | Safety | Safety score |
| BIG-Bench Hard | Reasoning | Accuracy |
| HumanEval | Code generation | Pass@1 |

### Deliverables
- Complete ablation results
- Pareto front visualizations
- Emergent behavior analysis report
- Full benchmark results

## Phase 4: Paper Writing (Week 7)

### Paper Structure (10 pages + appendix)

1. **Introduction** (1.5 pages): Motivation for heterogeneous multi-agent optimization
2. **Related Work** (1 page): Self-play, multi-agent LLM, game-theoretic ML
3. **GameRefine Framework** (2.5 pages): Agent design, game protocol, Nash arbitration
4. **Nash-DPO Training** (1.5 pages): Algorithm, convergence analysis
5. **Experiments** (3 pages): Main results, Pareto analysis, ablations, emergent behavior
6. **Conclusion** (0.5 pages)
7. **Appendix**: Full agent prompts, reward function details, additional ablations

### Key Figures

1. GameRefine framework diagram (game protocol visualization)
2. Pareto front comparison (4D projected to 2D pairs)
3. Nash-DPO convergence plots (per-iteration Nash product and agent rewards)
4. Ablation results (game rounds, agent count)
5. Game transcript examples (showing emergent strategic behavior)

## Compute Budget

| Component | GPU-hours |
|-----------|-----------|
| Individual agent training (4 agents × 64 GPU-hrs) | 256 |
| Game data generation (inference) | 200 |
| Nash-DPO training (5 iterations × 48 GPU-hrs) | 240 |
| Baselines (4 × 32 GPU-hrs) | 128 |
| Ablations (22 × ~24 GPU-hrs) | 528 |
| Evaluation (inference) | 100 |
| **Total** | **~1,452 GPU-hrs** |

## Risk Mitigation Checkpoints

| Checkpoint | Date | Decision |
|------------|------|----------|
| After Phase 0 | May 25 | Do reward functions produce meaningful scores? |
| After Phase 1 week 1 | Jun 1 | Do individual agents improve their dimension? |
| After Phase 1 | Jun 8 | Do game outputs beat individual agents? If no → revise game protocol |
| After Phase 2 week 1 | Jun 15 | Does Nash-DPO improve over independent DPO? If no → try larger beta/more iterations |
| After Phase 3 | Jun 29 | Is the Pareto front story compelling? If weak → focus on emergent behavior analysis |
