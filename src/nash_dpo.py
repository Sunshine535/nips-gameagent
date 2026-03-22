"""
Multi-objective Nash-DPO loss implementation.

In standard DPO, a single preference ranking drives the update.
Nash-DPO extends this to multi-agent settings where each agent has
different objectives, and the update seeks a Nash equilibrium where
no agent can unilaterally improve its reward by changing strategy.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class NashDPOLoss(nn.Module):
    """
    Multi-objective Nash-DPO loss.

    Given preference pairs from multiple evaluators (agents with different objectives),
    compute a DPO-style loss that balances all objectives via Nash bargaining.
    """

    def __init__(
        self,
        beta: float = 0.1,
        num_agents: int = 4,
        nash_iterations: int = 3,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.num_agents = num_agents
        self.nash_iterations = nash_iterations
        self.label_smoothing = label_smoothing
        # Learnable per-agent weights for Nash bargaining
        self.agent_weights = nn.Parameter(torch.ones(num_agents) / num_agents)

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        agent_preference_scores: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            policy_chosen_logps: (B,) log-probs of chosen under policy
            policy_rejected_logps: (B,) log-probs of rejected under policy
            reference_chosen_logps: (B,) log-probs of chosen under reference
            reference_rejected_logps: (B,) log-probs of rejected under reference
            agent_preference_scores: (B, num_agents) preference scores from each agent
                                     Positive = agrees with chosen > rejected

        Returns:
            dict with 'loss', 'chosen_rewards', 'rejected_rewards', 'agent_weights'
        """
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        logits = self.beta * (chosen_logratios - rejected_logratios)

        # Standard DPO loss
        dpo_loss = -F.logsigmoid(logits)

        if agent_preference_scores is not None:
            nash_weights = self._compute_nash_weights(agent_preference_scores)
            weighted_loss = (dpo_loss.unsqueeze(1) * nash_weights).sum(dim=1)
            loss = weighted_loss.mean()
        else:
            loss = dpo_loss.mean()
            nash_weights = F.softmax(self.agent_weights, dim=0).detach()

        if self.label_smoothing > 0:
            smooth_loss = -F.logsigmoid(-logits)
            loss = (1 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss.mean()

        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        reward_margin = (chosen_rewards - rejected_rewards).mean()

        return {
            "loss": loss,
            "chosen_rewards": chosen_rewards.mean(),
            "rejected_rewards": rejected_rewards.mean(),
            "reward_margin": reward_margin,
            "agent_weights": nash_weights if agent_preference_scores is not None
                             else F.softmax(self.agent_weights, dim=0).detach(),
        }

    def _compute_nash_weights(self, agent_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute Nash bargaining weights via iterative best-response.

        Each agent's weight reflects how much the current policy should
        attend to that agent's preferences to reach equilibrium.
        """
        B, N = agent_scores.shape
        weights = F.softmax(self.agent_weights, dim=0).unsqueeze(0).expand(B, -1)

        for _ in range(self.nash_iterations):
            # Each agent's "utility" is its preference agreement with current weighting
            utility = (agent_scores * weights).sum(dim=1, keepdim=True)  # (B, 1)

            # Nash bargaining solution: weight ∝ marginal contribution
            marginal = agent_scores - utility  # (B, N)
            # Agents whose preferences diverge most from consensus get more weight
            divergence = marginal.abs()
            new_weights = F.softmax(divergence / (divergence.sum(dim=1, keepdim=True) + 1e-8), dim=1)

            # Smooth update
            weights = 0.5 * weights + 0.5 * new_weights

        return weights


class NashDPOTrainerMixin:
    """Mixin to add Nash-DPO capabilities to TRL's DPOTrainer."""

    def compute_nash_dpo_loss(
        self,
        batch: dict,
        agent_roles: list[str],
    ) -> dict:
        """
        Compute Nash-DPO loss from a batch containing multi-agent preference data.

        Expects batch to have:
          - 'prompt', 'chosen', 'rejected' (standard DPO fields)
          - 'agent_scores_*' for each agent role (preference scores)
        """
        agent_scores = []
        for role in agent_roles:
            key = f"agent_scores_{role}"
            if key in batch:
                agent_scores.append(batch[key])

        if agent_scores:
            agent_preference_scores = torch.stack(agent_scores, dim=1)
        else:
            agent_preference_scores = None

        return agent_preference_scores


def create_nash_dpo_dataset(preference_pairs, agent_roles):
    """
    Convert preference pairs with multi-agent evaluations into a dataset
    suitable for Nash-DPO training.
    """
    records = []
    for pair in preference_pairs:
        record = {
            "prompt": pair.prompt,
            "chosen": pair.chosen,
            "rejected": pair.rejected,
            "margin": pair.margin,
            "chosen_agent": pair.chosen_agent,
            "rejected_agent": pair.rejected_agent,
        }
        records.append(record)
    return records
