"""
Formal Nash-DPO Loss and Nash Bargaining utilities.

All quantities (disagreement point, running losses, Nash weights) operate
in DPO-loss space to avoid the reward-vs-loss unit mismatch.
The disagreement point d_k is the expected per-objective DPO loss under a
random-preference policy, which equals -log(0.5) = ln(2) for each objective.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class NashBargainingProblem:
    """Formal definition of the Multi-Objective Alignment Game (MOAG)."""
    n_objectives: int
    feasible_set: np.ndarray
    disagreement_point: np.ndarray
    objective_names: list[str]


def compute_nash_bargaining_solution(
    utilities: np.ndarray,
    disagreement: np.ndarray,
) -> np.ndarray:
    """Find weights that maximize the Nash product over a discrete feasible set."""
    n_objectives = utilities.shape[1]
    surplus = utilities - disagreement[np.newaxis, :]

    valid = np.all(surplus > 0, axis=1)
    if not valid.any():
        return np.ones(n_objectives) / n_objectives

    surplus_valid = surplus[valid]

    log_product = np.sum(np.log(surplus_valid + 1e-10), axis=1)
    best_idx = np.argmax(log_product)
    optimal_utilities = utilities[valid][best_idx]

    weights = 1.0 / (optimal_utilities - disagreement + 1e-10)
    weights = weights / weights.sum()

    return weights


def compute_kkt_nash_weights(
    current_losses: torch.Tensor,
    disagreement_losses: torch.Tensor,
) -> torch.Tensor:
    """KKT-derived Nash bargaining weights in loss space.

    At the Nash bargaining optimum, the weight for objective k is
    proportional to 1/(d_k - L_k), where d_k is the disagreement loss
    and L_k is the current per-objective loss. Both are in the same
    (DPO loss) space.
    """
    surplus = disagreement_losses - current_losses
    surplus = torch.clamp(surplus, min=1e-8)

    weights = 1.0 / surplus
    weights = weights / weights.sum()

    return weights


class FormalNashDPOLoss(nn.Module):
    def __init__(
        self,
        beta: float = 0.1,
        n_objectives: int = 4,
        objective_names: Optional[list[str]] = None,
        disagreement_mode: str = "base_model",
        weight_update_ema: float = 0.1,
        warmup_steps: int = 100,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.n_objectives = n_objectives
        self.objective_names = objective_names or [f"obj_{i}" for i in range(n_objectives)]
        self.disagreement_mode = disagreement_mode
        self.weight_update_ema = weight_update_ema
        self.warmup_steps = warmup_steps
        self.label_smoothing = label_smoothing

        self.register_buffer(
            "nash_weights",
            torch.ones(n_objectives) / n_objectives,
        )
        self.register_buffer(
            "disagreement_losses",
            torch.ones(n_objectives),
        )
        self.register_buffer(
            "running_losses",
            torch.zeros(n_objectives),
        )
        self.update_count = 0

    def set_disagreement_point(self, losses: torch.Tensor):
        """Set the disagreement losses. Must be in DPO loss space (not utility)."""
        self.disagreement_losses.copy_(losses)

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        per_objective_preferences: torch.Tensor,
    ) -> dict:
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        logits = self.beta * (chosen_logratios - rejected_logratios)

        base_loss = -F.logsigmoid(logits)

        if self.label_smoothing > 0:
            smooth_loss = -F.logsigmoid(-logits)
            base_loss = (1 - self.label_smoothing) * base_loss + self.label_smoothing * smooth_loss

        per_obj_losses = []
        for k in range(self.n_objectives):
            pref_k = per_objective_preferences[:, k]
            agreement_mask = (pref_k > 0).float()
            disagreement_mask = (pref_k < 0).float()
            neutral_mask = (pref_k == 0).float()

            obj_loss = base_loss * agreement_mask + (-F.logsigmoid(-logits)) * disagreement_mask + base_loss * neutral_mask
            per_obj_losses.append(obj_loss.mean())

        per_obj_losses_tensor = torch.stack(per_obj_losses)

        self._update_nash_weights(per_obj_losses_tensor.detach())

        weighted_loss = (self.nash_weights * per_obj_losses_tensor).sum()

        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()

        return {
            "loss": weighted_loss,
            "per_objective_losses": {
                name: loss.item() for name, loss in zip(self.objective_names, per_obj_losses)
            },
            "nash_weights": {
                name: w.item() for name, w in zip(self.objective_names, self.nash_weights)
            },
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
            "nash_product": self._compute_nash_product(per_obj_losses_tensor).item(),
        }

    def _update_nash_weights(self, current_losses: torch.Tensor):
        self.update_count += 1

        tau = self.weight_update_ema
        self.running_losses = (1 - tau) * self.running_losses + tau * current_losses

        if self.update_count < self.warmup_steps:
            return

        new_weights = compute_kkt_nash_weights(
            current_losses=self.running_losses,
            disagreement_losses=self.disagreement_losses,
        )

        self.nash_weights = (1 - tau) * self.nash_weights + tau * new_weights

    def _compute_nash_product(self, losses: torch.Tensor) -> torch.Tensor:
        surplus = self.disagreement_losses - losses
        surplus = torch.clamp(surplus, min=1e-10)
        return torch.prod(surplus)


class MultiObjectiveEvaluator:
    def __init__(self, objective_fns: dict[str, callable]):
        self.objective_fns = objective_fns
        self.n_objectives = len(objective_fns)

    def evaluate(self, response: str, reference: Optional[str] = None, **kwargs) -> dict[str, float]:
        scores = {}
        for name, fn in self.objective_fns.items():
            try:
                if name in ("correctness", "efficiency") and reference:
                    scores[name] = fn(response, reference)
                else:
                    scores[name] = fn(response)
            except Exception as e:
                logger.warning("Objective %s failed: %s", name, e)
                scores[name] = 0.5
        return scores

    def generate_preference(
        self,
        response_a: str,
        response_b: str,
        reference: Optional[str] = None,
        **kwargs,
    ) -> dict:
        scores_a = self.evaluate(response_a, reference, **kwargs)
        scores_b = self.evaluate(response_b, reference, **kwargs)

        preferences = {}
        for name in self.objective_fns:
            diff = scores_a.get(name, 0.5) - scores_b.get(name, 0.5)
            preferences[name] = diff

        a_wins = sum(1 for v in preferences.values() if v > 0)
        b_wins = sum(1 for v in preferences.values() if v < 0)

        if a_wins > b_wins:
            chosen, rejected = response_a, response_b
            chosen_scores, rejected_scores = scores_a, scores_b
        else:
            chosen, rejected = response_b, response_a
            chosen_scores, rejected_scores = scores_b, scores_a

        return {
            "chosen": chosen,
            "rejected": rejected,
            "chosen_scores": chosen_scores,
            "rejected_scores": rejected_scores,
            "per_objective_preferences": {
                name: chosen_scores.get(name, 0.5) - rejected_scores.get(name, 0.5)
                for name in self.objective_fns
            },
        }


def compute_pareto_front(objective_vectors: np.ndarray) -> np.ndarray:
    n = len(objective_vectors)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if np.all(objective_vectors[j] >= objective_vectors[i]) and np.any(objective_vectors[j] > objective_vectors[i]):
                is_pareto[i] = False
                break

    return is_pareto


def nash_social_welfare(utilities: np.ndarray, disagreement: np.ndarray) -> float:
    surplus = utilities - disagreement
    surplus = np.maximum(surplus, 1e-10)
    return np.prod(surplus)


def kalai_smorodinsky_weights(
    ideal_point: np.ndarray,
    disagreement: np.ndarray,
    current_point: np.ndarray,
) -> np.ndarray:
    range_vec = ideal_point - disagreement
    range_vec = np.maximum(range_vec, 1e-10)
    normalized_deficit = (ideal_point - current_point) / range_vec
    weights = normalized_deficit / (normalized_deficit.sum() + 1e-10)
    return weights
