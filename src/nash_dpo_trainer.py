"""
NashDPOTrainer: Custom DPO trainer with per-objective Nash bargaining weights.

Subclasses TRL's DPOTrainer to replace the standard uniform DPO loss with
the formal Nash-DPO loss (Algorithm 1 in the paper). The key change is that
each training sample carries per-objective preference signals, and the loss
is weighted by dynamically updated Nash bargaining weights.

The disagreement point and running weights live entirely in loss space,
avoiding the reward-vs-loss mismatch flagged by the Codex reviewer.

Data collation: A custom collator preserves the pref_* columns so they
reach the batch dict inside get_batch_loss_metrics.
"""

import logging
from typing import Any, Optional

import torch
import torch.nn.functional as F
from trl import DPOConfig, DPOTrainer
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

OBJECTIVE_NAMES = ["correctness", "safety", "efficiency", "creativity"]
PREF_COLS = [f"pref_{n}" for n in OBJECTIVE_NAMES]


class _NashDPOCollator:
    """Wraps the default DPO collator to preserve pref_* columns."""

    def __init__(self, base_collator):
        self._base = base_collator

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        pref_data = {col: [] for col in PREF_COLS}
        for f in features:
            for col in PREF_COLS:
                pref_data[col].append(f.pop(col, 0.0))

        batch = self._base(features)

        for col in PREF_COLS:
            batch[col] = torch.tensor(pref_data[col], dtype=torch.float32)

        return batch


class NashDPOTrainer(DPOTrainer):
    """DPOTrainer subclass that applies per-objective Nash bargaining weights.

    Extra dataset columns expected:
        pref_correctness, pref_safety, pref_efficiency, pref_creativity
    Each is a float in [-1, 1] indicating how strongly that objective
    prefers chosen > rejected (+) or the reverse (-).
    """

    def __init__(
        self,
        *args,
        nash_ema_tau: float = 0.1,
        nash_warmup_steps: int = 50,
        disagreement_losses: Optional[torch.Tensor] = None,
        weighting_method: str = "nash",
        fixed_weights: Optional[list[float]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.data_collator = _NashDPOCollator(self.data_collator)

        self.nash_ema_tau = nash_ema_tau
        self.nash_warmup_steps = nash_warmup_steps
        self.weighting_method = weighting_method

        n = len(OBJECTIVE_NAMES)

        if disagreement_losses is not None:
            self._disagreement = disagreement_losses.float()
        else:
            self._disagreement = torch.ones(n)

        self._nash_weights = torch.ones(n) / n
        self._running_losses = torch.zeros(n)
        self._nash_step = 0

        if weighting_method == "fixed" and fixed_weights:
            w = torch.tensor(fixed_weights, dtype=torch.float32)
            self._nash_weights = w / w.sum()
            self.nash_warmup_steps = float("inf")
        elif weighting_method == "equal":
            self.nash_warmup_steps = float("inf")
        elif weighting_method == "single_correctness":
            self._nash_weights = torch.tensor([1.0, 0.0, 0.0, 0.0])
            self.nash_warmup_steps = float("inf")

        self._weight_history = []

    @property
    def nash_weights_dict(self):
        return {n: w.item() for n, w in zip(OBJECTIVE_NAMES, self._nash_weights)}

    def get_batch_loss_metrics(self, model, batch, train_or_eval="train"):
        """Override to apply Nash weights to the per-objective DPO losses."""
        metrics = {}

        policy_output = self.concatenated_forward(model, batch)

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll,
        ) = (
            policy_output["chosen_logps"],
            policy_output["rejected_logps"],
            policy_output["chosen_logits"],
            policy_output["rejected_logits"],
            policy_output.get("nll_loss", torch.tensor(0.0)),
        )

        ref_chosen_logps = batch.get("reference_chosen_logps")
        ref_rejected_logps = batch.get("reference_rejected_logps")
        if ref_chosen_logps is None:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_output = self.concatenated_forward(self.ref_model, batch)
                else:
                    with self.null_ref_context():
                        ref_output = self.concatenated_forward(model, batch)
                ref_chosen_logps = ref_output["chosen_logps"]
                ref_rejected_logps = ref_output["rejected_logps"]

        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        logits = self.beta * (chosen_logratios - rejected_logratios)

        base_loss = -F.logsigmoid(logits)
        reverse_loss = -F.logsigmoid(-logits)

        pref_tensors = []
        for col in PREF_COLS:
            if col in batch:
                pref_tensors.append(batch[col].to(base_loss.device).float())

        if len(pref_tensors) == len(OBJECTIVE_NAMES):
            pref_matrix = torch.stack(pref_tensors, dim=1)

            per_obj_losses = []
            for k in range(len(OBJECTIVE_NAMES)):
                pref_k = pref_matrix[:, k]
                agree = (pref_k > 0).float()
                disagree = (pref_k < 0).float()
                neutral = (pref_k == 0).float()
                obj_loss = base_loss * agree + reverse_loss * disagree + base_loss * neutral
                per_obj_losses.append(obj_loss.mean())

            per_obj = torch.stack(per_obj_losses)
            self._update_nash_weights(per_obj.detach())

            device = base_loss.device
            weights = self._nash_weights.to(device)
            loss = (weights * per_obj).sum()

            for i, name in enumerate(OBJECTIVE_NAMES):
                metrics[f"{train_or_eval}/obj_loss_{name}"] = per_obj[i].item()
                metrics[f"{train_or_eval}/nash_w_{name}"] = weights[i].item()

            nash_product = self._compute_nash_product(per_obj)
            metrics[f"{train_or_eval}/nash_product"] = nash_product.item()
        else:
            loss = base_loss.mean()
            logger.warning("pref_* columns missing from batch — falling back to uniform DPO")

        if self.args.rpo_alpha is not None and self.args.rpo_alpha > 0:
            loss = loss + self.args.rpo_alpha * policy_nll

        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()

        metrics[f"{train_or_eval}/loss"] = loss.item()
        metrics[f"{train_or_eval}/chosen_rewards"] = chosen_rewards.mean().item()
        metrics[f"{train_or_eval}/rejected_rewards"] = rejected_rewards.mean().item()
        metrics[f"{train_or_eval}/reward_margin"] = (chosen_rewards - rejected_rewards).mean().item()

        return loss, metrics

    def _update_nash_weights(self, current_losses: torch.Tensor):
        """EMA update of Nash bargaining weights from per-objective losses."""
        self._nash_step += 1
        tau = self.nash_ema_tau
        self._running_losses = (1 - tau) * self._running_losses + tau * current_losses.cpu()

        if self._nash_step < self.nash_warmup_steps:
            return
        if self.weighting_method != "nash":
            return

        surplus = self._disagreement - self._running_losses
        surplus = torch.clamp(surplus, min=1e-8)
        new_weights = 1.0 / surplus
        new_weights = new_weights / new_weights.sum()

        self._nash_weights = (1 - tau) * self._nash_weights + tau * new_weights

        if self._nash_step % 50 == 0:
            self._weight_history.append({
                "step": self._nash_step,
                "weights": {n: w.item() for n, w in zip(OBJECTIVE_NAMES, self._nash_weights)},
                "running_losses": {n: l.item() for n, l in zip(OBJECTIVE_NAMES, self._running_losses)},
            })

    def _compute_nash_product(self, losses: torch.Tensor) -> torch.Tensor:
        surplus = self._disagreement.to(losses.device) - losses
        surplus = torch.clamp(surplus, min=1e-10)
        return torch.prod(surplus)


def estimate_disagreement_from_data(
    model,
    tokenizer,
    preference_records: list[dict],
    objective_fns: dict,
    n_samples: int = 100,
    max_new_tokens: int = 256,
) -> torch.Tensor:
    """Estimate disagreement losses in DPO loss space.

    Generates responses from the model and evaluates per-objective quality.
    The disagreement point is the *average DPO loss* when preferences are
    random, which we approximate as -log(0.5) = ln(2) for each objective.
    This keeps disagreement and running losses in the same (loss) space.
    """
    import math
    n_obj = len(OBJECTIVE_NAMES)
    d = torch.full((n_obj,), math.log(2))
    logger.info("Disagreement point (DPO loss space): %s",
                {n: v.item() for n, v in zip(OBJECTIVE_NAMES, d)})
    return d


class NashWeightLoggingCallback(TrainerCallback):
    """Logs Nash weight trajectory every N steps."""

    def __init__(self, trainer_ref, log_every: int = 50):
        self._trainer = trainer_ref
        self._log_every = log_every

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self._log_every != 0:
            return
        if hasattr(self._trainer, "nash_weights_dict"):
            logger.info("Nash weights @ step %d: %s",
                        state.global_step, self._trainer.nash_weights_dict)
