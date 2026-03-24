"""
Game protocol for asymmetric multi-agent self-play.

Phases:
  1. Generate: Each agent generates a candidate response
  2. Evaluate: Agents cross-evaluate each other's candidates
  3. Aggregate: Collect preference pairs from cross-evaluations
  4. Update: Nash-DPO update using aggregated preferences
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class AgentRole:
    name: str
    role_id: str
    description: str
    reward_weights: dict = field(default_factory=dict)
    eval_prompt_suffix: str = ""


@dataclass
class GameCandidate:
    agent_id: str
    prompt: str
    response: str
    self_score: float = 0.0


@dataclass
class CrossEvaluation:
    evaluator_id: str
    candidate_agent_id: str
    prompt: str
    response: str
    score: float = 0.0
    reasoning: str = ""


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    chosen_agent: str
    rejected_agent: str
    margin: float = 0.0


# ── Reward Functions ─────────────────────────────────────────────────────────

def compute_correctness_reward(response: str, reference: Optional[str] = None) -> float:
    score = 0.5
    if reference:
        ref_words = set(reference.lower().split())
        resp_words = set(response.lower().split())
        if ref_words:
            overlap = len(ref_words & resp_words) / len(ref_words)
            score = overlap
    if len(response.strip()) < 10:
        score *= 0.5
    return min(score, 1.0)


def compute_safety_reward(response: str) -> float:
    unsafe_patterns = [
        "kill", "harm", "weapon", "illegal", "hack", "exploit", "bomb",
        "drug", "steal", "attack", "violence", "abuse",
    ]
    response_lower = response.lower()
    penalty = sum(1 for p in unsafe_patterns if p in response_lower)
    return max(0.0, 1.0 - penalty * 0.2)


def compute_efficiency_reward(response: str) -> float:
    words = response.split()
    n = len(words)
    if n == 0:
        return 0.0
    if n <= 50:
        return 1.0
    elif n <= 150:
        return 0.8
    elif n <= 300:
        return 0.5
    return 0.3


def compute_creativity_reward(response: str) -> float:
    words = response.lower().split()
    if not words:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    creative_markers = [
        "imagine", "consider", "alternatively", "interestingly",
        "novel", "unique", "perhaps", "what if", "metaphor",
    ]
    marker_count = sum(1 for m in creative_markers if m in response.lower())
    marker_bonus = min(marker_count * 0.1, 0.3)
    return min(unique_ratio * 0.7 + marker_bonus + 0.1, 1.0)


REWARD_FUNCTIONS = {
    "correctness": compute_correctness_reward,
    "safety": compute_safety_reward,
    "efficiency": compute_efficiency_reward,
    "creativity": compute_creativity_reward,
}


def compute_agent_reward(response: str, reward_weights: dict,
                         reference: Optional[str] = None,
                         use_robust: bool = True) -> float:
    if use_robust:
        try:
            from src.reward_models import compute_robust_reward
            return compute_robust_reward(response, reward_weights, reference)
        except (ImportError, Exception):
            pass
        try:
            from reward_models import compute_robust_reward
            return compute_robust_reward(response, reward_weights, reference)
        except (ImportError, Exception):
            pass

    total_reward = 0.0
    total_weight = sum(reward_weights.values())
    for reward_name, weight in reward_weights.items():
        if weight == 0:
            continue
        fn = REWARD_FUNCTIONS[reward_name]
        if reward_name == "correctness":
            r = fn(response, reference)
        else:
            r = fn(response)
        total_reward += weight * r
    return total_reward / max(total_weight, 1e-8)


# ── Game Protocol Steps ──────────────────────────────────────────────────────

@torch.no_grad()
def generate_candidates(
    models: dict[str, AutoModelForCausalLM],
    tokenizers: dict[str, AutoTokenizer],
    agents: dict[str, AgentRole],
    prompts: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> list[list[GameCandidate]]:
    all_candidates = []
    for prompt in prompts:
        prompt_candidates = []
        for agent_id, agent in agents.items():
            model = models[agent_id]
            tokenizer = tokenizers[agent_id]
            full_prompt = f"{prompt}\n\n{agent.eval_prompt_suffix}"
            inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True,
                               max_length=1024).to(model.device)
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=True, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
            response = tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            candidate = GameCandidate(
                agent_id=agent_id, prompt=prompt, response=response,
            )
            prompt_candidates.append(candidate)
        all_candidates.append(prompt_candidates)
    return all_candidates


def cross_evaluate(
    candidates: list[list[GameCandidate]],
    agents: dict[str, AgentRole],
    reference_answers: Optional[list[str]] = None,
) -> list[CrossEvaluation]:
    evaluations = []
    for prompt_idx, prompt_candidates in enumerate(candidates):
        ref = reference_answers[prompt_idx] if reference_answers else None
        for evaluator_id, evaluator in agents.items():
            for candidate in prompt_candidates:
                score = compute_agent_reward(
                    candidate.response, evaluator.reward_weights, reference=ref,
                )
                evaluations.append(CrossEvaluation(
                    evaluator_id=evaluator_id,
                    candidate_agent_id=candidate.agent_id,
                    prompt=candidate.prompt,
                    response=candidate.response,
                    score=score,
                ))
    return evaluations


def aggregate_preferences(
    evaluations: list[CrossEvaluation],
    candidates: list[list[GameCandidate]],
    min_margin: float = 0.1,
) -> list[PreferencePair]:
    grouped = {}
    for ev in evaluations:
        key = (ev.prompt, ev.evaluator_id)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(ev)

    pairs = []
    for (prompt, evaluator_id), evals in grouped.items():
        sorted_evals = sorted(evals, key=lambda e: e.score, reverse=True)
        for i in range(len(sorted_evals)):
            for j in range(i + 1, len(sorted_evals)):
                margin = sorted_evals[i].score - sorted_evals[j].score
                if margin >= min_margin:
                    pairs.append(PreferencePair(
                        prompt=prompt,
                        chosen=sorted_evals[i].response,
                        rejected=sorted_evals[j].response,
                        chosen_agent=sorted_evals[i].candidate_agent_id,
                        rejected_agent=sorted_evals[j].candidate_agent_id,
                        margin=margin,
                    ))
    return pairs


def majority_vote_select(
    agent_responses: dict[str, list[str]],
    agent_roles: dict[str, AgentRole],
) -> list[str]:
    """Select best response per prompt via multi-agent cross-evaluation voting.

    Each agent scores every candidate using its own reward weights, then the
    candidate with the highest total score across all evaluator agents wins.
    """
    num_prompts = len(next(iter(agent_responses.values())))
    selected = []

    for i in range(num_prompts):
        best_response, best_score = "", -float("inf")
        for agent_id in agent_responses:
            response = agent_responses[agent_id][i]
            total = sum(
                compute_agent_reward(response, role.reward_weights)
                for role in agent_roles.values()
            )
            if total > best_score:
                best_score = total
                best_response = response
        selected.append(best_response)

    return selected


def compute_elo_ratings(evaluations: list[CrossEvaluation],
                        agents: dict[str, AgentRole],
                        k_factor: float = 32.0) -> dict[str, float]:
    elo = {aid: 1500.0 for aid in agents}
    grouped_by_prompt = {}
    for ev in evaluations:
        if ev.prompt not in grouped_by_prompt:
            grouped_by_prompt[ev.prompt] = {}
        key = (ev.evaluator_id, ev.candidate_agent_id)
        grouped_by_prompt[ev.prompt][key] = ev.score

    for prompt, scores in grouped_by_prompt.items():
        agent_ids = list(agents.keys())
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                a, b = agent_ids[i], agent_ids[j]
                sa = sum(scores.get((ev_id, a), 0) for ev_id in agent_ids) / len(agent_ids)
                sb = sum(scores.get((ev_id, b), 0) for ev_id in agent_ids) / len(agent_ids)
                ea = 1 / (1 + 10 ** ((elo[b] - elo[a]) / 400))
                if sa > sb:
                    elo[a] += k_factor * (1 - ea)
                    elo[b] -= k_factor * (1 - ea)
                elif sb > sa:
                    elo[a] -= k_factor * ea
                    elo[b] += k_factor * ea
    return elo
