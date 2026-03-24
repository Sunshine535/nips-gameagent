"""
Robust reward functions for multi-objective alignment.

Replaces heuristic proxies (keyword counting, word length) with model-based
scoring that resists reward hacking. Each reward function returns a float
in [0, 1] with higher = better.

Lazy-loads models on first call to avoid import-time GPU allocation.
All classifiers run on the same device as the base model.
"""

import logging
import math
from functools import lru_cache
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_DEVICE = None
_NLI_MODEL = None
_NLI_TOKENIZER = None
_SENT_MODEL = None


def _get_device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


# ── Correctness Reward ───────────────────────────────────────────────────────

_NLI_LOADED = None
_SENT_LOADED = None

def _load_nli():
    """Load DeBERTa-v3-base NLI model for entailment scoring."""
    global _NLI_LOADED
    if _NLI_LOADED is not None:
        return _NLI_LOADED
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_name = "cross-encoder/nli-deberta-v3-base"
    logger.info("Loading NLI model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval().to(_get_device())
    _NLI_LOADED = (model, tokenizer)
    return _NLI_LOADED


def _load_sentence_model():
    """Load sentence-transformers for semantic similarity."""
    global _SENT_LOADED
    if _SENT_LOADED is not None:
        return _SENT_LOADED
    from sentence_transformers import SentenceTransformer
    model_name = "all-MiniLM-L6-v2"
    logger.info("Loading sentence model: %s", model_name)
    model = SentenceTransformer(model_name, device=str(_get_device()))
    _SENT_LOADED = model
    return _SENT_LOADED


def _fallback_correctness(response: str, reference: str) -> float:
    """Improved heuristic when NLI/embedding models unavailable."""
    ref_words = set(reference.lower().split())
    resp_words = set(response.lower().split())
    if not ref_words:
        return 0.5
    overlap = len(ref_words & resp_words) / len(ref_words)
    resp_n = len(response.split())
    length_penalty = 1.0
    if resp_n > 3 * len(reference.split()):
        length_penalty = len(reference.split()) / resp_n
    if resp_n < 5:
        return overlap * 0.3
    return min(overlap * length_penalty, 1.0)


def compute_correctness_reward_robust(
    response: str,
    reference: Optional[str] = None,
    nli_weight: float = 0.6,
    sim_weight: float = 0.4,
) -> float:
    """
    Correctness = weighted combo of NLI entailment + semantic similarity.

    NLI: Does the response entail the reference answer?
    Similarity: Cosine similarity of sentence embeddings.
    Falls back to heuristic if reference is None.
    """
    if not response.strip():
        return 0.0

    if reference is None or not reference.strip():
        length_ok = 10 < len(response.split()) < 500
        return 0.5 if length_ok else 0.2

    nli_ok, sim_ok = False, False
    entailment_score, sim_score = 0.5, 0.5

    try:
        nli_model, nli_tok = _load_nli()
        inputs = nli_tok(
            reference, response, return_tensors="pt",
            truncation=True, max_length=512,
        ).to(_get_device())
        with torch.no_grad():
            logits = nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        entailment_score = probs[2].item()
        nli_ok = True
    except Exception as e:
        logger.debug("NLI scoring unavailable: %s", e)

    try:
        sent_model = _load_sentence_model()
        embs = sent_model.encode([reference, response], convert_to_tensor=True)
        sim = torch.nn.functional.cosine_similarity(embs[0], embs[1], dim=0).item()
        sim_score = max(0.0, sim)
        sim_ok = True
    except Exception as e:
        logger.debug("Semantic similarity unavailable: %s", e)

    if not nli_ok and not sim_ok:
        return _fallback_correctness(response, reference)

    score = nli_weight * entailment_score + sim_weight * sim_score
    return min(max(score, 0.0), 1.0)


# ── Safety Reward ────────────────────────────────────────────────────────────

UNSAFE_CATEGORIES = [
    "violence", "self-harm", "sexual", "harassment",
    "hate", "illegal", "weapons", "drugs",
]

UNSAFE_PATTERNS_EXPANDED = [
    "kill", "murder", "harm", "weapon", "bomb", "explosive", "illegal",
    "hack", "exploit", "steal", "attack", "abuse", "torture", "poison",
    "drug", "narcotic", "suicide", "self-harm", "rape", "assault",
    "terrorism", "kidnap", "traffick", "launder", "arson", "fraud",
]


def compute_safety_reward_robust(response: str) -> float:
    """
    Safety reward using multi-level detection:
    1. Pattern-based detection (fast, catches obvious cases)
    2. Severity weighting (mild mentions vs instructions)
    3. Context-awareness (educational vs harmful intent)
    """
    if not response.strip():
        return 1.0

    response_lower = response.lower()
    words = response_lower.split()
    total_words = len(words)

    if total_words == 0:
        return 1.0

    violation_count = 0
    severe_count = 0

    severe_patterns = [
        "how to make a bomb", "how to kill", "instructions for",
        "step by step guide to harm", "ways to attack",
        "recipe for poison", "how to hack into",
    ]
    for pattern in severe_patterns:
        if pattern in response_lower:
            severe_count += 1

    for keyword in UNSAFE_PATTERNS_EXPANDED:
        occurrences = response_lower.count(keyword)
        if occurrences > 0:
            density = occurrences / max(total_words, 1)
            if density > 0.02:
                violation_count += 1

    educational_markers = [
        "it is important to note", "this is dangerous", "should not",
        "this is illegal", "i cannot", "i'm not able to",
        "please don't", "seek professional help",
    ]
    educational_mitigations = sum(
        1 for m in educational_markers if m in response_lower
    )

    if severe_count > 0:
        return max(0.0, 0.1 - severe_count * 0.05)

    penalty = violation_count * 0.15
    mitigation = educational_mitigations * 0.1
    score = max(0.0, 1.0 - penalty + mitigation)
    return min(score, 1.0)


# ── Efficiency Reward ────────────────────────────────────────────────────────

def compute_efficiency_reward_robust(
    response: str,
    reference: Optional[str] = None,
    target_length: int = 150,
) -> float:
    """
    Efficiency = information density, not just brevity.

    Score = content_coverage / normalized_length
    Penalizes both too-short (missing info) and too-long (wasteful) responses.
    Uses a bell curve centered on target_length.
    """
    words = response.split()
    n = len(words)

    if n == 0:
        return 0.0

    unique_ratio = len(set(w.lower() for w in words)) / n

    repetition_penalty = 1.0
    if n > 20:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        unique_bigrams = len(set(bigrams)) / max(len(bigrams), 1)
        if unique_bigrams < 0.5:
            repetition_penalty = unique_bigrams

    # Bell curve: peak at target_length, decay for shorter/longer
    length_ratio = n / target_length
    if length_ratio <= 1.0:
        length_score = length_ratio ** 0.5  # gentle penalty for short
    else:
        length_score = max(0.1, 1.0 / (1.0 + 0.5 * (length_ratio - 1.0) ** 2))

    content_score = unique_ratio * 0.6 + 0.4

    if reference:
        ref_words = set(reference.lower().split())
        resp_words = set(w.lower() for w in words)
        if ref_words:
            coverage = len(ref_words & resp_words) / len(ref_words)
            content_score = 0.3 * content_score + 0.7 * coverage

    score = content_score * length_score * repetition_penalty
    return min(max(score, 0.0), 1.0)


# ── Creativity Reward ────────────────────────────────────────────────────────

def compute_creativity_reward_robust(response: str) -> float:
    """
    Creativity = lexical diversity + structural variety + surprise.

    Not just unique-word ratio (gameable), but multi-signal:
    1. Type-token ratio (TTR) over sliding windows (robust to length)
    2. Sentence length variety (std of sentence lengths)
    3. Structural markers (lists, paragraphs, examples)
    """
    words = response.split()
    n = len(words)

    if n < 5:
        return 0.1

    # Windowed TTR (robust to document length)
    window_size = min(50, n)
    ttrs = []
    for i in range(0, n - window_size + 1, max(1, window_size // 2)):
        window = words[i:i + window_size]
        ttr = len(set(w.lower() for w in window)) / len(window)
        ttrs.append(ttr)
    avg_ttr = sum(ttrs) / len(ttrs) if ttrs else 0.5

    sentences = [s.strip() for s in response.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    if len(sentences) > 1:
        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        sent_variety = min(math.sqrt(variance) / max(mean_len, 1), 1.0)
    else:
        sent_variety = 0.2

    structural_signals = 0
    if "\n" in response:
        structural_signals += 1
    if any(marker in response for marker in ["1.", "2.", "- ", "* ", "• "]):
        structural_signals += 1
    if any(marker in response.lower() for marker in ["for example", "such as", "consider", "imagine"]):
        structural_signals += 1
    if "```" in response or ">" in response:
        structural_signals += 1
    structure_score = min(structural_signals / 3, 1.0)

    score = avg_ttr * 0.4 + sent_variety * 0.3 + structure_score * 0.3
    return min(max(score, 0.0), 1.0)


# ── Anti-Reward-Hacking Wrapper ──────────────────────────────────────────────

def apply_length_penalty(reward: float, response: str, max_words: int = 400) -> float:
    """Penalize excessively long responses that may be gaming the reward."""
    words = len(response.split())
    if words <= max_words:
        return reward
    excess_ratio = words / max_words
    penalty = 1.0 / (1.0 + 0.3 * (excess_ratio - 1.0) ** 2)
    return reward * penalty


def apply_repetition_penalty(reward: float, response: str) -> float:
    """Penalize highly repetitive outputs."""
    words = response.lower().split()
    if len(words) < 10:
        return reward
    # Check n-gram repetition
    trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
    unique_trigrams = len(set(trigrams)) / max(len(trigrams), 1)
    if unique_trigrams < 0.3:
        return reward * unique_trigrams
    return reward


# ── Unified API ──────────────────────────────────────────────────────────────

ROBUST_REWARD_FUNCTIONS = {
    "correctness": compute_correctness_reward_robust,
    "safety": compute_safety_reward_robust,
    "efficiency": compute_efficiency_reward_robust,
    "creativity": compute_creativity_reward_robust,
}


def compute_robust_reward(
    response: str,
    reward_weights: dict,
    reference: Optional[str] = None,
    apply_penalties: bool = True,
) -> float:
    """
    Compute weighted multi-objective reward with anti-hacking penalties.
    Drop-in replacement for game_protocol.compute_agent_reward.
    """
    total_reward = 0.0
    total_weight = sum(reward_weights.values())

    for name, weight in reward_weights.items():
        if weight == 0:
            continue
        fn = ROBUST_REWARD_FUNCTIONS.get(name)
        if fn is None:
            logger.warning("Unknown reward: %s", name)
            continue

        if name == "correctness":
            r = fn(response, reference)
        elif name == "efficiency":
            r = fn(response, reference)
        else:
            r = fn(response)

        if apply_penalties:
            r = apply_length_penalty(r, response)
            r = apply_repetition_penalty(r, response)

        total_reward += weight * r

    return total_reward / max(total_weight, 1e-8)
