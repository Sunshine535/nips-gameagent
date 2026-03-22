"""Game environment implementations for multi-agent decision training.
Supports: Prisoner's Dilemma, Coordination, Battle of Sexes, Stag Hunt,
Public Goods, Ultimatum, Auction, Negotiation."""

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GameState:
    """State of a game episode."""
    round_num: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_context(self, player_id: str) -> str:
        """Convert game state to natural language context for the LLM."""
        lines = [f"Round {self.round_num}. Your score: {self.scores.get(player_id, 0):.1f}"]
        if self.history:
            lines.append("History:")
            for h in self.history[-5:]:
                lines.append(f"  Round {h['round']}: {h.get('summary', '')}")
        return "\n".join(lines)


class GameEnvironment(ABC):
    """Abstract base class for game environments."""

    def __init__(self, config: dict):
        self.config = config
        self.name = config.get("name", "Unknown")
        self.num_rounds = config.get("num_rounds", 10)

    @abstractmethod
    def reset(self) -> GameState:
        pass

    @abstractmethod
    def step(self, actions: Dict[str, str]) -> Tuple[GameState, Dict[str, float]]:
        """Execute one round. Returns (new_state, rewards_per_player)."""
        pass

    @abstractmethod
    def get_action_space(self, player_id: str) -> List[str]:
        pass

    def get_prompt(self, state: GameState, player_id: str) -> str:
        """Generate decision prompt for a player."""
        context = state.to_prompt_context(player_id)
        actions = self.get_action_space(player_id)
        return (
            f"Game: {self.name}\n"
            f"Description: {self.config.get('description', '')}\n"
            f"{context}\n"
            f"Available actions: {', '.join(actions)}\n"
            f"Choose your action and explain your reasoning briefly."
        )


class MatrixGameEnvironment(GameEnvironment):
    """2-player matrix game (Prisoner's Dilemma, Coordination, etc.)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.actions = config["actions"]
        self.payoff_matrix = config["payoff_matrix"]
        self.state: Optional[GameState] = None

    def reset(self) -> GameState:
        self.state = GameState(
            round_num=0,
            scores={"player_0": 0.0, "player_1": 0.0},
        )
        return self.state

    def step(self, actions: Dict[str, str]) -> Tuple[GameState, Dict[str, float]]:
        a0 = actions.get("player_0", self.actions[0])
        a1 = actions.get("player_1", self.actions[0])

        key = f"{a0}_{a1}"
        payoffs = self.payoff_matrix.get(key, [0, 0])

        rewards = {"player_0": payoffs[0], "player_1": payoffs[1]}
        self.state.scores["player_0"] += payoffs[0]
        self.state.scores["player_1"] += payoffs[1]
        self.state.round_num += 1
        self.state.history.append({
            "round": self.state.round_num,
            "actions": actions,
            "payoffs": rewards,
            "summary": f"P0={a0}, P1={a1} → ({payoffs[0]}, {payoffs[1]})",
        })
        self.state.done = self.state.round_num >= self.num_rounds
        return self.state, rewards

    def get_action_space(self, player_id: str) -> List[str]:
        return self.actions


class PublicGoodsEnvironment(GameEnvironment):
    """N-player public goods game."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.num_players = config.get("num_players", 4)
        self.multiplier = config.get("multiplier", 2.0)
        self.endowment = config.get("endowment", 10)
        self.state: Optional[GameState] = None

    def reset(self) -> GameState:
        self.state = GameState(
            round_num=0,
            scores={f"player_{i}": 0.0 for i in range(self.num_players)},
        )
        return self.state

    def step(self, actions: Dict[str, str]) -> Tuple[GameState, Dict[str, float]]:
        contributions = {}
        for pid, action in actions.items():
            contributions[pid] = self.endowment if action == "contribute" else 0

        total_contributed = sum(contributions.values())
        public_good = total_contributed * self.multiplier / self.num_players

        rewards = {}
        for pid in actions:
            kept = self.endowment - contributions[pid]
            rewards[pid] = kept + public_good
            self.state.scores[pid] += rewards[pid]

        self.state.round_num += 1
        contrib_summary = ", ".join(f"{pid}={a}" for pid, a in actions.items())
        self.state.history.append({
            "round": self.state.round_num,
            "actions": actions,
            "contributions": contributions,
            "public_good": public_good,
            "summary": f"Contributions: {contrib_summary}. Pool={total_contributed:.0f}, Share={public_good:.1f}",
        })
        self.state.done = self.state.round_num >= self.num_rounds
        return self.state, rewards

    def get_action_space(self, player_id: str) -> List[str]:
        return ["contribute", "free_ride"]


class UltimatumEnvironment(GameEnvironment):
    """Ultimatum game: proposer offers, responder accepts/rejects."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.total = config.get("total_amount", 100)
        self.state: Optional[GameState] = None
        self.current_offer: Optional[int] = None

    def reset(self) -> GameState:
        self.state = GameState(
            round_num=0,
            scores={"proposer": 0.0, "responder": 0.0},
        )
        self.current_offer = None
        return self.state

    def step(self, actions: Dict[str, str]) -> Tuple[GameState, Dict[str, float]]:
        if self.current_offer is None:
            # Proposer phase
            try:
                offer = int(actions.get("proposer", "50"))
                offer = max(0, min(offer, self.total))
            except (ValueError, TypeError):
                offer = 50
            self.current_offer = offer
            return self.state, {"proposer": 0, "responder": 0}

        # Responder phase
        response = actions.get("responder", "accept").lower()
        accepted = "accept" in response

        if accepted:
            rewards = {"proposer": self.total - self.current_offer, "responder": self.current_offer}
        else:
            rewards = {"proposer": 0, "responder": 0}

        for pid, r in rewards.items():
            self.state.scores[pid] += r

        self.state.round_num += 1
        self.state.history.append({
            "round": self.state.round_num,
            "offer": self.current_offer,
            "accepted": accepted,
            "rewards": rewards,
            "summary": f"Offer={self.current_offer}, {'Accepted' if accepted else 'Rejected'} → {rewards}",
        })
        self.current_offer = None
        self.state.done = self.state.round_num >= self.num_rounds
        return self.state, rewards

    def get_action_space(self, player_id: str) -> List[str]:
        if player_id == "proposer":
            return [str(i) for i in range(0, self.total + 1, 10)]
        return ["accept", "reject"]


class AuctionEnvironment(GameEnvironment):
    """Sealed-bid first-price auction."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.num_players = config.get("num_players", 4)
        self.value_range = config.get("item_value_range", [50, 150])
        self.state: Optional[GameState] = None
        self.true_values: Dict[str, float] = {}

    def reset(self) -> GameState:
        self.state = GameState(
            round_num=0,
            scores={f"player_{i}": 0.0 for i in range(self.num_players)},
        )
        self.true_values = {
            f"player_{i}": random.uniform(*self.value_range) for i in range(self.num_players)
        }
        return self.state

    def step(self, actions: Dict[str, str]) -> Tuple[GameState, Dict[str, float]]:
        bids = {}
        for pid, action in actions.items():
            try:
                bids[pid] = float(action)
            except (ValueError, TypeError):
                bids[pid] = 0.0

        winner = max(bids, key=bids.get) if bids else None
        rewards = {pid: 0.0 for pid in actions}
        if winner:
            surplus = self.true_values.get(winner, 100) - bids[winner]
            rewards[winner] = max(surplus, 0.0)
            self.state.scores[winner] += rewards[winner]

        self.state.round_num += 1
        self.true_values = {
            f"player_{i}": random.uniform(*self.value_range) for i in range(self.num_players)
        }
        self.state.history.append({
            "round": self.state.round_num,
            "bids": bids,
            "winner": winner,
            "rewards": rewards,
            "summary": f"Bids: {bids}, Winner: {winner}, Reward: {rewards.get(winner, 0):.1f}",
        })
        self.state.done = self.state.round_num >= self.num_rounds
        return self.state, rewards

    def get_action_space(self, player_id: str) -> List[str]:
        return [str(v) for v in range(0, int(self.value_range[1]) + 1, 10)]


class NegotiationEnvironment(GameEnvironment):
    """Multi-issue negotiation between two parties."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.issues = config.get("issues", ["price", "delivery", "warranty"])
        self.max_rounds = config.get("max_rounds", 5)
        self.state: Optional[GameState] = None

    def reset(self) -> GameState:
        self.state = GameState(
            round_num=0,
            scores={"party_A": 0.0, "party_B": 0.0},
            info={"agreed_terms": {}, "pending_issues": list(self.issues)},
        )
        return self.state

    def step(self, actions: Dict[str, str]) -> Tuple[GameState, Dict[str, float]]:
        offer_a = actions.get("party_A", "")
        offer_b = actions.get("party_B", "")

        agreed = "agree" in offer_a.lower() and "agree" in offer_b.lower()
        rewards = {"party_A": 0.0, "party_B": 0.0}

        if agreed or self.state.round_num >= self.max_rounds - 1:
            if agreed:
                rewards = {"party_A": 5.0, "party_B": 5.0}
            else:
                rewards = {"party_A": 0.0, "party_B": 0.0}
            for pid, r in rewards.items():
                self.state.scores[pid] += r
            self.state.done = True
        else:
            rewards = {"party_A": -0.1, "party_B": -0.1}

        self.state.round_num += 1
        self.state.history.append({
            "round": self.state.round_num,
            "offers": actions,
            "agreed": agreed,
            "summary": f"A: {offer_a[:50]}, B: {offer_b[:50]}, Agreed: {agreed}",
        })
        self.state.done = self.state.done or self.state.round_num >= self.max_rounds
        return self.state, rewards

    def get_action_space(self, player_id: str) -> List[str]:
        return ["propose_high", "propose_medium", "propose_low", "agree", "reject"]


def create_environment(scenario_name: str, config: dict) -> GameEnvironment:
    """Factory function to create a game environment from config."""
    scenario = config["scenarios"][scenario_name]
    game_type = scenario.get("type", "")

    if scenario_name in ("prisoners_dilemma", "coordination_game", "battle_of_sexes",
                         "stag_hunt", "chicken", "matching_pennies"):
        return MatrixGameEnvironment(scenario)
    elif scenario_name == "public_goods":
        return PublicGoodsEnvironment(scenario)
    elif scenario_name == "ultimatum":
        return UltimatumEnvironment(scenario)
    elif scenario_name == "auction":
        return AuctionEnvironment(scenario)
    elif scenario_name == "negotiation":
        return NegotiationEnvironment(scenario)
    else:
        logger.warning("Unknown scenario %s, defaulting to MatrixGame", scenario_name)
        return MatrixGameEnvironment(scenario)
