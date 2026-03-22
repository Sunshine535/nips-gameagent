"""
Eight classic game-theory environments for LLM self-play.

Each game presents a scenario as a text prompt; the LLM responds with a
chosen action; a parser extracts the discrete action; payoffs follow the
game matrix.  Environments expose a uniform interface so the self-play
loop can iterate over them identically.
"""

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    game_name: str
    prompt: str
    player1_action: str
    player2_action: str
    player1_payoff: float
    player2_payoff: float
    nash_actions: list = field(default_factory=list)
    is_nash: bool = False


class GameEnvironment:
    """Base class for two-player normal-form games."""

    name: str = "base"
    actions: list[str] = []
    payoff_matrix: dict = {}  # (a1, a2) -> (p1, p2)
    nash_equilibria: list[tuple[str, str]] = []

    def build_prompt(self, player_id: int, history: Optional[list] = None) -> str:
        raise NotImplementedError

    def parse_action(self, response: str) -> str:
        """Extract the chosen action from an LLM response."""
        response_lower = response.lower().strip()
        for action in self.actions:
            if action.lower() in response_lower:
                return action
        return random.choice(self.actions)

    def get_payoffs(self, a1: str, a2: str) -> tuple[float, float]:
        return self.payoff_matrix.get((a1, a2), (0.0, 0.0))

    def play(self, response1: str, response2: str, prompt: str = "") -> GameResult:
        a1 = self.parse_action(response1)
        a2 = self.parse_action(response2)
        p1, p2 = self.get_payoffs(a1, a2)
        is_nash = (a1, a2) in self.nash_equilibria
        return GameResult(
            game_name=self.name, prompt=prompt,
            player1_action=a1, player2_action=a2,
            player1_payoff=p1, player2_payoff=p2,
            nash_actions=[(a, b) for a, b in self.nash_equilibria],
            is_nash=is_nash,
        )

    def nash_distance(self, a1: str, a2: str) -> float:
        """Distance from (a1,a2) to the nearest Nash equilibrium payoff."""
        p1, p2 = self.get_payoffs(a1, a2)
        best = float("inf")
        for na1, na2 in self.nash_equilibria:
            np1, np2 = self.get_payoffs(na1, na2)
            dist = ((p1 - np1) ** 2 + (p2 - np2) ** 2) ** 0.5
            best = min(best, dist)
        return best


# ── 1. Prisoners Dilemma ─────────────────────────────────────────────────────

class PrisonersDilemma(GameEnvironment):
    name = "prisoners_dilemma"
    actions = ["Cooperate", "Defect"]
    payoff_matrix = {
        ("Cooperate", "Cooperate"): (3.0, 3.0),
        ("Cooperate", "Defect"):    (0.0, 5.0),
        ("Defect",    "Cooperate"): (5.0, 0.0),
        ("Defect",    "Defect"):    (1.0, 1.0),
    }
    nash_equilibria = [("Defect", "Defect")]

    def build_prompt(self, player_id, history=None):
        h = ""
        if history:
            h = "\nPrevious rounds:\n" + "\n".join(
                f"  Round {i+1}: You={r[0]}, Opponent={r[1]}" for i, r in enumerate(history)
            ) + "\n"
        return (
            "You are playing the Prisoner's Dilemma.\n"
            "You and another player each independently choose to Cooperate or Defect.\n"
            "Payoffs: Both Cooperate -> 3 each. Both Defect -> 1 each.\n"
            "You Cooperate, they Defect -> you get 0, they get 5 (and vice versa).\n"
            f"{h}"
            "Choose your action. Respond with exactly one word: Cooperate or Defect."
        )


# ── 2. Battle of the Sexes ───────────────────────────────────────────────────

class BattleOfSexes(GameEnvironment):
    name = "battle_of_sexes"
    actions = ["Opera", "Football"]
    payoff_matrix = {
        ("Opera",    "Opera"):    (3.0, 2.0),
        ("Opera",    "Football"): (0.0, 0.0),
        ("Football", "Opera"):    (0.0, 0.0),
        ("Football", "Football"): (2.0, 3.0),
    }
    nash_equilibria = [("Opera", "Opera"), ("Football", "Football")]

    def build_prompt(self, player_id, history=None):
        pref = "Opera" if player_id == 0 else "Football"
        return (
            "You are playing the Battle of the Sexes game.\n"
            "Two players choose between Opera and Football.\n"
            f"You prefer {pref}. If you both choose the same, you both gain "
            "(3 for your preference, 2 for the other). If you mismatch, both get 0.\n"
            "Choose your action. Respond with exactly one word: Opera or Football."
        )


# ── 3. Stag Hunt ─────────────────────────────────────────────────────────────

class StagHunt(GameEnvironment):
    name = "stag_hunt"
    actions = ["Stag", "Hare"]
    payoff_matrix = {
        ("Stag", "Stag"): (4.0, 4.0),
        ("Stag", "Hare"): (0.0, 3.0),
        ("Hare", "Stag"): (3.0, 0.0),
        ("Hare", "Hare"): (2.0, 2.0),
    }
    nash_equilibria = [("Stag", "Stag"), ("Hare", "Hare")]

    def build_prompt(self, player_id, history=None):
        return (
            "You are playing the Stag Hunt game.\n"
            "Both choose Stag or Hare. Both Stag -> 4 each (best cooperative outcome).\n"
            "Both Hare -> 2 each. One Stag + one Hare -> Stag gets 0, Hare gets 3.\n"
            "Choose your action. Respond with exactly one word: Stag or Hare."
        )


# ── 4. Chicken (Hawk-Dove) ───────────────────────────────────────────────────

class Chicken(GameEnvironment):
    name = "chicken"
    actions = ["Swerve", "Straight"]
    payoff_matrix = {
        ("Swerve",   "Swerve"):   (3.0, 3.0),
        ("Swerve",   "Straight"): (1.0, 5.0),
        ("Straight", "Swerve"):   (5.0, 1.0),
        ("Straight", "Straight"): (0.0, 0.0),
    }
    nash_equilibria = [("Swerve", "Straight"), ("Straight", "Swerve")]

    def build_prompt(self, player_id, history=None):
        return (
            "You are playing the Chicken (Hawk-Dove) game.\n"
            "Both choose Swerve or Straight. Both Swerve -> 3 each.\n"
            "One Swerves, one goes Straight -> Straight gets 5, Swerve gets 1.\n"
            "Both Straight -> catastrophic: 0 each.\n"
            "Choose your action. Respond with exactly one word: Swerve or Straight."
        )


# ── 5. Matching Pennies ──────────────────────────────────────────────────────

class MatchingPennies(GameEnvironment):
    name = "matching_pennies"
    actions = ["Heads", "Tails"]
    payoff_matrix = {
        ("Heads", "Heads"): (1.0, -1.0),
        ("Heads", "Tails"): (-1.0, 1.0),
        ("Tails", "Heads"): (-1.0, 1.0),
        ("Tails", "Tails"): (1.0, -1.0),
    }
    nash_equilibria = []  # only mixed-strategy NE

    def build_prompt(self, player_id, history=None):
        role = "Matcher" if player_id == 0 else "Mismatcher"
        obj = "match" if player_id == 0 else "mismatch"
        return (
            f"You are the {role} in Matching Pennies.\n"
            "Both players show Heads or Tails simultaneously.\n"
            f"You want to {obj} the other player's choice.\n"
            f"Match -> {role} wins 1. Mismatch -> {role} loses 1.\n"
            "Choose your action. Respond with exactly one word: Heads or Tails."
        )

    def nash_distance(self, a1, a2):
        return 0.0  # mixed-strategy only; any play is equidistant


# ── 6. Public Goods Game ─────────────────────────────────────────────────────

class PublicGoods(GameEnvironment):
    name = "public_goods"
    actions = ["Contribute", "Free-ride"]
    payoff_matrix = {
        ("Contribute", "Contribute"):  (3.0, 3.0),
        ("Contribute", "Free-ride"):   (1.0, 4.0),
        ("Free-ride",  "Contribute"):  (4.0, 1.0),
        ("Free-ride",  "Free-ride"):   (1.5, 1.5),
    }
    nash_equilibria = [("Free-ride", "Free-ride")]

    def build_prompt(self, player_id, history=None):
        return (
            "You are in a Public Goods game with one other player.\n"
            "Each can Contribute to the public pool or Free-ride.\n"
            "Both Contribute -> 3 each. Both Free-ride -> 1.5 each.\n"
            "One Contributes, one Free-rides -> contributor gets 1, free-rider gets 4.\n"
            "Choose your action. Respond: Contribute or Free-ride."
        )

    def parse_action(self, response):
        r = response.lower()
        if "free" in r or "ride" in r:
            return "Free-ride"
        if "contrib" in r:
            return "Contribute"
        return random.choice(self.actions)


# ── 7. Ultimatum Game ────────────────────────────────────────────────────────

class UltimatumGame(GameEnvironment):
    name = "ultimatum"
    actions = ["Fair", "Greedy"]  # proposer: Fair=50/50, Greedy=80/20
    payoff_matrix = {
        ("Fair",   "Accept"): (5.0, 5.0),
        ("Fair",   "Reject"): (0.0, 0.0),
        ("Greedy", "Accept"): (8.0, 2.0),
        ("Greedy", "Reject"): (0.0, 0.0),
    }
    nash_equilibria = [("Greedy", "Accept")]

    def build_prompt(self, player_id, history=None):
        if player_id == 0:
            return (
                "You are the Proposer in the Ultimatum Game.\n"
                "You split 10 points. Choose Fair (5/5) or Greedy (8/2).\n"
                "The Responder can Accept or Reject. Reject -> both get 0.\n"
                "Choose your proposal. Respond with one word: Fair or Greedy."
            )
        return (
            "You are the Responder in the Ultimatum Game.\n"
            "The Proposer offers a split. You can Accept or Reject.\n"
            "If you Reject, both get 0. If you Accept, the split applies.\n"
            "Choose your action. Respond with one word: Accept or Reject."
        )

    def parse_action(self, response):
        r = response.lower()
        for a in ["Fair", "Greedy", "Accept", "Reject"]:
            if a.lower() in r:
                return a
        return random.choice(self.actions)

    def play(self, response1, response2, prompt=""):
        a1 = self.parse_action(response1)
        if a1 not in ("Fair", "Greedy"):
            a1 = "Fair"
        a2 = self.parse_action(response2)
        if a2 not in ("Accept", "Reject"):
            a2 = "Accept"
        p1, p2 = self.payoff_matrix.get((a1, a2), (0.0, 0.0))
        is_nash = (a1, a2) in self.nash_equilibria
        return GameResult(
            game_name=self.name, prompt=prompt,
            player1_action=a1, player2_action=a2,
            player1_payoff=p1, player2_payoff=p2,
            nash_actions=list(self.nash_equilibria),
            is_nash=is_nash,
        )


# ── 8. Coordination Game ─────────────────────────────────────────────────────

class CoordinationGame(GameEnvironment):
    name = "coordination"
    actions = ["Left", "Right"]
    payoff_matrix = {
        ("Left",  "Left"):  (2.0, 2.0),
        ("Left",  "Right"): (0.0, 0.0),
        ("Right", "Left"):  (0.0, 0.0),
        ("Right", "Right"): (2.0, 2.0),
    }
    nash_equilibria = [("Left", "Left"), ("Right", "Right")]

    def build_prompt(self, player_id, history=None):
        return (
            "You are in a Coordination Game.\n"
            "Both players choose Left or Right simultaneously.\n"
            "If you match (both Left or both Right), you each get 2 points.\n"
            "If you mismatch, both get 0.\n"
            "Choose your action. Respond with exactly one word: Left or Right."
        )


# ── Registry ─────────────────────────────────────────────────────────────────

ALL_GAMES: dict[str, GameEnvironment] = {
    "prisoners_dilemma": PrisonersDilemma(),
    "battle_of_sexes":   BattleOfSexes(),
    "stag_hunt":         StagHunt(),
    "chicken":           Chicken(),
    "matching_pennies":  MatchingPennies(),
    "public_goods":      PublicGoods(),
    "ultimatum":         UltimatumGame(),
    "coordination":      CoordinationGame(),
}


def get_game(name: str) -> GameEnvironment:
    if name not in ALL_GAMES:
        raise ValueError(f"Unknown game: {name}. Available: {list(ALL_GAMES.keys())}")
    return ALL_GAMES[name]


def list_games() -> list[str]:
    return list(ALL_GAMES.keys())
