"""
poker_env.py — River Endgame Environment
=========================================

Two-player zero-sum river endgame for No-Limit Hold'em.

Key design decisions
--------------------
* **Beta-distributed hand strength** parameterised by board texture
  (dry → polarised; wet → merged; neutral → mild).
* **Strictly zero-sum payoffs** — fold yields ±pot/2 net.
* **Human-fold-as-signal** — optional deliberate-fold mechanic that
  poisons an opposing bot's fold-frequency tracker.

References
----------
Zinkevich et al. (2007). "Regret Minimization in Games with
Incomplete Information." NeurIPS.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import ClassVar, Dict, Final, List, Tuple

import numpy as np
from scipy.stats import beta as beta_dist

# ── Constants ─────────────────────────────────────────────────────────

NUM_PLAYERS: Final[int] = 2
HERO: Final[int] = 0
OPPONENT: Final[int] = 1

ACTIONS: Final[Tuple[str, ...]] = (
    "FOLD", "CHECK", "CALL", "BET_HALF", "BET_POT", "ALL_IN",
)
ACTION_TO_IDX: Final[Dict[str, int]] = {a: i for i, a in enumerate(ACTIONS)}
NUM_ACTIONS: Final[int] = len(ACTIONS)

DEFAULT_HAND_BUCKETS: Final[int] = 10


# ── Board Texture ─────────────────────────────────────────────────────

class BoardTexture(Enum):
    """Board-texture classification (drives hand-strength distribution)."""
    DRY = auto()       # K♠ 7♦ 2♣  — low connectivity
    WET = auto()       # J♠ T♠ 9♥  — high draw density
    NEUTRAL = auto()   # A♥ 8♣ 3♦  — moderate texture


# ── Beta-Distributed Hand Strength ────────────────────────────────────

# Immutable lookup — shared across all instances.
_BETA_PARAMS: Final[Dict[BoardTexture, Tuple[float, float]]] = {
    BoardTexture.DRY:     (0.80, 0.80),   # U-shaped  → polarised
    BoardTexture.WET:     (2.50, 2.50),   # bell       → merged
    BoardTexture.NEUTRAL: (1.20, 1.20),   # mild       → slightly polar
}


@dataclass
class HandStrengthModel:
    """
    Draws hand-strength values s ∈ [0, 1] from Beta(α, β).

    The distribution shape is determined entirely by ``board_texture``:

    ===========  =====  =====  =============  ==========================
    Texture        α      β    Shape          Poker interpretation
    ===========  =====  =====  =============  ==========================
    Dry           0.80   0.80  U-shaped       Polarised (nuts-or-air)
    Wet           2.50   2.50  Bell-shaped    Merged / linear ranges
    Neutral       1.20   1.20  Mild-uniform   Moderate polarisation
    ===========  =====  =====  =============  ==========================
    """

    board_texture: BoardTexture = BoardTexture.DRY
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42)
    )

    @property
    def alpha(self) -> float:
        return _BETA_PARAMS[self.board_texture][0]

    @property
    def beta(self) -> float:
        return _BETA_PARAMS[self.board_texture][1]

    def sample(self, n: int = 1) -> np.ndarray:
        """Draw *n* hand-strength values from the texture's Beta."""
        return self.rng.beta(self.alpha, self.beta, size=n)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Beta PDF at *x*."""
        return beta_dist.pdf(x, self.alpha, self.beta)

    @property
    def params(self) -> Dict[str, float]:
        """Public accessor used by the visualisation layer."""
        return {"alpha": self.alpha, "beta": self.beta}


# ── Player State ──────────────────────────────────────────────────────

@dataclass
class PlayerState:
    """Per-player observable state at the river."""
    hand_strength: float          # ∈ [0, 1]
    stack: float
    is_hero: bool = True
    has_folded: bool = False
    total_invested: float = 0.0
    used_human_fold: bool = False  # flagged if fold was a deliberate signal

    def copy(self) -> "PlayerState":
        """Cheap shallow copy (all fields are scalars)."""
        return PlayerState(
            hand_strength=self.hand_strength,
            stack=self.stack,
            is_hero=self.is_hero,
            has_folded=self.has_folded,
            total_invested=self.total_invested,
            used_human_fold=self.used_human_fold,
        )


# ── Human-Fold-as-Signal ─────────────────────────────────────────────

@dataclass
class HumanFoldSignal:
    """
    Models the **deliberate fold** — a meta-game tactic.

    A human player intentionally folds a strong hand to inject a false
    datapoint into the opposing bot's fold-frequency tracker.  The bot
    then over-estimates the player's fold rate and increases its bluff
    frequency, at which point the player traps.

    The signal fires only when all three conditions hold:

    1. ``hand_strength ≥ hand_strength_floor`` (real equity to sacrifice).
    2. ``invested / pot ≤ sacrifice_ev_cap`` (bounded EV loss).
    3. Random roll < ``activation_probability`` (stochastic, unpredictable).
    """

    hand_strength_floor: float = 0.65
    sacrifice_ev_cap: float = 0.35
    activation_probability: float = 0.15
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(99)
    )

    def should_signal(
        self, hand_strength: float, pot: float, invested: float,
    ) -> bool:
        if hand_strength < self.hand_strength_floor:
            return False
        sacrifice = invested / pot if pot > 0 else 0.0
        if sacrifice > self.sacrifice_ev_cap:
            return False
        return float(self._rng.random()) < self.activation_probability


# ── Game State ────────────────────────────────────────────────────────

@dataclass
class GameState:
    """Full state of a river endgame decision node."""
    players: List[PlayerState] = field(default_factory=list)
    pot: float = 100.0
    history: List[str] = field(default_factory=list)
    active_player: int = HERO
    is_terminal: bool = False
    human_fold_deployed: bool = False


# ── River Endgame Environment ─────────────────────────────────────────

class RiverEndgameEnv:
    """
    Two-player zero-sum river endgame environment.

    Manages hand sampling, legal-action masking, state transitions,
    terminal payoffs, and the optional human-fold layer.

    Parameters
    ----------
    pot : float
        Starting pot.
    stack : float
        Effective stack (symmetric).
    board_texture : BoardTexture
        Drives the Beta hand-strength model.
    hand_buckets : int
        Information-abstraction granularity.
    enable_human_fold : bool
        Toggle the deliberate-fold signal.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        pot: float = 100.0,
        stack: float = 200.0,
        board_texture: BoardTexture = BoardTexture.DRY,
        hand_buckets: int = DEFAULT_HAND_BUCKETS,
        enable_human_fold: bool = False,
        seed: int = 42,
    ) -> None:
        self.initial_pot = pot
        self.initial_stack = stack
        self.board_texture = board_texture
        self.hand_buckets = hand_buckets
        self.enable_human_fold = enable_human_fold
        self.hand_model = HandStrengthModel(
            board_texture=board_texture,
            rng=np.random.default_rng(seed),
        )
        self.human_signal = HumanFoldSignal(
            _rng=np.random.default_rng(seed + 1),
        )
        self.human_folds_fired: int = 0
        self.total_games: int = 0

    # ── Game lifecycle ────────────────────────────────────────────

    def new_game(self) -> GameState:
        """Initialise a fresh deal with sampled hand strengths."""
        self.total_games += 1
        strengths = self.hand_model.sample(n=NUM_PLAYERS)
        return GameState(
            players=[
                PlayerState(
                    hand_strength=float(strengths[i]),
                    stack=self.initial_stack,
                    is_hero=(i == HERO),
                )
                for i in range(NUM_PLAYERS)
            ],
            pot=self.initial_pot,
            history=[],
            active_player=HERO,
            is_terminal=False,
        )

    # ── Legal actions ─────────────────────────────────────────────

    @staticmethod
    def legal_actions(state: GameState) -> List[int]:
        """
        Return sorted indices of legal actions for the active player.

        When facing a bet the player may FOLD, CALL, or raise
        (BET_HALF / BET_POT / ALL_IN).  When *not* facing a bet
        the player may CHECK or open with a bet/raise.
        """
        player = state.players[state.active_player]
        facing_bet = (
            len(state.history) > 0
            and state.history[-1] in ("BET_HALF", "BET_POT", "ALL_IN")
        )
        actions: List[int] = []

        if facing_bet:
            actions.append(ACTION_TO_IDX["FOLD"])
            actions.append(ACTION_TO_IDX["CALL"])
        else:
            actions.append(ACTION_TO_IDX["CHECK"])

        if player.stack > 0:
            actions.append(ACTION_TO_IDX["BET_HALF"])
            actions.append(ACTION_TO_IDX["BET_POT"])
            actions.append(ACTION_TO_IDX["ALL_IN"])

        return sorted(set(actions))

    # ── State transition ──────────────────────────────────────────

    def apply_action(self, state: GameState, action_idx: int) -> GameState:
        """Apply *action_idx* and return a **new** GameState (immutable)."""
        action_name = ACTIONS[action_idx]
        acting = state.active_player

        new_players = [p.copy() for p in state.players]
        new_pot = state.pot
        new_history = state.history + [action_name]
        new_terminal = False
        human_flag = state.human_fold_deployed

        if action_name == "FOLD":
            new_players[acting].has_folded = True
            new_terminal = True
            # Human-fold signal check
            if (
                self.enable_human_fold
                and acting == HERO
                and self.human_signal.should_signal(
                    new_players[acting].hand_strength,
                    new_pot,
                    new_players[acting].total_invested,
                )
            ):
                new_players[acting].used_human_fold = True
                human_flag = True
                self.human_folds_fired += 1

        elif action_name == "CHECK":
            # Both checked → showdown
            if len(new_history) >= 2 and new_history[-2] == "CHECK":
                new_terminal = True

        elif action_name == "CALL":
            # Match the previous bet → showdown
            prev_bet = self._bet_amount_from_action(
                state.history[-1] if state.history else "CHECK",
                state.pot - (new_players[acting].total_invested
                             + new_players[1 - acting].total_invested),
                new_players[acting].stack,
            )
            call_amount = min(prev_bet, new_players[acting].stack)
            new_players[acting].stack -= call_amount
            new_players[acting].total_invested += call_amount
            new_pot += call_amount
            new_terminal = True

        else:  # BET_HALF, BET_POT, ALL_IN
            bet = self._compute_bet(action_name, new_pot, new_players[acting].stack)
            new_players[acting].stack -= bet
            new_players[acting].total_invested += bet
            new_pot += bet
            # If opponent already bet, this is a raise → resolve
            prev = state.history[-1] if state.history else None
            if prev in ("BET_HALF", "BET_POT", "ALL_IN"):
                new_terminal = True

        return GameState(
            players=new_players,
            pot=new_pot,
            history=new_history,
            active_player=1 - acting,
            is_terminal=new_terminal,
            human_fold_deployed=human_flag,
        )

    # ── Terminal payoff (ZERO-SUM) ────────────────────────────────

    @staticmethod
    def terminal_payoff(state: GameState) -> Tuple[float, float]:
        """
        Compute (hero_payoff, opponent_payoff).

        **Invariant**: ``hero + opponent ≡ 0`` at every terminal node.

        On fold → winner nets pot/2 (above own contribution).
        On showdown → stronger hand nets pot/2; tie → (0, 0).
        """
        if not state.is_terminal:
            raise ValueError("terminal_payoff() called on non-terminal node.")

        hero = state.players[HERO]
        opp = state.players[OPPONENT]
        half = state.pot / 2.0

        if hero.has_folded:
            return (-half, half)
        if opp.has_folded:
            return (half, -half)
        if hero.hand_strength > opp.hand_strength:
            return (half, -half)
        if opp.hand_strength > hero.hand_strength:
            return (-half, half)
        return (0.0, 0.0)

    # ── Information set key ───────────────────────────────────────

    def info_set_key(self, state: GameState) -> str:
        """Unique key: ``P<id>|HS<bucket>|<action_history>``."""
        player = state.players[state.active_player]
        bucket = min(
            int(player.hand_strength * self.hand_buckets),
            self.hand_buckets - 1,
        )
        hist = "->".join(state.history) if state.history else "ROOT"
        return f"P{state.active_player}|HS{bucket}|{hist}"

    # ── Bet helpers ───────────────────────────────────────────────

    @staticmethod
    def _compute_bet(action: str, pot: float, stack: float) -> float:
        if action == "BET_HALF":
            return min(pot * 0.5, stack)
        if action == "BET_POT":
            return min(pot * 1.0, stack)
        if action == "ALL_IN":
            return stack
        return 0.0

    @staticmethod
    def _bet_amount_from_action(
        action: str, pot: float, stack: float,
    ) -> float:
        """Resolve the chip amount that a prior bet/raise committed."""
        return RiverEndgameEnv._compute_bet(action, pot, stack)

    @property
    def spr(self) -> float:
        """Stack-to-pot ratio."""
        if self.initial_pot <= 0:
            return float("inf")
        return self.initial_stack / self.initial_pot
