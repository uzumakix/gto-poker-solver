"""
cfr_solver.py — Counterfactual Regret Minimization Engine
==========================================================

Implements CFR / CFR+ with:
* Regret matching for strategy computation.
* Opponent-model tracking (human-fold-signal aware).
* Exploitative adjustment via MDF criterion.

References
----------
Zinkevich et al. (2007). NeurIPS.
Tammelin (2014). arXiv:1407.5042.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Final, List, Optional

import numpy as np

from .poker_env import (
    ACTIONS,
    ACTION_TO_IDX,
    HERO,
    NUM_ACTIONS,
    OPPONENT,
    GameState,
    RiverEndgameEnv,
)

logger = logging.getLogger(__name__)


# ── Information-Set Node ──────────────────────────────────────────────

@dataclass
class InfoSetNode:
    """Accumulates counterfactual regrets and strategy weights."""

    num_actions: int = NUM_ACTIONS
    regret_sum: np.ndarray = field(default=None)
    strategy_sum: np.ndarray = field(default=None)
    visits: int = 0

    def __post_init__(self) -> None:
        if self.regret_sum is None:
            self.regret_sum = np.zeros(self.num_actions, dtype=np.float64)
        if self.strategy_sum is None:
            self.strategy_sum = np.zeros(self.num_actions, dtype=np.float64)

    def current_strategy(
        self,
        realisation_weight: float = 1.0,
        cfr_plus: bool = False,
    ) -> np.ndarray:
        """
        Regret matching: σ(a) = R⁺(a) / Σ_b R⁺(b).

        With ``cfr_plus=True`` negative regrets are floored to zero
        *in-place* before normalisation (Tammelin 2014).
        """
        if cfr_plus:
            np.maximum(self.regret_sum, 0.0, out=self.regret_sum)

        positive = np.maximum(self.regret_sum, 0.0)
        total = positive.sum()

        strategy = (
            positive / total if total > 0
            else np.full(self.num_actions, 1.0 / self.num_actions)
        )
        self.strategy_sum += realisation_weight * strategy
        self.visits += 1
        return strategy

    def average_strategy(self) -> np.ndarray:
        """Time-averaged strategy (converges to Nash as T → ∞)."""
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        return np.full(self.num_actions, 1.0 / self.num_actions)


# ── Opponent Model ────────────────────────────────────────────────────

@dataclass
class OpponentModel:
    """
    Tracks the Opponent's empirical action frequencies.

    Separates *raw* fold rate from the *adjusted* rate that
    discounts detected human-fold signals.
    """

    total_actions: int = 0
    fold_count: int = 0
    check_count: int = 0
    call_count: int = 0
    bet_half_count: int = 0
    bet_pot_count: int = 0
    all_in_count: int = 0
    human_fold_signals_detected: int = 0

    _ATTR_MAP: ClassVar[Dict[str, str]] = {
        "FOLD": "fold_count",
        "CHECK": "check_count",
        "CALL": "call_count",
        "BET_HALF": "bet_half_count",
        "BET_POT": "bet_pot_count",
        "ALL_IN": "all_in_count",
    }

    def record(self, action_name: str) -> None:
        self.total_actions += 1
        attr = self._ATTR_MAP.get(action_name)
        if attr:
            setattr(self, attr, getattr(self, attr) + 1)

    def record_human_fold_signal(self) -> None:
        self.human_fold_signals_detected += 1

    @property
    def fold_to_cbet(self) -> float:
        """Raw Opponent fold frequency (all folds / all actions)."""
        return self.fold_count / self.total_actions if self.total_actions else 0.0

    @property
    def adjusted_fold_rate(self) -> float:
        """Fold rate after subtracting known human-fold signals."""
        genuine = max(self.fold_count - self.human_fold_signals_detected, 0)
        return genuine / self.total_actions if self.total_actions else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_actions": self.total_actions,
            "fold_count": self.fold_count,
            "fold_to_cbet": round(self.fold_to_cbet, 4),
            "adjusted_fold_rate": round(self.adjusted_fold_rate, 4),
            "human_fold_signals": self.human_fold_signals_detected,
            "action_distribution": {
                a: getattr(self, self._ATTR_MAP[a]) for a in ACTIONS
            },
        }


# ── CFR Solver ────────────────────────────────────────────────────────

class CFRSolver:
    """
    Counterfactual Regret Minimization solver for the river endgame.

    Parameters
    ----------
    env : RiverEndgameEnv
        Game environment.
    use_cfr_plus : bool
        Floor negative regrets (CFR+).  Default ``True``.
    """

    def __init__(
        self,
        env: RiverEndgameEnv,
        use_cfr_plus: bool = True,
    ) -> None:
        self.env = env
        self.use_cfr_plus = use_cfr_plus

        self.info_set_map: Dict[str, InfoSetNode] = {}
        self.opponent_model = OpponentModel()
        self.regret_history: List[float] = []
        self.exploitability_history: List[float] = []
        self._iterations_run: int = 0

    # ── Node lookup ───────────────────────────────────────────────

    def _get_node(self, key: str) -> InfoSetNode:
        try:
            return self.info_set_map[key]
        except KeyError:
            node = InfoSetNode()
            self.info_set_map[key] = node
            return node

    # ── Core traversal ────────────────────────────────────────────

    def cfr(
        self,
        state: GameState,
        reach_hero: float,
        reach_opponent: float,
    ) -> float:
        """Recursive CFR traversal.  Returns Hero's expected utility."""
        if state.is_terminal:
            return self.env.terminal_payoff(state)[HERO]

        acting = state.active_player
        node = self._get_node(self.env.info_set_key(state))
        legal = self.env.legal_actions(state)

        reach_self = reach_hero if acting == HERO else reach_opponent
        strategy = node.current_strategy(reach_self, cfr_plus=self.use_cfr_plus)

        action_utils = np.zeros(NUM_ACTIONS, dtype=np.float64)
        node_util = 0.0

        for a in legal:
            next_state = self.env.apply_action(state, a)

            if acting == OPPONENT:
                self.opponent_model.record(ACTIONS[a])
                if next_state.human_fold_deployed:
                    self.opponent_model.record_human_fold_signal()

            child_util = (
                self.cfr(next_state, reach_hero * strategy[a], reach_opponent)
                if acting == HERO
                else self.cfr(next_state, reach_hero, reach_opponent * strategy[a])
            )
            action_utils[a] = child_util
            node_util += strategy[a] * child_util

        # Counterfactual regret update
        cfr_reach = reach_opponent if acting == HERO else reach_hero
        for a in legal:
            node.regret_sum[a] += cfr_reach * (action_utils[a] - node_util)

        return node_util

    # ── Training ──────────────────────────────────────────────────

    def train(
        self,
        iterations: int = 10_000,
        log_every: Optional[int] = None,
        callback: Optional[Callable[[int, float, float], None]] = None,
    ) -> List[float]:
        """
        Run *iterations* of CFR self-play.

        Parameters
        ----------
        iterations : int
            Training iterations.
        log_every : int | None
            Print progress every *n* iterations (default: 10 times).
        callback : callable | None
            ``callback(t, mean_regret, exploitability)`` called each
            iteration.  Useful for external progress bars.

        Returns
        -------
        List[float]
            Mean-absolute-regret history, normalised by T.
        """
        if log_every is None:
            log_every = max(1, iterations // 10)

        cumulative = 0.0
        self.regret_history.clear()
        self.exploitability_history.clear()

        for t in range(1, iterations + 1):
            state = self.env.new_game()
            util = self.cfr(state, 1.0, 1.0)
            cumulative += abs(util)

            mean_regret = cumulative / t
            self.regret_history.append(mean_regret)
            self._iterations_run = t

            # ε upper bound (sum of max positive regrets / T)
            eps = sum(
                np.maximum(n.regret_sum, 0).max()
                for n in self.info_set_map.values()
            ) / t
            self.exploitability_history.append(eps)

            if callback is not None:
                callback(t, mean_regret, eps)

            if t % log_every == 0:
                logger.info(
                    "T=%7d  Mean|R|=%10.4f  ε≤%10.4f  OppFold=%5.1f%%  "
                    "InfoSets=%d",
                    t, mean_regret, eps,
                    self.opponent_model.fold_to_cbet * 100,
                    len(self.info_set_map),
                )
                print(
                    f"  T={t:>7,d}  "
                    f"Mean|R|={mean_regret:>10.4f}  "
                    f"ε≤{eps:>10.4f}  "
                    f"OppFold={self.opponent_model.fold_to_cbet:>5.1%}  "
                    f"InfoSets={len(self.info_set_map)}"
                )

        return self.regret_history

    # ── Exploitative adjustment ───────────────────────────────────

    def exploit_weakness(
        self,
        mdf_threshold: float = 0.50,
        bluff_percentile: float = 0.20,
    ) -> Dict[str, Any]:
        """
        If Opponent over-folds beyond *mdf_threshold*, shift the bottom
        *bluff_percentile* of Hero's range to pure All-In.

        Returns a structured report dict.
        """
        opp_fold = self.opponent_model.fold_to_cbet
        adj_fold = self.opponent_model.adjusted_fold_rate
        exploitable = opp_fold > mdf_threshold

        report: Dict[str, Any] = {
            "opponent_fold_frequency": round(opp_fold, 4),
            "adjusted_fold_rate": round(adj_fold, 4),
            "mdf_threshold": mdf_threshold,
            "classification": "OVER-FOLDING" if exploitable else "BALANCED",
            "adjustment_applied": False,
            "info_sets_adjusted": 0,
            "human_fold_signals": self.opponent_model.human_fold_signals_detected,
        }

        if not exploitable:
            print(
                f"  Opponent fold rate ({opp_fold:.1%}) ≤ threshold "
                f"({mdf_threshold:.1%}). No adjustment."
            )
            return report

        print(f"  Opponent OVER-FOLDING: {opp_fold:.1%} > {mdf_threshold:.1%}")
        if self.opponent_model.human_fold_signals_detected > 0:
            print(
                f"  ⚠ {self.opponent_model.human_fold_signals_detected} "
                f"human-fold signal(s) detected — raw fold rate may be "
                f"inflated.  Adjusted: {adj_fold:.1%}"
            )

        bucket_cutoff = int(bluff_percentile * self.env.hand_buckets)
        all_in_idx = ACTION_TO_IDX["ALL_IN"]
        count = 0

        for key, node in self.info_set_map.items():
            if not key.startswith("P0|"):
                continue
            try:
                bucket = int(key.split("|")[1].replace("HS", ""))
            except (IndexError, ValueError):
                continue
            if bucket >= bucket_cutoff:
                continue
            node.strategy_sum[:] = 0.0
            node.strategy_sum[all_in_idx] = 1.0
            node.regret_sum[:] = 0.0
            node.regret_sum[all_in_idx] = 1.0
            count += 1

        report["adjustment_applied"] = True
        report["info_sets_adjusted"] = count
        print(f"  → {count} info set(s) shifted to pure All-In.")
        return report

    # ── Reporting ─────────────────────────────────────────────────

    def get_average_strategies(self) -> Dict[str, np.ndarray]:
        """Return the averaged strategy for every discovered info set."""
        return {k: n.average_strategy() for k, n in self.info_set_map.items()}

    def summary(self) -> Dict[str, Any]:
        """Structured summary suitable for JSON serialisation."""
        return {
            "iterations": self._iterations_run,
            "info_sets": len(self.info_set_map),
            "final_mean_regret": (
                self.regret_history[-1] if self.regret_history else None
            ),
            "final_exploitability_bound": (
                self.exploitability_history[-1]
                if self.exploitability_history else None
            ),
            "cfr_plus": self.use_cfr_plus,
            "opponent": self.opponent_model.to_dict(),
            "env": {
                "pot": self.env.initial_pot,
                "stack": self.env.initial_stack,
                "spr": round(self.env.spr, 2),
                "board_texture": self.env.board_texture.name,
                "human_fold_enabled": self.env.enable_human_fold,
                "human_folds_fired": self.env.human_folds_fired,
            },
        }

    def print_summary(self) -> None:
        """Pretty-print session report to stdout."""
        s = self.summary()
        e, o = s["env"], s["opponent"]
        print()
        print("═" * 60)
        print("  CFR SOLVER — SESSION REPORT")
        print("═" * 60)
        print(f"  Iterations           : {s['iterations']:>10,d}")
        print(f"  Information Sets     : {s['info_sets']:>10,d}")
        print(f"  CFR+ Enabled         : {s['cfr_plus']}")
        print(f"  Final Mean |Regret|  : {s['final_mean_regret']:>10.4f}")
        print(f"  Exploitability ≤     : {s['final_exploitability_bound']:>10.4f}")
        print("─" * 60)
        print("  ENVIRONMENT")
        print(f"  Pot / Stack / SPR    : {e['pot']} / {e['stack']} / {e['spr']}")
        print(f"  Board Texture        : {e['board_texture']}")
        print(f"  Human-Fold Enabled   : {e['human_fold_enabled']}")
        if e["human_fold_enabled"]:
            print(f"  Human-Folds Fired    : {e['human_folds_fired']}")
        print("─" * 60)
        print("  OPPONENT MODEL")
        print(f"  Fold-to-CBet         : {o['fold_to_cbet']:.1%}")
        print(f"  Adjusted Fold Rate   : {o['adjusted_fold_rate']:.1%}")
        print(f"  Human-Fold Signals   : {o['human_fold_signals']}")
        print("═" * 60)
