"""Tests for poker_env.py — environment correctness."""

import numpy as np
import pytest
from gto_poker_solver.poker_env import (
    ACTION_TO_IDX,
    ACTIONS,
    HERO,
    OPPONENT,
    BoardTexture,
    HandStrengthModel,
    HumanFoldSignal,
    RiverEndgameEnv,
)


class TestZeroSumPayoff:
    """Every terminal node must satisfy hero + opponent ≡ 0."""

    def test_hero_folds(self, env):
        state = env.new_game()
        state = env.apply_action(state, ACTION_TO_IDX["FOLD"])
        h, o = env.terminal_payoff(state)
        assert h + o == 0.0
        assert h == -50.0  # lost pot/2

    def test_opponent_folds(self, env):
        state = env.new_game()
        state = env.apply_action(state, ACTION_TO_IDX["BET_HALF"])
        state = env.apply_action(state, ACTION_TO_IDX["FOLD"])
        h, o = env.terminal_payoff(state)
        assert h + o == 0.0
        assert h > 0

    def test_showdown(self, env):
        state = env.new_game()
        state = env.apply_action(state, ACTION_TO_IDX["CHECK"])
        state = env.apply_action(state, ACTION_TO_IDX["CHECK"])
        h, o = env.terminal_payoff(state)
        assert h + o == 0.0

    def test_fold_payoff_is_half_pot_not_full(self, env):
        """Critical fix: fold must yield ±pot/2, never ±pot."""
        state = env.new_game()
        state = env.apply_action(state, ACTION_TO_IDX["FOLD"])
        h, _ = env.terminal_payoff(state)
        assert abs(h) == state.pot / 2.0

    def test_call_resolves_to_showdown(self, env):
        """Bet → Call should be terminal (showdown)."""
        state = env.new_game()
        state = env.apply_action(state, ACTION_TO_IDX["BET_HALF"])
        state = env.apply_action(state, ACTION_TO_IDX["CALL"])
        assert state.is_terminal
        h, o = env.terminal_payoff(state)
        assert h + o == 0.0

    def test_non_terminal_raises_error(self, env):
        """Calling terminal_payoff on a non-terminal must raise."""
        state = env.new_game()
        with pytest.raises(ValueError, match="non-terminal"):
            env.terminal_payoff(state)


class TestLegalActions:
    """Action masking must be correct."""

    def test_opening_actions_include_check(self, env):
        state = env.new_game()
        legal = env.legal_actions(state)
        assert ACTION_TO_IDX["CHECK"] in legal
        assert ACTION_TO_IDX["FOLD"] not in legal

    def test_facing_bet_includes_fold_and_call(self, env):
        state = env.new_game()
        state = env.apply_action(state, ACTION_TO_IDX["BET_HALF"])
        legal = env.legal_actions(state)
        assert ACTION_TO_IDX["FOLD"] in legal
        assert ACTION_TO_IDX["CALL"] in legal
        assert ACTION_TO_IDX["CHECK"] not in legal


class TestHandStrength:
    """Beta-distributed hand strength (not Lennard-Jones)."""

    def test_in_unit_interval(self):
        model = HandStrengthModel(board_texture=BoardTexture.DRY)
        s = model.sample(10_000)
        assert np.all(s >= 0) and np.all(s <= 1)

    def test_dry_polarised(self):
        model = HandStrengthModel(board_texture=BoardTexture.DRY)
        s = model.sample(10_000)
        edges = ((s < 0.2) | (s > 0.8)).mean()
        middle = ((s > 0.4) & (s < 0.6)).mean()
        assert edges > middle

    def test_wet_merged(self):
        model = HandStrengthModel(board_texture=BoardTexture.WET)
        s = model.sample(10_000)
        assert ((s > 0.3) & (s < 0.7)).mean() > 0.5

    def test_neutral_exists(self):
        model = HandStrengthModel(board_texture=BoardTexture.NEUTRAL)
        s = model.sample(100)
        assert len(s) == 100


class TestHumanFoldSignal:
    def test_rejects_weak_hands(self):
        sig = HumanFoldSignal(hand_strength_floor=0.65, activation_probability=1.0)
        assert not sig.should_signal(0.30, 100, 10)

    def test_accepts_strong_hands(self):
        sig = HumanFoldSignal(hand_strength_floor=0.65, activation_probability=1.0)
        assert sig.should_signal(0.80, 100, 10)

    def test_rejects_high_sacrifice(self):
        sig = HumanFoldSignal(sacrifice_ev_cap=0.35, activation_probability=1.0)
        assert not sig.should_signal(0.90, 100, 50)  # 50/100 > 0.35

    def test_stochastic_activation(self):
        sig = HumanFoldSignal(activation_probability=0.0)
        # With p=0 it should never fire
        assert not sig.should_signal(0.99, 100, 0)
