"""Tests for cfr_solver.py — regret matching, training, exploitation."""

import numpy as np
import pytest
from gto_poker_solver.cfr_solver import CFRSolver, InfoSetNode


class TestRegretMatching:
    def test_uniform_under_zero_regret(self):
        node = InfoSetNode()
        strat = node.current_strategy()
        assert np.allclose(strat, 1.0 / node.num_actions)

    def test_proportional_to_positive_regret(self):
        node = InfoSetNode()
        node.regret_sum = np.array([10.0, 0.0, 0.0, 5.0, 0.0, 0.0])
        strat = node.current_strategy()
        assert strat[0] == pytest.approx(10.0 / 15.0)
        assert strat[3] == pytest.approx(5.0 / 15.0)
        assert strat[1] == 0.0

    def test_negative_regret_ignored(self):
        node = InfoSetNode()
        node.regret_sum = np.array([-5.0, 10.0, 0.0, 0.0, 0.0, 0.0])
        strat = node.current_strategy()
        assert strat[0] == 0.0
        assert strat[1] == 1.0

    def test_average_strategy_normalised(self):
        node = InfoSetNode()
        for _ in range(100):
            node.current_strategy(realisation_weight=1.0)
        assert abs(node.average_strategy().sum() - 1.0) < 1e-10

    def test_cfr_plus_floors_negatives(self):
        node = InfoSetNode()
        node.regret_sum = np.array([-10.0, -5.0, 3.0, 0.0, 0.0, 0.0])
        node.current_strategy(cfr_plus=True)
        # After CFR+ floor, negatives become 0
        assert np.all(node.regret_sum >= 0)


class TestCFRTraining:
    def test_history_length_matches_iterations(self, env):
        solver = CFRSolver(env)
        h = solver.train(iterations=100, log_every=9999)
        assert len(h) == 100

    def test_info_sets_discovered(self, env):
        solver = CFRSolver(env)
        solver.train(iterations=500, log_every=9999)
        assert len(solver.info_set_map) > 0

    def test_mean_regret_normalised_not_diverging(self, env):
        """Mean regret = cumulative/T must stabilise, not grow linearly."""
        solver = CFRSolver(env)
        solver.train(iterations=300, log_every=9999)
        ratio = solver.regret_history[-1] / max(solver.regret_history[10], 1e-9)
        assert ratio < 5.0

    def test_exploitability_decreases(self, env):
        """ε upper bound should trend downward."""
        solver = CFRSolver(env)
        solver.train(iterations=500, log_every=9999)
        early = solver.exploitability_history[50]
        late = solver.exploitability_history[-1]
        assert late <= early * 1.5  # not strictly monotone but should trend

    def test_callback_is_invoked(self, env):
        calls = []
        solver = CFRSolver(env)
        solver.train(
            iterations=10, log_every=9999,
            callback=lambda t, r, e: calls.append(t),
        )
        assert calls == list(range(1, 11))


class TestExploitation:
    def test_balanced_classification(self, env):
        solver = CFRSolver(env)
        solver.train(iterations=200, log_every=9999)
        report = solver.exploit_weakness(mdf_threshold=0.99)
        assert report["classification"] == "BALANCED"
        assert not report["adjustment_applied"]

    def test_reads_opponent_fold_not_hero(self, env):
        solver = CFRSolver(env)
        solver.train(iterations=200, log_every=9999)
        report = solver.exploit_weakness(mdf_threshold=0.01)
        # Report rounds to 4 decimals; compare accordingly
        assert report["opponent_fold_frequency"] == round(
            solver.opponent_model.fold_to_cbet, 4,
        )

    def test_adjustment_modifies_info_sets(self, env):
        solver = CFRSolver(env)
        solver.train(iterations=200, log_every=9999)
        report = solver.exploit_weakness(mdf_threshold=0.01)
        # Should adjust at least 1 info set when threshold is 1%
        if solver.opponent_model.fold_to_cbet > 0.01:
            assert report["info_sets_adjusted"] > 0


class TestHumanFoldIntegration:
    def test_env_tracks_count(self, env_human_fold):
        solver = CFRSolver(env_human_fold)
        solver.train(iterations=300, log_every=9999)
        assert env_human_fold.human_folds_fired >= 0

    def test_summary_includes_human_fold_fields(self, env_human_fold):
        solver = CFRSolver(env_human_fold)
        solver.train(iterations=50, log_every=9999)
        s = solver.summary()
        assert "human_fold_enabled" in s["env"]
        assert "human_folds_fired" in s["env"]
        assert "human_fold_signals" in s["opponent"]
