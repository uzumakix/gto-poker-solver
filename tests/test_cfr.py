import pytest
import numpy as np
import sys
sys.path.insert(0, '.')

from cfr import CFRSolver, Node


class TestGameValue:
    def test_converges_to_negative_one_eighteenth(self, trained_solver):
        gv = trained_solver.game_value()
        assert abs(gv - (-1/18)) < 0.01, f"game value {gv} not close to -1/18"


class TestExploitability:
    def test_low_exploitability(self, trained_solver):
        exploit = trained_solver.exploitability()
        assert exploit < 0.01, f"exploitability {exploit} too high after 10k iters"


class TestEquilibriumStrategies:
    def test_king_always_bets(self, trained_solver):
        table = trained_solver.get_strategy_table()
        bet_prob = table["K"][1]
        assert bet_prob > 0.65, f"King bet prob {bet_prob} too low"

    def test_jack_passes_initially(self, trained_solver):
        table = trained_solver.get_strategy_table()
        pass_prob = table["J"][0]
        assert pass_prob > 0.65, f"Jack pass prob {pass_prob} too low"


class TestStrategyValidity:
    def test_all_strategies_sum_to_one(self, trained_solver):
        table = trained_solver.get_strategy_table()
        for key, strat in table.items():
            total = strat.sum()
            assert abs(total - 1.0) < 1e-6, f"strategy for {key} sums to {total}"

    def test_node_average_strategy_sums_to_one(self):
        node = Node()
        node.strategy_sum = np.array([3.0, 7.0])
        avg = node.get_average_strategy()
        assert abs(avg.sum() - 1.0) < 1e-9

    def test_node_uniform_when_untrained(self):
        node = Node()
        avg = node.get_average_strategy()
        assert abs(avg[0] - 0.5) < 1e-9
        assert abs(avg[1] - 0.5) < 1e-9


class TestSolverConvergence:
    def test_quick_solver_has_nodes(self, quick_solver):
        assert len(quick_solver.nodes) > 0

    def test_more_iters_lower_exploitability(self, quick_solver, trained_solver):
        assert trained_solver.exploitability() <= quick_solver.exploitability() + 0.05
