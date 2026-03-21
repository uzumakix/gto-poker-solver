import sys
sys.path.insert(0, '.')

from kuhn import is_terminal, terminal_payoff_p0, info_set_key, ALL_DEALS


class TestIsTerminal:
    def test_terminal_histories(self):
        for h in ["pp", "bp", "bb", "pbp", "pbb"]:
            assert is_terminal(h), f"{h} should be terminal"

    def test_nonterminal_histories(self):
        for h in ["", "p", "b", "pb"]:
            assert not is_terminal(h), f"{h} should not be terminal"


class TestTerminalPayoff:
    def test_bp_is_fold(self):
        assert terminal_payoff_p0([0, 1], "bp") == 1
        assert terminal_payoff_p0([2, 0], "bp") == 1

    def test_pbp_is_fold(self):
        assert terminal_payoff_p0([1, 0], "pbp") == -1
        assert terminal_payoff_p0([2, 1], "pbp") == -1

    def test_pp_showdown_higher_wins(self):
        assert terminal_payoff_p0([2, 0], "pp") == 1

    def test_pp_showdown_lower_loses(self):
        assert terminal_payoff_p0([0, 2], "pp") == -1

    def test_bb_showdown_higher_wins(self):
        assert terminal_payoff_p0([2, 1], "bb") == 2


class TestInfoSetKey:
    def test_king_empty_history(self):
        assert info_set_key(2, "") == "K"

    def test_jack_with_history(self):
        assert info_set_key(0, "pb") == "J pb"


class TestAllDeals:
    def test_six_unique_deals(self):
        assert len(ALL_DEALS) == 6
        as_tuples = [tuple(d) for d in ALL_DEALS]
        assert len(set(as_tuples)) == 6
