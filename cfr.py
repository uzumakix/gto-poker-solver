"""
cfr.py -- Counterfactual Regret Minimization (CFR+) for Kuhn Poker
===================================================================

Implements vanilla CFR and CFR+ (Tammelin 2014). All values in the
recursion are from player 0's perspective. Player 0 maximizes,
player 1 minimizes. Regret signs are flipped for player 1.

The average strategy profile converges to a Nash equilibrium as
iterations grow. For Kuhn Poker the analytical equilibrium is known
(Kuhn 1950), so we can verify convergence directly.

References:
    Zinkevich, M. et al. (2007). "Regret Minimization in Games with
        Incomplete Information." NeurIPS.
    Tammelin, O. (2014). "Solving Large Imperfect Information Games
        Using CFR+." arXiv:1407.5042.
"""

import numpy as np

from kuhn import (
    ALL_DEALS,
    CARD_NAMES,
    CARDS,
    NUM_ACTIONS,
    ACTION_NAMES,
    info_set_key,
)

# terminal histories and payoffs from P0's perspective
TERMINALS = {
    "pp":  None,   # showdown for 1
    "bp":  1,      # P1 folded
    "bb":  None,   # showdown for 2
    "pbp": -1,     # P0 folded
    "pbb": None,   # showdown for 2
}


def terminal_payoff_p0(cards, history):
    """Return payoff from player 0's perspective (always)."""
    if history == "bp":
        return 1
    if history == "pbp":
        return -1
    # showdown
    stake = 1 if history == "pp" else 2
    if cards[0] > cards[1]:
        return stake
    else:
        return -stake


class Node:
    """Information set node storing cumulative regrets and strategy sums."""

    def __init__(self):
        self.regret_sum = np.zeros(NUM_ACTIONS, dtype=np.float64)
        self.strategy_sum = np.zeros(NUM_ACTIONS, dtype=np.float64)

    def get_strategy(self, realization_weight):
        """
        Regret-matching: convert positive regrets to a probability
        distribution. Accumulate into strategy_sum weighted by the
        player's own reach probability.
        """
        positive = np.maximum(self.regret_sum, 0.0)
        total = positive.sum()
        if total > 0:
            strategy = positive / total
        else:
            strategy = np.ones(NUM_ACTIONS) / NUM_ACTIONS
        self.strategy_sum += realization_weight * strategy
        return strategy

    def get_average_strategy(self):
        """Time-averaged strategy (converges to Nash)."""
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        return np.ones(NUM_ACTIONS) / NUM_ACTIONS


class CFRSolver:
    """
    CFR+ solver for Kuhn Poker.

    Parameters
    ----------
    use_cfr_plus : bool
        If True, floor cumulative regrets to zero after each update
        (CFR+ variant). Default True.
    """

    def __init__(self, use_cfr_plus=True):
        self.nodes = {}
        self.use_cfr_plus = use_cfr_plus

    def _get_node(self, key):
        if key not in self.nodes:
            self.nodes[key] = Node()
        return self.nodes[key]

    def _cfr(self, cards, history, p0, p1):
        """
        Recursive CFR traversal.

        All return values are from player 0's perspective.
        Player 0 maximizes, player 1 minimizes.
        """
        if history in TERMINALS:
            return terminal_payoff_p0(cards, history)

        plays = len(history)
        player = plays % 2

        card = cards[player]
        key = info_set_key(card, history)
        node = self._get_node(key)

        reach = p0 if player == 0 else p1
        strategy = node.get_strategy(reach)

        util = np.zeros(NUM_ACTIONS, dtype=np.float64)
        node_util = 0.0

        for a in range(NUM_ACTIONS):
            next_history = history + ACTION_NAMES[a]
            if player == 0:
                util[a] = self._cfr(cards, next_history,
                                    p0 * strategy[a], p1)
            else:
                util[a] = self._cfr(cards, next_history,
                                    p0, p1 * strategy[a])
            node_util += strategy[a] * util[a]

        # counterfactual regret update
        # for player 0: regret = util[a] - node_util (wants to maximize)
        # for player 1: regret = node_util - util[a] (wants to minimize p0 value)
        opp_reach = p1 if player == 0 else p0
        for a in range(NUM_ACTIONS):
            if player == 0:
                regret = util[a] - node_util
            else:
                regret = node_util - util[a]
            node.regret_sum[a] += opp_reach * regret

        # CFR+: floor regrets to zero
        if self.use_cfr_plus:
            np.maximum(node.regret_sum, 0.0, out=node.regret_sum)

        return node_util

    def train(self, iterations):
        """
        Run CFR for the given number of iterations.
        Each iteration traverses all 6 possible card deals.

        Returns list of (iteration, exploitability) checkpoints.
        """
        history_log = []

        for t in range(1, iterations + 1):
            for cards in ALL_DEALS:
                self._cfr(list(cards), "", 1.0, 1.0)

            if t % max(1, iterations // 200) == 0 or t == iterations:
                eps = self.exploitability()
                history_log.append((t, eps))

        return history_log

    def exploitability(self):
        """
        Compute exploitability of the current average strategy.

        Uses information-set-level best response: the BR player
        must choose one action per info set (same action for all
        deals sharing that info set). Enumerates all pure BR
        strategies (feasible since Kuhn has few info sets).

        Exploitability = (BR_value_p0 - BR_value_p1) / 2
        At Nash equilibrium this equals zero.
        """
        from itertools import product

        avg = {k: n.get_average_strategy() for k, n in self.nodes.items()}

        def _br_value(br_player):
            # collect info sets belonging to br_player
            br_keys = []
            for key in self.nodes:
                parts = key.split()
                history = parts[1] if len(parts) > 1 else ""
                if len(history) % 2 == br_player:
                    br_keys.append(key)

            best_val = None
            for actions in product(range(NUM_ACTIONS), repeat=len(br_keys)):
                policy = {}
                for i, key in enumerate(br_keys):
                    p = np.zeros(NUM_ACTIONS)
                    p[actions[i]] = 1.0
                    policy[key] = p

                total = 0.0
                for cards in ALL_DEALS:
                    total += self._eval_with_policy(
                        list(cards), "", br_player, policy, avg,
                    )
                val = total / len(ALL_DEALS)

                if best_val is None:
                    best_val = val
                elif br_player == 0 and val > best_val:
                    best_val = val
                elif br_player == 1 and val < best_val:
                    best_val = val

            return best_val

        br0 = _br_value(0)
        br1 = _br_value(1)
        return (br0 - br1) / 2.0

    def _eval_with_policy(self, cards, history, br_player,
                          br_policy, avg_strategies):
        """Evaluate P0's expected value under a specific BR policy."""
        if history in TERMINALS:
            return terminal_payoff_p0(cards, history)

        plays = len(history)
        player = plays % 2
        card = cards[player]
        key = info_set_key(card, history)

        if player == br_player:
            strategy = br_policy.get(key, np.ones(NUM_ACTIONS) / NUM_ACTIONS)
        else:
            strategy = avg_strategies.get(
                key, np.ones(NUM_ACTIONS) / NUM_ACTIONS
            )

        val = 0.0
        for a in range(NUM_ACTIONS):
            next_h = history + ACTION_NAMES[a]
            val += strategy[a] * self._eval_with_policy(
                cards, next_h, br_player, br_policy, avg_strategies,
            )
        return val

    def get_strategy_table(self):
        """
        Return a dict mapping info set keys to average strategy arrays.
        Sorted by card rank and history length.
        """
        result = {}
        for key in sorted(self.nodes.keys(),
                          key=lambda k: (len(k), k)):
            node = self.nodes[key]
            avg = node.get_average_strategy()
            result[key] = avg
        return result

    def game_value(self):
        """
        Expected value of the game for player 0 under the current
        average strategy profile. Nash value for Kuhn = -1/18.
        """
        avg = {k: n.get_average_strategy() for k, n in self.nodes.items()}
        total = 0.0
        for cards in ALL_DEALS:
            total += self._expected_value(list(cards), "", avg)
        return total / len(ALL_DEALS)

    def _expected_value(self, cards, history, avg_strategies):
        """Expected value from player 0's perspective under avg strategies."""
        if history in TERMINALS:
            return terminal_payoff_p0(cards, history)

        plays = len(history)
        player = plays % 2
        card = cards[player]
        key = info_set_key(card, history)

        if key in avg_strategies:
            strategy = avg_strategies[key]
        else:
            strategy = np.ones(NUM_ACTIONS) / NUM_ACTIONS

        val = 0.0
        for a in range(NUM_ACTIONS):
            next_h = history + ACTION_NAMES[a]
            val += strategy[a] * self._expected_value(
                cards, next_h, avg_strategies
            )
        return val
