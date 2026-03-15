"""
kuhn.py -- Kuhn Poker game definition
======================================

Three-card poker variant introduced by Harold Kuhn (1950).
Cards: Jack (0), Queen (1), King (2). Two players each ante 1 chip,
receive one card, then alternate between pass (check/fold) and
bet (bet/call) actions.

Terminal histories and payoffs (from current player's perspective):
    pp   -> showdown, winner takes pot (+-1)
    bp   -> player 1 folds, player 0 wins (+1 for current)
    bb   -> showdown, winner takes pot (+-2)
    pbp  -> player 0 folds, player 1 wins (+1 for current)
    pbb  -> showdown, winner takes pot (+-2)

Reference: Kuhn, H. W. (1950). "Simplified Two-Person Poker."
"""

import itertools

CARDS = [0, 1, 2]  # J=0, Q=1, K=2
CARD_NAMES = {0: "J", 1: "Q", 2: "K"}
NUM_ACTIONS = 2  # 0=pass, 1=bet
ACTION_NAMES = ["p", "b"]

# all 6 possible deals (order matters: [p0_card, p1_card])
ALL_DEALS = list(itertools.permutations(CARDS, 2))

TERMINAL_HISTORIES = {"pp", "bp", "bb", "pbp", "pbb"}


def is_terminal(history):
    return history in TERMINAL_HISTORIES


def terminal_payoff(cards, history):
    """
    Return payoff from the CURRENT player's perspective.
    Current player = len(history) % 2.
    """
    plays = len(history)
    player = plays % 2
    opponent = 1 - player
    player_higher = cards[player] > cards[opponent]

    if history[-1] == "p":
        if history == "pp":
            return 1 if player_higher else -1
        else:
            # opponent folded (bp or pbp)
            return 1
    else:
        # double bet (bb or pbb) -> showdown for 2
        return 2 if player_higher else -2


def info_set_key(card, history):
    """Unique information set string: card name + action history."""
    return CARD_NAMES[card] + " " + history if history else CARD_NAMES[card]
