"""
kuhn.py -- Kuhn Poker game definition
======================================

Three-card poker variant introduced by Harold Kuhn (1950).
Cards: Jack (0), Queen (1), King (2). Two players each ante 1 chip,
receive one card, then alternate between pass (check/fold) and
bet (bet/call) actions.

Game tree (P0 acts at even depths, P1 at odd):

              (deal)
             /      \\
          pass       bet
          /            \\
      P1:pass      P1:pass    P1:bet
       "pp"         "bp"       "bb"
     showdown     P1 folds   showdown
       +-1          +1         +-2
                              (P0 view)
          \\
         P1:bet
          /    \\
      P0:pass  P0:bet
       "pbp"    "pbb"
     P0 folds  showdown
       -1        +-2
     (P0 view)

Terminal payoffs (from player 0's perspective):
    pp   -> showdown, higher card wins +-1
    bp   -> P1 folded, P0 wins +1
    bb   -> showdown, higher card wins +-2
    pbp  -> P0 folded, P0 loses -1
    pbb  -> showdown, higher card wins +-2

Reference: Kuhn, H. W. (1950). "Simplified Two-Person Poker."
    Contributions to the Theory of Games, 1, 97-103.
"""

import itertools

CARDS = [0, 1, 2]  # J=0, Q=1, K=2
CARD_NAMES = {0: "J", 1: "Q", 2: "K"}
NUM_ACTIONS = 2  # 0=pass, 1=bet
ACTION_NAMES = ["p", "b"]

# all 6 possible deals (order matters: [p0_card, p1_card])
ALL_DEALS = list(itertools.permutations(CARDS, 2))

TERMINAL_HISTORIES = frozenset({"pp", "bp", "bb", "pbp", "pbb"})


def is_terminal(history):
    """Check whether a history string represents a terminal game state."""
    return history in TERMINAL_HISTORIES


def terminal_payoff_p0(cards, history):
    """
    Payoff from player 0's perspective at a terminal node.

    Parameters
    ----------
    cards : list[int]
        cards[0] = P0's card, cards[1] = P1's card.
    history : str
        Action history (must be terminal).

    Returns
    -------
    int
        +1/-1 for ante-only showdowns and folds,
        +2/-2 for bet-call showdowns.
    """
    if history == "bp":
        return 1       # P1 folded after P0 bet
    if history == "pbp":
        return -1      # P0 folded after P1 bet
    # showdown
    stake = 1 if history == "pp" else 2
    return stake if cards[0] > cards[1] else -stake


def info_set_key(card, history):
    """
    Unique information set string: card name + action history.

    A player's information set consists of their private card and the
    public action sequence. Two game states are in the same info set
    if and only if the acting player cannot distinguish them.
    """
    return CARD_NAMES[card] + " " + history if history else CARD_NAMES[card]
