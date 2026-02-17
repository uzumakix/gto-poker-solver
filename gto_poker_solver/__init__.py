"""
gto-poker-solver — CFR Nash equilibrium solver for NLHE river endgames.

Public API::

    from gto_poker_solver import CFRSolver, RiverEndgameEnv, BoardTexture

    env = RiverEndgameEnv(pot=100, stack=200, board_texture=BoardTexture.DRY)
    solver = CFRSolver(env)
    solver.train(iterations=10_000)
"""

__version__ = "2.1.0"

from .poker_env import BoardTexture, RiverEndgameEnv
from .cfr_solver import CFRSolver

__all__ = ["BoardTexture", "RiverEndgameEnv", "CFRSolver", "__version__"]
