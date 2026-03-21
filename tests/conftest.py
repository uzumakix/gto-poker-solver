import pytest
import sys
sys.path.insert(0, '.')
from cfr import CFRSolver

@pytest.fixture
def trained_solver():
    solver = CFRSolver(use_cfr_plus=True)
    solver.train(10000)
    return solver

@pytest.fixture
def quick_solver():
    solver = CFRSolver(use_cfr_plus=True)
    solver.train(1000)
    return solver
