"""Shared fixtures for the test suite."""

import pytest
from gto_poker_solver.poker_env import RiverEndgameEnv


@pytest.fixture
def env():
    """Standard environment: pot=100, stack=200, dry board."""
    return RiverEndgameEnv(pot=100.0, stack=200.0, seed=42)


@pytest.fixture
def env_human_fold():
    """Environment with human-fold-as-signal enabled."""
    return RiverEndgameEnv(
        pot=100.0, stack=200.0, enable_human_fold=True, seed=42,
    )
