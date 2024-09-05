"""Tests for the Jax solver."""

import sys
from pathlib import Path

import jax.numpy as np
import pytest

SRC_PATH = Path(__file__).parent.parent.parent / "src"
sys.path.append(SRC_PATH)
from simulation.jax_solver import JaxSolver


def diffeq_logistic_growth(t, z, args):
    """
    Example diffeq for logistic growth.
    """
    a_x, b_x, a_y, b_y = args
    x, y = z
    dxdt = a_x * (b_x - x)
    dydt = a_y * (b_y - y)
    return [dxdt, dydt]


def test_jax_solver_logistic_growth_happy_path():
    """
    Happy-path case for solving the logistic growth
    population model with PyTorch's torchdiffeq.
    """
    a_x = 0.1
    a_y = 0.2
    b_x = 100
    b_y = 300

    uut_solver = JaxSolver()
    sol = uut_solver.solve_ivp(
        func=diffeq_logistic_growth,
        ic=[1.0, 2.0],
        ti=0.0,
        tf=10.0,
        dt=0.1,
        args=(a_x, b_x, a_y, b_y),
    )
    assert sol.ts.size == 100
    assert len(sol.ys) == 2
    assert sol.ys[0].size == 100
    assert sol.ys[1].size == 100
