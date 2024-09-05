"""Tests for the Simulator class."""

import sys
import os

import torch

import pytest

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(SRC_PATH)
from simulation.simulator import Simulator


def diffeq_logistic_growth(t, z, args):
    """
    Example diffeq for logistic growth.
    """
    a_x, b_x, a_y, b_y = args
    x, y = z
    dxdt = a_x * (b_x - x)
    dydt = a_y * (b_y - y)
    return [dxdt, dydt]


@pytest.mark.parametrize("backend_option", [("scipy"), ("pytorch"), ("jax")])
def test_simulator_scipy_solver_logistic_growth_happy_path(backend_option):
    """
    Happy-path for running the Simulator object on a logistic
    growth model with various backends.
    """
    a_x = 0.5
    b_x = 200
    a_y = 0.02
    b_y = 100

    uut_solver = Simulator(
        backend=backend_option,
    )
    sol = uut_solver.generate_numeric_sol_ivp(
        diffeq_func=diffeq_logistic_growth,
        args=(0.5, 200, 0.02, 100),
        ic=[1.0, 2.0],
        ti=0.0,
        tf=10.0,
        dt=0.1,
    )

    if backend_option == "scipy":
        assert sol.success == True
    elif backend_option == "pytorch":
        assert sol.size() == torch.Size([100, 2])
    elif backend_option == "jax":
        assert sol.ts.size == 100
        assert len(sol.ys) == 2
        assert sol.ys[0].size == 100
        assert sol.ys[1].size == 100
