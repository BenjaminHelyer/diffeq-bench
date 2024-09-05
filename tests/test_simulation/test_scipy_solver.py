"""Tests for the SciPySolver class."""

import sys
import os

import pytest

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(SRC_PATH)
from simulation.scipy_solver import SciPySolver


def diffeq_logistic_growth(t, z, args):
    """
    Example diffeq for logistic growth.
    """
    a_x, b_x, a_y, b_y = args
    x, y = z

    dxdt = a_x * (b_x - x)
    dydt = a_y * (b_y - y)

    return [dxdt, dydt]


def test_scipy_solver_logistic_growth_happy_path():
    """
    Happy-path case for solving the logistic growth
    population model with SciPy.
    """
    a_x = 0.1
    a_y = 0.2
    b_x = 100
    b_y = 300

    uut_solver = SciPySolver()
    sol = uut_solver.solve_ivp(
        func=diffeq_logistic_growth,
        ic=[1.0, 2.0],
        ti=0.0,
        tf=10.0,
        dt=0.1,
        args=(a_x, b_x, a_y, b_y),
    )

    assert sol.success == True
