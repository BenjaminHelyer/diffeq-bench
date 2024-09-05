"""Tests for the PyTorch solver."""

import sys
import os

import torch
import pytest

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(SRC_PATH)
from simulation.pytorch_solver import PyTorchSolver


def diffeq_logistic_growth(t, z, args):
    """
    Example diffeq for logistic growth.
    """
    a_x, b_x, a_y, b_y = args
    x, y = z
    dxdt = a_x * (b_x - x)
    dydt = a_y * (b_y - y)
    return [dxdt, dydt]


def test_pytorch_solver_logistic_growth_happy_path():
    """
    Happy-path case for solving the logistic growth
    population model with PyTorch's torchdiffeq.
    """
    a_x = 0.1
    a_y = 0.2
    b_x = 100
    b_y = 300

    uut_solver = PyTorchSolver()
    sol = uut_solver.solve_ivp(
        func=diffeq_logistic_growth,
        ic=[1.0, 2.0],
        ti=0.0,
        tf=10.0,
        dt=0.1,
        args=(a_x, b_x, a_y, b_y),
    )
    assert sol.size() == torch.Size([100, 2])
