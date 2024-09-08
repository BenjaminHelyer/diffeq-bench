"""Tests for the Simulator class."""

import sys
import os

import torch

import pytest

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(SRC_PATH)
from simulation.simulator import Simulator


@pytest.fixture
def diffeq_logistic_growth():
    def _diffeq_logistic_growth(t, z, args):
        """
        Example diffeq for logistic growth.
        """
        a_x, b_x, a_y, b_y = args
        x, y = z
        dxdt = a_x * (b_x - x)
        dydt = a_y * (b_y - y)
        return [dxdt, dydt]

    return _diffeq_logistic_growth


@pytest.fixture
def logistic_growth_params():
    return (0.5, 200, 0.02, 100)


@pytest.mark.parametrize("backend_option", ["scipy", "pytorch", "jax"])
def test_simulator_logistic_growth_happy_path(
    backend_option, diffeq_logistic_growth, logistic_growth_params
):
    """
    Happy-path for running the Simulator object on a logistic
    growth model with various backends.
    """
    ic = [1.0, 2.0]
    ti = 0.0
    tf = 10.0
    dt = 0.1

    uut_solver = Simulator(
        backend=backend_option,
    )

    sol = uut_solver.generate_numeric_sol_ivp(
        diffeq_func=diffeq_logistic_growth,
        args=logistic_growth_params,
        ic=ic,
        ti=ti,
        tf=tf,
        dt=dt,
    )

    if backend_option == "scipy":
        assert sol.success is True
    elif backend_option == "pytorch":
        assert sol.size() == torch.Size([100, 2])
    elif backend_option == "jax":
        assert sol.ts.size == 100
        assert len(sol.ys) == 2
        assert sol.ys[0].size == 100
        assert sol.ys[1].size == 100


@pytest.mark.parametrize("backend_option", [("scipy"), ("pytorch"), ("jax")])
def test_simulator_cpu_sequential_solve_ics_logistic_growth(
    backend_option, diffeq_logistic_growth, logistic_growth_params
):
    """
    Tests the simulator's benchmarking capabilities
    for a benchmarking a set of initial conditions.

    This is the test for the sequential CPU benchmarks.
    """
    ics = [[1.0, 2.0], [2.0, 2.0], [1.0, 1.0]]

    uut_solver = Simulator(
        backend=backend_option,
    )
    uut_solver.cpu_sequential_solve_ics(
        diffeq_func=diffeq_logistic_growth,
        args=logistic_growth_params,
        ics=ics,
        ti=0.0,
        tf=10.0,
        dt=0.1,
    )
    assert len(uut_solver.sols) == 3
    for sol in uut_solver.sols:
        if backend_option == "scipy":
            assert sol.success == True
        elif backend_option == "pytorch":
            assert sol.size() == torch.Size([100, 2])
        elif backend_option == "jax":
            assert sol.ts.size == 100
            assert len(sol.ys) == 2
            assert sol.ys[0].size == 100
            assert sol.ys[1].size == 100


@pytest.mark.parametrize("backend_option", [("scipy"), ("pytorch"), ("jax")])
def test_simulator_cpu_parallel_solve_ics_logistic_growth(
    backend_option, diffeq_logistic_growth, logistic_growth_params
):
    """
    Tests the simulator's benchmarking capabilities
    for a benchmarking a set of initial conditions.

    This is the test for the parallel CPU benchmarks.
    """

    ics = [[1.0, 2.0], [2.0, 2.0], [1.0, 1.0]]

    uut_solver = Simulator(
        backend=backend_option,
    )
    uut_solver.cpu_parallel_solve_ics(
        diffeq_func=diffeq_logistic_growth,
        args=logistic_growth_params,
        ics=ics,
        ti=0.0,
        tf=10.0,
        dt=0.1,
        num_processes=10,
    )
    assert len(uut_solver.sols) == 3
    for sol in uut_solver.sols:
        if backend_option == "scipy":
            assert sol.success == True
        elif backend_option == "pytorch":
            assert sol.size() == torch.Size([100, 2])
        elif backend_option == "jax":
            assert sol.ts.size == 100
            assert len(sol.ys) == 2
            assert sol.ys[0].size == 100
            assert sol.ys[1].size == 100
