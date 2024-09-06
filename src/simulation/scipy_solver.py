"""Solver for differential equations using SciPy."""

from typing import Callable, List

import numpy as np
from scipy.integrate import solve_ivp as scipy_solve_ivp

from simulation.solver import DiffeqSolver


class SciPySolver(DiffeqSolver):
    """
    Thin wrapper around SciPy differential equation solvers.
    """

    def solve_ivp(
        self,
        func: Callable,
        ic: List[float],
        ti: float,
        tf: float,
        dt: float,
        args: List[float] = None,
    ):
        """
        Solver for an initial-value problem using SciPy's solve_ivp.
        """
        t_eval = self.calculate_t_eval(ti, tf, dt)
        sol = scipy_solve_ivp(
            fun=func,
            t_span=[ti, tf],
            y0=ic,
            t_eval=t_eval,
            args=[args],  # have to pass a list here since it tries to unpack the args
            method='RK45',
        )
        return sol

    def solve_bvp(self, func: Callable, bc, x):
        """
        Solver for boundary-value problem using SciPy's solve_bvp.
        """
        ...

    def calculate_t_eval(
        self,
        ti: float,
        tf: float,
        dt: float,
    ) -> List[float]:
        """
        Calculates the time eval points given initial, final, and delta time values.
        """
        return np.arange(
            ti, tf, dt
        )  # this will exclude tf in some cases but is probably OK for now
