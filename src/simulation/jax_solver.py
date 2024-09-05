"""Solver for differential equations using JAX via the diffrax library."""

from typing import Callable, List

from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt
import jax.numpy as jnp

from simulation.solver import DiffeqSolver


class JaxSolver(DiffeqSolver):
    """
    Thin wrapper around JAX differential equation solvers,
    via the diffrax library.
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
        Solver for an initial-value problem using diffrax's diffeqsolve.
        """
        t_saveat = self.calculate_t_saveat(ti, tf, dt)
        term = ODETerm(func)
        solver = Dopri5()
        sol = diffeqsolve(
            term,
            solver,
            t0=ti,
            t1=tf,
            dt0=0.01,
            y0=ic,
            saveat=SaveAt(ts=t_saveat),
            args=args,
        )
        return sol

    def solve_bvp(self, func: Callable, bc, x): ...

    def calculate_t_saveat(self, ti: float, tf: float, dt: float):
        """
        Calculates the SaveAt values for the independent axis.
        """
        return jnp.arange(ti, tf, dt)
