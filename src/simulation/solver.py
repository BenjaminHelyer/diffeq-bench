"""Abstract base class for a differential equation solver."""

from abc import ABC, abstractmethod
from typing import Callable, List


class DiffeqSolver(ABC):
    """
    Abstract base class for a differential equation solver.
    This is so we can lightly wrap various backend solvers
    using a common interface.
    """

    @abstractmethod
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
        Abstract method for solving an initial value problem.

        Parameters:
        - func: callable, the differential equation system function
        - ic: array-like, the initial condition(s) of the dependent variable(s)
        - ti, tf, dt: float, the initial, final, and deltas of the independent variable's axis
        - args (optional): array-like, arguments for func
        """
        pass

    @abstractmethod
    def solve_bvp(self, func: Callable, bc, x):
        """
        Abstract method for solving a boundary value problem.

        Parameters:
        - func: callable, the differential equation system function
        - bc: array-like, the boundary conditions
        - x: array-like, points of evaluation
        """
        pass
