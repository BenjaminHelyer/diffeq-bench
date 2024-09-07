"""Methods for generating data based on given differential equations."""

from typing import Callable, Union, Literal, List

import matplotlib.pyplot as plt

from simulation.scipy_solver import SciPySolver
from simulation.pytorch_solver import PyTorchSolver
from simulation.jax_solver import JaxSolver


class Simulator:
    """
    Light wrapper around high-level numeric differential equation solvers.
    Options include SciPy, PyTorch, and JAX back-ends.

    Includes capabilities such as plotting, performance testing,
    and other extensions to make benchmarking these solvers easier.
    """

    def __init__(
        self,
        backend: Union[Literal["scipy"], Literal["pytorch"], Literal["jax"]],
    ):
        """
        Initializes the simulator object.
        You must initiate a new Simulator for each backend solver
        that you wish to use.

        diffeq_func: function that returns the solution of the diffeq at specified points
        backend: literals specifying the backend to use, either one of SciPy, PyTorch, or JAX
        args (optional): array-like, arguments / coefficients for diffeq_func
        """
        self.backend = backend

        if self.backend == "scipy":
            self.solver = SciPySolver()
        elif self.backend == "pytorch":
            self.solver = PyTorchSolver()
        elif self.backend == "jax":
            self.solver = JaxSolver()
        else:
            raise NotImplementedError

        self.sols = []  # list of numeric solutions

    def generate_numeric_sol_ivp(
        self,
        diffeq_func: Callable,
        args: List[float],
        ic: List[float],
        ti: float,
        tf: float,
        dt: float,
    ):
        """
        Generates a numeric solution for an initial-value problem given
        the solver backend and function for the differential equation.
        """
        sol = self.solver.solve_ivp(
            func=diffeq_func, ic=ic, ti=ti, tf=tf, dt=dt, args=args
        )
        self.sols.append(sol)
        return sol

    def plot_numeric_sol_ivp(
        self,
        label_x: str = "t",
        label_y: str = "y",
        title: str = "Title",
    ):
        for sol in self.sols:
            if self.backend == "scipy":
                z = sol.y
                plt.plot(sol.t, z.T)
            else:
                raise NotImplementedError

            plt.xlabel(label_x)
            plt.ylabel(label_y)
            plt.title(title)

        return plt
    
    def clear_sims(
            self,
    ):
        self.sols = []
