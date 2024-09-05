"""Solver for differential equations based on PyTorch, via the library torchdiffeq."""

from typing import Callable, List

import torch
from torchdiffeq import odeint

from simulation.solver import DiffeqSolver


class PyTorchSolver(DiffeqSolver):
    """
    Thin wrapper around TorchDiffeq differential equation solvers.
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
        Solver for an initial-value problem using torchdiffeq's odeint.
        """
        t_eval = self.calculate_t_eval(ti, tf, dt)
        y0 = torch.tensor(ic)
        diffeq_nn = DiffEqNet(func, args)
        sol = odeint(diffeq_nn, y0, t_eval)
        return sol

    def solve_bvp(self, func: Callable, bc, x): ...

    def calculate_t_eval(
        self,
        ti: float,
        tf: float,
        dt: float,
    ) -> torch.Tensor:
        """
        Calculates the time eval points given initial, final, and delta time values.
        """
        return torch.arange(
            ti, tf, dt
        )  # this will exclude tf in some cases but is probably OK for now


class DiffEqNet(torch.nn.Module):
    """
    PyTorch neural network module which
    encapsulates the properties of the
    differential equation.
    """

    def __init__(
        self,
        func,
        args,
    ):
        """
        Initializes the neural network object.
        """
        super(DiffEqNet, self).__init__()
        self.args = args
        self.func = func

    def forward(self, t, z):
        """
        Forward pass for the neural network object.
        """
        # converting list to tensor at every step might add overhead
        # should probably find a better way to do this
        return torch.tensor(self.func(t, z, self.args))
