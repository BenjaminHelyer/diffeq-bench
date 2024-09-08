"""Module for utility functions for the Simulator class."""

import time
from typing import Callable


def benchmark_time(func: Callable):
    """
    Decorator to be used to benchmark the execution time
    of a different function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        print(f"Benchmarked time = {end_time - start_time}")
        print(f"Function benchmarked was: {func}")

    return wrapper
