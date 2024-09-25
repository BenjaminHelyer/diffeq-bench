# diffeq-bench
Library for benchmarking differential equation solvers and methods.

## Example: Benchmarking Across Many Initial Conditions
The animation below shows 100 initial conditions for a Lorenz system.
Numerically solving across each initial condition is slow when we
do so sequentially. However, when we solve across various initial
conditions in parallel via Python multiprocessing, we see
nearly a 5x speedup compared to the sequential approach. \
\
None of this requires anything fancy, but it does require
careful benchmarking to ensure that we are actually getting
a performance boost.
\
![Lorenz Animation Across 100 Initial Conditions](https://github.com/BenjaminHelyer/diffeq-bench/blob/main/media_assets/lorenz_trajectories_100_ics.gif)

