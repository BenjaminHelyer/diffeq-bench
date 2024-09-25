# diffeq-bench
Library for benchmarking differential equation solvers and methods.

## Example: Benchmarking Across Many Initial Conditions
This library can be used to benchmark performance
when running simulations across initial conditions. The
animation below shows 100 initial conditions for a Lorenz system;
benchmarking results indicated that adding CPU multiprocessing
led to speedups of 5x.
![Cool animation](https://github.com/BenjaminHelyer/diffeq-bench/blob/main/media_assets/lorenz_trajectories_100_ics.gif)

