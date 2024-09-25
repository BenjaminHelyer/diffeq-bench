# diffeq-bench
Library for benchmarking differential equation solvers and methods.

## Example: Benchmarking Across Many Initial Conditions
![Cool animation](./media_assets/lorenz_trajectories_100_ics.gif)
As an exmaple, this library can be used to benchmark performance
when running simulations across many initial conditions. The
animation below shows 100 initial conditions for a Lorenz system;
benchmarking results indicated that adding CPU multiprocessing
led to speedups of 5x.
