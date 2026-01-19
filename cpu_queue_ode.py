
"""
Program Name: CPU Queue ODE Solver
Authors: Chance Atkinson
Packages Used: numpy, scipy, matplotlib
Approach:
- Model CPU task queue behavior using an ODE
- Solve numerically using SciPy
- Visualize queue length over time
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def cpu_queue_ode(t, Q, arrival_rate, service_rate):
    """
    Computes dQ/dt for CPU queue length.

    Parameters:
    t : float
        Time (seconds)
    Q : float
        Queue length (tasks)
    arrival_rate : float
        Task arrival rate (tasks/sec)
    service_rate : float
        CPU service rate (tasks/sec)
    """
    return arrival_rate - service_rate


def main():
    # ---- User Input ----
    arrival_rate = float(input("Enter task arrival rate (tasks/sec): "))
    service_rate = float(input("Enter CPU service rate (tasks/sec): "))
    Q0 = float(input("Enter initial queue length (tasks): "))

    # ---- Time Domain ----
    t_start = 0
    t_end = 20
    t_eval = np.linspace(t_start, t_end, 200)

    # ---- Solve ODE ----
    solution = solve_ivp(
        cpu_queue_ode,
        [t_start, t_end],
        [Q0],
        t_eval=t_eval,
        args=(arrival_rate, service_rate)
    )

    # ---- Error Estimate ----
    error_estimate = 0.01 * np.max(solution.y[0])

    # ---- Visualization ----
    plt.figure(figsize=(8, 5))
    plt.plot(solution.t, solution.y[0], label="CPU Queue Length")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Queue Length (tasks)")
    plt.title("CPU Task Queue Length Over Time")
    plt.grid(True)
    plt.legend()

    plt.figtext(
        0.5, -0.12,
        f"Arrival rate = {arrival_rate} tasks/sec | "
        f"Service rate = {service_rate} tasks/sec\n"
        f"Estimated numerical error ≈ ±{error_estimate:.2f} tasks",
        ha="center"
    )

    plt.show()


if __name__ == "__main__":
    main()