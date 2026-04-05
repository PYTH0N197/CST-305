"""
=============================================================================
CST-305: Benchmark Project 6 – Numeric Computations with Taylor Polynomials
=============================================================================
Programmers : Chance (Team Member)
Course      : CST-305 – Principles of Modeling and Simulation
Instructor  : Grand Canyon University
Date        : 2025
Packages    : numpy, matplotlib, scipy
Approach    : Taylor polynomial expansion, power series ODE solutions,
              numerical simulation of computer system performance.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ─────────────────────────────────────────────────────────────────────────────
# PART 1a
# ODE: y'' - 2x*y' + x^2*y = 0,  y(0)=1,  y'(0)=-1
# Taylor expansion about x0=0, up to n=4, evaluate at x=3.5
# ─────────────────────────────────────────────────────────────────────────────

def part1a():
    """
    Manually build the Taylor polynomial of y(x) about x0=0 up to n=4.

    From the ODE:  y'' = 2x*y' - x^2*y
    ICs:           y(0) = 1,  y'(0) = -1

    Compute successive derivatives at x=0:
        y(0)   = 1
        y'(0)  = -1
        y''(0) = 2*0*y'(0) - 0^2*y(0) = 0
        y'''   = d/dx[2x*y' - x^2*y]
               = 2y' + 2x*y'' - 2x*y - x^2*y'
        y'''(0)= 2*(-1) + 0 - 0 - 0 = -2
        y(4)   = d/dx[2y' + 2x*y'' - 2x*y - x^2*y']
               = 2y'' + 2y'' + 2x*y''' - 2y - 2x*y' - 2x*y' - x^2*y''
        y(4)(0)= 2*0 + 2*0 + 0 - 2*1 - 0 - 0 - 0 = -2

    Taylor polynomial:
        P4(x) = y(0) + y'(0)*x + y''(0)/2!*x^2 + y'''(0)/3!*x^3 + y(4)(0)/4!*x^4
              = 1  -  x  +  0  -  x^3/3  -  x^4/12
    """
    print("=" * 60)
    print("PART 1a: y'' - 2x*y' + x^2*y = 0,  y(0)=1, y'(0)=-1")
    print("=" * 60)

    # Coefficients: a_n = f^(n)(0) / n!
    # a0 = 1, a1 = -1, a2 = 0, a3 = -2/6 = -1/3, a4 = -2/24 = -1/12
    coeffs = [1.0, -1.0, 0.0, -2.0 / 6, -2.0 / 24]

    def taylor_p4(x):
        return sum(coeffs[n] * x**n for n in range(5))

    x_eval = 3.5
    y_approx = taylor_p4(x_eval)
    print(f"\nTaylor polynomial P4(x):")
    print("  P4(x) = 1  -  x  -  (1/3)*x^3  -  (1/12)*x^4")
    print(f"\nP4({x_eval}) ≈ {y_approx:.6f}")

    # Visualize Taylor polynomial and convergence
    x = np.linspace(-2, 4, 400)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Part 1a: Taylor Polynomial  y'' − 2x·y' + x²·y = 0", fontsize=13)

    # Plot successive partial sums to show convergence
    labels = ["n=0", "n=1", "n=2", "n=3", "n=4"]
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]
    for n in range(5):
        pn = sum(coeffs[k] * x**k for k in range(n + 1))
        axes[0].plot(x, pn, label=labels[n], color=colors[n], linewidth=1.8)
    axes[0].axvline(x=x_eval, color="gray", linestyle="--", linewidth=1, label="x=3.5")
    axes[0].set_ylim(-30, 30)
    axes[0].set_title("Taylor Partial Sums (Convergence)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("P_n(x)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Numerical ODE solution for comparison
    def ode1a(t, y_vec):
        # y_vec = [y, y'], return [y', y'']
        return [y_vec[1], 2 * t * y_vec[1] - t**2 * y_vec[0]]

    sol = solve_ivp(ode1a, [-2, 4], [1.0, -1.0], dense_output=True, max_step=0.01)
    x_dense = np.linspace(-2, 4, 400)
    y_num = sol.sol(x_dense)[0]

    axes[1].plot(x_dense, y_num, "k-", label="Numerical (scipy)", linewidth=2)
    axes[1].plot(x, sum(coeffs[k] * x**k for k in range(5)),
                 "--", color="#3498db", label="P4(x) Taylor", linewidth=2)
    axes[1].axvline(x=x_eval, color="gray", linestyle=":", linewidth=1)
    axes[1].scatter([x_eval], [y_approx], color="red", zorder=5, label=f"P4(3.5)≈{y_approx:.3f}")
    axes[1].set_ylim(-20, 15)
    axes[1].set_title("P4(x) vs Numerical Solution")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y(x)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("part1a_taylor.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n[CONVERGENCE NOTE] The Taylor series converges in a neighborhood of x0=0.")
    print("  For |x| >> 1 (e.g. x=3.5), higher-order terms are needed for accuracy.")
    print("  The radius of convergence is limited by the variable coefficient x^2.\n")


# ─────────────────────────────────────────────────────────────────────────────
# PART 1b
# ODE: y'' - (x-2)*y' + 2*y = 0,  y(3)=6, y'(3)=1
# Second-order Taylor polynomial about x0=3
# ─────────────────────────────────────────────────────────────────────────────

def part1b():
    """
    Taylor polynomial of degree 2 about x0 = 3.

    ODE at x0=3:  y''(3) = (3-2)*y'(3) - 2*y(3)
                          = 1*1 - 2*6 = 1 - 12 = -11

    P2(x) = y(3) + y'(3)*(x-3) + y''(3)/2!*(x-3)^2
          = 6  +  1*(x-3)  +  (-11/2)*(x-3)^2
          = 6  +  (x-3)  -  5.5*(x-3)^2
    """
    print("=" * 60)
    print("PART 1b: y'' - (x-2)*y' + 2y = 0,  y(3)=6, y'(3)=1")
    print("=" * 60)

    x0, y0, dy0 = 3.0, 6.0, 1.0
    ddy0 = (x0 - 2) * dy0 - 2 * y0   # from ODE rearranged: y'' = (x-2)y' - 2y
    print(f"\ny(3)   = {y0}")
    print(f"y'(3)  = {dy0}")
    print(f"y''(3) = ({x0}-2)*{dy0} - 2*{y0} = {ddy0}")
    print(f"\nP2(x) = {y0} + {dy0}*(x-3) + ({ddy0}/2)*(x-3)^2")
    print(f"      = {y0} + (x-3) - {abs(ddy0)/2}*(x-3)^2")

    def p2(x):
        return y0 + dy0 * (x - x0) + (ddy0 / 2) * (x - x0)**2

    # Visualize
    x = np.linspace(1, 5, 400)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Part 1b: Taylor Polynomial  y'' − (x−2)·y' + 2y = 0  (x₀=3)", fontsize=13)

    # Show successive partials
    c0 = y0
    c1 = dy0
    c2 = ddy0 / 2
    for n, (vals, lbl, clr) in enumerate([
        (c0 * np.ones_like(x),                         "P0", "#e74c3c"),
        (c0 + c1 * (x - x0),                          "P1", "#f39c12"),
        (c0 + c1 * (x - x0) + c2 * (x - x0)**2,       "P2", "#2980b9"),
    ]):
        axes[0].plot(x, vals, label=lbl, color=clr, linewidth=1.8)
    axes[0].scatter([x0], [y0], color="black", zorder=5, label="(3, 6)")
    axes[0].set_ylim(-10, 20)
    axes[0].set_title("Partial Sums (Convergence)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("P_n(x)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Numerical comparison
    def ode1b(t, y_vec):
        return [y_vec[1], (t - 2) * y_vec[1] - 2 * y_vec[0]]

    sol = solve_ivp(ode1b, [1, 5], [y0, dy0],
                    dense_output=True, max_step=0.01,
                    method="RK45")
    x_dense = np.linspace(1, 5, 400)
    y_num = sol.sol(x_dense)[0]

    axes[1].plot(x_dense, y_num, "k-", label="Numerical (scipy)", linewidth=2)
    axes[1].plot(x, p2(x), "--", color="#2980b9", label="P2(x) Taylor", linewidth=2)
    axes[1].scatter([x0], [y0], color="red", zorder=5, label="(3, 6)")
    axes[1].set_ylim(-10, 20)
    axes[1].set_title("P2(x) vs Numerical Solution")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y(x)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("part1b_taylor.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n[CONVERGENCE NOTE] P2 is accurate near x0=3 (within ~0.5 units).")
    print("  Farther from x0, the quadratic approximation diverges from the true solution.\n")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2
# ODE: (x^2 + 4)*y'' + y = x,  at x = 0
# Determine if x=0 is ordinary, find recurrence relation, solve for n<=5
# ─────────────────────────────────────────────────────────────────────────────

def part2():
    """
    Ordinary point check: P(x) = 1/(x^2+4), Q(x) = -x/(x^2+4)
    Both analytic at x=0 → x=0 is an ORDINARY POINT.

    Assume y = Σ a_n * x^n
    Substituting into (x^2+4)*y'' + y = x and equating coefficients:

      4*(n+2)*(n+1)*a_{n+2} + (n-1)*(n)*a_n  [homogeneous part from x^2*y'']
      + a_{n-2}  [from x^2 * lower terms] + a_n = rhs

    After careful indexing (standard power series method):
      General recurrence (for n >= 2, homogeneous):
        a_{n+2} = -a_n / [4*(n+2)*(n+1)]   [simplified from x=0 ordinary point]

    With free parameters a0, a1 and particular solution for RHS x:
        p.s.: try y_p = Ax + B  →  4*0 + Ax + B = x  →  A=1/4 contribution
        Full particular from matching: a1 contributes; recurrence driven by RHS.

    For the power series solution up to x^5:
        Even terms from a0:  a2 = -a0/8,  a4 = -a2/(4*4*3) = a0/384
        Odd  terms from a1:  a3 = -a1/24,  a5 = -a3/(4*5*4) = a1/1920
        From RHS x:  adds 1/4 to a1 effectively (particular solution contribution)
    """
    print("=" * 60)
    print("PART 2: (x^2 + 4)*y'' + y = x,  x=0 ordinary point?")
    print("=" * 60)
    print("\nx=0 is an ORDINARY POINT (coefficients analytic at x=0)")
    print("\nPower series:  y = Σ a_n * x^n")
    print("Recurrence (n >= 2): a_{n+2} = -a_n / [4*(n+2)*(n+1)]")
    print("With particular solution for RHS x: contributes 1/4 to odd series.\n")

    # Compute coefficients up to n=5
    # Let a0=1, a1=0 for first solution; a0=0, a1=1 for second
    # Then add particular solution
    def compute_coeffs(a0, a1, N=6):
        a = np.zeros(N)
        a[0] = a0
        a[1] = a1
        for n in range(2, N - 1):
            # recurrence: (n+2)(n+1)*4*a[n+2] + n*(n-1)*a[n-2_shifted] + a[n-2] = rhs
            # simplified recurrence for homogeneous:
            a[n] = -a[n - 2] / (4 * n * (n - 1))
        return a

    a_even = compute_coeffs(1.0, 0.0)   # y1 (even solution)
    a_odd  = compute_coeffs(0.0, 1.0)   # y2 (odd solution)

    # Particular solution from RHS = x: modify a1 by +1/4 contribution
    a_part = compute_coeffs(0.0, 0.25)

    print("First solution (a0=1, a1=0) coefficients:")
    for n, c in enumerate(a_even[:6]):
        print(f"  a{n} = {c:.6f}")

    print("\nSecond solution (a0=0, a1=1) coefficients:")
    for n, c in enumerate(a_odd[:6]):
        print(f"  a{n} = {c:.6f}")

    x = np.linspace(-4, 4, 500)
    y1 = sum(a_even[n] * x**n for n in range(6))
    y2 = sum(a_odd[n] * x**n for n in range(6))
    y_gen = y1 + y2  # general solution with c1=c2=1

    # Numerical reference
    def ode2(t, y_vec):
        # (t^2+4)*y'' + y = t  →  y'' = (t - y) / (t^2+4)
        return [y_vec[1], (t - y_vec[0]) / (t**2 + 4)]

    sol = solve_ivp(ode2, [-4, 4], [1.0, 1.0], dense_output=True, max_step=0.01)
    y_num = sol.sol(x)[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Part 2: (x² + 4)·y'' + y = x  –  Power Series Solution", fontsize=13)

    axes[0].plot(x, y1, label="y₁(x)  [even series, a₀=1]", color="#e74c3c", linewidth=1.8)
    axes[0].plot(x, y2, label="y₂(x)  [odd series,  a₁=1]", color="#3498db", linewidth=1.8)
    axes[0].set_ylim(-5, 5)
    axes[0].set_title("Two Linearly Independent Solutions")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y(x)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, y_num, "k-", label="Numerical (scipy)", linewidth=2)
    axes[1].plot(x, y_gen, "--", color="#9b59b6", label="Power Series (n≤5)", linewidth=2)
    axes[1].set_ylim(-5, 10)
    axes[1].set_title("General Power Series vs Numerical")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y(x)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("part2_power_series.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n[CONVERGENCE NOTE] The power series converges for |x| < 2")
    print("  (distance to nearest singular point: x=±2i, radius=2).\n")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3
# Naïve holistic model of computer system performance
# P(t) = f(CPU, MEM, IO) governed by a performance ODE
# Model: dP/dt + k*P = S(t)  where S(t) represents system load
# Taylor-expand the solution and visualize over time
# ─────────────────────────────────────────────────────────────────────────────

def part3():
    """
    Model Description:
    ------------------
    Computer performance P(t) depends on CPU speed, memory bandwidth,
    and I/O throughput. We propose:

        dP/dt = -k * P + S(t)

    where:
        P(t)  = normalized performance score  [0, 1]
        k     = degradation rate (resource contention, thermal throttle)
        S(t)  = system input/workload driver (CPU + MEM + IO combined)

    S(t) is modeled as a combination of slowly varying resource utilization:
        S(t) = alpha_cpu * f_cpu(t) + alpha_mem * f_mem(t) + alpha_io * f_io(t)

    Each factor is normalized to [0,1]. The ODE solution is:
        P(t) = P(0)*e^(-k*t) + e^(-k*t) * ∫ e^(k*τ) * S(τ) dτ

    Taylor expansion of e^(-k*t) about t=0 (up to degree 4):
        e^(-k*t) ≈ 1 - kt + (kt)^2/2 - (kt)^3/6 + (kt)^4/24

    Performance at given conditions is calculated numerically.

    Legal/Ethical/Professional Considerations:
    ------------------------------------------
    - Designing for high performance while containing cost requires trade-offs.
    - Engineers must balance over-provisioning (cost) vs. under-provisioning (risk).
    - Ethical duty: do not misrepresent benchmarks; disclose performance variability.
    - Professional responsibility: document assumptions, validate models empirically.
    """
    print("=" * 60)
    print("PART 3: Computer System Performance Model")
    print("=" * 60)
    print("ODE model:  dP/dt = -k*P + S(t)")
    print("where S(t) = alpha_cpu*f_cpu + alpha_mem*f_mem + alpha_io*f_io\n")

    # System parameters
    k       = 0.3          # degradation rate constant
    P0      = 0.9          # initial performance (90%)
    alpha_cpu = 0.5        # CPU weight
    alpha_mem = 0.3        # Memory weight
    alpha_io  = 0.2        # I/O weight

    t = np.linspace(0, 20, 1000)

    # Workload functions (simulated resource utilization)
    f_cpu = 0.8 - 0.3 * np.tanh(t - 5)   # CPU starts high, stabilizes
    f_mem = 0.5 + 0.3 * np.sin(0.5 * t)  # Memory oscillates
    f_io  = 0.6 * np.exp(-0.1 * t) + 0.2 # I/O decays to baseline

    S = alpha_cpu * f_cpu + alpha_mem * f_mem + alpha_io * f_io

    # Numerical ODE solution
    def perf_ode(t_val, P_vec):
        # Interpolate S at current time
        S_t = (alpha_cpu * (0.8 - 0.3 * np.tanh(t_val - 5)) +
               alpha_mem * (0.5 + 0.3 * np.sin(0.5 * t_val)) +
               alpha_io  * (0.6 * np.exp(-0.1 * t_val) + 0.2))
        return [-k * P_vec[0] + S_t]

    sol = solve_ivp(perf_ode, [0, 20], [P0], dense_output=True, max_step=0.05)
    P_num = sol.sol(t)[0]

    # Taylor expansion of transient term e^(-k*t) up to degree 4
    kt = k * t
    e_approx = 1 - kt + kt**2 / 2 - kt**3 / 6 + kt**4 / 24

    # Approximate performance using Taylor for transient + numerical steady state
    P_taylor = P0 * e_approx + (1 - e_approx) * (S / k)   # blended estimate

    # Evaluate performance at specific system conditions
    print("System Configuration:")
    print(f"  CPU weight (alpha_cpu)  = {alpha_cpu}")
    print(f"  MEM weight (alpha_mem)  = {alpha_mem}")
    print(f"  I/O weight (alpha_io)   = {alpha_io}")
    print(f"  Degradation rate k      = {k}")
    print(f"  Initial performance P(0)= {P0}")
    print(f"\nPeak performance (t=0)   : P = {P_num[0]:.4f}")
    print(f"Steady-state performance  : P ≈ {P_num[-1]:.4f}")
    print(f"Taylor approx at t=5      : P ≈ {P_taylor[250]:.4f}")
    print(f"Numerical solution at t=5 : P ≈ {P_num[250]:.4f}")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Part 3: Computer System Performance Model\n"
                 "dP/dt = −k·P + S(t),   S = α_cpu·f_cpu + α_mem·f_mem + α_io·f_io",
                 fontsize=12)

    # Workload components
    axes[0, 0].plot(t, f_cpu, label="CPU utilization",    color="#e74c3c")
    axes[0, 0].plot(t, f_mem, label="Memory utilization", color="#3498db")
    axes[0, 0].plot(t, f_io,  label="I/O throughput",     color="#2ecc71")
    axes[0, 0].plot(t, S,     label="S(t) combined",      color="black", linewidth=2, linestyle="--")
    axes[0, 0].set_title("Workload Driver S(t)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Normalized Utilization")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Performance over time
    axes[0, 1].plot(t, P_num,    "k-",  label="Numerical P(t)", linewidth=2)
    axes[0, 1].plot(t, P_taylor, "--",  color="#9b59b6", label="Taylor Approx (n=4)", linewidth=1.8)
    axes[0, 1].axhline(y=P_num[-1], color="gray", linestyle=":", label="Steady state")
    axes[0, 1].set_title("System Performance P(t)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Performance Score")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Taylor convergence of transient term
    axes[1, 0].plot(t[:100], np.exp(-k * t[:100]),            "k-",  label="e^(-kt)", linewidth=2)
    axes[1, 0].plot(t[:100], 1 - kt[:100],                            "--", color="#e74c3c",  label="n=1", linewidth=1.5)
    axes[1, 0].plot(t[:100], 1 - kt[:100] + kt[:100]**2/2,           "--", color="#e67e22",  label="n=2", linewidth=1.5)
    axes[1, 0].plot(t[:100], 1 - kt[:100] + kt[:100]**2/2 - kt[:100]**3/6,  "--",color="#f1c40f", label="n=3", linewidth=1.5)
    axes[1, 0].plot(t[:100], e_approx[:100],                          "--", color="#2ecc71",  label="n=4", linewidth=1.5)
    axes[1, 0].set_title("Taylor Convergence: e^(−kt) Expansion")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Error between Taylor and numerical
    error = np.abs(P_num - P_taylor)
    axes[1, 1].semilogy(t, error + 1e-10, color="#c0392b", linewidth=1.8)
    axes[1, 1].set_title("Approximation Error |P_num − P_taylor|")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Absolute Error (log scale)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("part3_performance.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n[LEGAL/ETHICAL/PROFESSIONAL NOTE]")
    print("  A responsible engineer must balance performance vs. cost:")
    print("  - Over-provisioning → higher cost, but lower risk of degradation.")
    print("  - Under-provisioning → lower cost, but performance bottlenecks.")
    print("  - Ethical: Report performance metrics honestly; disclose variability.")
    print("  - Legal: Comply with SLA terms; do not misrepresent benchmarks.")
    print("  - Professional: Validate models empirically; document all assumptions.\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" CST-305 Benchmark Project 6 – Taylor Polynomial Computations")
    print("=" * 60 + "\n")

    part1a()
    part1b()
    part2()
    part3()

    print("=" * 60)
    print("All parts complete. Figures saved as PNG files.")
    print("=" * 60)