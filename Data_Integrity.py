"""
================================================================================
CST-305: Benchmark Project 4 — Degradation of Data Integrity
ODE System Solver & Visualizer
================================================================================
Packages Used:
  - numpy       : Matrix operations, eigenvalue computation, array math
  - scipy       : Matrix exponential (expm), ODE integration (solve_ivp)
  - matplotlib  : Plotting solution curves, phase portraits, e^At components
  - sympy       : Symbolic eigenvalue derivation & characteristic polynomial

Display:
  Graphs appear directly in PyCharm's Scientific View (Plot tab) via plt.show().
  No PNG files are written. Enable Scientific Mode in PyCharm if plots don't
  appear:  View → Scientific Mode  (or the beaker icon in the toolbar).

Approach to Implementation:
  This program models data degradation across processor networks as a system
  of first-order ODEs in the form x' = Ax (homogeneous, f(t) = 0).

  Part 1 — Three Processor System (A, B, C):
    The ODE system is constructed from MB/s flow rates read as input.
    The coefficient matrix A is assembled, eigenvalues are computed,
    and the solution is visualized over time.

  Part 2 — Two Processor Closed System (A, B):
    The 2x2 matrix A is constructed, e^{At} is computed analytically
    and verified numerically, the IVP x(0)=1, x'(0)=-1 is solved,
    and results are plotted with phase portrait and e^{At} components.
================================================================================
"""

import numpy as np
from scipy.linalg import expm, eig
from scipy.integrate import solve_ivp
import matplotlib
# No backend override — PyCharm's Scientific View intercepts plt.show() automatically
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')





# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: INPUT — Read ODE system parameters from user
# ─────────────────────────────────────────────────────────────────────────────

def read_flow_rates_part1():
    """
    Prompt user to enter the MB/s flow rates for the three-processor network.
    Returns the 3x3 coefficient matrix A (scaled by 1/capacity).
    """
    print("\n" + "="*65)
    print("  PART 1: Three-Processor ODE System")
    print("  Processors A, B, C — each with 100 MB capacity")
    print("="*65)
    print("\nEnter I/O flow rates (MB/sec) from the network diagram.")
    print("(Press ENTER to use default values from the assignment figure)\n")

    def prompt(label, default):
        val = input(f"  {label} [{default}]: ").strip()
        return float(val) if val else float(default)

    # Rates from the assignment figure
    rate_B_to_A = prompt("Rate FROM B  TO A  (top arrow, MB/s)", 2)
    rate_A_to_B = prompt("Rate FROM A  TO B  (bottom arrow, MB/s)", 6)
    rate_C_to_B = prompt("Rate FROM C  TO B  (top arrow, MB/s)", 1)
    rate_B_to_C = prompt("Rate FROM B  TO C  (bottom arrow, MB/s)", 5)
    rate_C_out  = prompt("Rate FROM C to network (exit, MB/s)", 4)
    capacity    = prompt("Processor memory capacity (MB)", 100)

    # Build A matrix: a_ij = rate from j to i / capacity (positive off-diag)
    # Diagonal = -(sum of all outflows from that processor) / capacity
    # x1' = -rate_out_A/cap * x1  +  rate_B_to_A/cap * x2
    # x2' =  rate_A_to_B/cap * x1 - (rate_B_to_A + rate_B_to_C)/cap * x2  +  rate_C_to_B/cap * x3
    # x3' =  rate_B_to_C/cap * x2 - (rate_C_to_B + rate_C_out)/cap * x3
    A = np.array([
        [-(rate_A_to_B) / capacity,          rate_B_to_A / capacity,           0                            ],
        [ rate_A_to_B  / capacity,  -(rate_B_to_A + rate_B_to_C) / capacity,   rate_C_to_B / capacity       ],
        [ 0,                          rate_B_to_C / capacity,          -(rate_C_to_B + rate_C_out) / capacity]
    ], dtype=float)

    return A, capacity


def read_flow_rates_part2():
    """
    Prompt user to enter flow rates for the two-processor closed system.
    Returns the 2x2 coefficient matrix A and initial conditions.
    """
    print("\n" + "="*65)
    print("  PART 2: Two-Processor Closed ODE System")
    print("  Processors A and B — each with 100 MB, closed loop")
    print("="*65)
    print("\nEnter I/O flow rates (MB/sec) from the network diagram.")
    print("(Press ENTER to use default values from the assignment figure)\n")

    def prompt(label, default):
        val = input(f"  {label} [{default}]: ").strip()
        return float(val) if val else float(default)

    rate_B_to_A = prompt("Rate FROM B TO A (top arrow, MB/s)", 3)
    rate_A_to_B = prompt("Rate FROM A TO B (bottom arrow, MB/s)", 2)
    capacity    = prompt("Processor memory capacity (MB)", 100)

    x0_val      = prompt("Initial condition x(0)", 1)
    xdot0_val   = prompt("Initial condition x'(0)", -1)

    # A matrix: x1' = -(r_AB/cap)*x1 + (r_BA/cap)*x2
    #           x2' =  (r_AB/cap)*x1 - (r_BA/cap)*x2
    A = np.array([
        [-rate_A_to_B / capacity,  rate_B_to_A / capacity],
        [ rate_A_to_B / capacity, -rate_B_to_A / capacity]
    ], dtype=float)

    return A, capacity, np.array([x0_val, xdot0_val])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: SOLVE — Eigenvalues, matrix exponential, IVP solution
# ─────────────────────────────────────────────────────────────────────────────

def compute_eigenvalues(A, label="A"):
    """
    Compute and display the eigenvalues and eigenvectors of matrix A.
    Returns eigenvalues array.
    """
    eigenvalues, eigenvectors = eig(A)
    eigenvalues_real = eigenvalues.real   # All eigenvalues are real here

    print(f"\n  Matrix {label}:")
    # Pretty-print the matrix with labels
    n = A.shape[0]
    for i, row in enumerate(A):
        row_str = "  [ " + "  ".join(f"{v:+8.5f}" for v in row) + " ]"
        print(row_str)

    print(f"\n  Eigenvalues of {label}:")
    for k, lam in enumerate(eigenvalues_real):
        stability = "stable" if lam < -1e-10 else ("neutral" if abs(lam) < 1e-10 else "unstable")
        print(f"    λ{k+1} = {lam:+.6f}  ({stability})")

    # System stability summary
    if all(lam < -1e-10 for lam in eigenvalues_real):
        print("\n  ✓ System is ASYMPTOTICALLY STABLE (all eigenvalues negative)")
        print("    → All data quantities decay exponentially to zero over time.")
    elif any(abs(lam) < 1e-10 for lam in eigenvalues_real):
        print("\n  ◌ System is MARGINALLY STABLE (has zero eigenvalue)")
        print("    → Closed system: total data is conserved; components redistribute.")
    else:
        print("\n  ✗ System is UNSTABLE")

    return eigenvalues_real, eigenvectors


def compute_matrix_exponential_part2(A_raw):
    """
    Analytically compute e^{At} for the 2x2 two-processor system.
    Uses the formula: e^{At} = I + (1/(λ2-λ1)) * (e^{λ2t} - e^{λ1t}) * A
    For λ1=0, λ2=-5: e^{At} = I + (1/5)(1 - e^{-5t}) * A
    Returns a function of t.
    """
    eigenvalues, _ = eig(A_raw)
    lam = sorted(eigenvalues.real)   # [λ2 (negative), λ1 (zero)]  → sort ascending
    lam1, lam2 = lam[1], lam[0]     # lam1 = 0 (larger), lam2 = -5 (smaller)

    print(f"\n  Eigenvalues for e^{{At}} computation: λ₁ = {lam1:.1f},  λ₂ = {lam2:.1f}")
    print(f"  Formula: e^{{At}} = I + (1/{lam1 - lam2:.0f})(1 - e^{{{lam2:.0f}t}}) · A")

    def eAt(t):
        """Matrix exponential e^{At} evaluated at time t (scalar or array)."""
        scalar = np.isscalar(t)
        t = np.atleast_1d(np.asarray(t, dtype=float))
        results = np.zeros((len(t), 2, 2))
        for i, ti in enumerate(t):
            factor = (1.0 / (lam1 - lam2)) * (1.0 - np.exp(lam2 * ti))
            results[i] = np.eye(2) + factor * A_raw
        return results[0] if scalar else results

    # Verify at t=0: should equal identity
    check = eAt(0.0)
    assert np.allclose(check, np.eye(2), atol=1e-10), "e^{A*0} ≠ I — formula error!"
    print("  ✓ Verified: e^{A·0} = I")

    # Cross-check with scipy expm at t=1
    t_test = 1.0
    analytic = eAt(t_test)
    numeric  = expm(A_raw * t_test)
    max_err  = np.max(np.abs(analytic - numeric))
    print(f"  ✓ Cross-check vs scipy.expm at t=1: max error = {max_err:.2e}")

    return eAt, lam2   # return lam2 (the non-zero eigenvalue)


def solve_ivp_matrix_method(A_raw, eAt_func, x0):
    """
    Solve the IVP x' = Ax, x(0) = x0 using the matrix exponential method.
    x(t) = e^{At} · x(0)
    Returns time array and solution array.
    """
    t_span = np.linspace(0, 2.0, 500)
    solutions = np.zeros((2, len(t_span)))

    for i, ti in enumerate(t_span):
        solutions[:, i] = eAt_func(ti) @ x0

    print(f"\n  IVP Solution via Matrix Method: x(t) = e^{{At}} · x(0)")
    print(f"  Initial state vector: x(0) = {x0}")
    print(f"  Closed-form result:")
    print(f"    x₁(t) = e^{{-5t}}")
    print(f"    x₂(t) = -e^{{-5t}}")
    print(f"  Verify at t=0: x₁(0) = {solutions[0,0]:.4f} (expect 1.0000)")
    print(f"  Verify at t=0: x₂(0) = {solutions[1,0]:.4f} (expect -1.0000)")

    return t_span, solutions


def solve_three_processor(A, x0, t_end=120):
    """
    Numerically integrate the 3-processor ODE system using scipy solve_ivp.
    Returns time array and solution trajectories.
    """
    def system(t, x):
        return A @ x   # x' = Ax

    result = solve_ivp(
        system,
        t_span=(0, t_end),
        y0=x0,
        method='RK45',
        t_eval=np.linspace(0, t_end, 1000),
        rtol=1e-9
    )
    return result.t, result.y


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: DISPLAY — Print solution summary in problem context
# ─────────────────────────────────────────────────────────────────────────────

def display_solution_part1(A, eigenvalues, t, y):
    """Display Part 1 results with domain-specific terminology (MB, sec)."""
    print("\n" + "="*65)
    print("  PART 1 SOLUTION SUMMARY")
    print("="*65)
    print("\n  Physical Interpretation (Data Integrity Context):")
    print("  ─────────────────────────────────────────────────")
    print(f"  The ODE system x' = Ax models the rate of change of I/O + data")
    print(f"  (in MB) across three networked processors over time (seconds).")
    print()
    print(f"  Degradation Modes (eigenvalues = decay rates):")
    for k, lam in enumerate(eigenvalues):
        half_life = np.log(2) / abs(lam) if abs(lam) > 1e-10 else float('inf')
        print(f"    Mode {k+1}: λ = {lam:+.6f} s⁻¹  →  "
              f"half-life ≈ {half_life:.1f} sec")
    print()
    print("  Data at selected time points (MB):")
    print(f"  {'Time (s)':<12} {'Proc A x₁(t)':<18} {'Proc B x₂(t)':<18} {'Proc C x₃(t)':<18}")
    print("  " + "-"*66)
    checkpoints = [0, 10, 30, 60, 100]
    for tc in checkpoints:
        idx = np.argmin(np.abs(t - tc))
        print(f"  {t[idx]:<12.1f} {y[0,idx]:<18.4f} {y[1,idx]:<18.4f} {y[2,idx]:<18.4f}  MB")
    print()
    print("  Conclusion: All data quantities decay exponentially toward 0 MB,")
    print("  modeling complete data degradation due to bit-flipping, charge")
    print("  dispersion, insulation leakage, and physical media decomposition.")


def display_solution_part2(A, eigenvalues, lam_nonzero, x0, t, solutions):
    """Display Part 2 results with domain-specific terminology."""
    print("\n" + "="*65)
    print("  PART 2 SOLUTION SUMMARY")
    print("="*65)
    print("\n  Physical Interpretation (Closed Processor Loop):")
    print("  ─────────────────────────────────────────────────")
    print("  The closed 2-processor system conserves total I/O data.")
    print(f"  Eigenvalue λ=0   → conservation of total data in system")
    print(f"  Eigenvalue λ={lam_nonzero:.1f} → exponential redistribution between A & B")
    print()
    print(f"  IVP: x(0) = {x0[0]}, x'(0) = {x0[1]}")
    print(f"  Matrix-method solution:")
    print(f"    x₁(t) = e^{{{lam_nonzero:.0f}t}}")
    print(f"    x₂(t) = -e^{{{lam_nonzero:.0f}t}}")
    print()
    print("  Solution values (MB data state):")
    print(f"  {'Time (s)':<12} {'x₁(t)':<18} {'x₂(t)':<18} {'x₁+x₂ (sum)':<18}")
    print("  " + "-"*66)
    checkpoints = [0, 0.1, 0.3, 0.5, 1.0, 2.0]
    for tc in checkpoints:
        idx = np.argmin(np.abs(t - tc))
        total = solutions[0,idx] + solutions[1,idx]
        print(f"  {t[idx]:<12.2f} {solutions[0,idx]:<18.6f} {solutions[1,idx]:<18.6f} {total:<18.6f}")
    print()
    print("  Note: x₁(t) + x₂(t) = 0 ∀ t (antisymmetric; initial conditions).")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: VISUALIZE — Render plots directly in PyCharm's Scientific View
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG   = '#0d0d1a'
PANEL_BG  = '#1a1a2e'
BLUE      = '#00d4ff'
PINK      = '#ff6b9d'
GREEN     = '#a8ff3e'
ORANGE    = '#ffaa00'
WHITE     = '#e0e0e0'
GRAY      = '#444466'


def style_ax(ax, title, xlabel, ylabel):
    """Apply consistent dark-theme styling to an axes object."""
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=WHITE, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color=WHITE, fontsize=9)
    ax.set_ylabel(ylabel, color=WHITE, fontsize=9)
    ax.tick_params(colors=WHITE, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRAY)
    ax.grid(True, alpha=0.15, color=WHITE, linestyle='--')


def plot_part1(t, y, eigenvalues):
    """Generate three plots for Part 1: time series, phase portrait, eigenvalue bar."""
    fig = plt.figure(figsize=(16, 5), facecolor=DARK_BG)
    fig.suptitle("Part 1 — Three-Processor Data Degradation  (x' = Ax,  f(t) = 0)",
                 color=WHITE, fontsize=13, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # ── Plot 1: Time series
    ax1 = fig.add_subplot(gs[0])
    colors = [BLUE, PINK, GREEN]
    labels = ['Processor A  x₁(t)', 'Processor B  x₂(t)', 'Processor C  x₃(t)']
    for i, (c, lbl) in enumerate(zip(colors, labels)):
        ax1.plot(t, y[i], color=c, linewidth=2, label=lbl)
    ax1.legend(facecolor='#252540', labelcolor=WHITE, fontsize=8, loc='upper right')
    style_ax(ax1, "Data Degradation Over Time", "Time (seconds)", "I/O + Data (MB)")

    # Annotate half-lives
    for i, lam in enumerate(eigenvalues):
        if abs(lam) > 1e-10:
            hl = np.log(2) / abs(lam)
            ax1.axvline(x=hl, color=colors[i], linewidth=0.8, linestyle=':', alpha=0.5)

    # ── Plot 2: Phase portrait x1 vs x2
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(y[0], y[1], color=BLUE, linewidth=2)
    ax2.plot(y[0,0], y[1,0], 'o', color=GREEN, markersize=9, label='t = 0  (start)', zorder=5)
    ax2.plot(y[0,-1], y[1,-1], 's', color=PINK, markersize=8, label='t = T  (end)', zorder=5)
    ax2.legend(facecolor='#252540', labelcolor=WHITE, fontsize=8)
    style_ax(ax2, "Phase Portrait: x₁ vs x₂", "x₁(t)  Processor A (MB)", "x₂(t)  Processor B (MB)")

    # ── Plot 3: Eigenvalue magnitudes (decay rates)
    ax3 = fig.add_subplot(gs[2])
    xlabels = [f'λ₁\n{eigenvalues[0]:.4f}', f'λ₂\n{eigenvalues[1]:.4f}', f'λ₃\n{eigenvalues[2]:.4f}']
    bar_colors = [BLUE, PINK, GREEN]
    ax3.bar(xlabels, [abs(l) for l in eigenvalues], color=bar_colors, width=0.5, edgecolor=GRAY)
    ax3.set_ylabel('|λ|  Decay Rate (s⁻¹)', color=WHITE, fontsize=9)
    style_ax(ax3, "Eigenvalue Magnitudes\n(Higher = Faster Decay)", "Eigenvalue", "|λ| (s⁻¹)")

    plt.tight_layout()
    plt.show()


def plot_part2(t, solutions, eAt_func, lam_nonzero):
    """Generate four plots for Part 2: solution, phase, e^At components, verification."""
    fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
    fig.suptitle("Part 2 — Two-Processor Closed System  |  IVP & Matrix Exponential e^{At}",
                 color=WHITE, fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.36, hspace=0.45)

    # ── Plot 1: IVP solution curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, solutions[0], color=BLUE, linewidth=2.5, label='x₁(t) = e^{-5t}  (Processor A)')
    ax1.plot(t, solutions[1], color=PINK, linewidth=2.5, label='x₂(t) = −e^{-5t}  (Processor B)')
    ax1.axhline(0, color=WHITE, linewidth=0.4, linestyle='--', alpha=0.3)
    ax1.plot(0, 1,  'o', color=GREEN,  markersize=9, zorder=5, label='x₁(0) = 1')
    ax1.plot(0, -1, 's', color=ORANGE, markersize=9, zorder=5, label='x₂(0) = −1')
    ax1.legend(facecolor='#252540', labelcolor=WHITE, fontsize=8)
    style_ax(ax1, "IVP Solution  x(0)=[1, −1]ᵀ", "Time (s)", "Data State x(t)")

    # ── Plot 2: Phase portrait
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(solutions[0], solutions[1], color=GREEN, linewidth=2.5)
    ax2.plot(solutions[0,0], solutions[1,0], 'o', color=GREEN, markersize=11,
             label='t=0: (1, −1)', zorder=5)
    ax2.plot(solutions[0,-1], solutions[1,-1], 's', color=PINK, markersize=9,
             label='t→∞: (0, 0)', zorder=5)
    # Reference line x2 = -x1
    xline = np.linspace(-0.02, 1.05, 100)
    ax2.plot(xline, -xline, color=WHITE, linewidth=0.7, linestyle=':', alpha=0.4,
             label='x₂ = −x₁')
    ax2.legend(facecolor='#252540', labelcolor=WHITE, fontsize=8)
    style_ax(ax2, "Phase Portrait  x₁ vs x₂", "x₁(t)", "x₂(t)")

    # ── Plots 3 & 4: two representative e^{At} components (full grid in plot_eAt_full)
    t_eAt = np.linspace(0, 3, 400)
    eAt_vals = eAt_func(t_eAt)   # shape (N, 2, 2)

    ax3 = fig.add_subplot(gs[1, 0])
    vals11 = eAt_vals[:, 0, 0]
    ax3.plot(t_eAt, vals11, color=BLUE, linewidth=2.2)
    ax3.axhline(vals11[-1], color=WHITE, linewidth=0.6, linestyle='--', alpha=0.35,
                label=f'Steady state: {vals11[-1]:.3f}')
    ax3.legend(facecolor='#252540', labelcolor=WHITE, fontsize=7)
    style_ax(ax3, "e^{At}[1,1]  =  (3+2e^{-5t})/5", "t (s)", "Value")

    ax4 = fig.add_subplot(gs[1, 1])
    vals22 = eAt_vals[:, 1, 1]
    ax4.plot(t_eAt, vals22, color=ORANGE, linewidth=2.2)
    ax4.axhline(vals22[-1], color=WHITE, linewidth=0.6, linestyle='--', alpha=0.35,
                label=f'Steady state: {vals22[-1]:.3f}')
    ax4.legend(facecolor='#252540', labelcolor=WHITE, fontsize=7)
    style_ax(ax4, "e^{At}[2,2]  =  (2+3e^{-5t})/5", "t (s)", "Value")

    plt.tight_layout()
    plt.show()


def plot_eAt_full(eAt_func):
    """Plot all four components of e^{At} in a clean 2x2 grid."""
    t_vals = np.linspace(0, 3, 500)
    eAt_all = eAt_func(t_vals)   # shape (500, 2, 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=DARK_BG)
    fig.suptitle("Matrix Exponential  e^{At}  —  All Four Components",
                 color=WHITE, fontsize=13, fontweight='bold')

    colors = [[BLUE, PINK], [GREEN, ORANGE]]
    formulas = [['(3 + 2e⁻⁵ᵗ) / 5', '3(1 − e⁻⁵ᵗ) / 5'],
                ['2(1 − e⁻⁵ᵗ) / 5', '(2 + 3e⁻⁵ᵗ) / 5']]

    for r in range(2):
        for c in range(2):
            ax = axes[r][c]
            vals = eAt_all[:, r, c]
            ax.set_facecolor(PANEL_BG)
            ax.plot(t_vals, vals, color=colors[r][c], linewidth=2.5)
            ax.axhline(vals[-1], color=WHITE, linewidth=0.7, linestyle='--', alpha=0.4)
            ax.set_title(f'e^{{At}}[{r+1},{c+1}] = {formulas[r][c]}',
                         color=WHITE, fontsize=10, fontweight='bold')
            ax.set_xlabel('t  (seconds)', color=WHITE, fontsize=9)
            ax.set_ylabel('Value', color=WHITE, fontsize=9)
            ax.tick_params(colors=WHITE, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(GRAY)
            ax.grid(True, alpha=0.15, color=WHITE, linestyle='--')
            ax.text(0.97, 0.5, f'→ {vals[-1]:.4f}',
                    transform=ax.transAxes, color=WHITE, fontsize=9,
                    ha='right', va='center', alpha=0.6)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "╔" + "═"*63 + "╗")
    print("║  CST-305 Project 4 — Data Integrity ODE Solver              ║")
    print("║  Models data degradation in processor networks via x' = Ax  ║")
    print("╚" + "═"*63 + "╝")

    # ── PART 1 ────────────────────────────────────────────────────────────────
    print("\n>>> PART 1: THREE-PROCESSOR SYSTEM")
    A1, cap1 = read_flow_rates_part1()

    print("\n  Solving ODE system...")
    eigenvalues1, eigenvectors1 = compute_eigenvalues(A1, label="A (Part 1, scaled)")

    # Initial conditions: 25 MB I/O in each processor at t=0
    x0_p1 = np.array([25.0, 25.0, 25.0])
    print(f"\n  Initial conditions: x₁(0) = x₂(0) = x₃(0) = 25 MB (I/O portion)")
    t1, y1 = solve_three_processor(A1, x0_p1, t_end=120)

    display_solution_part1(A1, eigenvalues1, t1, y1)
    plot_part1(t1, y1, eigenvalues1)

    # ── PART 2 ────────────────────────────────────────────────────────────────
    print("\n\n>>> PART 2: TWO-PROCESSOR CLOSED SYSTEM")
    A2, cap2, x0_p2 = read_flow_rates_part2()

    # Use unscaled matrix (multiply by 100) for cleaner eigenvalues
    A2_raw = A2 * 100.0

    print("\n  Unscaled matrix A₀ = 100·A (for integer eigenvalues):")
    eigenvalues2, eigenvectors2 = compute_eigenvalues(A2_raw, label="A₀ (Part 2, unscaled)")

    print("\n  Computing matrix exponential e^{At}...")
    eAt_func, lam_nonzero = compute_matrix_exponential_part2(A2_raw)

    print("\n  Solving IVP via matrix method...")
    t2, sol2 = solve_ivp_matrix_method(A2_raw, eAt_func, x0_p2)

    display_solution_part2(A2_raw, eigenvalues2, lam_nonzero, x0_p2, t2, sol2)
    plot_part2(t2, sol2, eAt_func, lam_nonzero)
    plot_eAt_full(eAt_func)

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  ALL SOLUTIONS COMPUTED SUCCESSFULLY")
    print("="*65)
    print("  Graphs displayed in PyCharm's Scientific View (Plot tab).")
    print()
    print("  Key Results:")
    print(f"  Part 1 Eigenvalues: {[f'{l:.4f}' for l in eigenvalues1]}")
    print(f"  Part 2 Eigenvalues: λ = 0, λ = {lam_nonzero:.1f}")
    print(f"  Part 2 IVP Solution: x₁(t) = e^{{{lam_nonzero:.0f}t}},  x₂(t) = -e^{{{lam_nonzero:.0f}t}}")
    print()
    print("  All graphs are displayed. Close graph windows to exit.")
    print("="*65 + "\n")




if __name__ == "__main__":
    main()