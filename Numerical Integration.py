"""
=============================================================================
CST-305: Project 8 – Numerical Integration (Riemann Sums)
=============================================================================
Programmer(s): Chance
Course:        CST-305 – Modeling and Simulation
Instructor:    Ricardo Citro
Date:          2025

Packages Used:
    - numpy       : Array operations and mathematical functions
    - matplotlib  : Plotting and visualization (PyCharm built-in viewer)
    - scipy       : Numerical integration (quad)

Approach:
    Part 1 – Four Riemann sum problems are implemented:
        (A) f(x) = sin(x) + 1 on [-π, π] with Left, Right, Midpoint Riemann sketches
        (B) f(x) = 3x + 2x² on [0, 1], formula derivation + limit as n→∞
        (C) Definite integral ∫₁ᵉ ln(x) dx via Riemann sum + exact value
        (D) f(x) = x² - x³ on [-1, 0], formula derivation + limit as n→∞

    Part 2 – Simulated media-server download rates (Mbps) over 30 minutes,
              stored in a table, integrated numerically to find total data (MB).
=============================================================================
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import integrate

# Use the PyCharm built-in Scientific View (shows plots inline in the IDE).
# PyCharm automatically injects backend_interagg when "Show plots in tool window"
# is enabled (File → Settings → Tools → Python Scientific).
# Running outside PyCharm: this block silently falls back to the system default.
try:
    matplotlib.use("module://backend_interagg")
except Exception:
    pass  # Fallback: matplotlib uses the system default backend

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.size": 10,
})

# =============================================================================
# HELPER UTILITIES
# =============================================================================

def riemann_sum(f, a, b, n, method="midpoint"):
    """
    General Riemann sum calculator.

    Parameters
    ----------
    f      : callable  – the integrand
    a, b   : float     – interval endpoints
    n      : int       – number of subintervals
    method : str       – 'left', 'right', or 'midpoint'

    Returns
    -------
    float  – approximate value of the integral
    """
    dx = (b - a) / n
    if method == "left":
        x = np.linspace(a, a + dx * (n - 1), n)
    elif method == "right":
        x = np.linspace(a + dx, b, n)
    elif method == "midpoint":
        x = np.linspace(a + dx / 2, b - dx / 2, n)
    else:
        raise ValueError("method must be 'left', 'right', or 'midpoint'")
    return np.sum(f(x) * dx)


def plot_riemann_rectangles(ax, f, a, b, n, method, color, alpha=0.4):
    """
    Draw Riemann rectangles on an existing Axes object.

    Parameters
    ----------
    ax     : matplotlib Axes
    f      : callable
    a, b   : float    – interval endpoints
    n      : int      – number of subintervals
    method : str      – 'left', 'right', or 'midpoint'
    color  : str      – face colour for rectangles
    alpha  : float    – transparency
    """
    dx = (b - a) / n
    for k in range(n):
        x_left = a + k * dx
        if method == "left":
            c = x_left
        elif method == "right":
            c = x_left + dx
        elif method == "midpoint":
            c = x_left + dx / 2
        height = f(c)
        rect = plt.Rectangle(
            (x_left, min(0, height)), dx, abs(height),
            edgecolor="black", facecolor=color, alpha=alpha, linewidth=0.8
        )
        ax.add_patch(rect)


# =============================================================================
# PART 1 – PROBLEM A
# f(x) = sin(x) + 1  on  [-π, π],  n = 4 subintervals
# Three separate sketches: left, right, midpoint
# =============================================================================
print("=" * 65)
print("PART 1 – Problem A")
print("f(x) = sin(x) + 1  on  [-π, π],  n = 4")
print("=" * 65)

f_A = lambda x: np.sin(x) + 1
a_A, b_A = -np.pi, np.pi
n_A = 4

exact_A, _ = integrate.quad(f_A, a_A, b_A)
print(f"Exact integral (scipy):  {exact_A:.6f}")

methods = ["left", "right", "midpoint"]
method_labels = ["Left-Hand Endpoint", "Right-Hand Endpoint", "Midpoint"]
colors = ["steelblue", "tomato", "seagreen"]

fig_A, axes_A = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
fig_A.suptitle("Part 1A: f(x) = sin(x)+1 on [−π, π]  |  n = 4 Subintervals",
               fontsize=12, fontweight="bold")

x_plot = np.linspace(a_A, b_A, 500)

for ax, method, label, color in zip(axes_A, methods, method_labels, colors):
    rs = riemann_sum(f_A, a_A, b_A, n_A, method)
    plot_riemann_rectangles(ax, f_A, a_A, b_A, n_A, method, color)
    ax.plot(x_plot, f_A(x_plot), "k-", linewidth=2, label="f(x)")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title(f"{label}\nRiemann Sum ≈ {rs:.4f}", fontsize=10)
    ax.set_xlabel("x")
    if ax == axes_A[0]:
        ax.set_ylabel("f(x)")
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", "0",
                        r"$\frac{\pi}{2}$", r"$\pi$"])
    ax.legend(fontsize=8)

    print(f"  {label:25s}:  {rs:.6f}")

print(f"  Exact value           :  {exact_A:.6f}")
plt.tight_layout()
plt.show()

# =============================================================================
# PART 1 – PROBLEM B
# f(x) = 3x + 2x²  on  [0, 1]
# Riemann sum formula (right endpoints), then limit as n → ∞
# =============================================================================
print()
print("=" * 65)
print("PART 1 – Problem B")
print("f(x) = 3x + 2x²  on  [0, 1]")
print("=" * 65)

print("""
Derivation of the Right-Hand Riemann Sum Formula:
--------------------------------------------------
  a = 0,  b = 1,  Δx = 1/n
  Right endpoint:  c_k = k/n  for k = 1, 2, ..., n

  S_n = Σ f(c_k) · Δx
       = Σ [3(k/n) + 2(k/n)²] · (1/n)
       = (3/n²) Σk + (2/n³) Σk²

  Using:  Σk = n(n+1)/2   and   Σk² = n(n+1)(2n+1)/6

  S_n = (3/n²) · [n(n+1)/2]  +  (2/n³) · [n(n+1)(2n+1)/6]
      = (3(n+1))/(2n)  +  ((n+1)(2n+1))/(3n²)

  As n → ∞:
      3(n+1)/(2n)         →  3/2
      (n+1)(2n+1)/(3n²)  →  2/3

  Exact Area = 3/2 + 2/3 = 9/6 + 4/6 = 13/6 ≈ 2.1667
""")

f_B = lambda x: 3 * x + 2 * x ** 2
a_B, b_B = 0.0, 1.0
exact_B, _ = integrate.quad(f_B, a_B, b_B)

# Verify numerically with large n
n_values = [4, 10, 100, 1000, 10000]
print("  n         Right-Sum      Error vs Exact")
print("  " + "-" * 45)
for nv in n_values:
    rs = riemann_sum(f_B, a_B, b_B, nv, "right")
    print(f"  {nv:6d}    {rs:.8f}   {abs(rs - exact_B):.2e}")
print(f"\n  Exact (scipy / formula):  {exact_B:.8f}  (= 13/6)")

# Plot the function and a 10-rectangle right-sum
fig_B, ax_B = plt.subplots(figsize=(7, 4))
x_plot = np.linspace(a_B, b_B, 500)
n_vis = 10
plot_riemann_rectangles(ax_B, f_B, a_B, b_B, n_vis, "right", "coral")
ax_B.plot(x_plot, f_B(x_plot), "k-", linewidth=2, label="f(x) = 3x + 2x²")
ax_B.fill_between(x_plot, f_B(x_plot), alpha=0.12, color="blue")
rs_vis = riemann_sum(f_B, a_B, b_B, n_vis, "right")
ax_B.set_title(
    f"Part 1B: f(x) = 3x+2x² on [0,1]  |  n={n_vis} right-endpoint rectangles\n"
    f"Riemann Sum ≈ {rs_vis:.4f}   Exact = {exact_B:.4f}  (= 13/6)",
    fontsize=10
)
ax_B.set_xlabel("x")
ax_B.set_ylabel("f(x)")
ax_B.legend()
plt.tight_layout()
plt.show()

# =============================================================================
# PART 1 – PROBLEM C
# ∫₁ᵉ ln(x) dx   (Riemann sum + exact value)
# =============================================================================
print()
print("=" * 65)
print("PART 1 – Problem C")
print("∫₁ᵉ ln(x) dx")
print("=" * 65)

f_C = lambda x: np.log(x)
a_C, b_C = 1.0, np.e
exact_C, _ = integrate.quad(f_C, a_C, b_C)

# Analytical: ∫ ln x dx = x·ln(x) − x  → [e·1 − e] − [1·0 − 1] = 0 + 1 = 1
print(f"\n  Analytical result: [x·ln(x) − x]₁ᵉ = (e·1−e) − (0−1) = 1.0")
print(f"  scipy.integrate.quad result:          {exact_C:.10f}")

n_values_C = [4, 10, 100, 1000, 100000]
print("\n  Midpoint Riemann sums (highest granularity):")
print("  n              Sum            Error")
print("  " + "-" * 50)
for nv in n_values_C:
    rs = riemann_sum(f_C, a_C, b_C, nv, "midpoint")
    print(f"  {nv:8d}      {rs:.10f}   {abs(rs - exact_C):.2e}")

# Plot with 1000 rectangles for high granularity
fig_C, axes_C = plt.subplots(1, 2, figsize=(13, 4))

# Left panel: coarse n=8 visual
n_vis_C = 8
x_plot_C = np.linspace(a_C, b_C, 500)
plot_riemann_rectangles(axes_C[0], f_C, a_C, b_C, n_vis_C, "midpoint", "mediumpurple")
axes_C[0].plot(x_plot_C, f_C(x_plot_C), "k-", linewidth=2, label="f(x) = ln(x)")
axes_C[0].fill_between(x_plot_C, f_C(x_plot_C), alpha=0.12, color="mediumpurple")
rs_vis_C = riemann_sum(f_C, a_C, b_C, n_vis_C, "midpoint")
axes_C[0].set_title(
    f"Part 1C: ∫₁ᵉ ln(x) dx  |  n={n_vis_C} midpoint rectangles\n"
    f"Sum ≈ {rs_vis_C:.5f}   Exact = {exact_C:.5f}",
    fontsize=10
)
axes_C[0].set_xlabel("x")
axes_C[0].set_ylabel("ln(x)")
axes_C[0].legend()

# Right panel: convergence curve
ns_C = np.logspace(0, 5, 200, dtype=int)
errors_C = [abs(riemann_sum(f_C, a_C, b_C, int(n), "midpoint") - exact_C)
            for n in ns_C]
axes_C[1].loglog(ns_C, errors_C, "b.-", markersize=2)
axes_C[1].set_title("Part 1C: Convergence of Midpoint Riemann Sum\n"
                    "Error vs. Number of Subintervals n", fontsize=10)
axes_C[1].set_xlabel("n (log scale)")
axes_C[1].set_ylabel("Absolute Error (log scale)")
axes_C[1].grid(True, which="both", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# =============================================================================
# PART 1 – PROBLEM D
# f(x) = x² − x³  on  [-1, 0]
# Riemann sum formula (right endpoints), then limit as n → ∞
# =============================================================================
print()
print("=" * 65)
print("PART 1 – Problem D")
print("f(x) = x² − x³  on  [-1, 0]")
print("=" * 65)

print("""
Derivation of the Right-Hand Riemann Sum Formula:
--------------------------------------------------
  a = -1,  b = 0,  Δx = 1/n
  Right endpoint:  c_k = -1 + k/n  for k = 1, 2, ..., n

  f(c_k) = c_k² − c_k³ = (-1 + k/n)² − (-1 + k/n)³

  Let u = -1 + k/n.

  S_n = Σ [u² − u³] · (1/n)

  Expanding:
    u²  = 1 − 2k/n + k²/n²
    u³  = -1 + 3k/n − 3k²/n² + k³/n³

  u² − u³ = 2 − 5k/n + 4k²/n² − k³/n³

  S_n = (1/n) Σ [2 − 5k/n + 4k²/n² − k³/n³]
      = 2 − (5/n²)·[n(n+1)/2]  +  (4/n³)·[n(n+1)(2n+1)/6]
            − (1/n⁴)·[n²(n+1)²/4]

  As n → ∞:
      2                    → 2
      5(n+1)/(2n)          → 5/2
      4(n+1)(2n+1)/(6n²)  → 8/6 = 4/3
      (n+1)²/(4n²)        → 1/4

  Exact Area = 2 − 5/2 + 4/3 − 1/4
             = 24/12 − 30/12 + 16/12 − 3/12 = 7/12 ≈ 0.5833
""")

f_D = lambda x: x ** 2 - x ** 3
a_D, b_D = -1.0, 0.0
exact_D, _ = integrate.quad(f_D, a_D, b_D)

print(f"  Exact (scipy):  {exact_D:.8f}  (= 7/12 ≈ {7/12:.8f})")

n_values_D = [4, 10, 100, 1000, 10000]
print("\n  n         Right-Sum      Error vs Exact")
print("  " + "-" * 45)
for nv in n_values_D:
    rs = riemann_sum(f_D, a_D, b_D, nv, "right")
    print(f"  {nv:6d}    {rs:.8f}   {abs(rs - exact_D):.2e}")

# Plot
fig_D, ax_D = plt.subplots(figsize=(7, 4))
x_plot_D = np.linspace(a_D, b_D, 500)
n_vis_D = 10
plot_riemann_rectangles(ax_D, f_D, a_D, b_D, n_vis_D, "right", "gold")
ax_D.plot(x_plot_D, f_D(x_plot_D), "k-", linewidth=2, label="f(x) = x²−x³")
ax_D.fill_between(x_plot_D, f_D(x_plot_D), alpha=0.15, color="gold")
rs_vis_D = riemann_sum(f_D, a_D, b_D, n_vis_D, "right")
ax_D.set_title(
    f"Part 1D: f(x) = x²−x³ on [−1, 0]  |  n={n_vis_D} right-endpoint rectangles\n"
    f"Sum ≈ {rs_vis_D:.4f}   Exact = {exact_D:.4f}  (= 7/12)",
    fontsize=10
)
ax_D.set_xlabel("x")
ax_D.set_ylabel("f(x)")
ax_D.legend()
plt.tight_layout()
plt.show()

# =============================================================================
# PART 2 – Media Server Download Rate Integration
# 30 download-rate measurements (Mbps) at 1-minute intervals → total MB
# =============================================================================
print()
print("=" * 65)
print("PART 2 – Media Server Download Rate Experiment")
print("=" * 65)

# Simulated download rates (Mbps) recorded at t = 0, 1, ..., 29 minutes
# Values mimic a realistic scenario: high burst, mid dip, recovery
download_rates_mbps = np.array([
    45.2, 47.8, 49.1, 48.6, 46.3,   # 0–4  min (ramp-up)
    44.0, 42.5, 40.1, 38.7, 36.2,   # 5–9  min (network congestion)
    35.0, 33.8, 35.5, 37.4, 39.0,   # 10–14 min (partial recovery)
    41.3, 43.6, 45.0, 46.8, 47.5,   # 15–19 min (good connection)
    48.1, 49.0, 48.3, 47.0, 45.5,   # 20–24 min (stable peak)
    44.2, 43.0, 41.8, 40.5, 39.7,   # 25–29 min (gradual decline)
])
t_minutes = np.arange(30)  # t = 0 … 29 minutes

# ---- Print the data table ----
print(f"\n  {'Minute':>6}  {'Rate (Mbps)':>12}")
print("  " + "-" * 22)
for t, r in zip(t_minutes, download_rates_mbps):
    print(f"  {t:6d}  {r:12.1f}")

# ---- Numerical integration using the trapezoidal rule ----
# R(t) = download rate in Mbps  →  Data = ∫R(t) dt  [Mbps · min]
# Convert: 1 Mbps · 1 min = 1 Mb/min · 60 s = 60 Mb = 7.5 MB
#   or equivalently: Mbps * min * 60 s/min / 8 bits = MB

total_mbps_min = np.trapezoid(download_rates_mbps, t_minutes)      # Mbps·min
total_megabits = total_mbps_min * 60                                 # Megabits
total_megabytes = total_megabits / 8                                 # Megabytes (MB)
total_gigabytes = total_megabytes / 1024                             # Gigabytes (GB)

print(f"\n  Trapezoidal Integration Result:")
print(f"  ∫₀³⁰ R(t) dt  =  {total_mbps_min:.2f}  Mbps·min")
print(f"                =  {total_megabits:.2f}  Megabits")
print(f"                =  {total_megabytes:.2f}  Megabytes (MB)")
print(f"                =  {total_gigabytes:.4f}  Gigabytes (GB)")
print()
print("  Interpretation:")
print(f"  Over the 30-minute experiment the media server transferred")
print(f"  approximately {total_megabytes:.1f} MB ({total_gigabytes:.3f} GB) of data.")
print(f"  This is computed by integrating the download rate function")
print(f"  R(t) across the experiment window [0, 30] minutes,")
print(f"  treating each 1-minute interval as one Riemann slab.")

# ---- Also apply our custom Riemann function for comparison ----
# Fit a continuous interpolant so riemann_sum() can sample arbitrary points
from scipy.interpolate import interp1d
R_continuous = interp1d(t_minutes, download_rates_mbps, kind="cubic",
                        fill_value="extrapolate")

print("\n  Comparison: Custom Riemann Sum vs. Trapezoid on interpolated R(t)")
print(f"  {'Method':20s}  {'∫R(t)dt (Mbps·min)':>22}  {'Total MB':>10}")
print("  " + "-" * 60)
for meth in ["left", "right", "midpoint"]:
    rs_p2 = riemann_sum(R_continuous, 0, 30, 30, meth)
    mb_p2 = rs_p2 * 60 / 8
    print(f"  {meth.capitalize():20s}  {rs_p2:22.4f}  {mb_p2:10.2f}")
trap_mb = total_megabytes
print(f"  {'Trapezoid (numpy)':20s}  {total_mbps_min:22.4f}  {trap_mb:10.2f}")

# ---- Plot Part 2 ----
fig_P2, axes_P2 = plt.subplots(1, 2, figsize=(13, 4))

# Left: download rate over time with shaded area
ax_p2a = axes_P2[0]
ax_p2a.bar(t_minutes, download_rates_mbps, color="steelblue", alpha=0.7,
           edgecolor="white", width=0.8, label="Rate (Mbps)")
t_fine = np.linspace(0, 29, 500)
ax_p2a.plot(t_fine, R_continuous(t_fine), "r-", linewidth=1.5, label="R(t) interpolated")
ax_p2a.fill_between(t_fine, R_continuous(t_fine), alpha=0.15, color="red")
ax_p2a.set_title(f"Part 2: Download Rate R(t)  |  ∫R dt ≈ {total_mbps_min:.1f} Mbps·min\n"
                 f"Total data transferred ≈ {total_megabytes:.1f} MB",
                 fontsize=10)
ax_p2a.set_xlabel("Time (minutes)")
ax_p2a.set_ylabel("Download Rate (Mbps)")
ax_p2a.legend()
ax_p2a.set_xlim(-0.5, 29.5)

# Right: cumulative data transferred
cumulative_mbps_min = np.array([np.trapezoid(download_rates_mbps[:k+1], t_minutes[:k+1])
                                 for k in range(len(t_minutes))])
cumulative_mb = cumulative_mbps_min * 60 / 8
ax_p2b = axes_P2[1]
ax_p2b.plot(t_minutes, cumulative_mb, "g-o", markersize=4, linewidth=2)
ax_p2b.set_title(f"Part 2: Cumulative Data Downloaded\nFinal ≈ {total_megabytes:.1f} MB",
                 fontsize=10)
ax_p2b.set_xlabel("Time (minutes)")
ax_p2b.set_ylabel("Cumulative Data (MB)")
ax_p2b.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

print()
print("=" * 65)
print("All computations and plots complete.")
print("=" * 65)