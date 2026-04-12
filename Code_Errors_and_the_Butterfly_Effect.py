"""
============================================================
CST-305: Project 7 – Code Errors and the Butterfly Effect
Combined Script – All Parts in One File

Programmers : Chance
Course      : CST-305, Grand Canyon University
Packages    : numpy, scipy, matplotlib

Approach:
  Startup – Terminal Parameter Input
    - Full parameter explainer printed before any window opens
    - User enters sigma, rho, beta, initial conditions, integration
      time, and animation FPS; pressing Enter accepts the classic default

  Part 1 – Lorenz Attractor  (uses terminal-entered values)
    - Fig 1 : Static attractor with user parameters + time colorbar
    - Fig 2 : 2x2 grid showing attractor shape across parameter regimes
    - Fig 10: Animated attractor driven by fig.canvas.new_timer()
              (works correctly in PyCharm's SciView panel).
              Sliders adjust sigma/rho/beta; Re-Animate replays;
              Clear wipes the canvas; info panel shows chaos status.

  Butterfly Effect
    - Fig 3: Side-by-side 3D trajectories from IC and IC+epsilon
    - Fig 4: Euclidean divergence vs time on a log scale
    - Fig 5: Code-error analog – running sum with per-step truncation

  Part 2 – M/M/1 Queue Analysis
    - Fig 6: Q1 – M/M/1/K buffer overflow probability
    - Fig 7: Q2 – Scaling analysis (lambda, mu scaled by k)
    - Fig 8: Q3 – Maximum arrival rate for E[TQ] < 6 min
    - Fig 9: Q4 – Server farm simulation across 6 dispatching policies

Usage (PyCharm):
    Hit the green Play button. Answer the terminal prompts, then each
    figure appears in PyCharm's built-in plot panel one at a time.
    Close each window to advance. The animated Lorenz window opens last
    and stays open so you can use the sliders.

Requirements:
    pip install numpy scipy matplotlib
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from scipy.integrate import solve_ivp
from collections import deque
import random
import heapq

# ── Colour palette ────────────────────────────────────────────────────────────
BG  = "#07090f"
BG2 = "#0f1420"
FG  = "#e8eaf6"
C1  = "#4fc3f7"   # sigma – cyan
C2  = "#ef5350"   # rho   – red
C3  = "#66bb6a"   # beta  – green
C4  = "#ffb347"   # orange
C5  = "#c084fc"   # purple
C6  = "#fb7185"   # pink

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   "#2a2a3e",
    "text.color":       FG,
    "axes.labelcolor":  FG,
    "xtick.color":      "#4a5568",
    "ytick.color":      "#4a5568",
    "grid.color":       "#151a28",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
})

CMAP = plt.get_cmap("plasma")


# =============================================================================
#  1.  TERMINAL PARAMETER INPUT
# =============================================================================
_EXPLAINER = """
╔══════════════════════════════════════════════════════════════════════════╗
║         CST-305  ·  Lorenz Attractor  ·  Interactive Visualizer         ║
╚══════════════════════════════════════════════════════════════════════════╝

  The Lorenz system models atmospheric convection with three coupled ODEs:

      dx/dt = σ(y − x)           [sigma]
      dy/dt = x(ρ − z) − y       [rho]
      dz/dt = xy − βz            [beta]

  ┌─────────────────────────────────────────────────────────────────────┐
  │  σ (sigma) – Prandtl number                                         │
  │    Ratio of fluid viscosity to thermal diffusivity.                 │
  │    Controls how tightly x and y are coupled.                        │
  │    Classic value: 10.  Higher σ → faster spiralling.                │
  ├─────────────────────────────────────────────────────────────────────┤
  │  ρ (rho)   – Rayleigh number                                        │
  │    Ratio of buoyancy force to viscous damping.                      │
  │    THE chaos driver.  Stable below ρ ≈ 1, periodic 1–24.74,        │
  │    and fully chaotic above 24.74.  Classic value: 28.               │
  ├─────────────────────────────────────────────────────────────────────┤
  │  β (beta)  – Geometric factor                                       │
  │    Aspect ratio of the convection cells; controls energy            │
  │    dissipation in the z-direction. Classic value: 8/3 ≈ 2.667.     │
  └─────────────────────────────────────────────────────────────────────┘

  Press Enter at any prompt to accept the classic default.
"""


def _ask_float(label, default):
    raw = input(f"  {label} [default {default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"    Bad input – using default {default}")
        return default


def _ask_int(label, default):
    raw = input(f"  {label} [default {default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"    Bad input – using default {default}")
        return default


def get_user_parameters():
    """Print the explainer, collect parameters interactively, return them."""
    print(_EXPLAINER)
    print("─" * 72)
    print("  Enter Lorenz parameters  (Enter = accept default)")
    print("─" * 72)
    sigma = _ask_float("σ  sigma  (Prandtl number,   typical 10   )", 10.0)
    rho   = _ask_float("ρ  rho    (Rayleigh number,  typical 28   )", 28.0)
    beta  = _ask_float("β  beta   (geometric factor, typical 2.667)", 8.0 / 3.0)
    print()
    print("  Initial conditions:")
    x0 = _ask_float("  x0", 1.0)
    y0 = _ask_float("  y0", 1.0)
    z0 = _ask_float("  z0", 1.0)
    print()
    T   = _ask_float("Integration time T in seconds  (typical 50)", 50.0)
    fps = _ask_int  ("Animation speed in FPS  (1=slow  60=fast  typical 30)", 30)
    print()
    print("─" * 72)
    print(f"  σ={sigma}  ρ={rho}  β={beta:.4f}  IC=({x0},{y0},{z0})  T={T}  FPS={fps}")
    print("─" * 72)
    print()
    return sigma, rho, beta, (x0, y0, z0), T, fps


# =============================================================================
#  2.  LORENZ ODE CORE
# =============================================================================
DEFAULT_SIGMA = 10.0
DEFAULT_RHO   = 28.0
DEFAULT_BETA  = 8.0 / 3.0
T_SPAN_DEF    = (0, 50)
IC_BASE       = np.array([1.0, 1.0, 1.0])


def lorenz(t, state, sigma, rho, beta):
    """Lorenz ODE: returns [dx/dt, dy/dt, dz/dt]."""
    x, y, z = state
    return [
        sigma * (y - x),        # dx/dt = σ(y − x)
        x * (rho - z) - y,      # dy/dt = x(ρ − z) − y
        x * y - beta * z,       # dz/dt = xy − βz
    ]


def solve_lorenz(sigma, rho, beta, ic=None, t_span=None, n_pts=10_000):
    """Integrate Lorenz ODE with RK45; return (x, y, z) arrays."""
    if ic is None:
        ic = IC_BASE.copy()
    if t_span is None:
        t_span = T_SPAN_DEF
    t_eval = np.linspace(t_span[0], t_span[1], n_pts)
    sol = solve_ivp(
        lorenz, t_span, list(ic),
        args=(sigma, rho, beta),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-9, atol=1e-12,
    )
    return sol.y[0], sol.y[1], sol.y[2]


# ── shared 3-D axis styling ───────────────────────────────────────────────────
def _style3d(ax, title=""):
    ax.set_facecolor(BG)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#1e2435")
    ax.tick_params(colors="#4a5568", labelsize=7)
    ax.set_xlabel("X", color=C1, labelpad=2)
    ax.set_ylabel("Y", color=C1, labelpad=2)
    ax.set_zlabel("Z", color=C1, labelpad=2)
    ax.grid(False)
    if title:
        ax.set_title(title, color=FG, fontsize=9, pad=4)


def _draw_traj(ax, x, y, z, cmap=None):
    """Render full trajectory colour-coded by time."""
    if cmap is None:
        cmap = CMAP
    n = len(x)
    cols = cmap(np.linspace(0, 1, n - 1))
    for i in range(n - 1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2],
                color=cols[i], lw=0.35, alpha=0.9)


# =============================================================================
#  3.  FIGURE 1 – Static Lorenz attractor (user parameters)
# =============================================================================
def plot_lorenz_static(sigma, rho, beta, ic, T):
    print("Plotting Figure 1: Lorenz Attractor – your parameters")
    x, y, z = solve_lorenz(sigma, rho, beta, ic=np.array(ic),
                            t_span=(0, T))
    fig = plt.figure(figsize=(10, 7), facecolor=BG)
    fig.suptitle(
        f"Lorenz Attractor  |  σ={sigma}  ρ={rho}  β={beta:.3f}  "
        f"IC=({ic[0]},{ic[1]},{ic[2]})  T={T}",
        color=FG, fontsize=11)
    ax = fig.add_subplot(111, projection="3d")
    _style3d(ax, "Colour = time  (purple=early  yellow=late)")
    _draw_traj(ax, x, y, z)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, T))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
    cb.set_label("Time t", color=FG, fontsize=9)
    cb.ax.yaxis.set_tick_params(color="#4a5568")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#4a5568", fontsize=8)
    plt.tight_layout()
    plt.show()


# =============================================================================
#  4.  FIGURE 2 – Parameter variation grid
# =============================================================================
def plot_lorenz_variants():
    print("Plotting Figure 2: Lorenz Attractor – parameter variants")
    configs = [
        (10.0, 10.0, 8/3, "rho=10  (periodic – below chaos threshold)"),
        (10.0, 28.0, 8/3, "rho=28  (classic chaotic attractor)"),
        (10.0, 99.9, 8/3, "rho=99.9  (highly chaotic)"),
        (5.0,  28.0, 8/3, "sigma=5  (lower Prandtl – weaker x-y coupling)"),
    ]
    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    fig.suptitle("Lorenz Attractor – Parameter Variations", color=FG, fontsize=13)
    for idx, (sig, rho, beta, title) in enumerate(configs, 1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        x, y, z = solve_lorenz(sig, rho, beta)
        _style3d(ax, title=title)
        _draw_traj(ax, x, y, z)
    plt.tight_layout()
    plt.show()


# =============================================================================
#  5.  FIGURE 3 – Butterfly Effect: diverging 3D trajectories
# =============================================================================
EPSILON = 1e-5


def plot_butterfly_3d():
    print("Plotting Figure 3: Butterfly Effect – diverging trajectories")
    T_BF = (0, 40)
    ic2  = IC_BASE.copy()
    ic2[0] += EPSILON

    x1, y1, z1 = solve_lorenz(DEFAULT_SIGMA, DEFAULT_RHO, DEFAULT_BETA,
                               ic=IC_BASE.copy(), t_span=T_BF, n_pts=8000)
    x2, y2, z2 = solve_lorenz(DEFAULT_SIGMA, DEFAULT_RHO, DEFAULT_BETA,
                               ic=ic2, t_span=T_BF, n_pts=8000)
    t_eval = np.linspace(*T_BF, 8000)

    fig = plt.figure(figsize=(14, 6), facecolor=BG)
    fig.suptitle(
        f"Butterfly Effect – Trajectories Diverging from ε = {EPSILON:.0e} Perturbation",
        color=FG, fontsize=12)
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    _style3d(ax1, "Baseline  IC = (1, 1, 1)")
    _draw_traj(ax1, x1, y1, z1, cmap=plt.get_cmap("cool"))
    _style3d(ax2, f"Perturbed  IC = (1+ε, 1, 1)")
    _draw_traj(ax2, x2, y2, z2, cmap=plt.get_cmap("hot"))
    plt.tight_layout()
    plt.show()

    return (x1, y1, z1), (x2, y2, z2), t_eval


# =============================================================================
#  6.  FIGURE 4 – Butterfly Effect: divergence growth
# =============================================================================
def plot_butterfly_divergence(traj1, traj2, t_eval):
    print("Plotting Figure 4: Butterfly Effect – divergence growth")
    x1, y1, z1 = traj1
    x2, y2, z2 = traj2
    div = np.linalg.norm(np.array([x2-x1, y2-y1, z2-z1]), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.semilogy(t_eval, div + 1e-16, color=C2, lw=1.8,
                label="||Δstate||  (baseline vs perturbed)")
    idx = np.argmax(div > 5.0)
    if idx:
        ax.axvline(t_eval[idx], color=C3, ls="--", lw=1.5,
                   label=f"Macro-divergence at t≈{t_eval[idx]:.1f}")
    ax.set_xlabel("Time t")
    ax.set_ylabel("||Δstate||  (log scale)")
    ax.set_title(f"Divergence of Lorenz Trajectories  (ε={EPSILON:.0e})", color=FG)
    ax.legend(facecolor=BG2, labelcolor=FG, fontsize=9)
    ax.grid(True)
    ax.tick_params(colors="#4a5568")
    for s in ax.spines.values():
        s.set_edgecolor("#2a2a3e")
    plt.tight_layout()
    plt.show()


# =============================================================================
#  7.  FIGURE 5 – Code-error butterfly effect analog
# =============================================================================
def _buggy_sum(values, dp=2):
    """Running sum where each step is rounded – simulates a code bug."""
    total = 0.0
    for v in values:
        total += v
        total = round(total, dp)   # << the bug: rounding error per step
    return total


def plot_code_error():
    print("Plotting Figure 5: Code Error – butterfly effect analog")
    rng  = np.random.default_rng(0)
    vals = rng.uniform(0.001, 0.999, size=1000)
    ns   = np.arange(1, len(vals) + 1)

    correct = np.cumsum(vals)
    buggy   = np.array([_buggy_sum(vals[:n]) for n in ns])
    err     = np.abs(correct - buggy)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(10, 7),
                                    facecolor=BG, sharex=True)
    fig.suptitle("Butterfly Effect in Code – Per-Step Truncation Error",
                 color=FG, fontsize=12)
    for ax in (axA, axB):
        ax.set_facecolor(BG)
        ax.tick_params(colors="#4a5568")
        for s in ax.spines.values():
            s.set_edgecolor("#2a2a3e")
        ax.grid(True)

    axA.plot(ns, correct, color=C1, lw=1.4, label="Correct sum")
    axA.plot(ns, buggy,   color=C2, lw=1.4, ls="--",
             label="Buggy sum  (round to 2 dp per step)")
    axA.set_ylabel("Running Sum")
    axA.set_title("Correct vs Buggy Running Sum", color=FG)
    axA.legend(facecolor=BG2, labelcolor=FG, fontsize=9)

    axB.semilogy(ns, err + 1e-15, color=C4, lw=1.4)
    axB.set_xlabel("Elements Processed")
    axB.set_ylabel("|Error|  (log scale)")
    axB.set_title("Accumulated Error – Mirrors Lorenz Divergence", color=FG)
    plt.tight_layout()
    plt.show()


# =============================================================================
#  8.  M/M/1 QUEUE HELPERS
# =============================================================================
def _mm1k(lam, mu, K):
    """P(overflow) = P_K for M/M/1/K queue (closed form)."""
    rho = lam / mu
    P0  = (1.0 / (K + 1)) if abs(rho - 1) < 1e-9 \
          else (1 - rho) / (1 - rho ** (K + 1))
    return rho, P0 * rho ** K


def _mm1_metrics(lam, mu):
    rho = lam / mu
    return rho, lam, rho / (1 - rho), 1 / (mu - lam)


# =============================================================================
#  9.  FIGURE 6 – Q1: M/M/1/K overflow
# =============================================================================
def plot_q1():
    print("Plotting Figure 6: Q1 – M/M/1/K overflow probability")
    lam, mu, K_given = 125, 500, 12
    rho, PK  = _mm1k(lam, mu, K_given)
    K_needed = next(K for K in range(1, 200) if _mm1k(lam, mu, K)[1] < 1e-6)
    print(f"  rho={rho:.4f}  P(overflow|K=12)={PK:.4e}  K_needed={K_needed}")

    Ks    = np.arange(1, 45)
    probs = [_mm1k(lam, mu, k)[1] for k in Ks]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.semilogy(Ks, probs, color=C1, lw=2, marker="o", ms=4)
    ax.axvline(K_given,  color=C2, ls="--", lw=1.8,
               label=f"K={K_given}  P_loss={PK:.2e}")
    ax.axvline(K_needed, color=C3, ls="--", lw=1.8,
               label=f"K={K_needed}  P_loss<1 ppm")
    ax.axhline(1e-6, color=C4, ls=":", lw=1.2, label="1 ppm target")
    ax.fill_between(Ks, probs, 1e-6,
                    where=[p > 1e-6 for p in probs],
                    color=C2, alpha=0.08)
    ax.annotate(f"P(K=12)={PK:.2e}",
                xy=(K_given, PK), xytext=(K_given+4, PK*12),
                color=C2, fontsize=8,
                arrowprops=dict(arrowstyle="->", color=C2))
    ax.set_xlabel("Buffer Size K")
    ax.set_ylabel("P(overflow)  [log scale]")
    ax.set_title(f"Q1 – M/M/1/K Overflow  (λ={lam} pps  μ={mu} pps  ρ={rho})",
                 color=FG)
    ax.legend(facecolor=BG2, labelcolor=FG, fontsize=9)
    ax.grid(True)
    ax.tick_params(colors="#4a5568")
    for s in ax.spines.values():
        s.set_edgecolor("#2a2a3e")
    plt.tight_layout()
    plt.show()


# =============================================================================
#  10.  FIGURE 7 – Q2: Scaling analysis
# =============================================================================
def plot_q2():
    print("Plotting Figure 7: Q2 – Scaling analysis")
    lam0, mu0 = 100, 200
    k_vals = np.linspace(0.5, 5, 300)
    rhos, Xs, ENs, ETs = [], [], [], []
    for k in k_vals:
        r, x, en, et = _mm1_metrics(k * lam0, k * mu0)
        rhos.append(r); Xs.append(x); ENs.append(en); ETs.append(et)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=BG)
    fig.suptitle("Q2 – Scaling Both λ and μ by k  (M/M/1, base ρ=0.5)",
                 color=FG, fontsize=12)
    specs = [
        ("Utilization  ρ",          rhos, C1, "UNCHANGED – ratio λ/μ invariant"),
        ("Throughput X  (jobs/s)",  Xs,   C2, "SCALES BY k – linear"),
        ("Mean # in system  E[N]",  ENs,  C3, "UNCHANGED – depends only on ρ"),
        ("Mean sojourn time  E[T]", ETs,  C4, "SCALES BY 1/k – faster service"),
    ]
    for ax, (title, data, col, note) in zip(axes.flat, specs):
        ax.set_facecolor(BG)
        ax.plot(k_vals, data, color=col, lw=2)
        ax.set_xlabel("Scale factor k")
        ax.set_title(title, color=FG, fontsize=10)
        ax.text(0.98, 0.05, note, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="#4a5568")
        ax.tick_params(colors="#4a5568")
        for s in ax.spines.values():
            s.set_edgecolor("#2a2a3e")
        ax.grid(True)
    plt.tight_layout()
    plt.show()


# =============================================================================
#  11.  FIGURE 8 – Q3: Maximum arrival rate
# =============================================================================
def plot_q3():
    print("Plotting Figure 8: Q3 – Maximum arrival rate")
    mu      = 1 / 3
    limit   = 6
    rho_max = (limit * mu) / (1 + limit * mu)
    lam_max = rho_max * mu
    print(f"  λ_max={lam_max:.4f} jobs/min  ρ_max={rho_max:.4f}")

    lams = np.linspace(0.01, mu - 0.0001, 500)
    rhos = lams / mu
    ETQs = rhos / (mu * (1 - rhos))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
    fig.suptitle("Q3 – Max Arrival Rate  (μ=1/3 job/min  E[TQ]<6 min)",
                 color=FG, fontsize=12)

    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.tick_params(colors="#4a5568")
        for s in ax.spines.values():
            s.set_edgecolor("#2a2a3e")
        ax.grid(True)

    ax1.plot(lams, ETQs, color=C1, lw=2)
    ax1.axhline(limit,   color=C2, ls="--", lw=1.8, label=f"E[TQ]={limit} min limit")
    ax1.axvline(lam_max, color=C3, ls="--", lw=1.8, label=f"λ_max={lam_max:.4f}")
    ax1.fill_between(lams, ETQs, limit,
                     where=ETQs > limit, color=C2, alpha=0.12,
                     label="Constraint violated")
    ax1.set_ylim(0, 30)
    ax1.set_xlabel("Arrival Rate λ  (jobs/min)")
    ax1.set_ylabel("E[TQ]  (min)")
    ax1.set_title("Mean Waiting Time vs Arrival Rate", color=FG)
    ax1.legend(facecolor=BG2, labelcolor=FG, fontsize=8)

    ax2.plot(lams, rhos, color=C4, lw=2)
    ax2.axvline(lam_max, color=C3, ls="--", lw=1.8,
                label=f"λ_max={lam_max:.4f}  ρ={rho_max:.4f}")
    ax2.axhline(rho_max, color=C5, ls=":", lw=1.5)
    ax2.set_xlabel("Arrival Rate λ  (jobs/min)")
    ax2.set_ylabel("Utilization ρ")
    ax2.set_title("Utilization vs Arrival Rate", color=FG)
    ax2.legend(facecolor=BG2, labelcolor=FG, fontsize=8)

    plt.tight_layout()
    plt.show()


# =============================================================================
#  12.  DISCRETE-EVENT SIMULATION (Q4)
# =============================================================================
class _Event:
    __slots__ = ("time", "kind", "host", "response_time")
    def __init__(self, time, kind, host, response_time=0):
        self.time = time; self.kind = kind
        self.host = host; self.response_time = response_time
    def __lt__(self, other): return self.time < other.time


def _simulate_farm(policy, n_hosts=3, lam=10.0, mu=5.0, n_jobs=50_000, seed=42):
    rng       = random.Random(seed)
    queues    = [deque() for _ in range(n_hosts)]
    busy      = [False] * n_hosts
    work_left = [0.0]   * n_hosts
    heap      = []
    t = 0.0; completed = 0; total_resp = 0.0; total_arr = 0
    rr = [0]
    sita_lo = -np.log(2/3) / mu
    sita_hi = -np.log(1/3) / mu

    def pick(svc):
        if policy == "Random":        return rng.randrange(n_hosts)
        if policy == "Round-Robin":   h = rr[0] % n_hosts; rr[0] += 1; return h
        if policy == "Shortest-Queue":return min(range(n_hosts), key=lambda h: len(queues[h])+int(busy[h]))
        if policy == "SITA":          return 0 if svc < sita_lo else (1 if svc < sita_hi else 2)
        if policy == "LWL":           return min(range(n_hosts), key=lambda h: work_left[h])
        if policy == "Central-Queue":
            free = [h for h in range(n_hosts) if not busy[h]]
            return free[0] if free else min(range(n_hosts), key=lambda h: len(queues[h]))
        raise ValueError(policy)

    heapq.heappush(heap, _Event(rng.expovariate(lam), "arrival", -1))
    while heap and completed < n_jobs:
        ev = heapq.heappop(heap); t = ev.time
        if ev.kind == "arrival":
            total_arr += 1
            svc = rng.expovariate(mu); h = pick(svc); arr = t
            if busy[h]:
                queues[h].append((arr, svc)); work_left[h] += svc
            else:
                busy[h] = True; work_left[h] = svc
                heapq.heappush(heap, _Event(t+svc, "departure", h, response_time=svc))
            if total_arr < n_jobs + 5000:
                heapq.heappush(heap, _Event(t+rng.expovariate(lam), "arrival", -1))
        else:
            h = ev.host; completed += 1; total_resp += ev.response_time
            work_left[h] = max(0.0, work_left[h] - ev.response_time)
            if queues[h]:
                arr, svc = queues[h].popleft(); work_left[h] = svc
                heapq.heappush(heap, _Event(t+svc, "departure", h, response_time=t-arr+svc))
            else:
                busy[h] = False
    return total_resp / max(completed, 1)


# =============================================================================
#  13.  FIGURE 9 – Q4: Dispatching policy comparison
# =============================================================================
def plot_q4():
    print("Plotting Figure 9: Q4 – Server farm dispatching (simulating…)")
    policies = ["Random", "Round-Robin", "Shortest-Queue", "SITA", "LWL", "Central-Queue"]
    colors   = [C1, C2, C3, C4, C5, C6]
    lam, mu  = 10.0, 5.0
    means    = {}
    for pol in policies:
        rt = _simulate_farm(pol, lam=lam, mu=mu)
        means[pol] = rt
        print(f"  {pol:<20s}  E[T]={rt:.4f}")
    best = min(means, key=means.get)
    print(f"  Best policy: {best}")
    base = means["Random"]
    imps = {p: (base - v) / base * 100 for p, v in means.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig.suptitle("Q4 – Server Farm Dispatching  (3 hosts  λ=10  μ=5  50 000 jobs)",
                 color=FG, fontsize=12)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.tick_params(axis="x", rotation=20, colors="#4a5568")
        ax.tick_params(axis="y", colors="#4a5568")
        ax.grid(True, axis="y")
        for s in ax.spines.values(): s.set_edgecolor("#2a2a3e")

    bars = ax1.bar(list(means.keys()), list(means.values()),
                   color=colors, width=0.6, edgecolor="none")
    for bar, val in zip(bars, means.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.003,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8, color=FG)
    ax1.set_ylabel("Mean Response Time E[T]")
    ax1.set_title("Mean Response Time per Policy", color=FG)

    ibars = ax2.bar(list(imps.keys()), list(imps.values()),
                    color=colors, width=0.6, edgecolor="none")
    for bar, val in zip(ibars, imps.values()):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 val + 0.2 if val >= 0 else val - 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8, color=FG)
    ax2.axhline(0, color="#4a5568", lw=0.8)
    ax2.set_ylabel("% Improvement over Random")
    ax2.set_title("Relative Improvement vs Random Baseline", color=FG)

    plt.tight_layout()
    plt.show()


# =============================================================================
#  14.  FIGURE 10 – Animated Interactive Lorenz (timer-based, PyCharm-safe)
# =============================================================================
def _build_info_panel(ax_info, sigma, rho, beta, ic):
    """Redraw the right-hand parameter info panel."""
    ax_info.cla()
    ax_info.set_facecolor(BG2)
    ax_info.set_xticks([]); ax_info.set_yticks([])
    for s in ax_info.spines.values():
        s.set_edgecolor("#1e2435")

    rows = [
        ("LORENZ SYSTEM",              FG,        11, True),
        ("",                           None,       6,  False),
        ("dx/dt = σ(y − x)",           C1,         9,  False),
        ("dy/dt = x(ρ − z) − y",       C2,         9,  False),
        ("dz/dt = xy − βz",            C3,         9,  False),
        ("",                           None,       6,  False),
        ("PARAMETERS",                 FG,         9,  True),
        ("",                           None,       5,  False),
        (f"σ = {sigma:.3f}",           C1,         9,  False),
        ("  Prandtl number",           "#7986cb",  7,  False),
        ("  viscosity/diffusivity",    "#4a5568",  7,  False),
        ("",                           None,       4,  False),
        (f"ρ = {rho:.3f}",             C2,         9,  False),
        ("  Rayleigh number",          "#ef9a9a",  7,  False),
        ("  buoyancy vs viscosity",    "#4a5568",  7,  False),
        (f"  chaos onset: ρ > 24.74",
         C2 if rho > 24.74 else "#4a5568",          7,  False),
        ("",                           None,       4,  False),
        (f"β = {beta:.4f}",            C3,         9,  False),
        ("  Geometric factor",         "#a5d6a7",  7,  False),
        ("  convection cell shape",    "#4a5568",  7,  False),
        ("",                           None,       5,  False),
        ("INITIAL CONDITIONS",         FG,         8,  True),
        (f"  x0={ic[0]:.2f}  y0={ic[1]:.2f}  z0={ic[2]:.2f}",
         FG, 8, False),
        ("",                           None,       5,  False),
        ("STATUS",                     FG,         8,  True),
        ("  CHAOTIC" if rho > 24.74 else "  PERIODIC/STABLE",
         C2 if rho > 24.74 else C3,   9,  True),
    ]

    y = 0.97
    for text, color, size, bold in rows:
        if color is None:
            y -= 0.01
            continue
        ax_info.text(0.06, y, text,
                     transform=ax_info.transAxes,
                     color=color, fontsize=size, va="top",
                     fontfamily="monospace",
                     fontweight="bold" if bold else "normal")
        y -= size * 0.013 + 0.004


class _LorenzAnimator:
    """
    Timer-driven animated Lorenz attractor.

    Uses fig.canvas.new_timer() instead of FuncAnimation so that
    PyCharm's SciView panel receives draw events correctly and the
    trajectory appears incrementally rather than staying blank.
    """

    def __init__(self, fig, ax3d, ax_info,
                 ax_sl_sig, ax_sl_rho, ax_sl_bet,
                 ax_btn_run, ax_btn_clr,
                 sigma, rho, beta, ic, T, fps):

        self.fig     = fig
        self.ax3d    = ax3d
        self.ax_info = ax_info
        self.ic      = ic
        self.T       = T
        self.fps     = fps
        self.sigma   = sigma
        self.rho     = rho
        self.beta    = beta

        self._timer   = None   # matplotlib timer handle
        self._frame   = 0      # current animation frame index
        self._indices = []     # list of (start_i, end_i) segment indices
        self._x = self._y = self._z = None
        self._cols = None

        # ── sliders ───────────────────────────────────────────────────────────
        self.sl_sig = Slider(ax_sl_sig, "", 1,   28,  valinit=sigma, color=C1)
        self.sl_rho = Slider(ax_sl_rho, "", 1,   60,  valinit=rho,   color=C2)
        self.sl_bet = Slider(ax_sl_bet, "", 0.1,  5,  valinit=beta,  color=C3)
        for sl in (self.sl_sig, self.sl_rho, self.sl_bet):
            sl.label.set_color(FG)
            sl.valtext.set_color(FG)
            sl.valtext.set_fontfamily("monospace")

        # ── buttons ───────────────────────────────────────────────────────────
        self.btn_run = Button(ax_btn_run, "Re-Animate",
                              color="#1a2035", hovercolor="#263050")
        self.btn_clr = Button(ax_btn_clr, "Clear",
                              color="#1a2035", hovercolor="#263050")
        for btn in (self.btn_run, self.btn_clr):
            btn.label.set_color(FG)
            btn.label.set_fontfamily("monospace")

        # ── wire callbacks ────────────────────────────────────────────────────
        self.btn_run.on_clicked(self._on_reanimate)
        self.btn_clr.on_clicked(self._on_clear)
        self.sl_sig.on_changed(self._on_slider)
        self.sl_rho.on_changed(self._on_slider)
        self.sl_bet.on_changed(self._on_slider)

        # ── kick off ──────────────────────────────────────────────────────────
        self._start()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _stop_timer(self):
        if self._timer is not None:
            try:
                self._timer.stop()
            except Exception:
                pass
            self._timer = None

    def _clear_ax(self):
        self.ax3d.cla()
        _style3d(self.ax3d,
                 title=f"σ={self.sigma:.2f}  ρ={self.rho:.2f}  β={self.beta:.3f}")

    def _start(self):
        """Integrate ODE, prepare frame data, start timer."""
        self._stop_timer()
        self._read_sliders()
        self._clear_ax()
        _build_info_panel(self.ax_info, self.sigma, self.rho, self.beta, self.ic)
        self.fig.canvas.draw()

        # integrate
        x, y, z = solve_lorenz(self.sigma, self.rho, self.beta,
                                ic=np.array(self.ic),
                                t_span=(0, self.T), n_pts=10_000)
        self._x, self._y, self._z = x, y, z

        # set axis limits from full solution
        pad = 2
        self.ax3d.set_xlim(x.min()-pad, x.max()+pad)
        self.ax3d.set_ylim(y.min()-pad, y.max()+pad)
        self.ax3d.set_zlim(z.min()-pad, z.max()+pad)

        # build colour array
        n = len(x)
        self._cols = CMAP(np.linspace(0, 1, n - 1))

        # decide how many points to draw per frame so animation
        # completes in roughly 8 seconds regardless of FPS
        target_frames = max(60, self.fps * 8)
        self._step  = max(1, n // target_frames)
        self._indices = list(range(0, n - self._step, self._step))
        self._frame = 0

        # start timer  (interval in ms)
        interval = max(1, int(1000 / self.fps))
        self._timer = self.fig.canvas.new_timer(interval=interval)
        self._timer.add_callback(self._tick)
        self._timer.start()

    def _tick(self):
        """Called by the timer every interval ms – draws one batch of segments."""
        if self._frame >= len(self._indices):
            self._stop_timer()
            return

        i    = self._indices[self._frame]
        end  = min(i + self._step + 1, len(self._x))
        self.ax3d.plot(
            self._x[i:end],
            self._y[i:end],
            self._z[i:end],
            color=self._cols[min(i, len(self._cols)-1)],
            lw=0.6, alpha=0.9,
        )
        self.fig.canvas.draw()
        self._frame += 1

    def _read_sliders(self):
        self.sigma = self.sl_sig.val
        self.rho   = self.sl_rho.val
        self.beta  = self.sl_bet.val

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_reanimate(self, _event):
        self._start()

    def _on_clear(self, _event):
        self._stop_timer()
        self._clear_ax()
        self.fig.canvas.draw()

    def _on_slider(self, _val):
        # live-update info panel while dragging; replay on Re-Animate
        _build_info_panel(self.ax_info,
                          self.sl_sig.val, self.sl_rho.val, self.sl_bet.val,
                          self.ic)
        self.fig.canvas.draw_idle()


def plot_animated_lorenz(sigma, rho, beta, ic, T, fps):
    print("Opening Figure 10: Animated Interactive Lorenz Attractor")

    fig = plt.figure(figsize=(15, 8.5), facecolor=BG)
    fig.suptitle("Lorenz Attractor – Animated Interactive Explorer",
                 color=FG, fontsize=13, y=0.98, fontfamily="monospace")

    # ── layout: 3D plot (left 75%) + info panel (right 25%) ──────────────────
    gs = gridspec.GridSpec(1, 2, figure=fig,
                           width_ratios=[3, 1],
                           left=0.03, right=0.97,
                           top=0.93, bottom=0.18,
                           wspace=0.04)
    ax3d    = fig.add_subplot(gs[0], projection="3d")
    ax_info = fig.add_subplot(gs[1])
    ax_info.set_xticks([]); ax_info.set_yticks([])
    _style3d(ax3d)

    # ── slider label text (above each slider) ─────────────────────────────────
    fig.text(0.04,  0.155, "σ  sigma  [1–28]",  color=C1, fontsize=8, fontfamily="monospace")
    fig.text(0.355, 0.155, "ρ  rho   [1–60]",   color=C2, fontsize=8, fontfamily="monospace")
    fig.text(0.645, 0.155, "β  beta  [0.1–5]",  color=C3, fontsize=8, fontfamily="monospace")

    # ── slider axes ───────────────────────────────────────────────────────────
    ax_sl_sig = fig.add_axes([0.04,  0.11, 0.28, 0.03], facecolor="#0f1420")
    ax_sl_rho = fig.add_axes([0.355, 0.11, 0.28, 0.03], facecolor="#0f1420")
    ax_sl_bet = fig.add_axes([0.645, 0.11, 0.19, 0.03], facecolor="#0f1420")
    ax_btn_run = fig.add_axes([0.855, 0.075, 0.12, 0.07])
    ax_btn_clr = fig.add_axes([0.855, 0.150, 0.12, 0.03])

    # ── create animator (wires everything up) ─────────────────────────────────
    _LorenzAnimator(fig, ax3d, ax_info,
                    ax_sl_sig, ax_sl_rho, ax_sl_bet,
                    ax_btn_run, ax_btn_clr,
                    sigma, rho, beta, ic, T, fps)

    plt.show()   # keeps window open; timer fires inside the event loop


# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    # ── Step 1: collect parameters from terminal BEFORE opening any window ────
    sigma, rho, beta, ic, T, fps = get_user_parameters()

    print("=" * 60)
    print("CST-305 Project 7 – opening figures one at a time.")
    print("Close each window to advance to the next.")
    print("=" * 60)

    # ── Part 1: Lorenz Attractor ──────────────────────────────────────────────
    plot_lorenz_static(sigma, rho, beta, ic, T)   # Fig 1  – user params
    plot_lorenz_variants()                         # Fig 2  – regime grid

    # ── Butterfly Effect ──────────────────────────────────────────────────────
    traj1, traj2, t_eval = plot_butterfly_3d()    # Fig 3  – diverging paths
    plot_butterfly_divergence(traj1, traj2, t_eval)  # Fig 4 – divergence plot
    plot_code_error()                              # Fig 5  – code bug analog

    # ── Part 2: Queue Theory ──────────────────────────────────────────────────
    plot_q1()   # Fig 6 – M/M/1/K overflow
    plot_q2()   # Fig 7 – scaling analysis
    plot_q3()   # Fig 8 – max arrival rate
    plot_q4()   # Fig 9 – server farm policies

    # ── Animated interactive window (stays open) ──────────────────────────────
    plot_animated_lorenz(sigma, rho, beta, ic, T, fps)   # Fig 10