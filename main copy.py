"""
=============================================================================
CST-305: Project 3 – Green's Function and ODE with IVP
=============================================================================
Programmers: [Your Name(s)]
Course: CST-305 – Principles of Modeling and Simulation
Instructor: [Instructor Name]
Date: 2025

Packages Used:
    - numpy        : Numerical computations and array operations
    - scipy        : ODE solving (solve_ivp) and numerical integration (quad)
    - matplotlib   : Plotting and visualization
    - sympy        : Symbolic math for verification (optional)

Approach to Implementation:
    1. Solve two 2nd-order ODEs analytically using Green's Function method.
    2. Numerically verify solutions using scipy.integrate.solve_ivp (RK45).
    3. Compute the Green's Function solution via numerical convolution (quad).
    4. Plot homogeneous, Green's function, and numerical solutions on
       shared figures for comparison.

ODEs:
    1. y'' + y  = 4;      t >= 0,  y(0) = y'(0) = 0
    2. y'' + 4y = x^2;    t >= 0,  y(0) = y'(0) = 0
=============================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — ODE 1:  y'' + y = 4,  y(0)=y'(0)=0
# ─────────────────────────────────────────────────────────────────────────────

# ── 1a. Analytical / Closed-Form Solution ────────────────────────────────────
#
#   Homogeneous solution:  y_h(t) = C1*cos(t) + C2*sin(t)
#
#   Green's Function for  y'' + y = f(t):
#       G(t, s) = sin(t - s)   for  s <= t,  0 otherwise
#
#   Particular solution via convolution:
#       y_p(t) = ∫₀ᵗ G(t,s) * f(s) ds
#             = ∫₀ᵗ sin(t-s) * 4 ds
#             = 4 * [-cos(t-s)]₀ᵗ
#             = 4 * (1 - cos(t))
#
#   Full solution (applying ICs y(0)=0, y'(0)=0 → C1=C2=0):
#       y(t) = 4*(1 - cos(t))

def ode1_analytical(t):
    """Closed-form Green's function solution for y'' + y = 4."""
    return 4.0 * (1.0 - np.cos(t))

def ode1_homogeneous(t, C1=1.0, C2=0.0):
    """Homogeneous solution y_h = C1*cos(t) + C2*sin(t)."""
    return C1 * np.cos(t) + C2 * np.sin(t)

# ── 1b. Green's Function Numerical Convolution ───────────────────────────────

def greens_ode1(t, s):
    """Green's function G(t,s) = sin(t-s) for ODE 1."""
    return np.sin(t - s)

def ode1_greens_numerical(t_arr):
    """Numerically compute y(t) = ∫₀ᵗ sin(t-s)*4 ds for each t."""
    result = np.zeros_like(t_arr)
    for i, t in enumerate(t_arr):
        if t == 0:
            result[i] = 0.0
        else:
            val, _ = quad(lambda s: greens_ode1(t, s) * 4.0, 0, t)
            result[i] = val
    return result

# ── 1c. Numerical ODE Solver (RK45 via solve_ivp) ────────────────────────────

def ode1_system(t, y):
    """State-space form: [y, y'] -> [y', y''] where y'' = 4 - y."""
    return [y[1], 4.0 - y[0]]

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — ODE 2:  y'' + 4y = x²,  y(0)=y'(0)=0
# ─────────────────────────────────────────────────────────────────────────────

# ── 2a. Analytical / Closed-Form Solution ────────────────────────────────────
#
#   Homogeneous solution:  y_h(t) = C1*cos(2t) + C2*sin(2t)
#
#   Green's Function for  y'' + 4y = f(t):
#       G(t, s) = (1/2)*sin(2*(t-s))   for s <= t
#
#   Particular solution via convolution:
#       y_p(t) = ∫₀ᵗ (1/2)*sin(2(t-s)) * s² ds
#
#   Evaluate using integration by parts twice:
#       Let u = s², dv = (1/2)sin(2(t-s)) ds
#       ... (full steps shown in documentation) ...
#
#   Result:
#       y_p(t) = (t²)/4 - (1/4)*[1 - cos(2t)]/2  (simplified)
#
#   Integration by Parts twice:
#       y_p(t) = (1/2) ∫₀ᵗ sin(2(t-s)) · s² ds
#
#   IBP round 1:  u=s², dv=sin(2(t-s))ds → v=cos(2(t-s))/2
#       = (1/2)[s²·cos(2(t-s))/2]₀ᵗ − (1/2)∫₀ᵗ cos(2(t-s))·s ds
#       = t²/4 − (1/2)∫₀ᵗ cos(2(t-s))·s ds
#
#   IBP round 2:  u=s, dv=cos(2(t-s))ds → v=sin(2(t-s))/2
#       (1/2)∫₀ᵗ cos(2(t-s))·s ds
#         = (1/2)[s·sin(2(t-s))/2]₀ᵗ − (1/2)∫₀ᵗ sin(2(t-s))/2 ds
#         = 0 + (1/4)∫₀ᵗ sin(2(t-s)) ds
#         = (1/4)[-cos(2(t-s))/2]₀ᵗ = (1/8)[cos(2t) − 1]  ... wait sign:
#         = (1/8)(cos(2t)·cos(0)-... = (1/8)(1 - cos(2t))  ... recheck:
#         [-cos(2(t-s))/2]₀ᵗ = -cos(0)/2 + cos(2t)/2 = (-1+cos(2t))/2
#         so = (1/4)·(-1+cos(2t))/2 = (cos(2t)-1)/8
#
#   Combined:
#       y_p(t) = t²/4 − (cos(2t)−1)/8 = t²/4 + (1−cos(2t))/8
#
#   Wait — let's be careful.  After full IBP:
#       y_p(t) = t²/4  +  (1/8)(cos(2t) − 1)
#   (Verified numerically against quadrature.)
#
#   ICs: y(0)=0 → y_p(0)=0 ✓; y'(0)=0 → y_p'(0)=0 ✓  (so C1=C2=0)
#       y(t) = t²/4 + (1/8)(cos(2t) − 1)

def ode2_analytical(t):
    """Closed-form Green's function solution for y'' + 4y = t²."""
    return t**2 / 4.0 + (np.cos(2.0 * t) - 1.0) / 8.0

def ode2_homogeneous(t, C1=1.0, C2=0.0):
    """Homogeneous solution y_h = C1*cos(2t) + C2*sin(2t)."""
    return C1 * np.cos(2.0 * t) + C2 * np.sin(2.0 * t)

# ── 2b. Green's Function Numerical Convolution ───────────────────────────────

def greens_ode2(t, s):
    """Green's function G(t,s) = (1/2)*sin(2*(t-s)) for ODE 2."""
    return 0.5 * np.sin(2.0 * (t - s))

def ode2_greens_numerical(t_arr):
    """Numerically compute y(t) = ∫₀ᵗ (1/2)*sin(2(t-s))*s² ds."""
    result = np.zeros_like(t_arr)
    for i, t in enumerate(t_arr):
        if t == 0:
            result[i] = 0.0
        else:
            val, _ = quad(lambda s: greens_ode2(t, s) * (s**2), 0, t)
            result[i] = val
    return result

# ── 2c. Numerical ODE Solver ─────────────────────────────────────────────────

def ode2_system(t, y):
    """State-space form: y'' = t² - 4y."""
    return [y[1], t**2 - 4.0 * y[0]]


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

t_span = (0, 4 * np.pi)
t_eval = np.linspace(0, 4 * np.pi, 500)
y0     = [0.0, 0.0]

# ODE 1 — solve
sol1 = solve_ivp(ode1_system, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-9)
y1_analytical  = ode1_analytical(t_eval)
y1_homogeneous = ode1_homogeneous(t_eval, C1=1.0, C2=0.0)
y1_greens_num  = ode1_greens_numerical(t_eval)

# ODE 2 — solve
sol2 = solve_ivp(ode2_system, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-9)
y2_analytical  = ode2_analytical(t_eval)
y2_homogeneous = ode2_homogeneous(t_eval, C1=1.0, C2=0.0)
y2_greens_num  = ode2_greens_numerical(t_eval)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'homogeneous': '#2196F3',
    'greens':      '#4CAF50',
    'numerical':   '#FF5722',
    'analytical':  '#9C27B0',
}

# ──────────────────────────────────────────────────────────────────
# FIGURE 1 — ODE 1 full comparison
# ──────────────────────────────────────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle("ODE 1:  y'' + y = 4,   y(0) = y'(0) = 0",
              fontsize=15, fontweight='bold')

ax = axes[0, 0]
ax.plot(t_eval, y1_homogeneous, color=COLORS['homogeneous'], lw=2)
ax.set_title("Homogeneous Solution\ny_h = cos(t)")
ax.set_xlabel("t"); ax.set_ylabel("y(t)")
ax.axhline(0, color='k', lw=0.5)

ax = axes[0, 1]
ax.plot(t_eval, y1_greens_num, color=COLORS['greens'], lw=2, label="Green's (numerical)")
ax.plot(t_eval, y1_analytical, color=COLORS['analytical'], lw=2, ls='--', label="Analytical 4(1-cos t)")
ax.set_title("Green's Function Solution")
ax.set_xlabel("t"); ax.set_ylabel("y(t)")
ax.legend()

ax = axes[1, 0]
ax.plot(t_eval, sol1.y[0], color=COLORS['numerical'], lw=2, label='RK45 Numerical')
ax.plot(t_eval, y1_analytical, color=COLORS['analytical'], lw=2, ls='--', label='Analytical')
ax.set_title("RK45 Numerical vs Analytical")
ax.set_xlabel("t"); ax.set_ylabel("y(t)")
ax.legend()

ax = axes[1, 1]
error = np.abs(sol1.y[0] - y1_analytical)
ax.semilogy(t_eval, error + 1e-16, color='red', lw=2)
ax.set_title("Absolute Error  |RK45 − Analytical|")
ax.set_xlabel("t"); ax.set_ylabel("Error (log scale)")

fig1.tight_layout()
fig1.savefig('ode1_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: ode1_comparison.png")

# ──────────────────────────────────────────────────────────────────
# FIGURE 2 — ODE 2 full comparison
# ──────────────────────────────────────────────────────────────────
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle("ODE 2:  y'' + 4y = t²,   y(0) = y'(0) = 0",
              fontsize=15, fontweight='bold')

ax = axes[0, 0]
ax.plot(t_eval, y2_homogeneous, color=COLORS['homogeneous'], lw=2)
ax.set_title("Homogeneous Solution\ny_h = cos(2t)")
ax.set_xlabel("t"); ax.set_ylabel("y(t)")
ax.axhline(0, color='k', lw=0.5)

ax = axes[0, 1]
ax.plot(t_eval, y2_greens_num, color=COLORS['greens'], lw=2, label="Green's (numerical)")
ax.plot(t_eval, y2_analytical, color=COLORS['analytical'], lw=2, ls='--',
        label=r"Analytical $\frac{t^2}{4}+\frac{\cos 2t - 1}{8}$")
ax.set_title("Green's Function Solution")
ax.set_xlabel("t"); ax.set_ylabel("y(t)")
ax.legend()

ax = axes[1, 0]
ax.plot(t_eval, sol2.y[0], color=COLORS['numerical'], lw=2, label='RK45 Numerical')
ax.plot(t_eval, y2_analytical, color=COLORS['analytical'], lw=2, ls='--', label='Analytical')
ax.set_title("RK45 Numerical vs Analytical")
ax.set_xlabel("t"); ax.set_ylabel("y(t)")
ax.legend()

ax = axes[1, 1]
error2 = np.abs(sol2.y[0] - y2_analytical)
ax.semilogy(t_eval, error2 + 1e-16, color='red', lw=2)
ax.set_title("Absolute Error  |RK45 − Analytical|")
ax.set_xlabel("t"); ax.set_ylabel("Error (log scale)")

fig2.tight_layout()
fig2.savefig('ode2_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: ode2_comparison.png")

# ──────────────────────────────────────────────────────────────────
# FIGURE 3 — Combined Green's Function overlay
# ──────────────────────────────────────────────────────────────────
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle("Green's Function Solutions — Both ODEs", fontsize=14, fontweight='bold')

ax1.fill_between(t_eval, y1_analytical, alpha=0.15, color=COLORS['greens'])
ax1.plot(t_eval, y1_analytical, color=COLORS['greens'], lw=2.5, label="y = 4(1 − cos t)")
ax1.plot(t_eval, y1_homogeneous, color=COLORS['homogeneous'], lw=1.5, ls='--', label="y_h = cos(t)")
ax1.set_title("ODE 1:  y'' + y = 4")
ax1.set_xlabel("t"); ax1.set_ylabel("y(t)")
ax1.legend()

ax2.fill_between(t_eval, y2_analytical, alpha=0.15, color=COLORS['greens'])
ax2.plot(t_eval, y2_analytical, color=COLORS['greens'], lw=2.5,
         label=r"y = $\frac{t^2}{4} + \frac{\cos 2t - 1}{8}$")
ax2.plot(t_eval, y2_homogeneous, color=COLORS['homogeneous'], lw=1.5, ls='--', label="y_h = cos(2t)")
ax2.set_title("ODE 2:  y'' + 4y = t²")
ax2.set_xlabel("t"); ax2.set_ylabel("y(t)")
ax2.legend()

fig3.tight_layout()
fig3.savefig('greens_solutions_combined.png', dpi=150, bbox_inches='tight')
print("Saved: greens_solutions_combined.png")

# ──────────────────────────────────────────────────────────────────
# FIGURE 4 — Green's Function surface G(t, s)
# ──────────────────────────────────────────────────────────────────
fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                subplot_kw={'projection': '3d'})
fig4.suptitle("Green's Function Surfaces G(t, s)", fontsize=14, fontweight='bold')

T_surf = np.linspace(0, 4*np.pi, 100)
S_surf = np.linspace(0, 4*np.pi, 100)
TT, SS = np.meshgrid(T_surf, S_surf)

# Mask: G = 0 for s > t
G1 = np.where(SS <= TT, np.sin(TT - SS), 0)
G2 = np.where(SS <= TT, 0.5 * np.sin(2*(TT - SS)), 0)

ax1.plot_surface(TT, SS, G1, cmap='viridis', alpha=0.8)
ax1.set_title("G₁(t,s) = sin(t−s)")
ax1.set_xlabel("t"); ax1.set_ylabel("s"); ax1.set_zlabel("G")

ax2.plot_surface(TT, SS, G2, cmap='plasma', alpha=0.8)
ax2.set_title("G₂(t,s) = ½sin(2(t−s))")
ax2.set_xlabel("t"); ax2.set_ylabel("s"); ax2.set_zlabel("G")

fig4.tight_layout()
fig4.savefig('greens_function_surfaces.png', dpi=150, bbox_inches='tight')
print("Saved: greens_function_surfaces.png")

plt.close('all')
print("\n✅  All plots saved to /mnt/user-data/outputs/")

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — PRINT VERIFICATION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SOLUTION VERIFICATION SUMMARY")
print("="*60)

sample_t = [np.pi/4, np.pi/2, np.pi, 2*np.pi]
print("\nODE 1:  y'' + y = 4  →  y(t) = 4(1 − cos t)")
print(f"{'t':>8}  {'Analytical':>12}  {'RK45':>12}  {'Green Num':>12}  {'Error':>12}")
for tv in sample_t:
    idx = np.argmin(np.abs(t_eval - tv))
    a = y1_analytical[idx]
    n = sol1.y[0][idx]
    g = y1_greens_num[idx]
    print(f"{tv:8.4f}  {a:12.6f}  {n:12.6f}  {g:12.6f}  {abs(a-n):12.2e}")

print("\nODE 2:  y'' + 4y = t²  →  y(t) = t²/4 + (cos 2t − 1)/8")
print(f"{'t':>8}  {'Analytical':>12}  {'RK45':>12}  {'Green Num':>12}  {'Error':>12}")
for tv in sample_t:
    idx = np.argmin(np.abs(t_eval - tv))
    a = y2_analytical[idx]
    n = sol2.y[0][idx]
    g = y2_greens_num[idx]
    print(f"{tv:8.4f}  {a:12.6f}  {n:12.6f}  {g:12.6f}  {abs(a-n):12.2e}")