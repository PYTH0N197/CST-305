# CST-305 Project 8 — Numerical Integration (Riemann Sums)

**Grand Canyon University — Principles of Modeling and Simulation**  
**Instructor:** Ricardo Citro  
**Student:** Chance  
**Year:** 2025

---

## Overview

This project implements numerical integration using **Riemann sums** to:

1. Evaluate four definite integrals using left, right, and midpoint endpoint methods
2. Derive closed-form Riemann sum formulas and compute their limits analytically
3. Apply numerical integration to estimate total data transferred by a media server over 30 minutes

All graphs are rendered inline in **PyCharm's Scientific View** (Plot panel).

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 or higher | [python.org](https://www.python.org/downloads/) |
| PyCharm | 2023.1+ (Community or Professional) | For built-in plot viewer |
| pip | Latest | Comes with Python |

---

## Installation

### 1. Clone or Download

```bash
git clone <your-repo-url>
cd cst305-project8
```

Or simply download `riemann_project8.py` directly.

### 2. Install Dependencies

```bash
pip install numpy matplotlib scipy
```

All three packages are available on PyPI and install in under a minute on most systems.

**Optional — create a virtual environment first (recommended):**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install numpy matplotlib scipy
```

---

## Running the Program

### In PyCharm (Recommended)

1. Open PyCharm and load the project folder (or open `riemann_project8.py` directly).
2. Enable the **Scientific View** plot panel:
   - Go to **File → Settings → Tools → Python Scientific**
   - Check **"Show plots in tool window"**
3. Right-click `riemann_project8.py` → **Run 'riemann_project8'**  
   *(or press `Shift+F10`)*

All graphs will appear in the **Plots** tab (bottom panel). Use the arrow buttons to navigate between the 5 figures.

### From the Terminal

```bash
python riemann_project8.py
```

> **Note:** Outside PyCharm, matplotlib will use your system's default backend (TkAgg, Qt5Agg, etc.). A plot window will open for each figure — close each one to advance to the next.

---

## Program Output

### Console Output

The program prints to the terminal:

- **Part 1A:** Exact integral value and all three Riemann sum approximations
- **Part 1B:** Riemann sum formula derivation + convergence table for n = 4, 10, 100, 1000, 10000
- **Part 1C:** Midpoint Riemann sum convergence table from n = 4 to n = 100,000
- **Part 1D:** Riemann sum formula derivation + convergence table
- **Part 2:** Full download rate table (30 entries), trapezoidal integration result (MB and GB), comparison of all four methods

### Graphs Generated

| Figure | Contents |
|---|---|
| **Figure 1** | Part 1A — 3-panel: Left, Right, Midpoint rectangles for sin(x)+1 on [−π, π] |
| **Figure 2** | Part 1B — Right-endpoint rectangles for 3x+2x² on [0,1] |
| **Figure 3** | Part 1C — Midpoint rectangles for ∫ln(x)dx + convergence error curve |
| **Figure 4** | Part 1D — Right-endpoint rectangles for x²−x³ on [−1,0] |
| **Figure 5** | Part 2 — Download rate bar chart with R(t) overlay + cumulative MB curve |

---

## Project Structure

```
cst305-project8/
│
├── riemann_project8.py          # Main Python script (all code)
├── README.md                    # This file
└── Project8_Numerical_Integration.docx   # Full project documentation
```

---

## Key Implementation Details

### Riemann Sum Function

```python
def riemann_sum(f, a, b, n, method="midpoint"):
    dx = (b - a) / n
    if method == "left":
        x = np.linspace(a, a + dx * (n - 1), n)
    elif method == "right":
        x = np.linspace(a + dx, b, n)
    elif method == "midpoint":
        x = np.linspace(a + dx / 2, b - dx / 2, n)
    return np.sum(f(x) * dx)
```

### PyCharm Plot Backend

```python
try:
    matplotlib.use("module://backend_interagg")
except Exception:
    pass  # Falls back to system default outside PyCharm
```

---

## Mathematical Problems Summary

| Problem | Function | Interval | Method | Exact Result |
|---|---|---|---|---|
| A | sin(x)+1 | [−π, π] | Left / Right / Midpoint | 2π ≈ 6.2832 |
| B | 3x+2x² | [0, 1] | Right → limit | 13/6 ≈ 2.1667 |
| C | ln(x) | [1, e] | Midpoint | 1.0000 |
| D | x²−x³ | [−1, 0] | Right → limit | 7/12 ≈ 0.5833 |

**Part 2 Result:** ≈ **9,360 MB (9.14 GB)** transferred over 30 minutes

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: numpy` | Run `pip install numpy matplotlib scipy` |
| Plots not showing in PyCharm | Enable **Show plots in tool window** in Settings → Tools → Python Scientific |
| `AttributeError: module 'numpy' has no attribute 'trapz'` | Update NumPy: `pip install --upgrade numpy` (script uses `np.trapezoid`, requires NumPy ≥ 1.25) |
| Plots open as separate windows | This is normal outside PyCharm — close each window to continue |

---

## References

- Thomas, G. B., Weir, M. D., & Hass, J. (2018). *Thomas' Calculus* (14th ed.). Pearson.
- NumPy Documentation: https://numpy.org/doc/stable/
- SciPy Documentation: https://docs.scipy.org/doc/scipy/
- Matplotlib Backends: https://matplotlib.org/stable/users/explain/figure/backends.html

---

*© 2025 — CST-305 Project 8 | Grand Canyon University*
