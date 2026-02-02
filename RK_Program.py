import numpy as np
import math
import time
import matplotlib.pyplot as plt

# ODE definition
def f(x, y):
    return -y + math.log(x)

# Parameters
x0 = 2.0
y0 = 1.0
h = 0.001          # small step to reach 1000–2000 points
N = 1000           # number of steps

# Storage
x_vals = np.zeros(N+1)
y_rk4 = np.zeros(N+1)

x_vals[0] = x0
y_rk4[0] = y0

# Timing
start_time = time.time()

# RK4 loop
for n in range(N):
    x = x_vals[n]
    y = y_rk4[n]

    k1 = f(x, y)
    k2 = f(x + h/2, y + h*k1/2)
    k3 = f(x + h/2, y + h*k2/2)
    k4 = f(x + h, y + h*k3)

    y_rk4[n+1] = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    x_vals[n+1] = x + h

rk4_time = time.time() - start_time

print("RK4 completed")
print("Steps:", N)
print("Execution time (seconds):", rk4_time)

# -------------------------------
# Traditional Numerical Method: Forward Euler
# -------------------------------

y_euler = np.zeros(N+1)
y_euler[0] = y0

start_time = time.time()

for n in range(N):
    y_euler[n+1] = y_euler[n] + h * f(x_vals[n], y_euler[n])

euler_time = time.time() - start_time

print("\nEuler Method completed")
print("Steps:", N)
print("Execution time (seconds):", euler_time)

from scipy.integrate import solve_ivp

def f_scipy(x, y):
    return -y + np.log(x)

t_span = (x0, x0 + N*h)
t_eval = x_vals  # same x points as RK4

start_time = time.time()

sol = solve_ivp(
    f_scipy,
    t_span,
    [y0],
    method='RK45',
    t_eval=t_eval
)

rkf_time = time.time() - start_time

y_rkf = sol.y[0]

print("\nRKF45 completed")
print("Steps:", len(sol.t))
print("Execution time (seconds):", rkf_time)

from scipy.special import expi

C = math.exp(2)*(1 - math.log(2) + math.exp(-2)*expi(2))

def y_exact(x):
    return np.log(x) - np.exp(-x)*expi(x) + C*np.exp(-x)

y_true = y_exact(x_vals)

error_euler = np.abs(y_true - y_euler)
error_rk4 = np.abs(y_true - y_rk4)
error_rkf = np.abs(y_true - y_rkf)

plt.figure()
plt.plot(x_vals, y_rk4, label="RK4 (manual)")
plt.plot(x_vals, y_rkf, '--', label="RKF45 (SciPy)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("RK4 vs RKF45")
plt.show()

plt.figure()
plt.plot(x_vals, y_true, label="Exact Solution")
plt.plot(x_vals, y_rk4, '--', label="RK4 Approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Numerical vs Exact Solution")
plt.show()

plt.figure()
plt.plot(x_vals, error_euler, label="Euler Error")
plt.plot(x_vals, error_rk4, label="RK4 Error")
plt.plot(x_vals, error_rkf, '--', label="RKF45 Error")
plt.xlabel("x")
plt.ylabel("Absolute Error")
plt.legend()
plt.title("Error Comparison: Euler vs RK4 vs RKF45")
plt.show()

# -------------------------------
# Additional Variations
# -------------------------------

def run_variation(y0_new, h_new):
    N_new = int((x_vals[-1] - x0) / h_new)
    y_temp = y0_new
    x_temp = x0

    start = time.time()
    for _ in range(N_new):
        y_temp += h_new * f(x_temp, y_temp)
        x_temp += h_new
    return N_new, time.time() - start

steps1, time1 = run_variation(0.5, 0.001)
steps2, time2 = run_variation(1.5, 0.0005)

print("\nPerformance Variations:")
print("Variation 1 (y0=0.5, h=0.001): steps =", steps1, "time =", time1)
print("Variation 2 (y0=1.5, h=0.0005): steps =", steps2, "time =", time2)