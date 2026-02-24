"""
CST-305: Project 3 — Green's Function and ODE with IVP
=======================================================
Part 1: Solve y'' + 3y' + 2y = e^{-t}, y(0)=0, y'(0)=0 using Green's function.
         Verify numerically and plot the Green's function, forcing function,
         homogeneous solutions, and the full solution.

Part 2: Model data propagation through an 8-node network as a damped second-order
         system D''(t) + alpha*D'(t) + beta*D(t) = f(t) at each node, solved via
         Green's function convolution. Visualize the network topology and animate
         the signal propagating hop-by-hop.

Factors modeled in Part 2:
  1. Network topology (graph structure)
  2. Latency at every node (contributes to damping coefficient alpha)
  3. Number of hops (signal attenuates with each hop via coupling strength)
  4. Data size (amplitude of the initial forcing pulse)
  5. Bandwidth (contributes to restoring coefficient beta)
  6. Packet loss rate (adds to damping coefficient alpha)
  7. Propagation delay between nodes (time shift before signal arrives)

Green's Function (for both parts):
  Given y'' + a*y' + b*y = f(t), y(0)=0, y'(0)=0,
  with characteristic roots r1, r2 of r^2 + a*r + b = 0:

      G(t, s) = [e^{r1(t-s)} - e^{r2(t-s)}] / (r1 - r2)   for 0 <= s <= t

      y(t) = integral from 0 to t of G(t,s) * f(s) ds

Required packages: numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
#  GREEN'S FUNCTION CORE (shared by Part 1 and Part 2)
# =============================================================================

def greens_function(t, s, r1, r2):
    """
    Green's function for y'' + a*y' + b*y = f(t) with y(0)=0, y'(0)=0.

    G(t,s) = [e^{r1(t-s)} - e^{r2(t-s)}] / (r1 - r2)  for s <= t
           = 0                                            for s > t

    Parameters:
        t   : evaluation time
        s   : source time
        r1  : first characteristic root
        r2  : second characteristic root
    Returns:
        Value of Green's function G(t, s)
    """
    if s > t:
        return 0.0
    tau = t - s
    return (np.exp(r1 * tau) - np.exp(r2 * tau)) / (r1 - r2)


def solve_with_greens(t_val, r1, r2, forcing_func, t_start=0):
    """
    Compute y(t) = integral from t_start to t_val of G(t_val, s) * f(s) ds
    using numerical quadrature (scipy.integrate.quad).

    Parameters:
        t_val        : time at which to evaluate the solution
        r1, r2       : characteristic roots
        forcing_func : callable f(s) representing the forcing/input function
        t_start      : lower bound of integration (default 0)
    Returns:
        y(t_val) computed via Green's function convolution
    """
    if t_val <= t_start:
        return 0.0
    integrand = lambda s: greens_function(t_val, s, r1, r2) * forcing_func(s)
    result, _ = quad(integrand, t_start, t_val, limit=100)
    return result


# =============================================================================
#  PART 1: Solve y'' + 3y' + 2y = e^{-t}, y(0)=0, y'(0)=0
# =============================================================================

def part1():
    """
    PART 1 — Analytical + Numerical Solution
    =========================================

    ODE:  y'' + 3y' + 2y = e^{-t},   y(0) = 0,  y'(0) = 0

    Step 1: Homogeneous equation y'' + 3y' + 2y = 0
            Characteristic equation: r^2 + 3r + 2 = 0  =>  (r+1)(r+2) = 0
            Roots: r1 = -1, r2 = -2
            Homogeneous solutions: y1 = e^{-t}, y2 = e^{-2t}

    Step 2: Wronskian
            W(s) = | e^{-s}    e^{-2s}  |
                   | -e^{-s}  -2e^{-2s} |
            W(s) = -2e^{-3s} + e^{-3s} = -e^{-3s}

    Step 3: Green's function
            G(t,s) = [y1(s)*y2(t) - y1(t)*y2(s)] / W(s)
                   = [e^{-s}*e^{-2t} - e^{-t}*e^{-2s}] / (-e^{-3s})
                   = e^{(s-t)} - e^{2(s-t)}

    Step 4: Particular solution via convolution
            y_p(t) = integral_0^t G(t,s) * e^{-s} ds
                   = integral_0^t [e^{(s-t)} - e^{2(s-t)}] * e^{-s} ds
                   = integral_0^t [e^{-t} - e^{(s-2t)}] ds
                   = t*e^{-t} - (e^{-t} - e^{-2t})
                   = t*e^{-t} - e^{-t} + e^{-2t}

    Step 5: Apply initial conditions to general solution
            y(t) = c1*e^{-t} + c2*e^{-2t} + t*e^{-t} - e^{-t} + e^{-2t}
            y(0) = 0  =>  c1 + c2 = 0
            y'(0) = 0 =>  -c1 - 2c2 = 0
            Solving: c1 = 0, c2 = 0

    FINAL SOLUTION:  y(t) = t*e^{-t} - e^{-t} + e^{-2t}

    Verification:
            y(0) = 0 - 1 + 1 = 0  ✓
            y'(0) = (2-0)*e^0 - 2*e^0 = 2 - 2 = 0  ✓
            y'' + 3y' + 2y = e^{-t}  ✓  (shown algebraically in documentation)
    """

    print("=" * 65)
    print("  PART 1: y'' + 3y' + 2y = e^(-t),  y(0)=0, y'(0)=0")
    print("=" * 65)

    # Characteristic roots
    r1, r2 = -1.0, -2.0
    print(f"\n  Characteristic roots: r1 = {r1}, r2 = {r2}")
    print(f"  Wronskian: W(s) = -e^(-3s)")
    print(f"  Green's function: G(t,s) = e^(s-t) - e^(2(s-t))  for s <= t")
    print(f"  Exact solution: y(t) = t*e^(-t) - e^(-t) + e^(-2t)")

    # Time domain
    t = np.linspace(0, 8, 500)

    # Forcing function
    g = lambda s: np.exp(-s)

    # Exact analytical solution
    y_exact = t * np.exp(-t) - np.exp(-t) + np.exp(-2 * t)

    # Numerical solution via Green's function convolution
    print("\n  Computing numerical solution via Green's function quadrature...")
    y_green = np.array([solve_with_greens(tv, r1, r2, g) for tv in t])

    # Verification: compute max error
    max_error = np.max(np.abs(y_exact - y_green))
    print(f"  Max |exact - numerical| = {max_error:.2e}")

    # Verify initial conditions
    print(f"\n  Verification:")
    print(f"    y(0)  = {y_exact[0]:.6f}  (should be 0)")
    y_prime = (1 - t) * np.exp(-t) + np.exp(-t) - 2 * np.exp(-2 * t)
    print(f"    y'(0) = {y_prime[0]:.6f}  (should be 0)")

    # Verify ODE: y'' + 3y' + 2y should equal e^{-t}
    y_pp = (t - 3) * np.exp(-t) + 4 * np.exp(-2 * t)
    y_p = (2 - t) * np.exp(-t) - 2 * np.exp(-2 * t)
    residual = y_pp + 3 * y_p + 2 * y_exact - np.exp(-t)
    print(f"    Max |y'' + 3y' + 2y - e^(-t)| = {np.max(np.abs(residual)):.2e}")

    # ---- PLOTTING ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Part 1: Green's Function Solution of  y'' + 3y' + 2y = e⁻ᵗ",
                 fontsize=14, fontweight='bold')

    # Plot 1: Green's function G(t, s) for various fixed s
    ax = axes[0, 0]
    for s_val in [0, 0.5, 1.0, 2.0, 3.0]:
        G_vals = [greens_function(tv, s_val, r1, r2) for tv in t]
        ax.plot(t, G_vals, label=f's = {s_val}', linewidth=1.5)
    ax.set_xlabel('t')
    ax.set_ylabel('G(t, s)')
    ax.set_title("Green's Function G(t, s) for Various s")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 0.35)

    # Plot 2: Forcing function g(t) = e^{-t}
    ax = axes[0, 1]
    g_vals = np.exp(-t)
    ax.plot(t, g_vals, 'r-', linewidth=2, label='g(t) = e⁻ᵗ')
    ax.fill_between(t, g_vals, alpha=0.2, color='red')
    ax.set_xlabel('t')
    ax.set_ylabel('g(t)')
    ax.set_title("Forcing Function g(t) = e⁻ᵗ")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Exact vs numerical solution
    ax = axes[1, 0]
    ax.plot(t, y_exact, 'b-', linewidth=2, label='Exact: te⁻ᵗ − e⁻ᵗ + e⁻²ᵗ')
    ax.plot(t[::10], y_green[::10], 'ro', markersize=4,
            label="Green's function (numerical)")
    ax.set_xlabel('t')
    ax.set_ylabel('y(t)')
    ax.set_title("Solution y(t) — Exact vs Green's Function")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Homogeneous solutions
    ax = axes[1, 1]
    ax.plot(t, np.exp(-t), 'g-', linewidth=2, label='y₁ = e⁻ᵗ')
    ax.plot(t, np.exp(-2 * t), 'm-', linewidth=2, label='y₂ = e⁻²ᵗ')
    ax.set_xlabel('t')
    ax.set_ylabel('y(t)')
    ax.set_title("Homogeneous Solutions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('part1_greens_function_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Saved: part1_greens_function_plots.png")


# =============================================================================
#  PART 2: Network Data Propagation Simulation
# =============================================================================

class NetworkNode:
    """
    Represents a single node in the network.

    Each node's data signal D(t) satisfies:
        D''(t) + alpha * D'(t) + beta * D(t) = f(t)

    where alpha and beta are derived from the node's physical characteristics.
    """
    def __init__(self, node_id, name, latency, bandwidth, packet_loss_rate, position):
        self.node_id = node_id
        self.name = name
        self.latency = latency                  # ms — contributes to damping
        self.bandwidth = bandwidth              # Mbps — contributes to beta
        self.packet_loss_rate = packet_loss_rate # 0 to 1
        self.position = position                # (x, y) for visualization

        # --- Derive ODE coefficients from physical parameters ---
        # alpha (damping): higher latency + packet loss = more damping
        self.alpha = 1.0 * latency / 10.0 + 3.0 * packet_loss_rate + 2.5
        # beta (restoring/bandwidth): higher bandwidth = higher natural frequency
        self.beta = 0.05 * bandwidth / 100.0 + 0.8

        # Characteristic roots: r^2 + alpha*r + beta = 0
        disc = self.alpha ** 2 - 4 * self.beta
        if disc >= 0:
            self.r1 = (-self.alpha + np.sqrt(disc)) / 2.0
            self.r2 = (-self.alpha - np.sqrt(disc)) / 2.0
        else:
            # Complex conjugate roots — use small separation for numerical stability
            self.r1 = -self.alpha / 2.0 + 0.01
            self.r2 = -self.alpha / 2.0 - 0.01

        self.signal = np.array([])  # populated during simulation


class NetworkEdge:
    """
    Represents a directed connection between two nodes.

    Attributes:
        propagation_delay  : time units before signal reaches target node
        coupling_strength  : 0 to 1, attenuates signal at each hop
    """
    def __init__(self, source_id, target_id, propagation_delay, coupling_strength):
        self.source_id = source_id
        self.target_id = target_id
        self.propagation_delay = propagation_delay
        self.coupling_strength = coupling_strength


def build_network():
    """
    Build a sample network with 8 nodes:
        Server -> Router1 -> Switch -> {Workstation1, Workstation2, Workstation3}
                  Router1 -> Router2 -> RemoteClient
    """
    nodes = [
        NetworkNode(0, "Server",        latency=5,  bandwidth=1000, packet_loss_rate=0.01, position=(0.5, 4.5)),
        NetworkNode(1, "Router 1",      latency=10, bandwidth=800,  packet_loss_rate=0.02, position=(2.0, 3.5)),
        NetworkNode(2, "Switch",        latency=3,  bandwidth=1000, packet_loss_rate=0.005, position=(3.5, 2.5)),
        NetworkNode(3, "Workstation 1", latency=8,  bandwidth=500,  packet_loss_rate=0.03, position=(2.0, 1.0)),
        NetworkNode(4, "Workstation 2", latency=12, bandwidth=300,  packet_loss_rate=0.05, position=(3.5, 0.5)),
        NetworkNode(5, "Workstation 3", latency=6,  bandwidth=600,  packet_loss_rate=0.02, position=(5.0, 1.0)),
        NetworkNode(6, "Router 2",      latency=15, bandwidth=400,  packet_loss_rate=0.04, position=(5.5, 3.5)),
        NetworkNode(7, "Remote Client", latency=25, bandwidth=200,  packet_loss_rate=0.08, position=(7.0, 4.5)),
    ]

    edges = [
        NetworkEdge(0, 1, propagation_delay=0.5, coupling_strength=0.70),
        NetworkEdge(1, 2, propagation_delay=0.3, coupling_strength=0.65),
        NetworkEdge(2, 3, propagation_delay=0.2, coupling_strength=0.55),
        NetworkEdge(2, 4, propagation_delay=0.2, coupling_strength=0.50),
        NetworkEdge(2, 5, propagation_delay=0.2, coupling_strength=0.55),
        NetworkEdge(1, 6, propagation_delay=0.8, coupling_strength=0.45),
        NetworkEdge(6, 7, propagation_delay=1.5, coupling_strength=0.30),
    ]

    return nodes, edges


def simulate_network(nodes, edges, t_max=12.0, dt=0.05, data_size=1.0):
    """
    Simulate data propagation through the network using Green's function.

    The Server (node 0) generates a Gaussian data pulse. The signal propagates
    hop-by-hop: at each downstream node, the forcing function is the delayed
    and attenuated signal from its upstream neighbor. Each node's ODE is solved
    via Green's function convolution.

    Parameters:
        t_max     : simulation duration
        dt        : time step
        data_size : amplitude scaling of the initial data pulse
    Returns:
        t : time array
    """
    t = np.arange(0, t_max, dt)
    n_steps = len(t)

    # Initialize all signals to zero
    for node in nodes:
        node.signal = np.zeros(n_steps)

    # Source node forcing function: a smooth Gaussian data pulse
    def source_pulse(t_val):
        return data_size * 5.0 * np.exp(-((t_val - 1.5) ** 2) / 0.3)

    # Solve source node (Node 0) using Green's function
    print("  Computing signal at Node 0 (Server)...")
    for i, t_val in enumerate(t):
        nodes[0].signal[i] = solve_with_greens(
            t_val, nodes[0].r1, nodes[0].r2, source_pulse
        )

    # Determine propagation order via BFS from source
    visited = {0}
    queue = [0]
    propagation_order = []

    while queue:
        current = queue.pop(0)
        for edge in edges:
            if edge.source_id == current and edge.target_id not in visited:
                propagation_order.append(edge)
                visited.add(edge.target_id)
                queue.append(edge.target_id)

    # Propagate signal through each edge in BFS order
    for edge in propagation_order:
        src_node = nodes[edge.source_id]
        tgt_node = nodes[edge.target_id]
        delay = edge.propagation_delay
        coupling = edge.coupling_strength

        print(f"  Computing signal at Node {tgt_node.node_id} ({tgt_node.name})...")

        # Forcing function for target = delayed & attenuated source signal
        src_signal_interp = lambda s, _src=src_node.signal, _t=t, _d=delay, _c=coupling: (
            _c * np.interp(s - _d, _t, _src, left=0, right=0)
        )

        for i, t_val in enumerate(t):
            if t_val > delay:
                tgt_node.signal[i] = solve_with_greens(
                    t_val, tgt_node.r1, tgt_node.r2, src_signal_interp, t_start=delay
                )

    return t


def plot_network_and_signals(nodes, edges, t):
    """Generate the comprehensive static visualization for Part 2."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    fig.suptitle("Part 2: Network Data Propagation — Green's Function Simulation",
                 fontsize=15, fontweight='bold', y=0.98)

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
              '#e67e22', '#9b59b6', '#1abc9c', '#34495e']

    # --- Network topology (top-left, 2 columns) ---
    ax_net = fig.add_subplot(gs[0, :2])
    ax_net.set_title("Network Topology", fontsize=13, fontweight='bold')
    ax_net.set_xlim(-0.5, 8.0)
    ax_net.set_ylim(-0.5, 5.5)
    ax_net.set_aspect('equal')
    ax_net.axis('off')

    for edge in edges:
        src = nodes[edge.source_id].position
        tgt = nodes[edge.target_id].position
        ax_net.annotate("", xy=tgt, xytext=src,
                        arrowprops=dict(arrowstyle='->', color='#555555',
                                        lw=1.5 + edge.coupling_strength * 2))
        mid = ((src[0] + tgt[0]) / 2, (src[1] + tgt[1]) / 2 + 0.15)
        ax_net.text(mid[0], mid[1],
                    f'c={edge.coupling_strength:.2f}\n\u0394t={edge.propagation_delay}',
                    fontsize=7, ha='center', color='#777777')

    for node in nodes:
        circle = plt.Circle(node.position, 0.25, color=colors[node.node_id], zorder=5)
        ax_net.add_patch(circle)
        ax_net.text(node.position[0], node.position[1], str(node.node_id),
                    ha='center', va='center', fontsize=10,
                    fontweight='bold', color='white', zorder=6)
        ax_net.text(node.position[0], node.position[1] - 0.45, node.name,
                    ha='center', va='top', fontsize=7, fontweight='bold')

    # --- Parameters table (top-right) ---
    ax_table = fig.add_subplot(gs[0, 2])
    ax_table.axis('off')
    ax_table.set_title("Node Parameters", fontsize=13, fontweight='bold')
    table_data = [["ID", "Name", "Lat(ms)", "BW", "Loss", "\u03b1", "\u03b2"]]
    for n in nodes:
        table_data.append([
            str(n.node_id), n.name[:8], f"{n.latency}", f"{n.bandwidth}",
            f"{n.packet_loss_rate:.3f}", f"{n.alpha:.2f}", f"{n.beta:.2f}"
        ])
    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.3)

    # --- Individual signal plots (bottom rows) ---
    for i in range(min(len(nodes), 6)):
        row = 1 + i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        ax.plot(t, nodes[i].signal, color=colors[i], linewidth=1.5)
        ax.fill_between(t, nodes[i].signal, alpha=0.15, color=colors[i])
        ax.set_title(f"Node {i}: {nodes[i].name}", fontsize=9, fontweight='bold')
        ax.set_xlabel('Time', fontsize=8)
        ax.set_ylabel('Signal', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        peak = np.max(nodes[i].signal)
        peak_t = t[np.argmax(nodes[i].signal)]
        if peak > 0.001:
            ax.annotate(f'peak={peak:.4f}\nt={peak_t:.1f}',
                        xy=(peak_t, peak), fontsize=6,
                        xytext=(peak_t + 1.5, peak * 0.75),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    plt.savefig('part2_network_simulation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: part2_network_simulation.png")


def plot_signal_comparison(nodes, t):
    """Plot all node signals overlaid for comparison."""
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
              '#e67e22', '#9b59b6', '#1abc9c', '#34495e']

    for i, node in enumerate(nodes):
        ax.plot(t, node.signal, color=colors[i], linewidth=2,
                label=f"{node.name} (\u03b1={node.alpha:.2f}, \u03b2={node.beta:.2f})")

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Signal Amplitude D(t)', fontsize=12)
    ax.set_title("Data Signal Propagation Across All Nodes\n"
                 "D''(t) + \u03b1\u00b7D'(t) + \u03b2\u00b7D(t) = f(t)  "
                 "solved via Green's Function", fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    ax.annotate('Data pulse originates\nat Server (Node 0)',
                xy=(1.8, nodes[0].signal[int(1.8 / 0.05)]),
                xytext=(4.0, nodes[0].signal[int(1.8 / 0.05)] * 1.1),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    plt.savefig('part2_signal_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: part2_signal_comparison.png")


def create_animation(nodes, edges, t):
    """Create animated GIF of data propagating through the network."""
    fig, (ax_net, ax_sig) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Network Data Propagation — Animated Simulation",
                 fontsize=14, fontweight='bold')

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
              '#e67e22', '#9b59b6', '#1abc9c', '#34495e']

    # --- Network panel ---
    ax_net.set_xlim(-0.5, 8.0)
    ax_net.set_ylim(-0.5, 5.5)
    ax_net.set_aspect('equal')
    ax_net.axis('off')
    ax_net.set_title("Network (node brightness = signal intensity)")

    for edge in edges:
        src = nodes[edge.source_id].position
        tgt = nodes[edge.target_id].position
        ax_net.annotate("", xy=tgt, xytext=src,
                        arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1.5))

    node_circles = []
    for node in nodes:
        circle = plt.Circle(node.position, 0.3, color=colors[node.node_id],
                            zorder=5, ec='black', lw=1.5)
        ax_net.add_patch(circle)
        node_circles.append(circle)
        ax_net.text(node.position[0], node.position[1] - 0.5,
                    node.name, ha='center', fontsize=7, fontweight='bold')

    # --- Signal panel ---
    max_sig = max(np.max(node.signal) for node in nodes) * 1.1
    ax_sig.set_xlim(0, t[-1])
    ax_sig.set_ylim(-0.02, max(max_sig, 0.1))
    ax_sig.set_xlabel('Time')
    ax_sig.set_ylabel('Signal Amplitude')
    ax_sig.set_title("Signal at Each Node Over Time")
    ax_sig.grid(True, alpha=0.3)

    lines = []
    for i, node in enumerate(nodes):
        line, = ax_sig.plot([], [], color=colors[i], linewidth=1.5, label=node.name)
        lines.append(line)
    ax_sig.legend(fontsize=6, loc='upper right', ncol=2)

    time_line = ax_sig.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    time_text = ax_net.text(0.0, 5.2, '', fontsize=11, fontweight='bold')

    step = 4  # skip frames for speed

    def animate(frame):
        idx = min(frame * step, len(t) - 1)
        current_t = t[idx]

        time_text.set_text(f't = {current_t:.2f}')
        time_line.set_xdata([current_t, current_t])

        for i, node in enumerate(nodes):
            lines[i].set_data(t[:idx + 1], node.signal[:idx + 1])

        for i, node in enumerate(nodes):
            intensity = min(node.signal[idx] / max(max_sig, 0.001), 1.0)
            r_base = int(colors[i][1:3], 16) / 255
            g_base = int(colors[i][3:5], 16) / 255
            b_base = int(colors[i][5:7], 16) / 255
            r = r_base + (1 - r_base) * intensity * 0.7
            g = g_base + (1 - g_base) * intensity * 0.7
            b = b_base * (1 - intensity * 0.5)
            node_circles[i].set_facecolor((r, g, b))
            node_circles[i].set_linewidth(1.5 + intensity * 3)

        return lines + node_circles + [time_line, time_text]

    n_frames = len(t) // step
    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                   interval=50, blit=False)

    print("  Saving animation (may take a moment)...")
    anim.save('part2_animation.gif', writer='pillow', fps=20, dpi=100)
    plt.close()
    print("  Saved: part2_animation.gif")


def part2():
    """Run the full Part 2 pipeline: build, simulate, visualize."""
    print("\n" + "=" * 65)
    print("  PART 2: Network Data Propagation Simulation")
    print("=" * 65)

    # Build network
    nodes, edges = build_network()

    print("\n  Network Nodes:")
    print(f"  {'ID':<4} {'Name':<16} {'Latency':<10} {'BW(Mbps)':<10} "
          f"{'Loss':<8} {chr(945):<8} {chr(946):<8}")
    print("  " + "-" * 64)
    for n in nodes:
        print(f"  {n.node_id:<4} {n.name:<16} {n.latency:<10} {n.bandwidth:<10} "
              f"{n.packet_loss_rate:<8.3f} {n.alpha:<8.3f} {n.beta:<8.3f}")

    # Simulate
    print("\n  Running Green's function simulation...")
    t = simulate_network(nodes, edges, t_max=12.0, dt=0.05, data_size=1.0)

    # Report peak signals
    print("\n  Peak signals (showing propagation delay and attenuation):")
    for node in nodes:
        peak = np.max(node.signal)
        peak_time = t[np.argmax(node.signal)]
        print(f"    {node.name:<16}: peak = {peak:.4f}  at t = {peak_time:.2f}")

    # Generate visualizations
    print("\n  Generating plots...")
    plot_network_and_signals(nodes, edges, t)
    plot_signal_comparison(nodes, t)

    print("\n  Generating animation...")
    create_animation(nodes, edges, t)


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 65)
    print("#  CST-305: Project 3 — Green's Function and ODE with IVP       #")
    print("#" * 65)

    part1()
    part2()

    print("\n" + "=" * 65)
    print("  All outputs generated successfully!")
    print("    Part 1: part1_greens_function_plots.png")
    print("    Part 2: part2_network_simulation.png")
    print("           part2_signal_comparison.png")
    print("           part2_animation.gif")
    print("=" * 65)