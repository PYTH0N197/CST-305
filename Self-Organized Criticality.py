"""
============================================================
CST-305: Benchmark Project 5 — Self-Organized Criticality
File System Fragmentation Simulator
============================================================
Programmer  : Chance
Course      : CST-305 — Principles of Modeling and Simulation
Packages    : numpy, matplotlib, random
Approach    : Simulate a fixed-size storage disk as a 1-D array of
              blocks. Files are placed via first-fit allocation;
              when no single contiguous run is large enough the file
              is fragmented across multiple extents. Fragmentation
              ratio F(t) and mean access delay D(t) are tracked at
              each step. A critical threshold θ triggers a
              "system-too-slow" alert, modelling SOC behaviour.
============================================================
"""

import random
import numpy as np
import matplotlib
# matplotlib.use("Agg")  # removed: using interactive backend for PyCharm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────
DISK_BLOCKS        = 200
CRITICAL_THRESHOLD = 0.75   # fragmentation ratio alert level
BASE_ACCESS_TIME   = 1.0    # ms — base seek time
SEEK_PENALTY       = 0.8    # ms — extra per additional extent
MAX_FILE_SIZE      = 20     # blocks
MIN_FILE_SIZE      = 2
RANDOM_SEED        = 42

# ─────────────────────────────────────────────────────────────
# FILE SYSTEM MODEL
# ─────────────────────────────────────────────────────────────
class FileSystem:
    def __init__(self, total_blocks=DISK_BLOCKS):
        self.total_blocks = total_blocks
        self.disk         = [None] * total_blocks
        self.files        = {}      # fid → {"size": int, "extents": [(start,len),...]}
        self.next_id      = 1
        self.history      = []

    def _free_runs(self):
        runs, i = [], 0
        while i < self.total_blocks:
            if self.disk[i] is None:
                j = i
                while j < self.total_blocks and self.disk[j] is None:
                    j += 1
                runs.append((i, j - i))
                i = j
            else:
                i += 1
        return runs

    def _allocate(self, fid, size):
        needed  = size
        extents = []
        for (start, length) in sorted(self._free_runs(), key=lambda r: r[0]):
            if needed == 0:
                break
            take = min(length, needed)
            for b in range(start, start + take):
                self.disk[b] = fid
            extents.append((start, take))
            needed -= take
        return extents if needed == 0 else []

    def _free_file(self, fid):
        for b in range(self.total_blocks):
            if self.disk[b] == fid:
                self.disk[b] = None

    def save_file(self, size):
        fid     = self.next_id
        extents = self._allocate(fid, size)
        if not extents:
            return None
        self.files[fid] = {"size": size, "extents": extents}
        self.next_id   += 1
        return fid

    def delete_file(self, fid):
        if fid not in self.files:
            return False
        self._free_file(fid)
        del self.files[fid]
        return True

    def access_file(self, fid):
        if fid not in self.files:
            return 0.0
        n = len(self.files[fid]["extents"])
        return BASE_ACCESS_TIME + (n - 1) * SEEK_PENALTY

    def fragmentation_ratio(self):
        if not self.files:
            return 0.0
        frag = sum(1 for f in self.files.values() if len(f["extents"]) > 1)
        return frag / len(self.files)

    def mean_access_delay(self):
        if not self.files:
            return BASE_ACCESS_TIME
        return float(np.mean([self.access_file(fid) for fid in self.files]))

    def used_blocks(self):
        return sum(1 for b in self.disk if b is not None)

    def snapshot(self, step, cmd):
        self.history.append({
            "step"       : step,
            "command"    : cmd,
            "frag_ratio" : self.fragmentation_ratio(),
            "mean_delay" : self.mean_access_delay(),
            "used_blocks": self.used_blocks(),
            "n_files"    : len(self.files),
            "disk_state" : list(self.disk),
        })


# ─────────────────────────────────────────────────────────────
# SIMULATION LOOP
# ─────────────────────────────────────────────────────────────
def run_simulation(n_steps=200, seed=RANDOM_SEED):
    """
    Interleaved save/delete workload designed to build fragmentation:

    Phase A (steps 0-59):   Save files of varied sizes until disk
                            reaches ~65% utilisation.
    Phase B (steps 60-199): Alternate between:
                              - Deleting a small file (punches holes)
                              - Saving a large file  (must fragment
                                to fit into scattered free space)
    """
    random.seed(seed)
    fs          = FileSystem()
    alert_fired = False
    TARGET_UTIL = 0.65

    print(f"{'Step':>5}  {'Command':<34}  {'Frag%':>6}  {'Delay(ms)':>10}  {'Used/Total':>12}")
    print("-" * 78)

    for step in range(n_steps):
        util  = fs.used_blocks() / fs.total_blocks
        label = ""

        if step < 60:
            # Phase A: warm-up with mixed-size saves
            if util < TARGET_UTIL or not fs.files:
                size = random.randint(MIN_FILE_SIZE, MAX_FILE_SIZE)
                fid  = fs.save_file(size)
                label = f"SAVE  {size:>3} blk -> fid={fid}" if fid else f"SAVE  {size:>3} blk -> FULL"
            else:
                # Above target — delete a random file to keep disk breathing
                fid = random.choice(list(fs.files.keys()))
                sz  = fs.files[fid]["size"]
                fs.delete_file(fid)
                label = f"DELETE fid={fid} ({sz} blk)"
        else:
            # Phase B: punch-hole + large-save cycle
            if step % 2 == 0 and fs.files:
                # Delete smallest available file → small hole
                small = min(fs.files.items(), key=lambda kv: kv[1]["size"])
                fid, info = small
                fs.delete_file(fid)
                label = f"DELETE fid={fid} ({info['size']} blk)"
            else:
                # Save large file — likely to fragment
                size = random.randint(MAX_FILE_SIZE // 2, MAX_FILE_SIZE)
                fid  = fs.save_file(size)
                label = f"SAVE  {size:>3} blk -> fid={fid}" if fid else f"SAVE  {size:>3} blk -> FULL"

        fs.snapshot(step, label)
        snap  = fs.history[-1]
        alert = ""
        if snap["frag_ratio"] >= CRITICAL_THRESHOLD and not alert_fired:
            alert       = "  *** CRITICAL ***"
            alert_fired = True

        print(f"{step:>5}  {label:<34}  {snap['frag_ratio']*100:>5.1f}%  "
              f"{snap['mean_delay']:>10.3f}  "
              f"{snap['used_blocks']:>5}/{fs.total_blocks}{alert}")

    return fs


# ─────────────────────────────────────────────────────────────
# VISUALIZATION 1 — DISK MAP
# ─────────────────────────────────────────────────────────────
def plot_disk_snapshots(fs, steps_to_show):
    n_plots = len(steps_to_show)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, n_plots * 1.7 + 1.0))
    if n_plots == 1:
        axes = [axes]

    all_fids  = sorted({b for snap in fs.history for b in snap["disk_state"] if b})
    cmap      = plt.get_cmap("tab20")  # compatible with matplotlib < 3.5
    fid_color = {fid: cmap((i % 20) / 20) for i, fid in enumerate(all_fids)}

    for ax, step in zip(axes, steps_to_show):
        snap       = fs.history[step]
        disk_state = snap["disk_state"]
        cols       = fs.total_blocks
        img        = np.zeros((1, cols, 4))

        for b, fid in enumerate(disk_state):
            if fid is None:
                img[0, b] = [0.93, 0.93, 0.93, 1.0]
            else:
                img[0, b] = list(fid_color.get(fid, [0.2, 0.4, 0.8, 1.0])[:3]) + [1.0]

        ax.imshow(img, aspect="auto", interpolation="nearest")
        ax.set_yticks([])
        ax.set_xticks(np.arange(0, cols + 1, 20))
        ax.set_xticklabels(np.arange(0, cols + 1, 20), fontsize=7)
        ax.set_title(
            f"Step {step:3d}  |  {snap['command'][:42]:<42}  |  "
            f"F(t)={snap['frag_ratio']*100:.1f}%   D(t)={snap['mean_delay']:.2f} ms",
            fontsize=8, loc="left"
        )

    free_patch = mpatches.Patch(color=(0.93, 0.93, 0.93), label="Free block")
    used_patch = mpatches.Patch(color=cmap(0), label="File data (colour per file ID)")
    fig.legend(handles=[free_patch, used_patch], loc="lower right", fontsize=8)
    fig.suptitle("Disk Block Map — Fragmentation Developing Over Time",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
# VISUALIZATION 2 — METRICS CHART
# ─────────────────────────────────────────────────────────────
def plot_metrics(fs):
    steps  = [s["step"]        for s in fs.history]
    frags  = [s["frag_ratio"]  for s in fs.history]
    delays = [s["mean_delay"]  for s in fs.history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Self-Organized Criticality — Metrics Approaching the Critical Point",
                 fontsize=13, fontweight="bold")

    # Fragmentation ratio
    ax1.plot(steps, frags, color="#2E75B6", linewidth=1.8, label="F(t) — Fragmentation Ratio")
    ax1.axhline(CRITICAL_THRESHOLD, color="#C00000", linestyle="--",
                linewidth=1.5, label=f"Critical threshold θ = {CRITICAL_THRESHOLD}")
    ax1.fill_between(steps, frags, CRITICAL_THRESHOLD,
                     where=[f >= CRITICAL_THRESHOLD for f in frags],
                     color="#C00000", alpha=0.15, label="Critical zone")

    for s, f in zip(steps, frags):
        if f >= CRITICAL_THRESHOLD:
            ax1.annotate("CRITICAL\nTHRESHOLD", xy=(s, f),
                         xytext=(s + 5, min(f + 0.08, 1.05)),
                         arrowprops=dict(arrowstyle="->", color="#C00000"),
                         fontsize=8, color="#C00000", fontweight="bold")
            break

    ax1.set_ylabel("F(t)  Fragmentation Ratio", fontsize=10)
    ax1.set_ylim(-0.05, 1.15)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#F9F9F9")

    # Access delay
    ax2.plot(steps, delays, color="#70AD47", linewidth=1.8, label="D(t) — Mean Access Delay (ms)")
    ax2.axhline(BASE_ACCESS_TIME, color="#7030A0", linestyle=":",
                linewidth=1.2, label=f"Base access time = {BASE_ACCESS_TIME} ms")
    ax2.fill_between(steps, BASE_ACCESS_TIME, delays,
                     where=[d > BASE_ACCESS_TIME for d in delays],
                     color="#70AD47", alpha=0.15, label="Fragmentation overhead")
    ax2.set_xlabel("Simulation Step", fontsize=10)
    ax2.set_ylabel("D(t)  Mean Access Delay [ms]", fontsize=10)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#F9F9F9")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  CST-305 Project 5 — File System SOC Simulation")
    print("=" * 65 + "\n")

    fs = run_simulation(n_steps=200)

    total     = len(fs.history)
    snapshots = [0, total // 5, 2 * total // 5, 3 * total // 5, 4 * total // 5, total - 1]

    plot_disk_snapshots(fs, snapshots)
    plot_metrics(fs)

    last = fs.history[-1]
    print(f"\n{'='*50}")
    print(f"  Final fragmentation ratio : {last['frag_ratio']*100:.1f}%")
    print(f"  Final mean access delay   : {last['mean_delay']:.3f} ms")
    print(f"  Files on disk             : {last['n_files']}")
    print(f"  Used / total blocks       : {last['used_blocks']} / {DISK_BLOCKS}")
    hit = last['frag_ratio'] >= CRITICAL_THRESHOLD
    print(f"  Critical threshold hit?   : {'YES — system too slow!' if hit else 'No'}")
    print(f"{'='*50}\n")