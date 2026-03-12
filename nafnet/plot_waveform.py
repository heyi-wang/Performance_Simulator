#!/usr/bin/env python3
"""plot_waveform.py — render the NAFNet simulator waveform.

Reads waveform.csv produced by nafnet_perf_sim and draws a stacked
digital-style activity chart for every hardware signal.

Because the accelerators handle millions of short (8-cycle) requests,
the raw bit-stream cannot be rendered pixel-by-pixel.  Instead the script
divides the simulation into N_BINS time windows and computes the exact
*occupancy fraction* (0.0–1.0) for each window using linear interpolation
of the cumulative-busy-time function.  The result looks like a waveform:
fully-idle windows appear at 0, fully-busy windows at 1, and windows that
straddle on/off transitions show the true fractional busy time.

Usage
-----
    python3 plot_waveform.py [waveform.csv] [waveform.png]

Both arguments are optional (defaults: waveform.csv, waveform.png in CWD).

Requirements
------------
    pip install matplotlib numpy
"""

import sys
import csv
import os
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required.  Install with: pip install numpy")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required.  Install with: pip install matplotlib")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BINS = 2000   # time-axis resolution (more bins → finer detail, slower render)


# ---------------------------------------------------------------------------
# Signal ordering and display metadata
# ---------------------------------------------------------------------------

SIGNAL_ORDER = [
    "worker_0", "worker_1", "worker_2", "worker_3",
    "mat_acc",  "vec_acc",
]

SIGNAL_COLORS = {
    "worker_0": "#4C9BE8",
    "worker_1": "#3A7EBF",
    "worker_2": "#2D639A",
    "worker_3": "#1F4A75",
    "mat_acc":  "#E8834C",
    "vec_acc":  "#BF5F3A",
}

SIGNAL_LABELS = {
    "worker_0": "Worker 0",
    "worker_1": "Worker 1",
    "worker_2": "Worker 2",
    "worker_3": "Worker 3",
    "mat_acc":  "Matrix Acc",
    "vec_acc":  "Vector Acc",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_events(path: str) -> dict:
    """Return  signal → sorted list of (cycle, value) transitions."""
    signals: dict[str, list] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            signals[row["signal"]].append((int(row["cycle"]), int(row["value"])))
    for sig in signals:
        signals[sig].sort(key=lambda x: x[0])
    return dict(signals)


# ---------------------------------------------------------------------------
# Signal sampling
# ---------------------------------------------------------------------------

def sample_signal(events: list, t_end: float, n_bins: int) -> np.ndarray:
    """Return the instantaneous signal value (0 or 1) at each bin centre.

    For each bin the signal value is taken from the most recent transition
    at or before that bin's centre point.  The result is strictly 0 or 1,
    giving a proper digital waveform display instead of an occupancy fraction.

    Algorithm: O(n_events + n_bins·log n_events) via np.searchsorted.
    """
    if not events:
        return np.zeros(n_bins)

    et = np.array([e[0] for e in events], dtype=np.float64)
    ev = np.array([e[1] for e in events], dtype=np.float64)

    bin_centers = np.linspace(t_end / (2 * n_bins),
                              t_end * (1 - 1 / (2 * n_bins)),
                              n_bins)

    # Last event index whose timestamp <= bin_center; -1 means before first event
    indices = np.searchsorted(et, bin_centers, side="right") - 1
    # Signal is 0 before the first logged transition
    return np.where(indices >= 0, ev[indices], 0.0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_waveforms(signals: dict, t_end: float, png_path: str):
    ordered = [s for s in SIGNAL_ORDER if s in signals]
    ordered += sorted(s for s in signals if s not in SIGNAL_ORDER)

    n = len(ordered)
    if n == 0:
        sys.exit("No signals found in waveform.csv")

    print(f"Sampling {n} signals over {t_end:.0f} cycles ({N_BINS} bins)…")

    # Sample instantaneous state (0/1) at each bin centre
    occ = {}
    for sig in ordered:
        occ[sig] = sample_signal(signals[sig], t_end, N_BINS)
        util = occ[sig].mean() * 100
        print(f"  {sig:<12s}  avg utilisation = {util:.1f}%")

    bin_centers = np.linspace(t_end / (2 * N_BINS),
                               t_end * (1 - 1 / (2 * N_BINS)),
                               N_BINS)

    row_height = 1.3
    fig_height = max(5.0, n * row_height + 1.8)
    fig, axes  = plt.subplots(n, 1, figsize=(20, fig_height),
                               sharex=True, squeeze=False)
    fig.subplots_adjust(hspace=0.06, left=0.12, right=0.97,
                        top=0.93, bottom=0.06)
    fig.suptitle(
        "NAFNet Hardware Activity Waveform\n"
        "(instantaneous state sampled at each time window centre; "
        "1 = busy, 0 = idle/stalled)",
        fontsize=11, fontweight="bold",
    )

    for row_idx, sig in enumerate(ordered):
        ax    = axes[row_idx][0]
        color = SIGNAL_COLORS.get(sig, "#888888")
        label = SIGNAL_LABELS.get(sig, sig)
        y     = occ[sig]

        ax.fill_between(bin_centers, y, alpha=0.45, color=color, linewidth=0)
        ax.step(bin_centers, y, where="mid", color=color, linewidth=0.8)

        avg = y.mean()
        ax.axhline(avg, color=color, linewidth=0.8, linestyle="--", alpha=0.7)
        ax.text(t_end * 1.002, avg, f"{avg*100:.0f}%",
                fontsize=7, va="center", color=color)

        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(["0", "½", "1"], fontsize=7)
        ax.set_ylabel(label, fontsize=9, rotation=0,
                      labelpad=65, va="center", ha="right")
        ax.yaxis.set_label_coords(-0.01, 0.5)
        ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1][0].set_xlabel("Simulation Cycles", fontsize=10)

    # X-axis tick formatting: show values in thousands (k) if large
    if t_end >= 1e6:
        axes[-1][0].xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
        )
        axes[-1][0].set_xlabel("Simulation Cycles (M = millions)", fontsize=10)
    elif t_end >= 1e3:
        axes[-1][0].xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k")
        )
        axes[-1][0].set_xlabel("Simulation Cycles (k = thousands)", fontsize=10)

    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"\nWaveform saved to {png_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "waveform.csv"
    png_path = sys.argv[2] if len(sys.argv) > 2 else "waveform.png"

    if not os.path.exists(csv_path):
        sys.exit(f"File not found: {csv_path}\n"
                 "Run nafnet_perf_sim first to generate it.")

    print(f"Loading {csv_path}…")
    signals = load_events(csv_path)

    t_end = max(
        (evts[-1][0] for evts in signals.values() if evts),
        default=1,
    )

    plot_waveforms(signals, float(t_end), png_path)


if __name__ == "__main__":
    main()
