#!/usr/bin/env python3
"""Stacked-bar plot of per-backend cycle breakdown for the nafblock sweep CSV.

Sister to kernel/matmul/plot_sweep.py --bar, but stacks the five nafblock
backend fractions instead of matmul's mat/vec/dma/scalar/stall categories.

Usage:
    python nafblock/plot_backend_breakdown.py \
        --input nafblock/full_sweep.csv \
        --x block_c \
        --filter tile_m=16,vec_count=2,dma_base_lat=10 \
        --output /tmp/nb_backend_bar.png

The matmul `plot_sweep.py` covers every other plot type (2D line, 3D scatter
/ surface / bar) without modification because its column projection is
column-agnostic; only the --bar mode hardcodes column names.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required. Install with: pip install matplotlib")


# (column, label, color, hatch) — height = total_cycles * fraction/100.
SEGMENTS = [
    ("layernorm_cycle_fraction_pct", "LayerNorm", "#1f77b4", "//"),
    ("matmul_cycle_fraction_pct",    "Matmul",    "#2ca02c", "\\\\"),
    ("dwconv_cycle_fraction_pct",    "DwConv",    "#ff7f0e", "xx"),
    ("pooling_cycle_fraction_pct",   "Pooling",   "#9467bd", ".."),
    ("vecops_cycle_fraction_pct",    "VecOps",    "#d62728", "++"),
]


def _parse_filter(value: str) -> dict[str, set[str]]:
    """Same syntax as plot_sweep.py: key=val or key=v1|v2|v3, comma-joined."""
    out: dict[str, set[str]] = {}
    if not value:
        return out
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise argparse.ArgumentTypeError(
                f"--filter: '{chunk}' must be key=val"
            )
        k, v = chunk.split("=", 1)
        out[k.strip()] = {p.strip() for p in v.split("|") if p.strip()}
    return out


def _row_matches(row: dict[str, str], flt: dict[str, set[str]]) -> bool:
    for k, vs in flt.items():
        rv = row.get(k, "")
        if rv not in vs:
            return False
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--x", required=True,
                   help="Column to use as bar-position (one bar per unique value).")
    p.add_argument("--filter", type=_parse_filter, default={},
                   help="key=val[|val][,key=val] (same syntax as plot_sweep.py).")
    p.add_argument("--require-pass", action="store_true",
                   help="Drop rows whose verification_status != PASS.")
    p.add_argument("--output", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--log-y", action="store_true")
    args = p.parse_args()

    with Path(args.input).open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if _row_matches(r, args.filter)]
    if args.require_pass:
        rows = [r for r in rows if r.get("verification_status") == "PASS"]
    if not rows:
        sys.exit("[plot] no rows matched")

    # One bar per unique --x value; fail loud on collisions.
    by_x: dict[str, dict[str, str]] = {}
    for r in rows:
        xv = r[args.x]
        if xv in by_x:
            sys.exit(
                f"[plot] multiple rows share {args.x}={xv}; "
                "narrow --filter so each x maps to one row."
            )
        by_x[xv] = r

    try:
        xs_sorted = sorted(by_x.keys(), key=lambda v: float(v))
    except ValueError:
        xs_sorted = sorted(by_x.keys())

    totals = [float(by_x[x]["total_cycles"]) for x in xs_sorted]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(xs_sorted) + 4), 5))
    bottoms = [0.0] * len(xs_sorted)
    for col, label, color, hatch in SEGMENTS:
        heights = [
            float(by_x[x][col]) / 100.0 * totals[i]
            for i, x in enumerate(xs_sorted)
        ]
        ax.bar(
            range(len(xs_sorted)), heights,
            bottom=bottoms,
            color=color, edgecolor="black", hatch=hatch, label=label,
        )
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_xticks(range(len(xs_sorted)))
    ax.set_xticklabels(xs_sorted)
    ax.set_xlabel(args.x)
    ax.set_ylabel("Total cycles")
    ax.set_title(args.title or f"Per-backend cycle breakdown vs {args.x}")
    if args.log_y:
        ax.set_yscale("log")
    ax.legend(title="Backend", loc="upper right", framealpha=0.9)

    # Auto-annotate fixed parameters (any column with a single unique value
    # across all matched rows that isn't the bar axis).
    candidate_keys = [
        "tile_m", "tile_k", "tile_n",
        "mat_latency", "mat_count", "vec_count", "vec_bytes", "n_workers",
        "block_c", "block_h", "block_w", "dma_base_lat",
    ]
    fixed_bits = []
    for k in candidate_keys:
        if k == args.x:
            continue
        vals = {r.get(k, "") for r in rows}
        if len(vals) == 1:
            v = next(iter(vals))
            if v:
                fixed_bits.append(f"{k}={v}")
    if fixed_bits:
        fig.text(
            0.5, 0.01, "Fixed: " + "  ·  ".join(fixed_bits),
            ha="center", fontsize=8, style="italic", color="#555",
        )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"[plot] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
