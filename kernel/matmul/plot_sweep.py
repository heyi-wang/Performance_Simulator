#!/usr/bin/env python3
"""Plot arbitrary slices of the full_sweep.csv dataset.

Example:
    python3 kernel/matmul/plot_sweep.py \\
        --input kernel/matmul/full_sweep.csv \\
        --x threads --y total_cycles \\
        --group-by mat_count \\
        --filter tile_m=8,gemm_m=128 \\
        --output /tmp/smoke.png

Filters accept comma-separated key=val pairs; values can be lists with '|'
(e.g. mat_count=1|2|4). Group-by accepts comma-separated column names --
one series per unique tuple of those columns.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    sys.exit("matplotlib is required. Install with: pip install matplotlib")


NUMERIC_COLUMNS = {
    "tile_m", "tile_k", "tile_n",
    "mat_latency", "mat_count", "vec_count", "vec_bytes",
    "gemm_m", "gemm_k", "gemm_n", "threads",
    "total_cycles", "actual_mat_accels", "actual_vec_accels",
    "wall_seconds", "build_ok", "run_ok",
}


def parse_filter(spec: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"bad --filter entry: '{part}'")
        key, value = part.split("=", 1)
        out[key.strip()] = [v.strip() for v in value.split("|") if v.strip()]
    return out


def parse_group_by(spec: str) -> list[str]:
    return [p.strip() for p in spec.split(",") if p.strip()]


def row_matches(row: dict[str, str], filters: dict[str, list[str]]) -> bool:
    for key, values in filters.items():
        if key not in row:
            return False
        if row[key] not in values:
            return False
    return True


def coerce(col: str, value: str):
    if col in NUMERIC_COLUMNS:
        if value == "":
            return None
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return None
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot slices of full_sweep.csv.")
    p.add_argument("--input", required=True, help="Path to full_sweep.csv.")
    p.add_argument("--x", required=True, help="Column to put on the x-axis.")
    p.add_argument(
        "--y", default="total_cycles",
        help="Metric column (default: total_cycles).",
    )
    p.add_argument(
        "--group-by", default="",
        help="Comma-separated columns; one series per unique tuple.",
    )
    p.add_argument(
        "--filter", default="",
        help="Comma-separated key=val; use key=a|b to accept multiple values.",
    )
    p.add_argument("--log-x", action="store_true", help="Log2-scale x-axis.")
    p.add_argument("--log-y", action="store_true", help="Log10-scale y-axis.")
    p.add_argument("--output", required=True, help="Output PNG path.")
    p.add_argument("--title", default="", help="Optional plot title.")
    p.add_argument(
        "--require-pass", action="store_true",
        help="Drop rows where verification_status != PASS.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    csv_path = Path(args.input)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    filters = parse_filter(args.filter)
    group_cols = parse_group_by(args.group_by)

    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.require_pass and row.get("verification_status") != "PASS":
                continue
            if not row_matches(row, filters):
                continue
            rows.append(row)

    if not rows:
        sys.exit("No rows match the filter.")

    series: dict[tuple, list[tuple]] = defaultdict(list)
    for row in rows:
        key = tuple(row[c] for c in group_cols) if group_cols else ("all",)
        xv = coerce(args.x, row[args.x])
        yv = coerce(args.y, row[args.y])
        if xv is None or yv is None:
            continue
        series[key].append((xv, yv))

    if not series:
        sys.exit("No (x, y) pairs to plot after filtering.")

    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        pass

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    for i, (key, pts) in enumerate(sorted(series.items(), key=lambda kv: kv[0])):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        label = (
            ", ".join(f"{c}={v}" for c, v in zip(group_cols, key))
            if group_cols else "all"
        )
        ax.plot(
            xs, ys,
            marker=markers[i % len(markers)],
            markersize=5.5,
            linewidth=1.6,
            label=label,
        )

    if args.log_x:
        ax.set_xscale("log", base=2)
        xs_all = sorted({p[0] for pts in series.values() for p in pts})
        try:
            ax.set_xticks(xs_all)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        except Exception:
            pass
    if args.log_y:
        ax.set_yscale("log", base=10)
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10))

    ax.set_xlabel(args.x + (" (log2)" if args.log_x else ""))
    ax.set_ylabel(args.y + (" (log10)" if args.log_y else ""))
    if args.title:
        ax.set_title(args.title, fontweight="bold")
    elif group_cols:
        ax.set_title(
            f"{args.y} vs {args.x}  grouped by {','.join(group_cols)}",
            fontweight="bold",
        )
    else:
        ax.set_title(f"{args.y} vs {args.x}", fontweight="bold")
    ax.grid(True, which="major", linestyle="-", linewidth=0.45, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if group_cols:
        ax.legend(fontsize="small", frameon=False, ncol=2)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
