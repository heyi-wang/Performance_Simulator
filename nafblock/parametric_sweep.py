#!/usr/bin/env python3
"""Parametric sweep of the NafBlock simulator.

Two axes, mirroring the kernel-level sweeps under ``kernel/<op>/parametric_sweep.py``:

  * **Block shape** (``--block-c/--block-h/--block-w``): the five canonical
    NAFNet32 levels by default ((C,H)=(32,64), (64,32), (128,16), (256,8),
    (512,4)) — each block is run with square spatial extent H==W.
  * **Hardware** (compile-time): ``MAT_ACCEL_COUNT`` and ``VEC_ACCEL_COUNT``
    overridden via ``EXTRA_CXXFLAGS``. Default: (2,4) only.

For every (HW, shape) point the script:

  1. Rebuilds ``nafblock/build/nafblock_perf_sim`` if HW changed.
  2. Runs the simulator and parses the report.
  3. Records total cycles, per-backend cycle/req breakdown, memory traffic,
     critical-path cycle fractions, and the per-layer table into a CSV row.
  4. Emits a PNG plot of total elapsed cycles vs block shape, one line
     per HW configuration.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    sys.exit("matplotlib is required. Install with: pip install matplotlib")


REPO_ROOT = Path(__file__).resolve().parents[1]
NAFBLOCK_DIR = REPO_ROOT / "nafblock"
BINARY = NAFBLOCK_DIR / "build" / "nafblock_perf_sim"

# Default block shapes: the five distinct NafBlock geometries used by
# the NAFNet32 reference network (see nafnet/nafnet_layers.h
# build_nafnet32_layers ENC_CH/ENC_H tables).
DEFAULT_SHAPES = [
    (32, 64),
    (64, 32),
    (128, 16),
    (256, 8),
    (512, 4),
]

# Default HW config matches config/hardware_config.h (MAT=2, VEC=4).
DEFAULT_HW = [(2, 4)]

TOTAL_ELAPSED_RE = re.compile(
    r"^Total Elapsed Cycles\s*\[cycles\]\s*:\s*(\d+)$", re.MULTILINE
)
TOTAL_MAT_REQ_RE = re.compile(
    r"^Total Matrix Accelerator Requests\s*\[requests\]\s*:\s*(\d+)$",
    re.MULTILINE,
)
TOTAL_VEC_REQ_RE = re.compile(
    r"^Total Vector Accelerator Requests\s*\[requests\]\s*:\s*(\d+)$",
    re.MULTILINE,
)
TOTAL_MEM_REQ_RE = re.compile(
    r"^Total Memory Requests\s*\[requests\]\s*:\s*(\d+)$", re.MULTILINE
)
TOTAL_RD_BYTES_RE = re.compile(
    r"^Total Read Bytes\s*\[bytes\]\s*:\s*(\d+)$", re.MULTILINE
)
TOTAL_WR_BYTES_RE = re.compile(
    r"^Total Write Bytes\s*\[bytes\]\s*:\s*(\d+)$", re.MULTILINE
)
TOTAL_STALL_RE = re.compile(
    r"^Total Stall Cycles\s*\[cycles\]\s*:\s*(\d+)$", re.MULTILINE
)
TOTAL_MEM_CYC_RE = re.compile(
    r"^Total Memory Cycles\s*\[cycles\]\s*:\s*(\d+)$", re.MULTILINE
)
TOTAL_SCALAR_RE = re.compile(
    r"^Total Scalar Cycles\s*\[cycles\]\s*:\s*(\d+)$", re.MULTILINE
)
MAT_ACCELS_RE = re.compile(
    r"^Matrix Accelerators\s*\[count\]\s*:\s*(\d+)$", re.MULTILINE
)
VEC_ACCELS_RE = re.compile(
    r"^Vector Accelerators\s*\[count\]\s*:\s*(\d+)$", re.MULTILINE
)
OVERALL_VERIF_RE = re.compile(
    r"^Overall Verification Status\s*:\s*(PASS|FAIL)$", re.MULTILINE
)
# Per-layer summary table row: <name> <BACKEND> <elapsed> <mat_reqs> <vec_reqs>
# <mem_reqs> <rd_bytes> <wr_bytes> <PASS/FAIL>
PER_LAYER_ROW_RE = re.compile(
    r"^(nafblock_\S+)\s+(LAYERNORM|MATMUL|DWCONV|POOLING|VECOPS)\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(PASS|FAIL)\s*$",
    re.MULTILINE,
)


def parse_shape_list(value: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for chunk in value.split(","):
        item = chunk.strip()
        if not item:
            continue
        parts = item.lower().split("x")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"shape '{item}' must be CxH (e.g. 32x64); H==W"
            )
        c = int(parts[0])
        h = int(parts[1])
        if c < 1 or h < 1:
            raise argparse.ArgumentTypeError("shape dims must be >= 1")
        shapes.append((c, h))
    if not shapes:
        raise argparse.ArgumentTypeError("at least one CxH shape required")
    return shapes


def parse_hw_list(value: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for chunk in value.split(","):
        item = chunk.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"hw '{item}' must be MAT:VEC (e.g. 2:4)"
            )
        out.append((int(parts[0]), int(parts[1])))
    if not out:
        raise argparse.ArgumentTypeError("at least one MAT:VEC pair required")
    return out


def rebuild_binary(mat_accels: int, vec_accels: int) -> None:
    extra = f"-DMAT_ACCEL_COUNT={mat_accels} -DVEC_ACCEL_COUNT={vec_accels}"
    print(f"[sweep] rebuilding mat={mat_accels} vec={vec_accels}", flush=True)
    for cmd in (["make", "clean"], ["make", f"EXTRA_CXXFLAGS={extra}"]):
        proc = subprocess.run(cmd, cwd=NAFBLOCK_DIR, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            raise RuntimeError(f"Build failed for mat={mat_accels} vec={vec_accels}")
    if not BINARY.exists():
        raise RuntimeError(f"Binary missing after build: {BINARY}")


def _match_int(rx: re.Pattern[str], text: str, default: int = -1) -> int:
    m = rx.search(text)
    return int(m.group(1)) if m else default


def per_backend_breakdown(text: str) -> dict[str, dict[str, int]]:
    """Aggregate per-layer rows by backend → cycle/req/byte totals."""
    out: dict[str, dict[str, int]] = {}
    for match in PER_LAYER_ROW_RE.finditer(text):
        backend = match.group(2)
        elapsed = int(match.group(3))
        mat_reqs = int(match.group(4))
        vec_reqs = int(match.group(5))
        mem_reqs = int(match.group(6))
        bucket = out.setdefault(
            backend,
            {"cycles": 0, "mat_reqs": 0, "vec_reqs": 0, "mem_reqs": 0, "layers": 0},
        )
        bucket["cycles"] += elapsed
        bucket["mat_reqs"] += mat_reqs
        bucket["vec_reqs"] += vec_reqs
        bucket["mem_reqs"] += mem_reqs
        bucket["layers"] += 1
    return out


def run_point(c: int, h: int, w: int) -> dict[str, int | str]:
    cmd = [
        str(BINARY),
        "--block-c", str(c),
        "--block-h", str(h),
        "--block-w", str(w),
    ]
    print(f"[sweep] running C={c} H={h} W={w}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"Simulator failed for C={c} H={h} W={w}")

    out = proc.stdout
    verif_match = OVERALL_VERIF_RE.search(out)
    if not verif_match:
        sys.stderr.write(out)
        raise RuntimeError("Could not find Overall Verification Status")
    verification = verif_match.group(1)
    if verification != "PASS":
        sys.stderr.write(out)
        raise RuntimeError(f"Verification FAIL for C={c} H={h} W={w}")

    total_cycles = _match_int(TOTAL_ELAPSED_RE, out)
    if total_cycles < 0:
        sys.stderr.write(out)
        raise RuntimeError("Could not parse Total Elapsed Cycles")

    backends = per_backend_breakdown(out)
    metrics: dict[str, int | str] = {
        "total_cycles":     total_cycles,
        "mat_reqs":         _match_int(TOTAL_MAT_REQ_RE, out, 0),
        "vec_reqs":         _match_int(TOTAL_VEC_REQ_RE, out, 0),
        "mem_reqs":         _match_int(TOTAL_MEM_REQ_RE, out, 0),
        "read_bytes":       _match_int(TOTAL_RD_BYTES_RE, out, 0),
        "write_bytes":      _match_int(TOTAL_WR_BYTES_RE, out, 0),
        "stall_cycles":     _match_int(TOTAL_STALL_RE, out, 0),
        "memory_cycles":    _match_int(TOTAL_MEM_CYC_RE, out, 0),
        "scalar_cycles":    _match_int(TOTAL_SCALAR_RE, out, 0),
        "actual_mat_accels": _match_int(MAT_ACCELS_RE, out, 0),
        "actual_vec_accels": _match_int(VEC_ACCELS_RE, out, 0),
        "verification":     verification,
        "wall_seconds":     f"{wall:.3f}",
    }
    for backend in ("LAYERNORM", "MATMUL", "DWCONV", "POOLING", "VECOPS"):
        bucket = backends.get(backend, {})
        metrics[f"{backend.lower()}_cycles"]   = bucket.get("cycles", 0)
        metrics[f"{backend.lower()}_mat_reqs"] = bucket.get("mat_reqs", 0)
        metrics[f"{backend.lower()}_vec_reqs"] = bucket.get("vec_reqs", 0)
        metrics[f"{backend.lower()}_mem_reqs"] = bucket.get("mem_reqs", 0)
    print(
        f"[sweep] C={c} H={h} W={w} "
        f"total={total_cycles} verif={verification} wall={wall:.2f}s",
        flush=True,
    )
    return metrics


def write_csv(rows: list[dict[str, int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "size_label", "block_c", "block_h", "block_w",
        "mat_accels", "vec_accels",
        "actual_mat_accels", "actual_vec_accels",
        "total_cycles", "verification",
        "mat_reqs", "vec_reqs", "mem_reqs",
        "read_bytes", "write_bytes",
        "stall_cycles", "memory_cycles", "scalar_cycles",
        "layernorm_cycles", "matmul_cycles", "dwconv_cycles",
        "pooling_cycles", "vecops_cycles",
        "layernorm_vec_reqs", "matmul_mat_reqs", "matmul_vec_reqs",
        "dwconv_vec_reqs", "pooling_vec_reqs", "vecops_vec_reqs",
        "wall_seconds",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def plot_rows(rows: list[dict[str, int | str]], png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        pass

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11.0, 8.4),
        gridspec_kw={"height_ratios": [1.0, 0.95], "hspace": 0.32},
    )

    hw_configs = sorted({(int(r["mat_accels"]), int(r["vec_accels"])) for r in rows})
    multi_hw = len(hw_configs) > 1

    # ---- Top: total cycles vs block size ----
    by_hw: dict[tuple[int, int], list[tuple[int, str, int]]] = {}
    for r in rows:
        key = (int(r["mat_accels"]), int(r["vec_accels"]))
        spatial_work = int(r["block_c"]) * int(r["block_h"]) * int(r["block_w"])
        by_hw.setdefault(key, []).append(
            (spatial_work, str(r["size_label"]), int(r["total_cycles"]))
        )

    markers = ["o", "s", "^", "D", "v", "P", "X"]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    sorted_hw = sorted(by_hw.items())
    x_labels: list[str] = []
    for idx, (hw, points) in enumerate(sorted_hw):
        points.sort()
        if not x_labels:
            x_labels = [p[1] for p in points]
        ax_top.plot(
            list(range(len(points))),
            [p[2] for p in points],
            marker=markers[idx % len(markers)],
            markersize=6.5,
            linewidth=1.8,
            color=color_cycle[idx % len(color_cycle)] if color_cycle else None,
            label=f"mat={hw[0]} vec={hw[1]}",
        )
    ax_top.set_xticks(list(range(len(x_labels))))
    ax_top.set_xticklabels(x_labels)
    ax_top.set_yscale("log", base=10)
    ax_top.yaxis.set_major_locator(mticker.LogLocator(base=10))
    ax_top.yaxis.set_minor_locator(
        mticker.LogLocator(base=10, subs=tuple(range(2, 10)))
    )
    ax_top.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_top.set_title(
        "NafBlock Performance Sweep — total elapsed cycles per block geometry",
        fontweight="bold",
    )
    ax_top.set_xlabel("Block shape [C × H × W]")
    ax_top.set_ylabel("Total Elapsed Cycles (log10)")
    ax_top.grid(True, which="major", linestyle="-", linewidth=0.45, alpha=0.35)
    ax_top.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.25)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    if multi_hw:
        ax_top.legend(title="Hardware", fontsize="small", frameon=False)

    # ---- Bottom: per-backend cycle breakdown stacked bars (first HW only) ----
    primary_hw = sorted_hw[0][0]
    primary_rows = [
        r for r in rows
        if (int(r["mat_accels"]), int(r["vec_accels"])) == primary_hw
    ]
    primary_rows.sort(key=lambda r: int(r["block_c"]) * int(r["block_h"]) * int(r["block_w"]))
    backend_order = ["MATMUL", "DWCONV", "LAYERNORM", "POOLING", "VECOPS"]
    backend_colors = {
        "MATMUL":    "#4878a8",
        "DWCONV":    "#d57068",
        "LAYERNORM": "#5fa052",
        "POOLING":   "#b079b8",
        "VECOPS":    "#e3a942",
    }
    x = list(range(len(primary_rows)))
    bottom = [0] * len(primary_rows)
    for backend in backend_order:
        key = f"{backend.lower()}_cycles"
        ys = [int(r[key]) for r in primary_rows]
        ax_bot.bar(
            x, ys, bottom=bottom, label=backend,
            color=backend_colors[backend], width=0.62,
        )
        bottom = [b + y for b, y in zip(bottom, ys)]
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([str(r["size_label"]) for r in primary_rows])
    ax_bot.set_title(
        f"Per-backend elapsed cycles (sum of layer rows) — mat={primary_hw[0]} vec={primary_hw[1]}",
        fontweight="bold",
    )
    ax_bot.set_xlabel("Block shape [C × H × W]")
    ax_bot.set_ylabel("Layer Elapsed Cycles (sum across 14 sub-layers)")
    ax_bot.grid(True, axis="y", linestyle="--", linewidth=0.45, alpha=0.4)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_bot.legend(
        title="Backend",
        fontsize="small",
        frameon=False,
        ncol=len(backend_order),
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep NafBlock shapes (and optionally hardware) for the nafblock simulator."
    )
    parser.add_argument(
        "--shapes",
        type=parse_shape_list,
        default=DEFAULT_SHAPES,
        help=(
            "Comma-separated CxH pairs (H is used for both height and width). "
            "Default: NAFNet32 levels 32x64,64x32,128x16,256x8,512x4."
        ),
    )
    parser.add_argument(
        "--hw",
        type=parse_hw_list,
        default=DEFAULT_HW,
        help=(
            "Comma-separated MAT:VEC accelerator-count pairs. Each triggers "
            "a recompile via EXTRA_CXXFLAGS=-DMAT_ACCEL_COUNT=...,VEC_ACCEL_COUNT=.... "
            "Default: 2:4 (matches config/hardware_config.h)."
        ),
    )
    parser.add_argument(
        "--csv",
        default=str(NAFBLOCK_DIR / "parametric_sweep.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--png",
        default=str(NAFBLOCK_DIR / "parametric_sweep.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--plot-from-csv",
        action="store_true",
        help="Regenerate PNG from existing CSV only.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.plot_from_csv:
        rows = read_csv(Path(args.csv))
        if not rows:
            parser.error(f"no rows in {args.csv}")
        plot_rows(rows, Path(args.png))
        print(f"[sweep] wrote plot to {args.png}")
        return 0

    rows: list[dict[str, int | str]] = []
    for mat_accels, vec_accels in args.hw:
        rebuild_binary(mat_accels, vec_accels)
        for (c, h) in args.shapes:
            w = h
            metrics = run_point(c, h, w)
            row: dict[str, int | str] = {
                "size_label":  f"{c}x{h}x{w}",
                "block_c":     c,
                "block_h":     h,
                "block_w":     w,
                "mat_accels":  mat_accels,
                "vec_accels":  vec_accels,
            }
            row.update(metrics)
            rows.append(row)

    write_csv(rows, Path(args.csv))
    plot_rows(rows, Path(args.png))
    print(f"[sweep] wrote CSV  to {args.csv}")
    print(f"[sweep] wrote plot to {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
