#!/usr/bin/env python3
"""Sweep vec_ops input size, thread count, and operation type.

Aligned with kernel/layer_norm/parametric_sweep.py,
kernel/dw_conv2d/parametric_sweep.py, and
kernel/pooling/parametric_sweep.py so the resulting CSV is consumable by
the existing matmul/pooling plotting helpers.

Vec_ops-specific dimension: --ops chooses which kernel functions from
vector_ops.h to sweep (default: all six). Each op carries its own
per-call insn count and rd/wr widths, so the sweep generates one curve
per (size, op) tuple.
"""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIZE_MULTIPLIERS = [1, 2, 4, 8]
DEFAULT_CHANNEL_MULTIPLIERS = [1]
# Sweep default channel count. Overrides whatever VOP_C is set to in
# vec_ops_config.h so that --max-workers up to 64 leaves every thread
# at least one channel to process.
SWEEP_BASE_CHANNELS = 64

# Op names accepted by `vec_ops_sim --op`. Order picked so the cheapest
# ops sort to the front of the legend.
ALL_OPS = [
    "elemwise_add",
    "elemwise_mul",
    "scalar_mul",
    "quantize_i32_to_i8",
    "quantize_i16_to_i8",
    "dequantize_i8_to_i32",
]


TOTAL_ELAPSED_RE = re.compile(
    r"^Total Elapsed Cycles\s*\[cycles\]\s*:\s*(\d+)$",
    re.MULTILINE,
)
VECTOR_ACCEL_RE = re.compile(
    r"^Vector Accelerators\s*\[count\]\s*:\s*(\d+)$",
    re.MULTILINE,
)
VERIFICATION_RE = re.compile(
    r"^Verification Status\s*:\s*(PASS|FAIL)$",
    re.MULTILINE,
)
SLOWEST_WORKER_RE = re.compile(
    r"^Critical-Path Worker\s*\[tid\]\s*:\s*(-?\d+)$",
    re.MULTILINE,
)
INSN_COUNT_RE = re.compile(
    r"^Op Vector Instructions\s*\[insns/request\]\s*:\s*(\d+)$",
    re.MULTILINE,
)
TILE_CAP_RE = re.compile(
    r"^Vector Accelerator Capacity\s*\[elements/request\]\s*:\s*(\d+)$",
    re.MULTILINE,
)
CYCLE_FRACTION_RE = {
    "vec_cycle_fraction_pct": re.compile(
        r"^Vec Cycle Fraction\s*\[%\]\s*:\s*([\d.]+)\s*%$",
        re.MULTILINE,
    ),
    "dma_cycle_fraction_pct": re.compile(
        r"^DMA Cycle Fraction\s*\[%\]\s*:\s*([\d.]+)\s*%$",
        re.MULTILINE,
    ),
    "scalar_cycle_fraction_pct": re.compile(
        r"^Scalar Cycle Fraction\s*\[%\]\s*:\s*([\d.]+)\s*%$",
        re.MULTILINE,
    ),
    "stall_cycle_fraction_pct": re.compile(
        r"^Stall Cycle Fraction\s*\[%\]\s*:\s*([\d.]+)\s*%$",
        re.MULTILINE,
    ),
}
CONFIG_VEC_ACCEL_RE = re.compile(
    r"^\s*#define\s+VEC_ACCEL_COUNT\s+(\d+)\s*$"
)
CONFIG_VECTOR_CAP_RE = re.compile(
    r"^\s*#define\s+VECTOR_ACC_CAP\s+(\d+)\s*$"
)
VOP_DIM_RE = re.compile(
    r"^\s*static\s+const\s+int\s+(VOP_[CHW])\s*=\s*(\d+)\s*;"
)


def parse_positive_int_list(value: str, name: str) -> list[int]:
    values: list[int] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed < 1:
            raise argparse.ArgumentTypeError(f"{name} values must be >= 1")
        values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError(f"expected at least one {name} value")
    return values


def parse_count_list(value: str) -> list[int]:
    return parse_positive_int_list(value, "accelerator count")


def parse_multiplier_list(value: str) -> list[int]:
    return parse_positive_int_list(value, "size multiplier")


def parse_op_list(value: str) -> list[str]:
    ops: list[str] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if item not in ALL_OPS:
            raise argparse.ArgumentTypeError(
                f"unknown op '{item}'. Choose from: {', '.join(ALL_OPS)}"
            )
        ops.append(item)
    if not ops:
        raise argparse.ArgumentTypeError("expected at least one op")
    return ops


def read_defaults() -> dict[str, int]:
    defaults = {
        "vec_accels": 1,
        "vec_bytes": 64,
        "channels": 32,
        "height": 64,
        "width": 64,
    }

    hardware_path = REPO_ROOT / "config" / "hardware_config.h"
    for line in hardware_path.read_text().splitlines():
        match = CONFIG_VEC_ACCEL_RE.match(line)
        if match:
            defaults["vec_accels"] = int(match.group(1))
            continue
        match = CONFIG_VECTOR_CAP_RE.match(line)
        if match:
            defaults["vec_bytes"] = int(match.group(1))

    vop_config_path = REPO_ROOT / "kernel" / "vec_ops" / "vec_ops_config.h"
    name_to_key = {
        "VOP_C": "channels",
        "VOP_H": "height",
        "VOP_W": "width",
    }
    for line in vop_config_path.read_text().splitlines():
        match = VOP_DIM_RE.match(line)
        if match:
            name, value = match.groups()
            key = name_to_key.get(name)
            if key is not None:
                defaults[key] = int(value)
    # Force the sweep to use SWEEP_BASE_CHANNELS so workers up to 64 each
    # own at least one channel. The kernel's VOP_C default stays untouched.
    defaults["channels"] = SWEEP_BASE_CHANNELS
    return defaults


def power_of_two_workers(max_workers: int) -> list[int]:
    workers: list[int] = []
    value = 1
    while value <= max_workers:
        workers.append(value)
        value *= 2
    return workers


def build_parser() -> argparse.ArgumentParser:
    defaults = read_defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Sweep vec_ops input size, threads, and op type; "
            "then plot elapsed cycles."
        )
    )
    parser.add_argument(
        "--max-workers",
        "--max-threads",
        dest="max_workers",
        type=int,
        default=64,
        help="Largest thread count to sweep. Points are 1, 2, 4, ... up to this value.",
    )
    parser.add_argument(
        "--size-multipliers",
        type=parse_multiplier_list,
        default=DEFAULT_SIZE_MULTIPLIERS,
        help=(
            "Comma-separated multipliers applied to height and width. "
            "Default: 1,2,4,8."
        ),
    )
    parser.add_argument(
        "--channel-multipliers",
        type=parse_multiplier_list,
        default=DEFAULT_CHANNEL_MULTIPLIERS,
        help=(
            "Comma-separated multipliers applied to the channel count. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--vec-accels",
        type=parse_count_list,
        default=[defaults["vec_accels"]],
        help=(
            "Comma-separated compile-time vector accelerator counts. "
            f"Default: {defaults['vec_accels']} from config/hardware_config.h."
        ),
    )
    parser.add_argument(
        "--vec-bytes",
        type=int,
        default=defaults["vec_bytes"],
        help=(
            "Vector datapath width column to write to CSV. "
            f"Default: {defaults['vec_bytes']} from config/hardware_config.h."
        ),
    )
    parser.add_argument(
        "--ops",
        type=parse_op_list,
        default=ALL_OPS,
        help=(
            "Comma-separated op names to sweep. "
            f"Default: all six ({','.join(ALL_OPS)})."
        ),
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=defaults["channels"],
        help=f"Base channel count before --channel-multipliers. Default: {defaults['channels']}.",
    )
    parser.add_argument(
        "--base-height",
        type=int,
        default=defaults["height"],
        help=f"Base input height before --size-multipliers. Default: {defaults['height']}.",
    )
    parser.add_argument(
        "--base-width",
        type=int,
        default=defaults["width"],
        help=f"Base input width before --size-multipliers. Default: {defaults['width']}.",
    )
    parser.add_argument(
        "--binary",
        default=str(REPO_ROOT / "kernel" / "build" / "vec_ops_sim"),
        help="Path to the vec_ops simulator binary.",
    )
    parser.add_argument(
        "--make-dir",
        default=str(REPO_ROOT / "kernel"),
        help="Directory containing the vec_ops Makefile.",
    )
    parser.add_argument(
        "--csv",
        default=str(REPO_ROOT / "kernel" / "vec_ops" / "parametric_sweep.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--png",
        default=str(REPO_ROOT / "kernel" / "vec_ops" / "parametric_sweep.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--plot-from-csv",
        action="store_true",
        help="Only regenerate the PNG from --csv; do not rebuild or rerun simulations.",
    )
    return parser


def rebuild_binary(make_dir: Path, vec_accels: int) -> None:
    extra = f"-DVEC_ACCEL_COUNT={vec_accels}"
    commands = (
        ["make", "clean"],
        ["make", "vec_ops", f"EXTRA_CXXFLAGS={extra}"],
    )

    print(f"[sweep] rebuilding vec={vec_accels}", flush=True)
    for command in commands:
        proc = subprocess.run(command, cwd=make_dir, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            raise RuntimeError(f"Build failed for vec={vec_accels}")


def _regex_value(pattern: re.Pattern[str], text: str, default: str = "") -> str:
    match = pattern.search(text)
    return match.group(1) if match else default


def run_point(
    binary: Path,
    workers: int,
    channels: int,
    height: int,
    width: int,
    op: str,
) -> dict[str, int | str]:
    command = [
        str(binary),
        "--workers", str(workers),
        "--channels", str(channels),
        "--height", str(height),
        "--width", str(width),
        "--op", op,
    ]

    print(
        "[sweep] running "
        f"op={op} shape={channels}x{height}x{width} workers={workers}",
        flush=True,
    )
    t0 = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True)
    wall_seconds = time.perf_counter() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(
            f"Simulator failed for op={op}, "
            f"shape={channels}x{height}x{width}, workers={workers}"
        )

    elapsed_match = TOTAL_ELAPSED_RE.search(proc.stdout)
    if not elapsed_match:
        sys.stderr.write(proc.stdout)
        raise RuntimeError(
            f"Could not find total elapsed cycles for op={op}, workers={workers}"
        )

    verification_match = VERIFICATION_RE.search(proc.stdout)
    if not verification_match:
        sys.stderr.write(proc.stdout)
        raise RuntimeError(
            f"Could not find verification status for op={op}, workers={workers}"
        )

    verification_status = verification_match.group(1)
    if verification_status != "PASS":
        sys.stderr.write(proc.stdout)
        raise RuntimeError(
            f"Verification failed for op={op}, "
            f"shape={channels}x{height}x{width}, workers={workers}"
        )

    vec_match = VECTOR_ACCEL_RE.search(proc.stdout)
    if not vec_match:
        sys.stderr.write(proc.stdout)
        raise RuntimeError("Could not find vector accelerator count in report")

    cycles = int(elapsed_match.group(1))
    actual_vec = int(vec_match.group(1))
    print(
        "[sweep] "
        f"op={op} shape={channels}x{height}x{width} workers={workers} "
        f"vec={actual_vec} total_cycles={cycles}",
        flush=True,
    )
    metrics: dict[str, int | str] = {
        "total_cycles": cycles,
        "actual_vec_accels": actual_vec,
        "verification_status": verification_status,
        "slowest_worker_tid": _regex_value(SLOWEST_WORKER_RE, proc.stdout, "-1"),
        "op_insn_count": _regex_value(INSN_COUNT_RE, proc.stdout, "0"),
        "op_tile_cap": _regex_value(TILE_CAP_RE, proc.stdout, "0"),
        "wall_seconds": f"{wall_seconds:.3f}",
        "build_ok": 1,
        "run_ok": 1,
    }
    for key, pattern in CYCLE_FRACTION_RE.items():
        metrics[key] = _regex_value(pattern, proc.stdout, "0.0")
    return metrics


def write_csv(rows: list[dict[str, int | str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        # Matmul/pooling full_sweep-compatible columns first.
        "tile_m",
        "tile_k",
        "tile_n",
        "mat_latency",
        "mat_count",
        "vec_count",
        "vec_bytes",
        "gemm_m",
        "gemm_k",
        "gemm_n",
        "threads",
        "dma_base_lat",
        "total_cycles",
        "verification_status",
        "actual_mat_accels",
        "actual_vec_accels",
        "slowest_worker_tid",
        "mat_util_pct",
        "vec_util_pct",
        "mat_cycle_fraction_pct",
        "vec_cycle_fraction_pct",
        "dma_cycle_fraction_pct",
        "scalar_cycle_fraction_pct",
        "stall_cycle_fraction_pct",
        "wall_seconds",
        "build_ok",
        "run_ok",
        # VecOps-specific aliases for humans and ad-hoc tooling.
        "size_label",
        "vop_op",
        "vop_insn_count",
        "vop_tile_cap",
        "vop_channels",
        "vop_height",
        "vop_width",
        "workers",
        "vec_accels",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def row_threads(row: dict[str, int | str]) -> int:
    value = row.get("threads") or row.get("workers")
    if value is None:
        raise KeyError("expected either 'threads' or 'workers' in sweep CSV")
    return int(value)


def plot_rows(rows: list[dict[str, int | str]], png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        pass

    fig, ax = plt.subplots(figsize=(11.5, 6.8))

    hardware_configs = sorted({int(row["vec_accels"]) for row in rows})
    include_hardware_in_label = len(hardware_configs) > 1
    size_labels = sorted({str(row["size_label"]) for row in rows})
    include_size_in_label = len(size_labels) > 1

    grouped: dict[str, list[tuple[int, int, int]]] = {}
    for row in rows:
        label_parts = [str(row["vop_op"])]
        if include_size_in_label:
            label_parts.append(str(row["size_label"]))
        if include_hardware_in_label:
            label_parts.append(f"vec={row['vec_accels']}")
        label = " | ".join(label_parts)
        workers = row_threads(row)
        cycles = int(row["total_cycles"])
        work = (
            int(row["vop_channels"]) *
            int(row["vop_height"]) *
            int(row["vop_width"])
        )
        grouped.setdefault(label, []).append((work, workers, cycles))

    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, (label, points) in enumerate(
        sorted(grouped.items(), key=lambda item: (item[1][0][0], item[0]))
    ):
        points.sort()
        ax.plot(
            [point[1] for point in points],
            [point[2] for point in points],
            marker=markers[idx % len(markers)],
            markersize=5.5,
            linewidth=1.7,
            color=color_cycle[idx % len(color_cycle)] if color_cycle else None,
            label=label,
        )

    worker_ticks = sorted({row_threads(row) for row in rows})
    hardware_label = ", ".join(f"vec={vec_accels}" for vec_accels in hardware_configs)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xticks(worker_ticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10, subs=tuple(range(2, 10)))
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_title(
        f"Vec Ops Performance Simulation (op × shape × threads), {hardware_label}",
        fontweight="bold",
    )
    ax.set_xlabel("Threads (log2 scale)")
    ax.set_ylabel("Total Elapsed Cycles (log10 scale)")
    ax.grid(True, which="major", linestyle="-", linewidth=0.45, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        title="op | [C,H,W]",
        fontsize="small",
        title_fontsize="small",
        frameon=False,
        ncol=2,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.max_workers < 1:
        parser.error("--max-workers/--max-threads must be >= 1")
    if args.base_channels < 1 or args.base_height < 1 or args.base_width < 1:
        parser.error("base dimensions must be >= 1")

    if args.plot_from_csv:
        rows = read_csv(Path(args.csv))
        if not rows:
            parser.error(f"No sweep rows found in {args.csv}")
        plot_rows(rows, Path(args.png))
        print(f"[sweep] wrote plot to {args.png}")
        return 0

    worker_values = power_of_two_workers(args.max_workers)
    binary = Path(args.binary)
    make_dir = Path(args.make_dir)
    rows: list[dict[str, int | str]] = []

    for vec_accels in args.vec_accels:
        rebuild_binary(make_dir, vec_accels)
        if not binary.exists():
            parser.error(f"Simulator binary not found after build: {binary}")

        for channel_multiplier in args.channel_multipliers:
            channels = args.base_channels * channel_multiplier
            for size_multiplier in args.size_multipliers:
                height = args.base_height * size_multiplier
                width = args.base_width * size_multiplier
                size_label = f"{channels}x{height}x{width}"
                for op in args.ops:
                    for workers in worker_values:
                        metrics = run_point(
                            binary,
                            workers,
                            channels,
                            height,
                            width,
                            op,
                        )
                        actual_vec = int(metrics["actual_vec_accels"])
                        rows.append(
                            {
                                "tile_m": 0,
                                "tile_k": 0,
                                "tile_n": 0,
                                "mat_latency": 0,
                                "mat_count": 0,
                                "vec_count": vec_accels,
                                "vec_bytes": args.vec_bytes,
                                "gemm_m": channels,
                                "gemm_k": height,
                                "gemm_n": width,
                                "threads": workers,
                                "dma_base_lat": 10,
                                "total_cycles": metrics["total_cycles"],
                                "verification_status": metrics["verification_status"],
                                "actual_mat_accels": 0,
                                "actual_vec_accels": actual_vec,
                                "slowest_worker_tid": metrics["slowest_worker_tid"],
                                "mat_util_pct": "0.0",
                                "vec_util_pct": "",
                                "mat_cycle_fraction_pct": "0.0",
                                "vec_cycle_fraction_pct": metrics["vec_cycle_fraction_pct"],
                                "dma_cycle_fraction_pct": metrics["dma_cycle_fraction_pct"],
                                "scalar_cycle_fraction_pct": metrics["scalar_cycle_fraction_pct"],
                                "stall_cycle_fraction_pct": metrics["stall_cycle_fraction_pct"],
                                "wall_seconds": metrics["wall_seconds"],
                                "build_ok": metrics["build_ok"],
                                "run_ok": metrics["run_ok"],
                                "size_label": size_label,
                                "vop_op": op,
                                "vop_insn_count": metrics["op_insn_count"],
                                "vop_tile_cap": metrics["op_tile_cap"],
                                "vop_channels": channels,
                                "vop_height": height,
                                "vop_width": width,
                                "workers": workers,
                                "vec_accels": vec_accels,
                            }
                        )

    write_csv(rows, Path(args.csv))
    plot_rows(rows, Path(args.png))

    print(f"[sweep] wrote CSV to {args.csv}")
    print(f"[sweep] wrote plot to {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
