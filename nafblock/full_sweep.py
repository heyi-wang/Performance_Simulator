#!/usr/bin/env python3
"""Full 8-dimensional parametric sweep driver for the NafBlock simulator.

Mirrors kernel/matmul/full_sweep.py. See nafblock/FULL_Sweep.md +
nafblock/Parametric_Sweep_HOWTO.md.

Dimensions:
    - matrix tile size  : MATMUL_M x MATMUL_K x MATMUL_N       (compile-time)
    - matrix latency    : MATMUL_ACC_CYCLE                     (compile-time)
    - matrix count      : MAT_ACCEL_COUNT                      (compile-time)
    - vector count      : VEC_ACCEL_COUNT                      (compile-time)
    - vector bytes      : VECTOR_ACC_CAP                       (compile-time)
    - worker count      : NAFBLOCK_N_WORKERS                   (compile-time)
    - block shape       : --block-c/--block-h/--block-w        (runtime CLI)
    - dma base latency  : --dma-base-lat                       (runtime CLI)

The 6 compile-time dimensions define a "hardware point". Each unique HW
point is built once into nafblock/.sweep_bin/<tag>/ then re-invoked across
the (block-shape x dma-base-lat) grid.

Designed so the user can validate the interface locally with --dry-run and a
tiny sub-grid, then run the full cartesian product on a server.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NAFBLOCK_DIR = REPO_ROOT / "nafblock"
DEFAULT_CSV = NAFBLOCK_DIR / "full_sweep.csv"
SWEEP_BUILD_ROOT = NAFBLOCK_DIR / ".sweep_bin"

# ------------------------------------------------------------
# Report parsers (regexes mirror nafblock/parametric_sweep.py).
# ------------------------------------------------------------
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
WORKERS_COUNT_RE = re.compile(
    r"^Workers\s*\[count\]\s*:\s*(\d+)$", re.MULTILINE
)
MAT_ACCELS_RE = re.compile(
    r"^Matrix Accelerators\s*\[count\]\s*:\s*(\d+)$", re.MULTILINE
)
VEC_ACCELS_RE = re.compile(
    r"^Vector Accelerators\s*\[count\]\s*:\s*(\d+)$", re.MULTILINE
)
VEC_CAP_RE = re.compile(
    r"^Vector Accelerator Capacity\s*\[elements/request\]\s*:\s*(\d+)$",
    re.MULTILINE,
)
OVERALL_VERIF_RE = re.compile(
    r"^Overall Verification Status\s*:\s*(PASS|FAIL)$", re.MULTILINE
)
# Pool-level compute utilization rows: "Matrix Accelerator  pool-level  ... X.YY %"
ACCEL_POOL_UTIL_RE = re.compile(
    r"^(Matrix|Vector) Accelerator\s+pool-level\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+"
    r"\s+([\d.]+)\s*%\s+([\d.]+)\s*%",
    re.MULTILINE,
)
# Per-layer summary table row.
PER_LAYER_ROW_RE = re.compile(
    r"^(nafblock_\S+)\s+(LAYERNORM|MATMUL|DWCONV|POOLING|VECOPS)\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(PASS|FAIL)\s*$",
    re.MULTILINE,
)

# ------------------------------------------------------------
# Defaults from nafblock/FULL_Sweep.md (5 unique block shapes after dedup of
# the user-supplied U-Net mirror; HOWTO documents the encoder/decoder
# multiplicity).
# ------------------------------------------------------------
DEFAULT_TILES: list[tuple[int, int, int]] = [
    (8, 8, 8),
    (16, 32, 64),
    (16, 16, 16),
    (32, 64, 128),
]
DEFAULT_MAT_LATENCIES: list[int] = [2, 4, 8, 16, 32, 64]
DEFAULT_MAT_COUNTS: list[int] = [1, 2, 4, 8]
DEFAULT_VEC_COUNTS: list[int] = [1, 2, 4, 8]
DEFAULT_VEC_BYTES: list[int] = [16, 32, 64, 128, 256]
DEFAULT_N_WORKERS: list[int] = [1, 2, 4, 8, 16, 32, 64]
DEFAULT_BLOCK_SHAPES: list[tuple[int, int, int]] = [
    (32, 512, 512),
    (64, 256, 256),
    (128, 128, 128),
    (256, 64, 64),
    (512, 32, 32),
]
DEFAULT_DMA_BASE_LATS: list[int] = [4, 8, 16, 32]

BACKENDS = ("LAYERNORM", "MATMUL", "DWCONV", "POOLING", "VECOPS")

KEY_FIELDS = [
    "tile_m", "tile_k", "tile_n",
    "mat_latency", "mat_count", "vec_count", "vec_bytes", "n_workers",
    "block_c", "block_h", "block_w", "dma_base_lat",
]

CSV_FIELDS = KEY_FIELDS + [
    "total_cycles", "verification_status",
    "actual_workers", "actual_mat_accels", "actual_vec_accels",
    "actual_vec_cap",
    "mat_pool_util_pct", "vec_pool_util_pct",
    "mat_pool_occupancy_pct", "vec_pool_occupancy_pct",
    "mat_reqs", "vec_reqs", "mem_reqs",
    "read_bytes", "write_bytes",
    "stall_cycles", "memory_cycles", "scalar_cycles",
] + [
    f"{b.lower()}_cycles" for b in BACKENDS
] + [
    f"{b.lower()}_cycle_fraction_pct" for b in BACKENDS
] + [
    f"{b.lower()}_mat_reqs" for b in BACKENDS
] + [
    f"{b.lower()}_vec_reqs" for b in BACKENDS
] + [
    f"{b.lower()}_mem_reqs" for b in BACKENDS
] + [
    "wall_seconds", "build_ok", "run_ok",
]


@dataclass(frozen=True)
class HwPoint:
    tile_m: int
    tile_k: int
    tile_n: int
    mat_latency: int
    mat_count: int
    vec_count: int
    vec_bytes: int
    n_workers: int

    @property
    def tag(self) -> str:
        raw = (
            f"m{self.tile_m}_k{self.tile_k}_n{self.tile_n}"
            f"_lat{self.mat_latency}_mc{self.mat_count}_vc{self.vec_count}"
            f"_vb{self.vec_bytes}_nw{self.n_workers}"
        )
        digest = hashlib.sha1(raw.encode()).hexdigest()[:8]
        return f"{raw}__{digest}"

    def extra_cxxflags(self) -> str:
        return (
            f"-DMATMUL_M={self.tile_m} -DMATMUL_K={self.tile_k} "
            f"-DMATMUL_N={self.tile_n} -DMATMUL_ACC_CYCLE={self.mat_latency} "
            f"-DMAT_ACCEL_COUNT={self.mat_count} "
            f"-DVEC_ACCEL_COUNT={self.vec_count} "
            f"-DVECTOR_ACC_CAP={self.vec_bytes} "
            f"-DNAFBLOCK_N_WORKERS={self.n_workers}"
        )


@dataclass(frozen=True)
class SweepPoint:
    hw: HwPoint
    block_c: int
    block_h: int
    block_w: int
    dma_base_lat: int

    def key(self) -> tuple:
        return (
            self.hw.tile_m, self.hw.tile_k, self.hw.tile_n,
            self.hw.mat_latency, self.hw.mat_count, self.hw.vec_count,
            self.hw.vec_bytes, self.hw.n_workers,
            self.block_c, self.block_h, self.block_w, self.dma_base_lat,
        )


# ------------------------------------------------------------
# CLI parsing helpers
# ------------------------------------------------------------
def _parse_int_list(value: str, name: str) -> list[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(f"{name} must have at least one value")
    out: list[int] = []
    for p in parts:
        try:
            v = int(p)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{name}: '{p}' is not an integer"
            ) from exc
        if v < 1:
            raise argparse.ArgumentTypeError(f"{name}: '{p}' must be >= 1")
        out.append(v)
    return out


def _parse_triple_list(value: str, name: str) -> list[tuple[int, int, int]]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(f"{name} must have at least one value")
    out: list[tuple[int, int, int]] = []
    for p in parts:
        pieces = p.lower().replace("*", "x").split("x")
        if len(pieces) != 3:
            raise argparse.ArgumentTypeError(
                f"{name}: '{p}' must be AxBxC (e.g. 16x32x64)"
            )
        try:
            triple = tuple(int(x) for x in pieces)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{name}: '{p}' has non-integer component"
            ) from exc
        if any(x < 1 for x in triple):
            raise argparse.ArgumentTypeError(f"{name}: '{p}' must be positive")
        out.append(triple)  # type: ignore[arg-type]
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Full 8-D nafblock parametric sweep. Defaults follow "
            "nafblock/FULL_Sweep.md; filter flags let you restrict to any "
            "sub-grid."
        )
    )
    p.add_argument(
        "--tile-sizes",
        type=lambda v: _parse_triple_list(v, "--tile-sizes"),
        default=DEFAULT_TILES,
        help="Matmul tile triples MxKxN. Default: full spec list.",
    )
    p.add_argument(
        "--mat-latencies",
        type=lambda v: _parse_int_list(v, "--mat-latencies"),
        default=DEFAULT_MAT_LATENCIES,
        help="Matrix accelerator latency (cycles). Default: 2,4,8,16,32,64.",
    )
    p.add_argument(
        "--mat-counts",
        type=lambda v: _parse_int_list(v, "--mat-counts"),
        default=DEFAULT_MAT_COUNTS,
        help="Matrix unit counts. Default: 1,2,4,8.",
    )
    p.add_argument(
        "--vec-counts",
        type=lambda v: _parse_int_list(v, "--vec-counts"),
        default=DEFAULT_VEC_COUNTS,
        help="Vector unit counts. Default: 1,2,4,8.",
    )
    p.add_argument(
        "--vec-bytes",
        type=lambda v: _parse_int_list(v, "--vec-bytes"),
        default=DEFAULT_VEC_BYTES,
        help="VECTOR_ACC_CAP widths in bytes. Default: 16,32,64,128,256.",
    )
    p.add_argument(
        "--n-workers",
        type=lambda v: _parse_int_list(v, "--n-workers"),
        default=DEFAULT_N_WORKERS,
        help=(
            "NAFBLOCK_N_WORKERS (per-block worker count). Default: "
            "1,2,4,8,16,32,64."
        ),
    )
    p.add_argument(
        "--block-shapes",
        type=lambda v: _parse_triple_list(v, "--block-shapes"),
        default=DEFAULT_BLOCK_SHAPES,
        help=(
            "Block shapes CxHxW. Default (5 unique NAFNet32 levels): "
            "32x512x512,64x256x256,128x128x128,256x64x64,512x32x32."
        ),
    )
    p.add_argument(
        "--dma-base-lats",
        type=lambda v: _parse_int_list(v, "--dma-base-lats"),
        default=DEFAULT_DMA_BASE_LATS,
        help=(
            "L2/DMA base latency (cycles). Forwarded to every kernel's "
            "l2_base_lat (and matmul's dma_base_lat). Default: 4,8,16,32."
        ),
    )
    p.add_argument(
        "--output",
        default=str(DEFAULT_CSV),
        help=f"Output CSV path. Default: {DEFAULT_CSV}.",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Parallel worker count (1 = serial). Each worker builds and runs "
            "one HW point at a time in its own private BUILDDIR; recommend "
            "--jobs $(nproc) on a server."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate points and print build/run commands without executing.",
    )
    p.add_argument(
        "--keep-build-dirs",
        action="store_true",
        help="Do not remove per-point build dirs after the sweep finishes.",
    )
    p.add_argument(
        "--build-root",
        default=str(SWEEP_BUILD_ROOT),
        help=f"Root for per-point build dirs. Default: {SWEEP_BUILD_ROOT}.",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite existing CSV instead of skipping already-computed rows.",
    )
    return p


# ------------------------------------------------------------
# Sweep grid
# ------------------------------------------------------------
def iter_hw_points(args: argparse.Namespace) -> list[HwPoint]:
    out: list[HwPoint] = []
    for (tm, tk, tn), lat, mc, vc, vb, nw in product(
        args.tile_sizes, args.mat_latencies,
        args.mat_counts, args.vec_counts, args.vec_bytes,
        args.n_workers,
    ):
        out.append(HwPoint(tm, tk, tn, lat, mc, vc, vb, nw))
    return out


def iter_sweep_points(
    args: argparse.Namespace, hw_points: list[HwPoint]
) -> list[SweepPoint]:
    runtime_grid = list(product(args.block_shapes, args.dma_base_lats))
    out: list[SweepPoint] = []
    for hw in hw_points:
        for (bc, bh, bw), dma_lat in runtime_grid:
            out.append(SweepPoint(hw, bc, bh, bw, dma_lat))
    return out


def load_existing_keys(csv_path: Path) -> set[tuple]:
    if not csv_path.exists():
        return set()
    keys: set[tuple] = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                keys.add(tuple(int(row[k]) for k in KEY_FIELDS))
            except (KeyError, ValueError):
                continue
    return keys


# ------------------------------------------------------------
# Build / run
# ------------------------------------------------------------
def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def build_hw_point(hw: HwPoint, build_root: Path) -> tuple[Path, str]:
    """Build the nafblock simulator for `hw` into a private BUILDDIR.

    Returns (binary_path, log). The nafblock Makefile derives TARGET from
    BUILDDIR (see nafblock/Makefile after the v3 sweep prep), so different
    hardware points compile concurrently into separate dirs.
    """
    point_dir = (build_root / hw.tag).resolve()
    point_dir.mkdir(parents=True, exist_ok=True)
    binary = point_dir / "nafblock_perf_sim"

    extra = hw.extra_cxxflags()
    build = _run(
        ["make",
         f"BUILDDIR={point_dir}",
         f"EXTRA_CXXFLAGS={extra}"],
        cwd=NAFBLOCK_DIR,
    )
    log = (
        f"$ make -C nafblock BUILDDIR='{point_dir}' EXTRA_CXXFLAGS='{extra}'\n"
        f"{build.stdout}{build.stderr}"
    )
    if build.returncode != 0 or not binary.exists():
        if build.returncode == 0:
            log += f"\nERROR: expected {binary} after build"
        return binary, log

    os.chmod(binary, 0o755)
    return binary, log


def _match_int(rx: re.Pattern[str], text: str, default: object = "") -> object:
    m = rx.search(text)
    return int(m.group(1)) if m else default


def _per_backend_breakdown(text: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {
        b: {"cycles": 0, "mat_reqs": 0, "vec_reqs": 0, "mem_reqs": 0}
        for b in BACKENDS
    }
    for match in PER_LAYER_ROW_RE.finditer(text):
        backend = match.group(2)
        out[backend]["cycles"]    += int(match.group(3))
        out[backend]["mat_reqs"]  += int(match.group(4))
        out[backend]["vec_reqs"]  += int(match.group(5))
        out[backend]["mem_reqs"]  += int(match.group(6))
    return out


def _parse_run_output(stdout: str) -> dict[str, object]:
    out: dict[str, object] = {}
    out["total_cycles"]   = _match_int(TOTAL_ELAPSED_RE, stdout)
    verif = OVERALL_VERIF_RE.search(stdout)
    out["verification_status"] = verif.group(1) if verif else ""
    out["actual_workers"]      = _match_int(WORKERS_COUNT_RE, stdout)
    out["actual_mat_accels"]   = _match_int(MAT_ACCELS_RE, stdout)
    out["actual_vec_accels"]   = _match_int(VEC_ACCELS_RE, stdout)
    out["actual_vec_cap"]      = _match_int(VEC_CAP_RE, stdout)
    out["mat_reqs"]    = _match_int(TOTAL_MAT_REQ_RE, stdout)
    out["vec_reqs"]    = _match_int(TOTAL_VEC_REQ_RE, stdout)
    out["mem_reqs"]    = _match_int(TOTAL_MEM_REQ_RE, stdout)
    out["read_bytes"]  = _match_int(TOTAL_RD_BYTES_RE, stdout)
    out["write_bytes"] = _match_int(TOTAL_WR_BYTES_RE, stdout)
    out["stall_cycles"]  = _match_int(TOTAL_STALL_RE, stdout)
    out["memory_cycles"] = _match_int(TOTAL_MEM_CYC_RE, stdout)
    out["scalar_cycles"] = _match_int(TOTAL_SCALAR_RE, stdout)

    # Pool utilization (compute and occupancy fractions).
    out["mat_pool_util_pct"] = ""
    out["vec_pool_util_pct"] = ""
    out["mat_pool_occupancy_pct"] = ""
    out["vec_pool_occupancy_pct"] = ""
    for kind, util, occ in ACCEL_POOL_UTIL_RE.findall(stdout):
        if kind == "Matrix":
            out["mat_pool_util_pct"]      = float(util)
            out["mat_pool_occupancy_pct"] = float(occ)
        else:
            out["vec_pool_util_pct"]      = float(util)
            out["vec_pool_occupancy_pct"] = float(occ)

    # Per-backend cycles + reqs + derived cycle fractions.
    breakdown = _per_backend_breakdown(stdout)
    total = out["total_cycles"] if isinstance(out["total_cycles"], int) else 0
    for b in BACKENDS:
        bl = b.lower()
        out[f"{bl}_cycles"]   = breakdown[b]["cycles"]
        out[f"{bl}_mat_reqs"] = breakdown[b]["mat_reqs"]
        out[f"{bl}_vec_reqs"] = breakdown[b]["vec_reqs"]
        out[f"{bl}_mem_reqs"] = breakdown[b]["mem_reqs"]
        out[f"{bl}_cycle_fraction_pct"] = (
            (100.0 * breakdown[b]["cycles"] / total) if total > 0 else ""
        )
    return out


def run_point(binary: Path, sp: SweepPoint) -> tuple[dict[str, object], float]:
    cmd = [
        str(binary),
        "--block-c", str(sp.block_c),
        "--block-h", str(sp.block_h),
        "--block-w", str(sp.block_w),
        "--dma-base-lat", str(sp.dma_base_lat),
    ]
    start = time.monotonic()
    proc = _run(cmd)
    elapsed = time.monotonic() - start
    fields = _parse_run_output(proc.stdout)
    fields["run_ok"] = 1 if proc.returncode == 0 else 0
    if proc.returncode != 0:
        sys.stderr.write(
            f"[sweep] run failed for {sp.key()} rc={proc.returncode}\n"
            f"{proc.stdout}\n{proc.stderr}\n"
        )
    return fields, elapsed


def row_for(
    sp: SweepPoint, fields: dict[str, object], wall: float, build_ok: int
) -> dict[str, object]:
    row: dict[str, object] = {
        "tile_m": sp.hw.tile_m,
        "tile_k": sp.hw.tile_k,
        "tile_n": sp.hw.tile_n,
        "mat_latency": sp.hw.mat_latency,
        "mat_count": sp.hw.mat_count,
        "vec_count": sp.hw.vec_count,
        "vec_bytes": sp.hw.vec_bytes,
        "n_workers": sp.hw.n_workers,
        "block_c": sp.block_c,
        "block_h": sp.block_h,
        "block_w": sp.block_w,
        "dma_base_lat": sp.dma_base_lat,
        "wall_seconds": f"{wall:.3f}",
        "build_ok": build_ok,
        "run_ok": fields.get("run_ok", 0),
    }
    for k in CSV_FIELDS:
        if k in row:
            continue
        v = fields.get(k, "")
        # Normalize floats to 3 decimal places for CSV stability.
        if isinstance(v, float):
            row[k] = f"{v:.3f}"
        else:
            row[k] = v
    return row


def open_csv_writer(csv_path: Path, append: bool) -> tuple[csv.DictWriter, "object"]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and csv_path.exists() else "w"
    f = csv_path.open(mode, newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if mode == "w":
        writer.writeheader()
    return writer, f


# ------------------------------------------------------------
# Executor
# ------------------------------------------------------------
def process_hw_point(
    hw: HwPoint,
    sweep_points: list[SweepPoint],
    build_root: Path,
    keep_build_dir: bool,
) -> dict:
    """Build one HW point, run its sweep points, return rows + log."""
    rows: list[dict] = []
    failures = 0

    binary, build_log = build_hw_point(hw, build_root)
    build_ok = 1 if binary.exists() else 0

    if not build_ok:
        for sp in sweep_points:
            rows.append(row_for(sp, {}, 0.0, 0))
            failures += 1
    else:
        for sp in sweep_points:
            fields, wall = run_point(binary, sp)
            rows.append(row_for(sp, fields, wall, 1))
            if fields.get("verification_status") != "PASS":
                failures += 1
        if not keep_build_dir:
            shutil.rmtree(build_root / hw.tag, ignore_errors=True)

    return {
        "tag": hw.tag,
        "rows": rows,
        "log": build_log,
        "failures": failures,
        "build_ok": build_ok,
    }


def execute_parallel(
    sweep_points: list[SweepPoint],
    existing: set[tuple],
    csv_path: Path,
    append: bool,
    build_root: Path,
    keep_build_dirs: bool,
    jobs: int,
) -> int:
    remaining_by_hw: dict[HwPoint, list[SweepPoint]] = {}
    for sp in sweep_points:
        if sp.key() in existing:
            continue
        remaining_by_hw.setdefault(sp.hw, []).append(sp)

    if not remaining_by_hw:
        print("[sweep] nothing to do -- all points already in CSV")
        return 0

    total = len(remaining_by_hw)
    total_runs = sum(len(v) for v in remaining_by_hw.values())
    print(
        f"[sweep] {total} hardware points, {total_runs} sweep runs, "
        f"jobs={jobs}",
        flush=True,
    )

    writer, fh = open_csv_writer(csv_path, append)
    failures = 0
    completed = 0
    interrupted = False

    try:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = {
                ex.submit(
                    process_hw_point, hw, sps, build_root, keep_build_dirs
                ): hw
                for hw, sps in remaining_by_hw.items()
            }
            try:
                for fut in as_completed(futures):
                    result = fut.result()
                    for row in result["rows"]:
                        writer.writerow(row)
                    fh.flush()
                    failures += result["failures"]
                    completed += 1
                    if not result["build_ok"]:
                        sys.stderr.write(result["log"] + "\n")
                    print(
                        f"[sweep] {completed}/{total} hw={result['tag']} "
                        f"runs={len(result['rows'])} fail={result['failures']}",
                        flush=True,
                    )
            except KeyboardInterrupt:
                interrupted = True
                print(
                    "[sweep] interrupted -- shutting down workers; "
                    "in-flight sims will finish before exit.",
                    file=sys.stderr,
                    flush=True,
                )
                ex.shutdown(cancel_futures=True)
    finally:
        fh.close()

    if interrupted:
        return 130
    return 0 if failures == 0 else 2


def dry_run(
    hw_points: list[HwPoint],
    sweep_points: list[SweepPoint],
    existing: set[tuple],
) -> None:
    print(f"[dry-run] hardware points: {len(hw_points)}")
    print(f"[dry-run] total sweep points: {len(sweep_points)}")
    skip = sum(1 for sp in sweep_points if sp.key() in existing)
    print(f"[dry-run] skipped (already in CSV): {skip}")
    print(f"[dry-run] to run: {len(sweep_points) - skip}")
    shown = 0
    for hw in hw_points:
        print(
            f"  build: make -C nafblock BUILDDIR=<...> "
            f"EXTRA_CXXFLAGS='{hw.extra_cxxflags()}'"
        )
        for sp in sweep_points:
            if sp.hw is not hw:
                continue
            print(
                f"    run: nafblock_perf_sim --block-c {sp.block_c} "
                f"--block-h {sp.block_h} --block-w {sp.block_w} "
                f"--dma-base-lat {sp.dma_base_lat}"
            )
            shown += 1
            if shown >= 6:
                break
        if shown >= 6:
            break


def main() -> int:
    args = build_parser().parse_args()
    csv_path = Path(args.output)
    build_root = Path(args.build_root)

    hw_points = iter_hw_points(args)
    sweep_points = iter_sweep_points(args, hw_points)

    existing = set() if args.no_resume else load_existing_keys(csv_path)
    if args.no_resume and csv_path.exists():
        csv_path.unlink()

    if args.dry_run:
        dry_run(hw_points, sweep_points, existing)
        return 0

    if args.jobs < 1:
        sys.exit("--jobs must be >= 1")

    append = bool(existing)
    return execute_parallel(
        sweep_points,
        existing,
        csv_path,
        append=append,
        build_root=build_root,
        keep_build_dirs=args.keep_build_dirs,
        jobs=args.jobs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
