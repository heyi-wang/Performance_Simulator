#!/usr/bin/env python3
"""Full 7-dimensional parametric sweep driver for the matmul simulator.

Dimensions (see kernel/matmul/Parametric_Sweep.md):
    - tile size       : MATMUL_M x MATMUL_K x MATMUL_N       (compile-time)
    - matrix latency  : MATMUL_ACC_CYCLE                     (compile-time)
    - matrix count    : MAT_ACCEL_COUNT                      (compile-time)
    - vector count    : VEC_ACCEL_COUNT                      (compile-time)
    - vector bytes    : VECTOR_ACC_CAP                       (compile-time)
    - gemm shape      : --gemm-m/--gemm-k/--gemm-n           (runtime CLI)
    - threads         : --threads                            (runtime CLI)

The 5 compile-time parameters define a "hardware point". Each unique hardware
point is built once into its own per-point binary under kernel/build/sweep/.
Each binary is then re-invoked across the (shape x threads) grid.

Results are stored one row per sweep point in a long/tidy CSV so any
parameter combination can be filtered/plotted later via plot_sweep.py.

Only the interface is intended for extensive use on a server -- this script
also supports --dry-run and filter flags so you can validate it locally
with a tiny subset.
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
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNEL_DIR = REPO_ROOT / "kernel"
DEFAULT_CSV = REPO_ROOT / "kernel" / "matmul" / "full_sweep.csv"
SWEEP_BUILD_ROOT = REPO_ROOT / "kernel" / "matmul" / ".sweep_bin"

TOTAL_ELAPSED_RE = re.compile(
    r"^(?:Total elapsed\s*:\s*(\d+)\s+cycles|Total Elapsed Cycles\s*\[cycles\]\s*:\s*(\d+))$",
    re.MULTILINE,
)
VERIFICATION_RE = re.compile(
    r"^Verification Status\s*:\s*(PASS|FAIL)$",
    re.MULTILINE,
)
ACCEL_COUNT_RE = re.compile(
    r"^(Matrix|Vector) Accelerators\s*\[count\]\s*:\s*(\d+)$",
    re.MULTILINE,
)

# Defaults from kernel/matmul/Parametric_Sweep.md.
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
DEFAULT_GEMM_SHAPES: list[tuple[int, int, int]] = [
    (128, 128, 128),
    (1024, 1024, 1024),
    (1024, 128, 1024),
    (128, 1024, 128),
    (4096, 4096, 4096),
    (4096, 1024, 4096),
    (1024, 4096, 1024),
]
DEFAULT_THREADS: list[int] = [1, 2, 4, 8, 16, 32, 64]

CSV_FIELDS = [
    "tile_m", "tile_k", "tile_n",
    "mat_latency", "mat_count", "vec_count", "vec_bytes",
    "gemm_m", "gemm_k", "gemm_n", "threads",
    "total_cycles", "verification_status",
    "actual_mat_accels", "actual_vec_accels",
    "wall_seconds", "build_ok", "run_ok",
]
KEY_FIELDS = [
    "tile_m", "tile_k", "tile_n",
    "mat_latency", "mat_count", "vec_count", "vec_bytes",
    "gemm_m", "gemm_k", "gemm_n", "threads",
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

    @property
    def tag(self) -> str:
        raw = (
            f"m{self.tile_m}_k{self.tile_k}_n{self.tile_n}"
            f"_lat{self.mat_latency}_mc{self.mat_count}_vc{self.vec_count}"
            f"_vb{self.vec_bytes}"
        )
        # Short hash keeps build-dir names bounded even for many points.
        digest = hashlib.sha1(raw.encode()).hexdigest()[:8]
        return f"{raw}__{digest}"

    def extra_cxxflags(self) -> str:
        return (
            f"-DMATMUL_M={self.tile_m} -DMATMUL_K={self.tile_k} "
            f"-DMATMUL_N={self.tile_n} -DMATMUL_ACC_CYCLE={self.mat_latency} "
            f"-DMAT_ACCEL_COUNT={self.mat_count} "
            f"-DVEC_ACCEL_COUNT={self.vec_count} "
            f"-DVECTOR_ACC_CAP={self.vec_bytes}"
        )


@dataclass(frozen=True)
class SweepPoint:
    hw: HwPoint
    gemm_m: int
    gemm_k: int
    gemm_n: int
    threads: int

    def key(self) -> tuple:
        return (
            self.hw.tile_m, self.hw.tile_k, self.hw.tile_n,
            self.hw.mat_latency, self.hw.mat_count,
            self.hw.vec_count, self.hw.vec_bytes,
            self.gemm_m, self.gemm_k, self.gemm_n, self.threads,
        )


def _parse_int_list(value: str, name: str) -> list[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(f"{name} must have at least one value")
    out: list[int] = []
    for p in parts:
        try:
            v = int(p)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name}: '{p}' is not an integer") from exc
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
        pieces = p.replace("*", "x").split("x")
        if len(pieces) != 3:
            raise argparse.ArgumentTypeError(
                f"{name}: '{p}' must be MxKxN (e.g. 16x32x64)"
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
            "Run the full 7-D matmul parametric sweep. Defaults match the "
            "parameter set in kernel/matmul/Parametric_Sweep.md; filter flags "
            "let you restrict to any sub-grid."
        )
    )
    p.add_argument(
        "--tile-sizes",
        type=lambda v: _parse_triple_list(v, "--tile-sizes"),
        default=DEFAULT_TILES,
        help=(
            "Comma-separated list of tile triples MxKxN "
            "(e.g. 8x8x8,16x32x64). Default: full spec list."
        ),
    )
    p.add_argument(
        "--mat-latencies",
        type=lambda v: _parse_int_list(v, "--mat-latencies"),
        default=DEFAULT_MAT_LATENCIES,
        help="Comma-separated matrix latencies (cycles). Default: 2,4,8,16,32,64.",
    )
    p.add_argument(
        "--mat-counts",
        type=lambda v: _parse_int_list(v, "--mat-counts"),
        default=DEFAULT_MAT_COUNTS,
        help="Comma-separated matrix unit counts. Default: 1,2,4,8.",
    )
    p.add_argument(
        "--vec-counts",
        type=lambda v: _parse_int_list(v, "--vec-counts"),
        default=DEFAULT_VEC_COUNTS,
        help="Comma-separated vector unit counts. Default: 1,2,4,8.",
    )
    p.add_argument(
        "--vec-bytes",
        type=lambda v: _parse_int_list(v, "--vec-bytes"),
        default=DEFAULT_VEC_BYTES,
        help="Comma-separated vector datapath widths in bytes. Default: 16,32,64,128,256.",
    )
    p.add_argument(
        "--gemm-shapes",
        type=lambda v: _parse_triple_list(v, "--gemm-shapes"),
        default=DEFAULT_GEMM_SHAPES,
        help="Comma-separated GEMM shapes MxKxN. Default: full spec list.",
    )
    p.add_argument(
        "--threads-list",
        type=lambda v: _parse_int_list(v, "--threads-list"),
        default=DEFAULT_THREADS,
        help="Comma-separated thread counts. Default: 1,2,4,8,16,32,64.",
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
        help="Parallel worker count (1 = serial). Each job uses its own build dir.",
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


def iter_hw_points(args: argparse.Namespace) -> list[HwPoint]:
    points: list[HwPoint] = []
    for (tm, tk, tn), lat, mc, vc, vb in product(
        args.tile_sizes, args.mat_latencies,
        args.mat_counts, args.vec_counts, args.vec_bytes,
    ):
        points.append(HwPoint(tm, tk, tn, lat, mc, vc, vb))
    return points


def iter_sweep_points(args: argparse.Namespace, hw_points: list[HwPoint]) -> list[SweepPoint]:
    runtime_grid = list(product(args.gemm_shapes, args.threads_list))
    points: list[SweepPoint] = []
    for hw in hw_points:
        for (gm, gk, gn), th in runtime_grid:
            points.append(SweepPoint(hw, gm, gk, gn, th))
    return points


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


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)


def build_hw_point(hw: HwPoint, build_root: Path) -> tuple[Path, str]:
    """Compile the matmul sim for this hardware point. Returns (binary_path, log).

    Uses a per-point out-of-tree build by running make in a temp dir layout
    that mirrors kernel/. We invoke the existing kernel Makefile with
    EXTRA_CXXFLAGS and an overridden build dir so per-point builds don't
    collide when --jobs > 1.
    """
    point_dir = build_root / hw.tag
    binary = point_dir / "matmul_sim"

    # We let make handle its own build dir inside KERNEL_DIR but redirect BUILD_DIR.
    # The kernel Makefile uses a hardcoded "build" dir; easiest portable approach
    # is to copy the produced binary to point_dir after a serialized make.
    # For --jobs > 1 we need isolation: do the make in a per-point symlink tree.
    #
    # Simpler and robust: run `make -j1` in the shared kernel dir under a lock,
    # then mv build/matmul_sim into point_dir. For parallelism, use --jobs 1
    # build-side but parallelize at the run step (runs dominate wall time).
    #
    # See runner logic below: parallel execution serializes builds.
    extra = hw.extra_cxxflags()
    log_lines: list[str] = []

    clean = _run(["make", "-s", "clean"], cwd=KERNEL_DIR)
    log_lines.append(f"$ make clean\n{clean.stdout}{clean.stderr}")

    build = _run(
        ["make", "matmul", f"EXTRA_CXXFLAGS={extra}"],
        cwd=KERNEL_DIR,
    )
    log_lines.append(
        f"$ make matmul EXTRA_CXXFLAGS='{extra}'\n{build.stdout}{build.stderr}"
    )
    if build.returncode != 0:
        return binary, "\n".join(log_lines)

    produced = KERNEL_DIR / "build" / "matmul_sim"
    if not produced.exists():
        log_lines.append(f"ERROR: expected {produced} after build")
        return binary, "\n".join(log_lines)

    point_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, binary)
    os.chmod(binary, 0o755)
    return binary, "\n".join(log_lines)


def _parse_run_output(stdout: str) -> dict[str, object]:
    out: dict[str, object] = {
        "total_cycles": "",
        "verification_status": "",
        "actual_mat_accels": "",
        "actual_vec_accels": "",
    }
    m = TOTAL_ELAPSED_RE.search(stdout)
    if m:
        out["total_cycles"] = int(next(g for g in m.groups() if g is not None))
    v = VERIFICATION_RE.search(stdout)
    if v:
        out["verification_status"] = v.group(1)
    for kind, count in ACCEL_COUNT_RE.findall(stdout):
        if kind == "Matrix":
            out["actual_mat_accels"] = int(count)
        else:
            out["actual_vec_accels"] = int(count)
    return out


def run_point(binary: Path, sp: SweepPoint) -> tuple[dict[str, object], float]:
    cmd = [
        str(binary),
        "--threads", str(sp.threads),
        "--gemm-m", str(sp.gemm_m),
        "--gemm-k", str(sp.gemm_k),
        "--gemm-n", str(sp.gemm_n),
    ]
    start = time.monotonic()
    proc = _run(cmd)
    elapsed = time.monotonic() - start
    fields = _parse_run_output(proc.stdout)
    fields["run_ok"] = 1 if proc.returncode == 0 else 0
    if proc.returncode != 0:
        # Surface the sim output so a server run can see why a point failed.
        sys.stderr.write(
            f"[sweep] run failed for {sp.key()} rc={proc.returncode}\n"
            f"{proc.stdout}\n{proc.stderr}\n"
        )
    return fields, elapsed


def row_for(sp: SweepPoint, fields: dict[str, object], wall: float, build_ok: int) -> dict[str, object]:
    return {
        "tile_m": sp.hw.tile_m,
        "tile_k": sp.hw.tile_k,
        "tile_n": sp.hw.tile_n,
        "mat_latency": sp.hw.mat_latency,
        "mat_count": sp.hw.mat_count,
        "vec_count": sp.hw.vec_count,
        "vec_bytes": sp.hw.vec_bytes,
        "gemm_m": sp.gemm_m,
        "gemm_k": sp.gemm_k,
        "gemm_n": sp.gemm_n,
        "threads": sp.threads,
        "total_cycles": fields.get("total_cycles", ""),
        "verification_status": fields.get("verification_status", ""),
        "actual_mat_accels": fields.get("actual_mat_accels", ""),
        "actual_vec_accels": fields.get("actual_vec_accels", ""),
        "wall_seconds": f"{wall:.3f}",
        "build_ok": build_ok,
        "run_ok": fields.get("run_ok", 0),
    }


def open_csv_writer(csv_path: Path, append: bool) -> tuple[csv.DictWriter, "object"]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and csv_path.exists() else "w"
    f = csv_path.open(mode, newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if mode == "w":
        writer.writeheader()
    return writer, f


def execute_serial(
    hw_points: list[HwPoint],
    sweep_points: list[SweepPoint],
    existing: set[tuple],
    csv_path: Path,
    append: bool,
    build_root: Path,
    keep_build_dirs: bool,
) -> int:
    writer, fh = open_csv_writer(csv_path, append)
    try:
        remaining_by_hw: dict[HwPoint, list[SweepPoint]] = {}
        for sp in sweep_points:
            if sp.key() in existing:
                continue
            remaining_by_hw.setdefault(sp.hw, []).append(sp)

        if not remaining_by_hw:
            print("[sweep] nothing to do -- all points already in CSV")
            return 0

        failures = 0
        for i, hw in enumerate(hw_points):
            sps = remaining_by_hw.get(hw)
            if not sps:
                continue
            print(
                f"[sweep] hw {i+1}/{len(hw_points)} {hw.tag} "
                f"({len(sps)} runs)",
                flush=True,
            )
            binary, build_log = build_hw_point(hw, build_root)
            build_ok = 1 if binary.exists() else 0
            if not build_ok:
                sys.stderr.write(build_log + "\n")
                for sp in sps:
                    row = row_for(sp, {}, 0.0, 0)
                    writer.writerow(row)
                    fh.flush()
                    failures += 1
                continue
            for sp in sps:
                fields, wall = run_point(binary, sp)
                row = row_for(sp, fields, wall, 1)
                writer.writerow(row)
                fh.flush()
                status = fields.get("verification_status", "?")
                cycles = fields.get("total_cycles", "?")
                print(
                    f"  [run] {hw.tag} gemm={sp.gemm_m}x{sp.gemm_k}x{sp.gemm_n} "
                    f"threads={sp.threads} cycles={cycles} verify={status} "
                    f"wall={wall:.2f}s",
                    flush=True,
                )
                if status != "PASS":
                    failures += 1
            if not keep_build_dirs:
                shutil.rmtree(build_root / hw.tag, ignore_errors=True)
        return 0 if failures == 0 else 2
    finally:
        fh.close()


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
    # Show a handful of commands as illustration.
    shown = 0
    for hw in hw_points:
        print(f"  build: make -C kernel matmul EXTRA_CXXFLAGS='{hw.extra_cxxflags()}'")
        for sp in sweep_points:
            if sp.hw is not hw:
                continue
            print(
                f"    run: matmul_sim --threads {sp.threads} "
                f"--gemm-m {sp.gemm_m} --gemm-k {sp.gemm_k} --gemm-n {sp.gemm_n}"
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

    if args.jobs > 1:
        # Parallel builds would need per-process working trees; keep it simple:
        # run the build stage serially, but the biggest wall-time cost per
        # hardware point is usually the run stage, which we could parallelize
        # per binary. For now we leave --jobs as a placeholder that forces
        # serial execution with a warning, since correctness > parallelism for
        # this initial interface.
        print(
            "[sweep] warning: --jobs > 1 not yet implemented safely; "
            "falling back to serial.",
            file=sys.stderr,
        )

    append = bool(existing)
    return execute_serial(
        hw_points,
        sweep_points,
        existing,
        csv_path,
        append=append,
        build_root=build_root,
        keep_build_dirs=args.keep_build_dirs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
