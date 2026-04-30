# Parametric Sweep — How to Run

Practical guide for driving the 7-dimensional matmul sweep defined in
[Parametric_Sweep.md](Parametric_Sweep.md).

## Scripts

| File | Purpose |
| --- | --- |
| [full_sweep.py](full_sweep.py)   | Runs the sweep: compiles one simulator binary per hardware point, executes each `(gemm_shape, threads)` run, streams rows into a CSV. |
| [plot_sweep.py](plot_sweep.py)   | Reads the CSV, filters and groups rows, renders a PNG. |
| [requirements.txt](requirements.txt) | Python deps (only `matplotlib`). |

## Parameter dimensions

| Flag | Default | Dimension |
| --- | --- | --- |
| `--tile-sizes`    | `8x8x8,16x32x64,16x16x16,32x64x128` | compile-time (`MATMUL_M/K/N`) |
| `--mat-latencies` | `2,4,8,16,32,64` | compile-time (`MATMUL_ACC_CYCLE`) |
| `--mat-counts`    | `1,2,4,8` | compile-time (`MAT_ACCEL_COUNT`) |
| `--vec-counts`    | `1,2,4,8` | compile-time (`VEC_ACCEL_COUNT`) |
| `--vec-bytes`     | `16,32,64,128,256` | compile-time (`VECTOR_ACC_CAP`) |
| `--gemm-shapes`   | `128x128x128, 1024x1024x1024, 1024x128x1024, 128x1024x128, 4096x4096x4096, 4096x1024x4096, 1024x4096x1024` | runtime CLI (`--gemm-m/k/n`) |
| `--threads-list`  | `1,2,4,8,16,32,64` | runtime CLI (`--threads`) |

Full default grid = 4 × 6 × 4 × 4 × 5 × 7 × 7 = **94 080 points**, requiring
**1 920 rebuilds** (one per unique combination of the 5 compile-time dims).

## One-time setup

```bash
python3 -m venv kernel/matmul/.venv
source kernel/matmul/.venv/bin/activate
pip install -r kernel/matmul/requirements.txt
```

All commands below assume the venv is active. The simulator itself is built
automatically by `full_sweep.py` via `make -C kernel matmul` with
`EXTRA_CXXFLAGS` — you do not need to pre-build it.

## Running the full sweep

```bash
python kernel/matmul/full_sweep.py
```

- Rows stream into `kernel/matmul/full_sweep.csv` (one row per point).
- Per-point binaries are stored under `kernel/matmul/.sweep_bin/<tag>/` and
  deleted after that hardware point finishes. Pass `--keep-build-dirs` to
  retain them.
- Exit code `0` means every run verified `PASS`; exit code `2` means at least
  one point failed verification or its build failed (the CSV still contains
  those rows for inspection).

### Previewing the grid before running

```bash
python kernel/matmul/full_sweep.py --dry-run
```

Prints the total point count, how many would be skipped by resume, and a few
sample build/run commands.

### Resuming

The CSV is keyed by the 11 parameter columns. Re-running with the same flags
skips any row already present; only missing rows are computed.

```bash
python kernel/matmul/full_sweep.py        # first run, partial, crashes
python kernel/matmul/full_sweep.py        # picks up where it left off
python kernel/matmul/full_sweep.py --no-resume   # force full overwrite
```

## Running a subset (any dimension)

Every dimension accepts a comma-separated list. Anything omitted falls back
to the full default list for that dimension.

### Single dimension — e.g. sweep only threads at one hardware point

```bash
python kernel/matmul/full_sweep.py \
    --tile-sizes 16x32x64 \
    --mat-latencies 4 \
    --mat-counts 4 \
    --vec-counts 4 \
    --vec-bytes 64 \
    --gemm-shapes 1024x1024x1024
# threads-list defaults to 1,2,4,8,16,32,64
```

### Two dimensions — e.g. vary matrix count × vector count at fixed everything else

```bash
python kernel/matmul/full_sweep.py \
    --tile-sizes 8x8x8 \
    --mat-latencies 4 \
    --mat-counts 1,2,4,8 \
    --vec-counts 1,2,4,8 \
    --vec-bytes 64 \
    --gemm-shapes 1024x1024x1024 \
    --threads-list 8
```

### Tile sweep at one workload / thread count

```bash
python kernel/matmul/full_sweep.py \
    --tile-sizes 8x8x8,16x16x16,16x32x64,32x64x128 \
    --mat-latencies 4 \
    --mat-counts 4 \
    --vec-counts 4 \
    --vec-bytes 64 \
    --gemm-shapes 1024x1024x1024 \
    --threads-list 16
```

### Writing to a separate CSV

Good practice when a sub-sweep should not pollute the main dataset:

```bash
python kernel/matmul/full_sweep.py \
    --tile-sizes 8x8x8 --mat-latencies 4 --mat-counts 4 \
    --vec-counts 4 --vec-bytes 64 \
    --gemm-shapes 1024x1024x1024 \
    --threads-list 1,2,4,8,16,32,64 \
    --output kernel/matmul/tile8_vs_threads.csv
```

## CSV schema

`full_sweep.csv` is long/tidy — one row per sweep point, parameters and
metrics side by side:

```
tile_m, tile_k, tile_n,
mat_latency, mat_count, vec_count, vec_bytes,
gemm_m, gemm_k, gemm_n, threads,
total_cycles, verification_status,
actual_mat_accels, actual_vec_accels,
wall_seconds, build_ok, run_ok
```

- The 11 leading columns are the **key**; `(key, metric)` pairs are what
  `plot_sweep.py` projects against.
- `verification_status` is `PASS` when the simulator's golden check agrees
  with request counts.
- `build_ok` / `run_ok` are `0`/`1`; filter on `build_ok=1,run_ok=1` when
  plotting to drop failed points.

## Plotting

```bash
python kernel/matmul/plot_sweep.py \
    --input kernel/matmul/full_sweep.csv \
    --x <column> --y <metric> \
    --group-by <col1,col2,...> \
    --filter <key=val[|val],...> \
    [--log-x] [--log-y] \
    [--3d --z <metric> --3d-style {scatter,surface,bar} [--log-z]] \
    [--style {hierarchical,flat}] \
    --output <path.png>
```

### Filter syntax

- `key=val` — keep rows where `key` equals `val`.
- `key=v1|v2|v3` — keep rows where `key` is in `{v1, v2, v3}`.
- Multiple keys are comma-separated: `--filter tile_m=16,gemm_m=1024,mat_count=1|2|4`.

### Group-by — hierarchical visual encoding (default)

In the default `--style hierarchical` mode, the columns in `--group-by` are
mapped to visual channels in order:

| Position | Visual channel | Effect |
| --- | --- | --- |
| 1st column | **Hue** | distinct base color per value |
| 2nd column | **Brightness** | light → dark within that hue |
| 3rd column | **Line style** | solid, dashed, dash-dot, dotted |
| 4th+ column | **Marker shape** | o, s, ^, D, … |

So `--group-by gemm_m,mat_count` puts every `gemm_m=1024` curve in shades of
one color (e.g. blue), with brightness varying by `mat_count`. Pass
`--style flat` to fall back to "one arbitrary color per series".

### Auto-annotated fixed dimensions

After filtering, any sweep parameter that is not on an axis or in
`--group-by` but has a single unique value across the matched rows is
listed below the plot, e.g.

```
Fixed: tile=16x32x64  ·  gemm=128x128x128  ·  vec_count=1  ·  vec_bytes=64
```

Pass `--no-fixed-annotation` to suppress.

### 3D mode

Pass `--3d` and supply a third axis via `--z`. In 3D mode, `--x` and `--y`
are the two horizontal axes and `--z` is the metric (default
`total_cycles`). Three rendering styles:

- `--3d-style scatter` — markers in 3D, colored by `--group-by` (default).
- `--3d-style surface` — `plot_trisurf` viridis surface with colorbar.
- `--3d-style bar` — `bar3d` for coarse / categorical x and y.

`--log-z` log-scales the vertical axis; `--log-x` / `--log-y` apply to the
horizontal axes (note that 3D log scales in matplotlib can be fragile —
prefer linear x / y when the values span a small range).

### Example plots

**Hierarchical 2D — threads vs cycles, hue=`mat_latency`, brightness=`mat_count`**

```bash
python kernel/matmul/plot_sweep.py \
    --input kernel/matmul/demo_sweep.csv \
    --x threads --y total_cycles \
    --group-by mat_latency,mat_count \
    --filter tile_m=8,tile_k=8,tile_n=8 \
    --log-x --log-y \
    --output kernel/matmul/demo_plots/demo_2d_hierarchical.png
```

Each `mat_latency` value gets one hue family; brightness varies with
`mat_count` within each family. Sample output:
[demo_plots/demo_2d_hierarchical.png](demo_plots/demo_2d_hierarchical.png).

**Flat 2D — single grouping column**

```bash
python kernel/matmul/plot_sweep.py \
    --input kernel/matmul/demo_sweep.csv \
    --x threads --y total_cycles \
    --group-by mat_count \
    --filter tile_m=8,mat_latency=4 \
    --log-x --log-y \
    --output kernel/matmul/demo_plots/demo_2d_flat.png
```

One distinct color per `mat_count`. Sample:
[demo_plots/demo_2d_flat.png](demo_plots/demo_2d_flat.png).

**3D scatter — `threads × mat_count × total_cycles`, color by `mat_latency`**

```bash
python kernel/matmul/plot_sweep.py \
    --input kernel/matmul/demo_sweep.csv \
    --x threads --y mat_count --z total_cycles \
    --3d --3d-style scatter \
    --group-by mat_latency \
    --filter tile_m=16,tile_k=32,tile_n=64 \
    --log-z \
    --output kernel/matmul/demo_plots/demo_3d_scatter.png
```

Sample: [demo_plots/demo_3d_scatter.png](demo_plots/demo_3d_scatter.png).

**3D surface — `plot_trisurf` viridis with colorbar**

```bash
python kernel/matmul/plot_sweep.py \
    --input kernel/matmul/demo_sweep.csv \
    --x threads --y mat_count --z total_cycles \
    --3d --3d-style surface \
    --filter tile_m=16,tile_k=32,tile_n=64,mat_latency=4 \
    --log-z \
    --output kernel/matmul/demo_plots/demo_3d_surface.png
```

Sample: [demo_plots/demo_3d_surface.png](demo_plots/demo_3d_surface.png).

### More 2D examples

**Tile size comparison at fixed 1024³ GEMM and 16 threads**

```bash
python kernel/matmul/plot_sweep.py \
    --input kernel/matmul/full_sweep.csv \
    --x tile_m --y total_cycles \
    --group-by tile_k,tile_n \
    --filter threads=16,gemm_m=1024,gemm_k=1024,gemm_n=1024,mat_latency=4,mat_count=4,vec_count=4,vec_bytes=64 \
    --output plots/tile_sweep.png
```

**Compare GEMM shapes at fixed hardware, thread sweep (hue=`gemm_m`, brightness=`gemm_n`)**

```bash
python kernel/matmul/plot_sweep.py \
    --input kernel/matmul/full_sweep.csv \
    --x threads --y total_cycles \
    --group-by gemm_m,gemm_n \
    --filter tile_m=16,tile_k=32,tile_n=64,mat_latency=4,mat_count=4,vec_count=4,vec_bytes=64 \
    --log-x --log-y \
    --output plots/gemm_shapes_vs_threads.png
```

### Drop failed rows

```bash
python kernel/matmul/plot_sweep.py ... --require-pass
```

Equivalent to adding `verification_status=PASS` to `--filter`.

## Smoke test (end-to-end sanity)

4 points, ~1 s total — use this to check that the interface still works after
touching any sweep code:

```bash
rm -f kernel/matmul/full_sweep.csv kernel/matmul/.sweep_bin

python kernel/matmul/full_sweep.py \
    --tile-sizes 8x8x8 \
    --mat-latencies 2 \
    --mat-counts 1,2 \
    --vec-counts 1 \
    --vec-bytes 64 \
    --gemm-shapes 128x128x128 \
    --threads-list 1,4

python kernel/matmul/plot_sweep.py \
    --input kernel/matmul/full_sweep.csv \
    --x threads --y total_cycles \
    --group-by mat_count \
    --filter tile_m=8,gemm_m=128 \
    --output /tmp/smoke.png --log-x
```

Expect: 4 CSV rows all `PASS`, PNG rendered to `/tmp/smoke.png`, a second
invocation of `full_sweep.py` prints `nothing to do`.

## Notes

- `--jobs N` runs the sweep in parallel using `N` worker processes. Each
  worker builds and runs **one hardware point at a time** in its own private
  `BUILDDIR` under `kernel/matmul/.sweep_bin/<tag>/`, so workers never
  contend on shared build state. Recommended for a server:

  ```bash
  python kernel/matmul/full_sweep.py --jobs $(nproc)
  ```

  CSV rows arrive in completion order (not input order) — `plot_sweep.py`
  already filters/groups by columns, so order is not load-bearing.
- Disk usage: peak ≈ `--jobs` × ~5 MB (~150 MB at `--jobs 30`). Per-point
  build dirs are deleted as soon as that hardware point's runs finish.
- `Ctrl+C` shuts the pool down cleanly: in-flight sims finish, pending
  hardware points are cancelled, and the CSV is left consistent — re-run
  the same command to resume.
- The CSV is append-friendly — safe to run from multiple shells against
  disjoint `--output` paths simultaneously.
