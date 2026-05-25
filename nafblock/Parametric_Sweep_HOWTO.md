# NafBlock Parametric Sweep — How to Run

Practical guide for driving the 8-dimensional NafBlock sweep defined in
[FULL_Sweep.md](FULL_Sweep.md). Mirrors the matmul project's
[kernel/matmul/Parametric_Sweep_HOWTO.md](../kernel/matmul/Parametric_Sweep_HOWTO.md).

## Scripts

| File | Purpose |
| --- | --- |
| [full_sweep.py](full_sweep.py) | Compiles one simulator binary per hardware point, executes each `(block shape × dma_base_lat)` run, streams rows into a CSV. |
| [parametric_sweep.py](parametric_sweep.py) | Quick-look 2-D sweep (shape × MAT:VEC counts) — kept for first-pass exploration. The 8-D sweep below supersedes it for thorough studies. |
| [../kernel/matmul/plot_sweep.py](../kernel/matmul/plot_sweep.py) | Column-agnostic plotter for 2-D line / 3-D scatter / surface / 3-D bar. Reads the CSV, filters / groups rows, renders a PNG. |
| [plot_backend_breakdown.py](plot_backend_breakdown.py) | CLI stacked-bar per-backend cycle breakdown (LayerNorm / Matmul / DwConv / Pooling / VecOps). Counterpart to `plot_sweep.py --bar`, which only knows matmul's `mat/vec/dma/scalar/stall` segments. |
| [../Plotting_Tool/app.py](../Plotting_Tool/app.py) | Interactive Streamlit + Plotly explorer. Schema-autodetects the nafblock CSV: stacked-bar uses the 5 backend fractions, `block = "CxHxW"` composite is auto-added, and all nafblock metric columns are correctly classified as non-params. Run `streamlit run Plotting_Tool/app.py` then upload `nafblock/full_sweep.csv`. |
| [../kernel/matmul/requirements.txt](../kernel/matmul/requirements.txt) | Python deps for the CLI plotters (`matplotlib`). |
| [../Plotting_Tool/requirements.txt](../Plotting_Tool/requirements.txt) | Python deps for the interactive explorer (`streamlit`, `plotly`, `pandas`, `streamlit-sortables`). |

## Parameter dimensions

| Flag | Default | Dimension |
| --- | --- | --- |
| `--tile-sizes`    | `8x8x8,16x32x64,16x16x16,32x64x128` | compile-time (`MATMUL_M/K/N`) |
| `--mat-latencies` | `2,4,8,16,32,64` | compile-time (`MATMUL_ACC_CYCLE`) |
| `--mat-counts`    | `1,2,4,8` | compile-time (`MAT_ACCEL_COUNT`) |
| `--vec-counts`    | `1,2,4,8` | compile-time (`VEC_ACCEL_COUNT`) |
| `--vec-bytes`     | `16,32,64,128,256` | compile-time (`VECTOR_ACC_CAP`) |
| `--n-workers`     | `1,2,4,8,16,32,64` | compile-time (`NAFBLOCK_N_WORKERS`) |
| `--block-shapes`  | `32x512x512,64x256x256,128x128x128,256x64x64,512x32x32` | runtime CLI (`--block-c/h/w`) |
| `--dma-base-lats` | `4,8,16,32` | runtime CLI (`--dma-base-lat`), mapped to every kernel's `l2_base_lat` and matmul's `dma_base_lat`. |

Full default grid = 4 × 6 × 4 × 4 × 5 × 7 × 5 × 4 = **268 800 points**, requiring
**13 440 rebuilds** (one per unique combination of the 6 compile-time dims).

### Block shape source

[FULL_Sweep.md](FULL_Sweep.md) lists 9 entries that walk down then back up the
NAFNet U-Net (encoder → bottleneck → decoder). For a deterministic
single-block simulator the encoder pass and decoder pass produce identical
cycles for the same shape, so the default `--block-shapes` is **deduped to 5
unique shapes**. To estimate the full network's NafBlock budget, sum each
non-bottleneck row's `total_cycles` twice (encoder + decoder) and the
bottleneck row (`128x128x128`) once.

## One-time setup

```bash
python3 -m venv kernel/matmul/.venv
source kernel/matmul/.venv/bin/activate
pip install -r kernel/matmul/requirements.txt
```

The simulator itself is built automatically by `full_sweep.py` via
`make -C nafblock` with `EXTRA_CXXFLAGS` — you do not need to pre-build it.

## Running the full sweep

```bash
python nafblock/full_sweep.py
```

- Rows stream into `nafblock/full_sweep.csv` (one row per point).
- Per-point binaries are stored under `nafblock/.sweep_bin/<tag>/` and
  deleted after that hardware point finishes. Pass `--keep-build-dirs` to
  retain them.
- Exit code `0` means every run verified `PASS`; exit code `2` means at least
  one point failed verification or its build failed (the CSV still contains
  those rows for inspection).

### Previewing the grid before running

```bash
python nafblock/full_sweep.py --dry-run
```

Prints the total point count, how many would be skipped by resume, and a few
sample build/run commands.

### Resuming

The CSV is keyed by the 12 parameter columns. Re-running with the same flags
skips any row already present; only missing rows are computed.

```bash
python nafblock/full_sweep.py        # first run, partial, crashes
python nafblock/full_sweep.py        # picks up where it left off
python nafblock/full_sweep.py --no-resume   # force full overwrite
```

## Running a subset (any dimension)

Every dimension accepts a comma-separated list. Anything omitted falls back
to the full default list for that dimension.

### Single dimension — DMA latency sweep at one hardware/shape point

```bash
python nafblock/full_sweep.py \
    --tile-sizes 16x32x64 --mat-latencies 8 --mat-counts 2 \
    --vec-counts 4 --vec-bytes 64 --n-workers 4 \
    --block-shapes 128x128x128 \
    --dma-base-lats 4,8,16,32
```

### Sweep block shape at fixed hardware

```bash
python nafblock/full_sweep.py \
    --tile-sizes 16x32x64 --mat-latencies 8 --mat-counts 2 \
    --vec-counts 4 --vec-bytes 64 --n-workers 4 \
    --dma-base-lats 8
# block-shapes defaults to the 5 NAFNet32 levels
```

### Two dimensions — vary `vec_count × vec_bytes` at fixed everything else

```bash
python nafblock/full_sweep.py \
    --tile-sizes 16x32x64 --mat-latencies 8 --mat-counts 2 \
    --vec-counts 1,2,4,8 --vec-bytes 16,32,64,128,256 --n-workers 4 \
    --block-shapes 128x128x128 \
    --dma-base-lats 8
```

### Worker-count sweep

```bash
python nafblock/full_sweep.py \
    --tile-sizes 16x32x64 --mat-latencies 8 --mat-counts 2 \
    --vec-counts 4 --vec-bytes 64 \
    --n-workers 1,2,4,8,16,32,64 \
    --block-shapes 128x128x128 \
    --dma-base-lats 8
```

### Writing to a separate CSV

Good practice when a sub-sweep should not pollute the main dataset:

```bash
python nafblock/full_sweep.py \
    --tile-sizes 8x8x8 --mat-latencies 4 --mat-counts 1 \
    --vec-counts 1 --vec-bytes 64 --n-workers 4 \
    --block-shapes 128x128x128 \
    --dma-base-lats 4,8,16,32 \
    --output nafblock/dma_lat_sweep.csv
```

## CSV schema

`full_sweep.csv` is long/tidy — one row per sweep point, parameters and
metrics side by side. The 12 leading columns are the **key**; everything else
is a metric.

### Key columns

| Column | Source |
| --- | --- |
| `tile_m`, `tile_k`, `tile_n` | `MATMUL_M/K/N` build flag |
| `mat_latency` | `MATMUL_ACC_CYCLE` |
| `mat_count`, `vec_count` | `MAT_ACCEL_COUNT`, `VEC_ACCEL_COUNT` |
| `vec_bytes` | `VECTOR_ACC_CAP` (bytes/call) |
| `n_workers` | `NAFBLOCK_N_WORKERS` |
| `block_c`, `block_h`, `block_w` | `--block-c/h/w` |
| `dma_base_lat` | `--dma-base-lat` |

### Metric columns

| Column | Source / meaning |
| --- | --- |
| `total_cycles` | `Total Elapsed Cycles` line from the report. |
| `verification_status` | `PASS` only when every per-layer + manifest check agrees. |
| `actual_workers`, `actual_mat_accels`, `actual_vec_accels`, `actual_vec_cap` | Observed values from the `Hardware Configuration` block — sanity-check vs. the build flags above. |
| `mat_pool_util_pct`, `vec_pool_util_pct` | `Compute Utilization [%]` for the matrix / vector accelerator pools (pool-level rows in the Accelerator Summary). |
| `mat_pool_occupancy_pct`, `vec_pool_occupancy_pct` | `Occupancy Utilization [%]` for the same pools. |
| `mat_reqs`, `vec_reqs`, `mem_reqs` | `Total Matrix / Vector Accelerator Requests`, `Total Memory Requests` aggregates. |
| `read_bytes`, `write_bytes` | `Total Read Bytes`, `Total Write Bytes`. |
| `stall_cycles`, `memory_cycles`, `scalar_cycles` | `Total Stall / Memory / Scalar Cycles` aggregates. |
| `<backend>_cycles` for `<backend>` ∈ `{layernorm, matmul, dwconv, pooling, vecops}` | Sum of `Elapsed Cycles` over the per-layer rows of that backend. **Sum of the five equals `total_cycles`** because the block runs sub-layers sequentially. |
| `<backend>_cycle_fraction_pct` | `100 × <backend>_cycles / total_cycles`. The five fractions **sum to 100 by construction** — use them with `plot_sweep.py --bar`. |
| `<backend>_mat_reqs`, `<backend>_vec_reqs`, `<backend>_mem_reqs` | Per-backend request aggregates. |
| `wall_seconds` | Real time spent inside the `nafblock_perf_sim` invocation. |
| `build_ok`, `run_ok` | `1` / `0` indicators; filter on `build_ok=1,run_ok=1` when plotting. |

### Verification

- `verification_status == PASS` means every per-layer kernel's expected-vs-actual
  request counts agree **and** the manifest validator's 14-row
  `(op, backend, phase_count, primary_vop, secondary_vop)` check passes.
- The cycle-fraction columns sum to 100 because nafblock layers run
  sequentially; this is **not** the matmul critical-path-worker convention.
  Stall / memory / scalar cycles are reported as absolute counts for
  diagnostic use — they overlap with backend cycles and do not add to 100%.

## Plotting

The plotter under `kernel/matmul/plot_sweep.py` is column-agnostic — point
it at the nafblock CSV. Every column documented above is a valid `--x`,
`--y`, `--z`, `--group-by`, or `--filter` value.

```bash
python kernel/matmul/plot_sweep.py \
    --input nafblock/full_sweep.csv \
    --x <column> --y <metric> \
    --group-by <col1,col2,...> \
    --filter <key=val[|val],...> \
    [--log-x] [--log-y] \
    [--3d --z <metric> --3d-style {scatter,surface,bar} [--log-z]] \
    [--style {hierarchical,flat}] \
    --output <path.png>
```

### Example plots

**Block shape vs total cycles, one line per vec_count**

```bash
python kernel/matmul/plot_sweep.py \
    --input nafblock/full_sweep.csv \
    --x block_c --y total_cycles \
    --group-by vec_count \
    --filter tile_m=16,tile_k=32,tile_n=64,mat_latency=8,mat_count=2,vec_bytes=64,n_workers=4,dma_base_lat=8 \
    --log-y \
    --output /tmp/nb_shape_vs_cycles.png
```

**Stacked bar — per-backend cycle breakdown by block shape**

Use the dedicated helper [plot_backend_breakdown.py](plot_backend_breakdown.py).
`kernel/matmul/plot_sweep.py --bar` hardcodes its 5 segments to matmul's
`mat/vec/dma/scalar/stall` columns; the nafblock CSV instead reports per-backend
fractions (LayerNorm / Matmul / DwConv / Pooling / VecOps), which is what this
script stacks:

```bash
python nafblock/plot_backend_breakdown.py \
    --input nafblock/full_sweep.csv \
    --x block_c \
    --filter tile_m=16,tile_k=32,tile_n=64,mat_latency=8,mat_count=2,vec_count=4,vec_bytes=64,n_workers=4,dma_base_lat=8 \
    --output /tmp/nb_backend_breakdown.png
```

Bar height = `total_cycles`; segment height = `total_cycles × fraction_pct / 100`,
one segment per `{layernorm, matmul, dwconv, pooling, vecops}` backend. Filter
syntax matches `plot_sweep.py` (`key=val` or `key=v1|v2|v3`, comma-joined).
Pass `--require-pass` to drop failed rows, `--log-y` for log scale.

**3D surface — vec_count × n_workers × total_cycles**

```bash
python kernel/matmul/plot_sweep.py \
    --input nafblock/full_sweep.csv \
    --x vec_count --y n_workers --z total_cycles \
    --3d --3d-style surface \
    --filter tile_m=16,tile_k=32,tile_n=64,mat_latency=8,mat_count=2,vec_bytes=64,block_c=128,block_h=128,block_w=128,dma_base_lat=8 \
    --log-z \
    --output /tmp/nb_3d_surface.png
```

### Drop failed rows

```bash
python kernel/matmul/plot_sweep.py ... --require-pass
```

Equivalent to adding `verification_status=PASS` to `--filter`.

## Whole-network estimation

The 9 spec entries in [FULL_Sweep.md](FULL_Sweep.md) walk down then back up
the U-Net (encoder → bottleneck → decoder); after dedup the CSV has 5 unique
rows per hardware point. To estimate the entire NAFNet's NafBlock budget at a
fixed hardware point, multiply each row's `total_cycles` by its multiplicity
in the encoder/decoder pattern:

| Block shape | Multiplicity (encoder + decoder) |
| --- | --- |
| `32×512×512` | 2 |
| `64×256×256` | 2 |
| `128×128×128` | 1 (bottleneck — appears once) |
| `256×64×64` | 2 |
| `512×32×32` | 2 |

`whole_net_cycles ≈ 2·(c_{32,512,512} + c_{64,256,256} + c_{256,64,64} + c_{512,32,32}) + c_{128,128,128}`

This is per-block; the actual NAFNet repeats each level `M_i` times — multiply
each term by the per-level block count if you want the full inference
estimate. The single-block sim under [nafblock/](.) does not multiply blocks
itself.

## Smoke test (end-to-end sanity)

4 points, ~30 s total — use this to check that the interface still works
after touching any sweep code:

```bash
rm -f nafblock/full_sweep.csv && rm -rf nafblock/.sweep_bin

python nafblock/full_sweep.py \
    --tile-sizes 16x32x64 \
    --mat-latencies 8 \
    --mat-counts 1 \
    --vec-counts 1,2 \
    --vec-bytes 64 \
    --n-workers 4 \
    --block-shapes 32x64x64,64x32x32 \
    --dma-base-lats 10 \
    --jobs 2

python kernel/matmul/plot_sweep.py \
    --input nafblock/full_sweep.csv \
    --x block_c --y total_cycles \
    --group-by vec_count \
    --filter tile_m=16 --log-y \
    --output /tmp/nb_smoke.png

python nafblock/plot_backend_breakdown.py \
    --input nafblock/full_sweep.csv \
    --x block_c \
    --filter tile_m=16,vec_count=2,dma_base_lat=10 \
    --output /tmp/nb_smoke_bar.png
```

Expect: 4 CSV rows all `PASS`, every row's five `*_cycle_fraction_pct` columns
summing to 100, PNGs rendered to `/tmp/nb_smoke.png` (2D — one line per
`vec_count`) and `/tmp/nb_smoke_bar.png` (stacked bar with 2 bars, each split
into 5 backend categories), and a second invocation of `full_sweep.py`
printing `nothing to do`.

## Notes

- `--jobs N` runs the sweep in parallel using `N` worker processes. Each
  worker builds and runs **one hardware point at a time** in its own private
  `BUILDDIR` under `nafblock/.sweep_bin/<tag>/`, so workers never contend on
  shared build state. Recommended for a server:

  ```bash
  python nafblock/full_sweep.py --jobs $(nproc)
  ```

  CSV rows arrive in completion order (not input order) — `plot_sweep.py`
  filters / groups by columns, so order is not load-bearing.
- Disk usage: peak ≈ `--jobs` × ~5 MB. Per-point build dirs are deleted as
  soon as that hardware point's runs finish.
- `Ctrl+C` shuts the pool down cleanly: in-flight sims finish, pending
  hardware points are cancelled, and the CSV is left consistent — re-run
  the same command to resume.
- The CSV is append-friendly — safe to run from multiple shells against
  disjoint `--output` paths simultaneously.
- The legacy `nafnet/` simulator is unrelated; this sweep targets only the
  standalone nafblock under `nafblock/`.
