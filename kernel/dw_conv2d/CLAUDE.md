# Dw_Conv2d simulator — calibrated state (v2 + v3)

## Scope
SystemC TLM-2.0 depth-wise 2D convolution sim under [kernel/dw_conv2d/](.).
Models a per-channel-parallel `Kh × Kw` depth-wise conv on `VEC_ACCEL_COUNT`
pinned vector accelerators with strip-mined output rows, L2 DMA prefetch, an
L1 writeback checkpoint embedded in the last sub-request, and a
fire-and-forget L2 DMA writeback per strip.

Shared building blocks (`Worker` / `Interconnect` / `AcceleratorPool` /
`L1L2Memory`) come from [src/](../../src/). The kernel reuses the shared
`Worker` via [WorkerPostProcessor](../../src/worker.h); strip-level logic
lives in `DwConvPostProcessor` ([dw_conv2d_top.cpp](dw_conv2d_top.cpp)).

## Hardware model

### Tensor / parallelism
From [dw_conv2d_config.h](dw_conv2d_config.h):
- `DW_C=64`, `DW_H=128`, `DW_W=128` — input `[C, H, W]`, channels-first.
- `DW_INPUT_ELEM_BYTES=1` (int8), `DW_OUTPUT_ELEM_BYTES=4` (int32 accumulator).
- `DW_KH=DW_KW=3`, `DW_PAD=1`, `DW_STRIDE=1`.
- `DW_NUM_WORKERS=16`; channels split evenly:
  `c_start = tid * DW_C / DW_NUM_WORKERS`.

### Accelerator / memory geometry (inherited from hardware_config.h)
All hardware knobs derive from [config/hardware_config.h](../../config/hardware_config.h)
so dw_conv2d, pooling and matmul see identical hardware:
- Vector accelerator: `DW_VEC_ACC_CAP = VECTOR_ACC_CAP = 64` B/call,
  `DW_VEC_ACC_CYCLE = VECTOR_ACC_CYCLE = 1` cycle.
- Pinned pool ([accelerator_pool.h:66-70](../../src/accelerator_pool.h#L66-L70)):
  `DW_VEC_ACC_INSTANCES = VEC_ACCEL_COUNT` units, **round-robin** worker map
  (worker `i` → unit `i % U`). With defaults (16 workers, 4 units) each
  unit serves 4 workers; each unit's per-instance row in the report shows
  `req_count = Kh*Kw × strips_assigned_to_that_unit`.
- Queue depth: `DW_ACC_QUEUE_DEPTH = max(HW_ACC_QUEUE_DEPTH, VEC_ACCEL_COUNT*4)`.
- L1: `DW_L1_BW = VECTOR_ACC_CAP`, `DW_L1_BASE_LAT=1`, `DW_L1_SLOTS=8`.
- DMA / L2: `DW_L2_BW = DW_L1_BW/4`, `DW_L2_BASE_LAT=10`, `DW_L2_SLOTS=16`.

### DMA scalar overhead
`DW_DMA_VEC_RD_SCALAR = HW_DMA_VEC_RD_SCALAR` and
`DW_DMA_VEC_WR_SCALAR = HW_DMA_VEC_WR_SCALAR` — charged once per L2 DMA via
`Worker::configure_dma_vec_cost` (matches matmul `DmaScalarMode::VecPerCall`).

## Per-strip request shape (v2)
For each `(channel, oh, strip)`:
| Stage | Count | rd_bytes | wr_bytes | Notes |
|-------|-------|----------|----------|-------|
| L2 DMA prefetch | 1 | sum of in-bounds bytes across all `(kh,kw)` (+ kernel bytes on `oh==0 && st==0` per channel) | 0 | scalar `dma_vec_rd_scalar` |
| Vec sub-requests (first `Kh*Kw - 1`) | each | per-`(kh,kw)` in-bounds `vl * input_elem_bytes` | 0 | scalar `scalar_overhead` per call |
| Vec sub-request (last `(kh,kw)`) | 1 | as above | `vl * output_elem_bytes` | L1 writeback checkpoint |
| L2 DMA writeback | 1 | 0 | `vl * output_elem_bytes` | fire-and-forget; scalar `dma_vec_wr_scalar` |

With defaults: `64 × 128 × 2` strips × 9 sub-requests = **147 456** vec calls;
per-unit balance = 36 864 reqs/unit (4 units).

## Communication model
Unchanged from project baseline: TLM-2.0 non-blocking only;
`Worker → Interconnect → AcceleratorPool → Memory`; backpressure via
`TLM_ACCEPTED` + deferred `END_REQ`; `ReqExt`/`TxnExt`/`MemoryAccessExt`
extensions carry cycles, byte counts, routing context, and L1/DMA classification.

## Pipelining

### Outer pipeline — per-strip submit/drain
Per `(channel, oh, strip)` the post-processor runs the following in
`DwConvPostProcessor::run_post_mat`:
1. L2 DMA prefetch via `issue_dma_begin(false, ...) / finish_dma` (blocking
   this strip).
2. Acquire per-unit `sc_semaphore`.
3. Submit all `Kh*Kw` sub-requests via `Worker::issue_begin`; the last one
   carries `wr = vl × output_elem_bytes`. `do_scalar(scalar_overhead)` per
   sub-request.
4. Release the unit lock (no other worker's strip can interleave its
   admissions between this strip's sub-requests on this unit).
5. Drain completion events via `Worker::issue_end`. Internal load / compute
   / write stages on the pinned unit overlap automatically.
6. Charge `dma_vec_wr_scalar` then `issue_dma_begin(true, ...)` for the L2
   writeback. Pushed onto a per-worker `write_inflight` deque bounded by
   `cfg.max_inflight_dma_writes` (default 2); when full, the oldest is
   reaped. All remaining writes drain at end of run.

### Inner pipeline — load / compute / writeback within one vec request
Pinned mode always pipelines its per-unit stages: each pinned vec unit spawns
`load_thread`, `compute_thread`, `write_thread` (capacity-1 stage queues), so
one request's L1 read overlaps with the next request's compute and the
previous request's L1 writeback on the same unit.

### Strict strip contiguity
The `sc_semaphore` plus the pinned per-unit FIFO means the sub-requests of
one strip are admitted to the unit contiguously — no other worker's
sub-request can interleave between them. Completion drain happens **after**
release so the queue can still overlap.

## Counters (verified by `compute_expected`)
For default geometry (64×128×128 input, 3×3 kernel, pad=1):
- `expected_vec_calls = Σ(channels × out_h × strips × Kh × Kw) = 147 456`.
- `expected_l1_reqs = total_nonzero_l1_rd + total_strips = 163 072`
  (sub-requests with an entirely out-of-bounds `kh` row emit no L1 read).
- `expected_l1_read_bytes` — sum of in-bounds input bytes (kernel weights
  are not double-counted on the L1 side; they ride only the L2 prefetch DMA).
- `expected_l1_write_bytes = vl × output_elem_bytes × total_strips`.
- `expected_l2_dma_reqs = 2 × total_strips` (one prefetch + one writeback).
- `expected_l2_read_bytes` = in-bounds inputs **+** kernel weights (1×Kh×Kw
  per first-strip-of-first-row per channel).
- `expected_l2_write_bytes = expected_l1_write_bytes` (same payload, L1→L2).

## Reporting
Aligned with matmul/pooling via the shared [report_formatter](../../src/report_formatter.h):
- Simulation Info: tensor shape, kernel shape, pad/stride, output shape.
- Hardware Config: workers / vec instances / binding mode / workers-per-unit
  / cap / cycle / queue depth / L1 + L2 bw/lat/slots.
- Worker Summary table via `report::make_worker_summary_table`.
- Accelerator Summary including per-instance vec rows via
  `report::make_per_instance_accel_rows`, plus L1 Memory and L2 DMA rows.
- Overall Summary: totals, BW, critical-path tid, **and cycle-fraction
  breakdown** (Vec / DMA / Scalar / Stall) normalized to the sum of
  categorized cycles on the critical-path worker — same convention as
  matmul/pooling.
- Verification block: PASS only when every expected vs. actual counter
  matches.

## CLI / build
```
make kernel-dwconv
./kernel/build/dw_conv2d_sim \
    [--workers N] [--channels C] [--height H] [--width W] \
    [--kernel-h Kh] [--kernel-w Kw] [--pad P] [--stride S] \
    [--max-inflight-vec N] [--max-inflight-dma-writes N] \
    [--dma-base-lat N]
```
Exit code 2 indicates verification or req-count mismatch. CLI lives in
[main.cpp](main.cpp); `DwConvRuntimeConfig::defaults()` seeds the run and
CLI values overwrite fields before `DwConvTop` is built.

## Sweeps (v3)
[parametric_sweep.py](parametric_sweep.py) sweeps input size and thread
count, rebuilds the simulator for each `--vec-accels` value, runs the binary
per `(workers × shape)` point, and emits a CSV column-compatible with
matmul/pooling plus dw-specific aliases (`dw_channels`, `dw_height`,
`dw_width`, `dw_kernel_h`, `dw_kernel_w`, `dw_pad`, `dw_stride`). Output:
[parametric_sweep.csv](parametric_sweep.csv) +
[parametric_sweep.png](parametric_sweep.png).

Typical invocation:
```
python3 kernel/dw_conv2d/parametric_sweep.py \
    --max-workers 64 --size-multipliers 1,2,4,8
```
Use `--plot-from-csv` to re-render the PNG without rebuilding/rerunning.

## Constraints
- Do not change the simulator structure or modeling strategy unless
  explicitly asked (see [CLAUDE.md](../../CLAUDE.md) at repo root).
- Do not modify `src/` without asking (see [kernel/CLAUDE.md](../CLAUDE.md)).
  The shared `Worker` already exposes everything dwconv needs
  (`issue_begin/issue_end`, `issue_dma_begin/finish_dma`,
  `configure_dma_vec_cost`, `max_inflight_vec_reqs`, post-processor hook).
- Hardware knobs are aligned with pooling/matmul so Nafnet integration sees
  the same memory/accelerator/scalar parameters across all kernels.
