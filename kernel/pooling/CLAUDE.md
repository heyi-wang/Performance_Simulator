# Pooling simulator — calibrated state

## Scope
SystemC TLM-2.0 Global Average Pooling sim under [kernel/pooling/](.). Models
GAP on `VEC_ACCEL_COUNT` vector accelerators with channel-parallel workers,
DMA prefetch + double-buffering, and an inline divide-by-N at the end of each
channel. Shared building blocks (Worker / Interconnect / AcceleratorPool /
L1L2Memory) come from [src/](../../src/). The kernel keeps a dedicated
`PoolWorker` SC_THREAD because the per-channel reduction loop with end-of-channel
writeback does not map cleanly onto `Worker::issue_stream` (which assumes a
fixed per-call rd/wr payload — pooling's last tile is the only one that writes,
and may be a partial tile).

## Hardware model

### Tensor / parallelism
From [pooling_config.h](pooling_config.h):
- `POOL_C=32`, `POOL_H=64`, `POOL_W=64` — input shape `[C,H,W]`, channels-first.
- `POOL_INPUT_ELEM_BYTES=1` (int8), `POOL_OUTPUT_ELEM_BYTES=4` (int32 accumulator).
- `POOL_NUM_WORKERS=1` (default); channels are distributed evenly:
  `c_start = tid * C / num_workers`.

### Accelerator / memory geometry (aligned with matmul)
All hardware knobs derive from [config/hardware_config.h](../../config/hardware_config.h)
so pooling and matmul see identical hardware:
- Vector accelerator: `POOL_VEC_ACC_CAP = VECTOR_ACC_CAP = 64` B/call,
  `POOL_VEC_ACC_CYCLE = VECTOR_ACC_CYCLE = 1` cycle, `POOL_VEC_ACC_INSTANCES = VEC_ACCEL_COUNT = 8`.
- Queue depth: `POOL_ACC_QUEUE_DEPTH = max(HW_ACC_QUEUE_DEPTH, VEC_ACCEL_COUNT*4)`.
- L1: `POOL_L1_BW = VECTOR_ACC_CAP` (one vec/cycle), `POOL_L1_BASE_LAT=1`, `POOL_L1_SLOTS=8`.
- DMA / L2: `POOL_L2_BW = POOL_L1_BW/4`, `POOL_L2_BASE_LAT=10`, `POOL_L2_SLOTS=16`.
- L1 tile buffers: `POOL_L1_TILE_BUFFERS = POOL_ACC_QUEUE_DEPTH` (matches matmul's
  read/compute/write stream depth = queue cap).

### DMA scalar overhead
Per-DMA setup cost (vector pipe). Mirrors matmul's `DmaScalarMode::VecPerCall`:
each pooling DMA carries one tile of int8 input or one int32 output scalar, so
the cost is charged once per call in [pooling_top.cpp](pooling_top.cpp) `issue_dma`
via `do_scalar(is_write ? cfg.dma_vec_wr_scalar : cfg.dma_vec_rd_scalar)` —
charged *before* `wait`, so it lands on the critical path the same way matmul does.

Defaults:
`POOL_DMA_VEC_RD_SCALAR = HW_DMA_VEC_RD_SCALAR`,
`POOL_DMA_VEC_WR_SCALAR = HW_DMA_VEC_WR_SCALAR`.

### Inline scalar divide
After all reduction tiles for a channel complete:
`output[c] = (int32_t)(total_sum / spatial)`.
Modeled as a CPU stall of `POOL_DIVIDE_CYCLES=4` followed by a one-tile
L2 DMA store of `POOL_OUTPUT_ELEM_BYTES`. No vec/coordinator split.

## Communication model
Unchanged from project baseline: TLM-2.0 non-blocking only;
`Worker → Interconnect → AcceleratorPool → Memory`; backpressure via
`TLM_ACCEPTED` + deferred `END_REQ`; `ReqExt`/`TxnExt` extensions carry
cycles, byte counts, and routing context.

## Pipelining

### Outer pipeline — DMA ↔ accelerator (mirrors `Worker::issue_stream`)
`PoolWorker::run` runs a 2-queue polling loop per channel
([pooling_top.cpp](pooling_top.cpp)):
- `read_inflight` — DMA loads in flight (issued via `issue_dma(false, ...)`).
- `accel_inflight` — vec-compute requests in flight (issued via `issue_begin`).

Per iteration the worker:
1. **issue_read** while `next_load_tile < tile_count`,
   `read_inflight + accel_inflight < l1_tile_buffers`, and
   `read_inflight < max_inflight_vec_reqs`.
2. **promote_read** — drain oldest DMA (`issue_end` on its handle),
   then `issue_begin` its compute with
   `wr = is_last_tile ? output_elem_bytes : 0`.
3. **do_scalar(scalar_overhead)** once per compute promotion.
4. **retire_accel** when the window is full or all reads have been issued.

Loop terminates when `next_load_tile == tile_count`, both queues empty.

Effective pipeline depth = `min(max_inflight_vec_reqs, l1_tile_buffers)`.
With `max_inflight=1` it reduces to the prior prefetch-one-ahead pattern.

### Inner pipeline — load / compute / writeback within one vec request
`PoolTop` constructs the vec `AcceleratorPool` with
`enable_unit_pipeline=true` so each unit spawns `load_thread`, `compute_thread`,
`write_thread` (capacity-1 stage queues). One request's L1 read overlaps with
the next request's compute and the previous request's L1 writeback on the same
unit. This is more aggressive than matmul's vec phase (matmul keeps
`unit_pipeline=false`).

### End-of-channel writeback semantics (Pooling.md L13)
The partial sum lives in the vec accelerator's accumulator register across
tiles of one channel. The L1 writeback happens **once per channel** on the
final tile (`wr = output_elem_bytes` only when `t == tile_count - 1`).
Then the inline divide and L2 DMA store run.

Counter consequences (verified in `collect_stats`):
- `expected_l1_write_bytes = channels × output_elem_bytes`.
- `expected_l1_reqs = vec_calls + channels` (one read per vec call +
  one final write per channel).
- L2 DMA expectations unchanged: one write DMA per channel.

## Reporting
Goes through the shared [report_formatter](../../src/report_formatter.h):
- Hardware Config block: surfaces L1 bw / lat / slots and L2 bw / lat / slots
  alongside the vec accelerator geometry.
- Per-worker info via `KernelWorkerInfo`.
- Per-vec-instance rows via `report::make_per_instance_accel_rows(vec_acc.units, "Vector")`.
- Overall Summary includes cycle-fraction lines
  (vec / dma / scalar / stall), normalized to categorized cycles on the
  critical-path worker, matching matmul's convention.

Total elapsed = slowest worker's `elapsed_cycles`. Verification predicate is
authoritative on byte/req counts; the `dma_accel_overlap_cycles > 0`
predicate was removed because the L1 buffer cap can serialize the bootstrap
window and produce zero overlap on small tile counts even when the pipeline
is wired correctly.

## CLI / build
```
make kernel-pooling
./kernel/build/pooling_sim [--workers N] [--channels C] [--height H] \
    [--width W] [--max-inflight-vec N] [--dma-base-lat N]
```
Exit code 2 indicates verification or req-count mismatch.

CLI parser is [main.cpp](main.cpp); `sc_main` lives there (the standalone
`sc_main` block in `pooling_top.cpp` was removed when `main.o` joined
`OBJS_POOL`). Workload defaults come from `PoolRuntimeConfig::defaults()`;
CLI values overwrite them before `PoolTop` is built.

## Sweeps
[parametric_sweep.py](parametric_sweep.py) drives parameter sweeps via `-D`
overrides on the compile line; results emitted as CSV.

## Constraints
- Do not change the simulator structure or modeling strategy unless
  explicitly asked (see [CLAUDE.md](../../CLAUDE.md) at repo root).
- Do not modify `src/` without asking (see [kernel/CLAUDE.md](../CLAUDE.md)).
  The shared `Worker` already exposes everything pooling needs
  (`max_inflight_vec_reqs`, `dma_vec_*_scalar`); the outer pipeline is
  re-implemented inside `PoolWorker` because `Worker::issue_stream`
  assumes fixed per-call rd/wr.
- Hardware knobs are aligned with matmul so Nafblock/Nafnet integration
  sees the same memory/accelerator/scalar parameters across both kernels.
