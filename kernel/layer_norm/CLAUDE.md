# LayerNorm2d simulator — calibrated state (v1 + v2 + v3)

## Scope
SystemC TLM-2.0 layer-normalization sim under [kernel/layer_norm/](.). Models
a per-channel, three-pass `[C, H, W]` LayerNorm2d on `VEC_ACCEL_COUNT` pinned
vector accelerators with L2 DMA prefetch per channel, on-unit int32
accumulators (sum / sum-of-squares), an L1 writeback on every pass-3 vec
request, and a fire-and-forget L2 DMA writeback per channel.

Shared building blocks (`Worker` / `Interconnect` / `AcceleratorPool` /
`L1L2Memory`) come from [src/](../../src/). The kernel reuses the shared
`Worker` via [WorkerPostProcessor](../../src/worker.h); per-channel logic
lives in `LayerNormPostProcessor` ([layer_norm_top.cpp](layer_norm_top.cpp)).

## Hardware model

### Tensor / parallelism
From [layer_norm_config.h](layer_norm_config.h):
- `LN_C=32`, `LN_H=64`, `LN_W=64` — input `[C, H, W]`, channels-first.
- `LN_INPUT_ELEM_BYTES=1` (int8), `LN_OUTPUT_ELEM_BYTES=1` (int8).
  Intermediate sums live in int32 in the pinned unit's register file.
- `LN_NUM_WORKERS=4`; channels split evenly:
  `c_start = tid * LN_C / LN_NUM_WORKERS`.

### Accelerator / memory geometry (inherited from hardware_config.h)
All hardware knobs derive from [config/hardware_config.h](../../config/hardware_config.h)
so layer_norm, dw_conv2d, pooling and matmul see identical hardware:
- Vector accelerator: `LN_VEC_ACC_CAP = VECTOR_ACC_CAP = 64` B/call.
- Pinned pool ([accelerator_pool.h:66-70](../../src/accelerator_pool.h#L66-L70)):
  `LN_VEC_ACC_INSTANCES = VEC_ACCEL_COUNT` units, **round-robin** worker map
  (worker `i` → unit `i % U`).
- Queue depth: `LN_ACC_QUEUE_DEPTH = max(HW_ACC_QUEUE_DEPTH, VEC_ACCEL_COUNT*4)`.
- L1: `LN_L1_BW = VECTOR_ACC_CAP`, `LN_L1_BASE_LAT=1`, `LN_L1_SLOTS=8`.
- DMA / L2: `LN_L2_BW = LN_L1_BW/4`, `LN_L2_BASE_LAT=10`, `LN_L2_SLOTS=16`.

### Per-pass compute cost (v2)
`LN_VEC_INSN_CYCLE = HW_VECTOR_INSN_CYCLE` (cycles per arithmetic intrinsic).
Each pass charges only its arithmetic vector intrinsics (load / store /
setvl / `vmv` identity-init excluded). Counted from the RVV reference
[kernel/layer_norm.h](../layer_norm.h) `mf_layernorm2d_i8`:

| Pass | Inner-loop arithmetic intrinsics                                                                                | Count |
|------|-----------------------------------------------------------------------------------------------------------------|-------|
| 1 (sum)        | `vwmul_vx`, `vredsum_vs`                                                                              | **2** |
| 2 (variance)   | `vwmul_vx`, `vsub_vx`, `vwmul_vv`, `vredsum_vs`                                                       | **4** |
| 3 (normalize)  | `vwmul_vx`, `vsub_vx`, `vwmul_vx`, `vmul_vx`, `vsra_vx`, `vadd_vx`, `vmax_vx`, `vmin_vx`, `vnsra_wx`, `vnsra_wx` | **10** |

Per-request service cycles = `LN_PASSk_INSNS * LN_VEC_INSN_CYCLE` via
`vec_request_cycles` ([src/common.h:18-22](../../src/common.h#L18-L22)).
The runtime `LayerNormRuntimeConfig::pass_cycles(int pass)` helper is used
both in `run_pass` (to set `svc_cycles` on the request) and in
`compute_expected` (to derive `expected_vec_acc_busy_cycles`).

### Scalar costs between passes
- `LN_MEAN_CYCLES=4` — integer `sum/N` charged once after pass 1.
- `LN_INVSTD_CYCLES=16` — `var/N + isqrt + 1/std` charged once after pass 2.
- `LN_SCALAR_OVERHEAD = HW_VEC_SCALAR_OVERHEAD` — per-vec-dispatch scalar
  overhead, charged via `Worker::do_scalar` before every vec request.

### DMA scalar overhead
`LN_DMA_VEC_RD_SCALAR = HW_DMA_VEC_RD_SCALAR` and
`LN_DMA_VEC_WR_SCALAR = HW_DMA_VEC_WR_SCALAR` — charged once per L2 DMA via
`Worker::configure_dma_vec_cost` (matches matmul `DmaScalarMode::VecPerCall`).

## Per-channel request shape
For each `channel` owned by a worker (let `tiles = ceil(H*W / VEC_ACC_CAP)`,
`vl = min(VEC_ACC_CAP, remaining)`):
| Stage | Count | rd_bytes | wr_bytes | Notes |
|-------|-------|----------|----------|-------|
| L2 DMA prefetch | 1 | `H*W * LN_INPUT_ELEM_BYTES` | 0 | scalar `dma_vec_rd_scalar` |
| Pass 1 vec reqs (sum) | `tiles` | `vl` | 0 | `svc = 2 * LN_VEC_INSN_CYCLE` |
| Scalar mean | 1 | — | — | `do_scalar(LN_MEAN_CYCLES)` |
| Pass 2 vec reqs (var) | `tiles` | `vl` | 0 | `svc = 4 * LN_VEC_INSN_CYCLE` |
| Scalar inv-std | 1 | — | — | `do_scalar(LN_INVSTD_CYCLES)` |
| Pass 3 vec reqs (norm)| `tiles` | `vl` | `vl * LN_OUTPUT_ELEM_BYTES` | `svc = 10 * LN_VEC_INSN_CYCLE`, L1 writeback per tile |
| L2 DMA writeback | 1 | 0 | `H*W * LN_OUTPUT_ELEM_BYTES` | fire-and-forget; scalar `dma_vec_wr_scalar` |

With defaults: `32 channels × 3 passes × 64 tiles = 6 144` vec reqs total
(2 048 per pass); `expected_vec_acc_busy_cycles = 32 × 64 × (2+4+10) × 1 =
32 768`.

## Communication model
Unchanged from project baseline: TLM-2.0 non-blocking only;
`Worker → Interconnect → AcceleratorPool → Memory`; backpressure via
`TLM_ACCEPTED` + deferred `END_REQ`; `ReqExt`/`TxnExt`/`MemoryAccessExt`
extensions carry cycles, byte counts, routing context, and L1/DMA classification.

## Pipelining

### Per-channel pass ordering
Within one channel, the three passes run back-to-back in
`LayerNormPostProcessor::run_post_mat`:
1. L2 DMA prefetch (blocking this channel).
2. Pass 1: submit `tiles` vec reqs, drain, `do_scalar(LN_MEAN_CYCLES)`.
3. Pass 2: submit `tiles` vec reqs, drain, `do_scalar(LN_INVSTD_CYCLES)`.
4. Pass 3: submit `tiles` vec reqs, drain.
5. L2 DMA writeback pushed onto per-worker `write_inflight` deque bounded by
   `cfg.max_inflight_dma_writes` (default 2); oldest reaped when full, all
   drained at end of run.

Cross-channel pass interleaving is **not** allowed for a single worker —
pass 2 needs the mean from pass 1, pass 3 needs `inv_std` from pass 2. The
pinned per-unit FIFO already serializes one worker's submissions; no extra
`sc_semaphore` is needed because channels are independent across workers.

### Inflight cap within a pass
`LN_MAX_INFLIGHT_VEC_REQS = LN_ACC_QUEUE_DEPTH` (default = per-unit queue
capacity) — the worker keeps its pinned unit fully fed during each pass.

### Inner pipeline — load / compute / writeback within one vec request
Pinned mode always pipelines its per-unit stages: each pinned vec unit spawns
`load_thread`, `compute_thread`, `write_thread` (capacity-1 stage queues), so
one request's L1 read overlaps with the next request's compute and the
previous request's L1 writeback on the same unit.

## Counters (verified by `compute_expected`)
For default geometry (`C=32, H=W=64`):
- `expected_pass1_reqs = expected_pass2_reqs = expected_pass3_reqs =
  C × tiles = 2 048`; `expected_vec_reqs = 6 144`.
- `expected_vec_acc_busy_cycles = C × tiles × (2+4+10) × LN_VEC_INSN_CYCLE =
  32 768` — verified end-to-end (v2 broadens the predicate to include this).
- `expected_l1_reqs = expected_vec_reqs + C × tiles = 8 192` (pass 3 emits
  an L1 read **and** an L1 write per tile; passes 1/2 emit reads only).
- `expected_l1_read_bytes = 3 × C × H × W × LN_INPUT_ELEM_BYTES` — each pass
  rewalks the channel from L1.
- `expected_l1_write_bytes = C × H × W × LN_OUTPUT_ELEM_BYTES` — pass 3 only.
- `expected_l2_dma_reqs = 2 × C` (one prefetch + one writeback per channel).
- `expected_l2_read_bytes  = C × H × W × LN_INPUT_ELEM_BYTES`.
- `expected_l2_write_bytes = C × H × W × LN_OUTPUT_ELEM_BYTES`.

Verification PASSes only when every expected vs. actual counter matches,
**including** `vec_acc_busy_cycles` and the per-pass req counts.

## Reporting
Aligned with matmul / pooling / dw_conv2d via the shared
[report_formatter](../../src/report_formatter.h):
- Simulation Info: op type, `[C, H, W]`, element types.
- Hardware Config: workers / vec instances / binding mode / L1+L2
  bw/lat/slots, **plus** matmul-style per-class insn counts:
    - `Vector Instruction Cycle [cycles/insn]`
    - `Pass-1 Vector Instructions [insns/request]` = 2
    - `Pass-2 Vector Instructions [insns/request]` = 4
    - `Pass-3 Vector Instructions [insns/request]` = 10
- Worker Summary table via `report::make_worker_summary_table`.
- Accelerator Summary including pool-level + per-instance vec rows via
  `report::make_per_instance_accel_rows`, plus L1 Memory + L2 DMA rows.
- Overall Summary: totals, BW, critical-path tid, **and cycle-fraction
  breakdown** (Vec / DMA / Scalar / Stall) on the critical-path worker —
  same normalization as matmul / pooling / dw_conv2d. Critical-path
  `vec_service` is computed per pass (`pass_cycles(1..3) * reqs_in_pass`)
  rather than the V1 flat `vec_calls * vec_acc_cycle`.
- Verification block: every expected vs. actual counter, including
  `Expected/Actual Vector Accelerator Busy Cycles` and per-pass req lines;
  PASS / FAIL line consumed by sweep scripts.

## CLI / build
```
make kernel-layernorm
./kernel/build/layer_norm_sim \
    [--workers N] [--channels C] [--height H] [--width W] \
    [--max-inflight-vec N] [--max-inflight-dma-writes N] \
    [--dma-base-lat N]
```
Exit code 2 indicates verification or req-count mismatch. CLI lives in
[main.cpp](main.cpp); `LayerNormRuntimeConfig::defaults()` seeds the run and
CLI values overwrite fields before `LayerNormTop` is built.

## Sweeps (v3)
[parametric_sweep.py](parametric_sweep.py) sweeps input size and thread
count, rebuilds the simulator once per `--vec-accels` value (via
`make kernel … EXTRA_CXXFLAGS=-DVEC_ACCEL_COUNT=N`), runs the binary per
`(workers × shape)` point, parses the report, and emits a CSV column-
compatible with matmul / pooling / dw_conv2d (`tile_*`, `mat_*`, `vec_*`,
`gemm_*`, `threads`, cycle-fraction columns) plus layer_norm-specific
aliases (`ln_channels`, `ln_height`, `ln_width`,
`max_inflight_dma_writes`). Output:
[parametric_sweep.csv](parametric_sweep.csv) +
[parametric_sweep.png](parametric_sweep.png).

Typical invocation:
```
python3 kernel/layer_norm/parametric_sweep.py \
    --max-workers 64 --size-multipliers 1,2,4,8
```
Use `--plot-from-csv` to re-render the PNG without rebuilding/rerunning.

### Observed scaling (default `C=32`, vec=4)
From the calibration run (`--max-workers 16 --size-multipliers 1,2,4`):

| Shape `[C,H,W]` | w=1 | w=2 | w=4 | w=8 | w=16 |
|-----------------|-----|-----|-----|-----|------|
| 32 × 64 × 64    |  62 858 |  31 562 |  15 914 |  10 596 |   9 112 |
| 32 × 128 × 128  | 247 946 | 124 490 |  62 762 |  42 084 |  36 376 |
| 32 × 256 × 256  | 988 298 | 496 202 | 250 154 | 168 036 | 145 432 |

Near-linear up to `workers = VEC_ACCEL_COUNT`, then knees once round-robin
pinning oversubscribes the four vec units (extra threads only parallelise
the scalar / DMA phases; vec throughput is unit-bound). Vec utilisation
therefore decreases beyond `workers = 4` even though total vec busy cycles
stay constant — see the V2 discussion notes.

## Constraints
- Do not change the simulator structure or modeling strategy unless
  explicitly asked (see [CLAUDE.md](../../CLAUDE.md) at repo root).
- Do not modify `src/` without asking (see [kernel/CLAUDE.md](../CLAUDE.md)).
  The shared `Worker` already exposes everything layer_norm needs
  (`issue_begin/issue_end`, `issue_dma_begin/finish_dma`,
  `configure_dma_vec_cost`, `max_inflight_vec_reqs`, post-processor hook).
- Hardware knobs are aligned with pooling / dw_conv2d / matmul so the
  nafnet bridge sees the same memory / accelerator / scalar parameters
  across all kernels. The bridge in
  [nafnet/nafnet_layers.h](../../nafnet/nafnet_layers.h) still uses the
  legacy 4-step int16 expected-stats path; `LN_VEC_ACC_CYCLE` and
  `LN_STEP3_CYCLES` are kept as back-compat aliases in
  [layer_norm_config.h](layer_norm_config.h) so it compiles. Updating the
  bridge's stale expected-stats logic is out of scope for V1/V2/V3.
