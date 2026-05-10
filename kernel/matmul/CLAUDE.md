# Matmul simulator — calibrated state

## Scope
SystemC TLM-2.0 matmul kernel sim under [kernel/matmul/](.). Models a K-split
GEMM on `MAT_ACCEL_COUNT` matrix accelerators with per-thread tree reduction
and parallel final quantization on `VEC_ACCEL_COUNT` vector accelerators.
Workers / Interconnect / AcceleratorPool / L1L2Memory come from [src/](../../src/).

## Hardware model

### Tile / accelerator geometry
From [config/hardware_config.h](../../config/hardware_config.h):
- `MATMUL_M=8`, `MATMUL_K=8`, `MATMUL_N=8` — matrix accelerator tile.
- `MAT_ACCEL_COUNT=4`, `VEC_ACCEL_COUNT=8`, `ACC_QUEUE_DEPTH=32`.
- `VECTOR_ACC_CAP=64` bytes — vector datapath width (one vector / cycle).
- `MATMUL_ACC_CYCLE=1`, `VECTOR_INSN_CYCLE=1`.

### Memory
- L1 (`l1_bw`) is **aligned with `VECTOR_ACC_CAP`** in [matmul_top.h](matmul_top.h):
  one vector-wide load is delivered to the vector accumulator in 1 cycle.
  Matrix accelerator data load/store overhead is **derived** from this BW
  via [memory.h:32-35](../../src/memory.h#L32-L35) (`base_lat + ceil(bytes / l1_bw)`).
  Example: a 256 B C-tile costs `1 + ceil(256/64) = 5` L1 cycles.
- DMA bandwidth (`dma_bw=64`) and base latency (`dma_base_lat=10`) are
  modeled separately in `MatmulRuntimeConfig`.
- L1 / DMA service time uses bounded parallel slots
  (`l1_slots=4`, `dma_slots=8`).

### DMA scalar overhead — phase-split

`Worker::issue_stream` accounts the per-DMA scalar setup cost via the
`DmaScalarMode` enum ([src/worker.h](../../src/worker.h)):

| Phase | Mode | Cost per read DMA | Cost per write DMA | Source |
|-------|------|-------------------|--------------------|--------|
| Matmul (matrix tiles) | `MatRow` | `HW_DMA_A_TILE_SCALAR` (per A tile DMA) | `dma_c_rows × HW_DMA_C_ROW_SCALAR` (one `dma.x` per C row) | A is one tile DMA; C still streams row-by-row |
| Reduction / quantization (vector pipe) | `VecPerCall` | `HW_DMA_VEC_RD_SCALAR` | `HW_DMA_VEC_WR_SCALAR` | one DMA carries one vector-wide payload |

The matmul phase additionally charges `HW_DMA_B_TILE_SCALAR` once per K-tile
inside [worker.cpp](../../src/worker.cpp) (`issue_gemm_reuse_stream`) for the
single B-tile DMA. The per-tile load model is the canonical strategy for
matrix-tile DMAs going forward (any future kernel modeling matrix-tile loads
should reuse it rather than re-introducing per-row load accounting).

Defaults in `hardware_config.h`:
`HW_DMA_A_TILE_SCALAR=20`, `HW_DMA_B_TILE_SCALAR=20`, `HW_DMA_C_ROW_SCALAR=1`,
`HW_DMA_VEC_RD_SCALAR=1`, `HW_DMA_VEC_WR_SCALAR=1`. All `#ifndef`-guarded.

Workers receive these costs from [matmul_top.cpp](matmul_top.cpp) via
`configure_dma_tile_load_cost(...)`, `configure_dma_c_row_cost(...)`, and
`configure_dma_vec_cost(...)`.

## Communication model
Unchanged from project baseline: TLM-2.0 non-blocking only;
`Worker → Interconnect → AcceleratorPool → Memory`; backpressure via
`TLM_ACCEPTED` + deferred `END_REQ`; `ReqExt`/`TxnExt` extensions carry
cycles, byte counts, and routing context.

## Phases per worker
1. Matmul ([worker.cpp](../../src/worker.cpp) `run()` → `issue_gemm_reuse_stream`):
   reads A and B tiles, writes C. Uses `DmaScalarMode::MatRow`.
2. Tree reduction (run inside the worker SC_THREAD via
   [accum_coordinator.cpp](accum_coordinator.cpp) `run_one_pair`):
   pairs of partial-sum vectors are accumulated on vector accelerators.
   Uses `DmaScalarMode::VecPerCall`.
3. Final quantization (workers run their own slice of the output via
   `run_final_quant`): vector accelerator. `DmaScalarMode::VecPerCall`.

Total elapsed = slowest worker's `elapsed_cycles`. The next layer
(in nafnet) starts only after every worker for the current layer
completes.

## Reporting
Per-thread report (from [matmul_sim/CLAUDE.md](../../matmul_sim/CLAUDE.md)
requirements):
- matrix accelerator calls
- vector accelerator calls in reduction phase (`accum_vec_calls`)
- vector accelerator calls in quantization phase (`quant_vec_calls`)
- reductions claimed (`reduction_pairs`)
- scalar cycles, stall cycles, memory cycles, elapsed cycles

Per-accelerator-instance report: requests, busy / occupied / queue-wait
cycles, compute and occupancy utilization.

Memory hierarchy report exposes L1 vs DMA byte counts, busy cycles, and
queue-wait cycles separately so the L1↔vector alignment effect is
visible (L1 busy grows when matrix tile loads exceed `VECTOR_ACC_CAP`).

## Build / run
```
make kernel-matmul
./kernel/build/matmul_sim --threads <T> --accum-registers <R>
```
Exit code 2 indicates verification or req-count mismatch.

## Sweeps
[parametric_sweep.py](parametric_sweep.py),
[sweep_workers.py](sweep_workers.py),
[hardware_sweep.py](hardware_sweep.py) drive parameter sweeps via `-D`
overrides on the compile line; results emitted as CSV.

[full_sweep.py](full_sweep.py) covers the full eight-dimensional spec from
[Parametric_Sweep.md](Parametric_Sweep.md): five compile-time dims
(`MATMUL_M/K/N`, `MATMUL_ACC_CYCLE`, `MAT_ACCEL_COUNT`, `VEC_ACCEL_COUNT`,
`VECTOR_ACC_CAP`) and three runtime dims (GEMM size, `--threads`,
`--dma-base-lat`). The CSV stores every sweep dim as a column plus per-class
cycle fractions (mat / vec / DMA / scalar / stall, summing to 100% on the
critical-path worker) and pool-level utilization. See
[Parametric_Sweep_HOWTO.md](Parametric_Sweep_HOWTO.md) for the runbook.

## Constraints
- Do not change the simulator structure or modeling strategy unless
  explicitly asked (see [CLAUDE.md](../../CLAUDE.md) at repo root).
- `Worker` changes in [src/](../../src/) must remain additive
  (default args preserve behavior for non-matmul kernels — `dw_conv2d`,
  `layer_norm`, `pooling`, `vec_ops` still build with no edits).
