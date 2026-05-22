# Vec_Ops simulator — calibrated state

## Scope
SystemC TLM-2.0 element-wise vector-operations sim under [kernel/vec_ops/](.).
Models per-channel-parallel `VopType` kernels on `VEC_ACCEL_COUNT` pinned
vector accelerators with L2 DMA prefetch + fire-and-forget writeback per
channel. Shared building blocks (`Worker` / `Interconnect` /
`AcceleratorPool` / `L1L2Memory`) come from [src/](../../src/).

Per-op service cost is dispatched through `vop_*` helpers in
[vec_ops_config.h](vec_ops_config.h); the kernel itself ([vec_ops_top.cpp](vec_ops_top.cpp))
treats every op uniformly — adding a new op is an additive switch-case
extension only.

## Hardware model

### Op set
| `VopType` | Insns | Input → Output | rd / vl | wr / vl |
|---|---|---|---|---|
| `VOP_ELEMWISE_ADD`         | 1 | int8 → int8   | 2 | 1 |
| `VOP_ELEMWISE_MUL`         | 1 | int8 → int16  | 2 | 2 |
| `VOP_SCALAR_MUL`           | 1 | int8 → int8   | 1 | 1 |
| `VOP_QUANTIZE_I32_TO_I8`   | 6 | int32 → int8  | 4 | 1 |
| `VOP_QUANTIZE_I16_TO_I8`   | 5 | int16 → int8  | 2 | 1 |
| `VOP_DEQUANTIZE_I8_TO_I32` | 3 | int8 → int32  | 1 | 4 |
| `VOP_DOT_PRODUCT_I8`       | 2 | int8·int8 → i32 scalar | 2 | 0/tile |

`VOP_DOT_PRODUCT_I8` was added for nafblock's `sca_conv` (1×1 conv mapped
to per-output-pixel dot products). Per-tile write is 0 because the i32
partial sum lives in the accumulator register across tiles.

### Tensor / parallelism
From [vec_ops_config.h](vec_ops_config.h):
- `VOP_C=32`, `VOP_H=64`, `VOP_W=64` — input `[C, H, W]`, channels-first.
- `VOP_NUM_WORKERS=16`; channels split evenly:
  `c_start = tid * VOP_C / VOP_NUM_WORKERS`.

### Accelerator / memory geometry (inherited from hardware_config.h)
- Vector accelerator: `VOP_VEC_ACC_CAP = VECTOR_ACC_CAP = 64` B/call,
  `VOP_VEC_INSN_CYCLE = HW_VECTOR_INSN_CYCLE`. Service cycles per request
  = `vop_insn_count(op) * VOP_VEC_INSN_CYCLE`.
- Pinned pool: `VOP_VEC_ACC_INSTANCES = VEC_ACCEL_COUNT` units,
  round-robin worker map. Inner load/compute/write stage pipeline always on.
- L1 / L2 BW + slots aligned with layer_norm / dw_conv2d / pooling.

### DMA scalar overhead
`HW_DMA_VEC_RD_SCALAR` / `HW_DMA_VEC_WR_SCALAR` charged once per L2 DMA
via `Worker::configure_dma_vec_cost` (`DmaScalarMode::VecPerCall`).

## Communication model
Unchanged from project baseline: TLM-2.0 non-blocking only;
`Worker → Interconnect → AcceleratorPool → Memory`; backpressure via
`TLM_ACCEPTED` + deferred `END_REQ`; `ReqExt`/`TxnExt`/`MemoryAccessExt`
extensions carry cycles, byte counts, routing context, and L1/DMA
classification.

## Per-channel request shape
For each `channel` owned by a worker (`tiles = ceil(H*W / tile_cap)`,
`vl = min(tile_cap, remaining)`):
| Stage | Count | rd_bytes | wr_bytes | Notes |
|---|---|---|---|---|
| L2 DMA prefetch | 1 | H·W · sum(input operand widths) | 0 | scalar `dma_vec_rd_scalar` |
| Vec reqs | `tiles` | `vop_rd_bytes(op, vl)` | `vop_wr_bytes(op, vl)` | `svc = vop_insn_count(op) * VOP_VEC_INSN_CYCLE` |
| L2 DMA writeback | 1 | 0 | `vop_wr_bytes(op, H·W)` | fire-and-forget |

`vop_tile_cap_elems(op)` accounts for the wider operand of each op so the
per-tile read budget stays within `VECTOR_ACC_CAP` bytes.

## Reporting
Same shared [report_formatter](../../src/report_formatter.h) as the other
vec-only kernels (layer_norm / dw_conv2d / pooling): Simulation Info,
Hardware Config, Worker Summary, per-instance vec rows + L1 / L2 DMA rows,
Overall Summary with cycle-fraction breakdown (Vec / DMA / Scalar / Stall)
on the critical-path worker.

## CLI / build
```
make kernel-vecops
./kernel/build/vec_ops_sim [--workers N] [--channels C] [--height H] \
    [--width W] [--op <name>] [--max-inflight-vec N] \
    [--max-inflight-dma-writes N] [--dma-base-lat N]
```
Op names match `vop_name(op)` (`mf_elemwise_add_i8`, `mf_elemwise_mul_i8_to_i16`,
`mf_dotprod_i8_to_i32`, etc.). Exit code 2 indicates verification or
req-count mismatch.

## Sweeps
[parametric_sweep.py](parametric_sweep.py) sweeps input size, op type and
worker count; emits [parametric_sweep.csv](parametric_sweep.csv) +
[parametric_sweep.png](parametric_sweep.png) column-compatible with the
other kernel sweeps.

## Constraints
- Do not change the simulator structure or modeling strategy unless
  explicitly asked (see [CLAUDE.md](../../CLAUDE.md) at repo root).
- Do not modify `src/` without asking (see [kernel/CLAUDE.md](../CLAUDE.md)).
- **Adding a new op is additive**: extend `VopType` and the six
  `vop_*` switch helpers in [vec_ops_config.h](vec_ops_config.h);
  no changes to [vec_ops_top.cpp](vec_ops_top.cpp) needed. This is the
  pattern used by `VOP_DOT_PRODUCT_I8` for nafblock's `sca_conv`.
- Hardware knobs aligned with layer_norm / dw_conv2d / pooling / matmul
  so the nafblock/nafnet bridges see identical hardware across kernels.
