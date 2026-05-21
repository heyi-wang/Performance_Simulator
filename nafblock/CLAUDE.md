# SystemC TLM performance simulator of Nafblock in Nafnet
This subproject of performance simulator builds a simulator for a nafblock in Nafnet

The nafblock structure is shown in the image: ![Nafblock](NAFBlock.png "Nafblock structure")


## Requirements v1

- Build simulator for one nafblock. Keep the flexibility to integrate this simulator to the whole nafnet simulator.
- For now ignore the existing thing in @/home/why/Desktop/Performance_Simulator/nafnet
- Parse structure of Nafblock and map the operations to corresponding kernels in the simulator.
- The nafblock input size is configurable

Update the status of simulator back to this file after modification.

## Status

**v1 implemented** — standalone simulator drives all 14 nafblock sub-layers
through the shared `src/` + `kernel/` infrastructure. No dependency on
`nafnet/`.

### Build / run
```
make nafblock                 # builds nafblock/build/nafblock_perf_sim
make -C nafblock run          # runs with default shape
./nafblock/build/nafblock_perf_sim [--block-c N] [--block-h N] [--block-w N]
```
Defaults: `C=32, H=64, W=64`. Exit codes: `0`=PASS, `1`=bad CLI, `2`=verification failure.

### Files
- [nafblock_config.h](nafblock_config.h) — `N_WORKERS` and default tensor shape.
- [nafblock_layers.h](nafblock_layers.h) — `NafBlockLayerDesc`, per-op factory
  helpers, and **`append_nafblock_layers(layers, id, prefix, C, H, W)`** —
  the public entry point for embedding this block in a future NafNet driver.
- [nafblock_kernel_bridge.h](nafblock_kernel_bridge.h) — `LayerDesc → kernel runtime config`
  translators (one `nb_make_*_cfg` per backend).
- [nafblock_sim.cpp](nafblock_sim.cpp) — `LayerRunner` polymorphic hierarchy
  + `NafBlockTop` orchestrator + CLI + report.
- [Makefile](Makefile) — links `src/` + `kernel/*_top.o` + `nafblock_sim.o`.

### Operation → kernel mapping
| # | Sub-layer | Op | Backend (kernel) |
|---|-----------|----|------------------|
| 1 | `norm1`           | LayerNorm        | `LayerNormTop` |
| 2 | `conv1`           | 1×1 Conv (C→2C)  | `MatmulTop` |
| 3 | `conv2_dw`        | DW 3×3 Conv      | `DwConvTop` |
| 4 | `simplegate1`     | Elem mul (2C→C)  | `VecOpsTop` (`VOP_ELEMWISE_MUL`) |
| 5 | `sca_gap`         | Global Avg Pool  | `PoolTop` |
| 6 | `sca_conv`        | 1×1 Conv (C→C)   | `VecOpsTop` (`VOP_DOT_PRODUCT_I8`) |
| 7 | `sca_scale`       | Channel scale    | `VecOpsTop` (`VOP_SCALAR_MUL`) |
| 8 | `conv3`           | 1×1 Conv (C→C)   | `MatmulTop` |
| 9 | `beta_residual`   | β·x + skip       | `VecOpsTop` (`VOP_SCALAR_MUL` → `VOP_ELEMWISE_ADD`) |
|10 | `norm2`           | LayerNorm        | `LayerNormTop` |
|11 | `conv4`           | 1×1 Conv (C→2C)  | `MatmulTop` |
|12 | `simplegate2`     | Elem mul (2C→C)  | `VecOpsTop` (`VOP_ELEMWISE_MUL`) |
|13 | `conv5`           | 1×1 Conv (C→C)   | `MatmulTop` |
|14 | `gamma_residual`  | γ·x + skip       | `VecOpsTop` (`VOP_SCALAR_MUL` → `VOP_ELEMWISE_ADD`) |

### Validated runs (2026-05-20)
- Default `C=32, H=W=64`: 14/14 layers PASS, total ≈3.31 M cycles.
- Custom `C=64, H=W=32`: 14/14 layers PASS, total ≈1.69 M cycles.
- Bad CLI (`--block-c 0`, unknown arg) → exit 1 with error on stderr.

### v2 update (2026-05-21) — `sca_conv` remapped to the vector unit
The SCA 1×1 conv operates on a 1×1×C feature map produced by Global Avg Pool.
Mapping it to `MatmulTop` (8×8×8 tiles) wastes the matrix unit because the
GEMM M dimension is 1. Semantically it's `C_out` independent dot products,
each a `C_in`-length kernel · the `C_in`-length input vector at (0,0); the
vector accelerator handles this with one `vwmul_vv` + `vredsum_vs` per pixel.

Changes:
- Added `VOP_DOT_PRODUCT_I8` to [../kernel/vec_ops/vec_ops_config.h](../kernel/vec_ops/vec_ops_config.h)
  — input bytes 1, output bytes 0 (accumulator-resident scalar), 2 insns,
  per-tile L1 write of 4 bytes (one i32 partial scalar; special-cased in
  `vop_wr_bytes` because the result size is independent of `vl`).
- New helper `nb_make_sca_conv(id, name, C)` in [nafblock_layers.h](nafblock_layers.h)
  re-encodes `C_in` into the `Hout` dimension so the existing
  `nb_make_vecops_cfg` produces `channels = C` and `spatial = C` without
  bridge changes.
- Manifest table row for `sca_conv` now expects `NB_BACKEND_VECOPS`.

Validated:
- `C=32, H=W=64`: `sca_conv` row = 32 vec reqs, 73 cycles (was 720+ on matmul); 14/14 PASS.
- `C=128, H=W=16`: `sca_conv` row = 256 vec reqs (128 channels × ⌈128/64⌉ tiles), 490 cycles, PASS.
- Standalone `make kernel-vecops && ./kernel/build/vec_ops_sim` still PASS — additive enum doesn't affect existing ops.

### Parametric sweep (2026-05-21)
[parametric_sweep.py](parametric_sweep.py) mirrors the per-kernel sweep scripts.
Sweeps **block shape × hardware accelerator counts**, parses the simulator
report, and emits `parametric_sweep.csv` + `parametric_sweep.png`.

```
# default: 5 NAFNet32 shapes × default HW (mat=2,vec=4)
/home/why/anaconda3/bin/python3 nafblock/parametric_sweep.py

# multi-HW: 5 shapes × 3 HW configs (15 points, ~22 s wall)
/home/why/anaconda3/bin/python3 nafblock/parametric_sweep.py --hw "1:2,2:4,4:8"

# custom shapes (CxH list, H is also used for W)
/home/why/anaconda3/bin/python3 nafblock/parametric_sweep.py --shapes "32x64,64x32"
```

HW changes recompile via `EXTRA_CXXFLAGS=-DMAT_ACCEL_COUNT=...,VEC_ACCEL_COUNT=...`;
shape changes only re-run the binary. The Makefile passes `EXTRA_CXXFLAGS`
through every compile rule (added with the sweep script).

Plot has two panels: total elapsed cycles vs block shape (one line per HW),
and per-backend cycle breakdown stacked bars for the first HW configuration.
The CSV also records per-backend cycle / request totals so other plots are
straightforward to build from the same data.

### Integration note
The 14-layer composition is exposed through
`append_nafblock_layers(std::vector<NafBlockLayerDesc> &, int &id, const char *prefix, int C, int H, int W)`
in `nafblock_layers.h`. A future NafNet driver can loop this helper across
encoder / middle / decoder levels — no changes to the kernels or the bridge
are needed.