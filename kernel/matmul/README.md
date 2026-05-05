# Matmul Performance Simulator

SystemC TLM-2.0 cycle estimator for a K-split GEMM kernel running on the modeled accelerator architecture (matrix-accelerator pool + vector-accelerator pool + L1/DMA memory). The top module is [matmul_top.cpp](matmul_top.cpp).

Each worker thread owns a slice of the K-dimension, runs its local matmul through the matrix-accelerator pool, then participates in a tree reduction and final quantization through the vector-accelerator pool driven by [accum_coordinator.cpp](accum_coordinator.cpp).

## Build

From the repository root:

```bash
make kernel-matmul              # build only the matmul simulator
make kernels                    # build every kernel simulator
```

Or from [kernel/](../):

```bash
make -C kernel matmul
```

The binary is written to [kernel/build/matmul_sim](../build/matmul_sim). Clean with `make -C kernel clean`.

Toolchain expectations (see [kernel/Makefile](../Makefile)):

- `g++` with `-std=c++17`
- SystemC headers in `/usr/include`, library in `/usr/lib/x86_64-linux-gnu`
- Override include/lib paths or add macros via `EXTRA_CXXFLAGS`

## Run

```bash
./kernel/build/matmul_sim [--threads N] [--accum-registers N] \
                          [--gemm-m M] [--gemm-k K] [--gemm-n N]
```

Runtime flags (parsed in [main.cpp:22-83](main.cpp#L22-L83)):

| Flag | Meaning | Default |
|------|---------|---------|
| `--threads N` | Number of K-split worker threads | `32` |
| `--accum-registers N` | C-tile accumulator registers per worker (controls M-batching) | `4` |
| `--gemm-m M` | Output rows | `128*128 = 16384` |
| `--gemm-k K` | Inner product extent | `64*3*3 = 576` |
| `--gemm-n N` | Output columns | `512` |

Exit codes: `0` pass, `1` bad argument, `2` verification mismatch (req counts or byte counts disagree with the analytical model).

Parameter sweeps are driven by the Python scripts in this directory ([sweep_workers.py](sweep_workers.py), [hardware_sweep.py](hardware_sweep.py), [parametric_sweep.py](parametric_sweep.py)); see [Parametric_Sweep_HOWTO.md](Parametric_Sweep_HOWTO.md).

## Compile-time Configuration

Hardware knobs are `#ifndef`-guarded macros — pass them through `EXTRA_CXXFLAGS` to override without editing headers. For example:

```bash
make -C kernel matmul EXTRA_CXXFLAGS="-DMAT_ACCEL_COUNT=8 -DVEC_ACCEL_COUNT=16 -DMATMUL_M=16"
```

### Accelerator counts and queues — [config/hardware_config.h](../../config/hardware_config.h)

| Macro | Default | Effect |
|-------|---------|--------|
| `MAT_ACCEL_COUNT` | `4` | Matrix-accelerator instances behind the shared pool |
| `VEC_ACCEL_COUNT` | `8` | Vector-accelerator instances behind the shared pool |
| `ACC_QUEUE_DEPTH` | `32` | Lower bound for the per-pool admission queue (also expanded to `4 × accel_count`) |
| `MEMORY_PARALLEL_SLOTS` | `MAT+VEC` | Concurrent memory slots (used as a default; matmul overrides via `l1_slots`/`dma_slots`) |

### Matrix tile and timing — [config/hardware_config.h](../../config/hardware_config.h)

| Macro | Default | Effect |
|-------|---------|--------|
| `MATMUL_M` | `8` | Rows per matrix-accelerator tile |
| `MATMUL_K` | `8` | K extent per matrix-accelerator tile |
| `MATMUL_N` | `8` | Columns per matrix-accelerator tile |
| `MATMUL_ACC_CYCLE` | `1` | Cycles per matrix-accelerator request |
| `MAT_SCALAR_OVERHEAD` | `25` | Scalar dispatch cost per matrix request |

### Vector unit and post-mat work — [config/hardware_config.h](../../config/hardware_config.h) / [matmul_config.h](matmul_config.h)

| Macro | Default | Effect |
|-------|---------|--------|
| `VECTOR_ACC_CAP` | `64` | Vector datapath width in bytes/request |
| `VECTOR_INSN_CYCLE` | `1` | Cycles per vector instruction |
| `VEC_SCALAR_OVERHEAD` | `8` | Scalar dispatch cost per vector request |
| `MATMUL_ACCUM_VEC_INSNS` | `1` | Vector instructions per accumulation request |
| `MATMUL_QUANT_VEC_INSNS` | `7` | Vector instructions per final-quantization request |

### Memory model — [config/hardware_config.h](../../config/hardware_config.h)

| Macro | Default | Effect |
|-------|---------|--------|
| `MEMORY_BASE_LAT` | `1` | Base memory latency (cycles) |
| `MEMORY_BYTES_PER_CYCLE` | `64` | Baseline memory bandwidth |
| `MATMUL_MEMORY_BYTES_PER_CYCLE` | `2 × MEMORY_BYTES_PER_CYCLE` | Bandwidth used by matmul memory |

L1/DMA split (latency, bandwidth, parallel slots) is set programmatically in [matmul_top.h:39-44](matmul_top.h#L39-L44) on `MatmulRuntimeConfig`. Override by editing those defaults or instantiating `MatmulRuntimeConfig` from your own driver — they are not exposed as `-D` macros.

### DMA descriptor scalar cost — [config/hardware_config.h](../../config/hardware_config.h)

| Macro | Default | Effect |
|-------|---------|--------|
| `DMA_A_ROW_SCALAR` | `10` | Scalar instructions per A-row DMA descriptor |
| `DMA_B_ROW_SCALAR` | `20` | Scalar instructions per B-row DMA descriptor |
| `DMA_C_ROW_SCALAR` | `10` | Scalar instructions per C-row DMA descriptor |

### Workload defaults — [matmul_config.h](matmul_config.h)

GEMM dimensions default to a conv-style mapping (`N·H·W × C_in·KH·KW × C_out` = `16384 × 576 × 512`). Override the GEMM shape at runtime via `--gemm-m / --gemm-k / --gemm-n`; the underlying conv-style fields are not separately exposed on the CLI.

## Example

```bash
# 16 workers, 8-tile accumulator window, custom GEMM, beefier matrix pool
make -C kernel matmul EXTRA_CXXFLAGS="-DMAT_ACCEL_COUNT=8"
./kernel/build/matmul_sim --threads 16 --accum-registers 8 \
                          --gemm-m 4096 --gemm-k 1024 --gemm-n 512
```

The report covers simulation info, hardware configuration, per-worker breakdown, per-accelerator utilization, memory traffic, and a verification block (`PASS` / `FAIL`).
