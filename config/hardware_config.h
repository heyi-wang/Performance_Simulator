#pragma once

#include <cstddef>
#include <cstdint>

// ============================================================
// Shared hardware configuration for reusable simulator blocks.
//
// Kernel-local config headers should keep workload geometry and
// derive shared hardware properties from the values below.
// ============================================================

#ifndef MAT_ACCEL_COUNT
#define MAT_ACCEL_COUNT 2
#endif

#ifndef VEC_ACCEL_COUNT
#define VEC_ACCEL_COUNT 4
#endif

#ifndef MEMORY_PARALLEL_SLOTS
#define MEMORY_PARALLEL_SLOTS (MAT_ACCEL_COUNT + VEC_ACCEL_COUNT)
#endif

#ifndef MEMORY_BASE_LAT
#define MEMORY_BASE_LAT 1
#endif

#ifndef MEMORY_BYTES_PER_CYCLE
#define MEMORY_BYTES_PER_CYCLE 64
#endif

#ifndef MATMUL_MEMORY_BYTES_PER_CYCLE
#define MATMUL_MEMORY_BYTES_PER_CYCLE (MEMORY_BYTES_PER_CYCLE * 2)
#endif

#ifndef DW_MEMORY_BYTES_PER_CYCLE
#define DW_MEMORY_BYTES_PER_CYCLE (MEMORY_BYTES_PER_CYCLE * 4)
#endif

#ifndef ACC_QUEUE_DEPTH
#define ACC_QUEUE_DEPTH 64
#endif

static const int MAT_ACCEL_COUNT_CFG = MAT_ACCEL_COUNT;
static const int VEC_ACCEL_COUNT_CFG = VEC_ACCEL_COUNT;
static const int MEMORY_PARALLEL_SLOTS_CFG = MEMORY_PARALLEL_SLOTS;

static const uint64_t HW_MEMORY_BASE_LAT = MEMORY_BASE_LAT;
static const uint64_t HW_MEMORY_BYTES_PER_CYCLE = MEMORY_BYTES_PER_CYCLE;
static const uint64_t HW_MATMUL_MEMORY_BYTES_PER_CYCLE =
    MATMUL_MEMORY_BYTES_PER_CYCLE;
static const uint64_t HW_DW_MEMORY_BYTES_PER_CYCLE =
    DW_MEMORY_BYTES_PER_CYCLE;
static const size_t HW_ACC_QUEUE_DEPTH = ACC_QUEUE_DEPTH;

// Matrix accelerator tile geometry and timing.
#ifndef MATMUL_M
#define MATMUL_M 8
#endif
#ifndef MATMUL_K
#define MATMUL_K 8
#endif
#ifndef MATMUL_N
#define MATMUL_N 8
#endif
#ifndef MATMUL_ACC_CYCLE
#define MATMUL_ACC_CYCLE 1
#endif

// Vector accelerator throughput (in bytes per cycle) and timing.
#ifndef VECTOR_ACC_CAP
#define VECTOR_ACC_CAP 64
#endif
#ifndef VECTOR_INSN_CYCLE
#define VECTOR_INSN_CYCLE 1
#endif

static const uint64_t HW_VECTOR_INSN_CYCLE = VECTOR_INSN_CYCLE;
// Compatibility alias for kernels that still model a flat per-request vector
// latency instead of task-type-specific instruction counts.
static const uint64_t VECTOR_ACC_CYCLE = HW_VECTOR_INSN_CYCLE;

// Scalar dispatch overhead per accelerator request.
#ifndef MAT_SCALAR_OVERHEAD
#define MAT_SCALAR_OVERHEAD 10
#endif
#ifndef VEC_SCALAR_OVERHEAD
#define VEC_SCALAR_OVERHEAD 8
#endif

static const uint64_t HW_MAT_SCALAR_OVERHEAD = MAT_SCALAR_OVERHEAD;
static const uint64_t HW_VEC_SCALAR_OVERHEAD = VEC_SCALAR_OVERHEAD;

// Scalar instruction cost for DMA descriptor setup of a matrix-tile load.
// One `dma.x` carries the whole A or B tile in a single descriptor, so the
// cost is charged once per tile DMA. C-tile writes still emit one DMA per
// row, hence the per-row scalar below.
#ifndef DMA_A_TILE_SCALAR
#define DMA_A_TILE_SCALAR 2
#endif
#ifndef DMA_B_TILE_SCALAR
#define DMA_B_TILE_SCALAR 4
#endif
#ifndef DMA_C_ROW_SCALAR
#define DMA_C_ROW_SCALAR 3
#endif

static const uint64_t HW_DMA_A_TILE_SCALAR = DMA_A_TILE_SCALAR;
static const uint64_t HW_DMA_B_TILE_SCALAR = DMA_B_TILE_SCALAR;
static const uint64_t HW_DMA_C_ROW_SCALAR  = DMA_C_ROW_SCALAR;

// Per-DMA scalar instruction cost for vector-pipe DMA loads/stores.
// Used by the reduction/quantization phase, which streams one vector per DMA
// rather than assembling matrix-tile rows; the per-row matmul-phase model
// above does not apply.
#ifndef DMA_VEC_RD_SCALAR
#define DMA_VEC_RD_SCALAR 1
#endif
#ifndef DMA_VEC_WR_SCALAR
#define DMA_VEC_WR_SCALAR 1
#endif

static const uint64_t HW_DMA_VEC_RD_SCALAR = DMA_VEC_RD_SCALAR;
static const uint64_t HW_DMA_VEC_WR_SCALAR = DMA_VEC_WR_SCALAR;
