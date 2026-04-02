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
#define ACC_QUEUE_DEPTH 32
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
static const uint64_t MATMUL_M = 8;
static const uint64_t MATMUL_K = 8;
static const uint64_t MATMUL_N = 8;
static const uint64_t MATMUL_ACC_CYCLE = 1;

// Vector accelerator throughput and timing.
static const uint64_t VECTOR_ACC_CAP = 64;
static const uint64_t VECTOR_ACC_CYCLE = 1;

// Scalar dispatch overhead per accelerator request.
static const uint64_t SCALAR_OVERHEAD = 16;
