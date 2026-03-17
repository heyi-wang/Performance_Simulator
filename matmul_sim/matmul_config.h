#pragma once
#include "config.h"
#include "src/common.h"

// ============================================================
// K-split GEMM configuration
//
// Unlike the base config (which splits M across threads),
// here all threads share the same output shape [GEMM_M x GEMM_N]
// but each thread handles a K-slice of the inner product.
//
//   Thread i computes: A[:, k_i : k_i+K_per_thread] × B[k_i : k_i+K_per_thread, :]
//   Producing a partial result of shape [GEMM_M x GEMM_N]
//   Final result = sum of all partial results  (K-split accumulation)
// ============================================================

// Full output matrix dimensions (M is NOT divided across threads for K-split).
static const uint64_t GEMM_M = CONV_N * CONV_H_OUT * CONV_W_OUT;
static const uint64_t GEMM_K = A_K;   // = CONV_C_IN * CONV_KH * CONV_KW
static const uint64_t GEMM_N = B_N;   // = CONV_C_OUT

// K sliced per thread (ceiling division so last thread may get less work)
static const uint64_t GEMM_K_PER_THREAD = ceil_div_u64(GEMM_K, (uint64_t)NUM_THREADS);

// Number of accelerator tiles each thread issues for its K-slice
static const uint64_t GEMM_TILE_M = ceil_div_u64(GEMM_M,             MATMUL_M);
static const uint64_t GEMM_TILE_K = ceil_div_u64(GEMM_K_PER_THREAD,  MATMUL_K);
static const uint64_t GEMM_TILE_N = ceil_div_u64(GEMM_N,             MATMUL_N);
static const uint64_t GEMM_ACCESS_MAT = GEMM_TILE_M * GEMM_TILE_K * GEMM_TILE_N;

// Memory traffic per tile (bytes)
static const uint64_t GEMM_A_BYTES = MATMUL_M * MATMUL_K * sizeof(int8_t);
static const uint64_t GEMM_B_BYTES = MATMUL_K * MATMUL_N * sizeof(int8_t);
static const uint64_t GEMM_C_BYTES = MATMUL_M * MATMUL_N * sizeof(int32_t);

// ============================================================
// Accumulation phase (tree reduction)
//
// Each pairwise accumulation sums two partial [GEMM_M x GEMM_N]
// matrices element-wise using the VectorAccelerator.
//
//   Number of vec_acc calls per pairwise accumulation
//     = ceil(GEMM_M * GEMM_N / VECTOR_ACC_CAP)
//   Memory per accumulation: read 2 partials, write 1 result
// ============================================================
static const uint64_t GEMM_PARTIAL_ELEMENTS = GEMM_M * GEMM_N;
static const uint64_t GEMM_ACCUM_VEC_CALLS  = ceil_div_u64(GEMM_PARTIAL_ELEMENTS, VECTOR_ACC_CAP);
static const uint64_t GEMM_QUANT_IN_ELEM_BYTES =
    (uint64_t)sizeof(int32_t);
static const uint64_t GEMM_QUANT_OUT_ELEM_BYTES =
    (uint64_t)sizeof(uint8_t);
static const uint64_t GEMM_ACCUM_IN_ELEM_BYTES  = GEMM_QUANT_OUT_ELEM_BYTES;
static const uint64_t GEMM_ACCUM_OUT_ELEM_BYTES = GEMM_QUANT_OUT_ELEM_BYTES;

// Read 2 partial matrices + write 1 result per vec_acc call
static const uint64_t GEMM_ACCUM_RD_BYTES =
    2 * VECTOR_ACC_CAP * GEMM_ACCUM_IN_ELEM_BYTES;
static const uint64_t GEMM_ACCUM_WR_BYTES =
    VECTOR_ACC_CAP * GEMM_ACCUM_OUT_ELEM_BYTES;

// ============================================================
// Per-worker output quantization
//
// After completing all mat tiles, each worker issues
// GEMM_QUANT_VEC_CALLS vec_acc requests to quantize its
// partial result [GEMM_M x GEMM_N] from fp32 → fp16.
// Only after quantization does it fire mat_done_ev so the
// AccumCoordinator can begin tree-reduction accumulation.
// ============================================================
static const uint64_t GEMM_QUANT_VEC_CALLS =
    ceil_div_u64(GEMM_PARTIAL_ELEMENTS, VECTOR_ACC_CAP);
static const uint64_t GEMM_QUANT_RD_BYTES =
    VECTOR_ACC_CAP * GEMM_QUANT_IN_ELEM_BYTES;       // read fp32 partial result
static const uint64_t GEMM_QUANT_WR_BYTES =
    VECTOR_ACC_CAP * GEMM_QUANT_OUT_ELEM_BYTES;      // write fp16 quantized result

// ============================================================
// Configurable accelerator queue depths
//
// Reduce VEC_ACC_QUEUE_CAP below NUM_THREADS to observe
// backpressure: some workers will stall during quantization.
// Set to NUM_THREADS for no stalls.
// ============================================================
static const size_t MAT_ACC_QUEUE_CAP =
    static_cast<size_t>(NUM_THREADS);
static const size_t VEC_ACC_QUEUE_CAP =
    static_cast<size_t>(NUM_THREADS >= 2 ? NUM_THREADS / 2 : 1);
