#pragma once
#include "hardware_config.h"
#include "common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

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

struct MatmulConfig
{
    static constexpr int default_thread_count = 32;

    int thread_count = default_thread_count;
    int mat_accel_count = MAT_ACCEL_COUNT;
    int vec_accel_count = VEC_ACCEL_COUNT;

    // Default workload mirrors the previous root conv-style GEMM mapping.
    static constexpr uint64_t workload_n = 1;
    static constexpr uint64_t workload_h = 128;
    static constexpr uint64_t workload_w = 128;
    static constexpr uint64_t workload_c_in = 64;
    static constexpr uint64_t workload_kh = 3;
    static constexpr uint64_t workload_kw = 3;
    static constexpr uint64_t workload_c_out = 512;

    // Full output matrix dimensions (M is NOT divided across threads for K-split).
    static constexpr uint64_t gemm_m = workload_n * workload_h * workload_w;
    static constexpr uint64_t gemm_k = workload_c_in * workload_kh * workload_kw;
    static constexpr uint64_t gemm_n = workload_c_out;

    // Number of accelerator tiles along M and N are independent of thread count.
    // Memory traffic per tile (bytes)
    static constexpr uint64_t gemm_a_bytes = MATMUL_M * MATMUL_K * sizeof(int8_t);
    static constexpr uint64_t gemm_b_bytes = MATMUL_K * MATMUL_N * sizeof(int8_t);
    static constexpr uint64_t gemm_c_bytes = MATMUL_M * MATMUL_N * sizeof(int32_t);

    // Reduction and final quantization traffic are per full output matrix.
    // Every worker owns a full [M x N] partial, but only the coordinator
    // quantizes once after the final reduction result is ready.
    static constexpr uint64_t gemm_partial_elements = gemm_m * gemm_n;
    static constexpr uint64_t gemm_quant_in_elem_bytes =
        static_cast<uint64_t>(sizeof(int32_t));
    static constexpr uint64_t gemm_quant_out_elem_bytes =
        static_cast<uint64_t>(sizeof(uint8_t));
    static constexpr uint64_t gemm_accum_in_elem_bytes = gemm_quant_in_elem_bytes;
    static constexpr uint64_t gemm_accum_out_elem_bytes = gemm_quant_in_elem_bytes;

    static constexpr uint64_t gemm_accum_rd_bytes =
        2 * VECTOR_ACC_CAP * gemm_accum_in_elem_bytes;
    static constexpr uint64_t gemm_accum_wr_bytes =
        VECTOR_ACC_CAP * gemm_accum_out_elem_bytes;
    static constexpr uint64_t gemm_quant_rd_bytes =
        VECTOR_ACC_CAP * gemm_quant_in_elem_bytes;
    static constexpr uint64_t gemm_quant_wr_bytes =
        VECTOR_ACC_CAP * gemm_quant_out_elem_bytes;

    explicit MatmulConfig(int threads = default_thread_count,
                          int mat_accels = MAT_ACCEL_COUNT,
                          int vec_accels = VEC_ACCEL_COUNT)
        : thread_count(std::max(threads, 1)),
          mat_accel_count(std::max(mat_accels, 1)),
          vec_accel_count(std::max(vec_accels, 1))
    {
    }

    static uint64_t gemm_tile_m()
    {
        return ceil_div_u64(gemm_m, MATMUL_M);
    }

    static uint64_t gemm_tile_n()
    {
        return ceil_div_u64(gemm_n, MATMUL_N);
    }

    static uint64_t gemm_accum_vec_calls()
    {
        return ceil_div_u64(gemm_partial_elements, VECTOR_ACC_CAP);
    }

    static uint64_t gemm_quant_vec_calls()
    {
        return ceil_div_u64(gemm_partial_elements, VECTOR_ACC_CAP);
    }

    uint64_t gemm_k_per_thread() const
    {
        return ceil_div_u64(gemm_k, static_cast<uint64_t>(thread_count));
    }

    uint64_t gemm_tile_k() const
    {
        return ceil_div_u64(gemm_k_per_thread(), MATMUL_K);
    }

    uint64_t gemm_access_mat() const
    {
        return gemm_tile_m() * gemm_tile_k() * gemm_tile_n();
    }

    size_t mat_acc_queue_cap() const
    {
        return std::max(HW_ACC_QUEUE_DEPTH,
                        static_cast<size_t>(mat_accel_count * 4));
    }

    size_t vec_acc_queue_cap() const
    {
        return std::max(HW_ACC_QUEUE_DEPTH,
                        static_cast<size_t>(vec_accel_count * 4));
    }

    uint64_t local_access_mat_for_thread(int tid) const
    {
        uint64_t k_begin = static_cast<uint64_t>(tid) * gemm_k_per_thread();
        uint64_t k_end = std::min<uint64_t>(gemm_k, k_begin + gemm_k_per_thread());
        uint64_t local_k = (k_begin < gemm_k) ? (k_end - k_begin) : 0;
        uint64_t local_tile_k = (local_k > 0) ? ceil_div_u64(local_k, MATMUL_K) : 0;
        return gemm_tile_m() * local_tile_k * gemm_tile_n();
    }
};
