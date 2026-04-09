#pragma once

#include "memory.h"
#include "interconnect.h"
#include "accelerator_pool.h"
#include "worker.h"
#include "accum_coordinator.h"
#include "matmul_config.h"
#include <algorithm>
#include <memory>
#include <systemc>
#include <iosfwd>
#include <vector>

struct MatmulRuntimeConfig
{
    int thread_count = MatmulConfig::default_thread_count;
    int mat_accel_count = MAT_ACCEL_COUNT;
    int vec_accel_count = VEC_ACCEL_COUNT;

    uint64_t workload_n = MatmulConfig::workload_n;
    uint64_t workload_h = MatmulConfig::workload_h;
    uint64_t workload_w = MatmulConfig::workload_w;
    uint64_t workload_c_in = MatmulConfig::workload_c_in;
    uint64_t workload_kh = MatmulConfig::workload_kh;
    uint64_t workload_kw = MatmulConfig::workload_kw;
    uint64_t workload_c_out = MatmulConfig::workload_c_out;

    uint64_t mat_cycle = MATMUL_ACC_CYCLE;
    uint64_t vec_cycle = VECTOR_ACC_CYCLE;
    uint64_t scalar_overhead = SCALAR_OVERHEAD;
    uint64_t memory_base_lat = HW_MEMORY_BASE_LAT;
    uint64_t memory_bw = HW_MATMUL_MEMORY_BYTES_PER_CYCLE;
    uint64_t memory_parallel_slots = MEMORY_PARALLEL_SLOTS_CFG;
    size_t mat_queue_cap_value =
        std::max(HW_ACC_QUEUE_DEPTH,
                 static_cast<size_t>(MAT_ACCEL_COUNT * 4));
    size_t vec_queue_cap_value =
        std::max(HW_ACC_QUEUE_DEPTH,
                 static_cast<size_t>(VEC_ACCEL_COUNT * 4));

    uint64_t gemm_quant_in_elem_bytes = static_cast<uint64_t>(sizeof(int32_t));
    uint64_t gemm_quant_out_elem_bytes = static_cast<uint64_t>(sizeof(uint8_t));
    uint64_t gemm_accum_in_elem_bytes = static_cast<uint64_t>(sizeof(int32_t));
    uint64_t gemm_accum_out_elem_bytes = static_cast<uint64_t>(sizeof(int32_t));

    static MatmulRuntimeConfig defaults(int threads = MatmulConfig::default_thread_count,
                                        int mat_accels = MAT_ACCEL_COUNT,
                                        int vec_accels = VEC_ACCEL_COUNT)
    {
        MatmulRuntimeConfig cfg;
        cfg.thread_count = std::max(threads, 1);
        cfg.mat_accel_count = std::max(mat_accels, 1);
        cfg.vec_accel_count = std::max(vec_accels, 1);
        cfg.mat_queue_cap_value =
            std::max(HW_ACC_QUEUE_DEPTH,
                     static_cast<size_t>(cfg.mat_accel_count * 4));
        cfg.vec_queue_cap_value =
            std::max(HW_ACC_QUEUE_DEPTH,
                     static_cast<size_t>(cfg.vec_accel_count * 4));
        return cfg;
    }

    uint64_t gemm_m() const { return workload_n * workload_h * workload_w; }
    uint64_t gemm_k() const { return workload_c_in * workload_kh * workload_kw; }
    uint64_t gemm_n() const { return workload_c_out; }
    uint64_t gemm_a_bytes() const { return MATMUL_M * MATMUL_K * sizeof(int8_t); }
    uint64_t gemm_b_bytes() const { return MATMUL_K * MATMUL_N * sizeof(int8_t); }
    uint64_t gemm_c_bytes() const { return MATMUL_M * MATMUL_N * sizeof(int32_t); }
    uint64_t gemm_tile_m() const { return ceil_div_u64(gemm_m(), MATMUL_M); }
    uint64_t gemm_tile_n() const { return ceil_div_u64(gemm_n(), MATMUL_N); }
    uint64_t gemm_partial_elements() const { return gemm_m() * gemm_n(); }
    uint64_t gemm_accum_vec_calls() const
    {
        return ceil_div_u64(gemm_partial_elements(), VECTOR_ACC_CAP);
    }
    uint64_t gemm_quant_vec_calls() const
    {
        return ceil_div_u64(gemm_partial_elements(), VECTOR_ACC_CAP);
    }
    uint64_t gemm_accum_rd_bytes() const
    {
        return 2 * VECTOR_ACC_CAP * gemm_accum_in_elem_bytes;
    }
    uint64_t gemm_accum_wr_bytes() const
    {
        return VECTOR_ACC_CAP * gemm_accum_out_elem_bytes;
    }
    uint64_t gemm_quant_rd_bytes() const
    {
        return VECTOR_ACC_CAP * gemm_quant_in_elem_bytes;
    }
    uint64_t gemm_quant_wr_bytes() const
    {
        return VECTOR_ACC_CAP * gemm_quant_out_elem_bytes;
    }
    uint64_t gemm_k_per_thread() const
    {
        return ceil_div_u64(gemm_k(), static_cast<uint64_t>(thread_count));
    }
    uint64_t local_k_extent_for_thread(int tid) const
    {
        uint64_t k_begin = static_cast<uint64_t>(tid) * gemm_k_per_thread();
        uint64_t k_end = std::min<uint64_t>(gemm_k(), k_begin + gemm_k_per_thread());
        return (k_begin < gemm_k()) ? (k_end - k_begin) : 0;
    }
    uint64_t local_access_mat_for_thread(int tid) const
    {
        uint64_t local_k = local_k_extent_for_thread(tid);
        uint64_t local_tile_k = (local_k > 0) ? ceil_div_u64(local_k, MATMUL_K) : 0;
        return gemm_tile_m() * local_tile_k * gemm_tile_n();
    }
    int active_thread_count() const
    {
        int active = 0;
        for (int tid = 0; tid < thread_count; ++tid)
            active += (local_k_extent_for_thread(tid) > 0) ? 1 : 0;
        return active;
    }
    size_t mat_acc_queue_cap() const { return mat_queue_cap_value; }
    size_t vec_acc_queue_cap() const { return vec_queue_cap_value; }
};

struct MatmulSimulationStats
{
    uint64_t total_elapsed = 0;
    uint64_t max_mat_elapsed = 0;
    uint64_t accum_overhead = 0;
    uint64_t total_macs = 0;
    uint64_t total_rd_bytes = 0;
    uint64_t total_wr_bytes = 0;
    uint64_t mat_req_total = 0;
    uint64_t vec_req_total = 0;
    uint64_t expected_mat_req_total = 0;
    uint64_t expected_vec_req_total = 0;
    uint64_t expected_accum_pairs = 0;
    uint64_t mat_busy_total = 0;
    uint64_t mat_occupied_total = 0;
    uint64_t mat_qwait_total = 0;
    uint64_t vec_busy_total = 0;
    uint64_t vec_occupied_total = 0;
    uint64_t vec_qwait_total = 0;
    uint64_t memory_reqs = 0;
    uint64_t memory_busy_cycles = 0;
    uint64_t memory_queue_wait_cycles = 0;
    uint64_t total_worker_stall = 0;
    uint64_t coordinator_stall = 0;
    uint64_t coordinator_compute = 0;
    double gflops = 0.0;
    double mat_occupancy = 0.0;
    double mat_compute_util = 0.0;
    double vec_occupancy = 0.0;
    double vec_compute_util = 0.0;
    double mem_bw = 0.0;
    bool verification_passed = false;
};

// ============================================================
// MatmulTop — top-level module for the K-split GEMM simulator.
//
// Topology (same hardware as Top, different workload):
//
//   workers[0..T-1]  ──┐
//   passive coordinator│
//          state       ├──► noc ──► mat_acc ──► memory
//                      │       └──► vec_acc ──► memory
//                      └── (mat_acc / vec_acc also reach memory via noc)
//
// Workers execute local matmul, worker-driven tree reduction,
// and worker-parallel final quantization. The AccumCoordinator
// remains as passive shared state and statistics storage.
// ============================================================
struct MatmulTop : sc_module
{
    MatmulRuntimeConfig cfg;
    AcceleratorPool mat_acc;
    AcceleratorPool vec_acc;
    Interconnect    noc;
    Memory          memory;

    std::vector<Worker *> workers;
    std::vector<Worker *> active_workers;
    AccumCoordinator     *coordinator = nullptr;
    sc_event *done_event = nullptr;
    std::unique_ptr<sc_fifo<int>> completion_fifo;

    SC_HAS_PROCESS(MatmulTop);
    MatmulTop(sc_module_name nm,
              const MatmulRuntimeConfig &cfg_,
              sc_event *start_event = nullptr,
              sc_event *done_event_ = nullptr);
    ~MatmulTop() override;

    MatmulSimulationStats collect_stats() const;
    std::vector<KernelWorkerInfo> collect_worker_info() const;
    void print_report(std::ostream &os) const;
    void done_monitor();
};
