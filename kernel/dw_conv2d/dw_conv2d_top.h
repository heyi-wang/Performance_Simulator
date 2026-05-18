#pragma once

#include "accelerator.h"
#include "accelerator_pool.h"
#include "common.h"
#include "dw_conv2d_config.h"
#include "interconnect.h"
#include "memory.h"
#include "worker.h"

#include <algorithm>
#include <memory>
#include <iosfwd>
#include <systemc>
#include <vector>

struct DwConvRuntimeConfig
{
    uint64_t input_elem_bytes = DW_INPUT_ELEM_BYTES;
    uint64_t output_elem_bytes = DW_OUTPUT_ELEM_BYTES;
    int channels = DW_C;
    int height = DW_H;
    int width = DW_W;
    int kernel_h = DW_KH;
    int kernel_w = DW_KW;
    int pad = DW_PAD;
    int stride = DW_STRIDE;
    int worker_count = DW_NUM_WORKERS;
    uint64_t vec_acc_cap = DW_VEC_ACC_CAP;
    uint64_t vec_acc_cycle = DW_VEC_ACC_CYCLE;
    int vec_acc_instances = DW_VEC_ACC_INSTANCES;
    uint64_t scalar_overhead = DW_SCALAR_OVERHEAD;
    uint64_t l1_base_lat = DW_L1_BASE_LAT_CFG;
    uint64_t l1_bw = DW_L1_BW_CFG;
    uint64_t l1_slots = DW_L1_SLOTS_CFG;
    uint64_t l2_base_lat = DW_L2_BASE_LAT_CFG;
    uint64_t l2_bw = DW_L2_BW_CFG;
    uint64_t l2_slots = DW_L2_SLOTS_CFG;
    size_t acc_queue_depth = DW_ACC_QUEUE_DEPTH;
    uint64_t dma_vec_rd_scalar = DW_DMA_VEC_RD_SCALAR_CFG;
    uint64_t dma_vec_wr_scalar = DW_DMA_VEC_WR_SCALAR_CFG;
    uint64_t max_inflight_vec_reqs = DW_MAX_INFLIGHT_VEC_REQS_CFG;
    uint64_t max_inflight_dma_writes = DW_MAX_INFLIGHT_DMA_WRITES_CFG;

    static DwConvRuntimeConfig defaults()
    {
        return DwConvRuntimeConfig{};
    }

    int out_h() const
    {
        return (height + 2 * pad - kernel_h) / stride + 1;
    }

    int out_w() const
    {
        return (width + 2 * pad - kernel_w) / stride + 1;
    }

    int strips_per_row() const
    {
        return static_cast<int>(
            ceil_div_u64(static_cast<uint64_t>(out_w()), vec_acc_cap));
    }

    int effective_vec_instances() const
    {
        return std::max(1, std::min(vec_acc_instances, worker_count));
    }
};

struct DwConvPostProcessor;

struct DwConvSimulationStats
{
    uint64_t total_vec_calls = 0;
    uint64_t total_rd_bytes = 0;
    uint64_t total_wr_bytes = 0;
    uint64_t total_wait_cycles = 0;
    uint64_t total_mem_cycles = 0;
    uint64_t max_elapsed_cycles = 0;
    uint64_t expected_vec_calls = 0;
    uint64_t expected_l1_read_bytes = 0;
    uint64_t expected_l1_write_bytes = 0;
    uint64_t expected_l1_reqs = 0;
    uint64_t expected_l2_read_bytes = 0;
    uint64_t expected_l2_write_bytes = 0;
    uint64_t expected_l2_dma_reqs = 0;
    uint64_t vec_acc_reqs = 0;
    uint64_t vec_acc_busy_cycles = 0;
    uint64_t vec_acc_occupied_cycles = 0;
    uint64_t vec_acc_queue_wait_cycles = 0;
    uint64_t l1_reqs = 0;
    uint64_t l1_read_bytes = 0;
    uint64_t l1_write_bytes = 0;
    uint64_t l1_busy_cycles = 0;
    uint64_t l1_queue_wait_cycles = 0;
    uint64_t l2_dma_reqs = 0;
    uint64_t l2_dma_read_bytes = 0;
    uint64_t l2_dma_write_bytes = 0;
    uint64_t l2_dma_busy_cycles = 0;
    uint64_t l2_dma_queue_wait_cycles = 0;
    // Back-compat aggregates used by nafnet bridge convert_stats(); populated
    // from the L2 DMA counters (matches pooling's convert_stats convention).
    uint64_t memory_reqs = 0;
    uint64_t memory_busy_cycles = 0;
    uint64_t memory_queue_wait_cycles = 0;
    double vec_util = 0.0;
    double vec_occupancy = 0.0;
    double l1_bw_observed = 0.0;
    double l2_bw_observed = 0.0;
    int slowest_worker_tid = -1;
    bool verification_passed = false;
};

struct DwConvTop : sc_module
{
    DwConvRuntimeConfig cfg;
    AcceleratorTLM  mat_acc;
    AcceleratorPool vec_acc;
    Interconnect    noc;
    L1L2Memory      memory;

    std::vector<Worker *> workers;
    std::vector<std::unique_ptr<DwConvPostProcessor>> post_processors;
    // Per-vec-unit strip-contiguity lock. A worker acquires its
    // assigned unit's semaphore before issuing the strip's first
    // sub-request and releases it after the last sub-request's
    // issue_end -- guaranteeing no other worker's strip interleaves
    // on the same unit (see Dw_Conv2d.md Requirements v2).
    std::vector<std::unique_ptr<sc_core::sc_semaphore>> per_unit_lock;
    sc_event *done_event = nullptr;
    std::unique_ptr<sc_fifo<int>> completion_fifo;

    SC_HAS_PROCESS(DwConvTop);
    explicit DwConvTop(sc_module_name name,
                       const DwConvRuntimeConfig &cfg_ =
                           DwConvRuntimeConfig::defaults(),
                       sc_event *start_event = nullptr,
                       sc_event *done_event_ = nullptr);
    ~DwConvTop() override;

    DwConvSimulationStats collect_stats() const;
    std::vector<KernelWorkerInfo> collect_worker_info() const;
    void print_report(std::ostream &os) const;
    void done_monitor();
};
