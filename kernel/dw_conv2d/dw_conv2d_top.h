#pragma once

#include "accelerator.h"
#include "accelerator_pool.h"
#include "common.h"
#include "dw_conv2d_config.h"
#include "interconnect.h"
#include "memory.h"

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
    uint64_t memory_base_lat = DW_MEM_BASE_LAT;
    uint64_t memory_bw = DW_MEM_BW;
    size_t acc_queue_depth = DW_ACC_QUEUE_DEPTH;

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
};

struct DwConvWorker;

struct DwConvSimulationStats
{
    uint64_t total_vec_calls = 0;
    uint64_t total_rd_bytes = 0;
    uint64_t total_wr_bytes = 0;
    uint64_t total_wait_cycles = 0;
    uint64_t total_mem_cycles = 0;
    uint64_t max_elapsed_cycles = 0;
    uint64_t expected_vec_calls = 0;
    uint64_t vec_acc_reqs = 0;
    uint64_t vec_acc_busy_cycles = 0;
    uint64_t vec_acc_occupied_cycles = 0;
    uint64_t vec_acc_queue_wait_cycles = 0;
    uint64_t memory_reqs = 0;
    uint64_t memory_busy_cycles = 0;
    uint64_t memory_queue_wait_cycles = 0;
    double vec_util = 0.0;
    double vec_occupancy = 0.0;
    double mem_bw = 0.0;
    bool verification_passed = false;
};

struct DwConvTop : sc_module
{
    DwConvRuntimeConfig cfg;
    AcceleratorTLM  mat_acc;
    AcceleratorPool vec_acc;
    Interconnect    noc;
    Memory          memory;

    std::vector<DwConvWorker *> workers;
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
