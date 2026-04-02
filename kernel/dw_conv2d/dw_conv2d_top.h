#pragma once

#include "accelerator.h"
#include "accelerator_pool.h"
#include "dw_conv2d_config.h"
#include "interconnect.h"
#include "memory.h"

#include <vector>

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
    AcceleratorTLM  mat_acc;
    AcceleratorPool vec_acc;
    Interconnect    noc;
    Memory          memory;

    std::vector<DwConvWorker *> workers;

    SC_HAS_PROCESS(DwConvTop);
    explicit DwConvTop(sc_module_name name);
    ~DwConvTop() override;

    DwConvSimulationStats collect_stats() const;
};
