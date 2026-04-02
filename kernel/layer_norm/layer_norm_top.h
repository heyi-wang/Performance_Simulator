#pragma once

#include "accelerator.h"
#include "accelerator_pool.h"
#include "interconnect.h"
#include "layer_norm_config.h"
#include "memory.h"

#include <vector>

struct LayerNormWorker;

struct LnStepStats
{
    uint64_t vec_reqs = 0;
    uint64_t accel_cycles = 0;
    uint64_t scalar_cycles = 0;
    uint64_t wait_cycles = 0;
    uint64_t mem_cycles = 0;
    uint64_t rd_bytes = 0;
    uint64_t wr_bytes = 0;
};

struct LayerNormSimulationStats
{
    LnStepStats steps[4];
    uint64_t total_vec_reqs = 0;
    uint64_t total_rd_bytes = 0;
    uint64_t total_wr_bytes = 0;
    uint64_t max_elapsed_cycles = 0;
    uint64_t expected_vec_reqs = 0;
    uint64_t vec_acc_reqs = 0;
    uint64_t vec_acc_busy_cycles = 0;
    uint64_t vec_acc_queue_wait_cycles = 0;
    uint64_t memory_reqs = 0;
    uint64_t memory_busy_cycles = 0;
    uint64_t memory_queue_wait_cycles = 0;
    double vec_util = 0.0;
    bool verification_passed = false;
};

struct LayerNormTop : sc_module
{
    AcceleratorTLM  mat_acc;
    AcceleratorPool vec_acc;
    Interconnect    noc;
    Memory          memory;

    std::vector<LayerNormWorker *> workers;

    SC_HAS_PROCESS(LayerNormTop);
    explicit LayerNormTop(sc_module_name name);
    ~LayerNormTop() override;

    LayerNormSimulationStats collect_stats() const;
};
