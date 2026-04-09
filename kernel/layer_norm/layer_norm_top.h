#pragma once

#include "accelerator.h"
#include "accelerator_pool.h"
#include "common.h"
#include "interconnect.h"
#include "layer_norm_config.h"
#include "memory.h"

#include <memory>
#include <iosfwd>
#include <systemc>
#include <vector>

struct LayerNormRuntimeConfig
{
    int channels = LN_C;
    int height = LN_H;
    int width = LN_W;
    int worker_count = LN_NUM_WORKERS;
    uint64_t vec_acc_cap = LN_VEC_ACC_CAP;
    uint64_t vec_acc_cycle = LN_VEC_ACC_CYCLE;
    int vec_acc_instances = LN_VEC_ACC_INSTANCES;
    uint64_t scalar_overhead = LN_SCALAR_OVERHEAD;
    uint64_t step3_cycles = LN_STEP3_CYCLES;
    uint64_t memory_base_lat = LN_MEM_BASE_LAT;
    uint64_t memory_bw = LN_MEM_BW;
    size_t acc_queue_depth = LN_ACC_QUEUE_DEPTH;

    static LayerNormRuntimeConfig defaults()
    {
        return LayerNormRuntimeConfig{};
    }

    int spatial() const { return height * width; }
    int tile_count() const
    {
        return static_cast<int>(
            ceil_div_u64(static_cast<uint64_t>(spatial()), vec_acc_cap));
    }
};

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
    uint64_t vec_acc_occupied_cycles = 0;
    uint64_t vec_acc_queue_wait_cycles = 0;
    uint64_t memory_reqs = 0;
    uint64_t memory_busy_cycles = 0;
    uint64_t memory_queue_wait_cycles = 0;
    double vec_util = 0.0;
    double vec_occupancy = 0.0;
    bool verification_passed = false;
};

struct LayerNormTop : sc_module
{
    LayerNormRuntimeConfig cfg;
    AcceleratorTLM  mat_acc;
    AcceleratorPool vec_acc;
    Interconnect    noc;
    Memory          memory;

    std::vector<LayerNormWorker *> workers;
    sc_event *done_event = nullptr;
    std::unique_ptr<sc_fifo<int>> completion_fifo;

    SC_HAS_PROCESS(LayerNormTop);
    explicit LayerNormTop(sc_module_name name,
                          const LayerNormRuntimeConfig &cfg_ =
                              LayerNormRuntimeConfig::defaults(),
                          sc_event *start_event = nullptr,
                          sc_event *done_event_ = nullptr);
    ~LayerNormTop() override;

    LayerNormSimulationStats collect_stats() const;
    std::vector<KernelWorkerInfo> collect_worker_info() const;
    void print_report(std::ostream &os) const;
    void done_monitor();
};
