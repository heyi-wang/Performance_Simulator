#pragma once

#include "accelerator.h"
#include "accelerator_pool.h"
#include "common.h"
#include "interconnect.h"
#include "memory.h"
#include "pooling_config.h"

#include <memory>
#include <iosfwd>
#include <systemc>
#include <vector>

struct PoolRuntimeConfig
{
    uint64_t input_elem_bytes = POOL_INPUT_ELEM_BYTES;
    uint64_t output_elem_bytes = POOL_OUTPUT_ELEM_BYTES;
    int channels = POOL_C;
    int height = POOL_H;
    int width = POOL_W;
    int worker_count = POOL_NUM_WORKERS;
    uint64_t vec_acc_cap = POOL_VEC_ACC_CAP;
    uint64_t vec_acc_cycle = POOL_VEC_ACC_CYCLE;
    int vec_acc_instances = POOL_VEC_ACC_INSTANCES;
    uint64_t scalar_overhead = POOL_SCALAR_OVERHEAD;
    uint64_t divide_cycles = POOL_DIVIDE_CYCLES;
    uint64_t memory_base_lat = POOL_MEM_BASE_LAT;
    uint64_t memory_bw = POOL_MEM_BW;
    size_t acc_queue_depth = POOL_ACC_QUEUE_DEPTH;

    static PoolRuntimeConfig defaults()
    {
        return PoolRuntimeConfig{};
    }

    int spatial() const { return height * width; }
    int tile_count() const
    {
        return static_cast<int>(
            ceil_div_u64(static_cast<uint64_t>(spatial()), vec_acc_cap));
    }
};

struct PoolWorker;

struct PoolSimulationStats
{
    uint64_t total_vec_calls = 0;
    uint64_t total_rd_bytes = 0;
    uint64_t total_wr_bytes = 0;
    uint64_t total_wait_cycles = 0;
    uint64_t total_mem_cycles = 0;
    uint64_t max_elapsed_cycles = 0;
    uint64_t expected_vec_calls = 0;
    uint64_t expected_rd_bytes = 0;
    uint64_t expected_wr_bytes = 0;
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

struct PoolTop : sc_module
{
    PoolRuntimeConfig cfg;
    AcceleratorTLM  mat_acc;
    AcceleratorPool vec_acc;
    Interconnect    noc;
    Memory          memory;

    std::vector<PoolWorker *> workers;
    sc_event *done_event = nullptr;
    std::unique_ptr<sc_fifo<int>> completion_fifo;

    SC_HAS_PROCESS(PoolTop);
    explicit PoolTop(sc_module_name name,
                     const PoolRuntimeConfig &cfg_ =
                         PoolRuntimeConfig::defaults(),
                     sc_event *start_event = nullptr,
                     sc_event *done_event_ = nullptr);
    ~PoolTop() override;

    PoolSimulationStats collect_stats() const;
    std::vector<KernelWorkerInfo> collect_worker_info() const;
    void print_report(std::ostream &os) const;
    void done_monitor();
};
