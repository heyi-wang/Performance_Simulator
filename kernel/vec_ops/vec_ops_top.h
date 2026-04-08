#pragma once

#include "accelerator.h"
#include "accelerator_pool.h"
#include "common.h"
#include "interconnect.h"
#include "memory.h"
#include "vec_ops_config.h"

#include <memory>
#include <systemc>
#include <vector>

struct VecOpsRuntimeConfig
{
    VopType op = VOP_SELECTED_OP;
    uint64_t elem_bytes = VOP_ELEM_BYTES;
    int channels = VOP_C;
    int height = VOP_H;
    int width = VOP_W;
    int worker_count = VOP_NUM_WORKERS;
    uint64_t vec_acc_cap = VOP_VEC_ACC_CAP;
    uint64_t vec_acc_cycle = VOP_VEC_ACC_CYCLE;
    int vec_acc_instances = VOP_VEC_ACC_INSTANCES;
    uint64_t scalar_overhead = VOP_SCALAR_OVERHEAD;
    uint64_t memory_base_lat = VOP_MEM_BASE_LAT;
    uint64_t memory_bw = VOP_MEM_BW;
    size_t acc_queue_depth = VOP_ACC_QUEUE_DEPTH;

    static VecOpsRuntimeConfig defaults()
    {
        return VecOpsRuntimeConfig{};
    }

    int spatial() const { return height * width; }
    uint64_t tile_cap() const
    {
        switch (vop_vec_shape(op))
        {
        case VopVecShape::E8M4:  return vec_acc_cap;
        case VopVecShape::E16M4: return (vec_acc_cap >= 2) ? (vec_acc_cap / 2) : 1;
        case VopVecShape::E32M4: return (vec_acc_cap >= 4) ? (vec_acc_cap / 4) : 1;
        case VopVecShape::E8M1:  return (vec_acc_cap >= 4) ? (vec_acc_cap / 4) : 1;
        }
        return vec_acc_cap;
    }
    int tile_count() const
    {
        return static_cast<int>(
            ceil_div_u64(static_cast<uint64_t>(spatial()), tile_cap()));
    }
};

struct VecOpsWorker;

struct VecOpsSimulationStats
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

struct VecOpsTop : sc_module
{
    VecOpsRuntimeConfig cfg;
    AcceleratorTLM  mat_acc;
    AcceleratorPool vec_acc;
    Interconnect    noc;
    Memory          memory;

    std::vector<VecOpsWorker *> workers;
    sc_event *done_event = nullptr;
    std::unique_ptr<sc_fifo<int>> completion_fifo;

    SC_HAS_PROCESS(VecOpsTop);
    explicit VecOpsTop(sc_module_name name,
                       const VecOpsRuntimeConfig &cfg_ =
                           VecOpsRuntimeConfig::defaults(),
                       sc_event *start_event = nullptr,
                       sc_event *done_event_ = nullptr);
    ~VecOpsTop() override;

    VecOpsSimulationStats collect_stats() const;
    void done_monitor();
};
