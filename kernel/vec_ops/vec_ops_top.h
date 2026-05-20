#pragma once

#include "accelerator.h"
#include "accelerator_pool.h"
#include "common.h"
#include "interconnect.h"
#include "memory.h"
#include "vec_ops_config.h"

#include <memory>
#include <iosfwd>
#include <systemc>
#include <vector>

struct VecOpsRuntimeConfig
{
    VopType op = VOP_SELECTED_OP;
    int channels = VOP_C;
    int height = VOP_H;
    int width = VOP_W;
    int worker_count = VOP_NUM_WORKERS;
    uint64_t vec_acc_cap = VOP_VEC_ACC_CAP;
    uint64_t vec_insn_cycle = VOP_VEC_INSN_CYCLE;
    int vec_acc_instances = VOP_VEC_ACC_INSTANCES;
    uint64_t scalar_overhead = VOP_SCALAR_OVERHEAD;
    uint64_t l1_base_lat = VOP_L1_BASE_LAT_CFG;
    uint64_t l1_bw = VOP_L1_BW_CFG;
    uint64_t l1_slots = VOP_L1_SLOTS_CFG;
    uint64_t l2_base_lat = VOP_L2_BASE_LAT_CFG;
    uint64_t l2_bw = VOP_L2_BW_CFG;
    uint64_t l2_slots = VOP_L2_SLOTS_CFG;
    size_t acc_queue_depth = VOP_ACC_QUEUE_DEPTH;
    uint64_t dma_vec_rd_scalar = VOP_DMA_VEC_RD_SCALAR_CFG;
    uint64_t dma_vec_wr_scalar = VOP_DMA_VEC_WR_SCALAR_CFG;
    uint64_t max_inflight_dma_writes = VOP_MAX_INFLIGHT_DMA_WRITES_CFG;

    static VecOpsRuntimeConfig defaults()
    {
        return VecOpsRuntimeConfig{};
    }

    int spatial() const { return height * width; }
    uint64_t tile_cap() const { return vop_tile_cap_elems(op); }
    int tile_count() const
    {
        return static_cast<int>(
            ceil_div_u64(static_cast<uint64_t>(spatial()), tile_cap()));
    }
    uint64_t service_cycles() const
    {
        return vop_insn_count(op) * vec_insn_cycle;
    }
};

struct VecOpsWorker;

struct VecOpsSimulationStats
{
    uint64_t total_vec_calls = 0;
    uint64_t total_rd_bytes = 0;        // L1 read bytes (vec-pipe)
    uint64_t total_wr_bytes = 0;        // L1 write bytes (vec-pipe)
    uint64_t total_wait_cycles = 0;
    uint64_t total_mem_cycles = 0;
    uint64_t max_elapsed_cycles = 0;

    // Expected (verification) counters
    uint64_t expected_vec_calls = 0;
    uint64_t expected_l1_read_bytes = 0;
    uint64_t expected_l1_write_bytes = 0;
    uint64_t expected_l1_reqs = 0;
    uint64_t expected_l2_read_bytes = 0;
    uint64_t expected_l2_write_bytes = 0;
    uint64_t expected_l2_dma_reqs = 0;
    uint64_t expected_vec_acc_busy_cycles = 0;

    // Vec pool counters
    uint64_t vec_acc_reqs = 0;
    uint64_t vec_acc_busy_cycles = 0;
    uint64_t vec_acc_occupied_cycles = 0;
    uint64_t vec_acc_queue_wait_cycles = 0;

    // L1 / L2 split memory counters
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

    // Back-compat aliases for the nafnet bridge, which still
    // sums into a flat `memory_*` set of fields. Populated from
    // the L2 DMA counters above.
    uint64_t memory_reqs = 0;
    uint64_t memory_busy_cycles = 0;
    uint64_t memory_queue_wait_cycles = 0;

    double vec_util = 0.0;
    double vec_occupancy = 0.0;
    double l1_bw_observed = 0.0;
    double l2_bw_observed = 0.0;

    // Cycle-fraction breakdown on the critical-path worker.
    int slowest_worker_tid = -1;
    uint64_t slowest_vec_cycles = 0;
    uint64_t slowest_dma_cycles = 0;
    uint64_t slowest_scalar_cycles = 0;
    uint64_t slowest_stall_cycles = 0;
    double vec_cycle_fraction = 0.0;
    double dma_cycle_fraction = 0.0;
    double scalar_cycle_fraction = 0.0;
    double stall_cycle_fraction = 0.0;

    bool verification_passed = false;
};

struct VecOpsTop : sc_module
{
    VecOpsRuntimeConfig cfg;
    AcceleratorTLM  mat_acc;
    AcceleratorPool vec_acc;
    Interconnect    noc;
    L1L2Memory      memory;

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
    std::vector<KernelWorkerInfo> collect_worker_info() const;
    void print_report(std::ostream &os) const;
    void done_monitor();
};
