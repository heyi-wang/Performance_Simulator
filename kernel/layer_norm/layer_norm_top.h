#pragma once

#include "accelerator.h"
#include "accelerator_pool.h"
#include "common.h"
#include "interconnect.h"
#include "layer_norm_config.h"
#include "memory.h"
#include "worker.h"

#include <algorithm>
#include <memory>
#include <iosfwd>
#include <systemc>
#include <vector>

struct LayerNormRuntimeConfig
{
    uint64_t input_elem_bytes = LN_INPUT_ELEM_BYTES;
    uint64_t output_elem_bytes = LN_OUTPUT_ELEM_BYTES;
    int channels = LN_C;
    int height = LN_H;
    int width = LN_W;
    int worker_count = LN_NUM_WORKERS;
    uint64_t vec_acc_cap = LN_VEC_ACC_CAP;
    uint64_t vec_insn_cycle = LN_VEC_INSN_CYCLE;
    uint64_t pass1_insns = LN_PASS1_INSNS_CFG;
    uint64_t pass2_insns = LN_PASS2_INSNS_CFG;
    uint64_t pass3_insns = LN_PASS3_INSNS_CFG;
    int vec_acc_instances = LN_VEC_ACC_INSTANCES;
    uint64_t scalar_overhead = LN_SCALAR_OVERHEAD;
    uint64_t mean_cycles = LN_MEAN_CYCLES_CFG;
    uint64_t invstd_cycles = LN_INVSTD_CYCLES_CFG;
    uint64_t l1_base_lat = LN_L1_BASE_LAT_CFG;
    uint64_t l1_bw = LN_L1_BW_CFG;
    uint64_t l1_slots = LN_L1_SLOTS_CFG;
    uint64_t l2_base_lat = LN_L2_BASE_LAT_CFG;
    uint64_t l2_bw = LN_L2_BW_CFG;
    uint64_t l2_slots = LN_L2_SLOTS_CFG;
    size_t acc_queue_depth = LN_ACC_QUEUE_DEPTH;
    uint64_t dma_vec_rd_scalar = LN_DMA_VEC_RD_SCALAR_CFG;
    uint64_t dma_vec_wr_scalar = LN_DMA_VEC_WR_SCALAR_CFG;
    uint64_t max_inflight_vec_reqs = LN_MAX_INFLIGHT_VEC_REQS_CFG;
    uint64_t max_inflight_dma_writes = LN_MAX_INFLIGHT_DMA_WRITES_CFG;

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

    int effective_vec_instances() const
    {
        return std::max(1, std::min(vec_acc_instances, worker_count));
    }

    uint64_t pass_insns(int pass) const
    {
        switch (pass)
        {
            case 1: return pass1_insns;
            case 2: return pass2_insns;
            case 3: return pass3_insns;
            default: return 0;
        }
    }

    uint64_t pass_cycles(int pass) const
    {
        return vec_request_cycles(vec_insn_cycle, pass_insns(pass));
    }
};

struct LayerNormPostProcessor;

struct LayerNormSimulationStats
{
    // Aggregate counters (new L1/L2 model).
    uint64_t total_vec_reqs = 0;
    uint64_t total_pass1_reqs = 0;
    uint64_t total_pass2_reqs = 0;
    uint64_t total_pass3_reqs = 0;
    uint64_t total_rd_bytes = 0;   // L1 read bytes (aggregate)
    uint64_t total_wr_bytes = 0;   // L1 write bytes (aggregate)
    uint64_t total_wait_cycles = 0;
    uint64_t total_mem_cycles = 0;
    uint64_t max_elapsed_cycles = 0;

    // Expected counters.
    uint64_t expected_vec_reqs = 0;
    uint64_t expected_pass1_reqs = 0;
    uint64_t expected_pass2_reqs = 0;
    uint64_t expected_pass3_reqs = 0;
    uint64_t expected_vec_acc_busy_cycles = 0;
    uint64_t expected_l1_read_bytes = 0;
    uint64_t expected_l1_write_bytes = 0;
    uint64_t expected_l1_reqs = 0;
    uint64_t expected_l2_read_bytes = 0;
    uint64_t expected_l2_write_bytes = 0;
    uint64_t expected_l2_dma_reqs = 0;

    // Accelerator stats.
    uint64_t vec_acc_reqs = 0;
    uint64_t vec_acc_busy_cycles = 0;
    uint64_t vec_acc_occupied_cycles = 0;
    uint64_t vec_acc_queue_wait_cycles = 0;

    // L1 stats.
    uint64_t l1_reqs = 0;
    uint64_t l1_read_bytes = 0;
    uint64_t l1_write_bytes = 0;
    uint64_t l1_busy_cycles = 0;
    uint64_t l1_queue_wait_cycles = 0;

    // L2 DMA stats.
    uint64_t l2_dma_reqs = 0;
    uint64_t l2_dma_read_bytes = 0;
    uint64_t l2_dma_write_bytes = 0;
    uint64_t l2_dma_busy_cycles = 0;
    uint64_t l2_dma_queue_wait_cycles = 0;

    // Back-compat aggregates expected by nafnet bridge convert_stats()
    // (mirror pooling/dw_conv2d's convention: memory_* == L2 DMA).
    uint64_t memory_reqs = 0;
    uint64_t memory_busy_cycles = 0;
    uint64_t memory_queue_wait_cycles = 0;

    double vec_util = 0.0;
    double vec_occupancy = 0.0;
    double l1_bw_observed = 0.0;
    double l2_bw_observed = 0.0;

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

struct LayerNormTop : sc_module
{
    LayerNormRuntimeConfig cfg;
    AcceleratorTLM  mat_acc;
    AcceleratorPool vec_acc;
    Interconnect    noc;
    L1L2Memory      memory;

    std::vector<Worker *> workers;
    std::vector<std::unique_ptr<LayerNormPostProcessor>> post_processors;
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
