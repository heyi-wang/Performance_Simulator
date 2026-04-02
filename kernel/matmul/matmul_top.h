#pragma once

#include "memory.h"
#include "interconnect.h"
#include "accelerator_pool.h"
#include "worker.h"
#include "accum_coordinator.h"
#include "matmul_config.h"
#include <iosfwd>
#include <vector>

struct MatmulSimulationStats
{
    uint64_t total_elapsed = 0;
    uint64_t max_mat_elapsed = 0;
    uint64_t accum_overhead = 0;
    uint64_t total_macs = 0;
    uint64_t mat_req_total = 0;
    uint64_t vec_req_total = 0;
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
    MatmulConfig    cfg;
    AcceleratorPool mat_acc;
    AcceleratorPool vec_acc;
    Interconnect    noc;
    Memory          memory;

    std::vector<Worker *> workers;
    AccumCoordinator     *coordinator = nullptr;

    SC_HAS_PROCESS(MatmulTop);
    MatmulTop(sc_module_name nm, const MatmulConfig &cfg_);
    ~MatmulTop() override;

    MatmulSimulationStats collect_stats() const;
    void print_report(std::ostream &os) const;
};
