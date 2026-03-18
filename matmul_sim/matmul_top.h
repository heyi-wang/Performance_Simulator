#pragma once

#include "../src/memory.h"
#include "../src/interconnect.h"
#include "../src/accelerator_pool.h"
#include "../src/worker.h"
#include "accum_coordinator.h"
#include "matmul_config.h"
#include <vector>

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
};
