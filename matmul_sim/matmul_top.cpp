#include "matmul_top.h"

MatmulTop::MatmulTop(sc_module_name nm, const MatmulConfig &cfg_)
    : sc_module(nm),
      cfg(cfg_),
      mat_acc("mat_acc", cfg.mat_accel_count, cfg.mat_acc_queue_cap()),
      vec_acc("vec_acc", cfg.vec_accel_count, cfg.vec_acc_queue_cap()),
      noc("noc"),
      memory("memory")
{
    // Bind accelerators and memory to the interconnect
    noc.to_mat.bind(mat_acc.tgt);
    noc.to_vec.bind(vec_acc.tgt);
    noc.to_mem.bind(memory.tgt);

    // Each physical accelerator instance reaches memory through the interconnect.
    for (auto &unit : mat_acc.units)
        unit->to_mem.bind(noc.tgt);
    for (auto &unit : vec_acc.units)
        unit->to_mem.bind(noc.tgt);

    // -------------------------------------------------------
    // Worker configuration (K-split):
    //   Workers only execute matmul for their K-slice.
    //   Quantization is deferred until the coordinator finishes
    //   the high-precision reduction tree.
    // -------------------------------------------------------
    for (int i = 0; i < cfg.thread_count; i++)
    {
        auto *w = new Worker(sc_gen_unique_name("worker"),
                             i,
                             cfg.local_access_mat_for_thread(i), // mat tiles for this worker's K-slice
                             0, // final quantization is deferred to the coordinator
                             MATMUL_ACC_CYCLE,
                             VECTOR_ACC_CYCLE,
                             SCALAR_OVERHEAD,
                             MatmulConfig::gemm_a_bytes,        // mat rd (A tile)
                             MatmulConfig::gemm_b_bytes,        // mat rd (B tile)
                             MatmulConfig::gemm_c_bytes,        // mat wr (C tile)
                             0,
                             0);
        workers.push_back(w);
        w->init.bind(noc.tgt);
    }

    // -------------------------------------------------------
    // AccumCoordinator — event-driven tree-reduction phase.
    // Starts as soon as any pair of partial results
    // is available (does not wait for all workers).
    // -------------------------------------------------------
    coordinator = new AccumCoordinator("accum_coord",
                                       VECTOR_ACC_CYCLE,
                                       MatmulConfig::gemm_accum_vec_calls(),
                                       MatmulConfig::gemm_quant_vec_calls(),
                                       static_cast<uint64_t>(cfg.vec_accel_count),
                                       SCALAR_OVERHEAD,
                                       MatmulConfig::gemm_accum_rd_bytes,
                                       MatmulConfig::gemm_accum_wr_bytes,
                                       MatmulConfig::gemm_quant_rd_bytes,
                                       MatmulConfig::gemm_quant_wr_bytes);

    coordinator->workers = workers;
    coordinator->init.bind(noc.tgt);
}

MatmulTop::~MatmulTop()
{
    for (auto *w : workers)
        delete w;
    delete coordinator;
}
