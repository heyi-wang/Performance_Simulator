#include "matmul_top.h"

MatmulTop::MatmulTop(sc_module_name nm)
    : sc_module(nm),
      mat_acc("mat_acc", MAT_ACC_QUEUE_CAP),
      vec_acc("vec_acc", VEC_ACC_QUEUE_CAP),
      noc("noc"),
      memory("memory")
{
    // Bind accelerators and memory to the interconnect
    noc.to_mat.bind(mat_acc.tgt);
    noc.to_vec.bind(vec_acc.tgt);
    noc.to_mem.bind(memory.tgt);

    // Accelerators reach memory through the interconnect
    mat_acc.to_mem.bind(noc.tgt);
    vec_acc.to_mem.bind(noc.tgt);

    // -------------------------------------------------------
    // Worker configuration (K-split):
    //   Phase 1 (mat): GEMM_ACCESS_MAT tiles for this thread's K-slice
    //   Phase 2 (quant): GEMM_QUANT_VEC_CALLS tiles to quantize
    //                    the partial result fp32 → fp16
    //   mat_done_ev fires after BOTH phases complete so the
    //   AccumCoordinator starts only on fully quantized data.
    // -------------------------------------------------------
    for (int i = 0; i < NUM_THREADS; i++)
    {
        auto *w = new Worker(sc_gen_unique_name("worker"),
                             i,
                             GEMM_ACCESS_MAT,        // mat tiles for K-slice
                             GEMM_QUANT_VEC_CALLS,   // quant tiles after mat
                             MATMUL_ACC_CYCLE,
                             VECTOR_ACC_CYCLE,
                             SCALAR_OVERHEAD,
                             GEMM_A_BYTES,            // mat rd (A tile)
                             GEMM_B_BYTES,            // mat rd (B tile)
                             GEMM_C_BYTES,            // mat wr (C tile)
                             GEMM_QUANT_RD_BYTES,     // vec rd (fp32 partial)
                             GEMM_QUANT_WR_BYTES);    // vec wr (fp16 quantized)
        workers.push_back(w);
        w->init.bind(noc.tgt);
    }

    // -------------------------------------------------------
    // AccumCoordinator — event-driven tree-reduction phase.
    // Starts as soon as any pair of quantized partial results
    // is available (does not wait for all workers).
    // -------------------------------------------------------
    coordinator = new AccumCoordinator("accum_coord",
                                       VECTOR_ACC_CYCLE,
                                       GEMM_ACCUM_VEC_CALLS,
                                       GEMM_ACCUM_RD_BYTES,
                                       GEMM_ACCUM_WR_BYTES);

    coordinator->workers = workers;
    coordinator->init.bind(noc.tgt);
}

MatmulTop::~MatmulTop()
{
    for (auto *w : workers)
        delete w;
    delete coordinator;
}
