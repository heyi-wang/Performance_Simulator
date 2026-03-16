#include "matmul_top.h"

MatmulTop::MatmulTop(sc_module_name nm)
    : sc_module(nm),
      mat_acc("mat_acc", NUM_THREADS),
      vec_acc("vec_acc", NUM_THREADS + 1), // +1 for AccumCoordinator concurrent requests
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
    //   - access_mat = tiles per worker for their K-slice
    //   - access_vec = 0  (accumulation handled by AccumCoordinator)
    //   - M is the full output-row count (not divided by threads)
    // -------------------------------------------------------
    for (int i = 0; i < NUM_THREADS; i++)
    {
        auto *w = new Worker(sc_gen_unique_name("worker"),
                             i,
                             GEMM_ACCESS_MAT,   // mat tiles for K-slice
                             0,                 // no vec calls from workers
                             MATMUL_ACC_CYCLE,
                             VECTOR_ACC_CYCLE,
                             SCALAR_OVERHEAD,
                             GEMM_A_BYTES,
                             GEMM_B_BYTES,
                             GEMM_C_BYTES);
        workers.push_back(w);
        w->init.bind(noc.tgt);
    }

    // -------------------------------------------------------
    // AccumCoordinator — tree-reduction accumulation phase
    // -------------------------------------------------------
    coordinator = new AccumCoordinator("accum_coord",
                                       VECTOR_ACC_CYCLE,
                                       GEMM_ACCUM_VEC_CALLS,
                                       GEMM_ACCUM_RD_BYTES,
                                       GEMM_ACCUM_WR_BYTES);

    // Share all workers with the coordinator so it can observe mat_done_ev
    coordinator->workers = workers;

    // Bind coordinator's TLM socket to the interconnect
    coordinator->init.bind(noc.tgt);
}

MatmulTop::~MatmulTop()
{
    for (auto *w : workers)
        delete w;
    delete coordinator;
}
