#include "top.h"

Top::Top(sc_module_name nm)
    : sc_module(nm),
      mat_acc("mat_acc", MAT_ACCEL_COUNT, NUM_THREADS),
      vec_acc("vec_acc", VEC_ACCEL_COUNT, NUM_THREADS),
      noc("noc"),
      memory("memory", 1, 128, MEMORY_PARALLEL_SLOTS_CFG)
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

    // Derive per-worker workload from config parameters
    uint64_t tile_M = ceil_div_u64(A_M, MATMUL_M);
    uint64_t tile_K = ceil_div_u64(A_K, MATMUL_K);
    uint64_t tile_N = ceil_div_u64(B_N, MATMUL_N);

    uint64_t access_mat    = tile_M * tile_K * tile_N;
    uint64_t access_vec    = ceil_div_u64(A_M * B_N, VECTOR_ACC_CAP);
    uint64_t mat_cycles    = MATMUL_ACC_CYCLE;
    uint64_t vec_cycles    = VECTOR_ACC_CYCLE;
    uint64_t scalar_cycles = SCALAR_OVERHEAD;

    uint64_t A_bytes = MATMUL_M * MATMUL_K * sizeof(float);
    uint64_t B_bytes = MATMUL_K * MATMUL_N * sizeof(float);
    uint64_t C_bytes = MATMUL_M * MATMUL_N * sizeof(float);

    for (int i = 0; i < NUM_THREADS; i++)
    {
        auto *w = new Worker(sc_gen_unique_name("worker"),
                             i,
                             access_mat,
                             access_vec,
                             mat_cycles,
                             vec_cycles,
                             scalar_cycles,
                             A_bytes,
                             B_bytes,
                             C_bytes,
                             0,
                             0,
                             NUM_THREADS,
                             NUM_THREADS);
        workers.push_back(w);
        w->init.bind(noc.tgt);
    }
}

Top::~Top()
{
    for (auto *w : workers)
        delete w;
}
