#include "matmul_top.h"

#include <algorithm>
#include <iomanip>
#include <iostream>

MatmulTop::MatmulTop(sc_module_name nm, const MatmulConfig &cfg_)
    : sc_module(nm),
      cfg(cfg_),
      mat_acc("mat_acc", cfg.mat_accel_count, cfg.mat_acc_queue_cap()),
      vec_acc("vec_acc", cfg.vec_accel_count, cfg.vec_acc_queue_cap()),
      noc("noc"),
      memory("memory",
             HW_MEMORY_BASE_LAT,
             HW_MATMUL_MEMORY_BYTES_PER_CYCLE,
             MEMORY_PARALLEL_SLOTS_CFG)
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
    // Passive shared reduction / quantization state. Existing workers execute
    // all post-mat work through this object; no additional threads are spawned.
    coordinator = new AccumCoordinator("accum_coord",
                                       MatmulConfig::gemm_accum_vec_calls(),
                                       MatmulConfig::gemm_quant_vec_calls(),
                                       MatmulConfig::gemm_accum_rd_bytes,
                                       MatmulConfig::gemm_accum_wr_bytes,
                                       MatmulConfig::gemm_quant_rd_bytes,
                                       MatmulConfig::gemm_quant_wr_bytes);

    // Worker configuration (K-split):
    //   Workers execute local matmul first, then run reduction and final
    //   quantization inside the same SC_THREAD via the passive coordinator.
    // -------------------------------------------------------
    for (int i = 0; i < cfg.thread_count; i++)
    {
        auto *w = new Worker(sc_gen_unique_name("worker"),
                             i,
                             cfg.local_access_mat_for_thread(i), // mat tiles for this worker's K-slice
                             0,
                             MATMUL_ACC_CYCLE,
                             VECTOR_ACC_CYCLE,
                             SCALAR_OVERHEAD,
                             MatmulConfig::gemm_a_bytes,        // mat rd (A tile)
                             MatmulConfig::gemm_b_bytes,        // mat rd (B tile)
                             MatmulConfig::gemm_c_bytes,        // mat wr (C tile)
                             0,
                             0,
                             cfg.mat_acc_queue_cap(),
                             cfg.vec_acc_queue_cap(),
                             coordinator);
        workers.push_back(w);
        w->init.bind(noc.tgt);
    }

    coordinator->configure_workers(workers);
}

MatmulTop::~MatmulTop()
{
    for (auto *w : workers)
        delete w;
    delete coordinator;
}

MatmulSimulationStats MatmulTop::collect_stats() const
{
    MatmulSimulationStats stats;

    for (const auto *w : workers)
    {
        stats.max_mat_elapsed = std::max(stats.max_mat_elapsed, w->mat_elapsed_cycles);
        stats.total_elapsed = std::max(stats.total_elapsed, w->elapsed_cycles);
        stats.total_worker_stall += w->stall_cycles;
    }

    stats.accum_overhead =
        (stats.total_elapsed > stats.max_mat_elapsed)
            ? (stats.total_elapsed - stats.max_mat_elapsed)
            : 0;

    stats.total_macs =
        MatmulConfig::gemm_m * MatmulConfig::gemm_k * MatmulConfig::gemm_n;
    stats.mat_req_total = mat_acc.req_count_total();
    stats.mat_busy_total = mat_acc.busy_cycles_total();
    stats.mat_occupied_total = mat_acc.occupied_cycles_total();
    stats.mat_qwait_total = mat_acc.queue_wait_cycles_total();
    stats.vec_req_total = vec_acc.req_count_total();
    stats.vec_busy_total = vec_acc.busy_cycles_total();
    stats.vec_occupied_total = vec_acc.occupied_cycles_total();
    stats.vec_qwait_total = vec_acc.queue_wait_cycles_total();
    stats.memory_reqs = memory.reqs;
    stats.memory_busy_cycles = memory.busy_cycles;
    stats.memory_queue_wait_cycles = memory.qwait_cycles;
    stats.coordinator_stall = coordinator->stall_cycles;
    stats.coordinator_compute = coordinator->compute_cycles;

    if (stats.total_elapsed > 0)
    {
        stats.gflops =
            (static_cast<double>(stats.total_macs) * 2.0) /
            static_cast<double>(stats.total_elapsed);
        stats.mem_bw =
            static_cast<double>(memory.busy_cycles) *
            static_cast<double>(HW_MATMUL_MEMORY_BYTES_PER_CYCLE) /
            static_cast<double>(stats.total_elapsed);
    }

    const double total_elapsed = static_cast<double>(stats.total_elapsed);
    const double mat_capacity =
        total_elapsed * static_cast<double>(mat_acc.instance_count());
    const double vec_capacity =
        total_elapsed * static_cast<double>(vec_acc.instance_count());

    if (mat_capacity > 0.0)
    {
        stats.mat_occupancy =
            static_cast<double>(stats.mat_occupied_total) / mat_capacity * 100.0;
        stats.mat_compute_util =
            static_cast<double>(stats.mat_busy_total) / mat_capacity * 100.0;
    }

    if (vec_capacity > 0.0)
    {
        stats.vec_occupancy =
            static_cast<double>(stats.vec_occupied_total) / vec_capacity * 100.0;
        stats.vec_compute_util =
            static_cast<double>(stats.vec_busy_total) / vec_capacity * 100.0;
    }

    return stats;
}

void MatmulTop::print_report(std::ostream &os) const
{
    const MatmulSimulationStats stats = collect_stats();

    os << "\n=== Accelerator stats ===\n";
    os << "mat_acc pool : units=" << mat_acc.instance_count()
       << "  reqs=" << stats.mat_req_total
       << "  busy_cycles=" << stats.mat_busy_total
       << "  occupied_cycles=" << stats.mat_occupied_total
       << "  shared_qwait_cycles=" << stats.mat_qwait_total << "\n";
    os << "vec_acc pool : units=" << vec_acc.instance_count()
       << "  reqs=" << stats.vec_req_total
       << "  busy_cycles=" << stats.vec_busy_total
       << "  occupied_cycles=" << stats.vec_occupied_total
       << "  shared_qwait_cycles=" << stats.vec_qwait_total << "\n";
    os << "memory  : reqs=" << stats.memory_reqs
       << "  busy_cycles=" << stats.memory_busy_cycles
       << "  queue_wait_cycles=" << stats.memory_queue_wait_cycles << "\n";

    os << "\n=== Performance summary ===\n";
    os << std::fixed << std::setprecision(2);
    os << "GEMM               : [" << MatmulConfig::gemm_m << " x "
       << MatmulConfig::gemm_k << " x " << MatmulConfig::gemm_n << "]\n";
    os << "Threads            : " << cfg.thread_count << "\n";
    os << "Total MACs         : " << stats.total_macs << "\n";
    os << "Mat phase          : " << stats.max_mat_elapsed
       << " cycles  (critical-path worker)\n";
    os << "Accum overhead     : " << stats.accum_overhead
       << " cycles  (beyond worker mat critical path)\n";
    os << "Total elapsed      : " << stats.total_elapsed << " cycles\n";
    os << "Throughput         : " << stats.gflops << " GFLOPS  (at 1 GHz)\n";
    os << "mat_acc occupancy  : " << stats.mat_occupancy << "%\n";
    os << "mat_acc compute    : " << stats.mat_compute_util << "%\n";
    os << "vec_acc occupancy  : " << stats.vec_occupancy << "%\n";
    os << "vec_acc compute    : " << stats.vec_compute_util << "%\n";
    os << "Mem BW avg         : " << stats.mem_bw << " bytes/cycle\n";
    os << "Worker mat stall   : " << stats.total_worker_stall
       << " cycles  (all worker-side backpressure across mat/reduction/quant)\n";
    os << "Post-mat stall     : " << stats.coordinator_stall
       << " cycles  (worker-driven reduction/final-quant vec backpressure)\n";
    os << "Post-mat compute   : " << stats.coordinator_compute
       << " cycles  (worker-driven reduction/final-quant scalar+service time)\n";
}
