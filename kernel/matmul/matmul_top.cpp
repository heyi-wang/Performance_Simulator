#include "matmul_top.h"

#include <algorithm>
#include <iomanip>
#include <iostream>

MatmulTop::MatmulTop(sc_module_name nm,
                     const MatmulRuntimeConfig &cfg_,
                     sc_event *start_event,
                     sc_event *done_event_)
    : sc_module(nm),
      cfg(cfg_),
      mat_acc("mat_acc", cfg.mat_accel_count, cfg.mat_acc_queue_cap()),
      vec_acc("vec_acc", cfg.vec_accel_count, cfg.vec_acc_queue_cap()),
      noc("noc"),
      memory("memory",
             cfg.memory_base_lat,
             cfg.memory_bw,
             cfg.memory_parallel_slots),
      done_event(done_event_)
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
                                       cfg.gemm_accum_vec_calls(),
                                       cfg.gemm_quant_vec_calls(),
                                       cfg.gemm_accum_rd_bytes(),
                                       cfg.gemm_accum_wr_bytes(),
                                       cfg.gemm_quant_rd_bytes(),
                                       cfg.gemm_quant_wr_bytes());

    if (done_event)
    {
        completion_fifo =
            std::make_unique<sc_fifo<int>>(sc_gen_unique_name("matmul_done_fifo"),
                                           cfg.thread_count + 1);
        SC_THREAD(done_monitor);
    }

    // Worker configuration (K-split):
    //   Workers execute local matmul first, then run reduction and final
    //   quantization inside the same SC_THREAD via the passive coordinator.
    // -------------------------------------------------------
    for (int i = 0; i < cfg.thread_count; i++)
    {
        WorkerPostProcessor *post_processor =
            (cfg.local_k_extent_for_thread(i) > 0) ? coordinator : nullptr;
        auto *w = new Worker(sc_gen_unique_name("worker"),
                             i,
                             cfg.local_access_mat_for_thread(i), // mat tiles for this worker's K-slice
                             0,
                             cfg.mat_cycle,
                             cfg.vec_cycle,
                             cfg.scalar_overhead,
                             cfg.gemm_a_bytes(),               // mat rd (A tile)
                             cfg.gemm_b_bytes(),               // mat rd (B tile)
                             cfg.gemm_c_bytes(),               // mat wr (C tile)
                             0,
                             0,
                             cfg.mat_acc_queue_cap(),
                             cfg.vec_acc_queue_cap(),
                             post_processor,
                             start_event,
                             completion_fifo.get());
        workers.push_back(w);
        if (post_processor)
            active_workers.push_back(w);
        w->init.bind(noc.tgt);
    }

    coordinator->configure_workers(active_workers);
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
        cfg.gemm_m() * cfg.gemm_k() * cfg.gemm_n();
    stats.mat_req_total = mat_acc.req_count_total();
    stats.mat_busy_total = mat_acc.busy_cycles_total();
    stats.mat_occupied_total = mat_acc.occupied_cycles_total();
    stats.mat_qwait_total = mat_acc.queue_wait_cycles_total();
    stats.vec_req_total = vec_acc.req_count_total();
    stats.expected_mat_req_total = 0;
    for (int tid = 0; tid < cfg.thread_count; ++tid)
        stats.expected_mat_req_total += cfg.local_access_mat_for_thread(tid);
    stats.expected_accum_pairs =
        static_cast<uint64_t>(std::max(cfg.active_thread_count() - 1, 0));
    stats.expected_vec_req_total =
        stats.expected_accum_pairs * cfg.gemm_accum_vec_calls() +
        ((cfg.active_thread_count() > 0) ? cfg.gemm_quant_vec_calls() : 0);
    stats.vec_busy_total = vec_acc.busy_cycles_total();
    stats.vec_occupied_total = vec_acc.occupied_cycles_total();
    stats.vec_qwait_total = vec_acc.queue_wait_cycles_total();
    stats.memory_reqs = memory.reqs;
    stats.memory_busy_cycles = memory.busy_cycles;
    stats.memory_queue_wait_cycles = memory.qwait_cycles;
    stats.coordinator_stall = coordinator->stall_cycles;
    stats.coordinator_compute = coordinator->compute_cycles;
    stats.total_rd_bytes =
        stats.mat_req_total * (cfg.gemm_a_bytes() + cfg.gemm_b_bytes()) +
        coordinator->accum_vec_calls_total * cfg.gemm_accum_rd_bytes() +
        coordinator->final_quant_calls_total * cfg.gemm_quant_rd_bytes();
    stats.total_wr_bytes =
        stats.mat_req_total * cfg.gemm_c_bytes() +
        coordinator->accum_vec_calls_total * cfg.gemm_accum_wr_bytes() +
        coordinator->final_quant_calls_total * cfg.gemm_quant_wr_bytes();

    if (stats.total_elapsed > 0)
    {
        stats.gflops =
            (static_cast<double>(stats.total_macs) * 2.0) /
            static_cast<double>(stats.total_elapsed);
        stats.mem_bw =
            static_cast<double>(memory.busy_cycles) *
            static_cast<double>(cfg.memory_bw) /
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

    stats.verification_passed =
        stats.mat_req_total == stats.expected_mat_req_total &&
        stats.vec_req_total == stats.expected_vec_req_total &&
        coordinator->quant_start_time >= coordinator->accum_end_time &&
        coordinator->accum_vec_calls_total ==
            stats.expected_accum_pairs * cfg.gemm_accum_vec_calls() &&
        coordinator->final_quant_calls_total ==
            ((cfg.active_thread_count() > 0) ? cfg.gemm_quant_vec_calls() : 0);

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
    os << "GEMM               : [" << cfg.gemm_m() << " x "
       << cfg.gemm_k() << " x " << cfg.gemm_n() << "]\n";
    os << "Threads            : " << cfg.thread_count << "\n";
    os << "Active threads     : " << cfg.active_thread_count() << "\n";
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
    os << "Read bytes         : " << stats.total_rd_bytes << "\n";
    os << "Write bytes        : " << stats.total_wr_bytes << "\n";
    os << "Verification       : " << (stats.verification_passed ? "PASS" : "FAIL") << "\n";
}

void MatmulTop::done_monitor()
{
    for (int i = 0; i < cfg.thread_count; ++i)
        completion_fifo->read();
    done_event->notify(SC_ZERO_TIME);
}
