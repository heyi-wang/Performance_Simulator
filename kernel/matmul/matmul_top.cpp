#include "matmul_top.h"

#include <algorithm>
#include <iomanip>
#include <iostream>

#include "report_formatter.h"

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
             cfg.l1_base_lat,
             cfg.l1_bw,
             cfg.dma_base_lat,
             cfg.dma_bw,
             cfg.l1_slots,
             cfg.dma_slots),
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
                                       cfg.gemm_accum_vec_cycles(),
                                       cfg.gemm_quant_vec_cycles(),
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
    workers.reserve(static_cast<size_t>(cfg.thread_count));
    active_workers.reserve(static_cast<size_t>(cfg.active_thread_count()));
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
                             cfg.mat_scalar_overhead,
                             cfg.vec_scalar_overhead,
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
        w->configure_gemm_reuse(cfg.gemm_tile_m(),
                                cfg.gemm_tile_n(),
                                cfg.local_tile_k_for_thread(i),
                                cfg.accumulator_register_count);
        w->configure_dma_row_cost(MATMUL_M,
                                  MATMUL_K,
                                  MATMUL_M,
                                  HW_DMA_A_ROW_SCALAR,
                                  HW_DMA_B_ROW_SCALAR,
                                  HW_DMA_C_ROW_SCALAR);
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
    const int active_threads = cfg.active_thread_count();
    const uint64_t accum_vec_calls = cfg.gemm_accum_vec_calls();
    const uint64_t quant_vec_calls =
        (active_threads > 0) ? cfg.gemm_quant_vec_calls() : 0;
    const uint64_t accum_rd_bytes = cfg.gemm_accum_rd_bytes();
    const uint64_t accum_wr_bytes = cfg.gemm_accum_wr_bytes();
    const uint64_t quant_rd_bytes = cfg.gemm_quant_rd_bytes();
    const uint64_t quant_wr_bytes = cfg.gemm_quant_wr_bytes();

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
    {
        stats.expected_mat_req_total += cfg.local_access_mat_for_thread(tid);
        const uint64_t mat_reqs = cfg.local_access_mat_for_thread(tid);
        const uint64_t output_tiles = cfg.local_output_tiles_for_thread(tid);
        stats.expected_l1_reqs += mat_reqs + output_tiles;
        stats.expected_l1_read_bytes += cfg.local_mat_l1_read_bytes_for_thread(tid);
        stats.expected_l1_write_bytes += cfg.local_mat_l1_write_bytes_for_thread(tid);
        stats.expected_dma_reqs +=
            mat_reqs + cfg.local_b_dma_tiles_for_thread(tid) + output_tiles;
        stats.expected_dma_read_bytes += cfg.local_mat_dma_read_bytes_for_thread(tid);
        stats.expected_dma_write_bytes += cfg.local_mat_dma_write_bytes_for_thread(tid);
    }
    stats.expected_accum_pairs =
        static_cast<uint64_t>(std::max(active_threads - 1, 0));
    stats.expected_vec_req_total =
        stats.expected_accum_pairs * accum_vec_calls + quant_vec_calls;
    const uint64_t expected_accum_vec_calls =
        stats.expected_accum_pairs * accum_vec_calls;
    const uint64_t expected_quant_vec_calls = quant_vec_calls;
    const uint64_t expected_vec_l1_read_bytes =
        expected_accum_vec_calls * accum_rd_bytes +
        expected_quant_vec_calls * quant_rd_bytes;
    const uint64_t expected_vec_l1_write_bytes =
        expected_accum_vec_calls * accum_wr_bytes +
        expected_quant_vec_calls * quant_wr_bytes;
    stats.expected_l1_reqs += stats.expected_vec_req_total * 2;
    stats.expected_l1_read_bytes += expected_vec_l1_read_bytes;
    stats.expected_l1_write_bytes += expected_vec_l1_write_bytes;
    stats.expected_dma_reqs += stats.expected_vec_req_total * 2;
    stats.expected_dma_read_bytes += expected_vec_l1_read_bytes;
    stats.expected_dma_write_bytes += expected_vec_l1_write_bytes;
    stats.vec_busy_total = vec_acc.busy_cycles_total();
    stats.vec_occupied_total = vec_acc.occupied_cycles_total();
    stats.vec_qwait_total = vec_acc.queue_wait_cycles_total();
    stats.l1_reqs = memory.l1_reqs;
    stats.l1_read_bytes = memory.l1_read_bytes;
    stats.l1_write_bytes = memory.l1_write_bytes;
    stats.l1_busy_cycles = memory.l1_busy_cycles;
    stats.l1_qwait_cycles = memory.l1_qwait_cycles;
    stats.dma_reqs = memory.dma_reqs;
    stats.dma_read_bytes = memory.dma_read_bytes;
    stats.dma_write_bytes = memory.dma_write_bytes;
    stats.dma_busy_cycles = memory.dma_busy_cycles;
    stats.dma_qwait_cycles = memory.dma_qwait_cycles;
    stats.memory_reqs = stats.l1_reqs + stats.dma_reqs;
    stats.memory_busy_cycles = stats.l1_busy_cycles + stats.dma_busy_cycles;
    stats.memory_queue_wait_cycles = stats.l1_qwait_cycles + stats.dma_qwait_cycles;
    stats.coordinator_stall = coordinator->stall_cycles;
    stats.coordinator_compute = coordinator->compute_cycles;
    stats.total_rd_bytes = stats.dma_read_bytes;
    stats.total_wr_bytes = stats.dma_write_bytes;

    if (stats.total_elapsed > 0)
    {
        stats.gflops =
            (static_cast<double>(stats.total_macs) * 2.0) /
            static_cast<double>(stats.total_elapsed);
        stats.mem_bw =
            static_cast<double>(stats.total_rd_bytes + stats.total_wr_bytes) /
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
        stats.l1_reqs == stats.expected_l1_reqs &&
        stats.l1_read_bytes == stats.expected_l1_read_bytes &&
        stats.l1_write_bytes == stats.expected_l1_write_bytes &&
        stats.dma_reqs == stats.expected_dma_reqs &&
        stats.dma_read_bytes == stats.expected_dma_read_bytes &&
        stats.dma_write_bytes == stats.expected_dma_write_bytes &&
        coordinator->quant_start_time >= coordinator->accum_end_time &&
        coordinator->accum_vec_calls_total ==
            stats.expected_accum_pairs * accum_vec_calls &&
        coordinator->final_quant_calls_total ==
            quant_vec_calls;

    return stats;
}

std::vector<KernelWorkerInfo> MatmulTop::collect_worker_info() const
{
    std::vector<KernelWorkerInfo> info;
    info.reserve(workers.size());

    for (const auto *w : workers)
    {
        KernelWorkerInfo wi;
        wi.tid = w->tid;
        wi.mat_reqs = w->mat_calls;
        wi.vec_reqs = w->vec_calls;
        wi.elapsed_cycles = w->elapsed_cycles;
        wi.stall_cycles = w->stall_cycles;
        wi.mem_cycles = w->mem_cycles_accum;

        const uint64_t service_cycles =
            w->mat_calls * w->mat_cycles +
            w->vec_service_cycles;
        wi.scalar_cycles = (w->compute_cycles >= service_cycles)
            ? (w->compute_cycles - service_cycles)
            : 0;

        wi.rd_bytes =
            cfg.local_mat_dma_read_bytes_for_thread(w->tid) +
            w->accum_vec_calls * cfg.gemm_accum_rd_bytes() +
            w->quant_vec_calls * cfg.gemm_quant_rd_bytes();
        wi.wr_bytes =
            cfg.local_mat_dma_write_bytes_for_thread(w->tid) +
            w->accum_vec_calls * cfg.gemm_accum_wr_bytes() +
            w->quant_vec_calls * cfg.gemm_quant_wr_bytes();
        info.push_back(wi);
    }

    return info;
}

void MatmulTop::print_report(std::ostream &os) const
{
    const MatmulSimulationStats stats = collect_stats();
    const std::vector<KernelWorkerInfo> worker_info = collect_worker_info();

    uint64_t total_scalar_cycles = 0;
    uint64_t total_stall_cycles = 0;
    uint64_t total_mem_cycles = 0;
    for (const auto &worker : worker_info)
    {
        total_scalar_cycles += worker.scalar_cycles;
        total_stall_cycles += worker.stall_cycles;
        total_mem_cycles += worker.mem_cycles;
    }

    report::print_section_title(os, "Simulation Info");
    report::print_fields(os, {
        {"Operation Type", "K-split General Matrix Multiply"},
        {"GEMM Shape", "[" + report::fmt_u64(cfg.gemm_m()) + " x " +
                       report::fmt_u64(cfg.gemm_k()) + " x " +
                       report::fmt_u64(cfg.gemm_n()) + "]"},
        {"K per Worker", report::fmt_u64(cfg.gemm_k_per_thread())},
        {"Active Workers [count]", report::fmt_int(cfg.active_thread_count())},
    });

    report::print_section_title(os, "Hardware Configuration");
    report::print_fields(os, {
        {"Workers [count]", report::fmt_int(cfg.thread_count)},
        {"Matrix Accelerators [count]", report::fmt_int(cfg.mat_accel_count)},
        {"Vector Accelerators [count]", report::fmt_int(cfg.vec_accel_count)},
        {"C Accumulator Registers [tiles]", report::fmt_u64(cfg.accumulator_register_count)},
        {"Matrix Accelerator Tile", "[" + report::fmt_u64(MATMUL_M) + " x " +
                                     report::fmt_u64(MATMUL_K) + " x " +
                                     report::fmt_u64(MATMUL_N) + "]"},
        {"Vector Accelerator Datapath [bytes/request]", report::fmt_u64(VECTOR_ACC_CAP)},
        {"Vector Instruction Cycle [cycles/insn]", report::fmt_u64(cfg.vec_cycle)},
        {"Accum Vector Instructions [insns/request]", report::fmt_u64(MATMUL_ACCUM_VEC_INSNS)},
        {"Quant Vector Instructions [insns/request]", report::fmt_u64(MATMUL_QUANT_VEC_INSNS)},
        {"Matrix Accelerator Queue Depth [requests]", report::fmt_u64(cfg.mat_acc_queue_cap())},
        {"Vector Accelerator Queue Depth [requests]", report::fmt_u64(cfg.vec_acc_queue_cap())},
        {"L1 Bandwidth [bytes/cycle]", report::fmt_u64(cfg.l1_bw)},
        {"L1 Base Latency [cycles]", report::fmt_u64(cfg.l1_base_lat)},
        {"L1 Parallel Slots", report::fmt_u64(cfg.l1_slots)},
        {"DMA Bandwidth [bytes/cycle]", report::fmt_u64(cfg.dma_bw)},
        {"DMA Base Latency [cycles]", report::fmt_u64(cfg.dma_base_lat)},
        {"DMA Parallel Slots", report::fmt_u64(cfg.dma_slots)},
    });

    report::print_section_title(os, "Worker Summary");
    report::print_table(os, report::make_worker_summary_table(worker_info));

    report::print_section_title(os, "Accelerator Summary");
    std::vector<report::AcceleratorSummaryRow> accel_rows;
    accel_rows.push_back({
        "Matrix Accelerator",
        "pool-level",
        report::fmt_int(cfg.mat_accel_count),
        report::fmt_u64(stats.mat_req_total),
        report::fmt_u64(stats.mat_qwait_total),
        report::fmt_u64(stats.mat_busy_total),
        report::fmt_u64(stats.mat_occupied_total),
        report::fmt_percent(stats.mat_compute_util),
        report::fmt_percent(stats.mat_occupancy),
        report::na(),
        report::na(),
    });
    for (auto &r : report::make_per_instance_accel_rows(
             "Matrix Accelerator", mat_acc.per_instance_stats(), stats.total_elapsed))
        accel_rows.push_back(std::move(r));

    accel_rows.push_back({
        "Vector Accelerator",
        "pool-level",
        report::fmt_int(cfg.vec_accel_count),
        report::fmt_u64(stats.vec_req_total),
        report::fmt_u64(stats.vec_qwait_total),
        report::fmt_u64(stats.vec_busy_total),
        report::fmt_u64(stats.vec_occupied_total),
        report::fmt_percent(stats.vec_compute_util),
        report::fmt_percent(stats.vec_occupancy),
        report::na(),
        report::na(),
    });
    for (auto &r : report::make_per_instance_accel_rows(
             "Vector Accelerator", vec_acc.per_instance_stats(), stats.total_elapsed))
        accel_rows.push_back(std::move(r));

    report::print_table(os, report::make_accelerator_summary_table(accel_rows));

    report::print_section_title(os, "Memory Hierarchy");
    report::print_fields(os, {
        {"L1 Engine Requests [requests]", report::fmt_u64(stats.l1_reqs)},
        {"Expected L1 Engine Requests [requests]", report::fmt_u64(stats.expected_l1_reqs)},
        {"L1 Engine Read Bytes [bytes]", report::fmt_u64(stats.l1_read_bytes)},
        {"Expected L1 Engine Read Bytes [bytes]", report::fmt_u64(stats.expected_l1_read_bytes)},
        {"L1 Engine Write Bytes [bytes]", report::fmt_u64(stats.l1_write_bytes)},
        {"Expected L1 Engine Write Bytes [bytes]", report::fmt_u64(stats.expected_l1_write_bytes)},
        {"L1 Engine Busy Cycles [cycles]", report::fmt_u64(stats.l1_busy_cycles)},
        {"L1 Engine Queue Wait [cycles]", report::fmt_u64(stats.l1_qwait_cycles)},
        {"DMA Engine Requests [requests]", report::fmt_u64(stats.dma_reqs)},
        {"Expected DMA Engine Requests [requests]", report::fmt_u64(stats.expected_dma_reqs)},
        {"DMA Engine Read Bytes [bytes]", report::fmt_u64(stats.dma_read_bytes)},
        {"Expected DMA Engine Read Bytes [bytes]", report::fmt_u64(stats.expected_dma_read_bytes)},
        {"DMA Engine Write Bytes [bytes]", report::fmt_u64(stats.dma_write_bytes)},
        {"Expected DMA Engine Write Bytes [bytes]", report::fmt_u64(stats.expected_dma_write_bytes)},
        {"DMA Engine Busy Cycles [cycles]", report::fmt_u64(stats.dma_busy_cycles)},
        {"DMA Engine Queue Wait [cycles]", report::fmt_u64(stats.dma_qwait_cycles)},
        {"Total Memory Requests [requests]", report::fmt_u64(stats.memory_reqs)},
        {"Total Memory Busy Cycles [cycles]", report::fmt_u64(stats.memory_busy_cycles)},
    });

    report::print_section_title(os, "Overall Summary");
    report::print_fields(os, {
        {"Total Elapsed Cycles [cycles]", report::fmt_u64(stats.total_elapsed)},
        {"Total Matrix Accelerator Requests [requests]", report::fmt_u64(stats.mat_req_total)},
        {"Total Vector Accelerator Requests [requests]", report::fmt_u64(stats.vec_req_total)},
        {"Total Memory Requests (L1+DMA) [requests]", report::fmt_u64(stats.memory_reqs)},
        {"Total Read Bytes [bytes]", report::fmt_u64(stats.total_rd_bytes)},
        {"Total Write Bytes [bytes]", report::fmt_u64(stats.total_wr_bytes)},
        {"Total Stall Cycles [cycles]", report::fmt_u64(total_stall_cycles)},
        {"Total Memory Cycles [cycles]", report::fmt_u64(total_mem_cycles)},
        {"Total Scalar Cycles [cycles]", report::fmt_u64(total_scalar_cycles)},
        {"Total MACs [operations]", report::fmt_u64(stats.total_macs)},
        {"Mat Phase Critical Path [cycles]", report::fmt_u64(stats.max_mat_elapsed)},
        {"Post-Mat Overhead [cycles]", report::fmt_u64(stats.accum_overhead)},
        {"Throughput @ 1 GHz [GFLOPS]", report::fmt_double(stats.gflops)},
        {"Average Memory Bandwidth [bytes/cycle]", report::fmt_rate(stats.mem_bw, "bytes/cycle")},
        {"Post-Mat Stall Cycles [cycles]", report::fmt_u64(stats.coordinator_stall)},
    });

    report::print_section_title(os, "Verification");
    report::print_fields(os, {
        {"Expected Matrix Accelerator Requests [requests]", report::fmt_u64(stats.expected_mat_req_total)},
        {"Actual Matrix Accelerator Requests [requests]", report::fmt_u64(stats.mat_req_total)},
        {"Expected Vector Accelerator Requests [requests]", report::fmt_u64(stats.expected_vec_req_total)},
        {"Actual Vector Accelerator Requests [requests]", report::fmt_u64(stats.vec_req_total)},
        {"Expected L1 Read Bytes [bytes]", report::fmt_u64(stats.expected_l1_read_bytes)},
        {"Actual L1 Read Bytes [bytes]", report::fmt_u64(stats.l1_read_bytes)},
        {"Expected L1 Write Bytes [bytes]", report::fmt_u64(stats.expected_l1_write_bytes)},
        {"Actual L1 Write Bytes [bytes]", report::fmt_u64(stats.l1_write_bytes)},
        {"Expected DMA Read Bytes [bytes]", report::fmt_u64(stats.expected_dma_read_bytes)},
        {"Actual DMA Read Bytes [bytes]", report::fmt_u64(stats.dma_read_bytes)},
        {"Expected DMA Write Bytes [bytes]", report::fmt_u64(stats.expected_dma_write_bytes)},
        {"Actual DMA Write Bytes [bytes]", report::fmt_u64(stats.dma_write_bytes)},
        {"Verification Status", stats.verification_passed ? "PASS" : "FAIL"},
    });
}

void MatmulTop::done_monitor()
{
    for (int i = 0; i < cfg.thread_count; ++i)
        completion_fifo->read();
    done_event->notify(SC_ZERO_TIME);
}
