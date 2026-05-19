#include "layer_norm_top.h"

#include <systemc>
#include <tlm>

#include <algorithm>
#include <deque>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "common.h"
#include "extensions.h"
#include "interconnect.h"
#include "report_formatter.h"
#include "worker.h"

using namespace sc_core;
using namespace tlm;

// ============================================================
// LayerNormPostProcessor - runs inside each Worker's SC_THREAD
// via WorkerPostProcessor.
//
// Per Layer_Norm.md v1 the kernel walks each owned channel in
// three passes (sum -> variance -> normalize). One L2 prefetch
// brings the channel input into L1, the three passes then re-
// read from L1, and a single L2 writeback stores the int8 result.
// The intermediate int32 sum / sum-of-squares live in the
// pinned vec unit's register across the tiles of one pass.
// ============================================================
struct LayerNormPostProcessor : WorkerPostProcessor
{
    const LayerNormRuntimeConfig &cfg;
    int tid;
    int n_workers;
    int my_unit_id = 0;

    // Per-worker L2-DMA byte / cycle accumulators (shared Worker
    // tracks compute / mem cycles but not L2-DMA bytes).
    uint64_t total_l2_rd_bytes = 0;
    uint64_t total_l2_wr_bytes = 0;
    uint64_t total_dma_cycles  = 0;

    // Per-pass vec request counters (V2 reporting).
    uint64_t pass1_reqs = 0;
    uint64_t pass2_reqs = 0;
    uint64_t pass3_reqs = 0;

    LayerNormPostProcessor(const LayerNormRuntimeConfig &cfg_,
                           int tid_,
                           int n_workers_,
                           int my_unit_id_)
        : cfg(cfg_),
          tid(tid_),
          n_workers(n_workers_),
          my_unit_id(my_unit_id_) {}

    // Run one of the three passes over a single channel resident
    // in L1. Each pass issues tile_count vec calls. Pass 3 carries
    // the L1 writeback payload on every tile (vl * output_elem_bytes);
    // passes 1/2 only read.
    void run_pass(Worker &worker, int pass, uint64_t &vec_calls_local)
    {
        const uint64_t spatial = static_cast<uint64_t>(cfg.spatial());
        const int tiles = cfg.tile_count();
        const size_t window =
            static_cast<size_t>(std::max<uint64_t>(cfg.max_inflight_vec_reqs, 1));

        // Per-pass service cycles = n_compute_intrinsics * vec_insn_cycle.
        const uint64_t svc_cycles = cfg.pass_cycles(pass);
        uint64_t *pass_counter =
            (pass == 1) ? &pass1_reqs :
            (pass == 2) ? &pass2_reqs :
                          &pass3_reqs;

        std::deque<Worker::PendingReq> inflight;

        auto retire_oldest = [&]() {
            worker.issue_end(inflight.front());
            inflight.pop_front();
        };

        for (int t = 0; t < tiles; ++t)
        {
            const uint64_t vl = std::min<uint64_t>(
                cfg.vec_acc_cap,
                spatial - static_cast<uint64_t>(t) * cfg.vec_acc_cap);
            const uint64_t rd = vl * cfg.input_elem_bytes;
            const uint64_t wr =
                (pass == 3) ? vl * cfg.output_elem_bytes : 0ULL;

            // Maintain the inflight window: retire oldest when full.
            if (inflight.size() >= window)
                retire_oldest();

            auto req = worker.issue_begin(Interconnect::ADDR_VEC,
                                          svc_cycles,
                                          rd, wr);
            ++worker.vec_calls;
            ++vec_calls_local;
            ++(*pass_counter);
            inflight.push_back(std::move(req));

            worker.do_scalar(cfg.scalar_overhead);
        }

        // Drain the rest of the pass before the next scalar
        // mean / inv_std math (the result depends on every tile).
        while (!inflight.empty())
            retire_oldest();
    }

    void run_post_mat(Worker &worker) override
    {
        const int c_start = (tid * cfg.channels) / n_workers;
        const int c_end   = ((tid + 1) * cfg.channels) / n_workers;
        if (c_start >= c_end)
            return;

        const uint64_t spatial = static_cast<uint64_t>(cfg.spatial());
        const uint64_t channel_in_bytes  = spatial * cfg.input_elem_bytes;
        const uint64_t channel_out_bytes = spatial * cfg.output_elem_bytes;
        const size_t max_dma_writes =
            static_cast<size_t>(
                std::max<uint64_t>(cfg.max_inflight_dma_writes, 1));

        worker.configure_dma_vec_cost(cfg.dma_vec_rd_scalar,
                                      cfg.dma_vec_wr_scalar);

        std::deque<Worker::DmaReq> write_inflight;
        uint64_t vec_calls_local = 0;

        for (int c = c_start; c < c_end; ++c)
        {
            // --- (1) L2 -> L1 prefetch of the channel input. ----
            // Charge the per-DMA scalar before issue, mirroring
            // dw_conv2d's prefetch path.
            if (cfg.dma_vec_rd_scalar > 0)
                worker.do_scalar(cfg.dma_vec_rd_scalar);
            const sc_time wait_start = sc_time_stamp();
            Worker::DmaReq rd = worker.issue_dma_begin(false, channel_in_bytes);
            worker.finish_dma(rd);
            total_dma_cycles += static_cast<uint64_t>(
                (sc_time_stamp() - wait_start) / CYCLE);
            total_l2_rd_bytes += channel_in_bytes;

            // --- (2) Pass 1: sum (running sum in vec register). -
            run_pass(worker, /*pass=*/1, vec_calls_local);

            // --- (3) Scalar mean = sum / spatial. ---------------
            if (cfg.mean_cycles > 0)
                worker.do_scalar(cfg.mean_cycles);

            // --- (4) Pass 2: sum-of-squares (about mean). -------
            run_pass(worker, /*pass=*/2, vec_calls_local);

            // --- (5) Scalar var/N + isqrt + 1/std. --------------
            if (cfg.invstd_cycles > 0)
                worker.do_scalar(cfg.invstd_cycles);

            // --- (6) Pass 3: normalize and writeback to L1. -----
            run_pass(worker, /*pass=*/3, vec_calls_local);

            // --- (7) Fire-and-forget L1 -> L2 channel store. ----
            if (cfg.dma_vec_wr_scalar > 0)
                worker.do_scalar(cfg.dma_vec_wr_scalar);
            if (write_inflight.size() >= max_dma_writes)
            {
                worker.finish_dma(write_inflight.front());
                write_inflight.pop_front();
            }
            Worker::DmaReq w =
                worker.issue_dma_begin(true, channel_out_bytes);
            total_l2_wr_bytes += channel_out_bytes;
            write_inflight.push_back(std::move(w));
        }

        while (!write_inflight.empty())
        {
            worker.finish_dma(write_inflight.front());
            write_inflight.pop_front();
        }
    }
};

// ============================================================
// Worker-to-vec-unit pin map: round-robin (worker i -> unit i % U).
// Matches dw_conv2d's mapping so each unit serves
// worker_count / effective_vec_instances workers.
// ============================================================
static std::vector<int> build_round_robin_map(int n_workers, int n_units)
{
    std::vector<int> m;
    m.reserve(static_cast<size_t>(std::max(n_workers, 0)));
    const int units = std::max(n_units, 1);
    for (int i = 0; i < n_workers; ++i)
        m.push_back(i % units);
    return m;
}

LayerNormTop::LayerNormTop(sc_module_name name,
                           const LayerNormRuntimeConfig &cfg_,
                           sc_event *start_event,
                           sc_event *done_event_)
    : sc_module(name),
      cfg(cfg_),
      mat_acc("mat_acc", cfg.acc_queue_depth),
      vec_acc("vec_acc",
              static_cast<size_t>(cfg.effective_vec_instances()),
              cfg.acc_queue_depth,
              build_round_robin_map(cfg.worker_count,
                                    cfg.effective_vec_instances()),
              std::vector<uint64_t>(
                  static_cast<size_t>(cfg.effective_vec_instances()), 0)),
      noc("noc"),
      memory("memory",
             cfg.l1_base_lat,
             cfg.l1_bw,
             cfg.l2_base_lat,
             cfg.l2_bw,
             cfg.l1_slots,
             cfg.l2_slots),
      done_event(done_event_)
{
    noc.to_mat.bind(mat_acc.tgt);
    noc.to_vec.bind(vec_acc.tgt);
    noc.to_mem.bind(memory.tgt);

    mat_acc.to_mem.bind(noc.tgt);
    for (auto &unit : vec_acc.units)
        unit->to_mem.bind(noc.tgt);

    if (done_event)
    {
        completion_fifo =
            std::make_unique<sc_fifo<int>>(sc_gen_unique_name("ln_done_fifo"),
                                           cfg.worker_count + 1);
        SC_THREAD(done_monitor);
    }

    const int n_units = cfg.effective_vec_instances();
    post_processors.reserve(static_cast<size_t>(cfg.worker_count));
    workers.reserve(static_cast<size_t>(cfg.worker_count));

    for (int i = 0; i < cfg.worker_count; ++i)
    {
        const int my_unit_id = i % n_units;
        post_processors.push_back(std::make_unique<LayerNormPostProcessor>(
            cfg, i, cfg.worker_count, my_unit_id));
        auto *w = new Worker(sc_gen_unique_name("ln_worker"),
                             /*tid=*/i,
                             /*access_mat=*/0,
                             /*access_vec=*/0,
                             /*mat_cycles=*/0,
                             /*vec_cycles=*/cfg.vec_insn_cycle,
                             /*mat_scalar_cycles=*/0,
                             /*vec_scalar_cycles=*/cfg.scalar_overhead,
                             /*A_bytes=*/0,
                             /*B_bytes=*/0,
                             /*C_bytes=*/0,
                             /*vec_rd=*/0,
                             /*vec_wr=*/0,
                             /*max_inflight_mat=*/1,
                             /*max_inflight_vec=*/cfg.max_inflight_vec_reqs,
                             /*post_processor=*/post_processors.back().get(),
                             /*start_event=*/start_event,
                             /*completion_fifo=*/completion_fifo.get());
        workers.push_back(w);
        w->init.bind(noc.tgt);
    }
}

LayerNormTop::~LayerNormTop()
{
    for (auto *w : workers)
        delete w;
}

// ------------------------------------------------------------
// Expected counters - three passes per channel over tile_count
// vec calls. Pass 3 writes vl * output_elem_bytes per call.
// One L2 prefetch + one L2 writeback per channel.
// ------------------------------------------------------------
static void compute_expected(const LayerNormRuntimeConfig &cfg,
                             LayerNormSimulationStats &s)
{
    const uint64_t spatial = static_cast<uint64_t>(cfg.spatial());
    const uint64_t tiles_per_pass = static_cast<uint64_t>(cfg.tile_count());
    const uint64_t channels = static_cast<uint64_t>(cfg.channels);

    s.expected_pass1_reqs = channels * tiles_per_pass;
    s.expected_pass2_reqs = channels * tiles_per_pass;
    s.expected_pass3_reqs = channels * tiles_per_pass;
    s.expected_vec_reqs =
        s.expected_pass1_reqs + s.expected_pass2_reqs + s.expected_pass3_reqs;

    // Total vec-unit busy cycles = sum over passes of (reqs * pass_cycles).
    s.expected_vec_acc_busy_cycles =
        s.expected_pass1_reqs * cfg.pass_cycles(1) +
        s.expected_pass2_reqs * cfg.pass_cycles(2) +
        s.expected_pass3_reqs * cfg.pass_cycles(3);
    // Every vec call emits exactly one L1 read of the tile's input
    // bytes. With pass-3 carrying L1 writes, the running totals are:
    //   L1 read bytes  = 3 * C * spatial * input_elem_bytes
    //   L1 write bytes = 1 * C * spatial * output_elem_bytes
    s.expected_l1_read_bytes  = 3ULL * channels * spatial * cfg.input_elem_bytes;
    s.expected_l1_write_bytes = channels * spatial * cfg.output_elem_bytes;
    // L1 reqs = one read per vec call across all three passes + one
    // write per pass-3 vec call.
    s.expected_l1_reqs =
        s.expected_vec_reqs + channels * tiles_per_pass;

    s.expected_l2_dma_reqs   = 2ULL * channels;
    s.expected_l2_read_bytes = channels * spatial * cfg.input_elem_bytes;
    s.expected_l2_write_bytes = channels * spatial * cfg.output_elem_bytes;
}

LayerNormSimulationStats LayerNormTop::collect_stats() const
{
    LayerNormSimulationStats stats;
    compute_expected(cfg, stats);

    for (size_t i = 0; i < workers.size(); ++i)
    {
        const Worker *w = workers[i];
        const LayerNormPostProcessor *pp = post_processors[i].get();
        stats.max_elapsed_cycles =
            std::max(stats.max_elapsed_cycles, w->elapsed_cycles);
        stats.total_vec_reqs   += w->vec_calls;
        stats.total_wait_cycles += w->wait_cycles;
        if (pp)
        {
            stats.total_pass1_reqs += pp->pass1_reqs;
            stats.total_pass2_reqs += pp->pass2_reqs;
            stats.total_pass3_reqs += pp->pass3_reqs;
            stats.total_rd_bytes  += pp->total_l2_rd_bytes;
            stats.total_wr_bytes  += pp->total_l2_wr_bytes;
            stats.total_mem_cycles += pp->total_dma_cycles +
                                       w->mem_cycles_accum;
        }
    }

    stats.vec_acc_reqs              = vec_acc.req_count_total();
    stats.vec_acc_busy_cycles       = vec_acc.busy_cycles_total();
    stats.vec_acc_occupied_cycles   = vec_acc.occupied_cycles_total();
    stats.vec_acc_queue_wait_cycles = vec_acc.queue_wait_cycles_total();

    stats.l1_reqs              = memory.l1_reqs;
    stats.l1_read_bytes        = memory.l1_read_bytes;
    stats.l1_write_bytes       = memory.l1_write_bytes;
    stats.l1_busy_cycles       = memory.l1_busy_cycles;
    stats.l1_queue_wait_cycles = memory.l1_qwait_cycles;

    stats.l2_dma_reqs              = memory.dma_reqs;
    stats.l2_dma_read_bytes        = memory.dma_read_bytes;
    stats.l2_dma_write_bytes       = memory.dma_write_bytes;
    stats.l2_dma_busy_cycles       = memory.dma_busy_cycles;
    stats.l2_dma_queue_wait_cycles = memory.dma_qwait_cycles;

    // Back-compat aggregates for the nafnet bridge convert_stats().
    stats.memory_reqs              = stats.l2_dma_reqs;
    stats.memory_busy_cycles       = stats.l2_dma_busy_cycles;
    stats.memory_queue_wait_cycles = stats.l2_dma_queue_wait_cycles;

    const double sim_cycles = static_cast<double>(sc_time_stamp() / CYCLE);
    const double vec_capacity =
        sim_cycles * static_cast<double>(vec_acc.instance_count());
    stats.vec_util = (vec_capacity > 0.0)
        ? static_cast<double>(stats.vec_acc_busy_cycles) / vec_capacity * 100.0
        : 0.0;
    stats.vec_occupancy = (vec_capacity > 0.0)
        ? static_cast<double>(stats.vec_acc_occupied_cycles) / vec_capacity * 100.0
        : 0.0;
    stats.l1_bw_observed = (sim_cycles > 0.0)
        ? static_cast<double>(stats.l1_read_bytes + stats.l1_write_bytes) /
              sim_cycles
        : 0.0;
    stats.l2_bw_observed = (sim_cycles > 0.0)
        ? static_cast<double>(stats.l2_dma_read_bytes + stats.l2_dma_write_bytes) /
              sim_cycles
        : 0.0;

    int slowest_idx = -1;
    uint64_t slowest_cycles = 0;
    for (size_t i = 0; i < workers.size(); ++i)
        if (workers[i]->elapsed_cycles > slowest_cycles)
        {
            slowest_cycles = workers[i]->elapsed_cycles;
            slowest_idx = static_cast<int>(i);
        }
    stats.slowest_worker_tid =
        (slowest_idx >= 0) ? workers[static_cast<size_t>(slowest_idx)]->tid : -1;

    if (slowest_idx >= 0)
    {
        const Worker *sw = workers[static_cast<size_t>(slowest_idx)];
        const LayerNormPostProcessor *spp =
            post_processors[static_cast<size_t>(slowest_idx)].get();
        // Per-pass insn counts -> per-pass cycle cost; sum over passes
        // gives the worker's total vec-unit service time.
        const uint64_t vec_service = spp
            ? spp->pass1_reqs * cfg.pass_cycles(1) +
              spp->pass2_reqs * cfg.pass_cycles(2) +
              spp->pass3_reqs * cfg.pass_cycles(3)
            : 0;
        const uint64_t dma_cycles  = (spp ? spp->total_dma_cycles : 0) +
                                     sw->mem_cycles_accum;
        const uint64_t scalar_cycles =
            (sw->compute_cycles >= sw->vec_service_cycles)
                ? (sw->compute_cycles - sw->vec_service_cycles) : 0;
        const uint64_t stall_cycles = sw->stall_cycles;

        stats.slowest_vec_cycles    = vec_service;
        stats.slowest_dma_cycles    = dma_cycles;
        stats.slowest_scalar_cycles = scalar_cycles;
        stats.slowest_stall_cycles  = stall_cycles;

        const uint64_t total_categorized =
            vec_service + dma_cycles + scalar_cycles + stall_cycles;
        if (total_categorized > 0)
        {
            const double denom = static_cast<double>(total_categorized);
            stats.vec_cycle_fraction    = vec_service   / denom * 100.0;
            stats.dma_cycle_fraction    = dma_cycles    / denom * 100.0;
            stats.scalar_cycle_fraction = scalar_cycles / denom * 100.0;
            stats.stall_cycle_fraction  = stall_cycles  / denom * 100.0;
        }
    }

    stats.verification_passed =
        stats.total_vec_reqs       == stats.expected_vec_reqs &&
        stats.total_pass1_reqs     == stats.expected_pass1_reqs &&
        stats.total_pass2_reqs     == stats.expected_pass2_reqs &&
        stats.total_pass3_reqs     == stats.expected_pass3_reqs &&
        stats.vec_acc_busy_cycles  == stats.expected_vec_acc_busy_cycles &&
        stats.l1_reqs              == stats.expected_l1_reqs &&
        stats.l1_read_bytes        == stats.expected_l1_read_bytes &&
        stats.l1_write_bytes       == stats.expected_l1_write_bytes &&
        stats.l2_dma_reqs          == stats.expected_l2_dma_reqs &&
        stats.l2_dma_read_bytes    == stats.expected_l2_read_bytes &&
        stats.l2_dma_write_bytes   == stats.expected_l2_write_bytes;
    return stats;
}

std::vector<KernelWorkerInfo> LayerNormTop::collect_worker_info() const
{
    std::vector<KernelWorkerInfo> info;
    info.reserve(workers.size());
    for (size_t i = 0; i < workers.size(); ++i)
    {
        const Worker *w = workers[i];
        const LayerNormPostProcessor *pp = post_processors[i].get();
        KernelWorkerInfo wi;
        wi.tid            = w->tid;
        wi.mat_reqs       = w->mat_calls;
        wi.vec_reqs       = w->vec_calls;
        wi.elapsed_cycles = w->elapsed_cycles;
        wi.stall_cycles   = w->stall_cycles;
        wi.mem_cycles     = (pp ? pp->total_dma_cycles : 0) +
                            w->mem_cycles_accum;
        const uint64_t service = w->vec_service_cycles;
        wi.scalar_cycles = (w->compute_cycles >= service)
            ? (w->compute_cycles - service) : 0;
        wi.rd_bytes = pp ? pp->total_l2_rd_bytes : 0;
        wi.wr_bytes = pp ? pp->total_l2_wr_bytes : 0;
        info.push_back(wi);
    }
    return info;
}

void LayerNormTop::print_report(std::ostream &os) const
{
    const LayerNormSimulationStats stats = collect_stats();
    const std::vector<KernelWorkerInfo> worker_info = collect_worker_info();

    uint64_t total_scalar_cycles = 0;
    uint64_t total_stall_cycles  = 0;
    uint64_t total_mem_cycles    = 0;
    for (const auto &worker : worker_info)
    {
        total_scalar_cycles += worker.scalar_cycles;
        total_stall_cycles  += worker.stall_cycles;
        total_mem_cycles    += worker.mem_cycles;
    }

    report::print_section_title(os, "Simulation Info");
    report::print_fields(os, {
        {"Operation Type", "Layer Normalization (int8, 3-pass)"},
        {"Input Tensor Shape", "[C=" + report::fmt_int(cfg.channels) +
                               ", H=" + report::fmt_int(cfg.height) +
                               ", W=" + report::fmt_int(cfg.width) + "]"},
        {"Input Element Type", "int8"},
        {"Output Element Type", "int8"},
    });

    const int pinned_vec_instances = cfg.effective_vec_instances();
    report::print_section_title(os, "Hardware Configuration");
    report::print_fields(os, {
        {"Workers [count]", report::fmt_int(cfg.worker_count)},
        {"Matrix Accelerators [count]", report::na()},
        {"Vector Accelerators [count]", report::fmt_int(pinned_vec_instances)},
        {"Worker->Vec Binding", "pinned round-robin (worker i -> unit i % U)"},
        {"Workers per Vec Unit", report::fmt_int(
            (cfg.worker_count + pinned_vec_instances - 1) /
            pinned_vec_instances)},
        {"Vector Accelerator Capacity [elements/request]",
         report::fmt_u64(cfg.vec_acc_cap)},
        {"Vector Instruction Cycle [cycles/insn]",
         report::fmt_u64(cfg.vec_insn_cycle)},
        {"Pass-1 Vector Instructions [insns/request]",
         report::fmt_u64(cfg.pass1_insns)},
        {"Pass-2 Vector Instructions [insns/request]",
         report::fmt_u64(cfg.pass2_insns)},
        {"Pass-3 Vector Instructions [insns/request]",
         report::fmt_u64(cfg.pass3_insns)},
        {"Accelerator Queue Depth [requests/unit]",
         report::fmt_u64(cfg.acc_queue_depth)},
        {"Max Inflight Vec Reqs / worker",
         report::fmt_u64(cfg.max_inflight_vec_reqs)},
        {"Max Inflight L2 DMA Writes / worker",
         report::fmt_u64(cfg.max_inflight_dma_writes)},
        {"L1 Bandwidth [bytes/cycle]", report::fmt_u64(cfg.l1_bw)},
        {"L1 Base Latency [cycles]", report::fmt_u64(cfg.l1_base_lat)},
        {"L1 Parallel Slots", report::fmt_u64(cfg.l1_slots)},
        {"L2 DMA Bandwidth [bytes/cycle]", report::fmt_u64(cfg.l2_bw)},
        {"L2 DMA Base Latency [cycles]", report::fmt_u64(cfg.l2_base_lat)},
        {"L2 DMA Parallel Slots", report::fmt_u64(cfg.l2_slots)},
    });

    report::print_section_title(os, "Worker Summary");
    report::print_table(os, report::make_worker_summary_table(worker_info));

    report::print_section_title(os, "Accelerator Summary");
    std::vector<report::AcceleratorSummaryRow> accel_rows;
    accel_rows.push_back({
        "Matrix Accelerator", "pool-level",
        report::na(), report::na(), report::na(), report::na(),
        report::na(), report::na(), report::na(), report::na(),
        report::na(), report::na(),
    });
    accel_rows.push_back({
        "Vector Accelerator", "pool-level",
        report::fmt_int(pinned_vec_instances),
        report::fmt_u64(stats.vec_acc_reqs),
        report::fmt_u64(stats.vec_acc_queue_wait_cycles),
        report::fmt_u64(stats.vec_acc_busy_cycles),
        report::fmt_u64(stats.vec_acc_occupied_cycles),
        report::fmt_percent(stats.vec_util),
        report::fmt_percent(stats.vec_occupancy),
        report::na(), report::na(), report::na(),
    });
    for (auto &r : report::make_per_instance_accel_rows(
             "Vector Accelerator", vec_acc.per_instance_stats(),
             stats.max_elapsed_cycles))
        accel_rows.push_back(std::move(r));
    accel_rows.push_back({
        "L1 Memory", "accelerator-side",
        report::fmt_int(pinned_vec_instances),
        report::fmt_u64(stats.l1_reqs),
        report::fmt_u64(stats.l1_queue_wait_cycles),
        report::fmt_u64(stats.l1_busy_cycles),
        report::na(), report::na(), report::na(),
        report::fmt_u64(stats.l1_read_bytes),
        report::fmt_u64(stats.l1_write_bytes),
        report::na(),
    });
    accel_rows.push_back({
        "L2 DMA", "prefetch/writeback", "1",
        report::fmt_u64(stats.l2_dma_reqs),
        report::fmt_u64(stats.l2_dma_queue_wait_cycles),
        report::fmt_u64(stats.l2_dma_busy_cycles),
        report::na(), report::na(), report::na(),
        report::fmt_u64(stats.l2_dma_read_bytes),
        report::fmt_u64(stats.l2_dma_write_bytes),
        report::na(),
    });
    report::print_table(os, report::make_accelerator_summary_table(accel_rows));

    report::print_section_title(os, "Overall Summary");
    report::print_fields(os, {
        {"Total Elapsed Cycles [cycles]",
         report::fmt_u64(stats.max_elapsed_cycles)},
        {"Total Matrix Accelerator Requests [requests]", report::na()},
        {"Total Vector Accelerator Requests [requests]",
         report::fmt_u64(stats.total_vec_reqs)},
        {"Total L1 Requests [requests]", report::fmt_u64(stats.l1_reqs)},
        {"Total L2 DMA Requests [requests]", report::fmt_u64(stats.l2_dma_reqs)},
        {"Total L1 Read Bytes [bytes]", report::fmt_u64(stats.l1_read_bytes)},
        {"Total L1 Write Bytes [bytes]", report::fmt_u64(stats.l1_write_bytes)},
        {"Total L2 DMA Read Bytes [bytes]",
         report::fmt_u64(stats.l2_dma_read_bytes)},
        {"Total L2 DMA Write Bytes [bytes]",
         report::fmt_u64(stats.l2_dma_write_bytes)},
        {"Total Stall Cycles [cycles]", report::fmt_u64(total_stall_cycles)},
        {"Total Memory Cycles [cycles]", report::fmt_u64(total_mem_cycles)},
        {"Total Scalar Cycles [cycles]", report::fmt_u64(total_scalar_cycles)},
        {"Average L1 Bandwidth [bytes/cycle]",
         report::fmt_rate(stats.l1_bw_observed, "bytes/cycle")},
        {"Average L2 DMA Bandwidth [bytes/cycle]",
         report::fmt_rate(stats.l2_bw_observed, "bytes/cycle")},
        {"Critical-Path Worker [tid]", report::fmt_int(stats.slowest_worker_tid)},
        {"Vec Cycle Fraction [%]",    report::fmt_percent(stats.vec_cycle_fraction)},
        {"DMA Cycle Fraction [%]",    report::fmt_percent(stats.dma_cycle_fraction)},
        {"Scalar Cycle Fraction [%]", report::fmt_percent(stats.scalar_cycle_fraction)},
        {"Stall Cycle Fraction [%]",  report::fmt_percent(stats.stall_cycle_fraction)},
    });

    report::print_section_title(os, "Verification");
    report::print_fields(os, {
        {"Expected Vector Accelerator Requests [requests]",
         report::fmt_u64(stats.expected_vec_reqs)},
        {"Actual Vector Accelerator Requests [requests]",
         report::fmt_u64(stats.total_vec_reqs)},
        {"Expected Pass-1 Vector Requests [requests]",
         report::fmt_u64(stats.expected_pass1_reqs)},
        {"Actual Pass-1 Vector Requests [requests]",
         report::fmt_u64(stats.total_pass1_reqs)},
        {"Expected Pass-2 Vector Requests [requests]",
         report::fmt_u64(stats.expected_pass2_reqs)},
        {"Actual Pass-2 Vector Requests [requests]",
         report::fmt_u64(stats.total_pass2_reqs)},
        {"Expected Pass-3 Vector Requests [requests]",
         report::fmt_u64(stats.expected_pass3_reqs)},
        {"Actual Pass-3 Vector Requests [requests]",
         report::fmt_u64(stats.total_pass3_reqs)},
        {"Expected Vector Accelerator Busy Cycles [cycles]",
         report::fmt_u64(stats.expected_vec_acc_busy_cycles)},
        {"Actual Vector Accelerator Busy Cycles [cycles]",
         report::fmt_u64(stats.vec_acc_busy_cycles)},
        {"Expected L1 Requests [requests]",
         report::fmt_u64(stats.expected_l1_reqs)},
        {"Actual L1 Requests [requests]", report::fmt_u64(stats.l1_reqs)},
        {"Expected L1 Read Bytes [bytes]",
         report::fmt_u64(stats.expected_l1_read_bytes)},
        {"Actual L1 Read Bytes [bytes]",
         report::fmt_u64(stats.l1_read_bytes)},
        {"Expected L1 Write Bytes [bytes]",
         report::fmt_u64(stats.expected_l1_write_bytes)},
        {"Actual L1 Write Bytes [bytes]",
         report::fmt_u64(stats.l1_write_bytes)},
        {"Expected L2 DMA Requests [requests]",
         report::fmt_u64(stats.expected_l2_dma_reqs)},
        {"Actual L2 DMA Requests [requests]",
         report::fmt_u64(stats.l2_dma_reqs)},
        {"Expected L2 DMA Read Bytes [bytes]",
         report::fmt_u64(stats.expected_l2_read_bytes)},
        {"Actual L2 DMA Read Bytes [bytes]",
         report::fmt_u64(stats.l2_dma_read_bytes)},
        {"Expected L2 DMA Write Bytes [bytes]",
         report::fmt_u64(stats.expected_l2_write_bytes)},
        {"Actual L2 DMA Write Bytes [bytes]",
         report::fmt_u64(stats.l2_dma_write_bytes)},
        {"Verification Status",
         stats.verification_passed ? "PASS" : "FAIL"},
    });
}

void LayerNormTop::done_monitor()
{
    for (int i = 0; i < cfg.worker_count; ++i)
        completion_fifo->read();
    done_event->notify(SC_ZERO_TIME);
}
