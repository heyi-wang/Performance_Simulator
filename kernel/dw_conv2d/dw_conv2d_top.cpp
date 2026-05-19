#include "dw_conv2d_top.h"

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
// DwConvPostProcessor (v2) — runs inside each Worker's SC_THREAD
// via WorkerPostProcessor.
//
// Per Dw_Conv2d.md Requirements v2, one output strip on a single
// channel emits Kh*Kw load-broadcast-compute requests to the
// vector unit. The last sub-request carries the L1 writeback
// payload (vl * output_elem_bytes); a fire-and-forget L2 DMA
// then writes the strip's output to L2. Sub-requests of one
// strip must be admitted contiguously to the assigned vec unit
// FIFO -- no other worker's strip may interleave between those
// sub-requests. Contiguity is enforced via a per-unit
// sc_semaphore held only while the worker submits the strip's
// sub-requests; completions are drained after releasing it.
// ============================================================
struct DwConvPostProcessor : WorkerPostProcessor
{
    const DwConvRuntimeConfig &cfg;
    int tid;
    int n_workers;
    int my_unit_id = 0;
    sc_semaphore *unit_lock = nullptr;

    // Per-worker byte / DMA-cycle accumulators (shared Worker
    // tracks compute/mem cycles but not L2-DMA cycles or bytes).
    uint64_t total_l1_rd_bytes = 0;
    uint64_t total_l1_wr_bytes = 0;
    uint64_t total_l2_rd_bytes = 0;
    uint64_t total_l2_wr_bytes = 0;
    uint64_t total_dma_cycles  = 0;

    DwConvPostProcessor(const DwConvRuntimeConfig &cfg_,
                        int tid_,
                        int n_workers_,
                        int my_unit_id_,
                        sc_semaphore *unit_lock_)
        : cfg(cfg_),
          tid(tid_),
          n_workers(n_workers_),
          my_unit_id(my_unit_id_),
          unit_lock(unit_lock_) {}

    // In-bounds input bytes for one (kh, kw) of one strip's vector
    // load (vl elements). Out-of-bounds region contributes 0 -- the
    // RVV kernel either skips or loads from the zero-padded
    // workspace, so the model charges only in-bounds traffic.
    uint64_t sub_rd_bytes(int oh,
                          uint64_t strip_start,
                          uint64_t vl,
                          int kh,
                          int kw) const
    {
        int ih = oh - cfg.pad + kh;
        if (ih < 0 || ih >= cfg.height)
            return 0;
        int64_t iw_base =
            static_cast<int64_t>(strip_start) - cfg.pad + kw;
        int64_t lo = std::max<int64_t>(0, iw_base);
        int64_t hi = std::min<int64_t>(
            static_cast<int64_t>(cfg.width),
            iw_base + static_cast<int64_t>(vl));
        if (hi <= lo)
            return 0;
        return static_cast<uint64_t>(hi - lo) * cfg.input_elem_bytes;
    }

    // Geometry for one strip in the flat (c, oh, st) iteration. The
    // flat enumeration lets the per-worker prefetch lookahead pipeline
    // straddle row and channel boundaries without resetting.
    struct StripGeom
    {
        int c = 0;
        int oh = 0;
        int st = 0;
        uint64_t strip_start = 0;
        uint64_t vl = 0;
        uint64_t prefetch_bytes = 0;
        uint64_t wr_bytes_strip = 0;
        bool first_of_channel = false;
    };

    void run_post_mat(Worker &worker) override
    {
        const int c_start = (tid * cfg.channels) / n_workers;
        const int c_end   = ((tid + 1) * cfg.channels) / n_workers;
        if (c_start >= c_end)
            return;

        const uint64_t kernel_rd_bytes =
            static_cast<uint64_t>(cfg.kernel_h * cfg.kernel_w) *
            cfg.input_elem_bytes;
        const size_t max_dma_writes =
            static_cast<size_t>(
                std::max<uint64_t>(cfg.max_inflight_dma_writes, 1));

        worker.configure_dma_vec_cost(cfg.dma_vec_rd_scalar,
                                      cfg.dma_vec_wr_scalar);

        std::deque<Worker::DmaReq> write_inflight;

        const int strips_per_row    = cfg.strips_per_row();
        const int rows              = cfg.out_h();
        const int strips_per_channel = rows * strips_per_row;
        const int n_channels        = c_end - c_start;
        const int n_strips_total    = n_channels * strips_per_channel;
        if (n_strips_total == 0)
            return;

        auto strip_geom = [&](int idx) {
            StripGeom g;
            const int c_local = idx / strips_per_channel;
            const int rem     = idx % strips_per_channel;
            g.c                = c_start + c_local;
            g.oh               = rem / strips_per_row;
            g.st               = rem % strips_per_row;
            g.first_of_channel = (rem == 0);
            g.strip_start      =
                static_cast<uint64_t>(g.st) * cfg.vec_acc_cap;
            g.vl = std::min<uint64_t>(
                cfg.vec_acc_cap,
                static_cast<uint64_t>(cfg.out_w()) - g.strip_start);
            g.wr_bytes_strip = g.vl * cfg.output_elem_bytes;
            uint64_t pb = 0;
            for (int kh = 0; kh < cfg.kernel_h; ++kh)
                for (int kw = 0; kw < cfg.kernel_w; ++kw)
                    pb += sub_rd_bytes(g.oh, g.strip_start, g.vl, kh, kw);
            if (g.first_of_channel)
                pb += kernel_rd_bytes;
            g.prefetch_bytes = pb;
            return g;
        };

        // --- Bootstrap: fire strip 0's prefetch before entering the loop. ---
        // The loop body then maintains a one-deep prefetch pipeline:
        // while we hold the unit lock / drain / writeback for strip s,
        // strip s+1's DMA is already inflight. By the time we loop back
        // to finish_dma it usually returns immediately.
        Worker::DmaReq pending_read;
        {
            const StripGeom g0 = strip_geom(0);
            if (cfg.dma_vec_rd_scalar > 0)
                worker.do_scalar(cfg.dma_vec_rd_scalar);
            pending_read = worker.issue_dma_begin(false, g0.prefetch_bytes);
        }

        for (int idx = 0; idx < n_strips_total; ++idx)
        {
            const StripGeom g = strip_geom(idx);

            // --- (1) Block until THIS strip's prefetch lands. -----
            // Account only the wall-clock spent inside finish_dma so
            // dma_cycles tracks worker-blocking time, not the DMA's
            // total inflight span (which now overlaps with submit/drain).
            const sc_time wait_start = sc_time_stamp();
            worker.finish_dma(pending_read);
            total_dma_cycles += static_cast<uint64_t>(
                (sc_time_stamp() - wait_start) / CYCLE);
            total_l2_rd_bytes += g.prefetch_bytes;

            // --- (1b) Launch NEXT strip's prefetch (lookahead). ---
            // Fired BEFORE acquiring the unit lock so its inflight
            // window covers lock-hold + drain + writeback scalar.
            if (idx + 1 < n_strips_total)
            {
                const StripGeom gn = strip_geom(idx + 1);
                if (cfg.dma_vec_rd_scalar > 0)
                    worker.do_scalar(cfg.dma_vec_rd_scalar);
                pending_read = worker.issue_dma_begin(false, gn.prefetch_bytes);
            }

            // --- (2) Acquire per-unit lock for admission contiguity --
            // Hold the lock across all kh*kw issue_begin calls, including
            // any queue-full stalls, so no other worker's strip can be
            // admitted to this unit between this strip's sub-requests.
            if (unit_lock)
                unit_lock->wait();

            // --- (3) Issue kh*kw sub-requests --------------
            // The last sub-request carries the L1 writeback
            // (vl * output_elem_bytes); earlier ones write
            // nothing (accumulator stays in the unit's
            // register and is committed on the final req).
            std::deque<Worker::PendingReq> pending;
            const int total_sub = cfg.kernel_h * cfg.kernel_w;
            int sub_idx = 0;
            for (int kh = 0; kh < cfg.kernel_h; ++kh)
            {
                for (int kw = 0; kw < cfg.kernel_w; ++kw, ++sub_idx)
                {
                    const uint64_t rd_sub =
                        sub_rd_bytes(g.oh, g.strip_start, g.vl, kh, kw);
                    const bool is_last = (sub_idx == total_sub - 1);
                    const uint64_t wr_sub =
                        is_last ? g.wr_bytes_strip : 0ULL;

                    auto req = worker.issue_begin(
                        Interconnect::ADDR_VEC,
                        cfg.vec_acc_cycle,
                        rd_sub,
                        wr_sub);
                    ++worker.vec_calls;
                    total_l1_rd_bytes += rd_sub;
                    total_l1_wr_bytes += wr_sub;
                    pending.push_back(std::move(req));

                    // Per-request scalar dispatch overhead
                    // (matches matmul/pooling per-vec-call).
                    worker.do_scalar(cfg.scalar_overhead);
                }
            }

            // --- (4) Release unit lock ---------------------
            // The strip is now fully admitted/enqueued. Let the next
            // worker queue behind it while this worker drains responses.
            if (unit_lock)
                unit_lock->post();

            // Drain all kh*kw sub-requests. Their internal
            // load/compute/write stages overlap via the
            // AcceleratorTLM's pipelined unit threads.
            for (auto &req : pending)
                worker.issue_end(req);

            // --- (5) Fire-and-forget L2 writeback ----------
            // Per the user's spec: charge the writeback DMA
            // scalar overhead at the end of the last
            // sub-request, then non-blocking L2 DMA.
            if (cfg.dma_vec_wr_scalar > 0)
                worker.do_scalar(cfg.dma_vec_wr_scalar);

            if (write_inflight.size() >= max_dma_writes)
            {
                worker.finish_dma(write_inflight.front());
                write_inflight.pop_front();
            }
            Worker::DmaReq w =
                worker.issue_dma_begin(true, g.wr_bytes_strip);
            total_l2_wr_bytes += g.wr_bytes_strip;
            write_inflight.push_back(std::move(w));
        }

        // Drain remaining fire-and-forget L2 writes.
        while (!write_inflight.empty())
        {
            worker.finish_dma(write_inflight.front());
            write_inflight.pop_front();
        }
    }
};

// ============================================================
// Worker-to-vec-unit pin map: round-robin (worker i -> unit i % U).
// With defaults (16 workers, 4 units) each unit serves 4 workers.
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

DwConvTop::DwConvTop(sc_module_name name,
                     const DwConvRuntimeConfig &cfg_,
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
            std::make_unique<sc_fifo<int>>(sc_gen_unique_name("dw_done_fifo"),
                                           cfg.worker_count + 1);
        SC_THREAD(done_monitor);
    }

    // Per-unit strip-contiguity locks (one binary semaphore per
    // pinned vec unit). A worker holds its assigned unit's lock
    // across the kh*kw sub-request submission and drain.
    const int n_units = cfg.effective_vec_instances();
    per_unit_lock.reserve(static_cast<size_t>(n_units));
    for (int u = 0; u < n_units; ++u)
        per_unit_lock.push_back(std::make_unique<sc_semaphore>(
            sc_gen_unique_name("dw_unit_lock"), 1));

    post_processors.reserve(static_cast<size_t>(cfg.worker_count));
    workers.reserve(static_cast<size_t>(cfg.worker_count));

    for (int i = 0; i < cfg.worker_count; ++i)
    {
        const int my_unit_id = i % n_units;
        post_processors.push_back(std::make_unique<DwConvPostProcessor>(
            cfg, i, cfg.worker_count,
            my_unit_id,
            per_unit_lock[static_cast<size_t>(my_unit_id)].get()));
        auto *w = new Worker(sc_gen_unique_name("dw_worker"),
                             /*tid=*/i,
                             /*access_mat=*/0,
                             /*access_vec=*/0,
                             /*mat_cycles=*/0,
                             /*vec_cycles=*/cfg.vec_acc_cycle,
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

DwConvTop::~DwConvTop()
{
    for (auto *w : workers)
        delete w;
}

// ------------------------------------------------------------
// Expected stats (v2) -- mirrors the new per-strip request shape:
//   per strip: Kh*Kw vec sub-requests; the sum of their L1 reads
//              equals the in-bounds input bytes (+ kernel bytes
//              on the first strip of the first row per channel);
//              one L1 write of vl*output_elem_bytes carried by
//              the last sub-request; 1 L2 DMA read prefetch + 1
//              L2 DMA write per strip.
// L1 request count per strip = Kh*Kw L1 reads + 1 L1 write
//                            = vec_calls/strip + 1.
// ------------------------------------------------------------
static void compute_expected(const DwConvRuntimeConfig &cfg,
                             DwConvSimulationStats &s)
{
    const uint64_t kernel_rd =
        static_cast<uint64_t>(cfg.kernel_h * cfg.kernel_w) *
        cfg.input_elem_bytes;
    const int strips = cfg.strips_per_row();
    const uint64_t sub_per_strip =
        static_cast<uint64_t>(cfg.kernel_h) *
        static_cast<uint64_t>(cfg.kernel_w);

    uint64_t total_strips        = 0;
    uint64_t total_nonzero_l1_rd = 0;  // sub-reqs with rd_sub > 0
    for (int c = 0; c < cfg.channels; ++c)
    {
        for (int oh = 0; oh < cfg.out_h(); ++oh)
        {
            for (int st = 0; st < strips; ++st)
            {
                const uint64_t strip_start =
                    static_cast<uint64_t>(st) * cfg.vec_acc_cap;
                const uint64_t vl = std::min<uint64_t>(
                    cfg.vec_acc_cap,
                    static_cast<uint64_t>(cfg.out_w()) - strip_start);

                uint64_t rd_in_bounds = 0;
                for (int kh = 0; kh < cfg.kernel_h; ++kh)
                {
                    int ih = oh - cfg.pad + kh;
                    if (ih < 0 || ih >= cfg.height)
                        continue;

                    for (int kw = 0; kw < cfg.kernel_w; ++kw)
                    {
                        int64_t iw_base =
                            static_cast<int64_t>(strip_start) - cfg.pad + kw;
                        int64_t lo = std::max<int64_t>(0, iw_base);
                        int64_t hi = std::min<int64_t>(
                            static_cast<int64_t>(cfg.width),
                            iw_base + static_cast<int64_t>(vl));
                        if (hi > lo)
                        {
                            rd_in_bounds += static_cast<uint64_t>(hi - lo) *
                                            cfg.input_elem_bytes;
                            ++total_nonzero_l1_rd;
                        }
                    }
                }
                const uint64_t wr = vl * cfg.output_elem_bytes;
                // L1 reads only carry in-bounds input bytes (no kernel
                // weights -- those ride on the L2 prefetch DMA below).
                s.expected_l1_read_bytes  += rd_in_bounds;
                s.expected_l1_write_bytes += wr;
                // L2 prefetch DMA additionally carries kernel weight
                // bytes on the first strip of the first row per channel.
                uint64_t l2_rd = rd_in_bounds;
                if (oh == 0 && st == 0)
                    l2_rd += kernel_rd;
                s.expected_l2_read_bytes  += l2_rd;
                s.expected_l2_write_bytes += wr;
                s.expected_vec_calls      += sub_per_strip;
                ++total_strips;
            }
        }
    }
    // L1 reqs = sub-requests that actually emit an L1 read (rd_sub > 0)
    // plus one L1 write per strip (the last sub-request's writeback).
    // Sub-requests where the entire kh-row is out of bounds emit no L1
    // read; that's why we count them explicitly above.
    s.expected_l1_reqs = total_nonzero_l1_rd + total_strips;
    // 1 L2 DMA read (prefetch) + 1 L2 DMA write (writeback) per strip.
    s.expected_l2_dma_reqs = 2 * total_strips;
}

DwConvSimulationStats DwConvTop::collect_stats() const
{
    DwConvSimulationStats stats;
    compute_expected(cfg, stats);

    for (size_t i = 0; i < workers.size(); ++i)
    {
        const Worker *w = workers[i];
        const DwConvPostProcessor *pp = post_processors[i].get();
        stats.max_elapsed_cycles =
            std::max(stats.max_elapsed_cycles, w->elapsed_cycles);
        stats.total_vec_calls   += w->vec_calls;
        stats.total_wait_cycles += w->wait_cycles;
        if (pp)
        {
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

    stats.l1_reqs            = memory.l1_reqs;
    stats.l1_read_bytes      = memory.l1_read_bytes;
    stats.l1_write_bytes     = memory.l1_write_bytes;
    stats.l1_busy_cycles     = memory.l1_busy_cycles;
    stats.l1_queue_wait_cycles = memory.l1_qwait_cycles;

    stats.l2_dma_reqs            = memory.dma_reqs;
    stats.l2_dma_read_bytes      = memory.dma_read_bytes;
    stats.l2_dma_write_bytes     = memory.dma_write_bytes;
    stats.l2_dma_busy_cycles     = memory.dma_busy_cycles;
    stats.l2_dma_queue_wait_cycles = memory.dma_qwait_cycles;

    // Back-compat aggregates for nafnet bridge convert_stats()
    // (mirror pooling's convention: memory_* == L2 DMA).
    stats.memory_reqs             = stats.l2_dma_reqs;
    stats.memory_busy_cycles      = stats.l2_dma_busy_cycles;
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
        const DwConvPostProcessor *spp =
            post_processors[static_cast<size_t>(slowest_idx)].get();
        const uint64_t vec_service = sw->vec_calls * cfg.vec_acc_cycle;
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
        stats.total_vec_calls    == stats.expected_vec_calls &&
        stats.l1_reqs            == stats.expected_l1_reqs &&
        stats.l1_read_bytes      == stats.expected_l1_read_bytes &&
        stats.l1_write_bytes     == stats.expected_l1_write_bytes &&
        stats.l2_dma_reqs        == stats.expected_l2_dma_reqs &&
        stats.l2_dma_read_bytes  == stats.expected_l2_read_bytes &&
        stats.l2_dma_write_bytes == stats.expected_l2_write_bytes;
    return stats;
}

std::vector<KernelWorkerInfo> DwConvTop::collect_worker_info() const
{
    std::vector<KernelWorkerInfo> info;
    info.reserve(workers.size());
    for (size_t i = 0; i < workers.size(); ++i)
    {
        const Worker *w = workers[i];
        const DwConvPostProcessor *pp = post_processors[i].get();
        KernelWorkerInfo wi;
        wi.tid           = w->tid;
        wi.mat_reqs      = w->mat_calls;
        wi.vec_reqs      = w->vec_calls;
        wi.elapsed_cycles = w->elapsed_cycles;
        wi.stall_cycles  = w->stall_cycles;
        wi.mem_cycles    = (pp ? pp->total_dma_cycles : 0) +
                           w->mem_cycles_accum;
        // scalar_cycles = compute_cycles - vec_service_cycles
        // (compute_cycles aggregates do_scalar + service; matches matmul).
        const uint64_t service = w->vec_service_cycles;
        wi.scalar_cycles = (w->compute_cycles >= service)
            ? (w->compute_cycles - service) : 0;
        wi.rd_bytes      = pp ? pp->total_l2_rd_bytes : 0;
        wi.wr_bytes      = pp ? pp->total_l2_wr_bytes : 0;
        info.push_back(wi);
    }
    return info;
}

void DwConvTop::print_report(std::ostream &os) const
{
    const DwConvSimulationStats stats = collect_stats();
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
        {"Operation Type", "Depth-wise Convolution"},
        {"Input Tensor Shape", "[C=" + report::fmt_int(cfg.channels) +
                               ", H=" + report::fmt_int(cfg.height) +
                               ", W=" + report::fmt_int(cfg.width) + "]"},
        {"Kernel Shape", "[" + report::fmt_int(cfg.kernel_h) +
                         " x " + report::fmt_int(cfg.kernel_w) + "]"},
        {"Padding", report::fmt_int(cfg.pad)},
        {"Stride", report::fmt_int(cfg.stride)},
        {"Output Tensor Shape", "[C=" + report::fmt_int(cfg.channels) +
                                ", H=" + report::fmt_int(cfg.out_h()) +
                                ", W=" + report::fmt_int(cfg.out_w()) + "]"},
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
        {"Vector Accelerator Cycle [cycles/request]",
         report::fmt_u64(cfg.vec_acc_cycle)},
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
         report::fmt_u64(stats.total_vec_calls)},
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
         report::fmt_u64(stats.expected_vec_calls)},
        {"Actual Vector Accelerator Requests [requests]",
         report::fmt_u64(stats.total_vec_calls)},
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

void DwConvTop::done_monitor()
{
    for (int i = 0; i < cfg.worker_count; ++i)
        completion_fifo->read();
    done_event->notify(SC_ZERO_TIME);
}

