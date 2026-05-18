#include "pooling_top.h"

#include <systemc>
#include <tlm>
#include <tlm_utils/peq_with_get.h>
#include <tlm_utils/simple_initiator_socket.h>

#include <algorithm>
#include <deque>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "extensions.h"
#include "report_formatter.h"

using namespace sc_core;
using namespace tlm;

struct PoolExt : tlm_extension<PoolExt>
{
    int channel_id = -1;
    int tile_idx = 0;

    tlm_extension_base *clone() const override
    {
        return new PoolExt(*this);
    }

    void copy_from(const tlm_extension_base &other) override
    {
        *this = static_cast<const PoolExt &>(other);
    }
};

struct PoolWorker : sc_module
{
    tlm_utils::simple_initiator_socket<PoolWorker> init;
    tlm_utils::peq_with_get<tlm_generic_payload> peq;

    int tid;
    int n_workers;
    const PoolRuntimeConfig &cfg;
    sc_event *start_event = nullptr;
    sc_fifo<int> *completion_fifo = nullptr;

    uint64_t vec_calls = 0;
    uint64_t total_scalar_cycles = 0;
    uint64_t total_divide_cycles = 0;
    uint64_t total_wait_cycles = 0;
    uint64_t total_stall_cycles = 0;
    uint64_t total_mem_cycles = 0;
    uint64_t total_rd_bytes = 0;
    uint64_t total_wr_bytes = 0;
    uint64_t total_l2_rd_bytes = 0;
    uint64_t total_l2_wr_bytes = 0;
    uint64_t dma_accel_overlap_cycles = 0;
    uint64_t elapsed_cycles = 0;

    struct DoneEntry
    {
        sc_event *ev = nullptr;
        sc_event *admit_ev = nullptr;
        bool fired = false;
        sc_time fired_time = SC_ZERO_TIME;
    };

    std::unordered_map<tlm_generic_payload *, DoneEntry *> done_map;

    struct PendingReq
    {
        tlm_generic_payload *gp = nullptr;
        ReqExt *req_ext = nullptr;
        TxnExt *tx_ext = nullptr;
        PoolExt *pool_ext = nullptr;
        MemoryAccessExt *mem_ext = nullptr;
        DoneEntry *done_entry = nullptr;
        uint64_t stall_cycles = 0;
        bool sync_done = false;
        bool direct_mem = false;
        sc_time submit_time = SC_ZERO_TIME;
        sc_time complete_time = SC_ZERO_TIME;
    };

    SC_HAS_PROCESS(PoolWorker);

    PoolWorker(sc_module_name name,
               int tid_,
               int n_workers_,
               const PoolRuntimeConfig &cfg_,
               sc_event *start_event_,
               sc_fifo<int> *completion_fifo_)
        : sc_module(name),
          init("init"),
          peq("peq"),
          tid(tid_),
          n_workers(n_workers_),
          cfg(cfg_),
          start_event(start_event_),
          completion_fifo(completion_fifo_)
    {
        init.register_nb_transport_bw(this, &PoolWorker::nb_transport_bw);
        SC_THREAD(peq_thread);
        SC_THREAD(run);
    }

    tlm_sync_enum nb_transport_bw(tlm_generic_payload &gp,
                                  tlm_phase &phase,
                                  sc_time &delay)
    {
        if (phase == BEGIN_RESP)
        {
            peq.notify(gp, delay);
            return TLM_ACCEPTED;
        }
        if (phase == END_REQ)
        {
            auto it = done_map.find(&gp);
            if (it != done_map.end() && it->second && it->second->admit_ev)
                it->second->admit_ev->notify(SC_ZERO_TIME);
            return TLM_ACCEPTED;
        }
        return TLM_ACCEPTED;
    }

    void peq_thread()
    {
        while (true)
        {
            wait(peq.get_event());
            while (auto *gp = peq.get_next_transaction())
            {
                auto it = done_map.find(gp);
                if (it != done_map.end() && it->second)
                {
                    it->second->fired = true;
                    it->second->fired_time = sc_time_stamp();
                    it->second->ev->notify(SC_ZERO_TIME);
                }
            }
        }
    }

    void do_scalar(uint64_t cyc)
    {
        total_scalar_cycles += cyc;
        wait(cyc * CYCLE);
    }

    PendingReq issue_begin(uint64_t rd, uint64_t wr, int channel_id, int tile_idx)
    {
        PendingReq p;

        auto *gp = new tlm_generic_payload();
        gp->set_command(TLM_IGNORE_COMMAND);
        gp->set_address(Interconnect::ADDR_VEC);
        gp->set_data_ptr(nullptr);
        gp->set_data_length(0);
        gp->set_streaming_width(0);
        gp->set_response_status(TLM_INCOMPLETE_RESPONSE);

        auto *req = new ReqExt(tid, cfg.vec_acc_cycle, rd, wr);
        auto *tx = new TxnExt();
        tx->src_worker = tid;
        auto *pool = new PoolExt();
        pool->channel_id = channel_id;
        pool->tile_idx = tile_idx;

        gp->set_extension(req);
        gp->set_extension(tx);
        gp->set_extension(pool);

        p.gp = gp;
        p.req_ext = req;
        p.tx_ext = tx;
        p.pool_ext = pool;
        p.done_entry = new DoneEntry();
        p.done_entry->ev = new sc_event();
        p.done_entry->admit_ev = new sc_event();
        done_map[gp] = p.done_entry;

        tlm_phase phase = BEGIN_REQ;
        sc_time delay = SC_ZERO_TIME;
        p.submit_time = sc_time_stamp();
        auto status = init->nb_transport_fw(*gp, phase, delay);

        if (status == TLM_ACCEPTED)
        {
            sc_time t_stall_start = sc_time_stamp();
            wait(*p.done_entry->admit_ev);
            p.stall_cycles = static_cast<uint64_t>((sc_time_stamp() - t_stall_start) / CYCLE);
        }
        else if (status == TLM_COMPLETED)
        {
            done_map.erase(gp);
            p.sync_done = true;
        }

        return p;
    }

    PendingReq issue_dma(bool is_write, uint64_t bytes, int channel_id, int tile_idx)
    {
        PendingReq p;
        p.direct_mem = true;

        auto *gp = new tlm_generic_payload();
        gp->set_command(is_write ? TLM_WRITE_COMMAND : TLM_READ_COMMAND);
        gp->set_address(Interconnect::ADDR_MEM);
        gp->set_data_ptr(nullptr);
        gp->set_data_length(static_cast<unsigned>(bytes));
        gp->set_streaming_width(static_cast<unsigned>(bytes));
        gp->set_response_status(TLM_INCOMPLETE_RESPONSE);

        auto *tx = new TxnExt();
        tx->src_worker = tid;
        auto *pool = new PoolExt();
        pool->channel_id = channel_id;
        pool->tile_idx = tile_idx;
        auto *mem = new MemoryAccessExt(MemoryAccessKind::Dma);

        gp->set_extension(tx);
        gp->set_extension(pool);
        gp->set_extension(mem);

        p.gp = gp;
        p.tx_ext = tx;
        p.pool_ext = pool;
        p.mem_ext = mem;
        p.done_entry = new DoneEntry();
        p.done_entry->ev = new sc_event();
        p.done_entry->admit_ev = new sc_event();
        done_map[gp] = p.done_entry;

        tlm_phase phase = BEGIN_REQ;
        sc_time delay = SC_ZERO_TIME;
        p.submit_time = sc_time_stamp();
        auto status = init->nb_transport_fw(*gp, phase, delay);
        if (status == TLM_COMPLETED)
        {
            done_map.erase(gp);
            p.sync_done = true;
        }

        // Per-DMA scalar setup cost (matches matmul's DmaScalarMode::VecPerCall).
        const uint64_t scalar_cost =
            is_write ? cfg.dma_vec_wr_scalar : cfg.dma_vec_rd_scalar;
        if (scalar_cost > 0)
            do_scalar(scalar_cost);

        return p;
    }

    void issue_end(PendingReq &p)
    {
        if (!p.sync_done && !p.done_entry->fired)
            wait(*p.done_entry->ev);

        p.complete_time = p.done_entry->fired ? p.done_entry->fired_time : sc_time_stamp();

        done_map.erase(p.gp);
        delete p.done_entry->ev;
        delete p.done_entry->admit_ev;
        delete p.done_entry;
        p.done_entry = nullptr;

        ReqExt *ext = nullptr;
        p.gp->get_extension(ext);
        if (p.direct_mem)
        {
            total_mem_cycles += static_cast<uint64_t>((p.complete_time - p.submit_time) / CYCLE);
        }
        else
        {
            total_wait_cycles += (ext ? ext->accel_qwait_cycles : 0);
            total_stall_cycles += p.stall_cycles;
            total_mem_cycles += ext ? ext->mem_cycles : 0;
        }

        tlm_phase end_phase = END_RESP;
        sc_time end_delay = SC_ZERO_TIME;
        init->nb_transport_fw(*p.gp, end_phase, end_delay);

        p.gp->clear_extension(p.req_ext);
        p.gp->clear_extension(p.tx_ext);
        p.gp->clear_extension(p.pool_ext);
        if (p.mem_ext)
            p.gp->clear_extension(p.mem_ext);
        delete p.req_ext;
        delete p.tx_ext;
        delete p.pool_ext;
        delete p.mem_ext;
        delete p.gp;
        p.gp = nullptr;
    }

    static uint64_t overlap_cycles(sc_time a_start,
                                   sc_time a_end,
                                   sc_time b_start,
                                   sc_time b_end)
    {
        const sc_time start = std::max(a_start, b_start);
        const sc_time end = std::min(a_end, b_end);
        return (end > start) ? static_cast<uint64_t>((end - start) / CYCLE) : 0;
    }

    static uint64_t tile_input_bytes(const PoolRuntimeConfig &cfg, int t)
    {
        const uint64_t tile_elems =
            std::min<uint64_t>(cfg.vec_acc_cap,
                               static_cast<uint64_t>(cfg.spatial()) -
                                   static_cast<uint64_t>(t) * cfg.vec_acc_cap);
        return tile_elems * cfg.input_elem_bytes;
    }

    struct InflightLoad
    {
        PendingReq dma;
        int tile_idx = 0;
        uint64_t rd_bytes = 0;
    };

    void run()
    {
        if (start_event)
            wait(*start_event);

        sc_time t_start = sc_time_stamp();
        int c_start = (tid * cfg.channels) / n_workers;
        int c_end = ((tid + 1) * cfg.channels) / n_workers;

        const size_t max_inflight =
            static_cast<size_t>(std::max<uint64_t>(cfg.max_inflight_vec_reqs, 1));
        const size_t l1_buffers =
            static_cast<size_t>(std::max(cfg.l1_tile_buffers, 1));
        const size_t max_dma_writes =
            static_cast<size_t>(std::max<uint64_t>(cfg.max_inflight_dma_writes, 1));

        std::deque<PendingReq> dma_write_inflight;

        for (int c = c_start; c < c_end; ++c)
        {
            // Two-queue polling loop modeled on Worker::issue_stream
            // (src/worker.cpp:534-554). Pipeline depth is gated by
            // min(max_inflight_vec_reqs, l1_tile_buffers).
            std::deque<InflightLoad> read_inflight;
            std::deque<PendingReq>   accel_inflight;
            int next_load_tile = 0;

            bool has_prev_compute = false;
            sc_time prev_compute_start = SC_ZERO_TIME;
            sc_time prev_compute_end = SC_ZERO_TIME;

            auto issue_read = [&]() -> bool {
                if (next_load_tile >= cfg.tile_count()) return false;
                if (read_inflight.size() >= max_inflight) return false;
                if (read_inflight.size() + accel_inflight.size() >= l1_buffers)
                    return false;
                InflightLoad in;
                in.tile_idx = next_load_tile;
                in.rd_bytes = tile_input_bytes(cfg, next_load_tile);
                in.dma = issue_dma(false, in.rd_bytes, c, next_load_tile);
                read_inflight.push_back(std::move(in));
                ++next_load_tile;
                return true;
            };

            auto promote_read = [&]() -> bool {
                if (read_inflight.empty()) return false;
                if (accel_inflight.size() >= max_inflight) return false;

                InflightLoad head = std::move(read_inflight.front());
                read_inflight.pop_front();
                issue_end(head.dma);
                total_l2_rd_bytes += head.rd_bytes;
                if (has_prev_compute)
                {
                    dma_accel_overlap_cycles += overlap_cycles(
                        prev_compute_start, prev_compute_end,
                        head.dma.submit_time, head.dma.complete_time);
                }

                // Per-request L1 writeback checkpoint; pinned vec unit
                // keeps the running sum in its register across tiles.
                const uint64_t wr = cfg.output_elem_bytes;
                auto req = issue_begin(head.rd_bytes, wr, c, head.tile_idx);
                ++vec_calls;
                total_rd_bytes += head.rd_bytes;
                total_wr_bytes += wr;
                accel_inflight.push_back(std::move(req));

                do_scalar(cfg.scalar_overhead);
                return true;
            };

            auto retire_accel = [&]() {
                PendingReq oldest = std::move(accel_inflight.front());
                accel_inflight.pop_front();
                issue_end(oldest);
                prev_compute_start = oldest.submit_time;
                prev_compute_end = oldest.complete_time;
                has_prev_compute = true;
            };

            while (next_load_tile < cfg.tile_count() ||
                   !read_inflight.empty() ||
                   !accel_inflight.empty())
            {
                bool progress = false;

                // Try to keep the DMA pipeline full.
                while (issue_read()) progress = true;

                // Promote any completed DMA to a compute submission.
                if (promote_read()) progress = true;

                // Retire the oldest compute when its slot is needed or
                // when no more reads can be issued/promoted right now.
                const bool window_full =
                    accel_inflight.size() >= max_inflight;
                const bool buffers_full =
                    read_inflight.size() + accel_inflight.size() >= l1_buffers;
                const bool no_more_reads =
                    next_load_tile >= cfg.tile_count() && read_inflight.empty();

                if (!accel_inflight.empty() &&
                    (window_full || buffers_full || no_more_reads))
                {
                    retire_accel();
                    progress = true;
                }

                if (!progress)
                {
                    // Should not happen: if any queue is non-empty there
                    // is always a wait we can do. Break to avoid an
                    // infinite loop in unforeseen states.
                    break;
                }
            }

            total_divide_cycles += cfg.divide_cycles;
            wait(cfg.divide_cycles * CYCLE);
            // Fire-and-forget L2 channel writeback bounded by a
            // per-worker inflight window; reap oldest when full.
            auto store_req = issue_dma(true, cfg.output_elem_bytes, c, -1);
            dma_write_inflight.push_back(std::move(store_req));
            total_l2_wr_bytes += cfg.output_elem_bytes;
            if (dma_write_inflight.size() > max_dma_writes)
            {
                issue_end(dma_write_inflight.front());
                dma_write_inflight.pop_front();
            }
        }

        while (!dma_write_inflight.empty())
        {
            issue_end(dma_write_inflight.front());
            dma_write_inflight.pop_front();
        }

        elapsed_cycles = static_cast<uint64_t>((sc_time_stamp() - t_start) / CYCLE);
        if (completion_fifo)
            completion_fifo->write(tid);
    }
};

static std::vector<int> build_identity_map(int n)
{
    std::vector<int> m;
    m.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) m.push_back(i);
    return m;
}

PoolTop::PoolTop(sc_module_name name,
                 const PoolRuntimeConfig &cfg_,
                 sc_event *start_event,
                 sc_event *done_event_)
    : sc_module(name),
      cfg(cfg_),
      mat_acc("mat_acc", cfg.acc_queue_depth),
      vec_acc("vec_acc",
              static_cast<size_t>(std::max(cfg.worker_count, 1)),
              cfg.acc_queue_depth,
              build_identity_map(std::max(cfg.worker_count, 1)),
              std::vector<uint64_t>(
                  static_cast<size_t>(std::max(cfg.worker_count, 1)), 0)),
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
            std::make_unique<sc_fifo<int>>(sc_gen_unique_name("pool_done_fifo"),
                                           cfg.worker_count + 1);
        SC_THREAD(done_monitor);
    }

    for (int i = 0; i < cfg.worker_count; ++i)
    {
        auto *w = new PoolWorker(sc_gen_unique_name("pool_worker"),
                                 i,
                                 cfg.worker_count,
                                 cfg,
                                 start_event,
                                 completion_fifo.get());
        workers.push_back(w);
        w->init.bind(noc.tgt);
    }
}

PoolTop::~PoolTop()
{
    for (auto *w : workers)
        delete w;
}

PoolSimulationStats PoolTop::collect_stats() const
{
    PoolSimulationStats stats;

    const PoolWorker *slowest = nullptr;
    for (const auto *w : workers)
    {
        if (w->elapsed_cycles > stats.max_elapsed_cycles)
        {
            stats.max_elapsed_cycles = w->elapsed_cycles;
            slowest = w;
        }
        stats.total_vec_calls += w->vec_calls;
        stats.total_rd_bytes += w->total_rd_bytes;
        stats.total_wr_bytes += w->total_wr_bytes;
        stats.total_wait_cycles += w->total_wait_cycles;
        stats.total_mem_cycles += w->total_mem_cycles;
        stats.dma_accel_overlap_cycles += w->dma_accel_overlap_cycles;
    }

    stats.expected_vec_calls =
        static_cast<uint64_t>(cfg.channels) * static_cast<uint64_t>(cfg.tile_count());
    stats.expected_l1_read_bytes =
        static_cast<uint64_t>(cfg.channels) * static_cast<uint64_t>(cfg.spatial()) *
        cfg.input_elem_bytes;
    // Per-request L1 writeback checkpoint: one read + one write per vec
    // call. Carry is kept in the pinned vec unit's register; channel
    // result lands in L2 via a separate fire-and-forget DMA.
    stats.expected_l1_write_bytes =
        stats.expected_vec_calls * cfg.output_elem_bytes;
    stats.expected_l1_reqs = stats.expected_vec_calls * 2;
    stats.expected_l2_read_bytes =
        static_cast<uint64_t>(cfg.channels) * static_cast<uint64_t>(cfg.spatial()) *
        cfg.input_elem_bytes;
    stats.expected_l2_write_bytes =
        static_cast<uint64_t>(cfg.channels) * cfg.output_elem_bytes;
    stats.expected_l2_dma_reqs =
        stats.expected_vec_calls + static_cast<uint64_t>(cfg.channels);
    stats.vec_acc_reqs = vec_acc.req_count_total();
    stats.vec_acc_busy_cycles = vec_acc.busy_cycles_total();
    stats.vec_acc_occupied_cycles = vec_acc.occupied_cycles_total();
    stats.vec_acc_queue_wait_cycles = vec_acc.queue_wait_cycles_total();
    stats.l1_reqs = memory.l1_reqs;
    stats.l1_read_bytes = memory.l1_read_bytes;
    stats.l1_write_bytes = memory.l1_write_bytes;
    stats.l1_busy_cycles = memory.l1_busy_cycles;
    stats.l1_queue_wait_cycles = memory.l1_qwait_cycles;
    stats.l2_dma_reqs = memory.dma_reqs;
    stats.l2_dma_read_bytes = memory.dma_read_bytes;
    stats.l2_dma_write_bytes = memory.dma_write_bytes;
    stats.l2_dma_busy_cycles = memory.dma_busy_cycles;
    stats.l2_dma_queue_wait_cycles = memory.dma_qwait_cycles;

    const double sim_cycles = static_cast<double>(sc_time_stamp() / CYCLE);
    const double vec_capacity =
        sim_cycles * static_cast<double>(vec_acc.instance_count());
    stats.vec_util = (vec_capacity > 0.0)
        ? static_cast<double>(vec_acc.busy_cycles_total()) / vec_capacity * 100.0
        : 0.0;
    stats.vec_occupancy = (vec_capacity > 0.0)
        ? static_cast<double>(vec_acc.occupied_cycles_total()) / vec_capacity * 100.0
        : 0.0;
    stats.l1_bw_observed = (sim_cycles > 0.0)
        ? static_cast<double>(stats.l1_read_bytes + stats.l1_write_bytes) / sim_cycles
        : 0.0;
    stats.l2_bw_observed = (sim_cycles > 0.0)
        ? static_cast<double>(stats.l2_dma_read_bytes + stats.l2_dma_write_bytes) / sim_cycles
        : 0.0;
    if (slowest != nullptr && stats.max_elapsed_cycles > 0)
    {
        // Per-category cycle counts attributed to the critical-path worker.
        // Categories can overlap on the wall-clock timeline (e.g. scalar
        // runs while a vec request is in service), so we normalize the
        // fractions by the sum of categorized cycles, not by elapsed
        // — same convention as matmul (see matmul_top.cpp).
        const uint64_t vec_service =
            slowest->vec_calls * cfg.vec_acc_cycle;
        const uint64_t dma_cycles = slowest->total_mem_cycles;
        const uint64_t scalar_cycles =
            slowest->total_scalar_cycles + slowest->total_divide_cycles;
        const uint64_t stall_cycles = slowest->total_stall_cycles;

        stats.slowest_worker_tid = slowest->tid;
        stats.slowest_vec_cycles = vec_service;
        stats.slowest_dma_cycles = dma_cycles;
        stats.slowest_scalar_cycles = scalar_cycles;
        stats.slowest_stall_cycles = stall_cycles;

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
        stats.total_vec_calls == stats.expected_vec_calls &&
        stats.total_rd_bytes == stats.expected_l1_read_bytes &&
        stats.total_wr_bytes == stats.expected_l1_write_bytes &&
        stats.l1_reqs == stats.expected_l1_reqs &&
        stats.l1_read_bytes == stats.expected_l1_read_bytes &&
        stats.l1_write_bytes == stats.expected_l1_write_bytes &&
        stats.l2_dma_reqs == stats.expected_l2_dma_reqs &&
        stats.l2_dma_read_bytes == stats.expected_l2_read_bytes &&
        stats.l2_dma_write_bytes == stats.expected_l2_write_bytes;
    return stats;
}

std::vector<KernelWorkerInfo> PoolTop::collect_worker_info() const
{
    std::vector<KernelWorkerInfo> info;
    info.reserve(workers.size());
    for (const auto *w : workers)
    {
        KernelWorkerInfo wi;
        wi.tid = w->tid;
        wi.vec_reqs = w->vec_calls;
        wi.scalar_cycles = w->total_scalar_cycles + w->total_divide_cycles;
        wi.stall_cycles = w->total_stall_cycles;
        wi.elapsed_cycles = w->elapsed_cycles;
        wi.mem_cycles = w->total_mem_cycles;
        wi.rd_bytes = w->total_rd_bytes;
        wi.wr_bytes = w->total_wr_bytes;
        info.push_back(wi);
    }
    return info;
}

void PoolTop::print_report(std::ostream &os) const
{
    const PoolSimulationStats stats = collect_stats();
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
        {"Operation Type", "Global Average Pooling"},
        {"Input Tensor Shape", "[C=" + report::fmt_int(cfg.channels) +
                               ", H=" + report::fmt_int(cfg.height) +
                               ", W=" + report::fmt_int(cfg.width) + "]"},
        {"Output Tensor Shape", "[C=" + report::fmt_int(cfg.channels) +
                                ", H=1, W=1]"},
    });

    const int pinned_vec_instances = std::max(cfg.worker_count, 1);
    report::print_section_title(os, "Hardware Configuration");
    report::print_fields(os, {
        {"Workers [count]", report::fmt_int(cfg.worker_count)},
        {"Matrix Accelerators [count]", report::na()},
        {"Vector Accelerators [count]", report::fmt_int(pinned_vec_instances)},
        {"Worker->Vec Binding", "pinned 1:1"},
        {"Matrix Accelerator Capacity", report::na()},
        {"Vector Accelerator Capacity [elements/request]", report::fmt_u64(cfg.vec_acc_cap)},
        {"Accelerator Queue Depth [requests]", report::fmt_u64(cfg.acc_queue_depth)},
        {"Max Inflight L2 DMA Writes / worker", report::fmt_u64(cfg.max_inflight_dma_writes)},
        {"L1 Bandwidth [bytes/cycle]", report::fmt_u64(cfg.l1_bw)},
        {"L1 Base Latency [cycles]", report::fmt_u64(cfg.l1_base_lat)},
        {"L1 Parallel Slots", report::fmt_u64(cfg.l1_slots)},
        {"L1 Tile Buffers [tiles]", report::fmt_int(cfg.l1_tile_buffers)},
        {"L2 DMA Bandwidth [bytes/cycle]", report::fmt_u64(cfg.l2_bw)},
        {"L2 DMA Base Latency [cycles]", report::fmt_u64(cfg.l2_base_lat)},
        {"L2 DMA Parallel Slots", report::fmt_u64(cfg.l2_slots)},
    });

    report::print_section_title(os, "Worker Summary");
    report::print_table(os, report::make_worker_summary_table(worker_info));

    report::print_section_title(os, "Accelerator Summary");
    std::vector<report::AcceleratorSummaryRow> accel_rows;
    accel_rows.push_back({
        "Matrix Accelerator",
        "pool-level",
        report::na(),
        report::na(),
        report::na(),
        report::na(),
        report::na(),
        report::na(),
        report::na(),
        report::na(),
        report::na(),
        report::na(),
    });
    accel_rows.push_back({
        "Vector Accelerator",
        "pool-level",
        report::fmt_int(pinned_vec_instances),
        report::fmt_u64(stats.vec_acc_reqs),
        report::fmt_u64(stats.vec_acc_queue_wait_cycles),
        report::fmt_u64(stats.vec_acc_busy_cycles),
        report::fmt_u64(stats.vec_acc_occupied_cycles),
        report::fmt_percent(stats.vec_util),
        report::fmt_percent(stats.vec_occupancy),
        report::na(),
        report::na(),
        report::na(),
    });
    for (auto &r : report::make_per_instance_accel_rows(
             "Vector Accelerator", vec_acc.per_instance_stats(),
             stats.max_elapsed_cycles))
        accel_rows.push_back(std::move(r));
    accel_rows.push_back({
        "L1 Memory",
        "accelerator-side",
        report::fmt_int(pinned_vec_instances),
        report::fmt_u64(stats.l1_reqs),
        report::fmt_u64(stats.l1_queue_wait_cycles),
        report::fmt_u64(stats.l1_busy_cycles),
        report::na(),
        report::na(),
        report::na(),
        report::fmt_u64(stats.l1_read_bytes),
        report::fmt_u64(stats.l1_write_bytes),
        report::na(),
    });
    accel_rows.push_back({
        "L2 DMA",
        "prefetch/writeback",
        "1",
        report::fmt_u64(stats.l2_dma_reqs),
        report::fmt_u64(stats.l2_dma_queue_wait_cycles),
        report::fmt_u64(stats.l2_dma_busy_cycles),
        report::na(),
        report::na(),
        report::na(),
        report::fmt_u64(stats.l2_dma_read_bytes),
        report::fmt_u64(stats.l2_dma_write_bytes),
        report::na(),
    });
    report::print_table(os, report::make_accelerator_summary_table(accel_rows));

    report::print_section_title(os, "Overall Summary");
    report::print_fields(os, {
        {"Total Elapsed Cycles [cycles]", report::fmt_u64(stats.max_elapsed_cycles)},
        {"Total Matrix Accelerator Requests [requests]", report::na()},
        {"Total Vector Accelerator Requests [requests]", report::fmt_u64(stats.total_vec_calls)},
        {"Total L1 Requests [requests]", report::fmt_u64(stats.l1_reqs)},
        {"Total L2 DMA Requests [requests]", report::fmt_u64(stats.l2_dma_reqs)},
        {"Total L1 Read Bytes [bytes]", report::fmt_u64(stats.l1_read_bytes)},
        {"Total L1 Write Bytes [bytes]", report::fmt_u64(stats.l1_write_bytes)},
        {"Total L2 DMA Read Bytes [bytes]", report::fmt_u64(stats.l2_dma_read_bytes)},
        {"Total L2 DMA Write Bytes [bytes]", report::fmt_u64(stats.l2_dma_write_bytes)},
        {"Total Stall Cycles [cycles]", report::fmt_u64(total_stall_cycles)},
        {"Total Memory Cycles [cycles]", report::fmt_u64(total_mem_cycles)},
        {"Total Scalar Cycles [cycles]", report::fmt_u64(total_scalar_cycles)},
        {"DMA/Accelerator Overlap [cycles]", report::fmt_u64(stats.dma_accel_overlap_cycles)},
        {"Average L1 Bandwidth [bytes/cycle]", report::fmt_rate(stats.l1_bw_observed, "bytes/cycle")},
        {"Average L2 DMA Bandwidth [bytes/cycle]", report::fmt_rate(stats.l2_bw_observed, "bytes/cycle")},
        {"Critical-Path Worker [tid]", report::fmt_int(stats.slowest_worker_tid)},
        {"Vec Cycle Fraction [%]",    report::fmt_percent(stats.vec_cycle_fraction)},
        {"DMA Cycle Fraction [%]",    report::fmt_percent(stats.dma_cycle_fraction)},
        {"Scalar Cycle Fraction [%]", report::fmt_percent(stats.scalar_cycle_fraction)},
        {"Stall Cycle Fraction [%]",  report::fmt_percent(stats.stall_cycle_fraction)},
    });

    report::print_section_title(os, "Verification");
    report::print_fields(os, {
        {"Expected Vector Accelerator Requests [requests]", report::fmt_u64(stats.expected_vec_calls)},
        {"Actual Vector Accelerator Requests [requests]", report::fmt_u64(stats.total_vec_calls)},
        {"Expected L1 Requests [requests]", report::fmt_u64(stats.expected_l1_reqs)},
        {"Actual L1 Requests [requests]", report::fmt_u64(stats.l1_reqs)},
        {"Expected L1 Read Bytes [bytes]", report::fmt_u64(stats.expected_l1_read_bytes)},
        {"Actual L1 Read Bytes [bytes]", report::fmt_u64(stats.l1_read_bytes)},
        {"Expected L1 Write Bytes [bytes]", report::fmt_u64(stats.expected_l1_write_bytes)},
        {"Actual L1 Write Bytes [bytes]", report::fmt_u64(stats.l1_write_bytes)},
        {"Expected L2 DMA Requests [requests]", report::fmt_u64(stats.expected_l2_dma_reqs)},
        {"Actual L2 DMA Requests [requests]", report::fmt_u64(stats.l2_dma_reqs)},
        {"Expected L2 DMA Read Bytes [bytes]", report::fmt_u64(stats.expected_l2_read_bytes)},
        {"Actual L2 DMA Read Bytes [bytes]", report::fmt_u64(stats.l2_dma_read_bytes)},
        {"Expected L2 DMA Write Bytes [bytes]", report::fmt_u64(stats.expected_l2_write_bytes)},
        {"Actual L2 DMA Write Bytes [bytes]", report::fmt_u64(stats.l2_dma_write_bytes)},
        {"DMA/Accelerator Overlap [cycles]", report::fmt_u64(stats.dma_accel_overlap_cycles)},
        {"Verification Status", stats.verification_passed ? "PASS" : "FAIL"},
    });
}

void PoolTop::done_monitor()
{
    for (int i = 0; i < cfg.worker_count; ++i)
        completion_fifo->read();
    done_event->notify(SC_ZERO_TIME);
}
