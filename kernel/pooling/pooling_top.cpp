#include "pooling_top.h"

#include <systemc>
#include <tlm>
#include <tlm_utils/peq_with_get.h>
#include <tlm_utils/simple_initiator_socket.h>

#include <algorithm>
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

    void run()
    {
        if (start_event)
            wait(*start_event);

        sc_time t_start = sc_time_stamp();
        int c_start = (tid * cfg.channels) / n_workers;
        int c_end = ((tid + 1) * cfg.channels) / n_workers;

        for (int c = c_start; c < c_end; ++c)
        {
            PendingReq load_req = issue_dma(false,
                                            std::min<uint64_t>(
                                                cfg.vec_acc_cap,
                                                static_cast<uint64_t>(cfg.spatial())) *
                                                cfg.input_elem_bytes,
                                            c,
                                            0);
            bool has_prev_compute = false;
            sc_time prev_compute_start = SC_ZERO_TIME;
            sc_time prev_compute_end = SC_ZERO_TIME;

            for (int t = 0; t < cfg.tile_count(); ++t)
            {
                const uint64_t tile_elems =
                    std::min<uint64_t>(cfg.vec_acc_cap,
                                       static_cast<uint64_t>(cfg.spatial()) -
                                           static_cast<uint64_t>(t) * cfg.vec_acc_cap);
                uint64_t rd = tile_elems * cfg.input_elem_bytes;

                issue_end(load_req);
                total_l2_rd_bytes += rd;
                if (has_prev_compute)
                {
                    dma_accel_overlap_cycles += overlap_cycles(prev_compute_start,
                                                               prev_compute_end,
                                                               load_req.submit_time,
                                                               load_req.complete_time);
                }

                auto req = issue_begin(rd, cfg.output_elem_bytes, c, t);
                ++vec_calls;
                total_rd_bytes += rd;
                total_wr_bytes += cfg.output_elem_bytes;

                PendingReq next_load;
                const bool has_next_tile = (t + 1) < cfg.tile_count();
                const bool prefetch_next = has_next_tile && cfg.l1_tile_buffers >= 2;
                if (prefetch_next)
                {
                    const uint64_t next_tile_elems =
                        std::min<uint64_t>(cfg.vec_acc_cap,
                                           static_cast<uint64_t>(cfg.spatial()) -
                                               static_cast<uint64_t>(t + 1) * cfg.vec_acc_cap);
                    next_load = issue_dma(false,
                                          next_tile_elems * cfg.input_elem_bytes,
                                          c,
                                          t + 1);
                }

                do_scalar(cfg.scalar_overhead);
                issue_end(req);

                prev_compute_start = req.submit_time;
                prev_compute_end = req.complete_time;
                has_prev_compute = true;

                if (has_next_tile)
                {
                    if (!prefetch_next)
                    {
                        const uint64_t next_tile_elems =
                            std::min<uint64_t>(cfg.vec_acc_cap,
                                               static_cast<uint64_t>(cfg.spatial()) -
                                                   static_cast<uint64_t>(t + 1) * cfg.vec_acc_cap);
                        next_load = issue_dma(false,
                                              next_tile_elems * cfg.input_elem_bytes,
                                              c,
                                              t + 1);
                    }
                    load_req = std::move(next_load);
                }
            }

            total_divide_cycles += cfg.divide_cycles;
            wait(cfg.divide_cycles * CYCLE);
            auto store_req = issue_dma(true, cfg.output_elem_bytes, c, -1);
            issue_end(store_req);
            total_l2_wr_bytes += cfg.output_elem_bytes;
        }

        elapsed_cycles = static_cast<uint64_t>((sc_time_stamp() - t_start) / CYCLE);
        if (completion_fifo)
            completion_fifo->write(tid);
    }
};

PoolTop::PoolTop(sc_module_name name,
                 const PoolRuntimeConfig &cfg_,
                 sc_event *start_event,
                 sc_event *done_event_)
    : sc_module(name),
      cfg(cfg_),
      mat_acc("mat_acc", cfg.acc_queue_depth),
      vec_acc("vec_acc",
              static_cast<size_t>(cfg.vec_acc_instances),
              cfg.acc_queue_depth),
      noc("noc"),
      memory("memory",
             cfg.l1_base_lat,
             cfg.l1_bw,
             cfg.l2_base_lat,
             cfg.l2_bw,
             static_cast<uint64_t>(cfg.vec_acc_instances),
             1),
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

    for (const auto *w : workers)
    {
        stats.max_elapsed_cycles = std::max(stats.max_elapsed_cycles, w->elapsed_cycles);
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
    stats.verification_passed =
        stats.total_vec_calls == stats.expected_vec_calls &&
        stats.total_rd_bytes == stats.expected_l1_read_bytes &&
        stats.total_wr_bytes == stats.expected_l1_write_bytes &&
        stats.l1_reqs == stats.expected_l1_reqs &&
        stats.l1_read_bytes == stats.expected_l1_read_bytes &&
        stats.l1_write_bytes == stats.expected_l1_write_bytes &&
        stats.l2_dma_reqs == stats.expected_l2_dma_reqs &&
        stats.l2_dma_read_bytes == stats.expected_l2_read_bytes &&
        stats.l2_dma_write_bytes == stats.expected_l2_write_bytes &&
        (cfg.tile_count() <= 1 || cfg.l1_tile_buffers < 2 ||
         stats.dma_accel_overlap_cycles > 0);
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

    report::print_section_title(os, "Hardware Configuration");
    report::print_fields(os, {
        {"Workers [count]", report::fmt_int(cfg.worker_count)},
        {"Matrix Accelerators [count]", report::na()},
        {"Vector Accelerators [count]", report::fmt_int(cfg.vec_acc_instances)},
        {"Matrix Accelerator Capacity", report::na()},
        {"Vector Accelerator Capacity [elements/request]", report::fmt_u64(cfg.vec_acc_cap)},
        {"Accelerator Queue Depth [requests]", report::fmt_u64(cfg.acc_queue_depth)},
        {"L1 Bandwidth [bytes/cycle]", report::fmt_u64(cfg.l1_bw)},
        {"L1 Base Latency [cycles]", report::fmt_u64(cfg.l1_base_lat)},
        {"L1 Tile Buffers [tiles]", report::fmt_int(cfg.l1_tile_buffers)},
        {"L2 DMA Bandwidth [bytes/cycle]", report::fmt_u64(cfg.l2_bw)},
        {"L2 DMA Base Latency [cycles]", report::fmt_u64(cfg.l2_base_lat)},
    });

    report::print_section_title(os, "Worker Summary");
    report::print_table(os, report::make_worker_summary_table(worker_info));

    report::print_section_title(os, "Accelerator Summary");
    report::print_table(os, report::make_accelerator_summary_table({
        {
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
        },
        {
            "Vector Accelerator",
            "pool-level",
            report::fmt_int(cfg.vec_acc_instances),
            report::fmt_u64(stats.vec_acc_reqs),
            report::fmt_u64(stats.vec_acc_queue_wait_cycles),
            report::fmt_u64(stats.vec_acc_busy_cycles),
            report::fmt_u64(stats.vec_acc_occupied_cycles),
            report::fmt_percent(stats.vec_util),
            report::fmt_percent(stats.vec_occupancy),
            report::na(),
            report::na(),
        },
        {
            "L1 Memory",
            "accelerator-side",
            report::fmt_int(cfg.vec_acc_instances),
            report::fmt_u64(stats.l1_reqs),
            report::fmt_u64(stats.l1_queue_wait_cycles),
            report::fmt_u64(stats.l1_busy_cycles),
            report::na(),
            report::na(),
            report::na(),
            report::fmt_u64(stats.l1_read_bytes),
            report::fmt_u64(stats.l1_write_bytes),
        },
        {
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
        },
    }));

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

#ifndef KERNEL_STANDALONE_MAIN
#define KERNEL_STANDALONE_MAIN 1
#endif

#if KERNEL_STANDALONE_MAIN
int sc_main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    PoolRuntimeConfig cfg = PoolRuntimeConfig::defaults();
    PoolTop top("pool_top", cfg);
    sc_start();

    top.print_report(std::cout);

    const PoolSimulationStats stats = top.collect_stats();
    return stats.verification_passed ? 0 : 2;
}
#endif
