#include "layer_norm_top.h"

#include <systemc>
#include <tlm>
#include <tlm_utils/peq_with_get.h>
#include <tlm_utils/simple_initiator_socket.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "extensions.h"
#include "report_formatter.h"

using namespace sc_core;
using namespace tlm;

struct LayerNormExt : tlm_extension<LayerNormExt>
{
    int channel_id = -1;
    int step = 0;

    tlm_extension_base *clone() const override
    {
        return new LayerNormExt(*this);
    }

    void copy_from(const tlm_extension_base &other) override
    {
        *this = static_cast<const LayerNormExt &>(other);
    }
};

struct LayerNormWorker : sc_module
{
    tlm_utils::simple_initiator_socket<LayerNormWorker> init;
    tlm_utils::peq_with_get<tlm_generic_payload> peq;

    int tid;
    int n_workers;
    const LayerNormRuntimeConfig &cfg;
    sc_event *start_event = nullptr;
    sc_fifo<int> *completion_fifo = nullptr;

    LnStepStats step_stats[4];
    uint64_t total_dispatch_cycles = 0;
    uint64_t total_scalar_cycles = 0;
    uint64_t total_wait_cycles = 0;
    uint64_t total_stall_cycles = 0;
    uint64_t total_mem_cycles = 0;
    uint64_t vec_calls = 0;
    uint64_t elapsed_cycles = 0;

    struct DoneEntry
    {
        sc_event *ev = nullptr;
        sc_event *admit_ev = nullptr;
        bool fired = false;
    };

    std::unordered_map<tlm_generic_payload *, DoneEntry *> done_map;

    struct PendingReq
    {
        tlm_generic_payload *gp = nullptr;
        ReqExt *req_ext = nullptr;
        TxnExt *tx_ext = nullptr;
        LayerNormExt *ln_ext = nullptr;
        DoneEntry *done_entry = nullptr;
        uint64_t svc_cyc = 0;
        uint64_t stall_cycles = 0;
        bool sync_done = false;
    };

    SC_HAS_PROCESS(LayerNormWorker);

    LayerNormWorker(sc_module_name name,
                    int tid_,
                    int n_workers_,
                    const LayerNormRuntimeConfig &cfg_,
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
        init.register_nb_transport_bw(this, &LayerNormWorker::nb_transport_bw);
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
                    it->second->ev->notify(SC_ZERO_TIME);
                }
            }
        }
    }

    void do_scalar(uint64_t cyc)
    {
        total_dispatch_cycles += cyc;
        wait(cyc * CYCLE);
    }

    PendingReq issue_begin(uint64_t addr,
                           uint64_t svc_cyc,
                           uint64_t rd,
                           uint64_t wr,
                           int channel_id,
                           int step)
    {
        PendingReq p;
        p.svc_cyc = svc_cyc;

        auto *gp = new tlm_generic_payload();
        gp->set_command(TLM_IGNORE_COMMAND);
        gp->set_address(addr);
        gp->set_data_ptr(nullptr);
        gp->set_data_length(0);
        gp->set_streaming_width(0);
        gp->set_response_status(TLM_INCOMPLETE_RESPONSE);

        auto *req = new ReqExt(tid, svc_cyc, rd, wr);
        auto *tx = new TxnExt();
        tx->src_worker = tid;
        auto *ln = new LayerNormExt();
        ln->channel_id = channel_id;
        ln->step = step;

        gp->set_extension(req);
        gp->set_extension(tx);
        gp->set_extension(ln);

        p.gp = gp;
        p.req_ext = req;
        p.tx_ext = tx;
        p.ln_ext = ln;
        p.done_entry = new DoneEntry();
        p.done_entry->ev = new sc_event();
        p.done_entry->admit_ev = new sc_event();
        done_map[gp] = p.done_entry;

        tlm_phase phase = BEGIN_REQ;
        sc_time delay = SC_ZERO_TIME;
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

    void issue_end(PendingReq &p, LnStepStats &s)
    {
        if (!p.sync_done && !p.done_entry->fired)
            wait(*p.done_entry->ev);

        done_map.erase(p.gp);
        delete p.done_entry->ev;
        delete p.done_entry->admit_ev;
        delete p.done_entry;
        p.done_entry = nullptr;

        ReqExt *ext = nullptr;
        p.gp->get_extension(ext);
        uint64_t qwait = ext ? ext->accel_qwait_cycles : 0;
        uint64_t mec = ext ? ext->mem_cycles : 0;

        s.wait_cycles += qwait;
        s.mem_cycles += mec;
        total_wait_cycles += qwait;
        total_stall_cycles += p.stall_cycles;
        total_mem_cycles += mec;

        tlm_phase end_phase = END_RESP;
        sc_time end_delay = SC_ZERO_TIME;
        init->nb_transport_fw(*p.gp, end_phase, end_delay);

        p.gp->clear_extension(p.req_ext);
        p.gp->clear_extension(p.tx_ext);
        p.gp->clear_extension(p.ln_ext);
        delete p.req_ext;
        delete p.tx_ext;
        delete p.ln_ext;
        delete p.gp;
        p.gp = nullptr;
    }

    void run_step(int channel_id, int step_id, LnStepStats &stats)
    {
        const uint64_t elem_bytes = 2;
        std::vector<PendingReq> pending;
        pending.reserve(static_cast<size_t>(cfg.tile_count()));

        for (int t = 0; t < cfg.tile_count(); ++t)
        {
            const uint64_t tile_elems =
                std::min<uint64_t>(cfg.vec_acc_cap,
                                   static_cast<uint64_t>(cfg.spatial()) -
                                       static_cast<uint64_t>(t) * cfg.vec_acc_cap);
            uint64_t rd = tile_elems * elem_bytes;
            uint64_t wr = 0;
            if (step_id == 4)
            {
                rd += 2 * elem_bytes;
                wr = tile_elems * elem_bytes;
            }

            auto req =
                issue_begin(Interconnect::ADDR_VEC,
                            cfg.vec_acc_cycle,
                            rd,
                            wr,
                            channel_id,
                            step_id);
            ++vec_calls;
            ++stats.vec_reqs;
            stats.accel_cycles += cfg.vec_acc_cycle;
            stats.rd_bytes += rd;
            stats.wr_bytes += wr;
            do_scalar(cfg.scalar_overhead);
            pending.push_back(std::move(req));
        }

        for (auto &req : pending)
            issue_end(req, stats);
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
            run_step(c, 1, step_stats[0]);
            run_step(c, 2, step_stats[1]);

            step_stats[2].scalar_cycles += cfg.step3_cycles;
            total_scalar_cycles += cfg.step3_cycles;
            wait(cfg.step3_cycles * CYCLE);

            run_step(c, 4, step_stats[3]);
        }

        elapsed_cycles = static_cast<uint64_t>((sc_time_stamp() - t_start) / CYCLE);
        if (completion_fifo)
            completion_fifo->write(tid);
    }
};

LayerNormTop::LayerNormTop(sc_module_name name,
                           const LayerNormRuntimeConfig &cfg_,
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
             cfg.memory_base_lat,
             cfg.memory_bw,
             static_cast<uint64_t>(cfg.vec_acc_instances)),
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

    for (int i = 0; i < cfg.worker_count; ++i)
    {
        auto *w = new LayerNormWorker(sc_gen_unique_name("ln_worker"),
                                      i,
                                      cfg.worker_count,
                                      cfg,
                                      start_event,
                                      completion_fifo.get());
        workers.push_back(w);
        w->init.bind(noc.tgt);
    }
}

LayerNormTop::~LayerNormTop()
{
    for (auto *w : workers)
        delete w;
}

LayerNormSimulationStats LayerNormTop::collect_stats() const
{
    LayerNormSimulationStats stats;

    for (const auto *w : workers)
    {
        for (int s = 0; s < 4; ++s)
        {
            stats.steps[s].vec_reqs += w->step_stats[s].vec_reqs;
            stats.steps[s].accel_cycles += w->step_stats[s].accel_cycles;
            stats.steps[s].scalar_cycles += w->step_stats[s].scalar_cycles;
            stats.steps[s].wait_cycles += w->step_stats[s].wait_cycles;
            stats.steps[s].mem_cycles += w->step_stats[s].mem_cycles;
            stats.steps[s].rd_bytes += w->step_stats[s].rd_bytes;
            stats.steps[s].wr_bytes += w->step_stats[s].wr_bytes;
        }

        stats.max_elapsed_cycles =
            std::max(stats.max_elapsed_cycles, w->elapsed_cycles);
    }

    for (int s = 0; s < 4; ++s)
    {
        stats.total_vec_reqs += stats.steps[s].vec_reqs;
        stats.total_rd_bytes += stats.steps[s].rd_bytes;
        stats.total_wr_bytes += stats.steps[s].wr_bytes;
    }

    stats.expected_vec_reqs =
        static_cast<uint64_t>(cfg.channels) * 3 * static_cast<uint64_t>(cfg.tile_count());
    stats.vec_acc_reqs = vec_acc.req_count_total();
    stats.vec_acc_busy_cycles = vec_acc.busy_cycles_total();
    stats.vec_acc_occupied_cycles = vec_acc.occupied_cycles_total();
    stats.vec_acc_queue_wait_cycles = vec_acc.queue_wait_cycles_total();
    stats.memory_reqs = memory.reqs;
    stats.memory_busy_cycles = memory.busy_cycles;
    stats.memory_queue_wait_cycles = memory.qwait_cycles;

    const double sim_cycles = static_cast<double>(sc_time_stamp() / CYCLE);
    const double vec_capacity =
        sim_cycles * static_cast<double>(vec_acc.instance_count());
    stats.vec_util = (vec_capacity > 0.0)
        ? static_cast<double>(vec_acc.busy_cycles_total()) / vec_capacity * 100.0
        : 0.0;
    stats.vec_occupancy = (vec_capacity > 0.0)
        ? static_cast<double>(vec_acc.occupied_cycles_total()) / vec_capacity * 100.0
        : 0.0;
    stats.verification_passed = (stats.total_vec_reqs == stats.expected_vec_reqs);
    return stats;
}

std::vector<KernelWorkerInfo> LayerNormTop::collect_worker_info() const
{
    std::vector<KernelWorkerInfo> info;
    info.reserve(workers.size());
    for (const auto *w : workers)
    {
        KernelWorkerInfo wi;
        wi.tid = w->tid;
        wi.vec_reqs = w->vec_calls;
        wi.scalar_cycles = w->total_scalar_cycles;
        wi.stall_cycles = w->total_stall_cycles;
        wi.elapsed_cycles = w->elapsed_cycles;
        wi.mem_cycles = w->total_mem_cycles;
        for (int s = 0; s < 4; ++s)
        {
            wi.rd_bytes += w->step_stats[s].rd_bytes;
            wi.wr_bytes += w->step_stats[s].wr_bytes;
        }
        info.push_back(wi);
    }
    return info;
}

void LayerNormTop::print_report(std::ostream &os) const
{
    const LayerNormSimulationStats stats = collect_stats();
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
        {"Operation Type", "Layer Normalization"},
        {"Input Tensor Shape", "[C=" + report::fmt_int(cfg.channels) +
                               ", H=" + report::fmt_int(cfg.height) +
                               ", W=" + report::fmt_int(cfg.width) + "]"},
        {"Input Element Type", "int16"},
    });

    report::print_section_title(os, "Hardware Configuration");
    report::print_fields(os, {
        {"Workers [count]", report::fmt_int(cfg.worker_count)},
        {"Matrix Accelerators [count]", report::na()},
        {"Vector Accelerators [count]", report::fmt_int(cfg.vec_acc_instances)},
        {"Matrix Accelerator Capacity", report::na()},
        {"Vector Accelerator Capacity [elements/request]", report::fmt_u64(cfg.vec_acc_cap)},
        {"Accelerator Queue Depth [requests]", report::fmt_u64(cfg.acc_queue_depth)},
        {"Memory Bandwidth [bytes/cycle]", report::fmt_u64(cfg.memory_bw)},
        {"Memory Base Latency [cycles]", report::fmt_u64(cfg.memory_base_lat)},
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
            "Memory",
            "shared resource",
            "1",
            report::fmt_u64(stats.memory_reqs),
            report::fmt_u64(stats.memory_queue_wait_cycles),
            report::fmt_u64(stats.memory_busy_cycles),
            report::na(),
            report::na(),
            report::na(),
            report::fmt_u64(stats.total_rd_bytes),
            report::fmt_u64(stats.total_wr_bytes),
        },
    }));

    report::print_section_title(os, "Overall Summary");
    report::print_fields(os, {
        {"Total Elapsed Cycles [cycles]", report::fmt_u64(stats.max_elapsed_cycles)},
        {"Total Matrix Accelerator Requests [requests]", report::na()},
        {"Total Vector Accelerator Requests [requests]", report::fmt_u64(stats.total_vec_reqs)},
        {"Total Memory Requests [requests]", report::fmt_u64(stats.memory_reqs)},
        {"Total Read Bytes [bytes]", report::fmt_u64(stats.total_rd_bytes)},
        {"Total Write Bytes [bytes]", report::fmt_u64(stats.total_wr_bytes)},
        {"Total Stall Cycles [cycles]", report::fmt_u64(total_stall_cycles)},
        {"Total Memory Cycles [cycles]", report::fmt_u64(total_mem_cycles)},
        {"Total Scalar Cycles [cycles]", report::fmt_u64(total_scalar_cycles)},
    });

    report::print_section_title(os, "Verification");
    report::print_fields(os, {
        {"Expected Vector Accelerator Requests [requests]", report::fmt_u64(stats.expected_vec_reqs)},
        {"Actual Vector Accelerator Requests [requests]", report::fmt_u64(stats.total_vec_reqs)},
        {"Verification Status", stats.verification_passed ? "PASS" : "FAIL"},
    });
}

void LayerNormTop::done_monitor()
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

    LayerNormRuntimeConfig cfg = LayerNormRuntimeConfig::defaults();
    LayerNormTop top("ln_top", cfg);
    sc_start();

    top.print_report(std::cout);

    const LayerNormSimulationStats stats = top.collect_stats();
    return stats.verification_passed ? 0 : 2;
}
#endif
