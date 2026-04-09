#include "dw_conv2d_top.h"

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

struct DwConvExt : tlm_extension<DwConvExt>
{
    int channel_id = -1;
    int out_row = 0;
    int strip_idx = 0;

    tlm_extension_base *clone() const override
    {
        return new DwConvExt(*this);
    }

    void copy_from(const tlm_extension_base &other) override
    {
        *this = static_cast<const DwConvExt &>(other);
    }
};

struct DwConvWorker : sc_module
{
    tlm_utils::simple_initiator_socket<DwConvWorker> init;
    tlm_utils::peq_with_get<tlm_generic_payload> peq;

    int tid;
    int n_workers;
    const DwConvRuntimeConfig &cfg;
    sc_event *start_event = nullptr;
    sc_fifo<int> *completion_fifo = nullptr;

    uint64_t vec_calls = 0;
    uint64_t total_scalar_cycles = 0;
    uint64_t total_wait_cycles = 0;
    uint64_t total_stall_cycles = 0;
    uint64_t total_mem_cycles = 0;
    uint64_t total_rd_bytes = 0;
    uint64_t total_wr_bytes = 0;
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
        DwConvExt *dw_ext = nullptr;
        DoneEntry *done_entry = nullptr;
        uint64_t stall_cycles = 0;
        bool sync_done = false;
    };

    SC_HAS_PROCESS(DwConvWorker);

    DwConvWorker(sc_module_name name,
                 int tid_,
                 int n_workers_,
                 const DwConvRuntimeConfig &cfg_,
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
        init.register_nb_transport_bw(this, &DwConvWorker::nb_transport_bw);
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
        total_scalar_cycles += cyc;
        wait(cyc * CYCLE);
    }

    uint64_t calc_strip_input_read_bytes(int oh,
                                         uint64_t strip_start,
                                         uint64_t vl) const
    {
        uint64_t rd = 0;

        for (int kh = 0; kh < cfg.kernel_h; ++kh)
        {
            int ih = oh - cfg.pad + kh;
            if (ih < 0 || ih >= cfg.height)
                continue;

            for (int kw = 0; kw < cfg.kernel_w; ++kw)
            {
                int64_t iw_base = static_cast<int64_t>(strip_start) - cfg.pad + kw;
                int64_t lo = std::max<int64_t>(0, iw_base);
                int64_t hi = std::min<int64_t>(
                    static_cast<int64_t>(cfg.width), iw_base + static_cast<int64_t>(vl));
                if (hi > lo)
                    rd += static_cast<uint64_t>(hi - lo) * cfg.input_elem_bytes;
            }
        }

        return rd;
    }

    PendingReq issue_begin(uint64_t rd,
                           uint64_t wr,
                           int channel_id,
                           int out_row,
                           int strip_idx)
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
        auto *dw = new DwConvExt();
        dw->channel_id = channel_id;
        dw->out_row = out_row;
        dw->strip_idx = strip_idx;

        gp->set_extension(req);
        gp->set_extension(tx);
        gp->set_extension(dw);

        p.gp = gp;
        p.req_ext = req;
        p.tx_ext = tx;
        p.dw_ext = dw;
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

    void issue_end(PendingReq &p)
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
        total_wait_cycles += (ext ? ext->accel_qwait_cycles : 0);
        total_stall_cycles += p.stall_cycles;
        total_mem_cycles += ext ? ext->mem_cycles : 0;

        tlm_phase end_phase = END_RESP;
        sc_time end_delay = SC_ZERO_TIME;
        init->nb_transport_fw(*p.gp, end_phase, end_delay);

        p.gp->clear_extension(p.req_ext);
        p.gp->clear_extension(p.tx_ext);
        p.gp->clear_extension(p.dw_ext);
        delete p.req_ext;
        delete p.tx_ext;
        delete p.dw_ext;
        delete p.gp;
        p.gp = nullptr;
    }

    void run()
    {
        if (start_event)
            wait(*start_event);

        sc_time t_start = sc_time_stamp();
        int c_start = (tid * cfg.channels) / n_workers;
        int c_end = ((tid + 1) * cfg.channels) / n_workers;
        const uint64_t kernel_rd_bytes =
            static_cast<uint64_t>(cfg.kernel_h * cfg.kernel_w) * cfg.input_elem_bytes;

        for (int c = c_start; c < c_end; ++c)
        {
            for (int oh = 0; oh < cfg.out_h(); ++oh)
            {
                std::vector<PendingReq> pending;
                pending.reserve(static_cast<size_t>(cfg.strips_per_row()));

                for (int s = 0; s < cfg.strips_per_row(); ++s)
                {
                    uint64_t strip_start = static_cast<uint64_t>(s) * cfg.vec_acc_cap;
                    uint64_t vl = std::min<uint64_t>(
                        cfg.vec_acc_cap,
                        static_cast<uint64_t>(cfg.out_w()) - strip_start);

                    uint64_t rd = calc_strip_input_read_bytes(oh, strip_start, vl);
                    if (oh == 0 && s == 0)
                        rd += kernel_rd_bytes;
                    uint64_t wr = vl * cfg.output_elem_bytes;

                    auto req = issue_begin(rd, wr, c, oh, s);
                    ++vec_calls;
                    total_rd_bytes += rd;
                    total_wr_bytes += wr;
                    do_scalar(cfg.scalar_overhead);
                    pending.push_back(std::move(req));
                }

                for (auto &req : pending)
                    issue_end(req);
            }
        }

        elapsed_cycles = static_cast<uint64_t>((sc_time_stamp() - t_start) / CYCLE);
        if (completion_fifo)
            completion_fifo->write(tid);
    }
};

DwConvTop::DwConvTop(sc_module_name name,
                     const DwConvRuntimeConfig &cfg_,
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
            std::make_unique<sc_fifo<int>>(sc_gen_unique_name("dw_done_fifo"),
                                           cfg.worker_count + 1);
        SC_THREAD(done_monitor);
    }

    for (int i = 0; i < cfg.worker_count; ++i)
    {
        auto *w = new DwConvWorker(sc_gen_unique_name("dw_worker"),
                                   i,
                                   cfg.worker_count,
                                   cfg,
                                   start_event,
                                   completion_fifo.get());
        workers.push_back(w);
        w->init.bind(noc.tgt);
    }
}

DwConvTop::~DwConvTop()
{
    for (auto *w : workers)
        delete w;
}

DwConvSimulationStats DwConvTop::collect_stats() const
{
    DwConvSimulationStats stats;
    const uint64_t calls_per_chan =
        static_cast<uint64_t>(cfg.out_h()) * static_cast<uint64_t>(cfg.strips_per_row());

    for (const auto *w : workers)
    {
        stats.max_elapsed_cycles = std::max(stats.max_elapsed_cycles, w->elapsed_cycles);
        stats.total_vec_calls += w->vec_calls;
        stats.total_rd_bytes += w->total_rd_bytes;
        stats.total_wr_bytes += w->total_wr_bytes;
        stats.total_wait_cycles += w->total_wait_cycles;
        stats.total_mem_cycles += w->total_mem_cycles;
    }

    stats.expected_vec_calls = static_cast<uint64_t>(cfg.channels) * calls_per_chan;
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
    stats.mem_bw = (sim_cycles > 0.0)
        ? static_cast<double>(memory.busy_cycles) * static_cast<double>(cfg.memory_bw) / sim_cycles
        : 0.0;
    stats.verification_passed = (stats.total_vec_calls == stats.expected_vec_calls);
    return stats;
}

std::vector<KernelWorkerInfo> DwConvTop::collect_worker_info() const
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
        wi.rd_bytes = w->total_rd_bytes;
        wi.wr_bytes = w->total_wr_bytes;
        info.push_back(wi);
    }
    return info;
}

void DwConvTop::print_report(std::ostream &os) const
{
    const DwConvSimulationStats stats = collect_stats();
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
        {"Total Vector Accelerator Requests [requests]", report::fmt_u64(stats.total_vec_calls)},
        {"Total Memory Requests [requests]", report::fmt_u64(stats.memory_reqs)},
        {"Total Read Bytes [bytes]", report::fmt_u64(stats.total_rd_bytes)},
        {"Total Write Bytes [bytes]", report::fmt_u64(stats.total_wr_bytes)},
        {"Total Stall Cycles [cycles]", report::fmt_u64(total_stall_cycles)},
        {"Total Memory Cycles [cycles]", report::fmt_u64(total_mem_cycles)},
        {"Total Scalar Cycles [cycles]", report::fmt_u64(total_scalar_cycles)},
        {"Average Memory Bandwidth [bytes/cycle]", report::fmt_rate(stats.mem_bw, "bytes/cycle")},
    });

    report::print_section_title(os, "Verification");
    report::print_fields(os, {
        {"Expected Vector Accelerator Requests [requests]", report::fmt_u64(stats.expected_vec_calls)},
        {"Actual Vector Accelerator Requests [requests]", report::fmt_u64(stats.total_vec_calls)},
        {"Verification Status", stats.verification_passed ? "PASS" : "FAIL"},
    });
}

void DwConvTop::done_monitor()
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

    DwConvRuntimeConfig cfg = DwConvRuntimeConfig::defaults();
    DwConvTop top("dw_top", cfg);
    sc_start();

    top.print_report(std::cout);

    const DwConvSimulationStats stats = top.collect_stats();
    return stats.verification_passed ? 0 : 2;
}
#endif
