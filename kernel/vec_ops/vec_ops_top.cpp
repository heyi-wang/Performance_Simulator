#include "vec_ops_top.h"

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

static uint64_t cfg_tile_cap(const VecOpsRuntimeConfig &cfg)
{
    return cfg.tile_cap();
}

static uint64_t cfg_rd_bytes(const VecOpsRuntimeConfig &cfg, uint64_t vl)
{
    switch (cfg.op)
    {
    case VOP_ELEMWISE_ADD:         return vl * cfg.elem_bytes * 2;
    case VOP_ELEMWISE_MUL:         return vl * cfg.elem_bytes * 2;
    case VOP_SCALAR_MUL:           return vl * cfg.elem_bytes;
    case VOP_QUANTIZE_I32_TO_I8:   return vl * 4;
    case VOP_DEQUANTIZE_I8_TO_I32: return vl * 1;
    case VOP_BIAS_ADD_I32:         return vl * 4;
    }
    return 0;
}

static uint64_t cfg_wr_bytes(const VecOpsRuntimeConfig &cfg, uint64_t vl)
{
    switch (cfg.op)
    {
    case VOP_ELEMWISE_ADD:         return vl * cfg.elem_bytes;
    case VOP_ELEMWISE_MUL:         return vl * cfg.elem_bytes;
    case VOP_SCALAR_MUL:           return vl * cfg.elem_bytes;
    case VOP_QUANTIZE_I32_TO_I8:   return vl * 1;
    case VOP_DEQUANTIZE_I8_TO_I32: return vl * 4;
    case VOP_BIAS_ADD_I32:         return vl * 4;
    }
    return 0;
}

static uint64_t cfg_extra_rd_bytes_per_channel(const VecOpsRuntimeConfig &cfg)
{
    return (cfg.op == VOP_BIAS_ADD_I32) ? 4 : 0;
}

struct VecOpsExt : tlm_extension<VecOpsExt>
{
    VopType op_type = VOP_SELECTED_OP;
    int channel_id = -1;
    int tile_idx = 0;

    tlm_extension_base *clone() const override
    {
        return new VecOpsExt(*this);
    }

    void copy_from(const tlm_extension_base &other) override
    {
        *this = static_cast<const VecOpsExt &>(other);
    }
};

struct VecOpsWorker : sc_module
{
    tlm_utils::simple_initiator_socket<VecOpsWorker> init;
    tlm_utils::peq_with_get<tlm_generic_payload> peq;

    int tid;
    int n_workers;
    const VecOpsRuntimeConfig &cfg;
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
        VecOpsExt *vop_ext = nullptr;
        DoneEntry *done_entry = nullptr;
        uint64_t stall_cycles = 0;
        sc_time submit_time = SC_ZERO_TIME;
        bool direct_mem = false;
        bool sync_done = false;
    };

    SC_HAS_PROCESS(VecOpsWorker);

    VecOpsWorker(sc_module_name name,
                 int tid_,
                 int n_workers_,
                 const VecOpsRuntimeConfig &cfg_,
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
        init.register_nb_transport_bw(this, &VecOpsWorker::nb_transport_bw);
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
        auto *vop = new VecOpsExt();
        vop->op_type = cfg.op;
        vop->channel_id = channel_id;
        vop->tile_idx = tile_idx;

        gp->set_extension(req);
        gp->set_extension(tx);
        gp->set_extension(vop);

        p.gp = gp;
        p.req_ext = req;
        p.tx_ext = tx;
        p.vop_ext = vop;
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

    PendingReq issue_mem_read(uint64_t bytes, int channel_id)
    {
        PendingReq p;
        p.direct_mem = true;

        auto *gp = new tlm_generic_payload();
        gp->set_command(TLM_READ_COMMAND);
        gp->set_address(Interconnect::ADDR_MEM);
        gp->set_data_ptr(nullptr);
        gp->set_data_length(static_cast<unsigned>(bytes));
        gp->set_streaming_width(static_cast<unsigned>(bytes));
        gp->set_response_status(TLM_INCOMPLETE_RESPONSE);

        auto *tx = new TxnExt();
        tx->src_worker = tid;
        auto *vop = new VecOpsExt();
        vop->op_type = cfg.op;
        vop->channel_id = channel_id;
        vop->tile_idx = -1;

        gp->set_extension(tx);
        gp->set_extension(vop);

        p.gp = gp;
        p.tx_ext = tx;
        p.vop_ext = vop;
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

        done_map.erase(p.gp);
        delete p.done_entry->ev;
        delete p.done_entry->admit_ev;
        delete p.done_entry;
        p.done_entry = nullptr;

        ReqExt *ext = nullptr;
        p.gp->get_extension(ext);
        if (p.direct_mem)
        {
            total_mem_cycles += static_cast<uint64_t>((sc_time_stamp() - p.submit_time) / CYCLE);
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
        p.gp->clear_extension(p.vop_ext);
        delete p.req_ext;
        delete p.tx_ext;
        delete p.vop_ext;
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
        const uint64_t tile_cap = cfg_tile_cap(cfg);
        const uint64_t extra_rd = cfg_extra_rd_bytes_per_channel(cfg);

        for (int c = c_start; c < c_end; ++c)
        {
            if (extra_rd != 0)
            {
                auto bias_rd = issue_mem_read(extra_rd, c);
                total_rd_bytes += extra_rd;
                issue_end(bias_rd);
            }

            std::vector<PendingReq> pending;
            pending.reserve(static_cast<size_t>(cfg.tile_count()));

            for (int t = 0; t < cfg.tile_count(); ++t)
            {
                uint64_t tile_elems =
                    std::min<uint64_t>(tile_cap,
                                       static_cast<uint64_t>(cfg.spatial()) -
                                           static_cast<uint64_t>(t) * tile_cap);
                uint64_t rd = cfg_rd_bytes(cfg, tile_elems);
                uint64_t wr = cfg_wr_bytes(cfg, tile_elems);
                auto req = issue_begin(rd, wr, c, t);
                ++vec_calls;
                total_rd_bytes += rd;
                total_wr_bytes += wr;
                do_scalar(cfg.scalar_overhead);
                pending.push_back(std::move(req));
            }

            for (auto &req : pending)
                issue_end(req);
        }

        elapsed_cycles = static_cast<uint64_t>((sc_time_stamp() - t_start) / CYCLE);
        if (completion_fifo)
            completion_fifo->write(tid);
    }
};

VecOpsTop::VecOpsTop(sc_module_name name,
                     const VecOpsRuntimeConfig &cfg_,
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
            std::make_unique<sc_fifo<int>>(sc_gen_unique_name("vec_done_fifo"),
                                           cfg.worker_count + 1);
        SC_THREAD(done_monitor);
    }

    for (int i = 0; i < cfg.worker_count; ++i)
    {
        auto *w = new VecOpsWorker(sc_gen_unique_name("vec_ops_worker"),
                                   i,
                                   cfg.worker_count,
                                   cfg,
                                   start_event,
                                   completion_fifo.get());
        workers.push_back(w);
        w->init.bind(noc.tgt);
    }
}

VecOpsTop::~VecOpsTop()
{
    for (auto *w : workers)
        delete w;
}

VecOpsSimulationStats VecOpsTop::collect_stats() const
{
    VecOpsSimulationStats stats;
    const uint64_t tile_cap = cfg_tile_cap(cfg);
    const uint64_t extra_rd_per_channel = cfg_extra_rd_bytes_per_channel(cfg);

    for (const auto *w : workers)
    {
        stats.max_elapsed_cycles = std::max(stats.max_elapsed_cycles, w->elapsed_cycles);
        stats.total_vec_calls += w->vec_calls;
        stats.total_rd_bytes += w->total_rd_bytes;
        stats.total_wr_bytes += w->total_wr_bytes;
        stats.total_wait_cycles += w->total_wait_cycles;
        stats.total_mem_cycles += w->total_mem_cycles;
    }

    stats.expected_vec_calls =
        static_cast<uint64_t>(cfg.channels) * static_cast<uint64_t>(cfg.tile_count());
    uint64_t per_chan_rd = 0;
    uint64_t per_chan_wr = 0;
    for (int t = 0; t < cfg.tile_count(); ++t)
    {
        uint64_t vl = std::min<uint64_t>(
            tile_cap,
            static_cast<uint64_t>(cfg.spatial()) - static_cast<uint64_t>(t) * tile_cap);
        per_chan_rd += cfg_rd_bytes(cfg, vl);
        per_chan_wr += cfg_wr_bytes(cfg, vl);
    }
    stats.expected_rd_bytes =
        static_cast<uint64_t>(cfg.channels) * (per_chan_rd + extra_rd_per_channel);
    stats.expected_wr_bytes = static_cast<uint64_t>(cfg.channels) * per_chan_wr;
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
    const uint64_t total_mem_bytes = stats.total_rd_bytes + stats.total_wr_bytes;
    stats.mem_bw = (sim_cycles > 0.0)
        ? static_cast<double>(total_mem_bytes) / sim_cycles
        : 0.0;
    stats.verification_passed =
        stats.total_vec_calls == stats.expected_vec_calls &&
        stats.total_rd_bytes == stats.expected_rd_bytes &&
        stats.total_wr_bytes == stats.expected_wr_bytes;
    return stats;
}

std::vector<KernelWorkerInfo> VecOpsTop::collect_worker_info() const
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

void VecOpsTop::print_report(std::ostream &os) const
{
    const VecOpsSimulationStats stats = collect_stats();
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
        {"Operation Type", vop_name(cfg.op)},
        {"Input Tensor Shape", "[C=" + report::fmt_int(cfg.channels) +
                               ", H=" + report::fmt_int(cfg.height) +
                               ", W=" + report::fmt_int(cfg.width) + "]"},
        {"Element Size [bytes]", report::fmt_u64(cfg.elem_bytes)},
    });

    report::print_section_title(os, "Hardware Configuration");
    report::print_fields(os, {
        {"Workers [count]", report::fmt_int(cfg.worker_count)},
        {"Matrix Accelerators [count]", report::na()},
        {"Vector Accelerators [count]", report::fmt_int(cfg.vec_acc_instances)},
        {"Matrix Accelerator Capacity", report::na()},
        {"Vector Accelerator Capacity [elements/request]", report::fmt_u64(cfg.tile_cap())},
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
        {"Expected Read Bytes [bytes]", report::fmt_u64(stats.expected_rd_bytes)},
        {"Actual Read Bytes [bytes]", report::fmt_u64(stats.total_rd_bytes)},
        {"Expected Write Bytes [bytes]", report::fmt_u64(stats.expected_wr_bytes)},
        {"Actual Write Bytes [bytes]", report::fmt_u64(stats.total_wr_bytes)},
        {"Verification Status", stats.verification_passed ? "PASS" : "FAIL"},
    });
}

void VecOpsTop::done_monitor()
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

    VecOpsRuntimeConfig cfg = VecOpsRuntimeConfig::defaults();
    VecOpsTop top("vec_ops_top", cfg);
    sc_start();

    top.print_report(std::cout);

    const VecOpsSimulationStats stats = top.collect_stats();
    return stats.verification_passed ? 0 : 2;
}
#endif
