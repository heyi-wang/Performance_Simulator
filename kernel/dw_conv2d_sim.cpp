// dw_conv2d_sim.cpp
// SystemC TLM-2.0 performance simulator for depth-wise 2D convolution in NAFNet.
//
// Maps mf_dw_conv2d_3x3_i16 (kernel/dw_conv2d.h) to the shared src/ hardware.
//
// Algorithm (from dw_conv2d.h, int16 path):
//   for c in [0, C):
//     load 9 kernel weights (scalars, hoisted per channel)
//     for oh in [0, outH):
//       strip-mine ow in [0, outW) with vl = DW_VEC_ACC_CAP:
//         acc = 0
//         for kh, kw in [0, kH) x [0, kW):
//           load vl int16 → broadcast kernel scalar → widening MAC → int32 acc
//         store vl int32 elements
//
// Each strip of DW_VEC_ACC_CAP output pixels maps to one vec_acc request.
//
// Workers are partitioned by channel: worker tid owns channels [c_start, c_end).
// Communication and synchronisation are consistent with layer_norm_sim.cpp:
//   nb_transport_fw/bw, TLM_UPDATED/TLM_ACCEPTED, admit_ev back-pressure.

#include <systemc>
#include <tlm>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/peq_with_get.h>

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <unordered_map>

#include "../src/common.h"
#include "../src/extensions.h"
#include "../src/accelerator.h"
#include "../src/accelerator_pool.h"
#include "../src/interconnect.h"
#include "../src/memory.h"

#include "dw_conv2d_config.h"

using namespace sc_core;
using namespace tlm;

// ============================================================
// DwConvExt — TLM extension carrying dw-conv request metadata.
// Attached alongside ReqExt and TxnExt on each transaction.
// AcceleratorTLM / Interconnect / Memory do not inspect this.
// ============================================================
struct DwConvExt : tlm_extension<DwConvExt>
{
    int channel_id = -1;   // channel this strip belongs to
    int out_row    =  0;   // output row index (oh)
    int strip_idx  =  0;   // strip index within the output row

    tlm_extension_base *clone() const override
    {
        return new DwConvExt(*this);
    }

    void copy_from(const tlm_extension_base &other) override
    {
        *this = static_cast<const DwConvExt &>(other);
    }
};

// ============================================================
// DwConvWorker — per-thread task generator.
//
// Channel assignment:
//   c_start = (tid   * DW_C) / n_workers
//   c_end   = ((tid+1) * DW_C) / n_workers
//
// For every assigned channel the worker iterates over all
// output rows.  Within each row it fire-then-drains all strips:
//
//   for oh in [0, outH):
//     for each strip in [0, strips_per_row):
//       issue_begin(...)    // fire request
//       do_scalar(overhead) // model dispatch scalar work
//     for each strip:
//       issue_end(...)      // collect result
//
// Memory traffic per strip (tile of DW_VEC_ACC_CAP output pixels):
//   rd = sum over valid (kh, kw, lane) input elements touched by the
//        int16 kernel's fast path / boundary path
//      + kH * kW * DW_INPUT_ELEM_BYTES on the first strip of each channel
//        (the 9 kernel weights are hoisted once per channel)
//   wr = vl * DW_OUTPUT_ELEM_BYTES
// where vl = min(DW_VEC_ACC_CAP, outW - strip*DW_VEC_ACC_CAP).
// ============================================================
struct DwConvWorker : sc_module
{
    tlm_utils::simple_initiator_socket<DwConvWorker>   init;
    tlm_utils::peq_with_get<tlm_generic_payload>       peq;

    int tid;
    int n_workers;

    // Statistics
    uint64_t vec_calls             = 0;   // total vec_acc requests issued
    uint64_t total_scalar_cycles   = 0;   // dispatch scalar overhead
    uint64_t total_wait_cycles     = 0;   // queue-wait + back-pressure stall
    uint64_t total_mem_cycles      = 0;   // memory service cycles
    uint64_t total_rd_bytes        = 0;   // total memory bytes read
    uint64_t total_wr_bytes        = 0;   // total memory bytes written
    uint64_t elapsed_cycles        = 0;   // wall-clock cycles start→finish

    // ----------------------------------------------------------
    // Per-request synchronisation bookkeeping.
    //   ev:       notified when AcceleratorTLM sends BEGIN_RESP.
    //   admit_ev: notified when AcceleratorTLM sends deferred
    //             END_REQ (back-pressure: queue slot granted).
    //   fired:    set true before notifying ev so issue_end can
    //             skip wait() if the response already arrived.
    // ----------------------------------------------------------
    struct DoneEntry
    {
        sc_event *ev       = nullptr;
        sc_event *admit_ev = nullptr;
        bool      fired    = false;
    };

    std::unordered_map<tlm_generic_payload *, DoneEntry *> done_map;

    // Handle returned by issue_begin; consumed by issue_end.
    struct PendingReq
    {
        tlm_generic_payload *gp           = nullptr;
        ReqExt              *req_ext      = nullptr;
        TxnExt              *tx_ext       = nullptr;
        DwConvExt           *dw_ext       = nullptr;
        DoneEntry           *done_entry   = nullptr;
        uint64_t             svc_cyc      = 0;
        uint64_t             stall_cycles = 0;
        bool                 sync_done    = false;
    };

    SC_HAS_PROCESS(DwConvWorker);

    DwConvWorker(sc_module_name name, int tid_, int n_workers_)
        : sc_module(name),
          init("init"),
          peq("peq"),
          tid(tid_),
          n_workers(n_workers_)
    {
        init.register_nb_transport_bw(this, &DwConvWorker::nb_transport_bw);
        SC_THREAD(peq_thread);
        SC_THREAD(run);
    }

    // ----------------------------------------------------------
    // nb_transport_bw: receive BEGIN_RESP or deferred END_REQ
    // ----------------------------------------------------------
    tlm_sync_enum nb_transport_bw(tlm_generic_payload &gp,
                                  tlm_phase           &phase,
                                  sc_time             &delay)
    {
        if (phase == BEGIN_RESP)
        {
            peq.notify(gp, delay);
            return TLM_ACCEPTED;
        }
        if (phase == END_REQ)
        {
            // Deferred admission: queue was full; accelerator grants a slot.
            auto it = done_map.find(&gp);
            if (it != done_map.end() && it->second && it->second->admit_ev)
                it->second->admit_ev->notify(SC_ZERO_TIME);
            return TLM_ACCEPTED;
        }
        return TLM_ACCEPTED;
    }

    // ----------------------------------------------------------
    // peq_thread: wake up done events when responses arrive.
    // Sets fired=true before notifying so issue_end can detect
    // a response that arrived before wait() was called.
    // ----------------------------------------------------------
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

    // ----------------------------------------------------------
    // do_scalar: model per-tile scalar dispatch overhead.
    // ----------------------------------------------------------
    void do_scalar(uint64_t cyc)
    {
        total_scalar_cycles += cyc;
        wait(cyc * CYCLE);
    }

    // ----------------------------------------------------------
    // calc_strip_input_read_bytes: exact input traffic for one
    // strip of mf_dw_conv2d_3x3_i16.
    //
    // Mirrors the kernel's bounds checks:
    //   ih = oh - pad + kh
    //   iw = strip_start - pad + kw + lane
    // and only counts input elements that are in range.
    // ----------------------------------------------------------
    uint64_t calc_strip_input_read_bytes(int oh,
                                         uint64_t strip_start,
                                         uint64_t vl) const
    {
        uint64_t rd = 0;

        for (int kh = 0; kh < DW_KH; ++kh)
        {
            int ih = oh - DW_PAD + kh;
            if (ih < 0 || ih >= DW_H)
                continue;

            for (int kw = 0; kw < DW_KW; ++kw)
            {
                int64_t iw_base = static_cast<int64_t>(strip_start) - DW_PAD + kw;
                int64_t lo = std::max<int64_t>(0, iw_base);
                int64_t hi = std::min<int64_t>(
                    static_cast<int64_t>(DW_W), iw_base + static_cast<int64_t>(vl));

                if (hi > lo)
                    rd += static_cast<uint64_t>(hi - lo) * DW_INPUT_ELEM_BYTES;
            }
        }

        return rd;
    }

    // ----------------------------------------------------------
    // issue_begin: fire one non-blocking vec_acc request.
    // Blocks only when the accelerator queue is full (TLM_ACCEPTED),
    // waiting on admit_ev until a slot is granted.
    // ----------------------------------------------------------
    PendingReq issue_begin(uint64_t addr,
                           uint64_t svc_cyc,
                           uint64_t rd,
                           uint64_t wr,
                           int      channel_id,
                           int      out_row,
                           int      strip_idx)
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
        auto *tx  = new TxnExt();
        tx->src_worker = tid;

        auto *dw  = new DwConvExt();
        dw->channel_id = channel_id;
        dw->out_row    = out_row;
        dw->strip_idx  = strip_idx;

        gp->set_extension(req);
        gp->set_extension(tx);
        gp->set_extension(dw);

        p.gp         = gp;
        p.req_ext    = req;
        p.tx_ext     = tx;
        p.dw_ext     = dw;
        p.done_entry = new DoneEntry();
        p.done_entry->ev       = new sc_event();
        p.done_entry->admit_ev = new sc_event();
        p.done_entry->fired    = false;
        done_map[gp] = p.done_entry;

        tlm_phase phase = BEGIN_REQ;
        sc_time   delay = SC_ZERO_TIME;
        auto status = init->nb_transport_fw(*gp, phase, delay);

        if (status == TLM_ACCEPTED)
        {
            // Queue was full: stall until accelerator grants a slot.
            sc_time t_stall_start = sc_time_stamp();
            wait(*p.done_entry->admit_ev);
            p.stall_cycles =
                (uint64_t)((sc_time_stamp() - t_stall_start) / CYCLE);
        }
        else if (status == TLM_COMPLETED)
        {
            done_map.erase(gp);
            p.sync_done = true;
        }
        return p;
    }

    // ----------------------------------------------------------
    // issue_end: wait for completion, harvest stats, send END_RESP.
    // ----------------------------------------------------------
    void issue_end(PendingReq &p)
    {
        if (!p.sync_done && !p.done_entry->fired)
            wait(*p.done_entry->ev);

        done_map.erase(p.gp);
        delete p.done_entry->ev;
        delete p.done_entry->admit_ev;
        delete p.done_entry;
        p.done_entry = nullptr;

        // Read back timing fields filled in by AcceleratorTLM.
        ReqExt *ext = nullptr;
        p.gp->get_extension(ext);
        uint64_t qwait = ext ? ext->accel_qwait_cycles : 0;
        uint64_t mec   = ext ? ext->mem_cycles          : 0;

        total_wait_cycles += qwait + p.stall_cycles;
        total_mem_cycles  += mec;

        // Acknowledge response back to the interconnect.
        tlm_phase end_phase = END_RESP;
        sc_time   end_delay = SC_ZERO_TIME;
        init->nb_transport_fw(*p.gp, end_phase, end_delay);

        // Release all extensions and the payload.
        p.gp->clear_extension(p.req_ext);
        p.gp->clear_extension(p.tx_ext);
        p.gp->clear_extension(p.dw_ext);
        delete p.req_ext; p.req_ext = nullptr;
        delete p.tx_ext;  p.tx_ext  = nullptr;
        delete p.dw_ext;  p.dw_ext  = nullptr;
        delete p.gp;      p.gp      = nullptr;
    }

    // ----------------------------------------------------------
    // run: main worker thread — process assigned channel slice.
    // ----------------------------------------------------------
    void run()
    {
        sc_time t_start = sc_time_stamp();

        // Channel range for this worker.
        int c_start = (tid       * DW_C) / n_workers;
        int c_end   = ((tid + 1) * DW_C) / n_workers;

        // Convolution geometry (const per simulation).
        const int outH = DW_OUT_H;
        const int outW = DW_OUT_W;

        // Number of vec_acc strips per output row.
        const int strips_per_row =
            (int)ceil_div_u64((uint64_t)outW, DW_VEC_ACC_CAP);

        const uint64_t kernel_rd_bytes =
            static_cast<uint64_t>(DW_KH * DW_KW) * DW_INPUT_ELEM_BYTES;

        for (int c = c_start; c < c_end; ++c)
        {
            for (int oh = 0; oh < outH; ++oh)
            {
                // ------------------------------------------------
                // Fire phase: issue all strips for this output row.
                // ------------------------------------------------
                std::vector<PendingReq> pending;
                pending.reserve((size_t)strips_per_row);

                for (int s = 0; s < strips_per_row; ++s)
                {
                    // Actual number of output pixels in this strip.
                    uint64_t strip_start = (uint64_t)s * DW_VEC_ACC_CAP;
                    uint64_t vl = std::min<uint64_t>(
                        DW_VEC_ACC_CAP,
                        (uint64_t)outW - strip_start);

                    // Exact input traffic for the valid overlap of this strip.
                    // Charge the hoisted 3x3 kernel weights once per channel.
                    uint64_t rd =
                        calc_strip_input_read_bytes(oh, strip_start, vl);
                    if (oh == 0 && s == 0)
                        rd += kernel_rd_bytes;
                    uint64_t wr = vl * DW_OUTPUT_ELEM_BYTES;

                    auto pm = issue_begin(Interconnect::ADDR_VEC,
                                          DW_VEC_ACC_CYCLE,
                                          rd, wr,
                                          c, oh, s);
                    ++vec_calls;
                    total_rd_bytes += rd;
                    total_wr_bytes += wr;

                    do_scalar(DW_SCALAR_OVERHEAD);
                    pending.push_back(std::move(pm));
                }

                // ------------------------------------------------
                // Drain phase: collect all responses for this row.
                // ------------------------------------------------
                for (auto &pm : pending)
                    issue_end(pm);
            }
        }

        elapsed_cycles =
            (uint64_t)((sc_time_stamp() - t_start) / CYCLE);
    }
};

// ============================================================
// DwConvTop — instantiates and wires the full simulator.
//
// Topology:
//   workers[0..N-1]  ──► noc ──► vec_acc ──► memory
//                             └──► mat_acc (dummy, never used)
//
// mat_acc is created only to satisfy the noc.to_mat binding.
// No worker ever sends to ADDR_MAT.
// ============================================================
struct DwConvTop : sc_module
{
    AcceleratorTLM  mat_acc;   // dummy: bound to satisfy noc.to_mat
    AcceleratorPool vec_acc;   // actual: pool of DW_VEC_ACC_INSTANCES units
    Interconnect    noc;
    Memory          memory;

    std::vector<DwConvWorker *> workers;

    SC_HAS_PROCESS(DwConvTop);

    DwConvTop(sc_module_name name)
        : sc_module(name),
          mat_acc("mat_acc", DW_ACC_QUEUE_DEPTH),
          vec_acc("vec_acc",
                  (size_t)DW_VEC_ACC_INSTANCES,
                  DW_ACC_QUEUE_DEPTH),
          noc("noc"),
          memory("memory",
                 DW_MEM_BASE_LAT,
                 DW_MEM_BW,
                 (uint64_t)DW_VEC_ACC_INSTANCES)
    {
        // Bind accelerators and memory to the interconnect.
        noc.to_mat.bind(mat_acc.tgt);
        noc.to_vec.bind(vec_acc.tgt);
        noc.to_mem.bind(memory.tgt);

        // Accelerators reach memory through the interconnect.
        mat_acc.to_mem.bind(noc.tgt);
        for (auto &unit : vec_acc.units)
            unit->to_mem.bind(noc.tgt);

        // Create and connect workers.
        for (int i = 0; i < DW_NUM_WORKERS; ++i)
        {
            auto *w = new DwConvWorker(
                          sc_gen_unique_name("dw_worker"),
                          i, DW_NUM_WORKERS);
            workers.push_back(w);
            w->init.bind(noc.tgt);
        }
    }

    ~DwConvTop() override
    {
        for (auto *w : workers)
            delete w;
    }
};

// ============================================================
// sc_main — run simulation and print three-section report.
// ============================================================
int sc_main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    DwConvTop top("dw_top");
    sc_start();

    // ----------------------------------------------------------
    // Derived geometry (mirrors dw_conv2d_config.h)
    // ----------------------------------------------------------
    const int    outH          = DW_OUT_H;
    const int    outW          = DW_OUT_W;
    const int    strips_per_row =
        (int)ceil_div_u64((uint64_t)outW, DW_VEC_ACC_CAP);
    const uint64_t calls_per_chan =
        (uint64_t)outH * (uint64_t)strips_per_row;
    const uint64_t expected_calls =
        (uint64_t)DW_C * calls_per_chan;

    // ----------------------------------------------------------
    // Section 0: Configuration header
    // ----------------------------------------------------------
    std::cout << "\n";
    std::cout << "==============================================\n";
    std::cout << "  Depth-wise Conv2d TLM Performance Simulation\n";
    std::cout << "  Algorithm : mf_dw_conv2d_3x3_i"
              << (DW_INPUT_ELEM_BYTES * 8) << "\n";
    std::cout << "  Input     : [C=" << DW_C
              << ", H=" << DW_H
              << ", W=" << DW_W << "]"
              << "  (int" << (DW_INPUT_ELEM_BYTES * 8) << ")\n";
    std::cout << "  Kernel    : [kH=" << DW_KH
              << ", kW=" << DW_KW << "]"
              << "  pad=" << DW_PAD
              << "  stride=" << DW_STRIDE << "\n";
    std::cout << "  Output    : [C=" << DW_C
              << ", outH=" << outH
              << ", outW=" << outW << "]"
              << "  (int" << (DW_OUTPUT_ELEM_BYTES * 8) << ")\n";
    std::cout << "  Input elem bytes  : " << DW_INPUT_ELEM_BYTES
              << "  (change DW_INPUT_ELEM_BYTES in dw_conv2d_config.h to switch)\n";
    std::cout << "  Output elem bytes : " << DW_OUTPUT_ELEM_BYTES << "\n";
    std::cout << "  Workers           : " << DW_NUM_WORKERS        << "\n";
    std::cout << "  vec_acc units     : " << DW_VEC_ACC_INSTANCES  << "\n";
    std::cout << "  VEC_ACC_CAP       : " << DW_VEC_ACC_CAP
              << " elements/call,  " << DW_VEC_ACC_CYCLE << " cycles/call\n";
    std::cout << "  Strips per row    : " << strips_per_row << "\n";
    std::cout << "  Calls per channel : " << calls_per_chan  << "\n";
    std::cout << "  Expected total    : " << expected_calls << " vec_acc calls\n";
    std::cout << "==============================================\n";

    // ----------------------------------------------------------
    // Section 1: Per-worker summary
    // ----------------------------------------------------------
    std::cout << "\n--- Per-Worker Summary ---\n";
    std::cout << std::left
              << std::setw(8)  << "Worker"
              << std::setw(14) << "ChanRange"
              << std::setw(10) << "VecCalls"
              << std::setw(14) << "ScalarCyc"
              << std::setw(12) << "WaitCyc"
              << std::setw(12) << "MemCyc"
              << std::setw(14) << "RdBytes"
              << std::setw(14) << "WrBytes"
              << std::setw(12) << "ElapsedCyc"
              << "\n";
    std::cout << std::string(110, '-') << "\n";

    uint64_t max_elapsed    = 0;
    uint64_t total_vec      = 0;
    uint64_t total_rd_all   = 0;
    uint64_t total_wr_all   = 0;
    uint64_t total_wait_all = 0;
    uint64_t total_mem_all  = 0;

    for (const auto *w : top.workers)
    {
        int c_start = (w->tid       * DW_C) / DW_NUM_WORKERS;
        int c_end   = ((w->tid + 1) * DW_C) / DW_NUM_WORKERS;
        std::string range = "[" + std::to_string(c_start)
                          + ","  + std::to_string(c_end)  + ")";
        std::cout << std::left
                  << std::setw(8)  << w->tid
                  << std::setw(14) << range
                  << std::setw(10) << w->vec_calls
                  << std::setw(14) << w->total_scalar_cycles
                  << std::setw(12) << w->total_wait_cycles
                  << std::setw(12) << w->total_mem_cycles
                  << std::setw(14) << w->total_rd_bytes
                  << std::setw(14) << w->total_wr_bytes
                  << std::setw(12) << w->elapsed_cycles
                  << "\n";

        max_elapsed    = std::max(max_elapsed,    w->elapsed_cycles);
        total_vec      += w->vec_calls;
        total_rd_all   += w->total_rd_bytes;
        total_wr_all   += w->total_wr_bytes;
        total_wait_all += w->total_wait_cycles;
        total_mem_all  += w->total_mem_cycles;
    }

    // ----------------------------------------------------------
    // Section 2: Global summary
    // ----------------------------------------------------------
    std::cout << "\n--- Global Summary ---\n";
    std::cout << "Simulation time           : " << sc_time_stamp()  << "\n";
    std::cout << "Max worker elapsed        : " << max_elapsed       << " cycles\n";
    std::cout << "Workers                   : " << DW_NUM_WORKERS    << "\n";
    std::cout << "Channels (C)              : " << DW_C              << "\n";
    std::cout << "Output spatial (outH*outW): " << outH << " x " << outW << "\n";
    std::cout << "Strips per row            : " << strips_per_row     << "\n";
    std::cout << "Total vec-acc calls       : " << total_vec
              << "  (expected " << expected_calls << ") "
              << (total_vec == expected_calls ? "[OK]" : "[MISMATCH]") << "\n";
    std::cout << "Total memory reads        : " << total_rd_all       << " bytes\n";
    std::cout << "Total memory writes       : " << total_wr_all       << " bytes\n";
    std::cout << "Total worker wait cycles  : " << total_wait_all     << "\n";
    std::cout << "Total worker mem cycles   : " << total_mem_all      << "\n";

    std::cout << "\n";
    std::cout << "vec_acc pool  : units="   << top.vec_acc.instance_count()
              << "  reqs="                  << top.vec_acc.req_count_total()
              << "  busy_cyc="              << top.vec_acc.busy_cycles_total()
              << "  occupied_cyc="          << top.vec_acc.occupied_cycles_total()
              << "  qwait_cyc="             << top.vec_acc.queue_wait_cycles_total()
              << "\n";
    std::cout << "memory        : reqs="    << top.memory.reqs
              << "  busy_cyc="              << top.memory.busy_cycles
              << "  qwait_cyc="             << top.memory.qwait_cycles
              << "\n";

    // Accelerator utilisation: busy cycles / (sim_cycles × num_units)
    double sim_cycles    = static_cast<double>(sc_time_stamp() / CYCLE);
    double vec_capacity  = sim_cycles *
                           static_cast<double>(top.vec_acc.instance_count());
    double vec_util = (vec_capacity > 0.0)
        ? static_cast<double>(top.vec_acc.busy_cycles_total())
              / vec_capacity * 100.0
        : 0.0;
    double vec_occ  = (vec_capacity > 0.0)
        ? static_cast<double>(top.vec_acc.occupied_cycles_total())
              / vec_capacity * 100.0
        : 0.0;

    // Memory bandwidth: bytes processed / total cycles
    double mem_bw = (sim_cycles > 0.0)
        ? static_cast<double>(top.memory.busy_cycles) * DW_MEM_BW
              / sim_cycles
        : 0.0;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "vec_acc compute util      : " << vec_util << "%\n";
    std::cout << "vec_acc occupancy         : " << vec_occ  << "%\n";
    std::cout << "Memory avg BW             : " << mem_bw   << " bytes/cycle\n";

    // Bottleneck hint
    if (vec_util > 80.0)
        std::cout << "Bottleneck hint           : vec_acc is the primary bottleneck.\n";
    else if (top.memory.busy_cycles > top.vec_acc.busy_cycles_total())
        std::cout << "Bottleneck hint           : memory bandwidth is the primary bottleneck.\n";
    else
        std::cout << "Bottleneck hint           : load is balanced between vec_acc and memory.\n";

    // ----------------------------------------------------------
    // Verification check
    // ----------------------------------------------------------
    std::cout << "\n--- Verification ---\n";
    bool pass = (total_vec == expected_calls);
    std::cout << (pass ? "  [PASS]" : "  [FAIL]")
              << " Total vec-acc calls == C * outH * ceil(outW/CAP)"
              << " (" << total_vec << " == " << expected_calls << ")\n";

    std::cout << (pass ? "\nAll checks passed.\n" : "\nSome checks FAILED.\n");
    return pass ? 0 : 2;
}
