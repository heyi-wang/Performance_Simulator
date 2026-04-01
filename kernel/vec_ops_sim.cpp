// vec_ops_sim.cpp
// SystemC TLM-2.0 performance simulator for element-wise vector operations
// in NAFNet, as defined in kernel/vector_ops.h.
//
// Supported operations (selected via VOP_SELECTED_OP in vec_ops_config.h):
//   ELEMWISE_ADD        mf_elemwise_add_i8        rd = 2*vl*elem  wr = vl*elem
//   ELEMWISE_MUL        mf_elemwise_mul_i8        rd = 2*vl*elem  wr = vl*elem
//   SCALAR_MUL          mf_elemwise_mul_scalar_i8  rd = vl*elem   wr = vl*elem
//   QUANTIZE_I32_TO_I8  mf_quantize_i32_to_i8     rd = vl*4      wr = vl*1
//   DEQUANTIZE_I8_TO_I32 mf_dequantize_i8_to_i32  rd = vl*1      wr = vl*4
//   BIAS_ADD_I32        mf_bias_add_i32           rd = vl*4      wr = vl*4
//
// All operations share the same RVV stripmining loop structure:
//   for each channel c in [0, C):
//     for each tile t in [0, n_tiles):
//       vl = min(VEC_ACC_CAP, spatial - t * VEC_ACC_CAP)
//       load input(s), compute, store output
//
// Memory traffic model
// --------------------
// READ  per tile : vop_rd_bytes(op, vl)  — depends on operation type
// WRITE per tile : vop_wr_bytes(op, vl)  — depends on operation type
//
// Unlike pooling_sim, all read AND write traffic flows through vec_acc.
// There is no scalar post-processing step and no direct memory writes.
//
// Workers are partitioned by channel: worker tid owns [c_start, c_end).
// Communication and synchronisation are consistent with pooling_sim.cpp:
//   nb_transport_fw/bw, TLM_UPDATED/TLM_ACCEPTED, admit_ev back-pressure,
//   fire-then-drain per-channel pattern.

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

#include "vec_ops_config.h"

using namespace sc_core;
using namespace tlm;

// ============================================================
// VecOpsExt — TLM extension carrying vector-op request metadata.
// Attached alongside ReqExt and TxnExt on each transaction.
// AcceleratorTLM / Interconnect / Memory do not inspect this.
// ============================================================
struct VecOpsExt : tlm_extension<VecOpsExt>
{
    VopType op_type   = VOP_SELECTED_OP;
    int     channel_id = -1;
    int     tile_idx   =  0;

    tlm_extension_base *clone() const override
    {
        return new VecOpsExt(*this);
    }

    void copy_from(const tlm_extension_base &other) override
    {
        *this = static_cast<const VecOpsExt &>(other);
    }
};

// ============================================================
// VecOpsWorker — per-thread task generator.
//
// Channel assignment:
//   c_start = (tid   * VOP_C) / n_workers
//   c_end   = ((tid+1) * VOP_C) / n_workers
//
// For every assigned channel the worker:
//   1. Fire phase  — issues all n_tiles vec_acc requests, each
//      carrying rd and wr bytes according to the selected op.
//   2. Drain phase — collects all responses for the channel.
//
// No scalar post-processing (unlike pooling_sim which has a
// divide + direct memory write step).
// ============================================================
struct VecOpsWorker : sc_module
{
    tlm_utils::simple_initiator_socket<VecOpsWorker> init;
    tlm_utils::peq_with_get<tlm_generic_payload>     peq;

    int tid;
    int n_workers;

    // Statistics
    uint64_t vec_calls           = 0;
    uint64_t total_scalar_cycles = 0;
    uint64_t total_wait_cycles   = 0;
    uint64_t total_mem_cycles    = 0;
    uint64_t total_rd_bytes      = 0;
    uint64_t total_wr_bytes      = 0;
    uint64_t elapsed_cycles      = 0;

    struct DoneEntry
    {
        sc_event *ev       = nullptr;
        sc_event *admit_ev = nullptr;
        bool      fired    = false;
    };

    std::unordered_map<tlm_generic_payload *, DoneEntry *> done_map;

    struct PendingReq
    {
        tlm_generic_payload *gp         = nullptr;
        ReqExt              *req_ext    = nullptr;
        TxnExt              *tx_ext     = nullptr;
        VecOpsExt           *vop_ext    = nullptr;
        DoneEntry           *done_entry = nullptr;
        uint64_t             svc_cyc    = 0;
        uint64_t             stall_cycles = 0;
        sc_time              submit_time = SC_ZERO_TIME;
        bool                 direct_mem = false;
        bool                 sync_done  = false;
    };

    SC_HAS_PROCESS(VecOpsWorker);

    VecOpsWorker(sc_module_name name, int tid_, int n_workers_)
        : sc_module(name),
          init("init"),
          peq("peq"),
          tid(tid_),
          n_workers(n_workers_)
    {
        init.register_nb_transport_bw(this, &VecOpsWorker::nb_transport_bw);
        SC_THREAD(peq_thread);
        SC_THREAD(run);
    }

    // ----------------------------------------------------------
    // nb_transport_bw: receive BEGIN_RESP or deferred END_REQ.
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
            auto it = done_map.find(&gp);
            if (it != done_map.end() && it->second && it->second->admit_ev)
                it->second->admit_ev->notify(SC_ZERO_TIME);
            return TLM_ACCEPTED;
        }
        return TLM_ACCEPTED;
    }

    // ----------------------------------------------------------
    // peq_thread: wake up done events when responses arrive.
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

    void do_scalar(uint64_t cyc)
    {
        total_scalar_cycles += cyc;
        wait(cyc * CYCLE);
    }

    // ----------------------------------------------------------
    // issue_begin: fire one non-blocking vec_acc request.
    // ----------------------------------------------------------
    PendingReq issue_begin(uint64_t addr,
                           uint64_t svc_cyc,
                           uint64_t rd,
                           uint64_t wr,
                           int      channel_id,
                           int      tile_idx)
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

        auto *vop = new VecOpsExt();
        vop->op_type    = VOP_SELECTED_OP;
        vop->channel_id = channel_id;
        vop->tile_idx   = tile_idx;

        gp->set_extension(req);
        gp->set_extension(tx);
        gp->set_extension(vop);

        p.gp         = gp;
        p.req_ext    = req;
        p.tx_ext     = tx;
        p.vop_ext    = vop;
        p.done_entry = new DoneEntry();
        p.done_entry->ev       = new sc_event();
        p.done_entry->admit_ev = new sc_event();
        p.done_entry->fired    = false;
        done_map[gp] = p.done_entry;

        tlm_phase phase = BEGIN_REQ;
        sc_time   delay = SC_ZERO_TIME;
        p.submit_time = sc_time_stamp();
        auto status = init->nb_transport_fw(*gp, phase, delay);

        if (status == TLM_ACCEPTED)
        {
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
    // issue_mem_read: fire one direct memory read request.
    // Used for scalar-side accesses that happen outside the vector
    // stripmining loop, such as mf_bias_add_i32 loading bias[c].
    // ----------------------------------------------------------
    PendingReq issue_mem_read(uint64_t bytes, int channel_id)
    {
        PendingReq p;
        p.direct_mem = true;

        auto *gp = new tlm_generic_payload();
        gp->set_command(TLM_READ_COMMAND);
        gp->set_address(Interconnect::ADDR_MEM);
        gp->set_data_ptr(nullptr);
        gp->set_data_length((unsigned)bytes);
        gp->set_streaming_width((unsigned)bytes);
        gp->set_response_status(TLM_INCOMPLETE_RESPONSE);

        auto *tx = new TxnExt();
        tx->src_worker = tid;

        auto *vop = new VecOpsExt();
        vop->op_type    = VOP_SELECTED_OP;
        vop->channel_id = channel_id;
        vop->tile_idx   = -1;

        gp->set_extension(tx);
        gp->set_extension(vop);

        p.gp         = gp;
        p.tx_ext     = tx;
        p.vop_ext    = vop;
        p.done_entry = new DoneEntry();
        p.done_entry->ev       = new sc_event();
        p.done_entry->admit_ev = new sc_event();
        p.done_entry->fired    = false;
        done_map[gp] = p.done_entry;

        tlm_phase phase = BEGIN_REQ;
        sc_time   delay = SC_ZERO_TIME;
        p.submit_time = sc_time_stamp();
        auto status = init->nb_transport_fw(*gp, phase, delay);

        if (status == TLM_COMPLETED)
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

        ReqExt *ext = nullptr;
        p.gp->get_extension(ext);
        uint64_t qwait = ext ? ext->accel_qwait_cycles : 0;
        uint64_t mec   = ext ? ext->mem_cycles          : 0;

        if (p.direct_mem)
        {
            total_mem_cycles +=
                (uint64_t)((sc_time_stamp() - p.submit_time) / CYCLE);
        }
        else
        {
            total_wait_cycles += qwait + p.stall_cycles;
            total_mem_cycles  += mec;
        }

        tlm_phase end_phase = END_RESP;
        sc_time   end_delay = SC_ZERO_TIME;
        init->nb_transport_fw(*p.gp, end_phase, end_delay);

        p.gp->clear_extension(p.req_ext);
        p.gp->clear_extension(p.tx_ext);
        p.gp->clear_extension(p.vop_ext);
        delete p.req_ext;  p.req_ext  = nullptr;
        delete p.tx_ext;   p.tx_ext   = nullptr;
        delete p.vop_ext;  p.vop_ext  = nullptr;
        delete p.gp;       p.gp       = nullptr;
    }

    // ----------------------------------------------------------
    // run: main worker thread — process assigned channel slice.
    //
    // For each channel:
    //   Fire phase : issue all tiles with rd/wr per the selected op
    //   Drain phase: collect all responses
    // ----------------------------------------------------------
    void run()
    {
        sc_time t_start = sc_time_stamp();

        const int      spatial   = VOP_H * VOP_W;
        const uint64_t tile_cap  = vop_tile_cap_elems(VOP_SELECTED_OP);
        const uint64_t extra_rd  = vop_extra_rd_bytes_per_channel(VOP_SELECTED_OP);
        const int      n_tiles   =
            (int)ceil_div_u64((uint64_t)spatial, tile_cap);

        int c_start = (tid       * VOP_C) / n_workers;
        int c_end   = ((tid + 1) * VOP_C) / n_workers;

        for (int c = c_start; c < c_end; ++c)
        {
            if (extra_rd != 0)
            {
                auto bias_rd = issue_mem_read(extra_rd, c);
                total_rd_bytes += extra_rd;
                issue_end(bias_rd);
            }

            // Fire phase
            std::vector<PendingReq> pending;
            pending.reserve((size_t)n_tiles);

            for (int t = 0; t < n_tiles; ++t)
            {
                const uint64_t tile_elems =
                    std::min<uint64_t>(tile_cap,
                                       (uint64_t)spatial -
                                       (uint64_t)t * tile_cap);

                uint64_t rd = vop_rd_bytes(VOP_SELECTED_OP, tile_elems);
                uint64_t wr = vop_wr_bytes(VOP_SELECTED_OP, tile_elems);

                auto pm = issue_begin(Interconnect::ADDR_VEC,
                                      VOP_VEC_ACC_CYCLE,
                                      rd, wr,
                                      c, t);
                ++vec_calls;
                total_rd_bytes += rd;
                total_wr_bytes += wr;

                do_scalar(VOP_SCALAR_OVERHEAD);
                pending.push_back(std::move(pm));
            }

            // Drain phase
            for (auto &pm : pending)
                issue_end(pm);
        }

        elapsed_cycles =
            (uint64_t)((sc_time_stamp() - t_start) / CYCLE);
    }
};

// ============================================================
// VecOpsTop — instantiates and wires the full simulator.
//
// Topology:
//   workers[0..N-1]  --> noc --> vec_acc --> memory
//                             \-> mat_acc (dummy, never used)
// ============================================================
struct VecOpsTop : sc_module
{
    AcceleratorTLM  mat_acc;
    AcceleratorPool vec_acc;
    Interconnect    noc;
    Memory          memory;

    std::vector<VecOpsWorker *> workers;

    SC_HAS_PROCESS(VecOpsTop);

    VecOpsTop(sc_module_name name)
        : sc_module(name),
          mat_acc("mat_acc", VOP_ACC_QUEUE_DEPTH),
          vec_acc("vec_acc",
                  (size_t)VOP_VEC_ACC_INSTANCES,
                  VOP_ACC_QUEUE_DEPTH),
          noc("noc"),
          memory("memory",
                 VOP_MEM_BASE_LAT,
                 VOP_MEM_BW,
                 (uint64_t)VOP_VEC_ACC_INSTANCES)
    {
        noc.to_mat.bind(mat_acc.tgt);
        noc.to_vec.bind(vec_acc.tgt);
        noc.to_mem.bind(memory.tgt);

        mat_acc.to_mem.bind(noc.tgt);
        for (auto &unit : vec_acc.units)
            unit->to_mem.bind(noc.tgt);

        for (int i = 0; i < VOP_NUM_WORKERS; ++i)
        {
            auto *w = new VecOpsWorker(
                          sc_gen_unique_name("vec_ops_worker"),
                          i, VOP_NUM_WORKERS);
            workers.push_back(w);
            w->init.bind(noc.tgt);
        }
    }

    ~VecOpsTop() override
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

    VecOpsTop top("vec_ops_top");
    sc_start();

    // ----------------------------------------------------------
    // Derived geometry
    // ----------------------------------------------------------
    const int      spatial = VOP_H * VOP_W;
    const uint64_t tile_cap = vop_tile_cap_elems(VOP_SELECTED_OP);
    const uint64_t extra_rd_per_channel =
        vop_extra_rd_bytes_per_channel(VOP_SELECTED_OP);
    const int      n_tiles = (int)ceil_div_u64((uint64_t)spatial,
                                                tile_cap);
    const uint64_t expected_vec_calls =
        (uint64_t)VOP_C * (uint64_t)n_tiles;

    // Compute expected total rd/wr by summing across all tiles per channel.
    uint64_t per_chan_rd = 0;
    uint64_t per_chan_wr = 0;
    for (int t = 0; t < n_tiles; ++t)
    {
        uint64_t vl = std::min<uint64_t>(
            tile_cap,
            (uint64_t)spatial - (uint64_t)t * tile_cap);
        per_chan_rd += vop_rd_bytes(VOP_SELECTED_OP, vl);
        per_chan_wr += vop_wr_bytes(VOP_SELECTED_OP, vl);
    }
    const uint64_t expected_rd_bytes =
        (uint64_t)VOP_C * (per_chan_rd + extra_rd_per_channel);
    const uint64_t expected_wr_bytes = (uint64_t)VOP_C * per_chan_wr;

    // ----------------------------------------------------------
    // Section 0: Configuration header
    // ----------------------------------------------------------
    std::cout << "\n";
    std::cout << "==============================================\n";
    std::cout << "  Vector Operations TLM Performance Simulation\n";
    std::cout << "  Operation : " << vop_name(VOP_SELECTED_OP) << "\n";
    std::cout << "  Input     : [C=" << VOP_C
              << ", H=" << VOP_H
              << ", W=" << VOP_W << "]"
              << "  (int" << (VOP_ELEM_BYTES * 8) << ")\n";
    std::cout << "  Output    : [C=" << VOP_C
              << ", H=" << VOP_H
              << ", W=" << VOP_W << "]\n";
    std::cout << "  Spatial elements (H*W)   : " << spatial          << "\n";
    std::cout << "  Base elem bytes          : " << VOP_ELEM_BYTES
              << "  (change VOP_ELEM_BYTES in vec_ops_config.h to switch)\n";
    std::cout << "  Workers                  : " << VOP_NUM_WORKERS  << "\n";
    std::cout << "  vec_acc units            : " << VOP_VEC_ACC_INSTANCES << "\n";
    std::cout << "  VEC_ACC_CAP              : " << VOP_VEC_ACC_CAP
              << " e8m4-baseline elements/call,  "
              << VOP_VEC_ACC_CYCLE << " cycles/call\n";
    std::cout << "  Active op tile cap       : " << tile_cap
              << " elements/call\n";
    std::cout << "  Tiles per channel        : " << n_tiles          << "\n";
    std::cout << "  Expected vec_acc calls   : " << expected_vec_calls << "\n";
    std::cout << "\n";
    std::cout << "  Memory traffic model (per tile, vl elements):\n";
    std::cout << "    READ  = vop_rd_bytes(" << vop_name(VOP_SELECTED_OP)
              << ", vl)\n";
    std::cout << "    WRITE = vop_wr_bytes(" << vop_name(VOP_SELECTED_OP)
              << ", vl)\n";
    if (extra_rd_per_channel != 0)
    {
        std::cout << "    Extra scalar read/channel = "
                  << extra_rd_per_channel << " bytes\n";
        std::cout << "    Vector tiles use vec_acc; scalar bias reads go direct to Memory\n";
    }
    else
    {
        std::cout << "    All traffic routed through vec_acc -> Memory\n";
    }
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
        int c_start = (w->tid       * VOP_C) / VOP_NUM_WORKERS;
        int c_end   = ((w->tid + 1) * VOP_C) / VOP_NUM_WORKERS;
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
    std::cout << "Simulation time              : " << sc_time_stamp()     << "\n";
    std::cout << "Max worker elapsed           : " << max_elapsed          << " cycles\n";
    std::cout << "Workers                      : " << VOP_NUM_WORKERS      << "\n";
    std::cout << "Channels (C)                 : " << VOP_C                << "\n";
    std::cout << "Spatial elements (H*W)       : " << spatial              << "\n";
    std::cout << "Vec tiles per channel        : " << n_tiles              << "\n";
    std::cout << "Total vec-acc calls          : " << total_vec
              << "  (expected " << expected_vec_calls << ") "
              << (total_vec == expected_vec_calls ? "[OK]" : "[MISMATCH]") << "\n";
    std::cout << "Total memory reads           : " << total_rd_all
              << " bytes"
              << "  (expected " << expected_rd_bytes << ") "
              << (total_rd_all == expected_rd_bytes ? "[OK]" : "[MISMATCH]") << "\n";
    std::cout << "Total memory writes          : " << total_wr_all
              << " bytes"
              << "  (expected " << expected_wr_bytes << ") "
              << (total_wr_all == expected_wr_bytes ? "[OK]" : "[MISMATCH]") << "\n";
    std::cout << "Total worker wait cycles     : " << total_wait_all       << "\n";
    std::cout << "Total worker mem cycles      : " << total_mem_all        << "\n";

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

    double sim_cycles   = static_cast<double>(sc_time_stamp() / CYCLE);
    double vec_capacity = sim_cycles *
                          static_cast<double>(top.vec_acc.instance_count());
    double vec_util = (vec_capacity > 0.0)
        ? static_cast<double>(top.vec_acc.busy_cycles_total())
              / vec_capacity * 100.0
        : 0.0;
    double vec_occ  = (vec_capacity > 0.0)
        ? static_cast<double>(top.vec_acc.occupied_cycles_total())
              / vec_capacity * 100.0
        : 0.0;

    const uint64_t total_mem_bytes = total_rd_all + total_wr_all;
    double mem_bw = (sim_cycles > 0.0)
        ? static_cast<double>(total_mem_bytes) / sim_cycles
        : 0.0;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "vec_acc compute util         : " << vec_util << "%\n";
    std::cout << "vec_acc occupancy            : " << vec_occ  << "%\n";
    std::cout << "Memory avg BW                : " << mem_bw   << " bytes/cycle\n";

    if (vec_util > 80.0)
        std::cout << "Bottleneck hint              : vec_acc is the primary bottleneck.\n";
    else if (top.memory.busy_cycles > top.vec_acc.busy_cycles_total())
        std::cout << "Bottleneck hint              : memory bandwidth is the primary bottleneck.\n";
    else
        std::cout << "Bottleneck hint              : load is balanced between vec_acc and memory.\n";

    // ----------------------------------------------------------
    // Verification
    // ----------------------------------------------------------
    std::cout << "\n--- Verification ---\n";
    bool ok_calls = (total_vec    == expected_vec_calls);
    bool ok_rd    = (total_rd_all == expected_rd_bytes);
    bool ok_wr    = (total_wr_all == expected_wr_bytes);

    std::cout << (ok_calls ? "  [PASS]" : "  [FAIL]")
              << " Total vec-acc calls == C * ceil(H*W/op_tile_cap)"
              << " (" << total_vec << " == " << expected_vec_calls << ")\n";
    std::cout << (ok_rd ? "  [PASS]" : "  [FAIL]")
              << " Total read bytes"
              << " (" << total_rd_all << " == " << expected_rd_bytes << ")\n";
    std::cout << (ok_wr ? "  [PASS]" : "  [FAIL]")
              << " Total write bytes"
              << " (" << total_wr_all << " == " << expected_wr_bytes << ")\n";

    bool pass = ok_calls && ok_rd && ok_wr;
    std::cout << (pass ? "\nAll checks passed.\n" : "\nSome checks FAILED.\n");
    return pass ? 0 : 2;
}
