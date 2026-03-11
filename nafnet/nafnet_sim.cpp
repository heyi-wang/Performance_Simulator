// nafnet_sim.cpp
// NAFNet-32 performance simulator built on the shared SystemC TLM-2.0
// base architecture (AcceleratorTLM, Interconnect, Memory from src/).
//
// Only NAFNet-specific code lives here; all base modules are reused as-is.

// SC_INCLUDE_DYNAMIC_PROCESSES is already passed via -D on the command line.
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
#include <cstdio>

// Base modules (compiled separately, included via headers)
#include "common.h"
#include "extensions.h"
#include "accelerator.h"
#include "interconnect.h"
#include "memory.h"

// NAFNet layer list + hardware config
#include "nafnet_hw_config.h"
#include "nafnet_layers.h"

using namespace sc_core;
using namespace tlm;

// ============================================================
// NafLayerExt — additional TLM extension carrying NAFNet
// layer metadata.  Attached alongside ReqExt and TxnExt.
// AcceleratorTLM / Interconnect / Memory do not read this.
// ============================================================
struct NafLayerExt : tlm_extension<NafLayerExt>
{
    int       layer_id   = -1;
    LayerType layer_type = LAYER_CONV;

    tlm_extension_base *clone() const override
    {
        return new NafLayerExt(*this);
    }
    void copy_from(const tlm_extension_base &ext) override
    {
        *this = static_cast<const NafLayerExt &>(ext);
    }
};

// ============================================================
// NafLayerStats — per-layer cycle and traffic accumulator.
// Each NafWorker holds one array; aggregated by sc_main.
// ============================================================
struct NafLayerStats
{
    uint64_t mat_reqs       = 0;  // matrix-acc requests
    uint64_t vec_reqs       = 0;  // vector-acc requests
    uint64_t compute_cycles = 0;  // sum of accelerator service cycles
    uint64_t wait_cycles    = 0;  // sum of queue-wait cycles
    uint64_t mem_cycles     = 0;  // sum of memory-access cycles
    uint64_t rd_bytes       = 0;
    uint64_t wr_bytes       = 0;
};

// ============================================================
// NafWorker — NAFNet task generator
//
// Each worker owns a full copy of the layer list and processes
// every layer, taking a 1/N_WORKERS share of the spatial work.
//
// Per CONV layer :
//   issue_begin(MAT)  →  do_scalar  →  issue_end
//   issue_begin(VEC)  →  do_scalar  →  issue_end   (requantize)
//
// Per DWCONV layer :
//   issue_begin(VEC)  →  do_scalar  →  issue_end
// ============================================================
struct NafWorker : sc_module
{
    tlm_utils::simple_initiator_socket<NafWorker>        init;
    tlm_utils::peq_with_get<tlm_generic_payload>         peq;

    int      tid;
    int      n_workers;

    const std::vector<LayerDesc> &layers; // shared, read-only

    // Per-layer statistics (indexed by LayerDesc::id)
    std::vector<NafLayerStats> layer_stats;

    // Global worker totals
    uint64_t total_compute_cycles = 0;
    uint64_t total_wait_cycles    = 0;
    uint64_t total_mem_cycles     = 0;
    uint64_t mat_calls            = 0;
    uint64_t vec_calls            = 0;
    uint64_t elapsed_cycles       = 0;

    // In-flight request registry
    std::unordered_map<tlm_generic_payload *, sc_event *> done_map;

    // Handle returned by issue_begin; consumed by issue_end.
    struct PendingReq
    {
        tlm_generic_payload *gp       = nullptr;
        ReqExt              *req_ext  = nullptr;
        TxnExt              *tx_ext   = nullptr;
        NafLayerExt         *naf_ext  = nullptr;
        sc_event            *done_ev  = nullptr;
        uint64_t             svc_cyc  = 0;
        bool                 sync_done = false;
    };

    SC_HAS_PROCESS(NafWorker);

    NafWorker(sc_module_name name,
              int tid_,
              int n_workers_,
              const std::vector<LayerDesc> &layers_)
        : sc_module(name),
          init("init"),
          peq("peq"),
          tid(tid_),
          n_workers(n_workers_),
          layers(layers_),
          layer_stats(layers_.size())
    {
        init.register_nb_transport_bw(this, &NafWorker::nb_transport_bw);
        SC_THREAD(peq_thread);
        SC_THREAD(run);
    }

    // ----------------------------------------------------------
    // nb_transport_bw: receive BEGIN_RESP from downstream
    // ----------------------------------------------------------
    tlm_sync_enum nb_transport_bw(tlm_generic_payload &gp,
                                  tlm_phase           &phase,
                                  sc_time             &delay)
    {
        if (phase == BEGIN_RESP)
            peq.notify(gp, delay);
        return TLM_ACCEPTED;
    }

    // ----------------------------------------------------------
    // peq_thread: wake up done events when responses arrive
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
                    it->second->notify(SC_ZERO_TIME);
            }
        }
    }

    // ----------------------------------------------------------
    // do_scalar: model CPU scalar overhead (cycles run in sim time)
    // ----------------------------------------------------------
    void do_scalar(uint64_t cyc)
    {
        total_compute_cycles += cyc;
        wait(cyc * CYCLE);
    }

    // ----------------------------------------------------------
    // issue_begin: fire a non-blocking accelerator request
    // ----------------------------------------------------------
    PendingReq issue_begin(uint64_t addr,
                           uint64_t svc_cyc,
                           uint64_t rd,
                           uint64_t wr,
                           int      layer_id,
                           LayerType ltype)
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

        auto *naf = new NafLayerExt();
        naf->layer_id   = layer_id;
        naf->layer_type = ltype;

        gp->set_extension(req);
        gp->set_extension(tx);
        gp->set_extension(naf);

        p.gp      = gp;
        p.req_ext = req;
        p.tx_ext  = tx;
        p.naf_ext = naf;
        p.done_ev = new sc_event();
        done_map[gp] = p.done_ev;

        tlm_phase phase = BEGIN_REQ;
        sc_time   delay = SC_ZERO_TIME;
        auto status = init->nb_transport_fw(*gp, phase, delay);

        if (status == TLM_COMPLETED)
        {
            done_map.erase(gp);
            p.sync_done = true;
        }
        return p;
    }

    // ----------------------------------------------------------
    // issue_end: wait for completion, harvest stats, send END_RESP
    // ----------------------------------------------------------
    void issue_end(PendingReq &p, NafLayerStats &s)
    {
        if (!p.sync_done)
            wait(*p.done_ev);

        done_map.erase(p.gp);
        delete p.done_ev;
        p.done_ev = nullptr;

        // Read back timing filled in by AcceleratorTLM
        ReqExt *ext = nullptr;
        p.gp->get_extension(ext);
        uint64_t qwait = ext ? ext->accel_qwait_cycles : 0;
        uint64_t mec   = ext ? ext->mem_cycles : 0;

        // Accumulate into global worker totals
        total_compute_cycles += p.svc_cyc;
        total_wait_cycles    += qwait;
        total_mem_cycles     += mec;

        // Accumulate into per-layer stats
        s.wait_cycles += qwait;
        s.mem_cycles  += mec;

        // Acknowledge END_RESP to the interconnect
        tlm_phase end_phase = END_RESP;
        sc_time   end_delay = SC_ZERO_TIME;
        init->nb_transport_fw(*p.gp, end_phase, end_delay);

        // Clean up
        p.gp->clear_extension(p.req_ext);
        p.gp->clear_extension(p.tx_ext);
        p.gp->clear_extension(p.naf_ext);
        delete p.req_ext; p.req_ext = nullptr;
        delete p.tx_ext;  p.tx_ext  = nullptr;
        delete p.naf_ext; p.naf_ext = nullptr;
        delete p.gp;      p.gp      = nullptr;
    }

    // ----------------------------------------------------------
    // run: main worker thread — iterate through all NAFNet layers
    // ----------------------------------------------------------
    void run()
    {
        sc_time t_start = sc_time_stamp();

        for (const LayerDesc &l : layers)
        {
            NafLayerStats &s = layer_stats[l.id];

            if (l.type == LAYER_CONV)
            {
                // Each worker handles 1/n_workers of the spatial work.
                uint64_t mat_cyc = cdiv64(conv_mat_cycles(l),
                                          (uint64_t)n_workers);
                uint64_t vec_cyc = cdiv64(conv_vec_quant_cycles(l),
                                          (uint64_t)n_workers);
                uint64_t rd      = cdiv64(layer_rd_bytes(l),
                                          (uint64_t)n_workers);
                uint64_t wr      = cdiv64(layer_wr_bytes(l),
                                          (uint64_t)n_workers);

                // --- Matrix accelerator request ---
                auto pm = issue_begin(Interconnect::ADDR_MAT,
                                      mat_cyc, rd, 0,
                                      l.id, l.type);
                mat_calls++;
                s.mat_reqs++;
                s.compute_cycles += mat_cyc;
                s.rd_bytes       += rd;

                do_scalar(SCALAR_OVERHEAD);
                issue_end(pm, s);

                // --- Vector accelerator request (requantize) ---
                auto pv = issue_begin(Interconnect::ADDR_VEC,
                                      vec_cyc, 0, wr,
                                      l.id, l.type);
                vec_calls++;
                s.vec_reqs++;
                s.compute_cycles += vec_cyc;
                s.wr_bytes       += wr;

                do_scalar(SCALAR_OVERHEAD);
                issue_end(pv, s);
            }
            else // LAYER_DWCONV
            {
                uint64_t vec_cyc = cdiv64(dwconv_vec_cycles(l),
                                          (uint64_t)n_workers);
                uint64_t rd      = cdiv64(layer_rd_bytes(l),
                                          (uint64_t)n_workers);
                uint64_t wr      = cdiv64(layer_wr_bytes(l),
                                          (uint64_t)n_workers);

                auto pv = issue_begin(Interconnect::ADDR_VEC,
                                      vec_cyc, rd, wr,
                                      l.id, l.type);
                vec_calls++;
                s.vec_reqs++;
                s.compute_cycles += vec_cyc;
                s.rd_bytes       += rd;
                s.wr_bytes       += wr;

                do_scalar(SCALAR_OVERHEAD);
                issue_end(pv, s);
            }
        }

        sc_time t_end     = sc_time_stamp();
        elapsed_cycles    = (uint64_t)((t_end - t_start) / CYCLE);
        uint64_t total    = total_compute_cycles
                          + total_wait_cycles
                          + total_mem_cycles;

        std::cout << "[Worker " << tid << "]"
                  << "  mat_calls="  << mat_calls
                  << "  vec_calls="  << vec_calls
                  << "  compute="    << total_compute_cycles
                  << "  wait="       << total_wait_cycles
                  << "  mem="        << total_mem_cycles
                  << "  total="      << total
                  << "  elapsed="    << elapsed_cycles
                  << "\n";
    }
};

// ============================================================
// NafTop — instantiates and wires the full NAFNet simulator
// ============================================================
struct NafTop : sc_module
{
    AcceleratorTLM mat_acc;
    AcceleratorTLM vec_acc;
    Interconnect   noc;
    Memory         memory;

    std::vector<NafWorker *> workers;
    std::vector<LayerDesc>   layers;   // shared layer list

    SC_CTOR(NafTop)
        : mat_acc("mat_acc"),
          vec_acc("vec_acc"),
          noc("noc"),
          memory("memory", MEM_BASE_LAT, MEM_BW)
    {
        // Bind accelerators and memory to the interconnect
        noc.to_mat.bind(mat_acc.tgt);
        noc.to_vec.bind(vec_acc.tgt);
        noc.to_mem.bind(memory.tgt);

        // Accelerators reach memory through the interconnect
        mat_acc.to_mem.bind(noc.tgt);
        vec_acc.to_mem.bind(noc.tgt);

        // Build the NAFNet layer list (shared across workers)
        layers = build_nafnet32_layers();

        // Create and connect workers
        for (int i = 0; i < N_WORKERS; ++i)
        {
            auto *w = new NafWorker(sc_gen_unique_name("naf_worker"),
                                    i, N_WORKERS, layers);
            workers.push_back(w);
            w->init.bind(noc.tgt);
        }
    }

    ~NafTop() override
    {
        for (auto *w : workers)
            delete w;
    }
};

// ============================================================
// sc_main — run simulation and print three-section report
// ============================================================
int sc_main(int /*argc*/, char * /*argv*/[])
{
    NafTop top("nafnet_top");
    sc_start();

    const auto  &layers   = top.layers;
    const size_t n_layers = layers.size();

    // ----------------------------------------------------------
    // Aggregate per-layer stats across all workers
    // ----------------------------------------------------------
    std::vector<NafLayerStats> global(n_layers);

    for (const auto *w : top.workers)
    {
        for (size_t i = 0; i < n_layers; ++i)
        {
            const NafLayerStats &ws = w->layer_stats[i];
            global[i].mat_reqs       += ws.mat_reqs;
            global[i].vec_reqs       += ws.vec_reqs;
            global[i].compute_cycles += ws.compute_cycles;
            global[i].wait_cycles    += ws.wait_cycles;
            global[i].mem_cycles     += ws.mem_cycles;
            global[i].rd_bytes       += ws.rd_bytes;
            global[i].wr_bytes       += ws.wr_bytes;
        }
    }

    // ----------------------------------------------------------
    // Report header
    // ----------------------------------------------------------
    std::cout << "\n";
    std::cout << "=========================================="
                 "==============================\n";
    std::cout << "  NAFNet-32 Performance Simulation Report"
                 "  (64x64 input, "
              << N_WORKERS << " workers)\n";
    std::cout << "=========================================="
                 "==============================\n";

    // ----------------------------------------------------------
    // Section 1: Per-layer summary
    // ----------------------------------------------------------
    std::cout << "\n--- Per-Layer Summary ---\n";
    std::cout << std::left
              << std::setw(4)  << "ID"
              << std::setw(28) << "Name"
              << std::setw(7)  << "Type"
              << std::setw(5)  << "Acc"
              << std::setw(7)  << "Reqs"
              << std::setw(12) << "ComputeCyc"
              << std::setw(12) << "WaitCyc"
              << std::setw(12) << "MemCyc"
              << std::setw(12) << "RdBytes"
              << std::setw(12) << "WrBytes"
              << "\n";
    std::cout << std::string(111, '-') << "\n";

    for (size_t i = 0; i < n_layers; ++i)
    {
        const LayerDesc    &l = layers[i];
        const NafLayerStats &s = global[i];
        uint64_t reqs = s.mat_reqs + s.vec_reqs;

        std::cout << std::left
                  << std::setw(4)  << l.id
                  << std::setw(28) << l.name
                  << std::setw(7)  << (l.type == LAYER_CONV ? "CONV" : "DWCONV")
                  << std::setw(5)  << (l.type == LAYER_CONV ? "MAT" : "VEC")
                  << std::setw(7)  << reqs
                  << std::setw(12) << s.compute_cycles
                  << std::setw(12) << s.wait_cycles
                  << std::setw(12) << s.mem_cycles
                  << std::setw(12) << s.rd_bytes
                  << std::setw(12) << s.wr_bytes
                  << "\n";
    }

    // ----------------------------------------------------------
    // Section 2: Per-worker summary
    // ----------------------------------------------------------
    std::cout << "\n--- Per-Worker Summary ---\n";
    std::cout << std::left
              << std::setw(8)  << "Worker"
              << std::setw(12) << "ComputeCyc"
              << std::setw(10) << "WaitCyc"
              << std::setw(10) << "MemCyc"
              << std::setw(10) << "MatReqs"
              << std::setw(10) << "VecReqs"
              << std::setw(12) << "ElapsedCyc"
              << "\n";
    std::cout << std::string(72, '-') << "\n";

    uint64_t max_elapsed = 0;
    for (const auto *w : top.workers)
    {
        std::cout << std::left
                  << std::setw(8)  << w->tid
                  << std::setw(12) << w->total_compute_cycles
                  << std::setw(10) << w->total_wait_cycles
                  << std::setw(10) << w->total_mem_cycles
                  << std::setw(10) << w->mat_calls
                  << std::setw(10) << w->vec_calls
                  << std::setw(12) << w->elapsed_cycles
                  << "\n";
        max_elapsed = std::max(max_elapsed, w->elapsed_cycles);
    }

    // ----------------------------------------------------------
    // Section 3: Global summary
    // ----------------------------------------------------------
    uint64_t total_mat_reqs = 0, total_vec_reqs = 0;
    uint64_t total_rd = 0, total_wr = 0;
    for (const auto &s : global)
    {
        total_mat_reqs += s.mat_reqs;
        total_vec_reqs += s.vec_reqs;
        total_rd       += s.rd_bytes;
        total_wr       += s.wr_bytes;
    }

    std::cout << "\n--- Global Summary ---\n";
    std::cout << "Simulation time        : " << sc_time_stamp() << "\n";
    std::cout << "Max worker elapsed     : " << max_elapsed      << " cycles\n";
    std::cout << "Workers                : " << N_WORKERS         << "\n";
    std::cout << "NAFNet layers modelled : " << n_layers          << "\n";
    std::cout << "Total mat-acc requests : " << total_mat_reqs    << "\n";
    std::cout << "Total vec-acc requests : " << total_vec_reqs    << "\n";
    std::cout << "Total memory reads     : " << total_rd          << " bytes\n";
    std::cout << "Total memory writes    : " << total_wr          << " bytes\n";
    std::cout << "\n";
    std::cout << "mat_acc : reqs="     << top.mat_acc.req_count
              << "  busy="             << top.mat_acc.busy_cycles
              << "  qwait="            << top.mat_acc.queue_wait_cycles
              << "\n";
    std::cout << "vec_acc : reqs="     << top.vec_acc.req_count
              << "  busy="             << top.vec_acc.busy_cycles
              << "  qwait="            << top.vec_acc.queue_wait_cycles
              << "\n";
    std::cout << "memory  : reqs="     << top.memory.reqs
              << "  busy="             << top.memory.busy_cycles
              << "  qwait="            << top.memory.qwait_cycles
              << "\n";

    // mat_acc utilisation
    double mat_busy  = static_cast<double>(top.mat_acc.busy_cycles);
    double mat_total = mat_busy
                     + static_cast<double>(top.mat_acc.queue_wait_cycles);
    double mat_util  = (mat_total > 0.0) ? mat_busy / mat_total * 100.0 : 0.0;

    double vec_busy  = static_cast<double>(top.vec_acc.busy_cycles);
    double vec_total = vec_busy
                     + static_cast<double>(top.vec_acc.queue_wait_cycles);
    double vec_util  = (vec_total > 0.0) ? vec_busy / vec_total * 100.0 : 0.0;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "\nmat_acc utilisation    : " << mat_util << "%\n";
    std::cout << "vec_acc utilisation    : " << vec_util  << "%\n";

    if (mat_util > vec_util + 10.0)
        std::cout << "Bottleneck hint        : matrix accelerator is the "
                     "primary bottleneck.\n";
    else if (vec_util > mat_util + 10.0)
        std::cout << "Bottleneck hint        : vector accelerator is the "
                     "primary bottleneck.\n";
    else
        std::cout << "Bottleneck hint        : load is balanced between "
                     "mat and vec accelerators.\n";

    return 0;
}
