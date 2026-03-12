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
#include <fstream>

// Base modules (compiled separately, included via headers)
#include "common.h"
#include "extensions.h"
#include "accelerator.h"
#include "interconnect.h"
#include "memory.h"

// NAFNet layer list + hardware config
#include "nafnet_hw_config.h"
#include "nafnet_layers.h"
#include "waveform.h"
#include "vcd_writer.h"

using namespace sc_core;
using namespace tlm;

// ============================================================
// Barrier — generation-based synchronisation barrier for
// N_WORKERS SC_THREADs.
//
// The generation counter prevents "stale" notifications from
// releasing a thread that has already moved on to the next
// barrier epoch.  The last arriving thread atomically resets
// the counter, increments the generation, and notifies; all
// waiting threads re-check generation before returning.
//
// NOTE: Barrier::sync() must only be called from an SC_THREAD
// (or from a function called within one) because it calls wait().
// ============================================================
struct Barrier
{
    int      n_workers;
    int      arrived    = 0;
    int      generation = 0;
    sc_event released;

    explicit Barrier(int n) : n_workers(n) {}

    void sync()
    {
        int my_gen = generation;
        if (++arrived == n_workers)
        {
            arrived = 0;
            ++generation;
            released.notify(SC_ZERO_TIME);
        }
        else
        {
            // Loop guards against spurious wakeups and reuse of
            // the released event across multiple barrier epochs.
            while (generation == my_gen)
                wait(released);
        }
    }
};

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
    uint64_t mat_reqs     = 0;  // matrix-acc requests
    uint64_t vec_reqs     = 0;  // vector-acc requests
    uint64_t accel_cycles = 0;  // accelerator service cycles (sum of tile cycles)
    uint64_t cpu_cycles   = 0;  // scalar CPU cycles (LAYER_SCALAR only)
    uint64_t wait_cycles  = 0;  // sum of queue-wait cycles
    uint64_t mem_cycles   = 0;  // sum of memory-access cycles
    uint64_t rd_bytes     = 0;
    uint64_t wr_bytes     = 0;
};

// ============================================================
// NafWorker — NAFNet task generator
//
// Each worker owns a full copy of the layer list.
//
// Multi-threaded layers (CONV, DWCONV, multithreaded=true):
//   All N workers synchronise at a barrier, then each takes
//   1/N_WORKERS share of the tile work in parallel.
//
// Single-threaded layers (LAYER_SCALAR, multithreaded=false):
//   Worker 0 executes the full layer on the scalar CPU.
//   Workers 1..N-1 skip the layer immediately and wait at
//   the next multi-threaded barrier.
// ============================================================
struct NafWorker : sc_module
{
    tlm_utils::simple_initiator_socket<NafWorker>        init;
    tlm_utils::peq_with_get<tlm_generic_payload>         peq;

    int      tid;
    int      n_workers;
    Barrier *barrier;  // shared barrier, owned by NafTop

    const std::vector<LayerDesc> &layers; // shared, read-only

    // Per-layer statistics (indexed by LayerDesc::id)
    std::vector<NafLayerStats> layer_stats;

    // Global worker totals
    uint64_t total_dispatch_cycles = 0; // CPU overhead per accelerator tile call (dispatch bookkeeping)
    uint64_t total_cpu_cycles      = 0; // CPU time for LAYER_SCALAR compute (LayerNorm, SimpleGate, etc.)
    uint64_t total_accel_cycles  = 0;  // accelerator service cycles
    uint64_t total_wait_cycles   = 0;
    uint64_t total_mem_cycles    = 0;
    uint64_t mat_calls           = 0;
    uint64_t vec_calls           = 0;
    uint64_t elapsed_cycles      = 0;

    // Synchronisation entry for one in-flight request.
    // fired is set to true by peq_thread before notifying done_ev, so
    // issue_end can skip the wait() if the response already arrived
    // (e.g. when accelerator service time < scalar overhead).
    // admit_ev is notified by nb_transport_bw when the accelerator sends
    // a deferred END_REQ (backpressure case: queue was full at issue time).
    struct DoneEntry
    {
        sc_event *ev       = nullptr;
        sc_event *admit_ev = nullptr;
        bool      fired    = false;
    };

    // In-flight request registry
    std::unordered_map<tlm_generic_payload *, DoneEntry *> done_map;

    // Handle returned by issue_begin; consumed by issue_end.
    struct PendingReq
    {
        tlm_generic_payload *gp          = nullptr;
        ReqExt              *req_ext     = nullptr;
        TxnExt              *tx_ext      = nullptr;
        NafLayerExt         *naf_ext     = nullptr;
        DoneEntry           *done_entry  = nullptr;
        uint64_t             svc_cyc     = 0;
        uint64_t             stall_cycles = 0; // cycles blocked waiting for queue slot
        bool                 sync_done   = false;
    };

    SC_HAS_PROCESS(NafWorker);

    NafWorker(sc_module_name name,
              int tid_,
              int n_workers_,
              const std::vector<LayerDesc> &layers_,
              Barrier *barrier_)
        : sc_module(name),
          init("init"),
          peq("peq"),
          tid(tid_),
          n_workers(n_workers_),
          barrier(barrier_),
          layers(layers_),
          layer_stats(layers_.size())
    {
        init.register_nb_transport_bw(this, &NafWorker::nb_transport_bw);
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
            // Deferred admission: the accelerator had a full queue and is now
            // granting a slot.  Wake up issue_begin which is waiting on admit_ev.
            auto it = done_map.find(&gp);
            if (it != done_map.end() && it->second && it->second->admit_ev)
                it->second->admit_ev->notify(SC_ZERO_TIME);
            return TLM_ACCEPTED;
        }
        return TLM_ACCEPTED;
    }

    // ----------------------------------------------------------
    // peq_thread: wake up done events when responses arrive.
    // Always set fired=true before notifying so issue_end can
    // detect a response that arrived before wait() was called.
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
    // do_scalar: model CPU scalar overhead (cycles run in sim time)
    // ----------------------------------------------------------
    void do_scalar(uint64_t cyc)
    {
        total_dispatch_cycles += cyc;
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

        p.gp         = gp;
        p.req_ext    = req;
        p.tx_ext     = tx;
        p.naf_ext    = naf;
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
            // Queue was full: block here until the accelerator grants a slot
            // (it will send END_REQ backward, which notifies admit_ev).
            std::string wname = "worker_" + std::to_string(tid);
            wave_log((uint64_t)(sc_time_stamp() / CYCLE), wname, 0); // stalled on full queue
            sc_time t_stall_start = sc_time_stamp();
            wait(*p.done_entry->admit_ev);
            p.stall_cycles = (uint64_t)((sc_time_stamp() - t_stall_start) / CYCLE);
            wave_log((uint64_t)(sc_time_stamp() / CYCLE), wname, 1); // queue slot granted
        }
        else if (status == TLM_COMPLETED)
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
        // If the response already arrived (fired=true) before we get
        // here, skip wait() to avoid blocking on a stale event.
        if (!p.sync_done && !p.done_entry->fired)
        {
            std::string wname = "worker_" + std::to_string(tid);
            wave_log((uint64_t)(sc_time_stamp() / CYCLE), wname, 0); // waiting for accelerator
            wait(*p.done_entry->ev);
            wave_log((uint64_t)(sc_time_stamp() / CYCLE), wname, 1); // accelerator done
        }

        done_map.erase(p.gp);
        delete p.done_entry->ev;
        delete p.done_entry->admit_ev;
        delete p.done_entry;
        p.done_entry = nullptr;

        // Read back timing filled in by AcceleratorTLM
        ReqExt *ext = nullptr;
        p.gp->get_extension(ext);
        uint64_t qwait = ext ? ext->accel_qwait_cycles : 0;
        uint64_t mec   = ext ? ext->mem_cycles : 0;

        // Accumulate into global worker totals
        total_accel_cycles += p.svc_cyc;
        total_wait_cycles  += qwait + p.stall_cycles;
        total_mem_cycles   += mec;

        // Accumulate into per-layer stats
        s.wait_cycles += qwait + p.stall_cycles;
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
    // run_accel_layer: execute one CONV or DWCONV layer.
    //   eff_workers: how many workers share this layer's tiles.
    // ----------------------------------------------------------
    void run_accel_layer(const LayerDesc &l, int eff_workers)
    {
        NafLayerStats &s = layer_stats[l.id];

        uint64_t rd_layer = cdiv64(layer_rd_bytes(l),  (uint64_t)eff_workers);
        uint64_t wr_layer = cdiv64(layer_wr_bytes(l),  (uint64_t)eff_workers);

        if (l.type == LAYER_CONV)
        {
            // --- MAT phase: one request per hardware tile ---
            // Only the first tile carries rd_bytes (DMA load into SRAM).
            uint64_t mat_t = cdiv64(conv_mat_tiles(l), (uint64_t)eff_workers);

            for (uint64_t t = 0; t < mat_t; ++t)
            {
                uint64_t rd = (t == 0) ? rd_layer : 0;
                auto pm = issue_begin(Interconnect::ADDR_MAT,
                                      MATMUL_ACC_CYCLE, rd, 0,
                                      l.id, l.type);
                ++mat_calls;
                ++s.mat_reqs;
                s.accel_cycles += MATMUL_ACC_CYCLE;
                s.rd_bytes     += rd;
                do_scalar(SCALAR_OVERHEAD);
                issue_end(pm, s);
            }

            // --- VEC phase: one request per requantize tile ---
            // Only the last tile carries wr_bytes (DMA write-back from SRAM).
            uint64_t vec_t = cdiv64(conv_vec_quant_tiles(l), (uint64_t)eff_workers);

            for (uint64_t t = 0; t < vec_t; ++t)
            {
                uint64_t wr = (t == vec_t - 1) ? wr_layer : 0;
                auto pv = issue_begin(Interconnect::ADDR_VEC,
                                      DWCONV_ACC_CYCLE, 0, wr,
                                      l.id, l.type);
                ++vec_calls;
                ++s.vec_reqs;
                s.accel_cycles += DWCONV_ACC_CYCLE;
                s.wr_bytes     += wr;
                do_scalar(SCALAR_OVERHEAD);
                issue_end(pv, s);
            }
        }
        else // LAYER_DWCONV
        {
            // --- VEC phase: one request per DW-conv tile ---
            // First tile: DMA load (rd_bytes). Last tile: DMA write-back (wr_bytes).
            uint64_t dw_t = cdiv64(dwconv_vec_tiles(l), (uint64_t)eff_workers);

            for (uint64_t t = 0; t < dw_t; ++t)
            {
                uint64_t rd = (t == 0)        ? rd_layer : 0;
                uint64_t wr = (t == dw_t - 1) ? wr_layer : 0;
                auto pv = issue_begin(Interconnect::ADDR_VEC,
                                      DWCONV_ACC_CYCLE, rd, wr,
                                      l.id, l.type);
                ++vec_calls;
                ++s.vec_reqs;
                s.accel_cycles += DWCONV_ACC_CYCLE;
                s.rd_bytes     += rd;
                s.wr_bytes     += wr;
                do_scalar(SCALAR_OVERHEAD);
                issue_end(pv, s);
            }
        }
    }

    // ----------------------------------------------------------
    // run_scalar_layer: execute one LAYER_SCALAR on the CPU.
    //   Only called by worker 0; other workers skip entirely.
    //   Memory traffic is NOT sent as a TLM transaction — the
    //   scalar pipeline accesses SRAM/cache, not DRAM directly.
    //   We model it as pure CPU time only.
    // ----------------------------------------------------------
    void run_scalar_layer(const LayerDesc &l)
    {
        NafLayerStats &s = layer_stats[l.id];
        uint64_t cyc = scalar_cpu_cycles(l);

        total_cpu_cycles += cyc;
        s.cpu_cycles     += cyc;
        s.rd_bytes       += layer_rd_bytes(l);
        s.wr_bytes       += layer_wr_bytes(l);

        wait(cyc * CYCLE);
    }

    // ----------------------------------------------------------
    // run: main worker thread — iterate through all NAFNet layers
    // ----------------------------------------------------------
    void run()
    {
        std::string wname = "worker_" + std::to_string(tid);
        sc_time t_start = sc_time_stamp();
        wave_log((uint64_t)(t_start / CYCLE), wname, 1);  // worker becomes active

        for (const LayerDesc &l : layers)
        {
            if (l.multithreaded)
            {
                // Synchronise all workers before starting this parallel layer.
                // The last arriving worker notifies the rest; all then proceed.
                wave_log((uint64_t)(sc_time_stamp() / CYCLE), wname, 0);  // entering barrier: idle
                barrier->sync();
                wave_log((uint64_t)(sc_time_stamp() / CYCLE), wname, 1);  // leaving barrier: busy
                run_accel_layer(l, n_workers);
            }
            else // LAYER_SCALAR — single-threaded, worker 0 only
            {
                if (tid == 0)
                    run_scalar_layer(l);
                // Workers 1..N-1 skip and fall through to the next iteration.
                // They will wait at the barrier of the next multithreaded layer.
            }
        }

        wave_log((uint64_t)(sc_time_stamp() / CYCLE), wname, 0);  // worker done: idle

        sc_time t_end     = sc_time_stamp();
        elapsed_cycles    = (uint64_t)((t_end - t_start) / CYCLE);
        uint64_t total    = total_dispatch_cycles
                          + total_cpu_cycles
                          + total_accel_cycles
                          + total_wait_cycles
                          + total_mem_cycles;

        std::cout << "[Worker " << tid << "]"
                  << "  mat_calls="  << mat_calls
                  << "  vec_calls="  << vec_calls
                  << "  dispatch="   << total_dispatch_cycles
                  << "  cpu="        << total_cpu_cycles
                  << "  accel="      << total_accel_cycles
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
    Barrier        barrier;

    std::vector<NafWorker *> workers;
    std::vector<LayerDesc>   layers;   // shared layer list

    SC_HAS_PROCESS(NafTop);

    NafTop(sc_module_name name, bool intro_only_ = false)
        : sc_module(name),
          mat_acc("mat_acc", ACC_QUEUE_DEPTH),
          vec_acc("vec_acc", ACC_QUEUE_DEPTH),
          noc("noc"),
          memory("memory", MEM_BASE_LAT, MEM_BW),
          barrier(N_WORKERS)
    {
        // Bind accelerators and memory to the interconnect
        noc.to_mat.bind(mat_acc.tgt);
        noc.to_vec.bind(vec_acc.tgt);
        noc.to_mem.bind(memory.tgt);

        // Accelerators reach memory through the interconnect
        mat_acc.to_mem.bind(noc.tgt);
        vec_acc.to_mem.bind(noc.tgt);

        // Wire busy-state callbacks for waveform logging
        mat_acc.set_busy_callback([](uint64_t t, bool b) {
            wave_log(t, "mat_acc", b ? 1 : 0);
        });
        vec_acc.set_busy_callback([](uint64_t t, bool b) {
            wave_log(t, "vec_acc", b ? 1 : 0);
        });

        // Build the NAFNet layer list (shared across workers)
        layers = build_nafnet32_layers();

        // In intro-only mode keep only the first layer (id=0, "intro" CONV)
        if (intro_only_)
            layers.resize(1);

        // Create and connect workers
        for (int i = 0; i < N_WORKERS; ++i)
        {
            auto *w = new NafWorker(sc_gen_unique_name("naf_worker"),
                                    i, N_WORKERS, layers, &barrier);
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
int sc_main(int argc, char *argv[])
{
    bool intro_only = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--intro-only")
            intro_only = true;
    }

    if (intro_only)
        std::cout << "[Mode] intro-only: simulating only the intro CONV layer\n";

    NafTop top("nafnet_top", intro_only);
    sc_start();

    // ----------------------------------------------------------
    // Export waveform to VCD for GTKWave
    // ----------------------------------------------------------
    {
        auto &evts = wave_events();
        // stable_sort preserves insertion order for same-cycle events,
        // which maintains causal ordering within cooperative SC_THREADs.
        std::stable_sort(evts.begin(), evts.end(),
                         [](const WaveEvent &a, const WaveEvent &b) {
                             return a.cycle < b.cycle;
                         });

        const char *vcd_name = intro_only ? "waveform_intro.vcd" : "waveform.vcd";
        write_vcd(evts, vcd_name);
        std::cout << "VCD waveform written to " << vcd_name
                  << " (" << evts.size() << " events)\n";
    }

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
            global[i].mat_reqs     += ws.mat_reqs;
            global[i].vec_reqs     += ws.vec_reqs;
            global[i].accel_cycles += ws.accel_cycles;
            global[i].cpu_cycles   += ws.cpu_cycles;
            global[i].wait_cycles  += ws.wait_cycles;
            global[i].mem_cycles   += ws.mem_cycles;
            global[i].rd_bytes     += ws.rd_bytes;
            global[i].wr_bytes     += ws.wr_bytes;
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
              << std::setw(8)  << "Type"
              << std::setw(5)  << "Acc"
              << std::setw(9)  << "Reqs"
              << std::setw(12) << "AccelCyc"
              << std::setw(10) << "CpuCyc"
              << std::setw(12) << "WaitCyc"
              << std::setw(12) << "MemCyc"
              << std::setw(12) << "RdBytes"
              << std::setw(12) << "WrBytes"
              << "\n";
    std::cout << std::string(124, '-') << "\n";

    uint64_t scalar_layer_count = 0;
    for (size_t i = 0; i < n_layers; ++i)
    {
        const LayerDesc     &l = layers[i];
        const NafLayerStats &s = global[i];
        uint64_t reqs = s.mat_reqs + s.vec_reqs;

        const char *type_str = (l.type == LAYER_CONV)   ? "CONV"   :
                               (l.type == LAYER_DWCONV) ? "DWCONV" : "SCALAR";
        const char *acc_str  = (l.type == LAYER_CONV)   ? "MAT"    :
                               (l.type == LAYER_DWCONV) ? "VEC"    : "CPU";

        std::cout << std::left
                  << std::setw(4)  << l.id
                  << std::setw(28) << l.name
                  << std::setw(8)  << type_str
                  << std::setw(5)  << acc_str
                  << std::setw(9)  << reqs
                  << std::setw(12) << s.accel_cycles
                  << std::setw(10) << s.cpu_cycles
                  << std::setw(12) << s.wait_cycles
                  << std::setw(12) << s.mem_cycles
                  << std::setw(12) << s.rd_bytes
                  << std::setw(12) << s.wr_bytes
                  << "\n";

        if (l.type == LAYER_SCALAR) ++scalar_layer_count;
    }

    // ----------------------------------------------------------
    // Section 2: Per-worker summary
    // ----------------------------------------------------------
    std::cout << "\n--- Per-Worker Summary ---\n";
    std::cout << std::left
              << std::setw(8)  << "Worker"
              << std::setw(12) << "DispatchCyc"
              << std::setw(10) << "CpuCyc"
              << std::setw(12) << "AccelCyc"
              << std::setw(10) << "WaitCyc"
              << std::setw(10) << "MemCyc"
              << std::setw(10) << "MatReqs"
              << std::setw(10) << "VecReqs"
              << std::setw(12) << "ElapsedCyc"
              << "\n";
    std::cout << std::string(94, '-') << "\n";

    uint64_t max_elapsed = 0;
    for (const auto *w : top.workers)
    {
        std::cout << std::left
                  << std::setw(8)  << w->tid
                  << std::setw(12) << w->total_dispatch_cycles
                  << std::setw(10) << w->total_cpu_cycles
                  << std::setw(12) << w->total_accel_cycles
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
    uint64_t total_cpu_cyc = 0;
    for (const auto &s : global)
    {
        total_mat_reqs += s.mat_reqs;
        total_vec_reqs += s.vec_reqs;
        total_rd       += s.rd_bytes;
        total_wr       += s.wr_bytes;
        total_cpu_cyc  += s.cpu_cycles;
    }

    std::cout << "\n--- Global Summary ---\n";
    std::cout << "Simulation time        : " << sc_time_stamp()    << "\n";
    std::cout << "Max worker elapsed     : " << max_elapsed         << " cycles\n";
    std::cout << "Workers                : " << N_WORKERS           << "\n";
    std::cout << "NAFNet layers total    : " << n_layers            << "\n";
    std::cout << "  Accel layers         : " << (n_layers - scalar_layer_count) << "\n";
    std::cout << "  Scalar layers        : " << scalar_layer_count  << "\n";
    std::cout << "Total mat-acc requests : " << total_mat_reqs      << "\n";
    std::cout << "Total vec-acc requests : " << total_vec_reqs      << "\n";
    std::cout << "Total scalar CPU cyc   : " << total_cpu_cyc       << "\n";
    std::cout << "Total memory reads     : " << total_rd            << " bytes\n";
    std::cout << "Total memory writes    : " << total_wr            << " bytes\n";
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

    // Utilisation = busy_cycles / total_simulation_cycles
    double sim_cycles = static_cast<double>(sc_time_stamp() / CYCLE);
    double mat_util   = (sim_cycles > 0.0)
                      ? static_cast<double>(top.mat_acc.busy_cycles)
                        / sim_cycles * 100.0
                      : 0.0;
    double vec_util   = (sim_cycles > 0.0)
                      ? static_cast<double>(top.vec_acc.busy_cycles)
                        / sim_cycles * 100.0
                      : 0.0;

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
