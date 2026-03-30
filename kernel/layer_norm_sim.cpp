// layer_norm_sim.cpp
// SystemC TLM-2.0 performance simulator for LayerNorm2d (int16) in NAFNet.
//
// Maps mf_layernorm2d_i16 (kernel/layer_norm.h) to the shared src/ hardware:
//   Step 1 — mean reduction   : ceil_div(H*W, LN_VEC_ACC_CAP) vec_acc requests
//   Step 2 — variance         : ceil_div(H*W, LN_VEC_ACC_CAP) vec_acc requests
//   Step 3 — inv_std_fp       : scalar CPU stall (isqrt + divide)
//   Step 4 — normalize+scale  : ceil_div(H*W, LN_VEC_ACC_CAP) vec_acc requests
//
// Workers are partitioned by channel: worker tid owns channels [c_start, c_end).
// Communication and synchronisation are consistent with nafnet_sim.cpp:
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

#include "layer_norm_config.h"

using namespace sc_core;
using namespace tlm;

// ============================================================
// LayerNormExt — TLM extension carrying layer norm metadata.
// Attached alongside ReqExt and TxnExt on each transaction.
// AcceleratorTLM / Interconnect / Memory do not inspect this.
// ============================================================
struct LayerNormExt : tlm_extension<LayerNormExt>
{
    int channel_id = -1;  // channel this request belongs to
    int step       =  0;  // 1=mean, 2=variance, 4=normalize

    tlm_extension_base *clone() const override
    {
        return new LayerNormExt(*this);
    }

    void copy_from(const tlm_extension_base &other) override
    {
        *this = static_cast<const LayerNormExt &>(other);
    }
};

// ============================================================
// LnStepStats — per-step cycle and traffic accumulator.
// Array index: 0=Step1, 1=Step2, 2=Step3, 3=Step4.
// ============================================================
struct LnStepStats
{
    uint64_t vec_reqs      = 0;  // number of vec_acc requests issued
    uint64_t accel_cycles  = 0;  // sum of service cycles requested
    uint64_t scalar_cycles = 0;  // CPU cycles (Step 3 only)
    uint64_t wait_cycles   = 0;  // queue-wait + back-pressure stall
    uint64_t mem_cycles    = 0;  // memory service cycles
    uint64_t rd_bytes      = 0;  // memory bytes read
    uint64_t wr_bytes      = 0;  // memory bytes written
};

// ============================================================
// LayerNormWorker — per-thread task generator.
//
// Channel assignment:
//   c_start = (tid   * LN_C) / n_workers
//   c_end   = ((tid+1) * LN_C) / n_workers
//
// For every channel, the worker issues requests in order:
//   Step 1: n_tiles vec_acc requests to compute the mean
//   Step 2: n_tiles vec_acc requests to compute the variance
//   Step 3: scalar stall for the isqrt / fixed-point reciprocal
//   Step 4: n_tiles vec_acc requests to normalise + scale + bias
//
// where n_tiles = ceil_div(H*W, LN_VEC_ACC_CAP).
//
// The fire-then-drain pattern (issue all tiles, then collect
// results) is used to allow multiple in-flight requests,
// matching the nafnet_sim.cpp / matmul_sim approach.
// ============================================================
struct LayerNormWorker : sc_module
{
    tlm_utils::simple_initiator_socket<LayerNormWorker>  init;
    tlm_utils::peq_with_get<tlm_generic_payload>         peq;

    int tid;
    int n_workers;

    // Per-step statistics (index 0..3 = Steps 1..4)
    LnStepStats step_stats[4];

    // Global worker totals
    uint64_t total_dispatch_cycles = 0;  // scalar overhead per tile dispatch
    uint64_t total_scalar_cycles   = 0;  // Step 3 CPU compute time
    uint64_t total_wait_cycles     = 0;  // queue-wait + stall (all steps)
    uint64_t total_mem_cycles      = 0;  // memory service (all steps)
    uint64_t vec_calls             = 0;  // total vec_acc requests issued
    uint64_t elapsed_cycles        = 0;  // wall-clock cycles from start to finish

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
        LayerNormExt        *ln_ext       = nullptr;
        DoneEntry           *done_entry   = nullptr;
        uint64_t             svc_cyc      = 0;
        uint64_t             stall_cycles = 0;   // back-pressure stall for this request
        bool                 sync_done    = false;
    };

    SC_HAS_PROCESS(LayerNormWorker);

    LayerNormWorker(sc_module_name name, int tid_, int n_workers_)
        : sc_module(name),
          init("init"),
          peq("peq"),
          tid(tid_),
          n_workers(n_workers_)
    {
        init.register_nb_transport_bw(this, &LayerNormWorker::nb_transport_bw);
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
            // Deferred admission: queue was full; accelerator now grants a slot.
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
    // Accumulates into total_dispatch_cycles.
    // ----------------------------------------------------------
    void do_scalar(uint64_t cyc)
    {
        total_dispatch_cycles += cyc;
        wait(cyc * CYCLE);
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
                           int      step)
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

        auto *ln  = new LayerNormExt();
        ln->channel_id = channel_id;
        ln->step       = step;

        gp->set_extension(req);
        gp->set_extension(tx);
        gp->set_extension(ln);

        p.gp         = gp;
        p.req_ext    = req;
        p.tx_ext     = tx;
        p.ln_ext     = ln;
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
            // Queue was full: block until the accelerator grants a slot.
            sc_time t_stall_start = sc_time_stamp();
            wait(*p.done_entry->admit_ev);
            p.stall_cycles = (uint64_t)((sc_time_stamp() - t_stall_start) / CYCLE);
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
    void issue_end(PendingReq &p, LnStepStats &s)
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

        // Accumulate into per-step and global worker totals.
        s.wait_cycles     += qwait + p.stall_cycles;
        s.mem_cycles      += mec;
        total_wait_cycles += qwait + p.stall_cycles;
        total_mem_cycles  += mec;

        // Acknowledge response back to the interconnect.
        tlm_phase end_phase = END_RESP;
        sc_time   end_delay = SC_ZERO_TIME;
        init->nb_transport_fw(*p.gp, end_phase, end_delay);

        // Release all extensions and the payload.
        p.gp->clear_extension(p.req_ext);
        p.gp->clear_extension(p.tx_ext);
        p.gp->clear_extension(p.ln_ext);
        delete p.req_ext; p.req_ext = nullptr;
        delete p.tx_ext;  p.tx_ext  = nullptr;
        delete p.ln_ext;  p.ln_ext  = nullptr;
        delete p.gp;      p.gp      = nullptr;
    }

    // ----------------------------------------------------------
    // run: main worker thread — process assigned channel slice.
    // ----------------------------------------------------------
    void run()
    {
        sc_time t_start = sc_time_stamp();

        const int      spatial   = LN_H * LN_W;
        const int      n_tiles   = (int)ceil_div_u64((uint64_t)spatial,
                                                      LN_VEC_ACC_CAP);
        const uint64_t elem_bytes = 2;                      // int16 bytes

        // Channel range for this worker (integer division gives even split).
        int c_start = (tid       * LN_C) / n_workers;
        int c_end   = ((tid + 1) * LN_C) / n_workers;

        for (int c = c_start; c < c_end; ++c)
        {
            // ================================================
            // Step 1: compute mean via sum reduction.
            //   Each tile independently reads the input slice it reduces.
            // ================================================
            {
                std::vector<PendingReq> pending;
                pending.reserve((size_t)n_tiles);

                for (int t = 0; t < n_tiles; ++t)
                {
                    const uint64_t tile_elems =
                        std::min<uint64_t>(LN_VEC_ACC_CAP,
                                           (uint64_t)spatial - (uint64_t)t * LN_VEC_ACC_CAP);
                    uint64_t rd = tile_elems * elem_bytes;
                    auto pm = issue_begin(Interconnect::ADDR_VEC,
                                         LN_VEC_ACC_CYCLE, rd, 0, c, 1);
                    ++vec_calls;
                    ++step_stats[0].vec_reqs;
                    step_stats[0].accel_cycles += LN_VEC_ACC_CYCLE;
                    step_stats[0].rd_bytes     += rd;
                    do_scalar(LN_SCALAR_OVERHEAD);
                    pending.push_back(std::move(pm));
                }
                for (auto &pm : pending)
                    issue_end(pm, step_stats[0]);
            }

            // ================================================
            // Step 2: compute variance via sum-of-squares.
            //   Each tile independently re-reads the input slice it reduces.
            // ================================================
            {
                std::vector<PendingReq> pending;
                pending.reserve((size_t)n_tiles);

                for (int t = 0; t < n_tiles; ++t)
                {
                    const uint64_t tile_elems =
                        std::min<uint64_t>(LN_VEC_ACC_CAP,
                                           (uint64_t)spatial - (uint64_t)t * LN_VEC_ACC_CAP);
                    uint64_t rd = tile_elems * elem_bytes;
                    auto pm = issue_begin(Interconnect::ADDR_VEC,
                                         LN_VEC_ACC_CYCLE, rd, 0, c, 2);
                    ++vec_calls;
                    ++step_stats[1].vec_reqs;
                    step_stats[1].accel_cycles += LN_VEC_ACC_CYCLE;
                    step_stats[1].rd_bytes     += rd;
                    do_scalar(LN_SCALAR_OVERHEAD);
                    pending.push_back(std::move(pm));
                }
                for (auto &pm : pending)
                    issue_end(pm, step_stats[1]);
            }

            // ================================================
            // Step 3: compute inv_std_fp — pure scalar.
            //   isqrt(var) + fixed-point reciprocal.
            //   No accelerator request; models the CPU pipeline.
            // ================================================
            {
                step_stats[2].scalar_cycles += LN_STEP3_CYCLES;
                total_scalar_cycles         += LN_STEP3_CYCLES;
                wait(LN_STEP3_CYCLES * CYCLE);
            }

            // ================================================
            // Step 4: normalise each element with gamma/beta.
            //   Each tile independently reads its input slice plus gamma/beta,
            //   then writes its output slice back to memory.
            // ================================================
            {
                std::vector<PendingReq> pending;
                pending.reserve((size_t)n_tiles);

                for (int t = 0; t < n_tiles; ++t)
                {
                    const uint64_t tile_elems =
                        std::min<uint64_t>(LN_VEC_ACC_CAP,
                                           (uint64_t)spatial - (uint64_t)t * LN_VEC_ACC_CAP);
                    // gamma[c] and beta[c] are each one int16 element.
                    uint64_t rd = tile_elems * elem_bytes + elem_bytes + elem_bytes;
                    uint64_t wr = tile_elems * elem_bytes;
                    auto pm = issue_begin(Interconnect::ADDR_VEC,
                                         LN_VEC_ACC_CYCLE, rd, wr, c, 4);
                    ++vec_calls;
                    ++step_stats[3].vec_reqs;
                    step_stats[3].accel_cycles += LN_VEC_ACC_CYCLE;
                    step_stats[3].rd_bytes     += rd;
                    step_stats[3].wr_bytes     += wr;
                    do_scalar(LN_SCALAR_OVERHEAD);
                    pending.push_back(std::move(pm));
                }
                for (auto &pm : pending)
                    issue_end(pm, step_stats[3]);
            }
        }

        elapsed_cycles = (uint64_t)((sc_time_stamp() - t_start) / CYCLE);
    }
};

// ============================================================
// LayerNormTop — instantiates and wires the full simulator.
//
// Topology:
//   workers[0..N-1]  ──► noc ──► vec_acc ──► memory
//                             └──► mat_acc (dummy, never used)
//
// mat_acc is created only to satisfy the noc.to_mat binding
// requirement.  No worker ever sends to ADDR_MAT.
// ============================================================
struct LayerNormTop : sc_module
{
    AcceleratorTLM  mat_acc;   // dummy: bound to satisfy noc.to_mat
    AcceleratorPool vec_acc;   // actual: pool of LN_VEC_ACC_INSTANCES units
    Interconnect    noc;
    Memory          memory;

    std::vector<LayerNormWorker *> workers;

    SC_HAS_PROCESS(LayerNormTop);

    LayerNormTop(sc_module_name name)
        : sc_module(name),
          mat_acc("mat_acc", LN_ACC_QUEUE_DEPTH),
          vec_acc("vec_acc",
                  (size_t)LN_VEC_ACC_INSTANCES,
                  LN_ACC_QUEUE_DEPTH),
          noc("noc"),
          memory("memory",
                 LN_MEM_BASE_LAT,
                 LN_MEM_BW,
                 (uint64_t)LN_VEC_ACC_INSTANCES)
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
        for (int i = 0; i < LN_NUM_WORKERS; ++i)
        {
            auto *w = new LayerNormWorker(
                          sc_gen_unique_name("ln_worker"),
                          i, LN_NUM_WORKERS);
            workers.push_back(w);
            w->init.bind(noc.tgt);
        }
    }

    ~LayerNormTop() override
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

    LayerNormTop top("ln_top");
    sc_start();

    // ----------------------------------------------------------
    // Aggregate per-step stats across all workers.
    // ----------------------------------------------------------
    LnStepStats global[4];
    for (const auto *w : top.workers)
    {
        for (int s = 0; s < 4; ++s)
        {
            global[s].vec_reqs      += w->step_stats[s].vec_reqs;
            global[s].accel_cycles  += w->step_stats[s].accel_cycles;
            global[s].scalar_cycles += w->step_stats[s].scalar_cycles;
            global[s].wait_cycles   += w->step_stats[s].wait_cycles;
            global[s].mem_cycles    += w->step_stats[s].mem_cycles;
            global[s].rd_bytes      += w->step_stats[s].rd_bytes;
            global[s].wr_bytes      += w->step_stats[s].wr_bytes;
        }
    }

    // ----------------------------------------------------------
    // Report header
    // ----------------------------------------------------------
    std::cout << "\n";
    std::cout << "==============================================\n";
    std::cout << "  LayerNorm2d TLM Performance Simulation\n";
    std::cout << "  Input : [C=" << LN_C
              << ", H=" << LN_H
              << ", W=" << LN_W << "]  int16\n";
    std::cout << "  Workers        : " << LN_NUM_WORKERS        << "\n";
    std::cout << "  vec_acc units  : " << LN_VEC_ACC_INSTANCES  << "\n";
    std::cout << "  VEC_ACC_CAP    : " << LN_VEC_ACC_CAP
              << " elements/call,  " << LN_VEC_ACC_CYCLE << " cycles/call\n";
    std::cout << "==============================================\n";

    // ----------------------------------------------------------
    // Section 1: Per-step summary (all workers combined)
    // ----------------------------------------------------------
    const char *step_names[4] = {
        "Step1(mean)     ",
        "Step2(variance) ",
        "Step3(inv_std)  ",
        "Step4(normalize)"
    };

    std::cout << "\n--- Per-Step Summary (all workers combined) ---\n";
    std::cout << std::left
              << std::setw(18) << "Step"
              << std::setw(10) << "VecReqs"
              << std::setw(12) << "AccelCyc"
              << std::setw(12) << "ScalarCyc"
              << std::setw(12) << "WaitCyc"
              << std::setw(10) << "MemCyc"
              << std::setw(12) << "RdBytes"
              << std::setw(12) << "WrBytes"
              << "\n";
    std::cout << std::string(98, '-') << "\n";

    for (int s = 0; s < 4; ++s)
    {
        std::cout << std::left
                  << std::setw(18) << step_names[s]
                  << std::setw(10) << global[s].vec_reqs
                  << std::setw(12) << global[s].accel_cycles
                  << std::setw(12) << global[s].scalar_cycles
                  << std::setw(12) << global[s].wait_cycles
                  << std::setw(10) << global[s].mem_cycles
                  << std::setw(12) << global[s].rd_bytes
                  << std::setw(12) << global[s].wr_bytes
                  << "\n";
    }

    // ----------------------------------------------------------
    // Section 2: Per-worker summary
    // ----------------------------------------------------------
    std::cout << "\n--- Per-Worker Summary ---\n";
    std::cout << std::left
              << std::setw(8)  << "Worker"
              << std::setw(12) << "ChanRange"
              << std::setw(10) << "VecCalls"
              << std::setw(14) << "DispatchCyc"
              << std::setw(12) << "ScalarCyc"
              << std::setw(12) << "WaitCyc"
              << std::setw(10) << "MemCyc"
              << std::setw(12) << "ElapsedCyc"
              << "\n";
    std::cout << std::string(90, '-') << "\n";

    uint64_t max_elapsed = 0;
    for (const auto *w : top.workers)
    {
        int c_start = (w->tid       * LN_C) / LN_NUM_WORKERS;
        int c_end   = ((w->tid + 1) * LN_C) / LN_NUM_WORKERS;
        std::string range = "[" + std::to_string(c_start)
                          + ","  + std::to_string(c_end)  + ")";
        std::cout << std::left
                  << std::setw(8)  << w->tid
                  << std::setw(12) << range
                  << std::setw(10) << w->vec_calls
                  << std::setw(14) << w->total_dispatch_cycles
                  << std::setw(12) << w->total_scalar_cycles
                  << std::setw(12) << w->total_wait_cycles
                  << std::setw(10) << w->total_mem_cycles
                  << std::setw(12) << w->elapsed_cycles
                  << "\n";
        max_elapsed = std::max(max_elapsed, w->elapsed_cycles);
    }

    // ----------------------------------------------------------
    // Section 3: Global summary
    // ----------------------------------------------------------
    uint64_t total_vec_reqs = 0;
    uint64_t total_rd = 0, total_wr = 0;
    for (int s = 0; s < 4; ++s)
    {
        total_vec_reqs += global[s].vec_reqs;
        total_rd       += global[s].rd_bytes;
        total_wr       += global[s].wr_bytes;
    }

    const int     spatial = LN_H * LN_W;
    const int     n_tiles = (int)ceil_div_u64((uint64_t)spatial,
                                              LN_VEC_ACC_CAP);
    // Expected total requests: C channels × 3 step phases × n_tiles
    const uint64_t expected_reqs = (uint64_t)LN_C * 3 * (uint64_t)n_tiles;

    std::cout << "\n--- Global Summary ---\n";
    std::cout << "Simulation time           : " << sc_time_stamp()  << "\n";
    std::cout << "Max worker elapsed        : " << max_elapsed      << " cycles\n";
    std::cout << "Workers                   : " << LN_NUM_WORKERS   << "\n";
    std::cout << "Channels (C)              : " << LN_C             << "\n";
    std::cout << "Spatial elements (H*W)    : " << spatial          << "\n";
    std::cout << "Vec tiles per step/chan    : " << n_tiles          << "\n";
    std::cout << "Total vec-acc requests    : " << total_vec_reqs
              << "  (expected " << expected_reqs << ")\n";
    std::cout << "Total memory reads        : " << total_rd         << " bytes\n";
    std::cout << "Total memory writes       : " << total_wr         << " bytes\n";

    std::cout << "\n";
    std::cout << "vec_acc pool  : units="  << top.vec_acc.instance_count()
              << "  reqs="                 << top.vec_acc.req_count_total()
              << "  busy_cyc="             << top.vec_acc.busy_cycles_total()
              << "  qwait_cyc="            << top.vec_acc.queue_wait_cycles_total()
              << "\n";
    std::cout << "memory        : reqs="   << top.memory.reqs
              << "  busy_cyc="             << top.memory.busy_cycles
              << "  qwait_cyc="            << top.memory.qwait_cycles
              << "\n";

    // Utilisation: busy cycles / (sim_cycles × num_units)
    double sim_cycles = static_cast<double>(sc_time_stamp() / CYCLE);
    double vec_capacity = sim_cycles * static_cast<double>(top.vec_acc.instance_count());
    double vec_util = (vec_capacity > 0.0)
        ? static_cast<double>(top.vec_acc.busy_cycles_total()) / vec_capacity * 100.0
        : 0.0;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "vec_acc utilisation       : " << vec_util << "%\n";

    if (vec_util > 80.0)
        std::cout << "Bottleneck hint           : vec_acc is the primary bottleneck.\n";
    else if (top.memory.busy_cycles > top.vec_acc.busy_cycles_total())
        std::cout << "Bottleneck hint           : memory bandwidth is the primary bottleneck.\n";
    else
        std::cout << "Bottleneck hint           : load is balanced between vec_acc and memory.\n";

    return 0;
}
