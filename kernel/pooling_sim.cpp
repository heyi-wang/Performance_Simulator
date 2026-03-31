// pooling_sim.cpp
// SystemC TLM-2.0 performance simulator for Global Average Pooling in NAFNet.
//
// Maps mf_global_avgpool_i16 (kernel/pooling.h) to the shared src/ hardware.
//
// Algorithm (from pooling.h, int16 path):
//   for c in [0, C):
//     total_sum = 0
//     for tile in [0, n_tiles):     // n_tiles = ceil(H*W / VEC_ACC_CAP)
//       vl = min(VEC_ACC_CAP, spatial - tile*VEC_ACC_CAP)
//       vle16(p, vl)                // unit-stride vector load: vl int16 elements
//       vwmul + vredsum              // widen to int32, reduce-sum
//       total_sum += scalar extract
//     output[c] = total_sum / spatial   // scalar divide + scalar store
//
// Memory traffic model
// --------------------
// READ  per tile : vl * POOL_INPUT_ELEM_BYTES
//   Mirrors the single vle16_v_i16m4 load per RVV iteration.
//   All spatial elements of the channel are read exactly once.
//
// WRITE per channel : POOL_OUTPUT_ELEM_BYTES (4 bytes, one int32)
//   The final "output[c] = total_sum / spatial" is a scalar store
//   that happens after all tiles. It is modelled as:
//     (a) a scalar CPU stall of POOL_DIVIDE_CYCLES for the integer divide, and
//     (b) a direct memory write request that bypasses vec_acc and
//         contributes to memory timing and bandwidth statistics.
//
// Workers are partitioned by channel: worker tid owns [c_start, c_end).
// Communication and synchronisation are consistent with dw_conv2d_sim.cpp:
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

#include "pooling_config.h"

using namespace sc_core;
using namespace tlm;

// ============================================================
// PoolExt — TLM extension carrying GAP request metadata.
// Attached alongside ReqExt and TxnExt on each transaction.
// AcceleratorTLM / Interconnect / Memory do not inspect this.
// ============================================================
struct PoolExt : tlm_extension<PoolExt>
{
    int channel_id = -1;  // channel this tile belongs to
    int tile_idx   =  0;  // tile index within the channel

    tlm_extension_base *clone() const override
    {
        return new PoolExt(*this);
    }

    void copy_from(const tlm_extension_base &other) override
    {
        *this = static_cast<const PoolExt &>(other);
    }
};

// ============================================================
// PoolWorker — per-thread task generator.
//
// Channel assignment:
//   c_start = (tid   * POOL_C) / n_workers
//   c_end   = ((tid+1) * POOL_C) / n_workers
//
// For every assigned channel the worker:
//   1. Fire phase  — issues all n_tiles reduction requests in order,
//      interleaving a scalar dispatch overhead between submissions.
//   2. Drain phase — collects all responses for the channel.
//   3. Scalar step — stalls POOL_DIVIDE_CYCLES for the integer divide
//      then issues a direct memory write of POOL_OUTPUT_ELEM_BYTES.
// ============================================================
struct PoolWorker : sc_module
{
    tlm_utils::simple_initiator_socket<PoolWorker>   init;
    tlm_utils::peq_with_get<tlm_generic_payload>     peq;

    int tid;
    int n_workers;

    // Statistics
    uint64_t vec_calls             = 0;   // total vec_acc requests issued
    uint64_t total_scalar_cycles   = 0;   // dispatch scalar overhead
    uint64_t total_divide_cycles   = 0;   // integer divide stall (one per channel)
    uint64_t total_wait_cycles     = 0;   // queue-wait + back-pressure stall
    uint64_t total_mem_cycles      = 0;   // memory service cycles
    uint64_t total_rd_bytes        = 0;   // total input bytes read via vec_acc
    uint64_t total_wr_bytes        = 0;   // total output bytes written (scalar)
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
        PoolExt             *pool_ext     = nullptr;
        DoneEntry           *done_entry   = nullptr;
        uint64_t             svc_cyc      = 0;
        uint64_t             stall_cycles = 0;
        bool                 sync_done    = false;
        bool                 direct_mem   = false;
        sc_time              submit_time  = SC_ZERO_TIME;
    };

    SC_HAS_PROCESS(PoolWorker);

    PoolWorker(sc_module_name name, int tid_, int n_workers_)
        : sc_module(name),
          init("init"),
          peq("peq"),
          tid(tid_),
          n_workers(n_workers_)
    {
        init.register_nb_transport_bw(this, &PoolWorker::nb_transport_bw);
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
    // issue_begin: fire one non-blocking vec_acc request.
    // Blocks only when the accelerator queue is full (TLM_ACCEPTED),
    // waiting on admit_ev until a slot is granted.
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

        auto *req  = new ReqExt(tid, svc_cyc, rd, wr);
        auto *tx   = new TxnExt();
        tx->src_worker = tid;

        auto *pool = new PoolExt();
        pool->channel_id = channel_id;
        pool->tile_idx   = tile_idx;

        gp->set_extension(req);
        gp->set_extension(tx);
        gp->set_extension(pool);

        p.gp         = gp;
        p.req_ext    = req;
        p.tx_ext     = tx;
        p.pool_ext   = pool;
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
    // issue_mem_write: fire one direct memory write request.
    // Used for the final scalar output[c] store so it contributes
    // to memory timing without occupying vec_acc.
    // ----------------------------------------------------------
    PendingReq issue_mem_write(uint64_t bytes, int channel_id)
    {
        PendingReq p;
        p.direct_mem = true;

        auto *gp = new tlm_generic_payload();
        gp->set_command(TLM_WRITE_COMMAND);
        gp->set_address(Interconnect::ADDR_MEM);
        gp->set_data_ptr(nullptr);
        gp->set_data_length((unsigned)bytes);
        gp->set_streaming_width((unsigned)bytes);
        gp->set_response_status(TLM_INCOMPLETE_RESPONSE);

        auto *tx = new TxnExt();
        tx->src_worker = tid;

        auto *pool = new PoolExt();
        pool->channel_id = channel_id;
        pool->tile_idx   = -1;

        gp->set_extension(tx);
        gp->set_extension(pool);

        p.gp         = gp;
        p.tx_ext     = tx;
        p.pool_ext   = pool;
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

        // Read back timing fields filled in by AcceleratorTLM.
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

        // Acknowledge response back to the interconnect.
        tlm_phase end_phase = END_RESP;
        sc_time   end_delay = SC_ZERO_TIME;
        init->nb_transport_fw(*p.gp, end_phase, end_delay);

        // Release all extensions and the payload.
        p.gp->clear_extension(p.req_ext);
        p.gp->clear_extension(p.tx_ext);
        p.gp->clear_extension(p.pool_ext);
        delete p.req_ext;  p.req_ext  = nullptr;
        delete p.tx_ext;   p.tx_ext   = nullptr;
        delete p.pool_ext; p.pool_ext = nullptr;
        delete p.gp;       p.gp       = nullptr;
    }

    // ----------------------------------------------------------
    // run: main worker thread — process assigned channel slice.
    //
    // For each channel:
    //   Fire phase : issue all reduction tiles (rd = vl * input_bytes, wr = 0)
    //   Drain phase: collect all responses
    //   Scalar step: stall POOL_DIVIDE_CYCLES for integer divide,
    //                then issue one direct memory write for output[c]
    //
    // The scalar store bypasses vec_acc but is still issued through the
    // Memory module so the final output traffic affects timing and BW.
    // ----------------------------------------------------------
    void run()
    {
        sc_time t_start = sc_time_stamp();

        const int      spatial = POOL_H * POOL_W;
        const int      n_tiles = (int)ceil_div_u64((uint64_t)spatial,
                                                    POOL_VEC_ACC_CAP);

        // Channel range for this worker.
        int c_start = (tid       * POOL_C) / n_workers;
        int c_end   = ((tid + 1) * POOL_C) / n_workers;

        for (int c = c_start; c < c_end; ++c)
        {
            // ------------------------------------------------
            // Fire phase: issue all reduction tiles.
            // Each tile reduces vl input elements to a partial
            // sum that stays in the vector register file.
            // ------------------------------------------------
            std::vector<PendingReq> pending;
            pending.reserve((size_t)n_tiles);

            for (int t = 0; t < n_tiles; ++t)
            {
                const uint64_t tile_elems =
                    std::min<uint64_t>(POOL_VEC_ACC_CAP,
                                       (uint64_t)spatial -
                                       (uint64_t)t * POOL_VEC_ACC_CAP);

                // READ: vl int16 elements from input[c, tile_start : tile_end]
                // WRITE: 0 — partial sums are held in vector registers;
                //        the final int32 output is written in the scalar step.
                uint64_t rd = tile_elems * POOL_INPUT_ELEM_BYTES;
                uint64_t wr = 0;

                auto pm = issue_begin(Interconnect::ADDR_VEC,
                                      POOL_VEC_ACC_CYCLE,
                                      rd, wr,
                                      c, t);
                ++vec_calls;
                total_rd_bytes += rd;

                do_scalar(POOL_SCALAR_OVERHEAD);
                pending.push_back(std::move(pm));
            }

            // ------------------------------------------------
            // Drain phase: collect all responses for this channel.
            // ------------------------------------------------
            for (auto &pm : pending)
                issue_end(pm);

            // ------------------------------------------------
            // Scalar post-processing:
            //   output[c] = (int32_t)(total_sum / spatial)
            //
            // Models the integer divide latency, then routes the
            // 4-byte scalar store through Memory without using vec_acc.
            // ------------------------------------------------
            total_divide_cycles += POOL_DIVIDE_CYCLES;
            wait(POOL_DIVIDE_CYCLES * CYCLE);
            auto store_req = issue_mem_write(POOL_OUTPUT_ELEM_BYTES, c);
            issue_end(store_req);
            total_wr_bytes += POOL_OUTPUT_ELEM_BYTES;
        }

        elapsed_cycles =
            (uint64_t)((sc_time_stamp() - t_start) / CYCLE);
    }
};

// ============================================================
// PoolTop — instantiates and wires the full simulator.
//
// Topology:
//   workers[0..N-1]  ──► noc ──► vec_acc ──► memory
//                             └──► mat_acc (dummy, never used)
//
// mat_acc is created only to satisfy the noc.to_mat binding.
// No worker ever sends to ADDR_MAT.
// ============================================================
struct PoolTop : sc_module
{
    AcceleratorTLM  mat_acc;   // dummy: bound to satisfy noc.to_mat
    AcceleratorPool vec_acc;   // actual: pool of POOL_VEC_ACC_INSTANCES units
    Interconnect    noc;
    Memory          memory;

    std::vector<PoolWorker *> workers;

    SC_HAS_PROCESS(PoolTop);

    PoolTop(sc_module_name name)
        : sc_module(name),
          mat_acc("mat_acc", POOL_ACC_QUEUE_DEPTH),
          vec_acc("vec_acc",
                  (size_t)POOL_VEC_ACC_INSTANCES,
                  POOL_ACC_QUEUE_DEPTH),
          noc("noc"),
          memory("memory",
                 POOL_MEM_BASE_LAT,
                 POOL_MEM_BW,
                 (uint64_t)POOL_VEC_ACC_INSTANCES)
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
        for (int i = 0; i < POOL_NUM_WORKERS; ++i)
        {
            auto *w = new PoolWorker(
                          sc_gen_unique_name("pool_worker"),
                          i, POOL_NUM_WORKERS);
            workers.push_back(w);
            w->init.bind(noc.tgt);
        }
    }

    ~PoolTop() override
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

    PoolTop top("pool_top");
    sc_start();

    // ----------------------------------------------------------
    // Derived geometry
    // ----------------------------------------------------------
    const int      spatial  = POOL_H * POOL_W;
    const int      n_tiles  = (int)ceil_div_u64((uint64_t)spatial,
                                                 POOL_VEC_ACC_CAP);
    const uint64_t expected_vec_calls =
        (uint64_t)POOL_C * (uint64_t)n_tiles;
    const uint64_t expected_rd_bytes =
        (uint64_t)POOL_C * (uint64_t)spatial * POOL_INPUT_ELEM_BYTES;
    const uint64_t expected_wr_bytes =
        (uint64_t)POOL_C * POOL_OUTPUT_ELEM_BYTES;

    // ----------------------------------------------------------
    // Section 0: Configuration header
    // ----------------------------------------------------------
    std::cout << "\n";
    std::cout << "==============================================\n";
    std::cout << "  Global Average Pooling TLM Performance Simulation\n";
    std::cout << "  Algorithm : mf_global_avgpool_i"
              << (POOL_INPUT_ELEM_BYTES * 8) << "\n";
    std::cout << "  Input     : [C=" << POOL_C
              << ", H=" << POOL_H
              << ", W=" << POOL_W << "]"
              << "  (int" << (POOL_INPUT_ELEM_BYTES * 8) << ")\n";
    std::cout << "  Output    : [C=" << POOL_C << "]"
              << "  (int" << (POOL_OUTPUT_ELEM_BYTES * 8) << ")\n";
    std::cout << "  Spatial elements (H*W)   : " << spatial          << "\n";
    std::cout << "  Input elem bytes         : " << POOL_INPUT_ELEM_BYTES
              << "  (change POOL_INPUT_ELEM_BYTES in pooling_config.h to switch)\n";
    std::cout << "  Output elem bytes        : " << POOL_OUTPUT_ELEM_BYTES << "\n";
    std::cout << "  Workers                  : " << POOL_NUM_WORKERS       << "\n";
    std::cout << "  vec_acc units            : " << POOL_VEC_ACC_INSTANCES  << "\n";
    std::cout << "  VEC_ACC_CAP              : " << POOL_VEC_ACC_CAP
              << " elements/call,  " << POOL_VEC_ACC_CYCLE << " cycles/call\n";
    std::cout << "  Tiles per channel        : " << n_tiles               << "\n";
    std::cout << "  Expected vec_acc calls   : " << expected_vec_calls    << "\n";
    std::cout << "\n";
    std::cout << "  Memory traffic model:\n";
    std::cout << "    READ  per tile    = vl * " << POOL_INPUT_ELEM_BYTES
              << " bytes  (vle" << (POOL_INPUT_ELEM_BYTES * 8)
              << " unit-stride load)\n";
    std::cout << "    WRITE per channel = " << POOL_OUTPUT_ELEM_BYTES
              << " bytes  (scalar store of output[c], after integer divide)\n";
    std::cout << "    Write bypasses vec_acc but is routed through Memory\n";
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
              << std::setw(14) << "DivideCyc"
              << std::setw(12) << "WaitCyc"
              << std::setw(12) << "MemCyc"
              << std::setw(14) << "RdBytes"
              << std::setw(14) << "WrBytes"
              << std::setw(12) << "ElapsedCyc"
              << "\n";
    std::cout << std::string(124, '-') << "\n";

    uint64_t max_elapsed    = 0;
    uint64_t total_vec      = 0;
    uint64_t total_rd_all   = 0;
    uint64_t total_wr_all   = 0;
    uint64_t total_wait_all = 0;
    uint64_t total_mem_all  = 0;

    for (const auto *w : top.workers)
    {
        int c_start = (w->tid       * POOL_C) / POOL_NUM_WORKERS;
        int c_end   = ((w->tid + 1) * POOL_C) / POOL_NUM_WORKERS;
        std::string range = "[" + std::to_string(c_start)
                          + ","  + std::to_string(c_end)  + ")";
        std::cout << std::left
                  << std::setw(8)  << w->tid
                  << std::setw(14) << range
                  << std::setw(10) << w->vec_calls
                  << std::setw(14) << w->total_scalar_cycles
                  << std::setw(14) << w->total_divide_cycles
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
    std::cout << "Workers                      : " << POOL_NUM_WORKERS     << "\n";
    std::cout << "Channels (C)                 : " << POOL_C               << "\n";
    std::cout << "Spatial elements (H*W)       : " << spatial              << "\n";
    std::cout << "Vec tiles per channel        : " << n_tiles              << "\n";
    std::cout << "Total vec-acc calls          : " << total_vec
              << "  (expected " << expected_vec_calls << ") "
              << (total_vec == expected_vec_calls ? "[OK]" : "[MISMATCH]") << "\n";
    std::cout << "Total memory reads (vec_acc) : " << total_rd_all
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

    // Accelerator utilisation: busy cycles / (sim_cycles × num_units)
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

    // Memory bandwidth: modeled bytes transferred / total cycles
    const uint64_t total_mem_bytes = total_rd_all + total_wr_all;
    double mem_bw = (sim_cycles > 0.0)
        ? static_cast<double>(total_mem_bytes) / sim_cycles
        : 0.0;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "vec_acc compute util         : " << vec_util << "%\n";
    std::cout << "vec_acc occupancy            : " << vec_occ  << "%\n";
    std::cout << "Memory avg BW                : " << mem_bw   << " bytes/cycle\n";

    // Bottleneck hint
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
              << " Total vec-acc calls == C * ceil(H*W/CAP)"
              << " (" << total_vec << " == " << expected_vec_calls << ")\n";
    std::cout << (ok_rd ? "  [PASS]" : "  [FAIL]")
              << " Total read bytes == C * H * W * input_bytes"
              << " (" << total_rd_all << " == " << expected_rd_bytes << ")\n";
    std::cout << (ok_wr ? "  [PASS]" : "  [FAIL]")
              << " Total write bytes == C * output_bytes"
              << " (" << total_wr_all << " == " << expected_wr_bytes << ")\n";

    bool pass = ok_calls && ok_rd && ok_wr;
    std::cout << (pass ? "\nAll checks passed.\n" : "\nSome checks FAILED.\n");
    return pass ? 0 : 2;
}
