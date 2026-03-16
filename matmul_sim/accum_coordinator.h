#pragma once

#include "../src/extensions.h"
#include "../src/worker.h"
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/peq_with_get.h>
#include <vector>
#include <unordered_map>

// ============================================================
// AccumCoordinator — dependency-aware partial-result accumulation
// for K-split GEMM.
//
// Execution model (event-driven tree reduction):
//   Each internal node of the binary reduction tree is an
//   independent SC_THREAD spawned during run().  A thread waits
//   for exactly its two input "ready" events, then immediately
//   issues accum_vec_calls VectorAccelerator requests, then
//   fires its own output event.
//
//   This means accumulation starts as soon as ANY pair of
//   partial results is available — without waiting for all
//   other workers to finish.
//
//   Leaf events wrap worker->mat_done_ev via lightweight proxy
//   threads so that the coordinator does not touch Worker
//   internals after sc_start().
//
// The coordinator owns its TLM initiator socket, bound to the
// interconnect just like a Worker.
// ============================================================
struct AccumCoordinator : sc_module
{
    tlm_utils::simple_initiator_socket<AccumCoordinator> init;
    tlm_utils::peq_with_get<tlm_generic_payload>         peq;

    std::vector<Worker *> workers;

    // Cycles the VectorAccelerator takes per vec_acc call
    uint64_t vec_svc_cycles;
    // Number of vec_acc calls per pairwise accumulation
    uint64_t accum_vec_calls;
    // Memory bytes per vec_acc call
    uint64_t accum_rd_bytes;
    uint64_t accum_wr_bytes;

    // Statistics filled in after run() completes
    sc_time  accum_end_time;              // sim_time when the final result is ready
    uint64_t vec_calls_total      = 0;   // total vec_acc calls issued
    uint64_t wait_cycles          = 0;   // accumulated queue-wait cycles
    uint64_t mem_cycles           = 0;   // accumulated memory cycles

    // Per-pair timing records: each entry corresponds to one pairwise
    // accumulation in tree order (left-to-right, bottom-up).
    // pair_start_times[i] = sim_time when both inputs for pair i were ready.
    // pair_end_times[i]   = sim_time when pair i's accumulation completed.
    std::vector<sc_time> pair_start_times;
    std::vector<sc_time> pair_end_times;
    sc_mutex             stats_mutex;   // protects the vectors above

    // Owns all sc_event objects created for the reduction tree so they
    // remain valid for the lifetime of the simulation.
    std::vector<sc_event *> tree_events_;

    std::unordered_map<tlm_generic_payload *, sc_event *> done_map;

    SC_HAS_PROCESS(AccumCoordinator);

    AccumCoordinator(sc_module_name name,
                     uint64_t vec_svc_cycles_,
                     uint64_t accum_vec_calls_,
                     uint64_t accum_rd_bytes_,
                     uint64_t accum_wr_bytes_);

    ~AccumCoordinator() override;

    tlm_sync_enum nb_transport_bw(tlm_generic_payload &gp,
                                  tlm_phase &phase,
                                  sc_time &delay);

    void peq_thread();

    struct PendingReq
    {
        tlm_generic_payload *gp        = nullptr;
        ReqExt              *req_ext   = nullptr;
        TxnExt              *tx_ext    = nullptr;
        sc_event            *done_ev   = nullptr;
        bool                 sync_done = false;
    };

    PendingReq issue_begin(uint64_t addr);
    void       issue_end(PendingReq &p);

    // Run one pairwise accumulation: issues accum_vec_calls sequential
    // vec_acc requests.  Records timing in pair_start/end_times.
    void run_one_pair();

    void run();
};
