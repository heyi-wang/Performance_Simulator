#pragma once

#include "../src/extensions.h"
#include "../src/worker.h"
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/peq_with_get.h>
#include <vector>
#include <unordered_map>

// ============================================================
// AccumCoordinator — dependency-aware partial-result
// accumulation for K-split GEMM.
//
// Execution model (event-driven tree reduction):
//   Each internal node of the binary reduction tree is an
//   independent SC_THREAD spawned during run().  A thread waits
//   for exactly its two input "ready" events, then immediately
//   issues accum_vec_calls VectorAccelerator requests, then
//   fires its own output event.
//
//   Accumulation starts as soon as ANY pair of quantized
//   partial results is available — without waiting for all
//   other workers to finish.
//
// Backpressure: when vec_acc queue is full, issue_begin blocks
// on admit_ev until the accelerator sends a deferred END_REQ.
// ============================================================
struct AccumCoordinator : sc_module
{
    tlm_utils::simple_initiator_socket<AccumCoordinator> init;
    tlm_utils::peq_with_get<tlm_generic_payload>         peq;

    std::vector<Worker *> workers;

    uint64_t vec_svc_cycles;   // VectorAccelerator service cycles per call
    uint64_t accum_vec_calls;  // vec_acc calls per pairwise accumulation
    uint64_t accum_rd_bytes;   // read bytes per vec_acc call
    uint64_t accum_wr_bytes;   // write bytes per vec_acc call

    // Statistics filled in after run() completes
    sc_time  accum_end_time;                // sim_time when final result is ready
    uint64_t vec_calls_total      = 0;      // total vec_acc calls issued
    uint64_t wait_cycles          = 0;      // queue-wait + backpressure stall
    uint64_t stall_cycles         = 0;      // backpressure stall only
    uint64_t mem_cycles           = 0;      // accumulated memory cycles

    // Per-pair timing (one entry per pairwise accumulation, tree order)
    std::vector<sc_time> pair_start_times;
    std::vector<sc_time> pair_end_times;
    sc_mutex             stats_mutex;       // protects the two vectors above

    // Owns all sc_event objects for the reduction tree
    std::vector<sc_event *> tree_events_;

    // ----------------------------------------------------------
    // DoneEntry — same backpressure pattern as Worker.
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
        DoneEntry           *done_entry   = nullptr;
        uint64_t             stall_cycles = 0;
        bool                 sync_done    = false;
    };

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

    PendingReq issue_begin(uint64_t addr);
    void       issue_end(PendingReq &p);

    // Run one pairwise accumulation: accum_vec_calls sequential vec_acc requests.
    void run_one_pair();

    void run();
};
