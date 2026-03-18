#pragma once

#include "../src/extensions.h"
#include "../src/worker.h"
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/peq_with_get.h>
#include <deque>
#include <vector>
#include <unordered_map>

// ============================================================
// AccumCoordinator — dependency-aware partial-result
// accumulation for K-split GEMM.
//
// Execution model (event-driven tree reduction):
//   Each internal node of the binary reduction tree is an
//   independent SC_THREAD spawned during run(). A thread waits
//   for exactly its two input "ready" events, then keeps a window
//   of vec requests in flight so different vec accelerators can
//   process that stage concurrently. After the root reduction
//   completes, one final quantization pass is issued with the same
//   pipelined request window.
//
//   Accumulation starts as soon as ANY pair of partial results
//   is available — without waiting for all other workers to finish.
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
    uint64_t final_quant_calls;// vec_acc calls for final output quantization
    uint64_t max_inflight_vec_reqs; // max outstanding vec requests per stage
    uint64_t scalar_cycles;    // scalar overhead cycles per vec_acc call
    uint64_t accum_rd_bytes;   // read bytes per vec_acc call
    uint64_t accum_wr_bytes;   // write bytes per vec_acc call
    uint64_t quant_rd_bytes;   // read bytes per final quant vec_acc call
    uint64_t quant_wr_bytes;   // write bytes per final quant vec_acc call

    // Statistics filled in after run() completes
    sc_time  accum_end_time;                // sim_time when final result is ready
    uint64_t vec_calls_total      = 0;      // total vec_acc calls issued
    uint64_t accum_vec_calls_total = 0;     // vec_acc calls spent on accumulation
    uint64_t final_quant_calls_total = 0;   // vec_acc calls spent on final quantization
    uint64_t compute_cycles       = 0;      // coordinator scalar-overhead cycles
    uint64_t wait_cycles          = 0;      // queue-wait + backpressure stall
    uint64_t stall_cycles         = 0;      // backpressure stall only
    uint64_t mem_cycles           = 0;      // accumulated memory cycles

    // Per-pair timing (one entry per pairwise accumulation, tree order)
    std::vector<sc_time> pair_start_times;
    std::vector<sc_time> pair_end_times;
    std::vector<sc_time> pair_left_ready_times;
    std::vector<sc_time> pair_right_ready_times;
    sc_mutex             stats_mutex;       // protects the two vectors above

    // Owns all sc_event objects for the reduction tree
    std::vector<sc_event *> tree_events_;
    std::vector<sc_time *>  tree_ready_times_;
    std::vector<bool *>     tree_ready_flags_;

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
                     uint64_t final_quant_calls_,
                     uint64_t max_inflight_vec_reqs_,
                     uint64_t scalar_cycles_,
                     uint64_t accum_rd_bytes_,
                     uint64_t accum_wr_bytes_,
                     uint64_t quant_rd_bytes_,
                     uint64_t quant_wr_bytes_);

    ~AccumCoordinator() override;

    tlm_sync_enum nb_transport_bw(tlm_generic_payload &gp,
                                  tlm_phase &phase,
                                  sc_time &delay);

    void peq_thread();

    PendingReq issue_begin(uint64_t addr, uint64_t rd_bytes, uint64_t wr_bytes);
    void       issue_end(PendingReq &p);
    void       do_scalar(uint64_t cyc);
    void       issue_vec_stream(uint64_t call_count,
                                uint64_t rd_bytes,
                                uint64_t wr_bytes,
                                uint64_t &stage_call_counter);

    // Run one pairwise accumulation with a pipelined vec request window.
    void run_one_pair(size_t pair_id, sc_time left_ready, sc_time right_ready);
    void run_final_quant();

    void run();
};
