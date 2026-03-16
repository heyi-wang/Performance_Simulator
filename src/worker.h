#pragma once

#include "extensions.h"
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/peq_with_get.h>
#include <unordered_map>

// ============================================================
// Worker — models one parallel compute thread
//
// Pipeline per accelerator type:
//   issue_begin[i] → do_scalar (∥ accel processes i)
//                  → issue_end[i] → issue_begin[i+1] → ...
// ============================================================
struct Worker : sc_module
{
    tlm_utils::simple_initiator_socket<Worker>   init;
    tlm_utils::peq_with_get<tlm_generic_payload> peq;

    int tid;

    uint64_t m;             // unused, kept for ABI compatibility
    uint64_t access_mat;
    uint64_t access_vec;
    uint64_t mat_cycles;
    uint64_t vec_cycles;
    uint64_t scalar_cycles;

    uint64_t compute_cycles   = 0;
    uint64_t wait_cycles      = 0;
    uint64_t mem_cycles_accum = 0;
    uint64_t mat_calls        = 0;
    uint64_t vec_calls        = 0;
    uint64_t elapsed_cycles   = 0;   // set at end of run(); used for GFLOPS reporting
    uint64_t mat_elapsed_cycles = 0; // cycles until end of mat phase; set before mat_done_ev

    // Fired when the matrix-multiply phase of run() completes.
    // AccumCoordinator waits on this to start the accumulation stage.
    sc_event  mat_done_ev;

    uint64_t A_bytes = 0;
    uint64_t B_bytes = 0;
    uint64_t C_bytes = 0;

    std::unordered_map<tlm_generic_payload *, sc_event *> done_map;

    // Handle for an in-flight accelerator request.
    // Returned by issue_begin(); must be finalized with issue_end().
    struct PendingReq
    {
        tlm_generic_payload *gp       = nullptr;
        ReqExt              *req_ext  = nullptr;
        TxnExt              *tx_ext   = nullptr;
        sc_event            *done_ev  = nullptr;
        uint64_t             svc_cycles = 0;
        bool                 sync_done  = false;
    };

    SC_HAS_PROCESS(Worker);

    Worker(sc_module_name name,
           int      tid_,
           uint64_t access_mat_,
           uint64_t access_vec_,
           uint64_t mat_cycles_,
           uint64_t vec_cycles_,
           uint64_t scalar_cycles_,
           uint64_t A_bytes_,
           uint64_t B_bytes_,
           uint64_t C_bytes_);

    tlm_sync_enum nb_transport_bw(tlm_generic_payload &gp,
                                  tlm_phase &phase,
                                  sc_time &delay);

    void peq_thread();
    void do_scalar(uint64_t cyc);

    PendingReq issue_begin(uint64_t addr, uint64_t svc_cycles);
    void       issue_end(PendingReq &p);

    void run();
};
