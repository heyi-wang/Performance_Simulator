#pragma once

#include "extensions.h"
#include <deque>
#include <systemc>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/peq_with_get.h>
#include <unordered_map>

struct Worker;
struct MemoryAccessExt;

// Optional hook for simulator-specific post-matmul work that still runs
// inside the worker's own SC_THREAD.
struct WorkerPostProcessor
{
    virtual ~WorkerPostProcessor() = default;
    virtual void run_post_mat(Worker &worker) = 0;
};

// ============================================================
// Worker — models one parallel compute thread
//
// Pipeline per accelerator type:
//   issue_begin[i] → do_scalar (∥ accel processes i)
//                  → issue_end[i] → issue_begin[i+1] → ...
//
// Backpressure: when the accelerator queue is full,
// nb_transport_fw returns TLM_ACCEPTED instead of TLM_UPDATED.
// issue_begin then blocks on admit_ev until the accelerator
// sends a deferred END_REQ backward to grant the slot.
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
    uint64_t max_inflight_mat_reqs = 1;
    uint64_t max_inflight_vec_reqs = 1;
    WorkerPostProcessor *post_processor = nullptr;
    sc_event *start_event = nullptr;
    sc_fifo<int> *completion_fifo = nullptr;

    uint64_t compute_cycles   = 0;
    uint64_t wait_cycles      = 0;   // accelerator queue wait only
    uint64_t stall_cycles     = 0;   // worker-side backpressure stall only
    uint64_t mem_cycles_accum = 0;
    uint64_t mat_calls        = 0;
    uint64_t vec_calls        = 0;
    uint64_t accum_vec_calls  = 0;
    uint64_t quant_vec_calls  = 0;
    uint64_t reduction_pairs  = 0;
    uint64_t elapsed_cycles   = 0;   // set at end of run(); used for GFLOPS reporting
    uint64_t mat_elapsed_cycles = 0; // cycles until the local mat phase completes
    sc_time  mat_done_time;

    // Fired after both mat tiles AND quantization vec tiles finish.
    // AccumCoordinator waits on this to start tree-reduction accumulation.
    sc_event mat_done_ev;

    uint64_t A_bytes      = 0;   // mat-request read bytes (A tile)
    uint64_t B_bytes      = 0;   // mat-request read bytes (B tile)
    uint64_t C_bytes      = 0;   // mat-request write bytes (C tile)
    uint64_t vec_rd_bytes = 0;   // vec-request read bytes  (0 → uses A+B)
    uint64_t vec_wr_bytes = 0;   // vec-request write bytes (0 → uses C)

    // ----------------------------------------------------------
    // DoneEntry — per-request synchronisation state.
    //   ev:       notified when accelerator sends BEGIN_RESP.
    //   admit_ev: notified when accelerator sends deferred END_REQ
    //             (backpressure: queue slot granted after stall).
    //   fired:    set true by peq_thread before notifying ev, so
    //             issue_end can skip wait() if response already
    //             arrived before issue_end is called.
    // ----------------------------------------------------------
    struct DoneEntry
    {
        sc_event *ev       = nullptr;
        sc_event *admit_ev = nullptr;
        bool      fired    = false;
    };

    std::unordered_map<tlm_generic_payload *, DoneEntry *> done_map;

    // Handle for an in-flight accelerator request.
    // Returned by issue_begin(); must be finalized with issue_end().
    struct PendingReq
    {
        tlm_generic_payload *gp           = nullptr;
        ReqExt              *req_ext      = nullptr;
        TxnExt              *tx_ext       = nullptr;
        DoneEntry           *done_entry   = nullptr;
        uint64_t             svc_cycles   = 0;
        uint64_t             stall_cycles = 0;   // backpressure stall for this request
        bool                 sync_done    = false;
    };

    // Handle for an in-flight DMA request.
    struct DmaReq
    {
        tlm_generic_payload *gp         = nullptr;
        MemoryAccessExt     *mem_ext    = nullptr;
        TxnExt              *tx_ext     = nullptr;
        DoneEntry           *done_entry = nullptr;
        bool                 sync_done  = true;
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
           uint64_t C_bytes_,
           uint64_t vec_rd_ = 0,
           uint64_t vec_wr_ = 0,
           uint64_t max_inflight_mat_reqs_ = 1,
           uint64_t max_inflight_vec_reqs_ = 1,
           WorkerPostProcessor *post_processor_ = nullptr,
           sc_event *start_event_ = nullptr,
           sc_fifo<int> *completion_fifo_ = nullptr);

    tlm_sync_enum nb_transport_bw(tlm_generic_payload &gp,
                                  tlm_phase &phase,
                                  sc_time &delay);

    void peq_thread();
    void do_scalar(uint64_t cyc);

    // issue_begin with explicit rd/wr bytes (used for vec/quant requests)
    PendingReq issue_begin(uint64_t addr, uint64_t svc_cycles, uint64_t rd, uint64_t wr);

    // Convenience overload: uses A_bytes+B_bytes and C_bytes (mat requests)
    PendingReq issue_begin(uint64_t addr, uint64_t svc_cycles);

    DmaReq issue_dma_begin(bool is_write, uint64_t bytes);
    void finish_dma(DmaReq &p);

    void issue_end(PendingReq &p);
    void issue_stream(uint64_t addr,
                      uint64_t call_count,
                      uint64_t svc_cycles,
                      uint64_t rd,
                      uint64_t wr,
                      uint64_t dma_rd,
                      uint64_t dma_wr,
                      uint64_t &call_counter,
                      uint64_t *phase_counter = nullptr,
                      uint64_t max_inflight = 1);

    void run();
};
