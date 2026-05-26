#pragma once

#include "extensions.h"
#include <tlm_utils/simple_target_socket.h>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/peq_with_get.h>
#include <deque>
#include <functional>

// ============================================================
// AcceleratorTLM — single-server FIFO accelerator
//   Accepts requests from workers via tgt.
//   Issues read/write sub-transactions to memory via to_mem.
//   Used for both the matrix and vector accelerator instances.
// ============================================================
struct AcceleratorTLM : sc_module
{
    tlm_utils::simple_target_socket<AcceleratorTLM>    tgt;
    tlm_utils::simple_initiator_socket<AcceleratorTLM> to_mem;

    tlm_utils::peq_with_get<tlm_generic_payload> peq;

    struct Entry
    {
        tlm_generic_payload *gp = nullptr;
        sc_time              enqueue_time;
        sc_time              t_load_start;
        sc_time              t_load_end;
        sc_time              t_compute_start;
        sc_time              t_compute_end;
        sc_time              t_write_start;
        sc_time              t_write_end;
    };

    std::deque<Entry> q;
    sc_event          q_nonempty;

    // Pipelined-mode inter-stage queues (capacity 1).
    bool                pipeline_enabled = false;
    std::deque<Entry>   loaded_q;
    std::deque<Entry>   computed_q;
    sc_event            loaded_q_changed;
    sc_event            computed_q_changed;

    // Wall-time tracking of "any stage in flight" used by pipelined mode.
    int       pipeline_active_stages = 0;
    sc_time   pipeline_busy_start;

    // Backpressure: requests that arrived when the queue was full.
    // Each GP here is waiting for a deferred END_REQ to be sent back.
    std::deque<tlm_generic_payload *> stall_fifo;

    // Total admitted slots currently in use:
    //   admitted = (entries in PEQ) + (entries in q) + (one being serviced)
    // nb_transport_fw increments this before accepting; service_thread
    // decrements it (or hands the slot to stall_fifo) after finishing.
    size_t admitted      = 0;
    size_t queue_capacity;

    // Optional callback: called with (cycle, busy) at every busy/idle transition.
    // Set via set_busy_callback() before sc_start().  Null by default.
    std::function<void(uint64_t, bool)> busy_cb;

    void set_busy_callback(std::function<void(uint64_t, bool)> cb)
    {
        busy_cb = std::move(cb);
    }

    uint64_t busy_cycles       = 0;
    uint64_t occupied_cycles   = 0;
    uint64_t queue_wait_cycles = 0;
    uint64_t req_count         = 0;

#ifdef PERFETTO_TRACE
    // End time of the last request this unit serviced; used to emit the idle
    // ("stall") lane in the Perfetto trace (gap until the next service start).
    sc_time perf_last_busy_end = SC_ZERO_TIME;
#endif

    SC_HAS_PROCESS(AcceleratorTLM);

    AcceleratorTLM(sc_module_name name, size_t cap, bool enable_pipeline = false);

    tlm_sync_enum nb_transport_fw(tlm_generic_payload &gp,
                                  tlm_phase &phase,
                                  sc_time &delay);

    tlm_sync_enum nb_transport_bw_mem(tlm_generic_payload &gp,
                                      tlm_phase &phase,
                                      sc_time &delay);

    void mem_access(bool is_write, uint64_t bytes);
    void enqueue_request(tlm_generic_payload &gp);

    void peq_thread();
    void service_thread();

    // Pipelined-mode stage threads.
    void load_thread();
    void compute_thread();
    void write_thread();

    // Helpers for pipelined-mode "any stage active" wall-time tracking.
    void stage_enter();
    void stage_exit();

    // Tail-end completion path shared by service_thread and write_thread:
    // sends BEGIN_RESP, drains stall_fifo / decrements admitted.
    void complete_request(Entry &e);
};
