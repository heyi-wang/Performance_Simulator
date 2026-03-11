#pragma once

#include "extensions.h"
#include <tlm_utils/simple_target_socket.h>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/peq_with_get.h>
#include <deque>
#include <unordered_map>

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
    };

    std::deque<Entry> q;
    sc_event          q_nonempty;

    // Backpressure: requests that arrived when the queue was full.
    // Each GP here is waiting for a deferred END_REQ to be sent back.
    std::deque<tlm_generic_payload *> stall_fifo;

    // Total admitted slots currently in use:
    //   admitted = (entries in PEQ) + (entries in q) + (one being serviced)
    // nb_transport_fw increments this before accepting; service_thread
    // decrements it (or hands the slot to stall_fifo) after finishing.
    size_t admitted      = 0;
    size_t queue_capacity;

    uint64_t busy_cycles       = 0;
    uint64_t queue_wait_cycles = 0;
    uint64_t req_count         = 0;

    std::unordered_map<tlm_generic_payload *, sc_event *> mem_done_map;

    SC_HAS_PROCESS(AcceleratorTLM);

    AcceleratorTLM(sc_module_name name, size_t cap);

    tlm_sync_enum nb_transport_fw(tlm_generic_payload &gp,
                                  tlm_phase &phase,
                                  sc_time &delay);

    tlm_sync_enum nb_transport_bw_mem(tlm_generic_payload &gp,
                                      tlm_phase &phase,
                                      sc_time &delay);

    void mem_access(bool is_write, uint64_t bytes);

    void peq_thread();
    void service_thread();
};
