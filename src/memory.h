#pragma once

#include "common.h"
#include <tlm_utils/simple_target_socket.h>
#include <tlm_utils/peq_with_get.h>
#include <deque>

// ============================================================
// Memory — non-blocking target, FIFO-fed multi-slot timing model
//   latency = base_lat_cycles + ceil(bytes / bytes_per_cycle)
// ============================================================
struct Memory : sc_module
{
    tlm_utils::simple_target_socket<Memory> tgt;

    uint64_t base_lat_cycles;
    uint64_t bytes_per_cycle;
    uint64_t parallel_slots;
    uint64_t active_reqs = 0;

    tlm_utils::peq_with_get<tlm_generic_payload> req_peq;
    tlm_utils::peq_with_get<tlm_generic_payload> resp_peq;

    struct Entry
    {
        tlm_generic_payload *gp = nullptr;
        sc_time              enqueue_time;
    };

    std::deque<Entry> q;
    sc_event          dispatch_ev;

    uint64_t reqs         = 0;
    uint64_t busy_cycles  = 0;
    uint64_t qwait_cycles = 0;

    SC_HAS_PROCESS(Memory);

    Memory(sc_module_name name,
           uint64_t base_lat_cycles_ = 1,
           uint64_t bytes_per_cycle_ = 128,
           uint64_t parallel_slots_ = 1);

    tlm_sync_enum nb_transport_fw(tlm_generic_payload &gp,
                                  tlm_phase &phase,
                                  sc_time &delay);

    void peq_thread();
    void dispatch_thread();
    void response_thread();
};
