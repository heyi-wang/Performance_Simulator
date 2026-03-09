#pragma once

#include "common.h"
#include <tlm_utils/simple_target_socket.h>
#include <tlm_utils/peq_with_get.h>
#include <deque>

// ============================================================
// Memory — non-blocking target, single-server FIFO timing model
//   latency = base_lat_cycles + ceil(bytes / bytes_per_cycle)
// ============================================================
struct Memory : sc_module
{
    tlm_utils::simple_target_socket<Memory> tgt;

    uint64_t base_lat_cycles;
    uint64_t bytes_per_cycle;

    tlm_utils::peq_with_get<tlm_generic_payload> peq;

    struct Entry
    {
        tlm_generic_payload *gp = nullptr;
        sc_time              enqueue_time;
    };

    std::deque<Entry> q;
    sc_event          q_nonempty;

    uint64_t reqs         = 0;
    uint64_t busy_cycles  = 0;
    uint64_t qwait_cycles = 0;

    SC_HAS_PROCESS(Memory);

    Memory(sc_module_name name,
           uint64_t base_lat_cycles_ = 20,
           uint64_t bytes_per_cycle_ = 32);

    tlm_sync_enum nb_transport_fw(tlm_generic_payload &gp,
                                  tlm_phase &phase,
                                  sc_time &delay);

    void peq_thread();
    void service_thread();
};
