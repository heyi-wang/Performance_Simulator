#pragma once

#include "accelerator.h"
#include <deque>
#include <memory>
#include <tlm_utils/simple_initiator_socket.h>

// ============================================================
// AcceleratorPool — one shared request queue feeding multiple
// identical accelerator instances.
//
// Upstream requestors see a single target socket per accelerator
// class, while requests are dispatched FIFO to the first free
// physical accelerator instance.
// ============================================================
struct AcceleratorPool : sc_module
{
    tlm_utils::simple_target_socket<AcceleratorPool> tgt;

    struct Entry
    {
        tlm_generic_payload *gp = nullptr;
        sc_time              enqueue_time;
    };

    std::vector<std::unique_ptr<AcceleratorTLM>> units;
    std::vector<std::unique_ptr<tlm_utils::simple_initiator_socket_tagged<AcceleratorPool>>> to_units;
    std::vector<bool> unit_busy;

    std::deque<Entry> q;
    sc_event          q_changed;
    std::deque<tlm_generic_payload *> stall_fifo;

    size_t admitted = 0;
    size_t queue_capacity = 0;
    uint64_t shared_queue_wait_cycles = 0;

    SC_HAS_PROCESS(AcceleratorPool);

    AcceleratorPool(sc_module_name name, size_t instance_count, size_t queue_capacity_);

    tlm_sync_enum nb_transport_fw(tlm_generic_payload &gp,
                                  tlm_phase &phase,
                                  sc_time &delay);

    tlm_sync_enum nb_transport_bw_unit(int id,
                                       tlm_generic_payload &gp,
                                       tlm_phase &phase,
                                       sc_time &delay);

    void dispatch_thread();

    size_t instance_count() const;
    bool   has_free_unit() const;
    int    find_free_unit() const;

    uint64_t req_count_total() const;
    uint64_t busy_cycles_total() const;
    uint64_t occupied_cycles_total() const;
    uint64_t queue_wait_cycles_total() const;
};
