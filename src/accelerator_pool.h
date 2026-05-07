#pragma once

#include "accelerator.h"
#include <deque>
#include <memory>
#include <tlm_utils/simple_initiator_socket.h>

// ============================================================
// AcceleratorPool — feeds N identical AcceleratorTLM instances.
//
// Two operating modes selected at construction:
//   * Shared-FIFO (default): one request queue, dynamically dispatched
//     to the first free unit. Used by every kernel except matmul-static.
//   * Per-accel pinned: one queue per unit; an upstream worker is
//     statically routed to a single unit via worker_to_accel_map.
//     The matmul kernel uses this mode for the matrix pool.
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

    // Shared-FIFO mode state
    std::deque<Entry> q;
    sc_event          q_changed;
    std::deque<tlm_generic_payload *> stall_fifo;
    size_t   admitted = 0;
    uint64_t shared_queue_wait_cycles = 0;

    size_t queue_capacity = 0;

    // Per-accel pinned mode state (populated only when per_accel_mode)
    bool per_accel_mode = false;
    std::vector<int>      worker_to_accel_map;
    std::vector<uint64_t> per_unit_register_count;
    std::vector<std::deque<Entry>>                    per_unit_q;
    std::vector<std::unique_ptr<sc_event>>            per_unit_q_changed;
    std::vector<std::deque<tlm_generic_payload *>>    per_unit_stall;
    std::vector<size_t>                               per_unit_admitted;
    std::vector<uint64_t>                             per_unit_qwait_cycles;

    SC_HAS_PROCESS(AcceleratorPool);

    // Shared-FIFO ctor (unchanged)
    AcceleratorPool(sc_module_name name, size_t instance_count, size_t queue_capacity_);

    // Per-accel pinned ctor
    AcceleratorPool(sc_module_name name,
                    size_t instance_count,
                    size_t per_unit_capacity,
                    std::vector<int> worker_to_accel,
                    std::vector<uint64_t> per_unit_registers);

    tlm_sync_enum nb_transport_fw(tlm_generic_payload &gp,
                                  tlm_phase &phase,
                                  sc_time &delay);

    tlm_sync_enum nb_transport_bw_unit(int id,
                                       tlm_generic_payload &gp,
                                       tlm_phase &phase,
                                       sc_time &delay);

    void dispatch_thread();
    void per_unit_dispatch_thread(int unit_id);

    size_t instance_count() const;
    bool   has_free_unit() const;
    int    find_free_unit() const;

    uint64_t req_count_total() const;
    uint64_t busy_cycles_total() const;
    uint64_t occupied_cycles_total() const;
    uint64_t queue_wait_cycles_total() const;

    std::vector<AccelInstanceStats> per_instance_stats() const;

private:
    void build_units(size_t count);
};
