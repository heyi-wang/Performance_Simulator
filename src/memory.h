#pragma once

#include "common.h"
#include <tlm_utils/simple_target_socket.h>
#include <tlm_utils/peq_with_get.h>
#include <deque>

enum class MemoryAccessKind
{
    Dma,
    L1,
};

struct MemoryAccessExt : tlm_extension<MemoryAccessExt>
{
    MemoryAccessKind kind = MemoryAccessKind::Dma;

    MemoryAccessExt() = default;
    explicit MemoryAccessExt(MemoryAccessKind kind_) : kind(kind_) {}

    tlm_extension_base *clone() const override
    {
        return new MemoryAccessExt(*this);
    }

    void copy_from(const tlm_extension_base &other) override
    {
        *this = static_cast<const MemoryAccessExt &>(other);
    }
};

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
    void enqueue_request(tlm_generic_payload &gp);
};

// ============================================================
// L1L2Memory - classified two-engine memory timing model.
//   MemoryAccessKind::L1  => accelerator read/write from/to L1.
//   MemoryAccessKind::Dma => DMA transfer between L2 and L1.
// ============================================================
struct L1L2Memory : sc_module
{
    tlm_utils::simple_target_socket<L1L2Memory> tgt;

    uint64_t l1_base_lat_cycles;
    uint64_t l1_bytes_per_cycle;
    uint64_t l1_parallel_slots;
    uint64_t l1_active_reqs = 0;

    uint64_t dma_base_lat_cycles;
    uint64_t dma_bytes_per_cycle;
    uint64_t dma_parallel_slots;
    uint64_t dma_active_reqs = 0;

    tlm_utils::peq_with_get<tlm_generic_payload> req_peq;
    tlm_utils::peq_with_get<tlm_generic_payload> resp_peq;

    struct Entry
    {
        tlm_generic_payload *gp = nullptr;
        sc_time              enqueue_time;
    };

    std::deque<Entry> l1_q;
    std::deque<Entry> dma_q;
    sc_event          l1_dispatch_ev;
    sc_event          dma_dispatch_ev;

    uint64_t l1_reqs = 0;
    uint64_t l1_read_reqs = 0;
    uint64_t l1_write_reqs = 0;
    uint64_t l1_read_bytes = 0;
    uint64_t l1_write_bytes = 0;
    uint64_t l1_busy_cycles = 0;
    uint64_t l1_qwait_cycles = 0;

    uint64_t dma_reqs = 0;
    uint64_t dma_read_reqs = 0;
    uint64_t dma_write_reqs = 0;
    uint64_t dma_read_bytes = 0;
    uint64_t dma_write_bytes = 0;
    uint64_t dma_busy_cycles = 0;
    uint64_t dma_qwait_cycles = 0;

    SC_HAS_PROCESS(L1L2Memory);

    L1L2Memory(sc_module_name name,
               uint64_t l1_base_lat_cycles_ = 1,
               uint64_t l1_bytes_per_cycle_ = 256,
               uint64_t dma_base_lat_cycles_ = 1,
               uint64_t dma_bytes_per_cycle_ = 128,
               uint64_t l1_parallel_slots_ = 1,
               uint64_t dma_parallel_slots_ = 1);

    tlm_sync_enum nb_transport_fw(tlm_generic_payload &gp,
                                  tlm_phase &phase,
                                  sc_time &delay);

    void peq_thread();
    void l1_dispatch_thread();
    void dma_dispatch_thread();
    void response_thread();
    void enqueue_request(tlm_generic_payload &gp);

private:
    static MemoryAccessKind access_kind(tlm_generic_payload &gp);
    static uint64_t transfer_cycles(uint64_t bytes,
                                    uint64_t base_lat_cycles,
                                    uint64_t bytes_per_cycle);
};
