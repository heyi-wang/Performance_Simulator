#include "memory.h"

// Required so SC_THREAD macro resolves SC_CURRENT_USER_MODULE outside class scope
SC_HAS_PROCESS(Memory);

Memory::Memory(sc_module_name name,
               uint64_t base_lat_cycles_,
               uint64_t bytes_per_cycle_,
               uint64_t parallel_slots_)
    : sc_module(name),
      tgt("tgt"),
      base_lat_cycles(base_lat_cycles_),
      bytes_per_cycle(bytes_per_cycle_),
      parallel_slots(parallel_slots_ == 0 ? 1 : parallel_slots_),
      req_peq("req_peq"),
      resp_peq("resp_peq")
{
    tgt.register_nb_transport_fw(this, &Memory::nb_transport_fw);
    SC_THREAD(peq_thread);
    SC_THREAD(dispatch_thread);
    SC_THREAD(response_thread);
}

tlm_sync_enum Memory::nb_transport_fw(tlm_generic_payload &gp,
                                      tlm_phase &phase,
                                      sc_time &delay)
{
    if (phase == BEGIN_REQ)
    {
        req_peq.notify(gp, delay);
        phase = END_REQ;
        delay = SC_ZERO_TIME;
        return TLM_UPDATED;
    }
    return TLM_ACCEPTED;
}

void Memory::enqueue_request(tlm_generic_payload &gp)
{
    Entry e;
    e.gp = &gp;
    e.enqueue_time = sc_time_stamp();
    q.push_back(e);
    dispatch_ev.notify(SC_ZERO_TIME);
}

void Memory::peq_thread()
{
    while (true)
    {
        wait(req_peq.get_event());
        while (auto *gp = req_peq.get_next_transaction())
        {
            enqueue_request(*gp);
        }
    }
}

void Memory::dispatch_thread()
{
    while (true)
    {
        while (q.empty() || active_reqs >= parallel_slots)
            wait(dispatch_ev);

        while (!q.empty() && active_reqs < parallel_slots)
        {
            Entry e = q.front();
            q.pop_front();

            uint64_t bytes   = e.gp->get_data_length();
            sc_time  t_start = sc_time_stamp();
            uint64_t qwait   = (uint64_t)((t_start - e.enqueue_time) / CYCLE);
            qwait_cycles += qwait;

            uint64_t xfer = (bytes_per_cycle == 0)
                                ? 0
                                : ceil_div_u64(bytes, bytes_per_cycle);
            uint64_t mem_lat = base_lat_cycles + xfer;

            reqs += 1;
            busy_cycles += mem_lat;
            active_reqs += 1;
            resp_peq.notify(*e.gp, mem_lat * CYCLE);
        }
    }
}

void Memory::response_thread()
{
    while (true)
    {
        wait(resp_peq.get_event());
        while (auto *gp = resp_peq.get_next_transaction())
        {
            gp->set_response_status(TLM_OK_RESPONSE);

            tlm_phase bw_phase = BEGIN_RESP;
            sc_time   bw_delay = SC_ZERO_TIME;
            tgt->nb_transport_bw(*gp, bw_phase, bw_delay);

            active_reqs -= 1;
            dispatch_ev.notify(SC_ZERO_TIME);
        }
    }
}

L1L2Memory::L1L2Memory(sc_module_name name,
                       uint64_t l1_base_lat_cycles_,
                       uint64_t l1_bytes_per_cycle_,
                       uint64_t dma_base_lat_cycles_,
                       uint64_t dma_bytes_per_cycle_,
                       uint64_t l1_parallel_slots_,
                       uint64_t dma_parallel_slots_)
    : sc_module(name),
      tgt("tgt"),
      l1_base_lat_cycles(l1_base_lat_cycles_),
      l1_bytes_per_cycle(l1_bytes_per_cycle_),
      l1_parallel_slots(l1_parallel_slots_ == 0 ? 1 : l1_parallel_slots_),
      dma_base_lat_cycles(dma_base_lat_cycles_),
      dma_bytes_per_cycle(dma_bytes_per_cycle_),
      dma_parallel_slots(dma_parallel_slots_ == 0 ? 1 : dma_parallel_slots_),
      req_peq("req_peq"),
      resp_peq("resp_peq")
{
    tgt.register_nb_transport_fw(this, &L1L2Memory::nb_transport_fw);
    SC_THREAD(peq_thread);
    SC_THREAD(l1_dispatch_thread);
    SC_THREAD(dma_dispatch_thread);
    SC_THREAD(response_thread);
}

tlm_sync_enum L1L2Memory::nb_transport_fw(tlm_generic_payload &gp,
                                          tlm_phase &phase,
                                          sc_time &delay)
{
    if (phase == BEGIN_REQ)
    {
        req_peq.notify(gp, delay);
        phase = END_REQ;
        delay = SC_ZERO_TIME;
        return TLM_UPDATED;
    }
    return TLM_ACCEPTED;
}

MemoryAccessKind L1L2Memory::access_kind(tlm_generic_payload &gp)
{
    MemoryAccessExt *ext = nullptr;
    gp.get_extension(ext);
    return ext ? ext->kind : MemoryAccessKind::Dma;
}

uint64_t L1L2Memory::transfer_cycles(uint64_t bytes,
                                     uint64_t base_lat_cycles,
                                     uint64_t bytes_per_cycle)
{
    uint64_t xfer = (bytes_per_cycle == 0)
                        ? 0
                        : ceil_div_u64(bytes, bytes_per_cycle);
    return base_lat_cycles + xfer;
}

void L1L2Memory::enqueue_request(tlm_generic_payload &gp)
{
    Entry e;
    e.gp = &gp;
    e.enqueue_time = sc_time_stamp();

    if (access_kind(gp) == MemoryAccessKind::L1)
    {
        l1_q.push_back(e);
        l1_dispatch_ev.notify(SC_ZERO_TIME);
    }
    else
    {
        dma_q.push_back(e);
        dma_dispatch_ev.notify(SC_ZERO_TIME);
    }
}

void L1L2Memory::peq_thread()
{
    while (true)
    {
        wait(req_peq.get_event());
        while (auto *gp = req_peq.get_next_transaction())
        {
            enqueue_request(*gp);
        }
    }
}

void L1L2Memory::l1_dispatch_thread()
{
    while (true)
    {
        while (l1_q.empty() || l1_active_reqs >= l1_parallel_slots)
            wait(l1_dispatch_ev);

        while (!l1_q.empty() && l1_active_reqs < l1_parallel_slots)
        {
            Entry e = l1_q.front();
            l1_q.pop_front();

            const uint64_t bytes = e.gp->get_data_length();
            const uint64_t qwait =
                static_cast<uint64_t>((sc_time_stamp() - e.enqueue_time) / CYCLE);
            const uint64_t lat =
                transfer_cycles(bytes, l1_base_lat_cycles, l1_bytes_per_cycle);

            l1_qwait_cycles += qwait;
            l1_busy_cycles += lat;
            l1_reqs += 1;
            if (e.gp->is_read())
            {
                l1_read_reqs += 1;
                l1_read_bytes += bytes;
            }
            else if (e.gp->is_write())
            {
                l1_write_reqs += 1;
                l1_write_bytes += bytes;
            }

            l1_active_reqs += 1;
            resp_peq.notify(*e.gp, lat * CYCLE);
        }
    }
}

void L1L2Memory::dma_dispatch_thread()
{
    while (true)
    {
        while (dma_q.empty() || dma_active_reqs >= dma_parallel_slots)
            wait(dma_dispatch_ev);

        while (!dma_q.empty() && dma_active_reqs < dma_parallel_slots)
        {
            Entry e = dma_q.front();
            dma_q.pop_front();

            const uint64_t bytes = e.gp->get_data_length();
            const uint64_t qwait =
                static_cast<uint64_t>((sc_time_stamp() - e.enqueue_time) / CYCLE);
            const uint64_t lat =
                transfer_cycles(bytes, dma_base_lat_cycles, dma_bytes_per_cycle);

            dma_qwait_cycles += qwait;
            dma_busy_cycles += lat;
            dma_reqs += 1;
            if (e.gp->is_read())
            {
                dma_read_reqs += 1;
                dma_read_bytes += bytes;
            }
            else if (e.gp->is_write())
            {
                dma_write_reqs += 1;
                dma_write_bytes += bytes;
            }

            dma_active_reqs += 1;
            resp_peq.notify(*e.gp, lat * CYCLE);
        }
    }
}

void L1L2Memory::response_thread()
{
    while (true)
    {
        wait(resp_peq.get_event());
        while (auto *gp = resp_peq.get_next_transaction())
        {
            gp->set_response_status(TLM_OK_RESPONSE);

            tlm_phase bw_phase = BEGIN_RESP;
            sc_time bw_delay = SC_ZERO_TIME;
            tgt->nb_transport_bw(*gp, bw_phase, bw_delay);

            MemoryAccessKind kind = access_kind(*gp);

            if (kind == MemoryAccessKind::L1)
            {
                if (l1_active_reqs > 0)
                    l1_active_reqs -= 1;
                l1_dispatch_ev.notify(SC_ZERO_TIME);
            }
            else
            {
                if (dma_active_reqs > 0)
                    dma_active_reqs -= 1;
                dma_dispatch_ev.notify(SC_ZERO_TIME);
            }
        }
    }
}
