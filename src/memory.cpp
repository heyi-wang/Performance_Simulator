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

void Memory::peq_thread()
{
    while (true)
    {
        wait(req_peq.get_event());
        while (auto *gp = req_peq.get_next_transaction())
        {
            Entry e;
            e.gp           = gp;
            e.enqueue_time = sc_time_stamp();
            q.push_back(e);
            dispatch_ev.notify(SC_ZERO_TIME);
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
