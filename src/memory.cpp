#include "memory.h"

// Required so SC_THREAD macro resolves SC_CURRENT_USER_MODULE outside class scope
SC_HAS_PROCESS(Memory);

Memory::Memory(sc_module_name name,
               uint64_t base_lat_cycles_,
               uint64_t bytes_per_cycle_)
    : sc_module(name),
      tgt("tgt"),
      base_lat_cycles(base_lat_cycles_),
      bytes_per_cycle(bytes_per_cycle_),
      peq("peq")
{
    tgt.register_nb_transport_fw(this, &Memory::nb_transport_fw);
    SC_THREAD(peq_thread);
    SC_THREAD(service_thread);
}

tlm_sync_enum Memory::nb_transport_fw(tlm_generic_payload &gp,
                                      tlm_phase &phase,
                                      sc_time &delay)
{
    if (phase == BEGIN_REQ)
    {
        peq.notify(gp, delay);
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
        wait(peq.get_event());
        while (auto *gp = peq.get_next_transaction())
        {
            Entry e;
            e.gp           = gp;
            e.enqueue_time = sc_time_stamp();
            q.push_back(e);
            q_nonempty.notify(SC_ZERO_TIME);
        }
    }
}

void Memory::service_thread()
{
    while (true)
    {
        while (q.empty())
            wait(q_nonempty);

        Entry    e      = q.front();
        q.pop_front();

        uint64_t bytes   = e.gp->get_data_length();
        sc_time  t_start = sc_time_stamp();
        uint64_t qwait   = (uint64_t)((t_start - e.enqueue_time) / CYCLE);
        qwait_cycles += qwait;

        uint64_t xfer      = (bytes_per_cycle == 0)
                                 ? 0
                                 : ceil_div_u64(bytes, bytes_per_cycle);
        uint64_t mem_lat   = base_lat_cycles + xfer;

        reqs        += 1;
        busy_cycles += mem_lat;
        wait(mem_lat * CYCLE);

        e.gp->set_response_status(TLM_OK_RESPONSE);

        tlm_phase bw_phase = BEGIN_RESP;
        sc_time   bw_delay = SC_ZERO_TIME;
        tgt->nb_transport_bw(*e.gp, bw_phase, bw_delay);
    }
}
