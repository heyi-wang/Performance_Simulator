#include "accelerator.h"
#include "interconnect.h"   // for Interconnect::ADDR_MEM

SC_HAS_PROCESS(AcceleratorTLM);

AcceleratorTLM::AcceleratorTLM(sc_module_name name, size_t cap)
    : sc_module(name),
      tgt("tgt"),
      to_mem("to_mem"),
      peq("peq"),
      queue_capacity(cap)
{
    tgt.register_nb_transport_fw(this, &AcceleratorTLM::nb_transport_fw);
    to_mem.register_nb_transport_bw(this, &AcceleratorTLM::nb_transport_bw_mem);
    SC_THREAD(peq_thread);
    SC_THREAD(service_thread);
}

tlm_sync_enum AcceleratorTLM::nb_transport_fw(tlm_generic_payload &gp,
                                              tlm_phase &phase,
                                              sc_time &delay)
{
    if (phase == BEGIN_REQ)
    {
        if (admitted < queue_capacity)
        {
            // Slot available: admit immediately through the PEQ.
            ++admitted;
            peq.notify(gp, delay);
            phase = END_REQ;
            delay = SC_ZERO_TIME;
            return TLM_UPDATED;
        }
        else
        {
            // Queue full: park the GP and stall the worker.
            // END_REQ will be sent back via nb_transport_bw once a slot opens.
            stall_fifo.push_back(&gp);
            return TLM_ACCEPTED;
        }
    }
    return TLM_ACCEPTED;
}

tlm_sync_enum AcceleratorTLM::nb_transport_bw_mem(tlm_generic_payload &gp,
                                                  tlm_phase &phase,
                                                  sc_time &delay)
{
    if (phase == BEGIN_RESP)
    {
        auto it = mem_done_map.find(&gp);
        if (it != mem_done_map.end() && it->second)
            it->second->notify(delay);

        phase = END_RESP;
        delay = SC_ZERO_TIME;
        return TLM_COMPLETED;
    }
    return TLM_ACCEPTED;
}

void AcceleratorTLM::mem_access(bool is_write, uint64_t bytes)
{
    if (bytes == 0)
        return;

    auto *gp = new tlm_generic_payload();
    gp->set_command(is_write ? TLM_WRITE_COMMAND : TLM_READ_COMMAND);
    gp->set_address(Interconnect::ADDR_MEM);
    gp->set_data_ptr(nullptr);
    gp->set_data_length((unsigned)bytes);
    gp->set_streaming_width((unsigned)bytes);
    gp->set_response_status(TLM_INCOMPLETE_RESPONSE);

    sc_event done_ev;
    mem_done_map[gp] = &done_ev;

    tlm_phase phase = BEGIN_REQ;
    sc_time   delay = SC_ZERO_TIME;
    auto status = to_mem->nb_transport_fw(*gp, phase, delay);

    if (status == TLM_COMPLETED)
    {
        mem_done_map.erase(gp);
        delete gp;
        return;
    }

    wait(done_ev);

    mem_done_map.erase(gp);
    delete gp;
}

void AcceleratorTLM::peq_thread()
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

void AcceleratorTLM::service_thread()
{
    while (true)
    {
        while (q.empty())
            wait(q_nonempty);

        Entry e = q.front();
        q.pop_front();

        ReqExt *ext = nullptr;
        e.gp->get_extension(ext);

        uint64_t svc    = ext ? ext->cycles : 0;
        sc_time  t_start = sc_time_stamp();
        uint64_t qwait  = (uint64_t)((t_start - e.enqueue_time) / CYCLE);

        if (ext)
            ext->accel_qwait_cycles = qwait;
        queue_wait_cycles += qwait;
        req_count         += 1;

        sc_time m0 = sc_time_stamp();
        mem_access(false, ext ? ext->rd_bytes : 0);
        mem_access(true,  ext ? ext->wr_bytes : 0);
        sc_time m1 = sc_time_stamp();

        if (ext)
            ext->mem_cycles = (uint64_t)((m1 - m0) / CYCLE);

        busy_cycles += svc;
        wait(svc * CYCLE);

        e.gp->set_response_status(TLM_OK_RESPONSE);

        // Release the admitted slot or hand it to the next stalled request.
        if (!stall_fifo.empty())
        {
            // Admit the oldest stalled GP: route it through the PEQ so that
            // peq_thread picks it up and enqueues it into q normally.
            tlm_generic_payload *next_gp = stall_fifo.front();
            stall_fifo.pop_front();
            sc_time zero = SC_ZERO_TIME;
            peq.notify(*next_gp, zero);

            // Send the deferred END_REQ back to the worker so that
            // issue_begin (which is blocked waiting for admit_ev) unblocks.
            tlm_phase end_req_phase = END_REQ;
            sc_time   end_req_delay = SC_ZERO_TIME;
            tgt->nb_transport_bw(*next_gp, end_req_phase, end_req_delay);
            // admitted stays the same: the slot is reused by next_gp.
        }
        else
        {
            --admitted;
        }

        tlm_phase phase = BEGIN_RESP;
        sc_time   delay = SC_ZERO_TIME;
        tgt->nb_transport_bw(*e.gp, phase, delay);
    }
}
