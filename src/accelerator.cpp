#include "accelerator.h"
#include "interconnect.h"   // for Interconnect::ADDR_MEM
#include "memory.h"

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
        TxnExt *tx = nullptr;
        gp.get_extension(tx);
        if (tx && tx->done_ev && tx->done_fired)
        {
            *tx->done_fired = true;
            tx->done_ev->notify(delay);
        }

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

    tlm_generic_payload gp;
    gp.set_command(is_write ? TLM_WRITE_COMMAND : TLM_READ_COMMAND);
    gp.set_address(Interconnect::ADDR_MEM);
    gp.set_data_ptr(nullptr);
    gp.set_data_length(static_cast<unsigned>(bytes));
    gp.set_streaming_width(static_cast<unsigned>(bytes));
    gp.set_response_status(TLM_INCOMPLETE_RESPONSE);

    MemoryAccessExt mem_ext(MemoryAccessKind::L1);
    gp.set_extension(&mem_ext);

    sc_event done_ev;
    bool done_fired = false;
    TxnExt tx;
    tx.done_ev = &done_ev;
    tx.done_fired = &done_fired;
    gp.set_extension(&tx);

    tlm_phase phase = BEGIN_REQ;
    sc_time   delay = SC_ZERO_TIME;
    auto status = to_mem->nb_transport_fw(gp, phase, delay);

    if (status == TLM_COMPLETED)
    {
        gp.clear_extension(&mem_ext);
        gp.clear_extension(&tx);
        return;
    }

    if (!done_fired)
        wait(done_ev);

    gp.clear_extension(&mem_ext);
    gp.clear_extension(&tx);
}

void AcceleratorTLM::enqueue_request(tlm_generic_payload &gp)
{
    Entry e;
    e.gp = &gp;
    e.enqueue_time = sc_time_stamp();
    q.push_back(e);
    q_nonempty.notify(SC_ZERO_TIME);
}

void AcceleratorTLM::peq_thread()
{
    while (true)
    {
        wait(peq.get_event());
        while (auto *gp = peq.get_next_transaction())
        {
            enqueue_request(*gp);
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
            ext->accel_qwait_cycles += qwait;
        queue_wait_cycles += qwait;
        req_count         += 1;

        // Signal busy start (includes memory access + compute wait)
        if (busy_cb)
            busy_cb((uint64_t)(sc_time_stamp() / CYCLE), true);

        sc_time m0 = sc_time_stamp();
        mem_access(false, ext ? ext->rd_bytes : 0);
        sc_time m1 = sc_time_stamp();

        busy_cycles += svc;
        wait(svc * CYCLE);

        sc_time m2 = sc_time_stamp();
        mem_access(true, ext ? ext->wr_bytes : 0);
        sc_time m3 = sc_time_stamp();

        if (ext)
            ext->mem_cycles =
                static_cast<uint64_t>(((m1 - m0) + (m3 - m2)) / CYCLE);

        occupied_cycles += (uint64_t)((sc_time_stamp() - t_start) / CYCLE);

        // Signal busy end (compute finished, about to send response)
        if (busy_cb)
            busy_cb((uint64_t)(sc_time_stamp() / CYCLE), false);

        e.gp->set_response_status(TLM_OK_RESPONSE);

        // Release the admitted slot or hand it to the next stalled request.
        if (!stall_fifo.empty())
        {
            tlm_generic_payload *next_gp = stall_fifo.front();
            stall_fifo.pop_front();
            peq.notify(*next_gp, SC_ZERO_TIME);

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
