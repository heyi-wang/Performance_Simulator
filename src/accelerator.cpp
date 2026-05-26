#include "accelerator.h"
#include "interconnect.h"   // for Interconnect::ADDR_MEM
#include "memory.h"
#include "perfetto_trace.h"

#ifdef PERFETTO_TRACE
#include <string>

// Collapse a per-layer hierarchical accelerator name (e.g.
// "nafblock_top.nb_matmul_top_2.mat_acc.accel_unit_0") into a stable per-unit
// group label shared across layers: "Matrix Unit 0" / "Vector Unit 1". This
// keeps the Perfetto trace to a fixed set of unit groups instead of one per
// layer instance.
static std::string accel_unit_group(const std::string &nm)
{
    const std::string cls =
        (nm.find(".mat_acc") != std::string::npos) ? "Matrix Unit" :
        (nm.find(".vec_acc") != std::string::npos) ? "Vector Unit" : "Accel Unit";
    const std::string key = "accel_unit_";
    const size_t p = nm.rfind(key);
    const std::string idx =
        (p != std::string::npos) ? nm.substr(p + key.size()) : "";
    return idx.empty() ? cls : cls + " " + idx;
}

// Emit one serviced request as load/compute/write lane spans on the unit's
// group, plus a "stall" (idle) span for the gap since the previous request.
static void perf_emit_service(const std::string &grp,
                              sc_time &last_busy_end,
                              sc_time t_load_start,    sc_time t_load_end,
                              sc_time t_compute_start, sc_time t_compute_end,
                              sc_time t_write_start,   sc_time t_write_end)
{
    if (last_busy_end > SC_ZERO_TIME && t_load_start > last_busy_end)
        PERF_TRACE_SPAN(grp, "stall", "stall",
                        static_cast<uint64_t>(last_busy_end / CYCLE),
                        static_cast<uint64_t>((t_load_start - last_busy_end) / CYCLE));
    PERF_TRACE_SPAN(grp, "load", "load",
                    static_cast<uint64_t>(t_load_start / CYCLE),
                    static_cast<uint64_t>((t_load_end - t_load_start) / CYCLE));
    PERF_TRACE_SPAN(grp, "compute", "compute",
                    static_cast<uint64_t>(t_compute_start / CYCLE),
                    static_cast<uint64_t>((t_compute_end - t_compute_start) / CYCLE));
    PERF_TRACE_SPAN(grp, "write", "write",
                    static_cast<uint64_t>(t_write_start / CYCLE),
                    static_cast<uint64_t>((t_write_end - t_write_start) / CYCLE));
    last_busy_end = t_write_end;
}
#endif

SC_HAS_PROCESS(AcceleratorTLM);

AcceleratorTLM::AcceleratorTLM(sc_module_name name, size_t cap, bool enable_pipeline)
    : sc_module(name),
      tgt("tgt"),
      to_mem("to_mem"),
      peq("peq"),
      pipeline_enabled(enable_pipeline),
      queue_capacity(cap)
{
    tgt.register_nb_transport_fw(this, &AcceleratorTLM::nb_transport_fw);
    to_mem.register_nb_transport_bw(this, &AcceleratorTLM::nb_transport_bw_mem);
    SC_THREAD(peq_thread);
    if (pipeline_enabled)
    {
        SC_THREAD(load_thread);
        SC_THREAD(compute_thread);
        SC_THREAD(write_thread);
    }
    else
    {
        SC_THREAD(service_thread);
    }
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

#ifdef PERFETTO_TRACE
        // Serial mode: load = read, compute = svc wait, write = write-back.
        if (perf_trace_enabled())
            perf_emit_service(accel_unit_group(name()), perf_last_busy_end,
                              m0, m1, m1, m2, m2, m3);
#endif

        complete_request(e);
    }
}

void AcceleratorTLM::complete_request(Entry &e)
{
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

// ============================================================
// Pipelined-mode stage threads
//   load_thread   : pop from q,        mem_access(read,  rd_bytes),  push loaded_q
//   compute_thread: pop from loaded_q, wait(svc cycles),             push computed_q
//   write_thread  : pop from computed_q, mem_access(write, wr_bytes), BEGIN_RESP
// Inter-stage queues are capacity-1, so each producer waits when the
// downstream queue is full -- this gives "exactly one tile ahead".
// ============================================================

void AcceleratorTLM::stage_enter()
{
    if (pipeline_active_stages == 0)
        pipeline_busy_start = sc_time_stamp();
    ++pipeline_active_stages;
}

void AcceleratorTLM::stage_exit()
{
    --pipeline_active_stages;
    if (pipeline_active_stages == 0)
    {
        occupied_cycles +=
            static_cast<uint64_t>((sc_time_stamp() - pipeline_busy_start) / CYCLE);
    }
}

void AcceleratorTLM::load_thread()
{
    while (true)
    {
        while (q.empty() || !loaded_q.empty())
            wait(q_nonempty | loaded_q_changed);

        Entry e = q.front();
        q.pop_front();

        ReqExt *ext = nullptr;
        e.gp->get_extension(ext);

        e.t_load_start = sc_time_stamp();
        uint64_t qwait =
            static_cast<uint64_t>((e.t_load_start - e.enqueue_time) / CYCLE);
        if (ext)
            ext->accel_qwait_cycles += qwait;
        queue_wait_cycles += qwait;
        req_count         += 1;

        stage_enter();
        if (busy_cb)
            busy_cb(static_cast<uint64_t>(sc_time_stamp() / CYCLE), true);

        mem_access(false, ext ? ext->rd_bytes : 0);
        e.t_load_end = sc_time_stamp();

        loaded_q.push_back(e);
        loaded_q_changed.notify(SC_ZERO_TIME);
    }
}

void AcceleratorTLM::compute_thread()
{
    while (true)
    {
        while (loaded_q.empty() || !computed_q.empty())
            wait(loaded_q_changed | computed_q_changed);

        Entry e = loaded_q.front();
        loaded_q.pop_front();
        loaded_q_changed.notify(SC_ZERO_TIME);

        ReqExt *ext = nullptr;
        e.gp->get_extension(ext);
        uint64_t svc = ext ? ext->cycles : 0;

        e.t_compute_start = sc_time_stamp();
        busy_cycles += svc;
        if (svc > 0)
            wait(svc * CYCLE);
        e.t_compute_end = sc_time_stamp();

        computed_q.push_back(e);
        computed_q_changed.notify(SC_ZERO_TIME);
    }
}

void AcceleratorTLM::write_thread()
{
    while (true)
    {
        while (computed_q.empty())
            wait(computed_q_changed);

        Entry e = computed_q.front();
        computed_q.pop_front();
        computed_q_changed.notify(SC_ZERO_TIME);

        ReqExt *ext = nullptr;
        e.gp->get_extension(ext);

        e.t_write_start = sc_time_stamp();
        mem_access(true, ext ? ext->wr_bytes : 0);
        e.t_write_end = sc_time_stamp();

        if (ext)
            ext->mem_cycles +=
                static_cast<uint64_t>(((e.t_load_end - e.t_load_start) +
                                       (e.t_write_end - e.t_write_start)) / CYCLE);

        if (busy_cb)
            busy_cb(static_cast<uint64_t>(sc_time_stamp() / CYCLE), false);
        stage_exit();

#ifdef PERFETTO_TRACE
        if (perf_trace_enabled())
            perf_emit_service(accel_unit_group(name()), perf_last_busy_end,
                              e.t_load_start,    e.t_load_end,
                              e.t_compute_start, e.t_compute_end,
                              e.t_write_start,   e.t_write_end);
#endif

        complete_request(e);
    }
}
