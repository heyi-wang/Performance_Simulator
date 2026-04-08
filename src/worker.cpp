#include "worker.h"
#include "interconnect.h"   // for Interconnect::ADDR_MAT / ADDR_VEC
#include <iostream>

SC_HAS_PROCESS(Worker);

Worker::Worker(sc_module_name name,
               int      tid_,
               uint64_t access_mat_,
               uint64_t access_vec_,
               uint64_t mat_cycles_,
               uint64_t vec_cycles_,
               uint64_t scalar_cycles_,
               uint64_t A_bytes_,
               uint64_t B_bytes_,
               uint64_t C_bytes_,
               uint64_t vec_rd_,
               uint64_t vec_wr_,
               uint64_t max_inflight_mat_reqs_,
               uint64_t max_inflight_vec_reqs_,
               WorkerPostProcessor *post_processor_,
               sc_event *start_event_,
               sc_fifo<int> *completion_fifo_)
    : sc_module(name),
      init("init"),
      peq("peq"),
      tid(tid_),
      m(0),
      access_mat(access_mat_),
      access_vec(access_vec_),
      mat_cycles(mat_cycles_),
      vec_cycles(vec_cycles_),
      scalar_cycles(scalar_cycles_),
      max_inflight_mat_reqs(std::max<uint64_t>(max_inflight_mat_reqs_, 1)),
      max_inflight_vec_reqs(std::max<uint64_t>(max_inflight_vec_reqs_, 1)),
      post_processor(post_processor_),
      start_event(start_event_),
      completion_fifo(completion_fifo_),
      A_bytes(A_bytes_),
      B_bytes(B_bytes_),
      C_bytes(C_bytes_),
      vec_rd_bytes(vec_rd_),
      vec_wr_bytes(vec_wr_)
{
    init.register_nb_transport_bw(this, &Worker::nb_transport_bw);
    SC_THREAD(peq_thread);
    SC_THREAD(run);
}

// ----------------------------------------------------------
// nb_transport_bw: receive BEGIN_RESP or deferred END_REQ
//
//   BEGIN_RESP — request completed; route through PEQ so
//                peq_thread can set fired and notify ev.
//   END_REQ    — accelerator queue was full when the request
//                was sent; now a slot is granted. Notify
//                admit_ev so issue_begin can unblock.
// ----------------------------------------------------------
tlm_sync_enum Worker::nb_transport_bw(tlm_generic_payload &gp,
                                      tlm_phase &phase,
                                      sc_time &delay)
{
    if (phase == BEGIN_RESP)
    {
        peq.notify(gp, delay);
        return TLM_ACCEPTED;
    }
    if (phase == END_REQ)
    {
        auto it = done_map.find(&gp);
        if (it != done_map.end() && it->second && it->second->admit_ev)
            it->second->admit_ev->notify(SC_ZERO_TIME);
        return TLM_ACCEPTED;
    }
    return TLM_ACCEPTED;
}

// ----------------------------------------------------------
// peq_thread: wake up done events when responses arrive.
// Always set fired=true BEFORE notifying so issue_end can
// detect a response that arrived before wait() was called.
// ----------------------------------------------------------
void Worker::peq_thread()
{
    while (true)
    {
        wait(peq.get_event());
        while (auto *gp = peq.get_next_transaction())
        {
            auto it = done_map.find(gp);
            if (it != done_map.end() && it->second)
            {
                it->second->fired = true;
                it->second->ev->notify(SC_ZERO_TIME);
            }
        }
    }
}

void Worker::do_scalar(uint64_t cyc)
{
    compute_cycles += cyc;
    wait(cyc * CYCLE);
}

// ----------------------------------------------------------
// issue_begin (explicit bytes): fire a non-blocking request.
// If the accelerator queue is full (TLM_ACCEPTED), block on
// admit_ev until a slot is granted via deferred END_REQ.
// ----------------------------------------------------------
Worker::PendingReq Worker::issue_begin(uint64_t addr,
                                       uint64_t svc_cycles,
                                       uint64_t rd,
                                       uint64_t wr)
{
    PendingReq p;
    p.svc_cycles = svc_cycles;

    auto *gp = new tlm_generic_payload();
    gp->set_command(TLM_IGNORE_COMMAND);
    gp->set_address(addr);
    gp->set_data_ptr(nullptr);
    gp->set_data_length(0);
    gp->set_streaming_width(0);
    gp->set_response_status(TLM_INCOMPLETE_RESPONSE);

    auto *req = new ReqExt(tid, svc_cycles, rd, wr);
    auto *tx  = new TxnExt();
    tx->src_worker = tid;

    gp->set_extension(req);
    gp->set_extension(tx);

    p.gp         = gp;
    p.req_ext    = req;
    p.tx_ext     = tx;
    p.done_entry = new DoneEntry();
    p.done_entry->ev       = new sc_event();
    p.done_entry->admit_ev = new sc_event();
    p.done_entry->fired    = false;
    done_map[gp] = p.done_entry;

    tlm_phase phase = BEGIN_REQ;
    sc_time   delay = SC_ZERO_TIME;
    auto status = init->nb_transport_fw(*gp, phase, delay);

    if (status == TLM_ACCEPTED)
    {
        // Queue was full: stall until the accelerator grants a slot.
        sc_time t_stall_start = sc_time_stamp();
        wait(*p.done_entry->admit_ev);
        p.stall_cycles = (uint64_t)((sc_time_stamp() - t_stall_start) / CYCLE);
    }
    else if (status == TLM_COMPLETED)
    {
        done_map.erase(gp);
        p.sync_done = true;
    }
    // TLM_UPDATED: slot granted immediately, no stall needed.

    return p;
}

// Convenience overload: use mat-request bytes (A+B reads, C writes)
Worker::PendingReq Worker::issue_begin(uint64_t addr, uint64_t svc_cycles)
{
    return issue_begin(addr, svc_cycles, A_bytes + B_bytes, C_bytes);
}

void Worker::issue_end(PendingReq &p)
{
    // If fired=true the response arrived before we got here; skip wait().
    if (!p.sync_done && !p.done_entry->fired)
        wait(*p.done_entry->ev);

    done_map.erase(p.gp);
    delete p.done_entry->ev;
    delete p.done_entry->admit_ev;
    delete p.done_entry;
    p.done_entry = nullptr;

    ReqExt *ext = nullptr;
    p.gp->get_extension(ext);

    uint64_t qwait = ext ? ext->accel_qwait_cycles : 0;
    wait_cycles      += qwait + p.stall_cycles;
    stall_cycles     += p.stall_cycles;
    compute_cycles   += p.svc_cycles;
    mem_cycles_accum += ext ? ext->mem_cycles : 0;

    tlm_phase end_phase = END_RESP;
    sc_time   end_delay = SC_ZERO_TIME;
    init->nb_transport_fw(*p.gp, end_phase, end_delay);

    p.gp->clear_extension(p.req_ext);
    p.gp->clear_extension(p.tx_ext);
    delete p.req_ext;
    delete p.tx_ext;
    delete p.gp;

    p.gp      = nullptr;
    p.req_ext = nullptr;
    p.tx_ext  = nullptr;
}

void Worker::issue_stream(uint64_t addr,
                          uint64_t call_count,
                          uint64_t svc_cycles,
                          uint64_t rd,
                          uint64_t wr,
                          uint64_t &call_counter,
                          uint64_t *phase_counter,
                          uint64_t max_inflight)
{
    if (call_count == 0)
        return;

    std::deque<PendingReq> inflight;
    uint64_t issued = 0;
    uint64_t window = std::max<uint64_t>(max_inflight, 1);

    auto issue_one = [&]() {
        inflight.push_back(issue_begin(addr, svc_cycles, rd, wr));
        ++call_counter;
        if (phase_counter)
            ++(*phase_counter);
        ++issued;
    };

    auto retire_one = [&]() {
        bool has_completed = false;
        for (auto &pending : inflight)
        {
            if (pending.sync_done || (pending.done_entry && pending.done_entry->fired))
            {
                has_completed = true;
                break;
            }
        }

        if (!has_completed)
        {
            sc_event_or_list wait_list;
            for (auto &pending : inflight)
            {
                if (!pending.sync_done && pending.done_entry && !pending.done_entry->fired)
                    wait_list |= *pending.done_entry->ev;
            }
            wait(wait_list);
        }

        for (auto it = inflight.begin(); it != inflight.end(); ++it)
        {
            if (it->sync_done || (it->done_entry && it->done_entry->fired))
            {
                issue_end(*it);
                inflight.erase(it);
                return;
            }
        }

        issue_end(inflight.front());
        inflight.pop_front();
    };

    issue_one();

    while (issued < call_count)
    {
        do_scalar(scalar_cycles);

        while (inflight.size() >= window)
            retire_one();

        issue_one();
    }

    while (!inflight.empty())
        retire_one();
}

void Worker::run()
{
    if (start_event)
        wait(*start_event);

    sc_time start = sc_time_stamp();

    // ----------------------------------------------------------
    // Phase 1: Matrix multiply (mat accelerator)
    // Pipeline: issue → scalar (overlapped) → issue_end → issue...
    // ----------------------------------------------------------
    if (access_mat > 0)
        issue_stream(Interconnect::ADDR_MAT,
                     access_mat,
                     mat_cycles,
                     A_bytes + B_bytes,
                     C_bytes,
                     mat_calls,
                     nullptr,
                     max_inflight_mat_reqs);

    if (post_processor)
    {
        // In the matmul simulator, the local mat phase completion is the handoff
        // point into worker-driven reduction / final quantization.
        mat_done_time = sc_time_stamp();
        mat_elapsed_cycles = (uint64_t)((mat_done_time - start) / CYCLE);
        mat_done_ev.notify(SC_ZERO_TIME);
        post_processor->run_post_mat(*this);
    }
    else
    {
        // ------------------------------------------------------
        // Phase 2: Output quantization (vec accelerator)
        // Reads fp32 partial result, writes fp16 quantized result.
        // vec_rd_bytes / vec_wr_bytes are set by the caller;
        // fall back to mat bytes if not explicitly configured.
        // ------------------------------------------------------
        if (access_vec > 0)
        {
            uint64_t vrd = (vec_rd_bytes > 0) ? vec_rd_bytes : (A_bytes + B_bytes);
            uint64_t vwr = (vec_wr_bytes > 0) ? vec_wr_bytes : C_bytes;
            issue_stream(Interconnect::ADDR_VEC,
                         access_vec,
                         vec_cycles,
                         vrd,
                         vwr,
                         vec_calls,
                         nullptr,
                         max_inflight_vec_reqs);
        }

        // Preserve the legacy behavior for the base simulator: mat_done marks
        // completion after both mat and worker-local vector phases.
        mat_done_time = sc_time_stamp();
        mat_elapsed_cycles = (uint64_t)((mat_done_time - start) / CYCLE);
        mat_done_ev.notify(SC_ZERO_TIME);
    }

    sc_time end        = sc_time_stamp();
    elapsed_cycles     = (uint64_t)((end - start) / CYCLE);
    if (completion_fifo)
        completion_fifo->write(tid);
}
