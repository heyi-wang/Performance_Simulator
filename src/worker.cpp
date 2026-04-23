#include "worker.h"
#include "interconnect.h"   // for Interconnect::ADDR_MAT / ADDR_VEC
#include "memory.h"         // for MemoryAccessExt / MemoryAccessKind
#include <algorithm>
#include <iostream>

SC_HAS_PROCESS(Worker);

Worker::Worker(sc_module_name name,
               int      tid_,
               uint64_t access_mat_,
               uint64_t access_vec_,
               uint64_t mat_cycles_,
               uint64_t vec_cycles_,
               uint64_t mat_scalar_cycles_,
               uint64_t vec_scalar_cycles_,
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
      mat_scalar_cycles(mat_scalar_cycles_),
      vec_scalar_cycles(vec_scalar_cycles_),
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
    TxnExt *tx = nullptr;
    gp.get_extension(tx);

    if (phase == BEGIN_RESP)
    {
        peq.notify(gp, delay);
        return TLM_ACCEPTED;
    }
    if (phase == END_REQ)
    {
        if (tx && tx->admit_ev)
        {
            if (tx->admit_fired)
                *tx->admit_fired = true;
            tx->admit_ev->notify(SC_ZERO_TIME);
        }
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
            TxnExt *tx = nullptr;
            gp->get_extension(tx);
            if (tx && tx->done_ev && tx->done_fired)
            {
                *tx->done_fired = true;
                tx->done_ev->notify(SC_ZERO_TIME);
            }
        }
    }
}

void Worker::do_scalar(uint64_t cyc)
{
    compute_cycles += cyc;
    wait(cyc * CYCLE);
}

Worker::PendingReqStorage *Worker::acquire_pending_req_storage()
{
    PendingReqStorage *storage = nullptr;
    if (!free_pending_reqs.empty())
    {
        storage = free_pending_reqs.back();
        free_pending_reqs.pop_back();
    }
    else
    {
        pending_req_pool.emplace_back();
        storage = &pending_req_pool.back();
    }

    storage->in_use = true;
    storage->done_entry.fired = false;
    storage->done_entry.admit_fired = false;
    storage->req_ext = ReqExt();
    storage->tx_ext = TxnExt();
    return storage;
}

Worker::DmaReqStorage *Worker::acquire_dma_req_storage()
{
    DmaReqStorage *storage = nullptr;
    if (!free_dma_reqs.empty())
    {
        storage = free_dma_reqs.back();
        free_dma_reqs.pop_back();
    }
    else
    {
        dma_req_pool.emplace_back();
        storage = &dma_req_pool.back();
    }

    storage->in_use = true;
    storage->done_entry.fired = false;
    storage->done_entry.admit_fired = false;
    storage->mem_ext = MemoryAccessExt(MemoryAccessKind::Dma);
    storage->tx_ext = TxnExt();
    return storage;
}

void Worker::release_pending_req_storage(PendingReqStorage *storage)
{
    if (!storage)
        return;
    storage->in_use = false;
    free_pending_reqs.push_back(storage);
}

void Worker::release_dma_req_storage(DmaReqStorage *storage)
{
    if (!storage)
        return;
    storage->in_use = false;
    free_dma_reqs.push_back(storage);
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

    auto *storage = acquire_pending_req_storage();
    auto *gp = &storage->gp;
    gp->set_command(TLM_IGNORE_COMMAND);
    gp->set_address(addr);
    gp->set_data_ptr(nullptr);
    gp->set_data_length(0);
    gp->set_streaming_width(0);
    gp->set_response_status(TLM_INCOMPLETE_RESPONSE);

    auto *req = &storage->req_ext;
    *req = ReqExt(tid, svc_cycles, rd, wr);
    auto *tx = &storage->tx_ext;
    tx->src_worker = tid;

    gp->set_extension(req);
    gp->set_extension(tx);

    p.gp         = gp;
    p.req_ext    = req;
    p.tx_ext     = tx;
    p.done_entry = &storage->done_entry;
    p.storage    = storage;
    tx->done_ev = &p.done_entry->ev;
    tx->admit_ev = &p.done_entry->admit_ev;
    tx->done_fired = &p.done_entry->fired;
    tx->admit_fired = &p.done_entry->admit_fired;

    tlm_phase phase = BEGIN_REQ;
    sc_time   delay = SC_ZERO_TIME;
    auto status = init->nb_transport_fw(*gp, phase, delay);

    if (status == TLM_ACCEPTED)
    {
        // Queue was full: stall until the accelerator grants a slot.
        sc_time t_stall_start = sc_time_stamp();
        if (!p.done_entry->admit_fired)
            wait(p.done_entry->admit_ev);
        p.stall_cycles = (uint64_t)((sc_time_stamp() - t_stall_start) / CYCLE);
    }
    else if (status == TLM_COMPLETED)
    {
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
        wait(p.done_entry->ev);

    ReqExt *ext = nullptr;
    p.gp->get_extension(ext);

    uint64_t qwait = ext ? ext->accel_qwait_cycles : 0;
    wait_cycles      += qwait;
    stall_cycles     += p.stall_cycles;
    compute_cycles   += p.svc_cycles;
    mem_cycles_accum += ext ? ext->mem_cycles : 0;
    if (p.gp->get_address() == Interconnect::ADDR_VEC)
        vec_service_cycles += p.svc_cycles;

    tlm_phase end_phase = END_RESP;
    sc_time   end_delay = SC_ZERO_TIME;
    init->nb_transport_fw(*p.gp, end_phase, end_delay);

    p.gp->clear_extension(p.req_ext);
    p.gp->clear_extension(p.tx_ext);
    release_pending_req_storage(p.storage);

    p.gp      = nullptr;
    p.req_ext = nullptr;
    p.tx_ext  = nullptr;
    p.done_entry = nullptr;
    p.storage = nullptr;
}

Worker::DmaReq Worker::issue_dma_begin(bool is_write, uint64_t bytes)
{
    DmaReq p;
    if (bytes == 0)
        return p;

    auto *storage = acquire_dma_req_storage();
    auto *gp = &storage->gp;
    gp->set_command(is_write ? TLM_WRITE_COMMAND : TLM_READ_COMMAND);
    gp->set_address(Interconnect::ADDR_MEM);
    gp->set_data_ptr(nullptr);
    gp->set_data_length(static_cast<unsigned>(bytes));
    gp->set_streaming_width(static_cast<unsigned>(bytes));
    gp->set_response_status(TLM_INCOMPLETE_RESPONSE);

    auto *mem_ext = &storage->mem_ext;
    mem_ext->kind = MemoryAccessKind::Dma;
    gp->set_extension(mem_ext);

    auto *tx = &storage->tx_ext;
    tx->src_worker = tid;
    gp->set_extension(tx);

    auto *de = &storage->done_entry;
    tx->done_ev = &de->ev;
    tx->admit_ev = &de->admit_ev;
    tx->done_fired = &de->fired;
    tx->admit_fired = &de->admit_fired;

    p.gp         = gp;
    p.mem_ext    = mem_ext;
    p.tx_ext     = tx;
    p.done_entry = de;
    p.storage    = storage;
    p.sync_done  = false;

    tlm_phase phase = BEGIN_REQ;
    sc_time   delay = SC_ZERO_TIME;
    auto status = init->nb_transport_fw(*gp, phase, delay);

    if (status == TLM_COMPLETED)
    {
        p.sync_done = true;
    }

    return p;
}

void Worker::finish_dma(DmaReq &p)
{
    if (!p.gp)
        return;

    if (!p.sync_done && !p.done_entry->fired)
        wait(p.done_entry->ev);

    p.done_entry = nullptr;

    tlm_phase end_phase = END_RESP;
    sc_time   end_delay = SC_ZERO_TIME;
    init->nb_transport_fw(*p.gp, end_phase, end_delay);

    p.gp->clear_extension(p.mem_ext);
    p.gp->clear_extension(p.tx_ext);
    release_dma_req_storage(p.storage);

    p.gp      = nullptr;
    p.mem_ext = nullptr;
    p.tx_ext  = nullptr;
    p.storage = nullptr;
}

void Worker::configure_gemm_reuse(uint64_t m_tiles,
                                  uint64_t n_tiles,
                                  uint64_t k_tiles,
                                  uint64_t accumulator_registers)
{
    gemm_m_tiles = m_tiles;
    gemm_n_tiles = n_tiles;
    gemm_k_tiles = k_tiles;
    accumulator_register_count = std::max<uint64_t>(accumulator_registers, 1);
}

void Worker::configure_dma_row_cost(uint64_t a_rows,
                                    uint64_t b_rows,
                                    uint64_t c_rows,
                                    uint64_t a_row_cost,
                                    uint64_t b_row_cost,
                                    uint64_t c_row_cost)
{
    dma_a_rows       = a_rows;
    dma_b_rows       = b_rows;
    dma_c_rows       = c_rows;
    dma_a_row_scalar = a_row_cost;
    dma_b_row_scalar = b_row_cost;
    dma_c_row_scalar = c_row_cost;
}

void Worker::issue_stream(uint64_t addr,
                          uint64_t call_count,
                          uint64_t svc_cycles,
                          uint64_t scalar_cycles,
                          uint64_t rd,
                          uint64_t wr,
                          uint64_t dma_rd,
                          uint64_t dma_wr,
                          uint64_t &call_counter,
                          uint64_t *phase_counter,
                          uint64_t max_inflight,
                          bool scalar_between_streams)
{
    if (call_count == 0)
        return;

    std::deque<DmaReq>     read_inflight;
    std::deque<PendingReq> accel_inflight;
    std::deque<DmaReq>     write_inflight;
    uint64_t reads_issued = 0;
    uint64_t accel_issued = 0;
    uint64_t window = std::max<uint64_t>(max_inflight, 1);

    auto dma_done = [](const DmaReq &p) {
        return !p.gp || p.sync_done ||
               (p.done_entry && p.done_entry->fired);
    };

    auto accel_done = [](const PendingReq &p) {
        return p.sync_done ||
               (p.done_entry && p.done_entry->fired);
    };

    auto issue_read = [&]() {
        if (dma_rd > 0 && dma_a_rows > 0 && dma_a_row_scalar > 0)
            do_scalar(dma_a_rows * dma_a_row_scalar);
        read_inflight.push_back(issue_dma_begin(false, dma_rd));
        ++reads_issued;
    };

    auto issue_accel = [&]() {
        if (accel_issued > 0 || (scalar_between_streams && call_counter > 0))
            do_scalar(scalar_cycles);
        accel_inflight.push_back(issue_begin(addr, svc_cycles, rd, wr));
        ++call_counter;
        if (phase_counter)
            ++(*phase_counter);
        ++accel_issued;
    };

    auto fill_reads = [&]() {
        while (reads_issued < call_count && read_inflight.size() < window)
            issue_read();
    };

    auto promote_reads = [&]() {
        bool progressed = false;
        while (!read_inflight.empty() &&
               accel_inflight.size() < window &&
               dma_done(read_inflight.front()))
        {
            finish_dma(read_inflight.front());
            read_inflight.pop_front();
            issue_accel();
            progressed = true;
        }
        return progressed;
    };

    auto retire_accels = [&]() {
        bool progressed = false;
        for (auto it = accel_inflight.begin();
             it != accel_inflight.end() && write_inflight.size() < window;)
        {
            if (accel_done(*it))
            {
                issue_end(*it);
                it = accel_inflight.erase(it);
                if (dma_wr > 0 && dma_c_rows > 0 && dma_c_row_scalar > 0)
                    do_scalar(dma_c_rows * dma_c_row_scalar);
                write_inflight.push_back(issue_dma_begin(true, dma_wr));
                progressed = true;
            }
            else
            {
                ++it;
            }
        }
        return progressed;
    };

    auto retire_writes = [&](bool wait_for_front) {
        bool progressed = false;
        while (!write_inflight.empty() &&
               (dma_done(write_inflight.front()) || wait_for_front))
        {
            finish_dma(write_inflight.front());
            write_inflight.pop_front();
            progressed = true;
            wait_for_front = false;
        }
        return progressed;
    };

    auto wait_for_progress = [&]() {
        sc_event_or_list wait_list;
        bool has_event = false;

        auto add_dma_event = [&](const DmaReq &p) {
            if (p.gp && !p.sync_done && p.done_entry && !p.done_entry->fired)
            {
                wait_list |= p.done_entry->ev;
                has_event = true;
            }
        };

        auto add_accel_event = [&](const PendingReq &p) {
            if (!p.sync_done && p.done_entry && !p.done_entry->fired)
            {
                wait_list |= p.done_entry->ev;
                has_event = true;
            }
        };

        for (const auto &p : read_inflight)
            add_dma_event(p);
        for (const auto &p : accel_inflight)
            add_accel_event(p);
        for (const auto &p : write_inflight)
            add_dma_event(p);

        if (has_event)
            wait(wait_list);
    };

    fill_reads();

    while (accel_issued < call_count ||
           !read_inflight.empty() ||
           !accel_inflight.empty() ||
           !write_inflight.empty())
    {
        bool progressed = false;

        fill_reads();
        progressed = promote_reads() || progressed;
        fill_reads();
        progressed = retire_accels() || progressed;
        progressed = retire_writes(false) || progressed;

        if (write_inflight.size() >= window)
            progressed = retire_writes(true) || progressed;

        if (!progressed)
            wait_for_progress();
    }
}

void Worker::issue_gemm_reuse_stream()
{
    if (gemm_m_tiles == 0 || gemm_n_tiles == 0 || gemm_k_tiles == 0)
        return;

    const uint64_t regs = std::max<uint64_t>(accumulator_register_count, 1);

    for (uint64_t mg = 0; mg < gemm_m_tiles; mg += regs)
    {
        const uint64_t m_batch = std::min<uint64_t>(regs, gemm_m_tiles - mg);

        for (uint64_t nt = 0; nt < gemm_n_tiles; ++nt)
        {
            for (uint64_t kt = 0; kt < gemm_k_tiles; ++kt)
            {
                if (dma_b_rows > 0 && dma_b_row_scalar > 0)
                    do_scalar(dma_b_rows * dma_b_row_scalar);
                DmaReq b_read = issue_dma_begin(false, B_bytes);
                finish_dma(b_read);

                const bool final_k = (kt + 1 == gemm_k_tiles);
                issue_stream(Interconnect::ADDR_MAT,
                             m_batch,
                             mat_cycles,
                             mat_scalar_cycles,
                             A_bytes + B_bytes,
                             final_k ? C_bytes : 0,
                             A_bytes,
                             final_k ? C_bytes : 0,
                             mat_calls,
                             nullptr,
                             max_inflight_mat_reqs,
                             true);
            }
        }
    }
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
    {
        if (gemm_m_tiles > 0 && gemm_n_tiles > 0 && gemm_k_tiles > 0)
            issue_gemm_reuse_stream();
        else
            issue_stream(Interconnect::ADDR_MAT,
                         access_mat,
                         mat_cycles,
                         mat_scalar_cycles,
                         A_bytes + B_bytes,
                         C_bytes,
                         A_bytes + B_bytes,
                         C_bytes,
                         mat_calls,
                         nullptr,
                         max_inflight_mat_reqs);
    }

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
                         vec_scalar_cycles,
                         vrd,
                         vwr,
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
