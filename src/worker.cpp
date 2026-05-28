#include "worker.h"
#include "interconnect.h"   // for Interconnect::ADDR_MAT / ADDR_VEC
#include "memory.h"         // for MemoryAccessExt / MemoryAccessKind
#include <algorithm>
#include <iostream>
#include "perfetto_trace.h"

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

void Worker::do_scalar(uint64_t cyc, const char *label)
{
    compute_cycles += cyc;
    wait(cyc * CYCLE);
    // start = end - cyc; computed inside the (no-op-when-off) macro args so the
    // untraced build pays nothing.
    PERF_TRACE_SPAN("Scalar Unit " + std::to_string(tid), "scalar", label,
                    static_cast<uint64_t>(sc_time_stamp() / CYCLE) - cyc, cyc);
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
        [[maybe_unused]] const char *stall_lane =
            (addr == Interconnect::ADDR_MAT) ? "stall (matrix FIFO full)" :
            (addr == Interconnect::ADDR_VEC) ? "stall (vector FIFO full)" :
                                               "stall (other)";
        // Slice name kept distinct from the lane name so Perfetto's UI doesn't
        // treat the lane as a single-category track and add a density-summary
        // row on top of the actual slices.
        PERF_TRACE_SPAN("Scalar Unit " + std::to_string(tid), stall_lane,
                        "stall",
                        static_cast<uint64_t>(t_stall_start / CYCLE),
                        p.stall_cycles);
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

void Worker::configure_dma_tile_load_cost(uint64_t a_tile_cost,
                                          uint64_t b_tile_cost)
{
    dma_a_tile_scalar = a_tile_cost;
    dma_b_tile_scalar = b_tile_cost;
}

void Worker::configure_dma_c_row_cost(uint64_t c_rows, uint64_t c_row_cost)
{
    dma_c_rows       = c_rows;
    dma_c_row_scalar = c_row_cost;
}

void Worker::configure_dma_vec_cost(uint64_t rd_cost, uint64_t wr_cost)
{
    dma_vec_rd_scalar = rd_cost;
    dma_vec_wr_scalar = wr_cost;
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
                          bool scalar_between_streams,
                          DmaScalarMode dma_scalar_mode)
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
        if (dma_rd > 0)
        {
            if (dma_scalar_mode == DmaScalarMode::MatRow)
            {
                if (dma_a_tile_scalar > 0)
                    do_scalar(dma_a_tile_scalar, "scalar: A-tile load");
            }
            else if (dma_vec_rd_scalar > 0)
            {
                do_scalar(dma_vec_rd_scalar, "scalar: vec tile load");
            }
        }
        read_inflight.push_back(issue_dma_begin(false, dma_rd));
        ++reads_issued;
    };

    auto issue_accel = [&]() {
        // Every dispatch pays the per-request scalar overhead, including the
        // very first one (no first-request exemption).
        do_scalar(scalar_cycles,
                  addr == Interconnect::ADDR_MAT
                      ? "scalar: request to mat unit"
                      : "scalar: request to vec unit");
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
                if (dma_wr > 0)
                {
                    if (dma_scalar_mode == DmaScalarMode::MatRow)
                    {
                        if (dma_c_rows > 0 && dma_c_row_scalar > 0)
                            do_scalar(dma_c_rows * dma_c_row_scalar,
                                      "scalar: C-row store");
                    }
                    else if (dma_vec_wr_scalar > 0)
                    {
                        do_scalar(dma_vec_wr_scalar, "scalar: vec tile store");
                    }
                }
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

    const uint64_t regs   = std::max<uint64_t>(accumulator_register_count, 1);
    const uint64_t window = std::max<uint64_t>(max_inflight_mat_reqs, 1);

    // Flatten every (mg, nt, kt) tile-group into one dispatch schedule. Each
    // dispatch consumes its own A-tile (one DMA read); the first dispatch of a
    // group also triggers that group's single shared B-tile read; a final-K
    // dispatch writes back a C tile. Running the whole GEMM as ONE pipeline
    // (instead of one drained issue_stream per group) lets the next batch's
    // B/A reads start as soon as the inflight window has room — i.e. once the
    // previous batch's requests are *sent*, not after it finishes computing.
    struct Disp { bool first_in_group; bool writes_c; };
    std::vector<Disp> sched;
    sched.reserve(static_cast<size_t>(gemm_m_tiles) *
                  static_cast<size_t>(gemm_n_tiles) *
                  static_cast<size_t>(gemm_k_tiles));
    for (uint64_t mg = 0; mg < gemm_m_tiles; mg += regs)
    {
        const uint64_t m_batch = std::min<uint64_t>(regs, gemm_m_tiles - mg);
        for (uint64_t nt = 0; nt < gemm_n_tiles; ++nt)
            for (uint64_t kt = 0; kt < gemm_k_tiles; ++kt)
            {
                const bool final_k = (kt + 1 == gemm_k_tiles);
                for (uint64_t i = 0; i < m_batch; ++i)
                    sched.push_back({ i == 0, final_k });
            }
    }
    const size_t total = sched.size();

    std::deque<DmaReq>     a_reads;        // one A-tile read per dispatch (in order)
    std::deque<DmaReq>     b_reads;        // one B-tile read per group (in order)
    std::deque<PendingReq> accel_inflight;
    std::deque<DmaReq>     write_inflight;

    size_t reads_issued  = 0;   // dispatches whose A-read has been issued
    size_t accels_issued = 0;   // dispatches promoted to the accelerator
    size_t accels_retired = 0;  // dispatches whose response has been consumed

    auto dma_done = [](const DmaReq &p) {
        return !p.gp || p.sync_done || (p.done_entry && p.done_entry->fired);
    };
    auto accel_done = [](const PendingReq &p) {
        return p.sync_done || (p.done_entry && p.done_entry->fired);
    };

    // Prepare ONE more DMA read (the next dispatch's A-tile, plus its group's
    // shared B-tile on the first dispatch of a group). Bounded only by the
    // accelerator FIFO depth: the in-flight request count (reads waiting on DMA
    // + dispatches queued on the accelerator) may not exceed `window`. The
    // scalar core keeps doing this whenever no dispatch is ready to send, so it
    // only goes idle once the FIFO is full (accel_inflight == window) and
    // backpressure stalls it.
    auto issue_one_read = [&]() {
        if (reads_issued >= total ||
            a_reads.size() + accel_inflight.size() >= window)
            return false;
        if (sched[reads_issued].first_in_group)
        {
            if (dma_b_tile_scalar > 0)
                do_scalar(dma_b_tile_scalar, "scalar: B-tile load");
            b_reads.push_back(issue_dma_begin(false, B_bytes));
        }
        if (dma_a_tile_scalar > 0)
            do_scalar(dma_a_tile_scalar, "scalar: A-tile load");
        a_reads.push_back(issue_dma_begin(false, A_bytes));
        ++reads_issued;
        return true;
    };

    // Promote completed reads into accelerator dispatches (in order).
    auto promote = [&]() {
        bool progressed = false;
        while (!a_reads.empty() && accel_inflight.size() < window &&
               dma_done(a_reads.front()))
        {
            if (sched[accels_issued].first_in_group)
            {
                finish_dma(b_reads.front());   // group's B (already complete)
                b_reads.pop_front();
            }
            finish_dma(a_reads.front());
            a_reads.pop_front();

            do_scalar(mat_scalar_cycles, "scalar: request to mat unit");
            accel_inflight.push_back(
                issue_begin(Interconnect::ADDR_MAT, mat_cycles,
                            A_bytes + B_bytes,
                            sched[accels_issued].writes_c ? C_bytes : 0));
            ++mat_calls;
            ++accels_issued;
            progressed = true;
        }
        return progressed;
    };

    // Retire finished dispatches in order; final-K ones spawn a C-write DMA.
    // The matmul matrix pool is per-worker pinned, so completion is in-order.
    auto retire_accels = [&]() {
        bool progressed = false;
        while (!accel_inflight.empty() && write_inflight.size() < window &&
               accel_done(accel_inflight.front()))
        {
            const bool writes_c = sched[accels_retired].writes_c;
            issue_end(accel_inflight.front());
            accel_inflight.pop_front();
            ++accels_retired;
            if (writes_c)
            {
                if (dma_c_rows > 0 && dma_c_row_scalar > 0)
                    do_scalar(dma_c_rows * dma_c_row_scalar,
                              "scalar: C-tile store");
                write_inflight.push_back(issue_dma_begin(true, C_bytes));
            }
            progressed = true;
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
        auto add_dma = [&](const DmaReq &p) {
            if (p.gp && !p.sync_done && p.done_entry && !p.done_entry->fired)
            { wait_list |= p.done_entry->ev; has_event = true; }
        };
        auto add_accel = [&](const PendingReq &p) {
            if (!p.sync_done && p.done_entry && !p.done_entry->fired)
            { wait_list |= p.done_entry->ev; has_event = true; }
        };
        for (const auto &p : a_reads)        add_dma(p);
        for (const auto &p : accel_inflight) add_accel(p);
        for (const auto &p : write_inflight) add_dma(p);
        if (has_event)
            wait(wait_list);
    };

    while (accels_issued < total ||
           !accel_inflight.empty() ||
           !write_inflight.empty())
    {
        // Reclaim finished work first (frees FIFO slots; final-K spawns C-write).
        bool progressed = retire_accels();
        progressed = retire_writes(false) || progressed;

        // Priority: send any request whose tiles are ready. Only when none is
        // ready does the scalar core fall back to preparing the next DMA read,
        // so it stays busy until the accelerator FIFO is full.
        if (promote())
            progressed = true;
        else if (issue_one_read())
            progressed = true;

        if (write_inflight.size() >= window)
            progressed = retire_writes(true) || progressed;

        if (!progressed)
            wait_for_progress();
    }
}

void Worker::run()
{
    if (start_event)
        wait(*start_event);

    sc_time start = sc_time_stamp();

    // Declare the Scalar Unit lanes in a fixed order so they always appear
    // (the DMA-FIFO-full lane stays empty: the memory model never stalls a
    // worker on a full DMA queue).
    {
        const std::string grp = "Scalar Unit " + std::to_string(tid);
        PERF_TRACE_DECLARE(grp, "scalar");
        PERF_TRACE_DECLARE(grp, "stall (matrix FIFO full)");
        PERF_TRACE_DECLARE(grp, "stall (vector FIFO full)");
        PERF_TRACE_DECLARE(grp, "stall (DMA FIFO full)");
    }

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
                         max_inflight_vec_reqs,
                         /*scalar_between_streams=*/false,
                         DmaScalarMode::VecPerCall);
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
