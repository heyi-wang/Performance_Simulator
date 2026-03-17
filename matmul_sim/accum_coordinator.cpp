#include "accum_coordinator.h"
#include "../src/interconnect.h"
#include <iostream>

SC_HAS_PROCESS(AccumCoordinator);

AccumCoordinator::AccumCoordinator(sc_module_name name,
                                   uint64_t vec_svc_cycles_,
                                   uint64_t accum_vec_calls_,
                                   uint64_t accum_rd_bytes_,
                                   uint64_t accum_wr_bytes_)
    : sc_module(name),
      init("init"),
      peq("peq"),
      vec_svc_cycles(vec_svc_cycles_),
      accum_vec_calls(accum_vec_calls_),
      accum_rd_bytes(accum_rd_bytes_),
      accum_wr_bytes(accum_wr_bytes_)
{
    init.register_nb_transport_bw(this, &AccumCoordinator::nb_transport_bw);
    SC_THREAD(peq_thread);
    SC_THREAD(run);
}

AccumCoordinator::~AccumCoordinator()
{
    for (auto *ev : tree_events_)
        delete ev;
}

// ----------------------------------------------------------
// nb_transport_bw: receive BEGIN_RESP or deferred END_REQ
//
//   BEGIN_RESP — request completed; route through PEQ.
//   END_REQ    — queue slot granted after stall; notify admit_ev.
// ----------------------------------------------------------
tlm_sync_enum AccumCoordinator::nb_transport_bw(tlm_generic_payload &gp,
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
// peq_thread: set fired=true BEFORE notifying so issue_end
// can skip wait() if the response arrived early.
// ----------------------------------------------------------
void AccumCoordinator::peq_thread()
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

AccumCoordinator::PendingReq AccumCoordinator::issue_begin(uint64_t addr)
{
    PendingReq p;

    auto *gp = new tlm_generic_payload();
    gp->set_command(TLM_IGNORE_COMMAND);
    gp->set_address(addr);
    gp->set_data_ptr(nullptr);
    gp->set_data_length(0);
    gp->set_streaming_width(0);
    gp->set_response_status(TLM_INCOMPLETE_RESPONSE);

    auto *req = new ReqExt(-1, vec_svc_cycles, accum_rd_bytes, accum_wr_bytes);
    auto *tx  = new TxnExt();
    tx->src_worker = -1;

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
        // Queue full: stall until the accelerator grants a slot.
        sc_time t_stall_start = sc_time_stamp();
        wait(*p.done_entry->admit_ev);
        p.stall_cycles = (uint64_t)((sc_time_stamp() - t_stall_start) / CYCLE);
    }
    else if (status == TLM_COMPLETED)
    {
        done_map.erase(gp);
        p.sync_done = true;
    }
    // TLM_UPDATED: slot admitted immediately, no stall.

    return p;
}

void AccumCoordinator::issue_end(PendingReq &p)
{
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
    wait_cycles  += qwait + p.stall_cycles;
    stall_cycles += p.stall_cycles;
    mem_cycles   += ext ? ext->mem_cycles : 0;

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

// ----------------------------------------------------------
// run_one_pair: issue accum_vec_calls sequential vec_acc
// requests for one pairwise accumulation.  Records timing.
// ----------------------------------------------------------
void AccumCoordinator::run_one_pair()
{
    sc_time t0 = sc_time_stamp();

    for (uint64_t i = 0; i < accum_vec_calls; i++)
    {
        auto p = issue_begin(Interconnect::ADDR_VEC);
        vec_calls_total++;
        issue_end(p);
    }

    sc_time t1 = sc_time_stamp();

    stats_mutex.lock();
    pair_start_times.push_back(t0);
    pair_end_times.push_back(t1);
    stats_mutex.unlock();
}

// ----------------------------------------------------------
// run() — build the event-driven reduction tree.
//
// For each internal node, an independent SC_THREAD is spawned
// that waits for its two inputs then immediately accumulates
// without any global barrier.  Accumulation begins as soon as
// any pair of quantized partial results is available.
// ----------------------------------------------------------
void AccumCoordinator::run()
{
    int n = (int)workers.size();
    if (n <= 1)
    {
        accum_end_time = sc_time_stamp();
        return;
    }

    // Create proxy leaf events: one per worker.
    // A lightweight spawned thread waits on worker->mat_done_ev
    // (which fires after mat+quant) then fires the proxy.
    std::vector<sc_event *> cur_evs;
    for (int i = 0; i < n; i++)
    {
        auto *ev = new sc_event();
        tree_events_.push_back(ev);
        cur_evs.push_back(ev);

        Worker *w = workers[i];
        sc_spawn([w, ev]() {
            sc_core::wait(w->mat_done_ev);
            ev->notify(SC_ZERO_TIME);
        });
    }

    // Build the reduction tree level by level.
    // Each pair spawns a thread that waits for both inputs
    // then immediately calls run_one_pair().
    while ((int)cur_evs.size() > 1)
    {
        int m     = (int)cur_evs.size();
        int pairs = m / 2;

        std::vector<sc_event *> next_evs;
        next_evs.reserve((m + 1) / 2);

        for (int i = 0; i < pairs; i++)
        {
            sc_event *left  = cur_evs[2 * i];
            sc_event *right = cur_evs[2 * i + 1];

            auto *out = new sc_event();
            tree_events_.push_back(out);
            next_evs.push_back(out);

            sc_spawn([this, left, right, out]() {
                sc_core::wait(*left);
                sc_core::wait(*right);
                run_one_pair();
                out->notify(SC_ZERO_TIME);
            });
        }

        // Odd survivor carries forward unchanged
        if (m % 2 == 1)
            next_evs.push_back(cur_evs[m - 1]);

        cur_evs = std::move(next_evs);
    }

    // Wait for the root event (entire tree complete)
    wait(*cur_evs[0]);

    accum_end_time = sc_time_stamp();

    std::cout << "[AccumCoordinator]"
              << " pairs_done="       << pair_start_times.size()
              << " vec_calls_total="  << vec_calls_total
              << " wait_cycles="      << wait_cycles
              << " stall_cycles="     << stall_cycles
              << " mem_cycles="       << mem_cycles
              << " accum_end="        << accum_end_time
              << "\n";
}
