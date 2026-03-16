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

tlm_sync_enum AccumCoordinator::nb_transport_bw(tlm_generic_payload &gp,
                                                 tlm_phase &phase,
                                                 sc_time &delay)
{
    if (phase == BEGIN_RESP)
        peq.notify(gp, delay);
    return TLM_ACCEPTED;
}

void AccumCoordinator::peq_thread()
{
    while (true)
    {
        wait(peq.get_event());
        while (auto *gp = peq.get_next_transaction())
        {
            auto it = done_map.find(gp);
            if (it != done_map.end() && it->second)
                it->second->notify(SC_ZERO_TIME);
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

    p.gp      = gp;
    p.req_ext = req;
    p.tx_ext  = tx;
    p.done_ev = new sc_event();
    done_map[gp] = p.done_ev;

    tlm_phase phase = BEGIN_REQ;
    sc_time   delay = SC_ZERO_TIME;
    auto status = init->nb_transport_fw(*gp, phase, delay);

    if (status == TLM_COMPLETED)
    {
        done_map.erase(gp);
        p.sync_done = true;
    }

    return p;
}

void AccumCoordinator::issue_end(PendingReq &p)
{
    if (!p.sync_done)
        wait(*p.done_ev);

    done_map.erase(p.gp);
    delete p.done_ev;
    p.done_ev = nullptr;

    ReqExt *ext = nullptr;
    p.gp->get_extension(ext);

    wait_cycles += ext ? ext->accel_qwait_cycles : 0;
    mem_cycles  += ext ? ext->mem_cycles : 0;

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

    // Record pair timing (protected because multiple spawned threads call this).
    stats_mutex.lock();
    pair_start_times.push_back(t0);
    pair_end_times.push_back(t1);
    stats_mutex.unlock();
}

// ============================================================
// run() — build the event-driven reduction tree and wait for
// the root result.
//
// For each internal node of the binary tree, an independent
// SC_THREAD is spawned that:
//   1. Waits for its left-input ready event.
//   2. Waits for its right-input ready event.
//   3. Immediately starts run_one_pair() (no global barrier).
//   4. Fires its own output ready event.
//
// As a result, accumulation begins as soon as any pair of
// partial results is available, overlapping with mat-phase
// work that is still in progress on other workers.
// ============================================================
void AccumCoordinator::run()
{
    int n = (int)workers.size();
    if (n <= 1)
    {
        accum_end_time = sc_time_stamp();
        return;
    }

    // ----------------------------------------------------------
    // Create proxy "leaf" events: one per worker.
    // A lightweight spawned thread waits on worker->mat_done_ev
    // and then fires the proxy, decoupling the coordinator from
    // the Worker's internal event object.
    // ----------------------------------------------------------
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

    // ----------------------------------------------------------
    // Build the reduction tree level by level.
    // For each pair (left, right) at the current level, spawn
    // a thread that blocks on both inputs then accumulates.
    // Odd survivors are forwarded unchanged to the next level.
    // ----------------------------------------------------------
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

            // Capture by value so the lambda is self-contained.
            sc_spawn([this, left, right, out]() {
                sc_core::wait(*left);
                sc_core::wait(*right);
                run_one_pair();
                out->notify(SC_ZERO_TIME);
            });
        }

        // Odd survivor: carries forward to the next level as-is.
        if (m % 2 == 1)
            next_evs.push_back(cur_evs[m - 1]);

        cur_evs = std::move(next_evs);
    }

    // Wait for the single remaining (root) event — this fires when
    // the entire reduction tree has completed.
    wait(*cur_evs[0]);

    accum_end_time = sc_time_stamp();

    std::cout << "[AccumCoordinator]"
              << " pairs_done=" << pair_start_times.size()
              << " vec_calls_total=" << vec_calls_total
              << " wait_cycles=" << wait_cycles
              << " mem_cycles=" << mem_cycles
              << " accum_end=" << accum_end_time
              << "\n";
}
