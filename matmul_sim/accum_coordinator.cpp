#include "accum_coordinator.h"
#include "../src/interconnect.h"
#include <iostream>

SC_HAS_PROCESS(AccumCoordinator);

AccumCoordinator::AccumCoordinator(sc_module_name name,
                                   uint64_t vec_svc_cycles_,
                                   uint64_t accum_vec_calls_,
                                   uint64_t final_quant_calls_,
                                   uint64_t max_inflight_vec_reqs_,
                                   uint64_t scalar_cycles_,
                                   uint64_t accum_rd_bytes_,
                                   uint64_t accum_wr_bytes_,
                                   uint64_t quant_rd_bytes_,
                                   uint64_t quant_wr_bytes_)
    : sc_module(name),
      init("init"),
      peq("peq"),
      vec_svc_cycles(vec_svc_cycles_),
      accum_vec_calls(accum_vec_calls_),
      final_quant_calls(final_quant_calls_),
      max_inflight_vec_reqs(std::max<uint64_t>(max_inflight_vec_reqs_, 1)),
      scalar_cycles(scalar_cycles_),
      accum_rd_bytes(accum_rd_bytes_),
      accum_wr_bytes(accum_wr_bytes_),
      quant_rd_bytes(quant_rd_bytes_),
      quant_wr_bytes(quant_wr_bytes_)
{
    init.register_nb_transport_bw(this, &AccumCoordinator::nb_transport_bw);
    SC_THREAD(peq_thread);
    SC_THREAD(run);
}

AccumCoordinator::~AccumCoordinator()
{
    for (auto *ev : tree_events_)
        delete ev;
    for (auto *t : tree_ready_times_)
        delete t;
    for (auto *ready : tree_ready_flags_)
        delete ready;
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

AccumCoordinator::PendingReq AccumCoordinator::issue_begin(uint64_t addr,
                                                           uint64_t rd_bytes,
                                                           uint64_t wr_bytes)
{
    PendingReq p;

    auto *gp = new tlm_generic_payload();
    gp->set_command(TLM_IGNORE_COMMAND);
    gp->set_address(addr);
    gp->set_data_ptr(nullptr);
    gp->set_data_length(0);
    gp->set_streaming_width(0);
    gp->set_response_status(TLM_INCOMPLETE_RESPONSE);

    auto *req = new ReqExt(-1, vec_svc_cycles, rd_bytes, wr_bytes);
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

void AccumCoordinator::do_scalar(uint64_t cyc)
{
    compute_cycles += cyc;
    wait(cyc * CYCLE);
}

void AccumCoordinator::issue_vec_stream(uint64_t call_count,
                                        uint64_t rd_bytes,
                                        uint64_t wr_bytes,
                                        uint64_t &stage_call_counter)
{
    if (call_count == 0)
        return;

    std::deque<PendingReq> inflight;
    uint64_t issued = 0;
    uint64_t window = std::max<uint64_t>(max_inflight_vec_reqs, 1);

    auto issue_one = [&]() {
        inflight.push_back(issue_begin(Interconnect::ADDR_VEC, rd_bytes, wr_bytes));
        ++vec_calls_total;
        ++stage_call_counter;
        ++issued;
    };

    while (issued < call_count && inflight.size() < window)
        issue_one();

    while (!inflight.empty())
    {
        if (issued < call_count)
            do_scalar(scalar_cycles);

        while (issued < call_count && inflight.size() < window)
            issue_one();

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

        bool retired_one = false;
        for (auto it = inflight.begin(); it != inflight.end();)
        {
            if (it->sync_done || (it->done_entry && it->done_entry->fired))
            {
                issue_end(*it);
                it = inflight.erase(it);
                retired_one = true;
                break;
            }
            else
            {
                ++it;
            }
        }

        if (!retired_one && !inflight.empty())
        {
            issue_end(inflight.front());
            inflight.pop_front();
        }
    }
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
// run_one_pair: issue accum_vec_calls with a pipelined in-flight
// vec request window so one reduction thread can occupy multiple
// vec accelerators concurrently. Records timing.
// ----------------------------------------------------------
void AccumCoordinator::run_one_pair(size_t pair_id,
                                    sc_time left_ready,
                                    sc_time right_ready)
{
    sc_time t0 = sc_time_stamp();

    issue_vec_stream(accum_vec_calls,
                     accum_rd_bytes,
                     accum_wr_bytes,
                     accum_vec_calls_total);

    sc_time t1 = sc_time_stamp();

    stats_mutex.lock();
    pair_start_times[pair_id] = t0;
    pair_end_times[pair_id] = t1;
    pair_left_ready_times[pair_id] = left_ready;
    pair_right_ready_times[pair_id] = right_ready;
    stats_mutex.unlock();
}

void AccumCoordinator::run_final_quant()
{
    issue_vec_stream(final_quant_calls,
                     quant_rd_bytes,
                     quant_wr_bytes,
                     final_quant_calls_total);
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
    if (n == 0)
    {
        accum_end_time = sc_time_stamp();
        return;
    }
    if (n == 1)
    {
        wait(workers[0]->mat_done_ev);
        run_final_quant();
        accum_end_time = sc_time_stamp();
        return;
    }

    // Create proxy leaf events: one per worker.
    // A lightweight spawned thread waits on worker->mat_done_ev
    // (which fires after mat+quant) then fires the proxy.
    struct ReadyNode
    {
        sc_event *ev = nullptr;
        sc_time  *ready_time = nullptr;
        bool     *ready = nullptr;
    };

    size_t total_pairs = (size_t)(n - 1);
    pair_start_times.assign(total_pairs, SC_ZERO_TIME);
    pair_end_times.assign(total_pairs, SC_ZERO_TIME);
    pair_left_ready_times.assign(total_pairs, SC_ZERO_TIME);
    pair_right_ready_times.assign(total_pairs, SC_ZERO_TIME);

    std::vector<ReadyNode> cur_nodes;
    for (int i = 0; i < n; i++)
    {
        auto *ev = new sc_event();
        auto *ready_time = new sc_time(SC_ZERO_TIME);
        auto *ready = new bool(false);
        tree_events_.push_back(ev);
        tree_ready_times_.push_back(ready_time);
        tree_ready_flags_.push_back(ready);
        cur_nodes.push_back({ev, ready_time, ready});

        Worker *w = workers[i];
        sc_spawn([w, ev, ready_time, ready]() {
            sc_core::wait(w->mat_done_ev);
            *ready_time = w->mat_done_time;
            *ready = true;
            ev->notify(SC_ZERO_TIME);
        });
    }

    // Build the reduction tree level by level.
    // Each pair spawns a thread that waits for both inputs
    // then immediately calls run_one_pair().
    size_t pair_id = 0;
    while ((int)cur_nodes.size() > 1)
    {
        int m     = (int)cur_nodes.size();
        int pairs = m / 2;

        std::vector<ReadyNode> next_nodes;
        next_nodes.reserve((m + 1) / 2);

        for (int i = 0; i < pairs; i++)
        {
            ReadyNode left  = cur_nodes[2 * i];
            ReadyNode right = cur_nodes[2 * i + 1];

            auto *out = new sc_event();
            auto *ready_time = new sc_time(SC_ZERO_TIME);
            auto *ready = new bool(false);
            tree_events_.push_back(out);
            tree_ready_times_.push_back(ready_time);
            tree_ready_flags_.push_back(ready);
            next_nodes.push_back({out, ready_time, ready});

            size_t this_pair = pair_id++;
            sc_spawn([this, left, right, out, ready_time, ready, this_pair]() {
                if (!*left.ready)
                    sc_core::wait(*left.ev);
                if (!*right.ready)
                    sc_core::wait(*right.ev);
                sc_time left_ready = *left.ready_time;
                sc_time right_ready = *right.ready_time;
                run_one_pair(this_pair, left_ready, right_ready);
                *ready_time = sc_time_stamp();
                *ready = true;
                out->notify(SC_ZERO_TIME);
            });
        }

        // Odd survivor carries forward unchanged
        if (m % 2 == 1)
            next_nodes.push_back(cur_nodes[m - 1]);

        cur_nodes = std::move(next_nodes);
    }

    // Wait for the root event (entire tree complete)
    wait(*cur_nodes[0].ev);

    run_final_quant();
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
