#include "accum_coordinator.h"
#include "interconnect.h"
#include <algorithm>
#include <iostream>

AccumCoordinator::AccumCoordinator(sc_module_name name,
                                   uint64_t accum_vec_calls_,
                                   uint64_t final_quant_calls_,
                                   uint64_t accum_vec_cycles_,
                                   uint64_t quant_vec_cycles_,
                                   uint64_t accum_rd_bytes_,
                                   uint64_t accum_wr_bytes_,
                                   uint64_t quant_rd_bytes_,
                                   uint64_t quant_wr_bytes_)
    : sc_module(name),
      accum_vec_calls(accum_vec_calls_),
      final_quant_calls(final_quant_calls_),
      accum_vec_cycles(accum_vec_cycles_),
      quant_vec_cycles(quant_vec_cycles_),
      accum_rd_bytes(accum_rd_bytes_),
      accum_wr_bytes(accum_wr_bytes_),
      quant_rd_bytes(quant_rd_bytes_),
      quant_wr_bytes(quant_wr_bytes_)
{
}

void AccumCoordinator::configure_workers(const std::vector<Worker *> &workers_)
{
    workers = workers_;
    build_reduction_tree();
}

void AccumCoordinator::build_reduction_tree()
{
    nodes.clear();
    tasks.clear();
    pair_start_times.clear();
    pair_end_times.clear();
    pair_left_ready_times.clear();
    pair_right_ready_times.clear();
    root_task_id = no_task;
    root_node_id = no_task;
    reduction_complete = false;
    accum_end_time = SC_ZERO_TIME;
    quant_start_time = SC_ZERO_TIME;
    quant_end_time = SC_ZERO_TIME;
    vec_calls_total = 0;
    accum_vec_calls_total = 0;
    final_quant_calls_total = 0;
    compute_cycles = 0;
    wait_cycles = 0;
    stall_cycles = 0;
    mem_cycles = 0;

    size_t n = workers.size();
    if (n == 0)
        return;

    nodes.resize(n);
    std::vector<size_t> cur_nodes;
    cur_nodes.reserve(n);
    for (size_t i = 0; i < n; ++i)
        cur_nodes.push_back(i);

    while (cur_nodes.size() > 1)
    {
        std::vector<size_t> next_nodes;
        next_nodes.reserve((cur_nodes.size() + 1) / 2);

        size_t pairs = cur_nodes.size() / 2;
        for (size_t i = 0; i < pairs; ++i)
        {
            size_t left = cur_nodes[2 * i];
            size_t right = cur_nodes[2 * i + 1];
            size_t out = nodes.size();
            nodes.push_back(NodeState{});

            size_t task_id = tasks.size();
            tasks.push_back({left, right, out, false, false, false});
            nodes[left].parent_task = task_id;
            nodes[right].parent_task = task_id;
            next_nodes.push_back(out);
        }

        if (cur_nodes.size() % 2 == 1)
            next_nodes.push_back(cur_nodes.back());

        cur_nodes = std::move(next_nodes);
    }

    root_node_id = cur_nodes.front();
    if (!tasks.empty())
        root_task_id = tasks.size() - 1;

    pair_start_times.assign(tasks.size(), SC_ZERO_TIME);
    pair_end_times.assign(tasks.size(), SC_ZERO_TIME);
    pair_left_ready_times.assign(tasks.size(), SC_ZERO_TIME);
    pair_right_ready_times.assign(tasks.size(), SC_ZERO_TIME);

    ready_task_fifo =
        std::make_unique<sc_fifo<int>>(sc_gen_unique_name("ready_task_fifo"),
                                       static_cast<int>(tasks.size() + workers.size() + 1));
}

void AccumCoordinator::maybe_enqueue_task_locked(size_t task_id)
{
    if (task_id == no_task || task_id >= tasks.size())
        return;

    auto &task = tasks[task_id];
    if (task.done || task.claimed || task.queued)
        return;
    if (!nodes[task.left].ready || !nodes[task.right].ready)
        return;

    task.queued = true;
    ready_task_fifo->nb_write(static_cast<int>(task_id));
}

void AccumCoordinator::mark_leaf_ready(size_t leaf_id, sc_time ready_time)
{
    state_mutex.lock();

    if (leaf_id < nodes.size())
    {
        nodes[leaf_id].ready = true;
        nodes[leaf_id].ready_time = ready_time;

        if (tasks.empty() && leaf_id == root_node_id && !reduction_complete)
        {
            reduction_complete = true;
            accum_end_time = ready_time;
            for (size_t i = 0; i < workers.size(); ++i)
                ready_task_fifo->nb_write(-1);
        }
        else
        {
            maybe_enqueue_task_locked(nodes[leaf_id].parent_task);
        }
    }

    state_mutex.unlock();
}

bool AccumCoordinator::claim_task(size_t task_id,
                                  sc_time &left_ready,
                                  sc_time &right_ready)
{
    state_mutex.lock();

    bool ok = false;
    if (task_id < tasks.size())
    {
        auto &task = tasks[task_id];
        if (!task.done && !task.claimed && task.queued &&
            nodes[task.left].ready && nodes[task.right].ready)
        {
            task.claimed = true;
            task.queued = false;
            left_ready = nodes[task.left].ready_time;
            right_ready = nodes[task.right].ready_time;
            ok = true;
        }
    }

    state_mutex.unlock();
    return ok;
}

void AccumCoordinator::finish_task(size_t task_id,
                                   sc_time start_time,
                                   sc_time end_time,
                                   sc_time left_ready,
                                   sc_time right_ready)
{
    stats_mutex.lock();
    pair_start_times[task_id] = start_time;
    pair_end_times[task_id] = end_time;
    pair_left_ready_times[task_id] = left_ready;
    pair_right_ready_times[task_id] = right_ready;
    stats_mutex.unlock();

    state_mutex.lock();

    auto &task = tasks[task_id];
    task.done = true;
    nodes[task.out].ready = true;
    nodes[task.out].ready_time = end_time;

    if (task_id == root_task_id && !reduction_complete)
    {
        reduction_complete = true;
        accum_end_time = end_time;
        for (size_t i = 0; i < workers.size(); ++i)
            ready_task_fifo->nb_write(-1);
    }
    else
    {
        maybe_enqueue_task_locked(nodes[task.out].parent_task);
    }

    state_mutex.unlock();
}

AccumCoordinator::WorkerSnapshot
AccumCoordinator::snapshot_worker(const Worker &worker) const
{
    WorkerSnapshot s;
    s.compute = worker.compute_cycles;
    s.wait = worker.wait_cycles;
    s.stall = worker.stall_cycles;
    s.mem = worker.mem_cycles_accum;
    s.vec_calls = worker.vec_calls;
    s.accum_vec_calls = worker.accum_vec_calls;
    s.quant_vec_calls = worker.quant_vec_calls;
    return s;
}

void AccumCoordinator::accumulate_worker_delta(const WorkerSnapshot &before,
                                               const WorkerSnapshot &after)
{
    stats_mutex.lock();
    compute_cycles += (after.compute - before.compute);
    wait_cycles += (after.wait - before.wait);
    stall_cycles += (after.stall - before.stall);
    mem_cycles += (after.mem - before.mem);
    vec_calls_total += (after.vec_calls - before.vec_calls);
    accum_vec_calls_total += (after.accum_vec_calls - before.accum_vec_calls);
    final_quant_calls_total += (after.quant_vec_calls - before.quant_vec_calls);
    stats_mutex.unlock();
}

void AccumCoordinator::run_one_pair(Worker &worker,
                                    size_t task_id,
                                    sc_time left_ready,
                                    sc_time right_ready)
{
    (void)left_ready;
    (void)right_ready;

    WorkerSnapshot before = snapshot_worker(worker);
    sc_time start_time = sc_time_stamp();

    worker.issue_stream(Interconnect::ADDR_VEC,
                        accum_vec_calls,
                        accum_vec_cycles,
                        worker.vec_scalar_cycles,
                        accum_rd_bytes,
                        accum_wr_bytes,
                        accum_rd_bytes,
                        accum_wr_bytes,
                        worker.vec_calls,
                        &worker.accum_vec_calls,
                        worker.max_inflight_vec_reqs);
    worker.reduction_pairs++;

    sc_time end_time = sc_time_stamp();
    WorkerSnapshot after = snapshot_worker(worker);

    accumulate_worker_delta(before, after);
    finish_task(task_id, start_time, end_time, left_ready, right_ready);
}

uint64_t AccumCoordinator::quant_calls_for_worker(int tid) const
{
    uint64_t total = final_quant_calls;
    uint64_t workers_count = static_cast<uint64_t>(std::max<size_t>(workers.size(), 1));
    uint64_t base = total / workers_count;
    uint64_t rem = total % workers_count;
    uint64_t idx = static_cast<uint64_t>(std::max(tid, 0));
    return base + ((idx < rem) ? 1 : 0);
}

void AccumCoordinator::run_final_quant(Worker &worker)
{
    uint64_t quant_calls = quant_calls_for_worker(worker.tid);
    if (quant_calls == 0)
        return;

    stats_mutex.lock();
    if (quant_start_time == SC_ZERO_TIME)
        quant_start_time = sc_time_stamp();
    stats_mutex.unlock();

    WorkerSnapshot before = snapshot_worker(worker);
    worker.issue_stream(Interconnect::ADDR_VEC,
                        quant_calls,
                        quant_vec_cycles,
                        worker.vec_scalar_cycles,
                        quant_rd_bytes,
                        quant_wr_bytes,
                        quant_rd_bytes,
                        quant_wr_bytes,
                        worker.vec_calls,
                        &worker.quant_vec_calls,
                        worker.max_inflight_vec_reqs);
    WorkerSnapshot after = snapshot_worker(worker);

    accumulate_worker_delta(before, after);

    stats_mutex.lock();
    if (sc_time_stamp() > quant_end_time)
        quant_end_time = sc_time_stamp();
    stats_mutex.unlock();
}

void AccumCoordinator::run_post_mat(Worker &worker)
{
    mark_leaf_ready(static_cast<size_t>(worker.tid), worker.mat_done_time);

    while (true)
    {
        int item = ready_task_fifo->read();
        if (item < 0)
            break;

        size_t task_id = static_cast<size_t>(item);
        sc_time left_ready = SC_ZERO_TIME;
        sc_time right_ready = SC_ZERO_TIME;
        if (!claim_task(task_id, left_ready, right_ready))
            continue;

        run_one_pair(worker, task_id, left_ready, right_ready);
    }

    run_final_quant(worker);
}
