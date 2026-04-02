#pragma once

#include "worker.h"
#include <deque>
#include <limits>
#include <memory>
#include <systemc>
#include <vector>

// ============================================================
// AccumCoordinator — passive shared state for worker-driven
// reduction and final quantization in the K-split GEMM simulator.
//
// No SC_THREADs or sc_spawn() are used here. Existing workers call
// run_post_mat() inside their own SC_THREAD after finishing the local
// matmul phase.
// ============================================================
struct AccumCoordinator : sc_module, WorkerPostProcessor
{
    std::vector<Worker *> workers;

    uint64_t accum_vec_calls;   // vec_acc calls per pairwise accumulation
    uint64_t final_quant_calls; // total vec_acc calls for final quantization
    uint64_t accum_rd_bytes;
    uint64_t accum_wr_bytes;
    uint64_t quant_rd_bytes;
    uint64_t quant_wr_bytes;

    // Statistics filled in by workers while they execute post-mat work.
    sc_time  accum_end_time       = SC_ZERO_TIME; // root reduction completion
    sc_time  quant_start_time     = SC_ZERO_TIME;
    sc_time  quant_end_time       = SC_ZERO_TIME;
    uint64_t vec_calls_total      = 0;
    uint64_t accum_vec_calls_total = 0;
    uint64_t final_quant_calls_total = 0;
    uint64_t compute_cycles       = 0;
    uint64_t wait_cycles          = 0;
    uint64_t stall_cycles         = 0;
    uint64_t mem_cycles           = 0;

    // Per-pair timing (one entry per pairwise accumulation, tree order).
    std::vector<sc_time> pair_start_times;
    std::vector<sc_time> pair_end_times;
    std::vector<sc_time> pair_left_ready_times;
    std::vector<sc_time> pair_right_ready_times;
    sc_mutex             stats_mutex;

    struct NodeState
    {
        bool    ready = false;
        sc_time ready_time = SC_ZERO_TIME;
        size_t  parent_task = std::numeric_limits<size_t>::max();
    };

    struct TaskState
    {
        size_t left = 0;
        size_t right = 0;
        size_t out = 0;
        bool   queued = false;
        bool   claimed = false;
        bool   done = false;
    };

    std::vector<NodeState> nodes;
    std::vector<TaskState> tasks;
    std::unique_ptr<sc_fifo<int>> ready_task_fifo;
    sc_mutex                      state_mutex;

    size_t root_task_id = std::numeric_limits<size_t>::max();
    size_t root_node_id = std::numeric_limits<size_t>::max();
    bool   reduction_complete = false;

    explicit AccumCoordinator(sc_module_name name,
                              uint64_t accum_vec_calls_,
                              uint64_t final_quant_calls_,
                              uint64_t accum_rd_bytes_,
                              uint64_t accum_wr_bytes_,
                              uint64_t quant_rd_bytes_,
                              uint64_t quant_wr_bytes_);

    void configure_workers(const std::vector<Worker *> &workers_);
    void run_post_mat(Worker &worker) override;

private:
    struct WorkerSnapshot
    {
        uint64_t compute = 0;
        uint64_t wait = 0;
        uint64_t stall = 0;
        uint64_t mem = 0;
        uint64_t vec_calls = 0;
        uint64_t accum_vec_calls = 0;
        uint64_t quant_vec_calls = 0;
    };

    static constexpr size_t no_task = std::numeric_limits<size_t>::max();

    void build_reduction_tree();
    void mark_leaf_ready(size_t leaf_id, sc_time ready_time);
    void maybe_enqueue_task_locked(size_t task_id);
    bool claim_task(size_t task_id, sc_time &left_ready, sc_time &right_ready);
    void finish_task(size_t task_id,
                     sc_time start_time,
                     sc_time end_time,
                     sc_time left_ready,
                     sc_time right_ready);

    void run_one_pair(Worker &worker,
                      size_t task_id,
                      sc_time left_ready,
                      sc_time right_ready);
    void run_final_quant(Worker &worker);

    WorkerSnapshot snapshot_worker(const Worker &worker) const;
    void accumulate_worker_delta(const WorkerSnapshot &before,
                                 const WorkerSnapshot &after);

    uint64_t quant_calls_for_worker(int tid) const;
};
