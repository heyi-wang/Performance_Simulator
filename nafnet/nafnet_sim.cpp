// nafnet_sim.cpp
// NAFNet performance simulator built on the shared SystemC TLM-2.0
// hardware model in src/. This version can simulate a standalone
// NafBlock using the same tiling and phase structure as the
// kernel-level simulators under kernel/.

#include <systemc>
#include <tlm>
#include <tlm_utils/peq_with_get.h>
#include <tlm_utils/simple_initiator_socket.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "accelerator.h"
#include "accelerator_pool.h"
#include "common.h"
#include "extensions.h"
#include "interconnect.h"
#include "memory.h"
#include "nafnet_hw_config.h"
#include "nafnet_layers.h"
#include "vcd_writer.h"
#include "waveform.h"

using namespace sc_core;
using namespace tlm;

struct Barrier
{
    int      n_workers;
    int      arrived    = 0;
    int      generation = 0;
    sc_event released;

    explicit Barrier(int n) : n_workers(n) {}

    void sync()
    {
        int my_gen = generation;
        if (++arrived == n_workers)
        {
            arrived = 0;
            ++generation;
            released.notify(SC_ZERO_TIME);
        }
        else
        {
            while (generation == my_gen)
                wait(released);
        }
    }
};

struct NafLayerExt : tlm_extension<NafLayerExt>
{
    int          layer_id = -1;
    LayerOpKind  op_kind = LAYER_OP_CONV;
    LayerBackend backend = BACKEND_MATMUL;

    tlm_extension_base *clone() const override
    {
        return new NafLayerExt(*this);
    }

    void copy_from(const tlm_extension_base &ext) override
    {
        *this = static_cast<const NafLayerExt &>(ext);
    }
};

struct NafLayerStats
{
    uint64_t mat_reqs = 0;
    uint64_t vec_reqs = 0;
    uint64_t mem_reqs = 0;
    uint64_t accel_cycles = 0;
    uint64_t cpu_cycles = 0;
    uint64_t wait_cycles = 0;
    uint64_t mem_cycles = 0;
    uint64_t rd_bytes = 0;
    uint64_t wr_bytes = 0;
};

struct NafOptions
{
    bool intro_only = false;
    bool nafblock_only = false;
    int  block_c = 32;
    int  block_h = 64;
    int  block_w = 64;
};

struct NafConvReduceCoordinator
{
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

    static constexpr size_t no_task = std::numeric_limits<size_t>::max();

    int layer_id = -1;
    int worker_count = 0;
    int active_worker_count = 0;

    std::vector<int> active_tids;
    std::vector<int> leaf_index_by_tid;
    std::vector<NodeState> nodes;
    std::vector<TaskState> tasks;
    std::unique_ptr<sc_fifo<int>> ready_task_fifo;
    sc_mutex state_mutex;
    sc_mutex stats_mutex;

    size_t root_task_id = no_task;
    size_t root_node_id = no_task;
    bool   reduction_complete = false;
    bool   quant_started = false;
    bool   quant_finished = false;

    std::vector<sc_time> worker_mat_done_times;
    std::vector<sc_time> pair_start_times;
    std::vector<sc_time> pair_end_times;
    std::vector<sc_time> pair_left_ready_times;
    std::vector<sc_time> pair_right_ready_times;
    sc_time accum_end_time = SC_ZERO_TIME;
    sc_time quant_start_time = SC_ZERO_TIME;
    sc_time quant_end_time = SC_ZERO_TIME;

    NafConvReduceCoordinator(int layer_id_,
                             int worker_count_,
                             const std::vector<int> &active_tids_)
        : layer_id(layer_id_),
          worker_count(worker_count_),
          active_worker_count(static_cast<int>(active_tids_.size())),
          active_tids(active_tids_),
          leaf_index_by_tid(static_cast<size_t>(std::max(worker_count_, 0)), -1),
          worker_mat_done_times(static_cast<size_t>(std::max(worker_count_, 0)),
                                SC_ZERO_TIME)
    {
        build_reduction_tree();
    }

    void build_reduction_tree()
    {
        nodes.clear();
        tasks.clear();
        root_task_id = no_task;
        root_node_id = no_task;
        reduction_complete = false;
        quant_started = false;
        quant_finished = false;
        accum_end_time = SC_ZERO_TIME;
        quant_start_time = SC_ZERO_TIME;
        quant_end_time = SC_ZERO_TIME;
        std::fill(leaf_index_by_tid.begin(), leaf_index_by_tid.end(), -1);
        std::fill(worker_mat_done_times.begin(), worker_mat_done_times.end(), SC_ZERO_TIME);

        size_t n = active_tids.size();
        nodes.resize(n);

        std::vector<size_t> cur_nodes;
        cur_nodes.reserve(n);
        for (size_t i = 0; i < n; ++i)
        {
            cur_nodes.push_back(i);
            int tid = active_tids[i];
            if (tid >= 0 && tid < worker_count)
                leaf_index_by_tid[static_cast<size_t>(tid)] = static_cast<int>(i);
        }

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

        if (!cur_nodes.empty())
            root_node_id = cur_nodes.front();
        if (!tasks.empty())
            root_task_id = tasks.size() - 1;

        pair_start_times.assign(tasks.size(), SC_ZERO_TIME);
        pair_end_times.assign(tasks.size(), SC_ZERO_TIME);
        pair_left_ready_times.assign(tasks.size(), SC_ZERO_TIME);
        pair_right_ready_times.assign(tasks.size(), SC_ZERO_TIME);

        ready_task_fifo =
            std::make_unique<sc_fifo<int>>(sc_gen_unique_name("naf_ready_task_fifo"),
                                           static_cast<int>(tasks.size() + worker_count + 1));

        if (active_tids.empty())
            complete_reduction_locked(SC_ZERO_TIME);
    }

    void complete_reduction_locked(sc_time done_time)
    {
        if (reduction_complete)
            return;

        reduction_complete = true;
        accum_end_time = done_time;
        for (int i = 0; i < worker_count; ++i)
            ready_task_fifo->nb_write(-1);
    }

    void maybe_enqueue_task_locked(size_t task_id)
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

    void mark_worker_mat_done(int tid, sc_time ready_time)
    {
        state_mutex.lock();

        if (tid >= 0 && tid < worker_count)
            worker_mat_done_times[static_cast<size_t>(tid)] = ready_time;

        size_t leaf_id = no_task;
        if (tid >= 0 && tid < worker_count)
        {
            int leaf = leaf_index_by_tid[static_cast<size_t>(tid)];
            if (leaf >= 0)
                leaf_id = static_cast<size_t>(leaf);
        }

        if (leaf_id != no_task && leaf_id < nodes.size())
        {
            nodes[leaf_id].ready = true;
            nodes[leaf_id].ready_time = ready_time;

            if (tasks.empty() && leaf_id == root_node_id)
                complete_reduction_locked(ready_time);
            else
                maybe_enqueue_task_locked(nodes[leaf_id].parent_task);
        }
        else if (!reduction_complete && active_tids.empty())
        {
            complete_reduction_locked(ready_time);
        }

        state_mutex.unlock();
    }

    bool claim_task(size_t task_id, sc_time &left_ready, sc_time &right_ready)
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

    bool next_ready_task(size_t &task_id, sc_time &left_ready, sc_time &right_ready)
    {
        while (true)
        {
            int item = ready_task_fifo->read();
            if (item < 0)
                return false;

            size_t candidate = static_cast<size_t>(item);
            if (claim_task(candidate, left_ready, right_ready))
            {
                task_id = candidate;
                return true;
            }
        }
    }

    void finish_task(size_t task_id,
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

        if (task_id == root_task_id)
            complete_reduction_locked(end_time);
        else
            maybe_enqueue_task_locked(nodes[task.out].parent_task);

        state_mutex.unlock();
    }

    void note_quant_start(sc_time start_time)
    {
        stats_mutex.lock();
        if (!quant_started)
        {
            quant_started = true;
            quant_start_time = start_time;
        }
        stats_mutex.unlock();
    }

    void note_quant_end(sc_time end_time)
    {
        stats_mutex.lock();
        if (!quant_finished || end_time > quant_end_time)
        {
            quant_finished = true;
            quant_end_time = end_time;
        }
        stats_mutex.unlock();
    }

    sc_time earliest_pair_start() const
    {
        sc_time earliest = SC_ZERO_TIME;
        bool found = false;
        for (const auto &t : pair_start_times)
        {
            if (t == SC_ZERO_TIME && !found)
                continue;
            if (!found || t < earliest)
            {
                earliest = t;
                found = true;
            }
        }
        return earliest;
    }

    sc_time latest_active_mat_done() const
    {
        sc_time latest = SC_ZERO_TIME;
        for (int tid : active_tids)
        {
            if (tid >= 0 && tid < worker_count)
                latest = std::max(latest, worker_mat_done_times[static_cast<size_t>(tid)]);
        }
        return latest;
    }
};

struct NafWorker : sc_module
{
    tlm_utils::simple_initiator_socket<NafWorker> init;
    tlm_utils::peq_with_get<tlm_generic_payload>  peq;

    int      tid;
    int      n_workers;
    Barrier *barrier;

    const std::vector<LayerDesc> &layers;
    const std::vector<NafConvReduceCoordinator *> &conv_reduce;
    std::vector<NafLayerStats>    layer_stats;

    uint64_t total_dispatch_cycles = 0;
    uint64_t total_cpu_cycles      = 0;
    uint64_t total_accel_cycles    = 0;
    uint64_t total_wait_cycles     = 0;
    uint64_t total_mem_cycles      = 0;
    uint64_t mat_calls             = 0;
    uint64_t vec_calls             = 0;
    uint64_t mem_calls             = 0;
    uint64_t elapsed_cycles        = 0;

    struct DoneEntry
    {
        sc_event *ev       = nullptr;
        sc_event *admit_ev = nullptr;
        bool      fired    = false;
    };

    std::unordered_map<tlm_generic_payload *, DoneEntry *> done_map;

    struct PendingReq
    {
        tlm_generic_payload *gp           = nullptr;
        ReqExt              *req_ext      = nullptr;
        TxnExt              *tx_ext       = nullptr;
        NafLayerExt         *naf_ext      = nullptr;
        DoneEntry           *done_entry   = nullptr;
        uint64_t             svc_cyc      = 0;
        uint64_t             stall_cycles = 0;
        sc_time              submit_time  = SC_ZERO_TIME;
        bool                 direct_mem   = false;
        bool                 sync_done    = false;
    };

    SC_HAS_PROCESS(NafWorker);

    NafWorker(sc_module_name name,
              int tid_,
              int n_workers_,
              const std::vector<LayerDesc> &layers_,
              const std::vector<NafConvReduceCoordinator *> &conv_reduce_,
              Barrier *barrier_)
        : sc_module(name),
          init("init"),
          peq("peq"),
          tid(tid_),
          n_workers(n_workers_),
          barrier(barrier_),
          layers(layers_),
          conv_reduce(conv_reduce_),
          layer_stats(layers_.size())
    {
        init.register_nb_transport_bw(this, &NafWorker::nb_transport_bw);
        SC_THREAD(peq_thread);
        SC_THREAD(run);
    }

    tlm_sync_enum nb_transport_bw(tlm_generic_payload &gp,
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

    void peq_thread()
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

    void do_scalar(uint64_t cyc)
    {
        total_dispatch_cycles += cyc;
        wait(cyc * CYCLE);
    }

    PendingReq issue_begin(uint64_t addr,
                           uint64_t svc_cyc,
                           uint64_t rd,
                           uint64_t wr,
                           int layer_id,
                           LayerOpKind op_kind,
                           LayerBackend backend)
    {
        PendingReq p;
        p.svc_cyc = svc_cyc;

        auto *gp = new tlm_generic_payload();
        gp->set_command(TLM_IGNORE_COMMAND);
        gp->set_address(addr);
        gp->set_data_ptr(nullptr);
        gp->set_data_length(0);
        gp->set_streaming_width(0);
        gp->set_response_status(TLM_INCOMPLETE_RESPONSE);

        auto *req = new ReqExt(tid, svc_cyc, rd, wr);
        auto *tx  = new TxnExt();
        auto *naf = new NafLayerExt();

        tx->src_worker = tid;
        naf->layer_id = layer_id;
        naf->op_kind = op_kind;
        naf->backend = backend;

        gp->set_extension(req);
        gp->set_extension(tx);
        gp->set_extension(naf);

        p.gp         = gp;
        p.req_ext    = req;
        p.tx_ext     = tx;
        p.naf_ext    = naf;
        p.done_entry = new DoneEntry();
        p.done_entry->ev       = new sc_event();
        p.done_entry->admit_ev = new sc_event();
        done_map[gp] = p.done_entry;

        tlm_phase phase = BEGIN_REQ;
        sc_time   delay = SC_ZERO_TIME;
        auto status = init->nb_transport_fw(*gp, phase, delay);

        if (status == TLM_ACCEPTED)
        {
            std::string wname = "worker_" + std::to_string(tid);
            wave_log((uint64_t)(sc_time_stamp() / CYCLE), wname, 0);
            sc_time t_stall_start = sc_time_stamp();
            wait(*p.done_entry->admit_ev);
            p.stall_cycles = (uint64_t)((sc_time_stamp() - t_stall_start) / CYCLE);
            wave_log((uint64_t)(sc_time_stamp() / CYCLE), wname, 1);
        }
        else if (status == TLM_COMPLETED)
        {
            done_map.erase(gp);
            p.sync_done = true;
        }

        return p;
    }

    PendingReq issue_mem_access(bool is_write,
                                uint64_t bytes,
                                int layer_id,
                                LayerOpKind op_kind,
                                LayerBackend backend)
    {
        PendingReq p;
        p.direct_mem = true;

        auto *gp = new tlm_generic_payload();
        gp->set_command(is_write ? TLM_WRITE_COMMAND : TLM_READ_COMMAND);
        gp->set_address(Interconnect::ADDR_MEM);
        gp->set_data_ptr(nullptr);
        gp->set_data_length((unsigned)bytes);
        gp->set_streaming_width((unsigned)bytes);
        gp->set_response_status(TLM_INCOMPLETE_RESPONSE);

        auto *tx  = new TxnExt();
        auto *naf = new NafLayerExt();
        tx->src_worker = tid;
        naf->layer_id = layer_id;
        naf->op_kind = op_kind;
        naf->backend = backend;

        gp->set_extension(tx);
        gp->set_extension(naf);

        p.gp         = gp;
        p.tx_ext     = tx;
        p.naf_ext    = naf;
        p.done_entry = new DoneEntry();
        p.done_entry->ev       = new sc_event();
        p.done_entry->admit_ev = new sc_event();
        p.submit_time = sc_time_stamp();
        done_map[gp] = p.done_entry;

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

    void issue_end(PendingReq &p, NafLayerStats &s)
    {
        if (!p.sync_done && !p.done_entry->fired)
        {
            std::string wname = "worker_" + std::to_string(tid);
            wave_log((uint64_t)(sc_time_stamp() / CYCLE), wname, 0);
            wait(*p.done_entry->ev);
            wave_log((uint64_t)(sc_time_stamp() / CYCLE), wname, 1);
        }

        done_map.erase(p.gp);
        delete p.done_entry->ev;
        delete p.done_entry->admit_ev;
        delete p.done_entry;
        p.done_entry = nullptr;

        ReqExt *ext = nullptr;
        p.gp->get_extension(ext);
        uint64_t qwait = ext ? ext->accel_qwait_cycles : 0;
        uint64_t mec   = ext ? ext->mem_cycles : 0;

        if (p.direct_mem)
        {
            uint64_t direct_mem_cycles =
                (uint64_t)((sc_time_stamp() - p.submit_time) / CYCLE);
            total_mem_cycles += direct_mem_cycles;
            s.mem_cycles     += direct_mem_cycles;
        }
        else
        {
            total_accel_cycles += p.svc_cyc;
            total_wait_cycles  += qwait + p.stall_cycles;
            total_mem_cycles   += mec;
            s.wait_cycles      += qwait + p.stall_cycles;
            s.mem_cycles       += mec;
        }

        tlm_phase end_phase = END_RESP;
        sc_time   end_delay = SC_ZERO_TIME;
        init->nb_transport_fw(*p.gp, end_phase, end_delay);

        p.gp->clear_extension(p.req_ext);
        p.gp->clear_extension(p.tx_ext);
        p.gp->clear_extension(p.naf_ext);
        delete p.req_ext; p.req_ext = nullptr;
        delete p.tx_ext;  p.tx_ext  = nullptr;
        delete p.naf_ext; p.naf_ext = nullptr;
        delete p.gp;      p.gp      = nullptr;
    }

    void note_accel_issue(NafLayerStats &s,
                          bool is_mat,
                          uint64_t svc_cyc,
                          uint64_t rd,
                          uint64_t wr)
    {
        if (is_mat)
        {
            ++mat_calls;
            ++s.mat_reqs;
        }
        else
        {
            ++vec_calls;
            ++s.vec_reqs;
        }
        s.accel_cycles += svc_cyc;
        s.rd_bytes     += rd;
        s.wr_bytes     += wr;
    }

    void note_mem_issue(NafLayerStats &s, uint64_t rd, uint64_t wr)
    {
        ++mem_calls;
        ++s.mem_reqs;
        s.rd_bytes += rd;
        s.wr_bytes += wr;
    }

    void run_conv_layer(const LayerDesc &l)
    {
        NafLayerStats &s = layer_stats[l.id];
        NafConvReduceCoordinator *coord =
            (l.id >= 0 && static_cast<size_t>(l.id) < conv_reduce.size())
                ? conv_reduce[static_cast<size_t>(l.id)]
                : nullptr;

        uint64_t local_mat_reqs = naf_conv_local_mat_reqs(l, tid, n_workers);
        std::vector<PendingReq> pending;
        pending.reserve((size_t)local_mat_reqs);

        for (uint64_t i = 0; i < local_mat_reqs; ++i)
        {
            uint64_t rd = naf_matmul_a_bytes() + naf_matmul_b_bytes();
            uint64_t wr = naf_matmul_c_bytes();
            auto req = issue_begin(Interconnect::ADDR_MAT,
                                   MATMUL_ACC_CYCLE,
                                   rd, wr,
                                   l.id, l.op_kind, l.backend);
            note_accel_issue(s, true, MATMUL_ACC_CYCLE, rd, wr);
            do_scalar(SCALAR_OVERHEAD);
            pending.push_back(std::move(req));
        }

        for (auto &req : pending)
            issue_end(req, s);

        if (coord)
            coord->mark_worker_mat_done(tid, sc_time_stamp());

        uint64_t reduce_calls = naf_conv_reduce_calls(l);
        if (coord)
        {
            size_t task_id = NafConvReduceCoordinator::no_task;
            sc_time left_ready = SC_ZERO_TIME;
            sc_time right_ready = SC_ZERO_TIME;
            while (coord->next_ready_task(task_id, left_ready, right_ready))
            {
                sc_time pair_start = sc_time_stamp();
                std::vector<PendingReq> reduce_pending;
                reduce_pending.reserve((size_t)reduce_calls);
                for (uint64_t call = 0; call < reduce_calls; ++call)
                {
                    uint64_t rd = naf_conv_reduce_rd_bytes();
                    uint64_t wr = naf_conv_reduce_wr_bytes();
                    auto req = issue_begin(Interconnect::ADDR_VEC,
                                           VECTOR_ACC_CYCLE,
                                           rd, wr,
                                           l.id, l.op_kind, l.backend);
                    note_accel_issue(s, false, VECTOR_ACC_CYCLE, rd, wr);
                    do_scalar(SCALAR_OVERHEAD);
                    reduce_pending.push_back(std::move(req));
                }
                for (auto &req : reduce_pending)
                    issue_end(req, s);
                coord->finish_task(task_id, pair_start, sc_time_stamp(),
                                   left_ready, right_ready);
            }
        }

        uint64_t quant_calls = even_share(naf_conv_quant_calls(l), tid, n_workers);
        if (quant_calls > 0)
        {
            if (coord)
                coord->note_quant_start(sc_time_stamp());

            std::vector<PendingReq> quant_pending;
            quant_pending.reserve((size_t)quant_calls);
            for (uint64_t call = 0; call < quant_calls; ++call)
            {
                uint64_t rd = naf_conv_quant_rd_bytes();
                uint64_t wr = naf_conv_quant_wr_bytes();
                auto req = issue_begin(Interconnect::ADDR_VEC,
                                       VECTOR_ACC_CYCLE,
                                       rd, wr,
                                       l.id, l.op_kind, l.backend);
                note_accel_issue(s, false, VECTOR_ACC_CYCLE, rd, wr);
                do_scalar(SCALAR_OVERHEAD);
                quant_pending.push_back(std::move(req));
            }
            for (auto &req : quant_pending)
                issue_end(req, s);

            if (coord)
                coord->note_quant_end(sc_time_stamp());
        }
        else if (coord)
        {
            coord->note_quant_start(sc_time_stamp());
            coord->note_quant_end(sc_time_stamp());
        }
    }

    void run_dwconv_layer(const LayerDesc &l)
    {
        NafLayerStats &s = layer_stats[l.id];
        auto [c_start, c_end] = channel_range(l.Cout, tid, n_workers);
        uint64_t strips_per_row = cdiv64((uint64_t)l.Wout, DW_VEC_ACC_CAP);

        for (int c = c_start; c < c_end; ++c)
        {
            (void)c;
            for (int oh = 0; oh < l.Hout; ++oh)
            {
                std::vector<PendingReq> pending;
                pending.reserve((size_t)strips_per_row);

                for (uint64_t strip = 0; strip < strips_per_row; ++strip)
                {
                    uint64_t strip_start = strip * DW_VEC_ACC_CAP;
                    uint64_t vl = std::min<uint64_t>(DW_VEC_ACC_CAP,
                                                     (uint64_t)l.Wout - strip_start);
                    uint64_t rd = naf_dwconv_strip_rd_bytes(l, oh, strip_start, vl);
                    if (oh == 0 && strip == 0)
                        rd += (uint64_t)l.Kh * l.Kw * DW_INPUT_ELEM_BYTES;
                    uint64_t wr = vl * DW_OUTPUT_ELEM_BYTES;
                    auto req = issue_begin(Interconnect::ADDR_VEC,
                                           DW_VEC_ACC_CYCLE,
                                           rd, wr,
                                           l.id, l.op_kind, l.backend);
                    note_accel_issue(s, false, DW_VEC_ACC_CYCLE, rd, wr);
                    do_scalar(DW_SCALAR_OVERHEAD);
                    pending.push_back(std::move(req));
                }

                for (auto &req : pending)
                    issue_end(req, s);
            }
        }
    }

    void run_layernorm_layer(const LayerDesc &l)
    {
        NafLayerStats &s = layer_stats[l.id];
        auto [c_start, c_end] = channel_range(l.Cout, tid, n_workers);
        uint64_t spatial = (uint64_t)l.Hin * l.Win;
        uint64_t n_tiles = cdiv64(spatial, LN_VEC_ACC_CAP);
        const uint64_t elem_bytes = 2;

        for (int c = c_start; c < c_end; ++c)
        {
            (void)c;

            for (int step = 0; step < 2; ++step)
            {
                std::vector<PendingReq> pending;
                pending.reserve((size_t)n_tiles);

                for (uint64_t t = 0; t < n_tiles; ++t)
                {
                    uint64_t tile_elems = std::min<uint64_t>(LN_VEC_ACC_CAP, spatial - t * LN_VEC_ACC_CAP);
                    uint64_t rd = tile_elems * elem_bytes;
                    auto req = issue_begin(Interconnect::ADDR_VEC,
                                           LN_VEC_ACC_CYCLE,
                                           rd, 0,
                                           l.id, l.op_kind, l.backend);
                    note_accel_issue(s, false, LN_VEC_ACC_CYCLE, rd, 0);
                    do_scalar(LN_SCALAR_OVERHEAD);
                    pending.push_back(std::move(req));
                }

                for (auto &req : pending)
                    issue_end(req, s);
            }

            total_cpu_cycles += LN_STEP3_CYCLES;
            s.cpu_cycles     += LN_STEP3_CYCLES;
            wait(LN_STEP3_CYCLES * CYCLE);

            std::vector<PendingReq> pending;
            pending.reserve((size_t)n_tiles);
            for (uint64_t t = 0; t < n_tiles; ++t)
            {
                uint64_t tile_elems = std::min<uint64_t>(LN_VEC_ACC_CAP, spatial - t * LN_VEC_ACC_CAP);
                uint64_t rd = tile_elems * elem_bytes + 2 * elem_bytes;
                uint64_t wr = tile_elems * elem_bytes;
                auto req = issue_begin(Interconnect::ADDR_VEC,
                                       LN_VEC_ACC_CYCLE,
                                       rd, wr,
                                       l.id, l.op_kind, l.backend);
                note_accel_issue(s, false, LN_VEC_ACC_CYCLE, rd, wr);
                do_scalar(LN_SCALAR_OVERHEAD);
                pending.push_back(std::move(req));
            }
            for (auto &req : pending)
                issue_end(req, s);
        }
    }

    void run_vecop_phase(const LayerDesc &l, VopType op, NafLayerStats &s)
    {
        auto [c_start, c_end] = channel_range(l.Cout, tid, n_workers);
        uint64_t spatial = (uint64_t)l.Hout * l.Wout;
        uint64_t tile_cap = vop_tile_cap_elems(op);
        uint64_t n_tiles = cdiv64(spatial, tile_cap);

        for (int c = c_start; c < c_end; ++c)
        {
            (void)c;
            uint64_t extra_rd = vop_extra_rd_bytes_per_channel(op);
            if (extra_rd > 0)
            {
                auto mem_req = issue_mem_access(false, extra_rd, l.id, l.op_kind, l.backend);
                note_mem_issue(s, extra_rd, 0);
                issue_end(mem_req, s);
            }

            std::vector<PendingReq> pending;
            pending.reserve((size_t)n_tiles);
            for (uint64_t t = 0; t < n_tiles; ++t)
            {
                uint64_t tile_elems = std::min<uint64_t>(tile_cap, spatial - t * tile_cap);
                uint64_t rd = vop_rd_bytes(op, tile_elems);
                uint64_t wr = vop_wr_bytes(op, tile_elems);
                auto req = issue_begin(Interconnect::ADDR_VEC,
                                       VOP_VEC_ACC_CYCLE,
                                       rd, wr,
                                       l.id, l.op_kind, l.backend);
                note_accel_issue(s, false, VOP_VEC_ACC_CYCLE, rd, wr);
                do_scalar(VOP_SCALAR_OVERHEAD);
                pending.push_back(std::move(req));
            }
            for (auto &req : pending)
                issue_end(req, s);
        }
    }

    void run_pool_layer(const LayerDesc &l)
    {
        NafLayerStats &s = layer_stats[l.id];
        auto [c_start, c_end] = channel_range(l.Cin, tid, n_workers);
        uint64_t spatial = (uint64_t)l.Hin * l.Win;
        uint64_t n_tiles = cdiv64(spatial, POOL_VEC_ACC_CAP);

        for (int c = c_start; c < c_end; ++c)
        {
            std::vector<PendingReq> pending;
            pending.reserve((size_t)n_tiles);

            for (uint64_t t = 0; t < n_tiles; ++t)
            {
                uint64_t tile_elems = std::min<uint64_t>(POOL_VEC_ACC_CAP, spatial - t * POOL_VEC_ACC_CAP);
                uint64_t rd = tile_elems * POOL_INPUT_ELEM_BYTES;
                auto req = issue_begin(Interconnect::ADDR_VEC,
                                       POOL_VEC_ACC_CYCLE,
                                       rd, 0,
                                       l.id, l.op_kind, l.backend);
                note_accel_issue(s, false, POOL_VEC_ACC_CYCLE, rd, 0);
                do_scalar(POOL_SCALAR_OVERHEAD);
                pending.push_back(std::move(req));
            }
            for (auto &req : pending)
                issue_end(req, s);

            total_cpu_cycles += POOL_DIVIDE_CYCLES;
            s.cpu_cycles     += POOL_DIVIDE_CYCLES;
            wait(POOL_DIVIDE_CYCLES * CYCLE);

            auto store_req = issue_mem_access(true, POOL_OUTPUT_ELEM_BYTES,
                                              l.id, l.op_kind, l.backend);
            note_mem_issue(s, 0, POOL_OUTPUT_ELEM_BYTES);
            issue_end(store_req, s);
            (void)c;
        }
    }

    void run_layer(const LayerDesc &l)
    {
        switch (l.backend)
        {
        case BACKEND_MATMUL:
            run_conv_layer(l);
            break;
        case BACKEND_DWCONV:
            run_dwconv_layer(l);
            break;
        case BACKEND_LAYERNORM:
            run_layernorm_layer(l);
            break;
        case BACKEND_POOLING:
            run_pool_layer(l);
            break;
        case BACKEND_VECOPS:
            run_vecop_phase(l, l.primary_vop, layer_stats[l.id]);
            if (l.phase_count > 1)
                run_vecop_phase(l, l.secondary_vop, layer_stats[l.id]);
            break;
        }
    }

    void run()
    {
        std::string wname = "worker_" + std::to_string(tid);
        sc_time t_start = sc_time_stamp();
        wave_log((uint64_t)(t_start / CYCLE), wname, 1);

        for (const LayerDesc &l : layers)
        {
            wave_log((uint64_t)(sc_time_stamp() / CYCLE), wname, 0);
            barrier->sync();
            wave_log((uint64_t)(sc_time_stamp() / CYCLE), wname, 1);
            run_layer(l);
        }

        wave_log((uint64_t)(sc_time_stamp() / CYCLE), wname, 0);

        sc_time t_end = sc_time_stamp();
        elapsed_cycles = (uint64_t)((t_end - t_start) / CYCLE);
    }
};

struct NafTop : sc_module
{
    AcceleratorPool mat_acc;
    AcceleratorPool vec_acc;
    Interconnect    noc;
    Memory          memory;
    Barrier         barrier;

    std::vector<NafWorker *> workers;
    std::vector<LayerDesc>   layers;
    std::vector<std::unique_ptr<NafConvReduceCoordinator>> conv_reduce_owned;
    std::vector<NafConvReduceCoordinator *>                conv_reduce;
    NafOptions               opts;

    SC_HAS_PROCESS(NafTop);

    NafTop(sc_module_name name, const NafOptions &opts_)
        : sc_module(name),
          mat_acc("mat_acc",
                  static_cast<size_t>(MAT_ACCEL_COUNT_CFG),
                  std::max(HW_ACC_QUEUE_DEPTH,
                           static_cast<size_t>(MAT_ACCEL_COUNT_CFG * 4))),
          vec_acc("vec_acc",
                  static_cast<size_t>(VEC_ACCEL_COUNT_CFG),
                  std::max(HW_ACC_QUEUE_DEPTH,
                           static_cast<size_t>(VEC_ACCEL_COUNT_CFG * 4))),
          noc("noc"),
          memory("memory",
                 HW_MEMORY_BASE_LAT,
                 HW_MEMORY_BYTES_PER_CYCLE,
                 MEMORY_PARALLEL_SLOTS_CFG),
          barrier(N_WORKERS),
          opts(opts_)
    {
        noc.to_mat.bind(mat_acc.tgt);
        noc.to_vec.bind(vec_acc.tgt);
        noc.to_mem.bind(memory.tgt);

        for (auto &unit : mat_acc.units)
            unit->to_mem.bind(noc.tgt);
        for (auto &unit : vec_acc.units)
            unit->to_mem.bind(noc.tgt);

        if (opts.nafblock_only)
            layers = build_nafblock_only_layers(opts.block_c, opts.block_h, opts.block_w);
        else
            layers = build_nafnet32_layers();

        if (opts.intro_only && !opts.nafblock_only)
            layers.resize(1);

        conv_reduce.resize(layers.size(), nullptr);
        conv_reduce_owned.reserve(layers.size());
        for (const auto &l : layers)
        {
            if (l.backend != BACKEND_MATMUL)
                continue;

            std::vector<int> active_tids;
            active_tids.reserve((size_t)N_WORKERS);
            for (int tid = 0; tid < N_WORKERS; ++tid)
            {
                if (naf_conv_local_k_extent(l, tid, N_WORKERS) > 0)
                    active_tids.push_back(tid);
            }

            auto coord = std::make_unique<NafConvReduceCoordinator>(l.id,
                                                                    N_WORKERS,
                                                                    active_tids);
            conv_reduce[static_cast<size_t>(l.id)] = coord.get();
            conv_reduce_owned.push_back(std::move(coord));
        }

        for (int i = 0; i < N_WORKERS; ++i)
        {
            auto *w = new NafWorker(sc_gen_unique_name("naf_worker"),
                                    i, N_WORKERS, layers, conv_reduce, &barrier);
            workers.push_back(w);
            w->init.bind(noc.tgt);
        }
    }

    ~NafTop() override
    {
        for (auto *w : workers)
            delete w;
    }
};

static bool parse_args(int argc, char *argv[], NafOptions &opts)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--intro-only")
        {
            opts.intro_only = true;
        }
        else if (arg == "--nafblock-only")
        {
            opts.nafblock_only = true;
        }
        else if (arg == "--block-c" || arg == "--block-h" || arg == "--block-w")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for " << arg << "\n";
                return false;
            }
            int value = std::stoi(argv[++i]);
            if (value <= 0)
            {
                std::cerr << "Invalid value for " << arg << ": " << value << "\n";
                return false;
            }
            if (arg == "--block-c") opts.block_c = value;
            if (arg == "--block-h") opts.block_h = value;
            if (arg == "--block-w") opts.block_w = value;
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout
                << "Usage: " << argv[0]
                << " [--intro-only] [--nafblock-only] [--block-c N] [--block-h N] [--block-w N]\n";
            return false;
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }
    return true;
}

static void print_nafblock_verification(const std::vector<LayerDesc> &layers,
                                        const std::vector<NafLayerStats> &global,
                                        const std::vector<NafConvReduceCoordinator *> &conv_reduce,
                                        int n_workers)
{
    std::string manifest_error;
    bool manifest_ok = validate_nafblock_manifest(layers, "nafblock",
                                                  layers[0].Cin,
                                                  layers[0].Hin,
                                                  layers[0].Win,
                                                  &manifest_error);

    std::cout << "\n--- NafBlock Verification ---\n";
    std::cout << "["
              << (manifest_ok ? "PASS" : "FAIL")
              << "] manifest"
              << (manifest_ok ? "" : (": " + manifest_error))
              << "\n";

    bool analytics_ok = true;
    for (size_t i = 0; i < layers.size(); ++i)
    {
        LayerExpectedStats exp = expected_layer_stats(layers[i], n_workers);
        const NafLayerStats &obs = global[i];
        bool ok =
            exp.mat_reqs == obs.mat_reqs &&
            exp.vec_reqs == obs.vec_reqs &&
            exp.mem_reqs == obs.mem_reqs &&
            exp.accel_cycles == obs.accel_cycles &&
            exp.cpu_cycles == obs.cpu_cycles &&
            exp.rd_bytes == obs.rd_bytes &&
            exp.wr_bytes == obs.wr_bytes;
        analytics_ok = analytics_ok && ok;
        std::cout << "["
                  << (ok ? "PASS" : "FAIL")
                  << "] " << layers[i].name
                  << "  reqs(mat/vec/mem)="
                  << obs.mat_reqs << "/" << obs.vec_reqs << "/" << obs.mem_reqs
                  << "  bytes(rd/wr)=" << obs.rd_bytes << "/" << obs.wr_bytes
                  << "\n";
    }

    std::cout << "["
              << (analytics_ok ? "PASS" : "FAIL")
              << "] aggregate analytic counts\n";

    bool timing_ok = true;
    for (const auto &l : layers)
    {
        if (l.backend != BACKEND_MATMUL)
            continue;

        const auto *coord =
            (l.id >= 0 && static_cast<size_t>(l.id) < conv_reduce.size())
                ? conv_reduce[static_cast<size_t>(l.id)]
                : nullptr;
        if (!coord)
            continue;

        bool deps_ok = true;
        for (size_t i = 0; i < coord->pair_start_times.size(); ++i)
        {
            if (coord->pair_start_times[i] < coord->pair_left_ready_times[i] ||
                coord->pair_start_times[i] < coord->pair_right_ready_times[i])
            {
                deps_ok = false;
                break;
            }
        }

        bool quant_ok = !coord->quant_started ||
                        coord->quant_start_time >= coord->accum_end_time;
        bool pair_count_ok =
            coord->tasks.size() ==
            static_cast<size_t>(std::max(coord->active_worker_count - 1, 0));

        timing_ok = timing_ok && deps_ok && quant_ok && pair_count_ok;
        std::cout << "["
                  << ((deps_ok && quant_ok && pair_count_ok) ? "PASS" : "FAIL")
                  << "] " << l.name
                  << "  active=" << coord->active_worker_count
                  << "  pairs=" << coord->tasks.size()
                  << "  accum_end=" << coord->accum_end_time
                  << "  quant_start=" << coord->quant_start_time
                  << "\n";

        sc_time earliest_pair = coord->earliest_pair_start();
        sc_time latest_mat = coord->latest_active_mat_done();
        if (!coord->pair_start_times.empty())
        {
            if (earliest_pair < latest_mat)
            {
                std::cout << "[PASS] " << l.name
                          << " early-start overlap observed: first_pair=" << earliest_pair
                          << "  latest_active_mat=" << latest_mat << "\n";
            }
            else
            {
                std::cout << "[INFO] " << l.name
                          << " no early-start overlap observed in this run: first_pair="
                          << earliest_pair
                          << "  latest_active_mat=" << latest_mat << "\n";
            }
        }
        else
        {
            std::cout << "[INFO] " << l.name
                      << " reduction tree skipped (active workers="
                      << coord->active_worker_count << ")\n";
        }
    }

    std::cout << "["
              << (timing_ok ? "PASS" : "FAIL")
              << "] reduction timing invariants\n";
}

int sc_main(int argc, char *argv[])
{
    NafOptions opts;
    if (!parse_args(argc, argv, opts))
        return (argc > 1) ? 1 : 0;

    if (opts.nafblock_only)
    {
        std::cout << "[Mode] nafblock-only: C=" << opts.block_c
                  << " H=" << opts.block_h
                  << " W=" << opts.block_w << "\n";
    }
    else if (opts.intro_only)
    {
        std::cout << "[Mode] intro-only: simulating only the intro CONV layer\n";
    }

    NafTop top("nafnet_top", opts);

    if (opts.nafblock_only)
    {
        std::string err;
        if (!validate_nafblock_manifest(top.layers, "nafblock",
                                        opts.block_c, opts.block_h, opts.block_w, &err))
        {
            std::cerr << "NafBlock manifest validation failed before simulation: "
                      << err << "\n";
            return 2;
        }
    }

    sc_start();

    {
        auto &evts = wave_events();
        std::stable_sort(evts.begin(), evts.end(),
                         [](const WaveEvent &a, const WaveEvent &b) {
                             return a.cycle < b.cycle;
                         });

        const char *vcd_name = opts.nafblock_only
                             ? "waveform_nafblock.vcd"
                             : (opts.intro_only ? "waveform_intro.vcd" : "waveform.vcd");
        write_vcd(evts, vcd_name);
        std::cout << "VCD waveform written to " << vcd_name
                  << " (" << evts.size() << " events)\n";
    }

    const auto  &layers = top.layers;
    const size_t n_layers = layers.size();
    std::vector<NafLayerStats> global(n_layers);

    for (const auto *w : top.workers)
    {
        for (size_t i = 0; i < n_layers; ++i)
        {
            const NafLayerStats &ws = w->layer_stats[i];
            global[i].mat_reqs     += ws.mat_reqs;
            global[i].vec_reqs     += ws.vec_reqs;
            global[i].mem_reqs     += ws.mem_reqs;
            global[i].accel_cycles += ws.accel_cycles;
            global[i].cpu_cycles   += ws.cpu_cycles;
            global[i].wait_cycles  += ws.wait_cycles;
            global[i].mem_cycles   += ws.mem_cycles;
            global[i].rd_bytes     += ws.rd_bytes;
            global[i].wr_bytes     += ws.wr_bytes;
        }
    }

    std::cout << "\n====================================================================\n";
    std::cout << "  NAFNet Performance Simulation Report"
              << "  (" << N_WORKERS << " workers)\n";
    std::cout << "====================================================================\n";

    std::cout << "\n--- Per-Layer Summary ---\n";
    std::cout << std::left
              << std::setw(4)  << "ID"
              << std::setw(24) << "Name"
              << std::setw(12) << "Op"
              << std::setw(11) << "Backend"
              << std::setw(7)  << "Phases"
              << std::setw(8)  << "MatReq"
              << std::setw(8)  << "VecReq"
              << std::setw(8)  << "MemReq"
              << std::setw(12) << "AccelCyc"
              << std::setw(10) << "CpuCyc"
              << std::setw(10) << "WaitCyc"
              << std::setw(10) << "MemCyc"
              << std::setw(12) << "RdBytes"
              << std::setw(12) << "WrBytes"
              << "\n";
    std::cout << std::string(156, '-') << "\n";

    for (size_t i = 0; i < n_layers; ++i)
    {
        const LayerDesc &l = layers[i];
        const NafLayerStats &s = global[i];
        std::cout << std::left
                  << std::setw(4)  << l.id
                  << std::setw(24) << l.name
                  << std::setw(12) << layer_op_kind_str(l.op_kind)
                  << std::setw(11) << layer_backend_str(l.backend)
                  << std::setw(7)  << l.phase_count
                  << std::setw(8)  << s.mat_reqs
                  << std::setw(8)  << s.vec_reqs
                  << std::setw(8)  << s.mem_reqs
                  << std::setw(12) << s.accel_cycles
                  << std::setw(10) << s.cpu_cycles
                  << std::setw(10) << s.wait_cycles
                  << std::setw(10) << s.mem_cycles
                  << std::setw(12) << s.rd_bytes
                  << std::setw(12) << s.wr_bytes
                  << "\n";
    }

    std::cout << "\n--- Per-Worker Summary ---\n";
    std::cout << std::left
              << std::setw(8)  << "Worker"
              << std::setw(12) << "Dispatch"
              << std::setw(10) << "CpuCyc"
              << std::setw(12) << "AccelCyc"
              << std::setw(10) << "WaitCyc"
              << std::setw(10) << "MemCyc"
              << std::setw(10) << "MatReq"
              << std::setw(10) << "VecReq"
              << std::setw(10) << "MemReq"
              << std::setw(12) << "Elapsed"
              << "\n";
    std::cout << std::string(104, '-') << "\n";

    uint64_t max_elapsed = 0;
    for (const auto *w : top.workers)
    {
        std::cout << std::left
                  << std::setw(8)  << w->tid
                  << std::setw(12) << w->total_dispatch_cycles
                  << std::setw(10) << w->total_cpu_cycles
                  << std::setw(12) << w->total_accel_cycles
                  << std::setw(10) << w->total_wait_cycles
                  << std::setw(10) << w->total_mem_cycles
                  << std::setw(10) << w->mat_calls
                  << std::setw(10) << w->vec_calls
                  << std::setw(10) << w->mem_calls
                  << std::setw(12) << w->elapsed_cycles
                  << "\n";
        max_elapsed = std::max(max_elapsed, w->elapsed_cycles);
    }

    uint64_t total_mat_reqs = 0;
    uint64_t total_vec_reqs = 0;
    uint64_t total_mem_reqs = 0;
    uint64_t total_rd_bytes = 0;
    uint64_t total_wr_bytes = 0;
    uint64_t total_cpu_cycles = 0;
    for (const auto &s : global)
    {
        total_mat_reqs += s.mat_reqs;
        total_vec_reqs += s.vec_reqs;
        total_mem_reqs += s.mem_reqs;
        total_rd_bytes += s.rd_bytes;
        total_wr_bytes += s.wr_bytes;
        total_cpu_cycles += s.cpu_cycles;
    }

    std::cout << "\n--- Global Summary ---\n";
    std::cout << "Simulation time        : " << sc_time_stamp()    << "\n";
    std::cout << "Max worker elapsed     : " << max_elapsed         << " cycles\n";
    std::cout << "Workers                : " << N_WORKERS           << "\n";
    std::cout << "Layers                 : " << n_layers            << "\n";
    std::cout << "Total mat-acc requests : " << total_mat_reqs      << "\n";
    std::cout << "Total vec-acc requests : " << total_vec_reqs      << "\n";
    std::cout << "Total direct mem reqs  : " << total_mem_reqs      << "\n";
    std::cout << "Total scalar CPU cyc   : " << total_cpu_cycles    << "\n";
    std::cout << "Total memory reads     : " << total_rd_bytes      << " bytes\n";
    std::cout << "Total memory writes    : " << total_wr_bytes      << " bytes\n";
    std::cout << "mat_acc : units=" << top.mat_acc.instance_count()
              << "  reqs=" << top.mat_acc.req_count_total()
              << "  busy=" << top.mat_acc.busy_cycles_total()
              << "  qwait=" << top.mat_acc.queue_wait_cycles_total()
              << "\n";
    std::cout << "vec_acc : units=" << top.vec_acc.instance_count()
              << "  reqs=" << top.vec_acc.req_count_total()
              << "  busy=" << top.vec_acc.busy_cycles_total()
              << "  qwait=" << top.vec_acc.queue_wait_cycles_total()
              << "\n";
    std::cout << "memory  : reqs=" << top.memory.reqs
              << "  busy=" << top.memory.busy_cycles
              << "  qwait=" << top.memory.qwait_cycles
              << "\n";

    double sim_cycles = static_cast<double>(sc_time_stamp() / CYCLE);
    double mat_util = (sim_cycles > 0.0)
                    ? static_cast<double>(top.mat_acc.busy_cycles_total())
                        / (sim_cycles * static_cast<double>(top.mat_acc.instance_count())) * 100.0
                    : 0.0;
    double vec_util = (sim_cycles > 0.0)
                    ? static_cast<double>(top.vec_acc.busy_cycles_total())
                        / (sim_cycles * static_cast<double>(top.vec_acc.instance_count())) * 100.0
                    : 0.0;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "mat_acc utilisation    : " << mat_util << "%\n";
    std::cout << "vec_acc utilisation    : " << vec_util << "%\n";

    if (opts.nafblock_only)
        print_nafblock_verification(layers, global, top.conv_reduce, N_WORKERS);

    return 0;
}
