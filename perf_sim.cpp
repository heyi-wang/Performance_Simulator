// tlm_perf_sim_nb.cpp
#define SC_INCLUDE_DYNAMIC_PROCESSES

#include <systemc>
#include <tlm>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>
#include <tlm_utils/multi_passthrough_target_socket.h>
#include <tlm_utils/multi_passthrough_initiator_socket.h>
#include <tlm_utils/peq_with_get.h>

#include <deque>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <cstdint>
#include <algorithm>

#include "config.h"

using namespace sc_core;
using namespace tlm;

// ---------------------------
// Time base
// ---------------------------
static const sc_time CYCLE = sc_time(1, SC_NS);

// ---------------------------
// Helpers
// ---------------------------
static inline uint64_t ceil_div_u64(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

// ============================================================
// Request extension: accelerator + memory metadata
// ============================================================
struct ReqExt : tlm_extension<ReqExt>
{
    int src_id = -1;
    uint64_t cycles = 0;

    uint64_t accel_qwait_cycles = 0;
    uint64_t mem_cycles = 0;

    uint64_t rd_bytes = 0;
    uint64_t wr_bytes = 0;

    ReqExt() = default;
    ReqExt(int s, uint64_t a, uint64_t r, uint64_t w)
        : src_id(s), cycles(a), rd_bytes(r), wr_bytes(w) {}

    tlm_extension_base *clone() const override
    {
        return new ReqExt(*this);
    }

    void copy_from(const tlm_extension_base &ext) override
    {
        *this = static_cast<const ReqExt &>(ext);
    }
};

// ============================================================
// Transaction context extension
// Used to remember original source and completion event
// ============================================================
struct TxnExt : tlm_extension<TxnExt>
{
    int src_worker = -1;

    sc_event *done_ev = nullptr;

    // Used for routing responses back through interconnect
    int upstream_id = -1;

    tlm_extension_base *clone() const override
    {
        return new TxnExt(*this);
    }

    void copy_from(const tlm_extension_base &ext) override
    {
        *this = static_cast<const TxnExt &>(ext);
    }
};

// ============================================================
// Memory module (non-blocking target)
// Single-server FIFO timing model
// ============================================================
struct Memory : sc_module
{
    tlm_utils::simple_target_socket<Memory> tgt;

    uint64_t base_lat_cycles;
    uint64_t bytes_per_cycle;

    tlm_utils::peq_with_get<tlm_generic_payload> peq;

    struct Entry
    {
        tlm_generic_payload *gp = nullptr;
        sc_time enqueue_time;
    };

    std::deque<Entry> q;
    sc_event q_nonempty;

    uint64_t reqs = 0;
    uint64_t busy_cycles = 0;
    uint64_t qwait_cycles = 0;

    SC_HAS_PROCESS(Memory);

    Memory(sc_module_name name,
           uint64_t base_lat_cycles_ = 20,
           uint64_t bytes_per_cycle_ = 32)
        : sc_module(name),
          tgt("tgt"),
          base_lat_cycles(base_lat_cycles_),
          bytes_per_cycle(bytes_per_cycle_),
          peq("peq")
    {
        tgt.register_nb_transport_fw(this, &Memory::nb_transport_fw);
        SC_THREAD(peq_thread);
        SC_THREAD(service_thread);
    }

    tlm_sync_enum nb_transport_fw(tlm_generic_payload &gp,
                                  tlm_phase &phase,
                                  sc_time &delay)
    {
        if (phase == BEGIN_REQ)
        {
            // gp.acquire();
            peq.notify(gp, delay);
            phase = END_REQ;
            delay = SC_ZERO_TIME;
            return TLM_UPDATED;
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
                Entry e;
                e.gp = gp;
                e.enqueue_time = sc_time_stamp();
                q.push_back(e);
                q_nonempty.notify(SC_ZERO_TIME);
            }
        }
    }

    void service_thread()
    {
        while (true)
        {
            while (q.empty())
                wait(q_nonempty);

            Entry e = q.front();
            q.pop_front();

            uint64_t bytes = e.gp->get_data_length();

            sc_time t_start = sc_time_stamp();
            uint64_t qwait = (uint64_t)((t_start - e.enqueue_time) / CYCLE);
            qwait_cycles += qwait;

            uint64_t xfer = (bytes_per_cycle == 0) ? 0 : ceil_div_u64(bytes, bytes_per_cycle);
            uint64_t mem_cycles = base_lat_cycles + xfer;

            reqs += 1;
            busy_cycles += mem_cycles;

            wait(mem_cycles * CYCLE);

            e.gp->set_response_status(TLM_OK_RESPONSE);

            tlm_phase bw_phase = BEGIN_RESP;
            sc_time bw_delay = SC_ZERO_TIME;
            auto status = tgt->nb_transport_bw(*e.gp, bw_phase, bw_delay);

            if (status == TLM_COMPLETED)
            {
                // e.gp->release();
            }
            // If TLM_ACCEPTED or TLM_UPDATED, initiator will finish with END_RESP
        }
    }
};

// ============================================================
// Interconnect
// - many workers -> router target
// - router -> mat / vec / mem
// - responses routed back to originating worker
// ============================================================
struct Interconnect : sc_module
{
    tlm_utils::multi_passthrough_target_socket<Interconnect> tgt;
    // tlm_utils::multi_passthrough_initiator_socket<Interconnect> to_workers;

    tlm_utils::simple_initiator_socket<Interconnect> to_mat;
    tlm_utils::simple_initiator_socket<Interconnect> to_vec;
    tlm_utils::simple_initiator_socket<Interconnect> to_mem;

    static constexpr uint64_t ADDR_MAT = 0x1000;
    static constexpr uint64_t ADDR_VEC = 0x2000;
    static constexpr uint64_t ADDR_MEM = 0x3000;

    SC_CTOR(Interconnect)
        : tgt("tgt"),
          to_mat("to_mat"),
          to_vec("to_vec"),
          to_mem("to_mem")
    {
        tgt.register_nb_transport_fw(this, &Interconnect::nb_transport_fw);
        to_mat.register_nb_transport_bw(this, &Interconnect::nb_transport_bw_mat);
        to_vec.register_nb_transport_bw(this, &Interconnect::nb_transport_bw_vec);
        to_mem.register_nb_transport_bw(this, &Interconnect::nb_transport_bw_mem);
    }

    tlm_sync_enum nb_transport_fw(int id,
                                  tlm_generic_payload &gp,
                                  tlm_phase &phase,
                                  sc_time &delay)
    {
        TxnExt *tx = nullptr;
        gp.get_extension(tx);
        if (!tx)
        {
            tx = new TxnExt();
            gp.set_extension(tx);
        }
        tx->upstream_id = id;

        uint64_t addr = gp.get_address();

        if (addr == ADDR_MAT)
            return to_mat->nb_transport_fw(gp, phase, delay);
        else if (addr == ADDR_VEC)
            return to_vec->nb_transport_fw(gp, phase, delay);
        else if (addr == ADDR_MEM)
            return to_mem->nb_transport_fw(gp, phase, delay);

        gp.set_response_status(TLM_ADDRESS_ERROR_RESPONSE);
        phase = BEGIN_RESP;
        return tgt[id]->nb_transport_bw(gp, phase, delay);
    }

    tlm_sync_enum route_back(tlm_generic_payload &gp,
                             tlm_phase &phase,
                             sc_time &delay)
    {
        TxnExt *tx = nullptr;
        gp.get_extension(tx);
        if (!tx || tx->upstream_id < 0)
            return TLM_COMPLETED;

        return tgt[tx->upstream_id]->nb_transport_bw(gp, phase, delay);
    }

    tlm_sync_enum nb_transport_bw_mat(tlm_generic_payload &gp,
                                      tlm_phase &phase,
                                      sc_time &delay)
    {
        return route_back(gp, phase, delay);
    }

    tlm_sync_enum nb_transport_bw_vec(tlm_generic_payload &gp,
                                      tlm_phase &phase,
                                      sc_time &delay)
    {
        return route_back(gp, phase, delay);
    }

    tlm_sync_enum nb_transport_bw_mem(tlm_generic_payload &gp,
                                      tlm_phase &phase,
                                      sc_time &delay)
    {
        return route_back(gp, phase, delay);
    }
};

// ============================================================
// Accelerator target
// Single-server FIFO + memory accesses
// ============================================================
struct AcceleratorTLM : sc_module
{
    tlm_utils::simple_target_socket<AcceleratorTLM> tgt;
    tlm_utils::simple_initiator_socket<AcceleratorTLM> to_mem;

    tlm_utils::peq_with_get<tlm_generic_payload> peq;

    struct Entry
    {
        tlm_generic_payload *gp = nullptr;
        sc_time enqueue_time;
    };

    std::deque<Entry> q;
    sc_event q_nonempty;

    uint64_t busy_cycles = 0;
    uint64_t queue_wait_cycles = 0;
    uint64_t req_count = 0;

    // One local event per memory sub-transaction
    std::unordered_map<tlm_generic_payload *, sc_event *> mem_done_map;

    SC_HAS_PROCESS(AcceleratorTLM);

    explicit AcceleratorTLM(sc_module_name name)
        : sc_module(name),
          tgt("tgt"),
          to_mem("to_mem"),
          peq("peq")
    {
        tgt.register_nb_transport_fw(this, &AcceleratorTLM::nb_transport_fw);
        to_mem.register_nb_transport_bw(this, &AcceleratorTLM::nb_transport_bw_mem);
        SC_THREAD(peq_thread);
        SC_THREAD(service_thread);
    }

    tlm_sync_enum nb_transport_fw(tlm_generic_payload &gp,
                                  tlm_phase &phase,
                                  sc_time &delay)
    {
        if (phase == BEGIN_REQ)
        {
            // gp.acquire();
            peq.notify(gp, delay);
            phase = END_REQ;
            delay = SC_ZERO_TIME;
            return TLM_UPDATED;
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
                Entry e;
                e.gp = gp;
                e.enqueue_time = sc_time_stamp();
                q.push_back(e);
                q_nonempty.notify(SC_ZERO_TIME);
            }
        }
    }

    void mem_access(bool is_write, uint64_t bytes)
    {
        if (bytes == 0)
            return;

        auto *gp = new tlm_generic_payload();
        // gp->acquire();

        gp->set_command(is_write ? TLM_WRITE_COMMAND : TLM_READ_COMMAND);
        gp->set_address(Interconnect::ADDR_MEM);
        gp->set_data_ptr(nullptr);
        gp->set_data_length((unsigned)bytes);
        gp->set_streaming_width((unsigned)bytes);
        gp->set_response_status(TLM_INCOMPLETE_RESPONSE);

        sc_event done_ev;
        mem_done_map[gp] = &done_ev;

        tlm_phase phase = BEGIN_REQ;
        sc_time delay = SC_ZERO_TIME;

        auto status = to_mem->nb_transport_fw(*gp, phase, delay);

        if (status == TLM_COMPLETED)
        {
            mem_done_map.erase(gp);
            // gp->release();
            delete gp;
            return;
        }

        wait(done_ev);

        mem_done_map.erase(gp);
        // gp->release();
        delete gp;
    }

    tlm_sync_enum nb_transport_bw_mem(tlm_generic_payload &gp,
                                      tlm_phase &phase,
                                      sc_time &delay)
    {
        if (phase == BEGIN_RESP)
        {
            auto it = mem_done_map.find(&gp);
            if (it != mem_done_map.end() && it->second)
            {
                it->second->notify(delay);
            }

            phase = END_RESP;
            delay = SC_ZERO_TIME;
            return TLM_COMPLETED;
        }

        return TLM_ACCEPTED;
    }

    void service_thread()
    {
        while (true)
        {
            while (q.empty())
                wait(q_nonempty);

            Entry e = q.front();
            q.pop_front();

            ReqExt *ext = nullptr;
            e.gp->get_extension(ext);

            uint64_t svc = ext ? ext->cycles : 0;

            sc_time t_start = sc_time_stamp();
            uint64_t qwait = (uint64_t)((t_start - e.enqueue_time) / CYCLE);
            if (ext)
                ext->accel_qwait_cycles = qwait;
            queue_wait_cycles += qwait;

            req_count += 1;

            sc_time m0 = sc_time_stamp();
            mem_access(false, ext ? ext->rd_bytes : 0);
            mem_access(true, ext ? ext->wr_bytes : 0);
            sc_time m1 = sc_time_stamp();

            if (ext)
                ext->mem_cycles = (uint64_t)((m1 - m0) / CYCLE);

            busy_cycles += svc;
            wait(svc * CYCLE);

            e.gp->set_response_status(TLM_OK_RESPONSE);

            tlm_phase phase = BEGIN_RESP;
            sc_time delay = SC_ZERO_TIME;
            auto status = tgt->nb_transport_bw(*e.gp, phase, delay);

            if (status == TLM_COMPLETED)
            {
                // e.gp->release();
            }
        }
    }
};

// ============================================================
// Worker
// Uses non-blocking initiator transport
// ============================================================
struct Worker : sc_module
{
    tlm_utils::simple_initiator_socket<Worker> init;
    tlm_utils::peq_with_get<tlm_generic_payload> peq;

    int tid;

    uint64_t m;
    uint64_t access_mat;
    uint64_t access_vec;
    uint64_t mat_cycles;
    uint64_t vec_cycles;
    uint64_t scalar_cycles;

    uint64_t compute_cycles = 0;
    uint64_t wait_cycles = 0;
    uint64_t mem_cycles_accum = 0;
    uint64_t mat_calls = 0;
    uint64_t vec_calls = 0;

    uint64_t A_bytes = 0;
    uint64_t B_bytes = 0;
    uint64_t C_bytes = 0;

    std::unordered_map<tlm_generic_payload *, sc_event *> done_map;

    SC_HAS_PROCESS(Worker);

    Worker(sc_module_name name,
           int tid_,
           uint64_t access_mat_,
           uint64_t access_vec_,
           uint64_t mat_cycles_,
           uint64_t vec_cycles_,
           uint64_t scalar_cycles_,
           uint64_t A_bytes_,
           uint64_t B_bytes_,
           uint64_t C_bytes_)
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
          A_bytes(A_bytes_),
          B_bytes(B_bytes_),
          C_bytes(C_bytes_)
    {
        init.register_nb_transport_bw(this, &Worker::nb_transport_bw);
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
                    it->second->notify(SC_ZERO_TIME);
                }
            }
        }
    }

    void do_scalar(uint64_t cyc)
    {
        compute_cycles += cyc;
        wait(cyc * CYCLE);
    }

    void issue(uint64_t addr, uint64_t svc_cycles)
    {
        auto *gp = new tlm_generic_payload();
        // gp->acquire();

        gp->set_command(TLM_IGNORE_COMMAND);
        gp->set_address(addr);
        gp->set_data_ptr(nullptr);
        gp->set_data_length(0);
        gp->set_streaming_width(0);
        gp->set_response_status(TLM_INCOMPLETE_RESPONSE);

        auto *req = new ReqExt(tid, svc_cycles, A_bytes + B_bytes, C_bytes);
        auto *tx = new TxnExt();
        tx->src_worker = tid;

        gp->set_extension(req);
        gp->set_extension(tx);

        sc_event done_ev;
        done_map[gp] = &done_ev;

        sc_time t0 = sc_time_stamp();
        tlm_phase phase = BEGIN_REQ;
        sc_time delay = SC_ZERO_TIME;

        auto status = init->nb_transport_fw(*gp, phase, delay);

        if (status == TLM_COMPLETED)
        {
            // immediate completion case
        }
        else
        {
            wait(done_ev);
        }

        sc_time t1 = sc_time_stamp();

        ReqExt *ext = nullptr;
        gp->get_extension(ext);

        wait_cycles += ext ? ext->accel_qwait_cycles : 0;
        compute_cycles += svc_cycles;
        mem_cycles_accum += ext ? ext->mem_cycles : 0;

        // Close response handshake
        tlm_phase end_phase = END_RESP;
        sc_time end_delay = SC_ZERO_TIME;
        init->nb_transport_fw(*gp, end_phase, end_delay);

        done_map.erase(gp);

        gp->clear_extension(req);
        gp->clear_extension(tx);
        delete req;
        delete tx;

        // gp->release();
        delete gp;

        (void)t0;
        (void)t1;
    }

    void run()
    {
        sc_time start = sc_time_stamp();

        for (uint64_t i = 0; i < access_mat; i++)
        {
            do_scalar(scalar_cycles);
            issue(Interconnect::ADDR_MAT, mat_cycles);
            mat_calls++;
        }

        for (uint64_t i = 0; i < access_vec; i++)
        {
            do_scalar(scalar_cycles);
            issue(Interconnect::ADDR_VEC, vec_cycles);
            vec_calls++;
        }

        sc_time end = sc_time_stamp();
        uint64_t elapsed_cycles = (uint64_t)((end - start) / CYCLE);
        uint64_t total_cycles = compute_cycles + wait_cycles + mem_cycles_accum;

        std::cout << "[T" << tid << "]"
                  << " mat_calls=" << mat_calls
                  << " vec_calls=" << vec_calls
                  << " wait=" << wait_cycles
                  << " compute=" << compute_cycles
                  << " mem=" << mem_cycles_accum
                  << " total=" << total_cycles
                  << " elapsed=" << elapsed_cycles
                  << " @ sim_time=" << sc_time_stamp()
                  << "\n";
    }
};

// ============================================================
// Top
// ============================================================
struct Top : sc_module
{
    AcceleratorTLM mat_acc;
    AcceleratorTLM vec_acc;
    Interconnect noc;
    Memory memory;

    std::vector<Worker *> workers;

    SC_CTOR(Top)
        : mat_acc("mat_acc"),
          vec_acc("vec_acc"),
          noc("noc"),
          memory("memory")
    {
        noc.to_mat.bind(mat_acc.tgt);
        noc.to_vec.bind(vec_acc.tgt);
        noc.to_mem.bind(memory.tgt);

        mat_acc.to_mem.bind(noc.tgt);
        vec_acc.to_mem.bind(noc.tgt);

        int N = NUM_THREADS;

        uint64_t tile_M = ceil_div_u64(A_M, MATMUL_M);
        uint64_t tile_K = ceil_div_u64(A_K, MATMUL_K);
        uint64_t tile_N = ceil_div_u64(B_N, MATMUL_N);

        uint64_t access_mat = tile_M * tile_K * tile_N;
        uint64_t access_vec = ceil_div_u64(A_M * B_N, VECTOR_ACC_CAP);

        uint64_t mat_cycles = MATMUL_ACC_CYCLE;
        uint64_t vec_cycles = VECTOR_ACC_CYCLE;
        uint64_t scalar_cycles = SCALAR_OVERHEAD;

        uint64_t A_bytes = MATMUL_M * MATMUL_K * sizeof(float);
        uint64_t B_bytes = MATMUL_K * MATMUL_N * sizeof(float);
        uint64_t C_bytes = MATMUL_M * MATMUL_N * sizeof(float);

        for (int i = 0; i < N; i++)
        {
            auto *w = new Worker(sc_gen_unique_name("worker"),
                                 i,
                                 access_mat,
                                 access_vec,
                                 mat_cycles,
                                 vec_cycles,
                                 scalar_cycles,
                                 A_bytes,
                                 B_bytes,
                                 C_bytes);

            workers.push_back(w);

            w->init.bind(noc.tgt);
            // noc.to_workers.bind(w->init);
        }
    }

    ~Top() override
    {
        for (auto *w : workers)
            delete w;
    }
};

int sc_main(int argc, char *argv[])
{
    Top top("top");
    sc_start();

    std::cout << "\n=== Accelerator stats ===\n";
    std::cout << "mat_acc: reqs=" << top.mat_acc.req_count
              << " busy_cycles=" << top.mat_acc.busy_cycles
              << " queue_wait_cycles=" << top.mat_acc.queue_wait_cycles
              << "\n";

    std::cout << "vec_acc: reqs=" << top.vec_acc.req_count
              << " busy_cycles=" << top.vec_acc.busy_cycles
              << " queue_wait_cycles=" << top.vec_acc.queue_wait_cycles
              << "\n";

    std::cout << "memory : reqs=" << top.memory.reqs
              << " busy_cycles=" << top.memory.busy_cycles
              << " queue_wait_cycles=" << top.memory.qwait_cycles
              << "\n";

    return 0;
}
