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

tlm_sync_enum Worker::nb_transport_bw(tlm_generic_payload &gp,
                                      tlm_phase &phase,
                                      sc_time &delay)
{
    if (phase == BEGIN_RESP)
        peq.notify(gp, delay);
    return TLM_ACCEPTED;
}

void Worker::peq_thread()
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

void Worker::do_scalar(uint64_t cyc)
{
    compute_cycles += cyc;
    wait(cyc * CYCLE);
}

Worker::PendingReq Worker::issue_begin(uint64_t addr, uint64_t svc_cycles)
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

    auto *req = new ReqExt(tid, svc_cycles, A_bytes + B_bytes, C_bytes);
    auto *tx  = new TxnExt();
    tx->src_worker = tid;

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

void Worker::issue_end(PendingReq &p)
{
    if (!p.sync_done)
        wait(*p.done_ev);

    done_map.erase(p.gp);
    delete p.done_ev;
    p.done_ev = nullptr;

    ReqExt *ext = nullptr;
    p.gp->get_extension(ext);

    wait_cycles      += ext ? ext->accel_qwait_cycles : 0;
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

void Worker::run()
{
    sc_time start = sc_time_stamp();

    // Pipeline pattern:
    //   send[i] → do_scalar (overlapped with accel processing i)
    //           → issue_end[i] → send[i+1] → ...
    if (access_mat > 0)
    {
        auto pending = issue_begin(Interconnect::ADDR_MAT, mat_cycles);
        mat_calls++;

        for (uint64_t i = 1; i < access_mat; i++)
        {
            do_scalar(scalar_cycles);
            issue_end(pending);
            pending = issue_begin(Interconnect::ADDR_MAT, mat_cycles);
            mat_calls++;
        }
        issue_end(pending);
    }

    if (access_vec > 0)
    {
        auto pending = issue_begin(Interconnect::ADDR_VEC, vec_cycles);
        vec_calls++;

        for (uint64_t i = 1; i < access_vec; i++)
        {
            do_scalar(scalar_cycles);
            issue_end(pending);
            pending = issue_begin(Interconnect::ADDR_VEC, vec_cycles);
            vec_calls++;
        }
        issue_end(pending);
    }

    sc_time  end           = sc_time_stamp();
    uint64_t elapsed_cycles = (uint64_t)((end - start) / CYCLE);
    uint64_t total_cycles   = compute_cycles + wait_cycles + mem_cycles_accum;

    std::cout << "[T" << tid << "]"
              << " mat_calls="  << mat_calls
              << " vec_calls="  << vec_calls
              << " wait="       << wait_cycles
              << " compute="    << compute_cycles
              << " mem="        << mem_cycles_accum
              << " total="      << total_cycles
              << " elapsed="    << elapsed_cycles
              << " @ sim_time=" << sc_time_stamp()
              << "\n";
}
