#include "accelerator_pool.h"

SC_HAS_PROCESS(AcceleratorPool);

AcceleratorPool::AcceleratorPool(sc_module_name name,
                                 size_t instance_count_,
                                 size_t queue_capacity_)
    : sc_module(name),
      tgt("tgt"),
      queue_capacity(queue_capacity_)
{
    tgt.register_nb_transport_fw(this, &AcceleratorPool::nb_transport_fw);
    SC_THREAD(dispatch_thread);

    size_t count = std::max<size_t>(instance_count_, 1);
    units.reserve(count);
    to_units.reserve(count);
    unit_busy.assign(count, false);

    for (size_t i = 0; i < count; ++i)
    {
        units.push_back(std::make_unique<AcceleratorTLM>(
            sc_gen_unique_name("accel_unit"),
            1));
        to_units.push_back(
            std::make_unique<tlm_utils::simple_initiator_socket_tagged<AcceleratorPool>>(
                sc_gen_unique_name("to_unit")));
        to_units.back()->register_nb_transport_bw(this, &AcceleratorPool::nb_transport_bw_unit, static_cast<int>(i));
        to_units.back()->bind(units.back()->tgt);
    }
}

tlm_sync_enum AcceleratorPool::nb_transport_fw(tlm_generic_payload &gp,
                                               tlm_phase &phase,
                                               sc_time &delay)
{
    if (phase == BEGIN_REQ)
    {
        if (admitted < queue_capacity)
        {
            ++admitted;
            q.push_back({&gp, sc_time_stamp() + delay});
            q_changed.notify(delay);
            phase = END_REQ;
            delay = SC_ZERO_TIME;
            return TLM_UPDATED;
        }

        stall_fifo.push_back(&gp);
        return TLM_ACCEPTED;
    }

    return TLM_ACCEPTED;
}

tlm_sync_enum AcceleratorPool::nb_transport_bw_unit(int id,
                                                    tlm_generic_payload &gp,
                                                    tlm_phase &phase,
                                                    sc_time &delay)
{
    if (id >= 0 && static_cast<size_t>(id) < unit_busy.size() && phase == BEGIN_RESP)
    {
        unit_busy[static_cast<size_t>(id)] = false;
        if (!stall_fifo.empty())
        {
            tlm_generic_payload *next_gp = stall_fifo.front();
            stall_fifo.pop_front();
            q.push_back({next_gp, sc_time_stamp() + delay});

            tlm_phase end_req_phase = END_REQ;
            sc_time   end_req_delay = delay;
            tgt->nb_transport_bw(*next_gp, end_req_phase, end_req_delay);
        }
        else if (admitted > 0)
        {
            --admitted;
        }
        q_changed.notify(delay);
    }

    return tgt->nb_transport_bw(gp, phase, delay);
}

void AcceleratorPool::dispatch_thread()
{
    while (true)
    {
        while (q.empty() || !has_free_unit())
            wait(q_changed);

        while (!q.empty())
        {
            int unit_id = find_free_unit();
            if (unit_id < 0)
                break;

            Entry e = q.front();
            q.pop_front();
            unit_busy[static_cast<size_t>(unit_id)] = true;

            ReqExt *ext = nullptr;
            e.gp->get_extension(ext);
            uint64_t qwait =
                static_cast<uint64_t>((sc_time_stamp() - e.enqueue_time) / CYCLE);
            if (ext)
                ext->accel_qwait_cycles += qwait;
            shared_queue_wait_cycles += qwait;

            tlm_phase phase = BEGIN_REQ;
            sc_time   delay = SC_ZERO_TIME;
            auto status = (*to_units[static_cast<size_t>(unit_id)])->nb_transport_fw(*e.gp, phase, delay);

            if (status == TLM_ACCEPTED)
            {
                // The pool only dispatches to known-free units, so this should not happen.
                // If it does, requeue the request and try again later.
                unit_busy[static_cast<size_t>(unit_id)] = false;
                q.push_front(e);
                q_changed.notify(SC_ZERO_TIME);
                break;
            }

        }
    }
}

size_t AcceleratorPool::instance_count() const
{
    return units.size();
}

bool AcceleratorPool::has_free_unit() const
{
    for (bool busy : unit_busy)
        if (!busy)
            return true;
    return false;
}

int AcceleratorPool::find_free_unit() const
{
    for (size_t i = 0; i < unit_busy.size(); ++i)
        if (!unit_busy[i])
            return static_cast<int>(i);
    return -1;
}

uint64_t AcceleratorPool::req_count_total() const
{
    uint64_t total = 0;
    for (const auto &unit : units)
        total += unit->req_count;
    return total;
}

uint64_t AcceleratorPool::busy_cycles_total() const
{
    uint64_t total = 0;
    for (const auto &unit : units)
        total += unit->busy_cycles;
    return total;
}

uint64_t AcceleratorPool::occupied_cycles_total() const
{
    uint64_t total = 0;
    for (const auto &unit : units)
        total += unit->occupied_cycles;
    return total;
}

uint64_t AcceleratorPool::queue_wait_cycles_total() const
{
    uint64_t total = shared_queue_wait_cycles;
    for (const auto &unit : units)
        total += unit->queue_wait_cycles;
    return total;
}

std::vector<AccelInstanceStats> AcceleratorPool::per_instance_stats() const
{
    std::vector<AccelInstanceStats> out;
    out.reserve(units.size());
    for (size_t i = 0; i < units.size(); ++i)
    {
        AccelInstanceStats s;
        s.instance_id      = static_cast<int>(i);
        s.req_count        = units[i]->req_count;
        s.busy_cycles      = units[i]->busy_cycles;
        s.occupied_cycles  = units[i]->occupied_cycles;
        s.queue_wait_cycles = units[i]->queue_wait_cycles;
        out.push_back(s);
    }
    return out;
}
