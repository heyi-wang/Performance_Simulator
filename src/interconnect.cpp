#include "interconnect.h"

Interconnect::Interconnect(sc_module_name nm)
    : sc_module(nm),
      tgt("tgt"),
      to_mat("to_mat"),
      to_vec("to_vec"),
      to_mem("to_mem")
{
    tgt.register_nb_transport_fw(this, &Interconnect::nb_transport_fw);
    to_mat.register_nb_transport_bw(this, &Interconnect::nb_transport_bw_mat);
    to_vec.register_nb_transport_bw(this, &Interconnect::nb_transport_bw_vec);
    to_mem.register_nb_transport_bw(this, &Interconnect::nb_transport_bw_mem);
}

tlm_sync_enum Interconnect::nb_transport_fw(int id,
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

tlm_sync_enum Interconnect::route_back(tlm_generic_payload &gp,
                                       tlm_phase &phase,
                                       sc_time &delay)
{
    TxnExt *tx = nullptr;
    gp.get_extension(tx);
    if (!tx || tx->upstream_id < 0)
        return TLM_COMPLETED;

    return tgt[tx->upstream_id]->nb_transport_bw(gp, phase, delay);
}

tlm_sync_enum Interconnect::nb_transport_bw_mat(tlm_generic_payload &gp,
                                                tlm_phase &phase,
                                                sc_time &delay)
{
    return route_back(gp, phase, delay);
}

tlm_sync_enum Interconnect::nb_transport_bw_vec(tlm_generic_payload &gp,
                                                tlm_phase &phase,
                                                sc_time &delay)
{
    return route_back(gp, phase, delay);
}

tlm_sync_enum Interconnect::nb_transport_bw_mem(tlm_generic_payload &gp,
                                                tlm_phase &phase,
                                                sc_time &delay)
{
    return route_back(gp, phase, delay);
}
