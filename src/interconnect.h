#pragma once

#include "extensions.h"
#include <tlm_utils/multi_passthrough_target_socket.h>
#include <tlm_utils/simple_initiator_socket.h>

// ============================================================
// Interconnect — address-based router
//   Workers / Accelerators → tgt (multi-passthrough, indexed)
//   tgt → to_mat / to_vec / to_mem  (address decode)
//   Responses routed back via TxnExt::upstream_id
// ============================================================
struct Interconnect : sc_module
{
    tlm_utils::multi_passthrough_target_socket<Interconnect> tgt;

    tlm_utils::simple_initiator_socket<Interconnect> to_mat;
    tlm_utils::simple_initiator_socket<Interconnect> to_vec;
    tlm_utils::simple_initiator_socket<Interconnect> to_mem;

    static constexpr uint64_t ADDR_MAT = 0x1000;
    static constexpr uint64_t ADDR_VEC = 0x2000;
    static constexpr uint64_t ADDR_MEM = 0x3000;

    SC_CTOR(Interconnect);

    tlm_sync_enum nb_transport_fw(int id,
                                  tlm_generic_payload &gp,
                                  tlm_phase &phase,
                                  sc_time &delay);

    tlm_sync_enum route_back(tlm_generic_payload &gp,
                             tlm_phase &phase,
                             sc_time &delay);

    tlm_sync_enum nb_transport_bw_mat(tlm_generic_payload &gp,
                                      tlm_phase &phase,
                                      sc_time &delay);

    tlm_sync_enum nb_transport_bw_vec(tlm_generic_payload &gp,
                                      tlm_phase &phase,
                                      sc_time &delay);

    tlm_sync_enum nb_transport_bw_mem(tlm_generic_payload &gp,
                                      tlm_phase &phase,
                                      sc_time &delay);
};
