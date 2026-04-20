#pragma once

#include "common.h"

// ============================================================
// ReqExt: carries accelerator request metadata
// (source thread, computation cycles, memory traffic)
// ============================================================
struct ReqExt : tlm_extension<ReqExt>
{
    int      src_id             = -1;
    uint64_t cycles             = 0;
    uint64_t accel_qwait_cycles = 0;
    uint64_t mem_cycles         = 0;
    uint64_t rd_bytes           = 0;
    uint64_t wr_bytes           = 0;

    ReqExt() = default;
    ReqExt(int s, uint64_t a, uint64_t r, uint64_t w)
        : src_id(s), cycles(a), rd_bytes(r), wr_bytes(w) {}

    tlm_extension_base *clone() const override { return new ReqExt(*this); }
    void copy_from(const tlm_extension_base &ext) override
    {
        *this = static_cast<const ReqExt &>(ext);
    }
};

// ============================================================
// TxnExt: carries transaction routing context
// (source worker id, upstream socket id for response routing)
// ============================================================
struct TxnExt : tlm_extension<TxnExt>
{
    int       src_worker   = -1;
    sc_event *done_ev      = nullptr;
    sc_event *admit_ev     = nullptr;
    bool     *done_fired   = nullptr;
    bool     *admit_fired  = nullptr;
    int       upstream_id  = -1;

    tlm_extension_base *clone() const override { return new TxnExt(*this); }
    void copy_from(const tlm_extension_base &ext) override
    {
        *this = static_cast<const TxnExt &>(ext);
    }
};
