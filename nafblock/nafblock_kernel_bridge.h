#pragma once

#include <vector>

#include "../kernel/dw_conv2d/dw_conv2d_top.h"
#include "../kernel/layer_norm/layer_norm_top.h"
#include "../kernel/matmul/matmul_top.h"
#include "../kernel/pooling/pooling_top.h"
#include "../kernel/vec_ops/vec_ops_top.h"
#include "nafblock_config.h"
#include "nafblock_layers.h"

inline MatmulRuntimeConfig nb_make_matmul_cfg(const NafBlockLayerDesc &layer,
                                              int worker_count = nafblock_cfg::N_WORKERS)
{
    // A<->B operand swap: treat weights as A (M=Cout) and input as B (N=Hout*Wout).
    // gemm_m = n*h*w, gemm_k = c_in*kh*kw, gemm_n = c_out
    // To get gemm_m = Cout and gemm_n = Hout*Wout while keeping gemm_k = Cin*Kh*Kw,
    // pack Cout into workload_h and Hout*Wout into workload_c_out.
    MatmulRuntimeConfig cfg = MatmulRuntimeConfig::defaults(worker_count);
    const uint64_t spatial = static_cast<uint64_t>(layer.Hout)
                           * static_cast<uint64_t>(layer.Wout);
    cfg.workload_n     = 1;
    cfg.workload_h     = static_cast<uint64_t>(layer.Cout);
    cfg.workload_w     = 1;
    cfg.workload_c_in  = static_cast<uint64_t>(layer.Cin);
    cfg.workload_kh    = static_cast<uint64_t>(layer.Kh);
    cfg.workload_kw    = static_cast<uint64_t>(layer.Kw);
    cfg.workload_c_out = spatial;
    return cfg;
}

inline DwConvRuntimeConfig nb_make_dwconv_cfg(const NafBlockLayerDesc &layer)
{
    DwConvRuntimeConfig cfg = DwConvRuntimeConfig::defaults();
    cfg.channels     = layer.Cout;
    cfg.height       = layer.Hin;
    cfg.width        = layer.Win;
    cfg.kernel_h     = layer.Kh;
    cfg.kernel_w     = layer.Kw;
    cfg.pad          = layer.pad;
    cfg.stride       = layer.stride;
    cfg.worker_count = nafblock_cfg::N_WORKERS;
    // Match the C reference (dw_conv3x3_bias_requant in nafnet_c_ops.h): the
    // dwconv kernel's int32 accumulator must be requantized to int8 before
    // the next sub-layer consumes it.
    cfg.requant_enabled = true;
    return cfg;
}

inline LayerNormRuntimeConfig nb_make_layernorm_cfg(const NafBlockLayerDesc &layer)
{
    LayerNormRuntimeConfig cfg = LayerNormRuntimeConfig::defaults();
    cfg.channels     = layer.Cin;
    cfg.height       = layer.Hin;
    cfg.width        = layer.Win;
    cfg.worker_count = nafblock_cfg::N_WORKERS;
    return cfg;
}

inline PoolRuntimeConfig nb_make_pool_cfg(const NafBlockLayerDesc &layer)
{
    PoolRuntimeConfig cfg = PoolRuntimeConfig::defaults();
    cfg.channels     = layer.Cin;
    cfg.height       = layer.Hin;
    cfg.width        = layer.Win;
    cfg.worker_count = nafblock_cfg::N_WORKERS;
    return cfg;
}

inline VecOpsRuntimeConfig nb_make_vecops_cfg(const NafBlockLayerDesc &layer,
                                              VopType op,
                                              int phase_idx = 0)
{
    VecOpsRuntimeConfig cfg = VecOpsRuntimeConfig::defaults();
    cfg.op           = op;
    cfg.channels     = layer.Cout;
    cfg.height       = layer.Hout;
    cfg.width        = layer.Wout;
    // Phase-2 shape override: non-zero secondary_* fields replace the primary
    // shape for the second phase. Currently used by sca_conv to size the
    // i32 -> i8 requant over the C partial sums instead of C*C tiles.
    if (phase_idx == 1)
    {
        if (layer.secondary_Cout > 0) cfg.channels = layer.secondary_Cout;
        if (layer.secondary_Hout > 0) cfg.height   = layer.secondary_Hout;
        if (layer.secondary_Wout > 0) cfg.width    = layer.secondary_Wout;
    }
    cfg.worker_count = nafblock_cfg::N_WORKERS;
    return cfg;
}

inline std::vector<VopType> nb_vecops_phases(const NafBlockLayerDesc &layer)
{
    std::vector<VopType> phases;
    phases.push_back(layer.primary_vop);
    if (layer.phase_count > 1)
        phases.push_back(layer.secondary_vop);
    return phases;
}
