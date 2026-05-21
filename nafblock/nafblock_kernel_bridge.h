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
    MatmulRuntimeConfig cfg = MatmulRuntimeConfig::defaults(worker_count);
    cfg.workload_n     = 1;
    cfg.workload_h     = static_cast<uint64_t>(layer.Hout);
    cfg.workload_w     = static_cast<uint64_t>(layer.Wout);
    cfg.workload_c_in  = static_cast<uint64_t>(layer.Cin);
    cfg.workload_kh    = static_cast<uint64_t>(layer.Kh);
    cfg.workload_kw    = static_cast<uint64_t>(layer.Kw);
    cfg.workload_c_out = static_cast<uint64_t>(layer.Cout);
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
                                              VopType op)
{
    VecOpsRuntimeConfig cfg = VecOpsRuntimeConfig::defaults();
    cfg.op           = op;
    cfg.channels     = layer.Cout;
    cfg.height       = layer.Hout;
    cfg.width        = layer.Wout;
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
