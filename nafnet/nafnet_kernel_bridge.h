#pragma once

#include <vector>

#include "../kernel/dw_conv2d/dw_conv2d_top.h"
#include "../kernel/layer_norm/layer_norm_top.h"
#include "../kernel/matmul/matmul_top.h"
#include "../kernel/pooling/pooling_top.h"
#include "../kernel/vec_ops/vec_ops_top.h"
#include "nafnet_hw_config.h"
#include "nafnet_layers.h"

inline MatmulRuntimeConfig make_matmul_runtime_config(const LayerDesc &layer,
                                                      int worker_count = N_WORKERS)
{
    MatmulRuntimeConfig cfg = MatmulRuntimeConfig::defaults(worker_count);
    cfg.workload_n = 1;
    cfg.workload_h = static_cast<uint64_t>(layer.Hout);
    cfg.workload_w = static_cast<uint64_t>(layer.Wout);
    cfg.workload_c_in = static_cast<uint64_t>(layer.Cin);
    cfg.workload_kh = static_cast<uint64_t>(layer.Kh);
    cfg.workload_kw = static_cast<uint64_t>(layer.Kw);
    cfg.workload_c_out = static_cast<uint64_t>(layer.Cout);
    return cfg;
}

inline DwConvRuntimeConfig make_dwconv_runtime_config(const LayerDesc &layer)
{
    DwConvRuntimeConfig cfg = DwConvRuntimeConfig::defaults();
    cfg.channels = layer.Cout;
    cfg.height = layer.Hin;
    cfg.width = layer.Win;
    cfg.kernel_h = layer.Kh;
    cfg.kernel_w = layer.Kw;
    cfg.pad = layer.pad;
    cfg.stride = layer.stride;
    cfg.worker_count = N_WORKERS;
    return cfg;
}

inline LayerNormRuntimeConfig make_layernorm_runtime_config(const LayerDesc &layer)
{
    LayerNormRuntimeConfig cfg = LayerNormRuntimeConfig::defaults();
    cfg.channels = layer.Cin;
    cfg.height = layer.Hin;
    cfg.width = layer.Win;
    cfg.worker_count = N_WORKERS;
    return cfg;
}

inline PoolRuntimeConfig make_pool_runtime_config(const LayerDesc &layer)
{
    PoolRuntimeConfig cfg = PoolRuntimeConfig::defaults();
    cfg.channels = layer.Cin;
    cfg.height = layer.Hin;
    cfg.width = layer.Win;
    cfg.worker_count = N_WORKERS;
    return cfg;
}

inline VecOpsRuntimeConfig make_vecops_runtime_config(const LayerDesc &layer, VopType op)
{
    VecOpsRuntimeConfig cfg = VecOpsRuntimeConfig::defaults();
    cfg.op = op;
    cfg.channels = layer.Cout;
    cfg.height = layer.Hout;
    cfg.width = layer.Wout;
    cfg.worker_count = N_WORKERS;
    return cfg;
}

inline std::vector<VopType> vecops_phases_for_layer(const LayerDesc &layer)
{
    std::vector<VopType> phases;
    phases.push_back(layer.primary_vop);
    if (layer.phase_count > 1)
        phases.push_back(layer.secondary_vop);
    return phases;
}
