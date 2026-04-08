#include <cstdlib>
#include <iostream>
#include <vector>

#include "nafnet_kernel_bridge.h"

static bool check(bool cond, const char *msg)
{
    std::cout << (cond ? "[PASS] " : "[FAIL] ") << msg << "\n";
    return cond;
}

int sc_main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    bool ok = true;

    const auto layers = build_nafblock_only_layers(32, 64, 64);

    const LayerDesc &conv1 = layers[1];
    const MatmulRuntimeConfig conv1_cfg = make_matmul_runtime_config(conv1);
    ok &= check(conv1_cfg.gemm_m() == 64u * 64u, "conv1 maps H*W to GEMM M");
    ok &= check(conv1_cfg.gemm_k() == 32u, "conv1 maps Cin*Kh*Kw to GEMM K");
    ok &= check(conv1_cfg.gemm_n() == 64u, "conv1 maps Cout to GEMM N");

    const LayerDesc &dw = layers[2];
    const DwConvRuntimeConfig dw_cfg = make_dwconv_runtime_config(dw);
    ok &= check(dw_cfg.channels == 64, "dwconv keeps channel count");
    ok &= check(dw_cfg.kernel_h == 3 && dw_cfg.kernel_w == 3, "dwconv keeps kernel shape");
    ok &= check(dw_cfg.out_h() == 64 && dw_cfg.out_w() == 64, "dwconv derives output shape");

    const LayerDesc &norm1 = layers[0];
    const LayerNormRuntimeConfig ln_cfg = make_layernorm_runtime_config(norm1);
    ok &= check(ln_cfg.channels == 32 && ln_cfg.height == 64 && ln_cfg.width == 64,
                "layernorm keeps tensor shape");

    const LayerDesc &gap = layers[4];
    const PoolRuntimeConfig pool_cfg = make_pool_runtime_config(gap);
    ok &= check(pool_cfg.channels == 32 && pool_cfg.spatial() == 64 * 64,
                "pooling keeps channel/spatial shape");

    const LayerDesc &scale = layers[6];
    const auto scale_phases = vecops_phases_for_layer(scale);
    ok &= check(scale_phases.size() == 1 && scale_phases[0] == VOP_SCALAR_MUL,
                "scale layer maps to scalar mul");

    const LayerDesc &residual = layers[8];
    const auto residual_phases = vecops_phases_for_layer(residual);
    ok &= check(residual_phases.size() == 2 &&
                    residual_phases[0] == VOP_SCALAR_MUL &&
                    residual_phases[1] == VOP_ELEMWISE_ADD,
                "residual layer maps to scalar mul then add");

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
