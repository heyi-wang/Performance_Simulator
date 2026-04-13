#include <cstdlib>
#include <iostream>
#include <string>
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

    const LayerDesc &sca_conv = layers[5];
    ok &= check(sca_conv.op_kind == LAYER_OP_SCA_CONV && sca_conv.backend == BACKEND_VECOPS,
                "sca conv uses dedicated vecops layer kind");
    const auto sca_conv_phases = vecops_phases_for_layer(sca_conv);
    ok &= check(sca_conv_phases.size() == 2 &&
                    sca_conv_phases[0] == VOP_SCA_DOT_I8_TO_I32 &&
                    sca_conv_phases[1] == VOP_SCA_BIAS_QUANT_I32_TO_I8,
                "sca conv maps to reduction then bias-quant phases");

    const VecOpsRuntimeConfig sca_dot_cfg =
        make_vecops_runtime_config(sca_conv, sca_conv_phases[0]);
    ok &= check(sca_dot_cfg.channels == 32 && sca_dot_cfg.height == 1 &&
                    sca_dot_cfg.width == 1 && sca_dot_cfg.reduce_len == 32,
                "sca reduction config keeps output channels and reduction length");

    const LayerExpectedStats sca_conv_stats = expected_layer_stats(sca_conv, N_WORKERS);
    const uint64_t sca_dot_tiles = (32u + VOP_VEC_ACC_CAP - 1) / VOP_VEC_ACC_CAP;
    ok &= check(sca_conv_stats.mat_reqs == 0, "sca conv does not use matmul requests");
    ok &= check(sca_conv_stats.vec_reqs == 32u * sca_dot_tiles + 32u,
                "sca conv vector requests match reduction plus finalize");
    ok &= check(sca_conv_stats.rd_bytes == 2u * 32u * 32u + 32u * (4u + 4u),
                "sca conv reads pooled activations, weights, accumulator, and bias");
    ok &= check(sca_conv_stats.wr_bytes == 32u,
                "sca conv writes one attention byte per output channel");

    const LayerDesc &scale = layers[6];
    const auto scale_phases = vecops_phases_for_layer(scale);
    ok &= check(scale_phases.size() == 1 && scale_phases[0] == VOP_SCALAR_MUL,
                "scale layer maps to scalar mul");

    int id = 0;
    const LayerDesc pixelshuffle = make_pixelshuffle_layer(id, "pixelshuffle", 32, 32, 1024);
    ok &= check(pixelshuffle.op_kind == LAYER_OP_PIXELSHUFFLE &&
                    pixelshuffle.backend == BACKEND_VECOPS,
                "pixelshuffle layer uses dedicated op kind and vecops backend");
    ok &= check(pixelshuffle.Hin == 32 && pixelshuffle.Win == 32 && pixelshuffle.Cin == 1024 &&
                    pixelshuffle.Hout == 64 && pixelshuffle.Wout == 64 && pixelshuffle.Cout == 256,
                "pixelshuffle layer expands spatial shape and reduces channels");

    const auto pixelshuffle_phases = vecops_phases_for_layer(pixelshuffle);
    ok &= check(pixelshuffle_phases.size() == 1 &&
                    pixelshuffle_phases[0] == VOP_PIXELSHUFFLE_MOVE,
                "pixelshuffle layer maps to a dedicated move phase");

    const VecOpsRuntimeConfig pixelshuffle_cfg =
        make_vecops_runtime_config(pixelshuffle, pixelshuffle_phases[0]);
    ok &= check(pixelshuffle_cfg.channels == 256 && pixelshuffle_cfg.height == 64 &&
                    pixelshuffle_cfg.width == 64 && pixelshuffle_cfg.reduce_len == 0,
                "pixelshuffle vecops config follows output tensor shape");

    const LayerExpectedStats pixelshuffle_stats = expected_layer_stats(pixelshuffle, N_WORKERS);
    const uint64_t pixelshuffle_elems = 256u * 64u * 64u;
    ok &= check(pixelshuffle_stats.rd_bytes == pixelshuffle_elems,
                "pixelshuffle reads one input byte per output element");
    ok &= check(pixelshuffle_stats.wr_bytes == pixelshuffle_elems,
                "pixelshuffle writes one output byte per output element");

    const LayerDesc skip_add = make_skip_add_layer(id, "skip_add", 64, 64, 32);
    ok &= check(skip_add.op_kind == LAYER_OP_SKIP_ADD && skip_add.backend == BACKEND_VECOPS,
                "skip-add layer uses dedicated op kind and vecops backend");
    ok &= check(skip_add.Hin == 64 && skip_add.Win == 64 && skip_add.Cin == 32 &&
                    skip_add.Hout == 64 && skip_add.Wout == 64 && skip_add.Cout == 32,
                "skip-add layer preserves tensor shape");

    const auto skip_add_phases = vecops_phases_for_layer(skip_add);
    ok &= check(skip_add_phases.size() == 1 && skip_add_phases[0] == VOP_ELEMWISE_ADD,
                "skip-add layer maps to element-wise add");

    const VecOpsRuntimeConfig skip_add_cfg =
        make_vecops_runtime_config(skip_add, skip_add_phases[0]);
    ok &= check(skip_add_cfg.channels == 32 && skip_add_cfg.height == 64 &&
                    skip_add_cfg.width == 64,
                "skip-add vecops config preserves tensor shape");

    const LayerExpectedStats skip_add_stats = expected_layer_stats(skip_add, N_WORKERS);
    const uint64_t skip_add_elems = 32u * 64u * 64u;
    ok &= check(skip_add_stats.rd_bytes == 2u * skip_add_elems,
                "skip-add reads two input tensors");
    ok &= check(skip_add_stats.wr_bytes == skip_add_elems,
                "skip-add writes one output tensor");
    ok &= check(skip_add_stats.vec_reqs ==
                    skip_add_elems / VOP_VEC_ACC_CAP +
                        ((skip_add_elems % VOP_VEC_ACC_CAP) ? 1u : 0u),
                "skip-add vector requests follow vec-op tiling");

    const LayerDesc &residual = layers[8];
    const auto residual_phases = vecops_phases_for_layer(residual);
    ok &= check(residual_phases.size() == 2 &&
                    residual_phases[0] == VOP_SCALAR_MUL &&
                    residual_phases[1] == VOP_ELEMWISE_ADD,
                "residual layer maps to scalar mul then add");

    const auto full_layers = build_nafnet32_layers();
    auto find_layer = [&](const char *name) -> const LayerDesc * {
        for (const auto &layer : full_layers)
        {
            if (std::string(layer.name) == name)
                return &layer;
        }
        return nullptr;
    };

    ok &= check(full_layers.size() == 522, "full NAFNet manifest layer count matches inference path");

    const LayerDesc *intro = find_layer("intro");
    ok &= check(intro && intro->Hin == 512 && intro->Win == 512 &&
                    intro->Hout == 512 && intro->Wout == 512,
                "full NAFNet manifest starts with 512x512 intro conv");

    const LayerDesc *ups0 = find_layer("ups_0");
    ok &= check(ups0 && ups0->Hin == 32 && ups0->Win == 32 &&
                    ups0->Cin == 512 && ups0->Cout == 1024,
                "decoder stage 0 starts with 512-to-1024 upsample conv");

    const LayerDesc *ups0_pixelshuffle = find_layer("ups_0_pixelshuffle");
    ok &= check(ups0_pixelshuffle &&
                    ups0_pixelshuffle->op_kind == LAYER_OP_PIXELSHUFFLE &&
                    ups0_pixelshuffle->Hin == 32 && ups0_pixelshuffle->Win == 32 &&
                    ups0_pixelshuffle->Cin == 1024 &&
                    ups0_pixelshuffle->Hout == 64 && ups0_pixelshuffle->Wout == 64 &&
                    ups0_pixelshuffle->Cout == 256,
                "decoder stage 0 includes pixelshuffle after upsample conv");

    const LayerDesc *dec0_skip_add = find_layer("dec0_skip_add");
    ok &= check(dec0_skip_add && dec0_skip_add->op_kind == LAYER_OP_SKIP_ADD &&
                    dec0_skip_add->Hout == 64 && dec0_skip_add->Wout == 64 &&
                    dec0_skip_add->Cout == 256,
                "decoder stage 0 includes explicit skip-add before decoder blocks");

    ok &= check(find_layer("dec0_blk0_norm1") && find_layer("dec0_blk1_norm1"),
                "decoder stage 0 includes both decoder blocks");

    const LayerDesc *ending = find_layer("ending");
    ok &= check(ending && ending->Hin == 512 && ending->Win == 512 &&
                    ending->Kh == 3 && ending->Kw == 3,
                "full NAFNet manifest ends with 3x3 ending conv at full resolution");

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
