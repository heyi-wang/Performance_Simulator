#pragma once

#include <cstdio>
#include <string>
#include <vector>

#include "../kernel/vec_ops/vec_ops_config.h"

enum NafBlockOpKind
{
    NB_OP_LAYERNORM,
    NB_OP_CONV,
    NB_OP_DWCONV,
    NB_OP_SIMPLEGATE,
    NB_OP_GAP,
    NB_OP_SCA_SCALE,
    NB_OP_RESIDUAL,
};

enum NafBlockBackend
{
    NB_BACKEND_MATMUL,
    NB_BACKEND_DWCONV,
    NB_BACKEND_LAYERNORM,
    NB_BACKEND_POOLING,
    NB_BACKEND_VECOPS,
};

struct NafBlockLayerDesc
{
    int             id = 0;
    char            name[64]{};
    NafBlockOpKind  op_kind = NB_OP_CONV;
    NafBlockBackend backend = NB_BACKEND_MATMUL;

    int Hin = 0, Win = 0, Cin = 0;
    int Hout = 0, Wout = 0, Cout = 0;
    int Kh = 1, Kw = 1, stride = 1, pad = 0, groups = 1;

    int     phase_count   = 1;
    VopType primary_vop   = VOP_ELEMWISE_MUL;
    VopType secondary_vop = VOP_ELEMWISE_MUL;

    // Optional shape overrides for phase 2. Default 0 = inherit the primary
    // phase's (Hout, Wout, Cout). Used by sca_conv's quant epilogue, where
    // phase 1's dot-product re-encoding (Hout=C, Cout=C) would otherwise make
    // phase 2 iterate C*C elements instead of the real C scalar partials.
    int secondary_Hout = 0;
    int secondary_Wout = 0;
    int secondary_Cout = 0;
};

inline const char *nb_op_kind_str(NafBlockOpKind op)
{
    switch (op)
    {
    case NB_OP_LAYERNORM:  return "LAYERNORM";
    case NB_OP_CONV:       return "CONV";
    case NB_OP_DWCONV:     return "DWCONV";
    case NB_OP_SIMPLEGATE: return "SIMPLEGATE";
    case NB_OP_GAP:        return "GAP";
    case NB_OP_SCA_SCALE:  return "SCA_SCALE";
    case NB_OP_RESIDUAL:   return "RESIDUAL";
    }
    return "UNKNOWN";
}

inline const char *nb_backend_str(NafBlockBackend backend)
{
    switch (backend)
    {
    case NB_BACKEND_MATMUL:    return "MATMUL";
    case NB_BACKEND_DWCONV:    return "DWCONV";
    case NB_BACKEND_LAYERNORM: return "LAYERNORM";
    case NB_BACKEND_POOLING:   return "POOLING";
    case NB_BACKEND_VECOPS:    return "VECOPS";
    }
    return "UNKNOWN";
}

inline NafBlockLayerDesc nb_make_layer(int &id_ctr,
                                       const char *name,
                                       NafBlockOpKind op_kind,
                                       NafBlockBackend backend,
                                       int Hin, int Win, int Cin,
                                       int Hout, int Wout, int Cout,
                                       int Kh, int Kw,
                                       int stride, int pad, int groups,
                                       int phase_count,
                                       VopType primary_vop = VOP_ELEMWISE_MUL,
                                       VopType secondary_vop = VOP_ELEMWISE_MUL)
{
    NafBlockLayerDesc l{};
    l.id = id_ctr++;
    std::snprintf(l.name, sizeof(l.name), "%s", name);
    l.op_kind       = op_kind;
    l.backend       = backend;
    l.Hin = Hin;   l.Win = Win;   l.Cin = Cin;
    l.Hout = Hout; l.Wout = Wout; l.Cout = Cout;
    l.Kh = Kh;     l.Kw = Kw;
    l.stride = stride; l.pad = pad; l.groups = groups;
    l.phase_count   = phase_count;
    l.primary_vop   = primary_vop;
    l.secondary_vop = secondary_vop;
    return l;
}

inline NafBlockLayerDesc nb_make_layernorm(int &id, const char *name,
                                           int H, int W, int C)
{
    return nb_make_layer(id, name, NB_OP_LAYERNORM, NB_BACKEND_LAYERNORM,
                         H, W, C, H, W, C,
                         1, 1, 1, 0, 1, 1);
}

inline NafBlockLayerDesc nb_make_conv(int &id, const char *name,
                                      int Hin, int Win, int Cin,
                                      int Hout, int Wout, int Cout,
                                      int Kh, int Kw,
                                      int stride, int pad, int groups = 1)
{
    return nb_make_layer(id, name, NB_OP_CONV, NB_BACKEND_MATMUL,
                         Hin, Win, Cin, Hout, Wout, Cout,
                         Kh, Kw, stride, pad, groups, 1);
}

inline NafBlockLayerDesc nb_make_dwconv(int &id, const char *name,
                                        int H, int W, int C,
                                        int Kh, int Kw, int pad)
{
    return nb_make_layer(id, name, NB_OP_DWCONV, NB_BACKEND_DWCONV,
                         H, W, C, H, W, C,
                         Kh, Kw, 1, pad, C, 1);
}

// SimpleGate: split a 2C tensor along channels into two halves and elementwise
// multiply, then requant the i16 product back to i8 (matches simplegate_i8 in
// nafnet_c_ops.h: prod = (a*b)*M >> shift). Phase 1 = widening i8*i8 -> i16,
// phase 2 = i16 -> i8 quant with M+shift.
inline NafBlockLayerDesc nb_make_simplegate(int &id, const char *name,
                                            int H, int W, int C)
{
    return nb_make_layer(id, name, NB_OP_SIMPLEGATE, NB_BACKEND_VECOPS,
                         H, W, 2 * C, H, W, C,
                         1, 1, 1, 0, 1, 2,
                         VOP_ELEMWISE_MUL,
                         VOP_QUANTIZE_I16_TO_I8);
}

// Global Average Pool: collapse H*W -> 1*1 per channel.
inline NafBlockLayerDesc nb_make_gap(int &id, const char *name,
                                     int H, int W, int C)
{
    return nb_make_layer(id, name, NB_OP_GAP, NB_BACKEND_POOLING,
                         H, W, C, 1, 1, C,
                         1, 1, 1, 0, 1, 1);
}

// SCA scale: broadcast a 1x1xC scalar onto an HxWxC tensor (channel-wise mul),
// then requant the i16 product back to i8. Matches sca_i8's step 3 in
// nafnet_c_ops.h: out = (x[c,h,w] * attn[c]) * M >> shift. Phase 1 = widening
// i8*i8 -> i16 (attn[c] is broadcast per channel by the worker pinning),
// phase 2 = i16 -> i8 quant with M+shift.
inline NafBlockLayerDesc nb_make_scale(int &id, const char *name,
                                       int H, int W, int C)
{
    return nb_make_layer(id, name, NB_OP_SCA_SCALE, NB_BACKEND_VECOPS,
                         H, W, C, H, W, C,
                         1, 1, 1, 0, 1, 2,
                         VOP_ELEMWISE_MUL,
                         VOP_QUANTIZE_I16_TO_I8);
}

// SCA 1x1 conv on a 1x1 spatial feature map. Semantically C_out independent
// dot products, each over a C_in-length kernel and the C_in-length input
// vector at spatial (0,0); C_in == C_out == C. Mapped to the vector unit
// (one vwmul_vv + vredsum_vs per output pixel). We re-encode C_in into the
// LayerDesc's Hout dimension so the existing nb_make_vecops_cfg sees
// channels = C_out output pixels and spatial = C_in MACs per dot product.
inline NafBlockLayerDesc nb_make_sca_conv(int &id, const char *name, int C)
{
    // Phase 1: C output pixels, each a C-element dot product (re-encoded as
    // Hout=C, Cout=C so the existing nb_make_vecops_cfg sees C channels with
    // spatial=C). Phase 2: a single C-element i32->i8 requant over the C
    // partial sums; override the shape to (Cout=1, Hout=C, Wout=1) so it
    // iterates exactly C elements, not C*C.
    NafBlockLayerDesc l = nb_make_layer(
        id, name, NB_OP_CONV, NB_BACKEND_VECOPS,
        /*Hin*/  1, /*Win*/ 1, /*Cin*/  C,
        /*Hout*/ C, /*Wout*/ 1, /*Cout*/ C,
        /*Kh*/   1, /*Kw*/  1,
        /*stride*/ 1, /*pad*/ 0, /*groups*/ 1,
        /*phase_count*/ 2,
        VOP_DOT_PRODUCT_I8,
        VOP_QUANTIZE_I32_TO_I8);
    l.secondary_Cout = 1;
    l.secondary_Hout = C;
    l.secondary_Wout = 1;
    return l;
}

// Residual with scaling: out = clip(x + (y * beta_i16) >> frac_bits). Phase 1
// is the fused i8*i16 broadcast multiply + arithmetic shift + clip
// (VOP_SCALE_REQUANT_I8); phase 2 is the elementwise i8 add of the skip
// connection. Matches residual_scale_add_i8 in nafnet_c_ops.h.
inline NafBlockLayerDesc nb_make_residual(int &id, const char *name,
                                          int H, int W, int C)
{
    return nb_make_layer(id, name, NB_OP_RESIDUAL, NB_BACKEND_VECOPS,
                         H, W, C, H, W, C,
                         1, 1, 1, 0, 1, 2,
                         VOP_SCALE_REQUANT_I8,
                         VOP_ELEMWISE_ADD);
}

// Integration entry point. Appends the 14 sub-layers of one NafBlock with the
// given prefix into `layers`, bumping `id` as it goes. A future nafnet driver
// can call this in a loop to embed the block at any level of the network.
inline void append_nafblock_layers(std::vector<NafBlockLayerDesc> &layers,
                                   int &id,
                                   const char *prefix,
                                   int C, int H, int W)
{
    char nm[64];

    // ---- MBConv ----
    std::snprintf(nm, sizeof(nm), "%s_norm1", prefix);
    layers.push_back(nb_make_layernorm(id, nm, H, W, C));

    std::snprintf(nm, sizeof(nm), "%s_conv1", prefix);
    layers.push_back(nb_make_conv(id, nm, H, W, C, H, W, 2 * C, 1, 1, 1, 0));

    std::snprintf(nm, sizeof(nm), "%s_conv2_dw", prefix);
    layers.push_back(nb_make_dwconv(id, nm, H, W, 2 * C, 3, 3, 1));

    std::snprintf(nm, sizeof(nm), "%s_simplegate1", prefix);
    layers.push_back(nb_make_simplegate(id, nm, H, W, C));

    std::snprintf(nm, sizeof(nm), "%s_sca_gap", prefix);
    layers.push_back(nb_make_gap(id, nm, H, W, C));

    std::snprintf(nm, sizeof(nm), "%s_sca_conv", prefix);
    layers.push_back(nb_make_sca_conv(id, nm, C));

    std::snprintf(nm, sizeof(nm), "%s_sca_scale", prefix);
    layers.push_back(nb_make_scale(id, nm, H, W, C));

    std::snprintf(nm, sizeof(nm), "%s_conv3", prefix);
    layers.push_back(nb_make_conv(id, nm, H, W, C, H, W, C, 1, 1, 1, 0));

    std::snprintf(nm, sizeof(nm), "%s_beta_residual", prefix);
    layers.push_back(nb_make_residual(id, nm, H, W, C));

    // ---- FFN ----
    std::snprintf(nm, sizeof(nm), "%s_norm2", prefix);
    layers.push_back(nb_make_layernorm(id, nm, H, W, C));

    std::snprintf(nm, sizeof(nm), "%s_conv4", prefix);
    layers.push_back(nb_make_conv(id, nm, H, W, C, H, W, 2 * C, 1, 1, 1, 0));

    std::snprintf(nm, sizeof(nm), "%s_simplegate2", prefix);
    layers.push_back(nb_make_simplegate(id, nm, H, W, C));

    std::snprintf(nm, sizeof(nm), "%s_conv5", prefix);
    layers.push_back(nb_make_conv(id, nm, H, W, C, H, W, C, 1, 1, 1, 0));

    std::snprintf(nm, sizeof(nm), "%s_gamma_residual", prefix);
    layers.push_back(nb_make_residual(id, nm, H, W, C));
}

inline std::vector<NafBlockLayerDesc> build_nafblock_layers(int C, int H, int W)
{
    std::vector<NafBlockLayerDesc> layers;
    layers.reserve(14);
    int id = 0;
    append_nafblock_layers(layers, id, "nafblock", C, H, W);
    return layers;
}

// Canonical 14-entry manifest used by the pre-sim sanity check.
// VECOPS rows additionally pin the expected phase composition so any
// drift in the per-helper quantize epilogue (Changes B / C) fails closed.
struct NafBlockManifestEntry
{
    const char     *suffix;
    NafBlockOpKind  op_kind;
    NafBlockBackend backend;
    int             phase_count;    // 0 = don't check (non-VECOPS rows)
    VopType         primary_vop;
    VopType         secondary_vop;
};

inline const NafBlockManifestEntry *nafblock_manifest()
{
    static const NafBlockManifestEntry m[14] = {
        {"norm1",           NB_OP_LAYERNORM,  NB_BACKEND_LAYERNORM, 0, VOP_ELEMWISE_MUL,    VOP_ELEMWISE_MUL},
        {"conv1",           NB_OP_CONV,       NB_BACKEND_MATMUL,    0, VOP_ELEMWISE_MUL,    VOP_ELEMWISE_MUL},
        {"conv2_dw",        NB_OP_DWCONV,     NB_BACKEND_DWCONV,    0, VOP_ELEMWISE_MUL,    VOP_ELEMWISE_MUL},
        {"simplegate1",     NB_OP_SIMPLEGATE, NB_BACKEND_VECOPS,    2, VOP_ELEMWISE_MUL,    VOP_QUANTIZE_I16_TO_I8},
        {"sca_gap",         NB_OP_GAP,        NB_BACKEND_POOLING,   0, VOP_ELEMWISE_MUL,    VOP_ELEMWISE_MUL},
        {"sca_conv",        NB_OP_CONV,       NB_BACKEND_VECOPS,    2, VOP_DOT_PRODUCT_I8,  VOP_QUANTIZE_I32_TO_I8},
        {"sca_scale",       NB_OP_SCA_SCALE,  NB_BACKEND_VECOPS,    2, VOP_ELEMWISE_MUL,    VOP_QUANTIZE_I16_TO_I8},
        {"conv3",           NB_OP_CONV,       NB_BACKEND_MATMUL,    0, VOP_ELEMWISE_MUL,    VOP_ELEMWISE_MUL},
        {"beta_residual",   NB_OP_RESIDUAL,   NB_BACKEND_VECOPS,    2, VOP_SCALE_REQUANT_I8, VOP_ELEMWISE_ADD},
        {"norm2",           NB_OP_LAYERNORM,  NB_BACKEND_LAYERNORM, 0, VOP_ELEMWISE_MUL,    VOP_ELEMWISE_MUL},
        {"conv4",           NB_OP_CONV,       NB_BACKEND_MATMUL,    0, VOP_ELEMWISE_MUL,    VOP_ELEMWISE_MUL},
        {"simplegate2",     NB_OP_SIMPLEGATE, NB_BACKEND_VECOPS,    2, VOP_ELEMWISE_MUL,    VOP_QUANTIZE_I16_TO_I8},
        {"conv5",           NB_OP_CONV,       NB_BACKEND_MATMUL,    0, VOP_ELEMWISE_MUL,    VOP_ELEMWISE_MUL},
        {"gamma_residual",  NB_OP_RESIDUAL,   NB_BACKEND_VECOPS,    2, VOP_SCALE_REQUANT_I8, VOP_ELEMWISE_ADD},
    };
    return m;
}

inline bool validate_nafblock_manifest(const std::vector<NafBlockLayerDesc> &layers,
                                       const char *prefix,
                                       std::string *error = nullptr)
{
    if (layers.size() != 14)
    {
        if (error) *error = "manifest must contain exactly 14 sub-layers";
        return false;
    }
    const NafBlockManifestEntry *m = nafblock_manifest();
    for (size_t i = 0; i < 14; ++i)
    {
        char expected[64];
        std::snprintf(expected, sizeof(expected), "%s_%s", prefix, m[i].suffix);
        if (std::string(layers[i].name) != expected)
        {
            if (error) *error = "unexpected layer name at index " +
                                std::to_string(i) + ": '" + layers[i].name +
                                "' expected '" + expected + "'";
            return false;
        }
        if (layers[i].op_kind != m[i].op_kind ||
            layers[i].backend != m[i].backend)
        {
            if (error) *error = "unexpected op/backend at index " +
                                std::to_string(i);
            return false;
        }
        if (m[i].phase_count > 0)
        {
            if (layers[i].phase_count != m[i].phase_count ||
                layers[i].primary_vop != m[i].primary_vop ||
                (m[i].phase_count > 1 &&
                 layers[i].secondary_vop != m[i].secondary_vop))
            {
                if (error) *error = "unexpected phase composition at index " +
                                    std::to_string(i) + " (" + layers[i].name + ")";
                return false;
            }
        }
    }
    return true;
}
