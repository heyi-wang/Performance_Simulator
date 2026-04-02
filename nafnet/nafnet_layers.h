#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "../kernel/dw_conv2d/dw_conv2d_config.h"
#include "../kernel/layer_norm/layer_norm_config.h"
#include "../kernel/pooling/pooling_config.h"
#include "../kernel/vec_ops/vec_ops_config.h"
#include "common.h"
#include "nafnet_hw_config.h"

enum LayerOpKind
{
    LAYER_OP_CONV,
    LAYER_OP_DWCONV,
    LAYER_OP_LAYERNORM,
    LAYER_OP_SIMPLEGATE,
    LAYER_OP_GAP,
    LAYER_OP_SCA_SCALE,
    LAYER_OP_RESIDUAL,
};

enum LayerBackend
{
    BACKEND_MATMUL,
    BACKEND_DWCONV,
    BACKEND_LAYERNORM,
    BACKEND_POOLING,
    BACKEND_VECOPS,
};

struct LayerDesc
{
    int          id = 0;
    char         name[64]{};
    LayerOpKind  op_kind = LAYER_OP_CONV;
    LayerBackend backend = BACKEND_MATMUL;

    int Hin = 0, Win = 0, Cin = 0;
    int Hout = 0, Wout = 0, Cout = 0;
    int Kh = 1, Kw = 1, stride = 1, pad = 0, groups = 1;

    bool    multithreaded = true;
    int     phase_count = 1;
    VopType primary_vop = VOP_ELEMWISE_MUL;
    VopType secondary_vop = VOP_ELEMWISE_MUL;
};

struct LayerExpectedStats
{
    uint64_t mat_reqs = 0;
    uint64_t vec_reqs = 0;
    uint64_t mem_reqs = 0;
    uint64_t accel_cycles = 0;
    uint64_t cpu_cycles = 0;
    uint64_t rd_bytes = 0;
    uint64_t wr_bytes = 0;
};

static inline uint64_t cdiv64(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

static inline std::pair<int, int> channel_range(int total, int tid, int n_workers)
{
    return {
        (tid * total) / n_workers,
        ((tid + 1) * total) / n_workers,
    };
}

static inline uint64_t even_share(uint64_t total, int tid, int n_workers)
{
    uint64_t base = total / static_cast<uint64_t>(n_workers);
    uint64_t rem  = total % static_cast<uint64_t>(n_workers);
    return base + ((static_cast<uint64_t>(tid) < rem) ? 1u : 0u);
}

static inline const char *layer_op_kind_str(LayerOpKind op)
{
    switch (op)
    {
    case LAYER_OP_CONV:       return "CONV";
    case LAYER_OP_DWCONV:     return "DWCONV";
    case LAYER_OP_LAYERNORM:  return "LAYERNORM";
    case LAYER_OP_SIMPLEGATE: return "SIMPLEGATE";
    case LAYER_OP_GAP:        return "GAP";
    case LAYER_OP_SCA_SCALE:  return "SCA_SCALE";
    case LAYER_OP_RESIDUAL:   return "RESIDUAL";
    }
    return "UNKNOWN";
}

static inline const char *layer_backend_str(LayerBackend backend)
{
    switch (backend)
    {
    case BACKEND_MATMUL:    return "MATMUL";
    case BACKEND_DWCONV:    return "DWCONV";
    case BACKEND_LAYERNORM: return "LAYERNORM";
    case BACKEND_POOLING:   return "POOLING";
    case BACKEND_VECOPS:    return "VECOPS";
    }
    return "UNKNOWN";
}

static inline LayerDesc make_layer(int &id_ctr,
                                   const char *name,
                                   LayerOpKind op_kind,
                                   LayerBackend backend,
                                   int Hin, int Win, int Cin,
                                   int Hout, int Wout, int Cout,
                                   int Kh, int Kw,
                                   int stride, int pad, int groups,
                                   int phase_count,
                                   VopType primary_vop = VOP_ELEMWISE_MUL,
                                   VopType secondary_vop = VOP_ELEMWISE_MUL)
{
    LayerDesc l{};
    l.id = id_ctr++;
    std::snprintf(l.name, sizeof(l.name), "%s", name);
    l.op_kind = op_kind;
    l.backend = backend;
    l.Hin = Hin;
    l.Win = Win;
    l.Cin = Cin;
    l.Hout = Hout;
    l.Wout = Wout;
    l.Cout = Cout;
    l.Kh = Kh;
    l.Kw = Kw;
    l.stride = stride;
    l.pad = pad;
    l.groups = groups;
    l.multithreaded = true;
    l.phase_count = phase_count;
    l.primary_vop = primary_vop;
    l.secondary_vop = secondary_vop;
    return l;
}

static inline LayerDesc make_conv_layer(int &id_ctr,
                                        const char *name,
                                        int Hin, int Win, int Cin,
                                        int Hout, int Wout, int Cout,
                                        int Kh, int Kw,
                                        int stride, int pad, int groups = 1)
{
    return make_layer(id_ctr, name,
                      LAYER_OP_CONV, BACKEND_MATMUL,
                      Hin, Win, Cin, Hout, Wout, Cout,
                      Kh, Kw, stride, pad, groups, 3);
}

static inline LayerDesc make_dwconv_layer(int &id_ctr,
                                          const char *name,
                                          int H, int W, int C,
                                          int Kh, int Kw, int pad)
{
    return make_layer(id_ctr, name,
                      LAYER_OP_DWCONV, BACKEND_DWCONV,
                      H, W, C, H, W, C,
                      Kh, Kw, 1, pad, C, 1);
}

static inline LayerDesc make_layernorm_layer(int &id_ctr,
                                             const char *name,
                                             int H, int W, int C)
{
    return make_layer(id_ctr, name,
                      LAYER_OP_LAYERNORM, BACKEND_LAYERNORM,
                      H, W, C, H, W, C,
                      1, 1, 1, 0, 1, 4);
}

static inline LayerDesc make_simplegate_layer(int &id_ctr,
                                              const char *name,
                                              int H, int W, int C)
{
    return make_layer(id_ctr, name,
                      LAYER_OP_SIMPLEGATE, BACKEND_VECOPS,
                      H, W, 2 * C, H, W, C,
                      1, 1, 1, 0, 1, 1, VOP_ELEMWISE_MUL);
}

static inline LayerDesc make_gap_layer(int &id_ctr,
                                       const char *name,
                                       int H, int W, int C)
{
    return make_layer(id_ctr, name,
                      LAYER_OP_GAP, BACKEND_POOLING,
                      H, W, C, 1, 1, C,
                      1, 1, 1, 0, 1, 3);
}

static inline LayerDesc make_scale_layer(int &id_ctr,
                                         const char *name,
                                         int H, int W, int C)
{
    return make_layer(id_ctr, name,
                      LAYER_OP_SCA_SCALE, BACKEND_VECOPS,
                      H, W, C, H, W, C,
                      1, 1, 1, 0, 1, 1, VOP_SCALAR_MUL);
}

static inline LayerDesc make_residual_layer(int &id_ctr,
                                            const char *name,
                                            int H, int W, int C)
{
    return make_layer(id_ctr, name,
                      LAYER_OP_RESIDUAL, BACKEND_VECOPS,
                      H, W, C, H, W, C,
                      1, 1, 1, 0, 1, 2,
                      VOP_SCALAR_MUL,
                      VOP_ELEMWISE_ADD);
}

static inline uint64_t naf_matmul_a_bytes()
{
    return MATMUL_M * MATMUL_K * sizeof(int8_t);
}

static inline uint64_t naf_matmul_b_bytes()
{
    return MATMUL_K * MATMUL_N * sizeof(int8_t);
}

static inline uint64_t naf_matmul_c_bytes()
{
    return MATMUL_M * MATMUL_N * sizeof(int32_t);
}

static inline uint64_t naf_conv_gemm_m(const LayerDesc &l)
{
    return static_cast<uint64_t>(l.Hout) * l.Wout;
}

static inline uint64_t naf_conv_gemm_k(const LayerDesc &l)
{
    return static_cast<uint64_t>(l.Cin) * l.Kh * l.Kw;
}

static inline uint64_t naf_conv_gemm_n(const LayerDesc &l)
{
    return static_cast<uint64_t>(l.Cout);
}

static inline uint64_t naf_conv_tile_m(const LayerDesc &l)
{
    return cdiv64(naf_conv_gemm_m(l), MATMUL_M);
}

static inline uint64_t naf_conv_tile_n(const LayerDesc &l)
{
    return cdiv64(naf_conv_gemm_n(l), MATMUL_N);
}

static inline uint64_t naf_conv_k_per_worker(const LayerDesc &l, int n_workers)
{
    return cdiv64(naf_conv_gemm_k(l), static_cast<uint64_t>(n_workers));
}

static inline uint64_t naf_conv_local_k_extent(const LayerDesc &l, int tid, int n_workers)
{
    uint64_t total_k = naf_conv_gemm_k(l);
    uint64_t chunk   = naf_conv_k_per_worker(l, n_workers);
    uint64_t begin   = static_cast<uint64_t>(tid) * chunk;
    uint64_t end     = std::min<uint64_t>(total_k, begin + chunk);
    return (begin < total_k) ? (end - begin) : 0;
}

static inline uint64_t naf_conv_local_tile_k(const LayerDesc &l, int tid, int n_workers)
{
    uint64_t local_k = naf_conv_local_k_extent(l, tid, n_workers);
    return (local_k > 0) ? cdiv64(local_k, MATMUL_K) : 0;
}

static inline uint64_t naf_conv_local_mat_reqs(const LayerDesc &l, int tid, int n_workers)
{
    return naf_conv_tile_m(l) * naf_conv_tile_n(l) * naf_conv_local_tile_k(l, tid, n_workers);
}

static inline int naf_conv_active_workers(const LayerDesc &l, int n_workers)
{
    int active = 0;
    for (int tid = 0; tid < n_workers; ++tid)
    {
        if (naf_conv_local_k_extent(l, tid, n_workers) > 0)
            ++active;
    }
    return active;
}

static inline uint64_t naf_conv_partial_elements(const LayerDesc &l)
{
    return static_cast<uint64_t>(l.Hout) * l.Wout * l.Cout;
}

static inline uint64_t naf_conv_reduce_calls(const LayerDesc &l)
{
    return cdiv64(naf_conv_partial_elements(l), VECTOR_ACC_CAP);
}

static inline uint64_t naf_conv_quant_calls(const LayerDesc &l)
{
    return cdiv64(naf_conv_partial_elements(l), VECTOR_ACC_CAP);
}

static inline uint64_t naf_conv_reduce_rd_bytes()
{
    return 2 * VECTOR_ACC_CAP * sizeof(int32_t);
}

static inline uint64_t naf_conv_reduce_wr_bytes()
{
    return VECTOR_ACC_CAP * sizeof(int32_t);
}

static inline uint64_t naf_conv_quant_rd_bytes()
{
    return VECTOR_ACC_CAP * sizeof(int32_t);
}

static inline uint64_t naf_conv_quant_wr_bytes()
{
    return VECTOR_ACC_CAP * sizeof(uint8_t);
}

static inline uint64_t naf_vecop_requests_per_channel(uint64_t spatial, VopType op)
{
    return cdiv64(spatial, vop_tile_cap_elems(op));
}

static inline uint64_t naf_dwconv_strip_rd_bytes(const LayerDesc &l,
                                                 int oh,
                                                 uint64_t strip_start,
                                                 uint64_t vl)
{
    uint64_t rd = 0;
    for (int kh = 0; kh < l.Kh; ++kh)
    {
        int ih = oh - l.pad + kh;
        if (ih < 0 || ih >= l.Hin)
            continue;

        for (int kw = 0; kw < l.Kw; ++kw)
        {
            int64_t iw_base = static_cast<int64_t>(strip_start) - l.pad + kw;
            int64_t lo = std::max<int64_t>(0, iw_base);
            int64_t hi = std::min<int64_t>(static_cast<int64_t>(l.Win),
                                           iw_base + static_cast<int64_t>(vl));
            if (hi > lo)
                rd += static_cast<uint64_t>(hi - lo) * DW_INPUT_ELEM_BYTES;
        }
    }
    return rd;
}

static inline LayerExpectedStats expected_layer_stats(const LayerDesc &l, int n_workers)
{
    LayerExpectedStats stats{};

    switch (l.backend)
    {
    case BACKEND_MATMUL:
    {
        for (int tid = 0; tid < n_workers; ++tid)
            stats.mat_reqs += naf_conv_local_mat_reqs(l, tid, n_workers);

        int active_workers = naf_conv_active_workers(l, n_workers);
        uint64_t reduce_pairs = (active_workers > 0)
                              ? static_cast<uint64_t>(active_workers - 1)
                              : 0;
        uint64_t reduce_calls = naf_conv_reduce_calls(l);
        uint64_t quant_calls  = naf_conv_quant_calls(l);

        stats.vec_reqs = reduce_pairs * reduce_calls + quant_calls;
        stats.accel_cycles =
            stats.mat_reqs * MATMUL_ACC_CYCLE +
            stats.vec_reqs * VECTOR_ACC_CYCLE;
        stats.rd_bytes =
            stats.mat_reqs * (naf_matmul_a_bytes() + naf_matmul_b_bytes()) +
            reduce_pairs * reduce_calls * naf_conv_reduce_rd_bytes() +
            quant_calls * naf_conv_quant_rd_bytes();
        stats.wr_bytes =
            stats.mat_reqs * naf_matmul_c_bytes() +
            reduce_pairs * reduce_calls * naf_conv_reduce_wr_bytes() +
            quant_calls * naf_conv_quant_wr_bytes();
        break;
    }

    case BACKEND_DWCONV:
    {
        uint64_t strips_per_row = cdiv64(static_cast<uint64_t>(l.Wout), DW_VEC_ACC_CAP);
        for (int c = 0; c < l.Cout; ++c)
        {
            for (int oh = 0; oh < l.Hout; ++oh)
            {
                for (uint64_t strip = 0; strip < strips_per_row; ++strip)
                {
                    uint64_t strip_start = strip * DW_VEC_ACC_CAP;
                    uint64_t vl = std::min<uint64_t>(DW_VEC_ACC_CAP,
                                                     static_cast<uint64_t>(l.Wout) - strip_start);
                    uint64_t rd = naf_dwconv_strip_rd_bytes(l, oh, strip_start, vl);
                    if (oh == 0 && strip == 0)
                        rd += static_cast<uint64_t>(l.Kh) * l.Kw * DW_INPUT_ELEM_BYTES;
                    uint64_t wr = vl * DW_OUTPUT_ELEM_BYTES;
                    ++stats.vec_reqs;
                    stats.accel_cycles += DW_VEC_ACC_CYCLE;
                    stats.rd_bytes += rd;
                    stats.wr_bytes += wr;
                }
            }
        }
        break;
    }

    case BACKEND_LAYERNORM:
    {
        uint64_t spatial = static_cast<uint64_t>(l.Hin) * l.Win;
        uint64_t n_tiles = cdiv64(spatial, LN_VEC_ACC_CAP);
        const uint64_t elem_bytes = 2;
        stats.vec_reqs = static_cast<uint64_t>(l.Cout) * 3 * n_tiles;
        stats.accel_cycles = stats.vec_reqs * LN_VEC_ACC_CYCLE;
        stats.cpu_cycles = static_cast<uint64_t>(l.Cout) * LN_STEP3_CYCLES;
        for (int c = 0; c < l.Cout; ++c)
        {
            for (uint64_t t = 0; t < n_tiles; ++t)
            {
                uint64_t tile_elems = std::min<uint64_t>(LN_VEC_ACC_CAP, spatial - t * LN_VEC_ACC_CAP);
                stats.rd_bytes += tile_elems * elem_bytes;                 // step 1
                stats.rd_bytes += tile_elems * elem_bytes;                 // step 2
                stats.rd_bytes += tile_elems * elem_bytes + 2 * elem_bytes; // step 4
                stats.wr_bytes += tile_elems * elem_bytes;                 // step 4
            }
        }
        break;
    }

    case BACKEND_POOLING:
    {
        uint64_t spatial = static_cast<uint64_t>(l.Hin) * l.Win;
        uint64_t n_tiles = cdiv64(spatial, POOL_VEC_ACC_CAP);
        stats.vec_reqs = static_cast<uint64_t>(l.Cin) * n_tiles;
        stats.mem_reqs = l.Cout;
        stats.accel_cycles = stats.vec_reqs * POOL_VEC_ACC_CYCLE;
        stats.cpu_cycles = static_cast<uint64_t>(l.Cout) * POOL_DIVIDE_CYCLES;
        for (int c = 0; c < l.Cin; ++c)
        {
            for (uint64_t t = 0; t < n_tiles; ++t)
            {
                uint64_t tile_elems = std::min<uint64_t>(POOL_VEC_ACC_CAP, spatial - t * POOL_VEC_ACC_CAP);
                stats.rd_bytes += tile_elems * POOL_INPUT_ELEM_BYTES;
            }
        }
        stats.wr_bytes += static_cast<uint64_t>(l.Cout) * POOL_OUTPUT_ELEM_BYTES;
        break;
    }

    case BACKEND_VECOPS:
    {
        uint64_t spatial = static_cast<uint64_t>(l.Hout) * l.Wout;
        auto accumulate_op = [&](VopType op) {
            uint64_t tile_cap = vop_tile_cap_elems(op);
            uint64_t n_tiles = cdiv64(spatial, tile_cap);
            stats.vec_reqs += static_cast<uint64_t>(l.Cout) * n_tiles;
            stats.accel_cycles += static_cast<uint64_t>(l.Cout) * n_tiles * VOP_VEC_ACC_CYCLE;
            for (int c = 0; c < l.Cout; ++c)
            {
                for (uint64_t t = 0; t < n_tiles; ++t)
                {
                    uint64_t tile_elems = std::min<uint64_t>(tile_cap, spatial - t * tile_cap);
                    stats.rd_bytes += vop_rd_bytes(op, tile_elems);
                    stats.wr_bytes += vop_wr_bytes(op, tile_elems);
                }
            }
        };

        accumulate_op(l.primary_vop);
        if (l.phase_count > 1)
            accumulate_op(l.secondary_vop);
        break;
    }
    }

    return stats;
}

static inline void append_nafblock_layers(std::vector<LayerDesc> &layers,
                                          int &id,
                                          const char *prefix,
                                          int C, int H, int W)
{
    char nm[64];

    std::snprintf(nm, sizeof(nm), "%s_norm1", prefix);
    layers.push_back(make_layernorm_layer(id, nm, H, W, C));

    std::snprintf(nm, sizeof(nm), "%s_conv1", prefix);
    layers.push_back(make_conv_layer(id, nm, H, W, C, H, W, 2 * C, 1, 1, 1, 0));

    std::snprintf(nm, sizeof(nm), "%s_conv2_dw", prefix);
    layers.push_back(make_dwconv_layer(id, nm, H, W, 2 * C, 3, 3, 1));

    std::snprintf(nm, sizeof(nm), "%s_simplegate1", prefix);
    layers.push_back(make_simplegate_layer(id, nm, H, W, C));

    std::snprintf(nm, sizeof(nm), "%s_sca_gap", prefix);
    layers.push_back(make_gap_layer(id, nm, H, W, C));

    std::snprintf(nm, sizeof(nm), "%s_sca_conv", prefix);
    layers.push_back(make_conv_layer(id, nm, 1, 1, C, 1, 1, C, 1, 1, 1, 0));

    std::snprintf(nm, sizeof(nm), "%s_sca_scale", prefix);
    layers.push_back(make_scale_layer(id, nm, H, W, C));

    std::snprintf(nm, sizeof(nm), "%s_conv3", prefix);
    layers.push_back(make_conv_layer(id, nm, H, W, C, H, W, C, 1, 1, 1, 0));

    std::snprintf(nm, sizeof(nm), "%s_beta_residual", prefix);
    layers.push_back(make_residual_layer(id, nm, H, W, C));

    std::snprintf(nm, sizeof(nm), "%s_norm2", prefix);
    layers.push_back(make_layernorm_layer(id, nm, H, W, C));

    std::snprintf(nm, sizeof(nm), "%s_conv4", prefix);
    layers.push_back(make_conv_layer(id, nm, H, W, C, H, W, 2 * C, 1, 1, 1, 0));

    std::snprintf(nm, sizeof(nm), "%s_simplegate2", prefix);
    layers.push_back(make_simplegate_layer(id, nm, H, W, C));

    std::snprintf(nm, sizeof(nm), "%s_conv5", prefix);
    layers.push_back(make_conv_layer(id, nm, H, W, C, H, W, C, 1, 1, 1, 0));

    std::snprintf(nm, sizeof(nm), "%s_gamma_residual", prefix);
    layers.push_back(make_residual_layer(id, nm, H, W, C));
}

static inline std::vector<LayerDesc> build_nafblock_only_layers(int C, int H, int W)
{
    std::vector<LayerDesc> layers;
    layers.reserve(16);
    int id = 0;
    append_nafblock_layers(layers, id, "nafblock", C, H, W);
    return layers;
}

static inline bool validate_nafblock_manifest(const std::vector<LayerDesc> &layers,
                                              const char *prefix,
                                              int C, int H, int W,
                                              std::string *error = nullptr)
{
    static const std::array<const char *, 14> suffixes = {{
        "norm1", "conv1", "conv2_dw", "simplegate1", "sca_gap", "sca_conv", "sca_scale",
        "conv3", "beta_residual", "norm2", "conv4", "simplegate2", "conv5", "gamma_residual"
    }};
    static const std::array<LayerOpKind, 14> ops = {{
        LAYER_OP_LAYERNORM, LAYER_OP_CONV, LAYER_OP_DWCONV, LAYER_OP_SIMPLEGATE,
        LAYER_OP_GAP, LAYER_OP_CONV, LAYER_OP_SCA_SCALE, LAYER_OP_CONV,
        LAYER_OP_RESIDUAL, LAYER_OP_LAYERNORM, LAYER_OP_CONV, LAYER_OP_SIMPLEGATE,
        LAYER_OP_CONV, LAYER_OP_RESIDUAL
    }};
    static const std::array<LayerBackend, 14> backends = {{
        BACKEND_LAYERNORM, BACKEND_MATMUL, BACKEND_DWCONV, BACKEND_VECOPS,
        BACKEND_POOLING, BACKEND_MATMUL, BACKEND_VECOPS, BACKEND_MATMUL,
        BACKEND_VECOPS, BACKEND_LAYERNORM, BACKEND_MATMUL, BACKEND_VECOPS,
        BACKEND_MATMUL, BACKEND_VECOPS
    }};

    if (layers.size() != suffixes.size())
    {
        if (error) *error = "manifest must contain exactly 14 sub-layers";
        return false;
    }

    for (size_t i = 0; i < layers.size(); ++i)
    {
        char expected_name[64];
        std::snprintf(expected_name, sizeof(expected_name), "%s_%s", prefix, suffixes[i]);
        if (std::string(layers[i].name) != expected_name)
        {
            if (error) *error = "unexpected layer name at index " + std::to_string(i);
            return false;
        }
        if (layers[i].op_kind != ops[i] || layers[i].backend != backends[i])
        {
            if (error) *error = "unexpected op/backend at index " + std::to_string(i);
            return false;
        }
    }

    const LayerDesc &norm1 = layers[0];
    const LayerDesc &conv1 = layers[1];
    const LayerDesc &dw    = layers[2];
    const LayerDesc &gap   = layers[4];
    const LayerDesc &sca_c = layers[5];
    const LayerDesc &res1  = layers[8];

    bool ok =
        norm1.Cin == C && norm1.Hin == H && norm1.Win == W &&
        conv1.Cin == C && conv1.Cout == 2 * C && conv1.Hout == H && conv1.Wout == W &&
        dw.Cin == 2 * C && dw.Cout == 2 * C && dw.groups == 2 * C &&
        gap.Cin == C && gap.Cout == C && gap.Hout == 1 && gap.Wout == 1 &&
        sca_c.Hin == 1 && sca_c.Win == 1 && sca_c.Cin == C && sca_c.Cout == C &&
        res1.Cin == C && res1.Cout == C && res1.Hout == H && res1.Wout == W;

    if (!ok && error)
        *error = "one or more NafBlock tensor shapes do not match nafblock()";

    return ok;
}

inline std::vector<LayerDesc> build_nafnet32_layers()
{
    std::vector<LayerDesc> layers;
    layers.reserve(768);
    int id = 0;
    char nm[64];

    static const int ENC_CH[4]   = {  32,  64, 128, 256 };
    static const int ENC_H[4]    = {  64,  32,  16,   8 };
    static const int ENC_NBLK[4] = {   2,   2,   4,   8 };

    layers.push_back(make_conv_layer(id, "intro", 64, 64, 3, 64, 64, 32, 3, 3, 1, 1));

    for (int lvl = 0; lvl < 4; ++lvl)
    {
        int C = ENC_CH[lvl];
        int H = ENC_H[lvl];

        for (int b = 0; b < ENC_NBLK[lvl]; ++b)
        {
            std::snprintf(nm, sizeof(nm), "enc%d_blk%d", lvl, b);
            append_nafblock_layers(layers, id, nm, C, H, H);
        }

        std::snprintf(nm, sizeof(nm), "down_%d", lvl);
        layers.push_back(make_conv_layer(id, nm, H, H, C, H / 2, H / 2, 2 * C, 2, 2, 2, 0));
    }

    for (int b = 0; b < 12; ++b)
    {
        std::snprintf(nm, sizeof(nm), "mid_blk%d", b);
        append_nafblock_layers(layers, id, nm, 512, 4, 4);
    }

    static const int DEC_CH[4]  = { 256, 128,  64,  32 };
    static const int DEC_H[4]   = {   8,  16,  32,  64 };
    static const int DEC_CIN[4] = { 512, 256, 128,  64 };

    for (int lvl = 0; lvl < 4; ++lvl)
    {
        int Cin = DEC_CIN[lvl];
        int Hin = (lvl == 0) ? 4 : DEC_H[lvl - 1];

        std::snprintf(nm, sizeof(nm), "ups_%d", lvl);
        layers.push_back(make_conv_layer(id, nm,
                                         Hin, Hin, Cin,
                                         Hin, Hin, DEC_CH[lvl] * 4,
                                         1, 1, 1, 0));

        std::snprintf(nm, sizeof(nm), "dec%d_blk0", lvl);
        append_nafblock_layers(layers, id, nm, DEC_CH[lvl], DEC_H[lvl], DEC_H[lvl]);
    }

    layers.push_back(make_conv_layer(id, "ending", 64, 64, 32, 64, 64, 3, 1, 1, 1, 0));

    return layers;
}
