#pragma once

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>

#include "nafnet_hw_config.h"

// ============================================================
// LayerType — which accelerator handles this layer
// ============================================================
enum LayerType
{
    LAYER_CONV,   // standard / pointwise conv  →  matrix accelerator
    LAYER_DWCONV  // depth-wise conv            →  vector accelerator
};

// ============================================================
// LayerDesc — static description of one NAFNet layer
// ============================================================
struct LayerDesc
{
    int  id;
    char name[64];
    LayerType type;

    // Input tensor shape
    int Hin, Win, Cin;
    // Output tensor shape
    int Hout, Wout, Cout;
    // Kernel / stride / padding / groups
    int Kh, Kw, stride, pad, groups;
};

// ------------------------------------------------------------
// Internal ceil-division helper
// ------------------------------------------------------------
static inline uint64_t cdiv64(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

// ============================================================
// Cycle estimation helpers
//   All use hardware constants from nafnet_hw_config.h
// ============================================================

// Matrix-accelerator service cycles for a CONV layer.
//   GEMM is tiled M × K × N; each tile costs MATMUL_ACC_CYCLE cycles.
inline uint64_t conv_mat_cycles(const LayerDesc &l)
{
    uint64_t sp = cdiv64((uint64_t)l.Hout * l.Wout, MATMUL_M);
    uint64_t co = cdiv64((uint64_t)l.Cout,            MATMUL_N);
    uint64_t ci = cdiv64((uint64_t)l.Cin * l.Kh * l.Kw, MATMUL_K);
    return sp * co * ci * MATMUL_ACC_CYCLE;
}

// Vector-accelerator requantize cycles that follow each CONV layer.
//   Processes one element per DWCONV_CAP slots.
inline uint64_t conv_vec_quant_cycles(const LayerDesc &l)
{
    return cdiv64((uint64_t)l.Hout * l.Wout * l.Cout, DWCONV_CAP)
           * DWCONV_ACC_CYCLE;
}

// Vector-accelerator service cycles for a DWCONV layer.
//   Total MACs = Hout × Wout × Cout × Kh × Kw (one accumulate per element).
inline uint64_t dwconv_vec_cycles(const LayerDesc &l)
{
    return cdiv64((uint64_t)l.Hout * l.Wout * l.Cout
                  * (uint64_t)l.Kh * l.Kw,
                  DWCONV_CAP)
           * DWCONV_ACC_CYCLE;
}

// ============================================================
// Memory-traffic helpers (int8, 1 byte per element)
// ============================================================

// Read bytes: input feature map + weight tensor
inline uint64_t layer_rd_bytes(const LayerDesc &l)
{
    uint64_t ifmap   = (uint64_t)l.Hin * l.Win * l.Cin;
    uint64_t weights = (uint64_t)(l.Cin / l.groups) * l.Cout * l.Kh * l.Kw;
    return ifmap + weights;
}

// Write bytes: output feature map
inline uint64_t layer_wr_bytes(const LayerDesc &l)
{
    return (uint64_t)l.Hout * l.Wout * l.Cout;
}

// ============================================================
// Internal factory helper
// ============================================================
static inline LayerDesc make_layer(int &id_ctr, const char *name,
                                   LayerType type,
                                   int Hin, int Win, int Cin,
                                   int Hout, int Wout, int Cout,
                                   int Kh, int Kw,
                                   int stride, int pad, int groups)
{
    LayerDesc l{};
    l.id = id_ctr++;
    std::snprintf(l.name, sizeof(l.name), "%s", name);
    l.type   = type;
    l.Hin    = Hin;  l.Win  = Win;  l.Cin  = Cin;
    l.Hout   = Hout; l.Wout = Wout; l.Cout = Cout;
    l.Kh     = Kh;   l.Kw   = Kw;
    l.stride = stride; l.pad = pad; l.groups = groups;
    return l;
}

// ============================================================
// add_nafblock_layers
//
// Appends the 5 accelerator-mapped operations of one NAFBlock.
//
//   Branch 1: conv1(C→2C) · dw_conv3(2C) · conv2(C→C)
//   Branch 2: conv1(C→2C) · conv2(C→C)
//
// Scalar steps (LayerNorm, SimpleGate, SCA, residual-add)
// are not modelled as accelerator requests in the first version.
// ============================================================
static inline void add_nafblock_layers(std::vector<LayerDesc> &layers,
                                       int &id,
                                       const char *prefix,
                                       int C, int H, int W)
{
    char nm[64];

    // Branch 1 – expand conv1 (C → 2C)
    std::snprintf(nm, sizeof(nm), "%s_b1_expand", prefix);
    layers.push_back(make_layer(id, nm, LAYER_CONV,
                                H, W, C, H, W, 2*C, 1, 1, 1, 0, 1));

    // Branch 1 – depth-wise conv (2C, groups = 2C, 3×3)
    std::snprintf(nm, sizeof(nm), "%s_b1_dw", prefix);
    layers.push_back(make_layer(id, nm, LAYER_DWCONV,
                                H, W, 2*C, H, W, 2*C, 3, 3, 1, 1, 2*C));

    // Branch 1 – project conv (C → C, after SimpleGate halves channels)
    std::snprintf(nm, sizeof(nm), "%s_b1_proj", prefix);
    layers.push_back(make_layer(id, nm, LAYER_CONV,
                                H, W, C, H, W, C, 1, 1, 1, 0, 1));

    // Branch 2 – expand conv1 (C → 2C)
    std::snprintf(nm, sizeof(nm), "%s_b2_expand", prefix);
    layers.push_back(make_layer(id, nm, LAYER_CONV,
                                H, W, C, H, W, 2*C, 1, 1, 1, 0, 1));

    // Branch 2 – project conv (C → C, after SimpleGate)
    std::snprintf(nm, sizeof(nm), "%s_b2_proj", prefix);
    layers.push_back(make_layer(id, nm, LAYER_CONV,
                                H, W, C, H, W, C, 1, 1, 1, 0, 1));
}

// ============================================================
// build_nafnet32_layers
//
// Returns the ordered list of all accelerator-mapped operations
// in a NAFNet-32 forward pass (64×64 input image patch).
//
// Network structure:
//   Intro (3→32, 64×64)
//   Encoder L0  : 2 NAFBlocks @ C=32,  64×64  → down_0 (32→64)
//   Encoder L1  : 2 NAFBlocks @ C=64,  32×32  → down_1 (64→128)
//   Encoder L2  : 4 NAFBlocks @ C=128, 16×16  → down_2 (128→256)
//   Encoder L3  : 8 NAFBlocks @ C=256,  8×8   → down_3 (256→512)
//   Middle      : 12 NAFBlocks @ C=512,  4×4
//   Decoder L0  : ups_0 (512→1024, pixel-shuffle→256@ 8×8) + 1 block
//   Decoder L1  : ups_1 (256→512,  pixel-shuffle→128@16×16) + 1 block
//   Decoder L2  : ups_2 (128→256,  pixel-shuffle→ 64@32×32) + 1 block
//   Decoder L3  : ups_3 ( 64→128,  pixel-shuffle→ 32@64×64) + 1 block
//   Ending (32→3, 64×64)
// ============================================================
inline std::vector<LayerDesc> build_nafnet32_layers()
{
    std::vector<LayerDesc> layers;
    layers.reserve(256);
    int id = 0;
    char nm[64];

    // Encoder config
    static const int ENC_CH[4]   = {  32,  64, 128, 256 };
    static const int ENC_H[4]    = {  64,  32,  16,   8 };
    static const int ENC_NBLK[4] = {   2,   2,   4,   8 };

    // Intro: 3×3 conv, 3 → 32, 64×64
    layers.push_back(make_layer(id, "intro", LAYER_CONV,
                                64, 64, 3, 64, 64, 32, 3, 3, 1, 1, 1));

    // Encoder levels 0–3
    for (int lvl = 0; lvl < 4; ++lvl)
    {
        int C = ENC_CH[lvl];
        int H = ENC_H[lvl];

        for (int b = 0; b < ENC_NBLK[lvl]; ++b)
        {
            std::snprintf(nm, sizeof(nm), "enc%d_blk%d", lvl, b);
            add_nafblock_layers(layers, id, nm, C, H, H);
        }

        // Downsample: 2×2 stride-2 conv, C → 2C
        std::snprintf(nm, sizeof(nm), "down_%d", lvl);
        layers.push_back(make_layer(id, nm, LAYER_CONV,
                                    H, H, C, H/2, H/2, 2*C, 2, 2, 2, 0, 1));
    }

    // Middle: 12 NAFBlocks @ C=512, 4×4
    for (int b = 0; b < 12; ++b)
    {
        std::snprintf(nm, sizeof(nm), "mid_blk%d", b);
        add_nafblock_layers(layers, id, nm, 512, 4, 4);
    }

    // Decoder levels 0–3
    // ups_k: conv1×1 expands C → 4C so pixel-shuffle (r=2) yields C at 2H
    static const int DEC_CH[4]  = { 256, 128,  64,  32 };
    static const int DEC_H[4]   = {   8,  16,  32,  64 };
    static const int DEC_CIN[4] = { 512, 256, 128,  64 }; // input to ups_k

    for (int lvl = 0; lvl < 4; ++lvl)
    {
        int Cin = DEC_CIN[lvl];
        int Hin = (lvl == 0) ? 4 : DEC_H[lvl - 1]; // spatial before ups

        // Upsample conv (operates on the low-res map before pixel-shuffle)
        std::snprintf(nm, sizeof(nm), "ups_%d", lvl);
        layers.push_back(make_layer(id, nm, LAYER_CONV,
                                    Hin, Hin, Cin,
                                    Hin, Hin, DEC_CH[lvl] * 4,
                                    1, 1, 1, 0, 1));

        // Decoder NAFBlock (after pixel-shuffle at higher resolution)
        std::snprintf(nm, sizeof(nm), "dec%d_blk0", lvl);
        add_nafblock_layers(layers, id, nm,
                            DEC_CH[lvl], DEC_H[lvl], DEC_H[lvl]);
    }

    // Ending: 1×1 conv, 32 → 3, 64×64
    layers.push_back(make_layer(id, "ending", LAYER_CONV,
                                64, 64, 32, 64, 64, 3, 1, 1, 1, 0, 1));

    return layers;
}
