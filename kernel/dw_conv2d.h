/*
 * mf_dw_conv2d.h - MatrixFlow Depthwise Separable Convolution Kernels
 *
 * Implements depthwise convolution using RISC-V Vector (RVV) instructions.
 * Depthwise convolution applies a separate filter per channel -- it cannot
 * be expressed as a single GeMM and therefore uses RVV instead of tmma.
 *
 * Supported operations:
 *   - 3x3 depthwise convolution (int8, int16)
 *
 * Data layout:
 *   input:  [C, H, W] row-major
 *   kernel: [C, 3, 3] row-major  (one 3x3 filter per channel)
 *   output: [C, outH, outW] row-major
 *
 * Vectorization strategy:
 *   - For each channel and output row, vectorize across the W dimension
 *   - Load 3 input rows covering the 3x3 window
 *   - For each of the 9 kernel elements, broadcast-multiply and accumulate
 *   - Uses widening multiply-accumulate for int8->int32 / int16->int32
 *
 * All inputs and outputs reside in L1.
 */

#ifndef MF_DW_CONV2D_H
#define MF_DW_CONV2D_H

#include "mf_kernel.h"
#include <riscv_vector.h>

/* ========================================================================
 * INT8 Depthwise 3x3 Convolution
 * ======================================================================== */

/*
 * mf_dw_conv2d_3x3_i8 - Depthwise 3x3 Convolution (int8 -> int32)
 *
 * Each channel c has its own 3x3 filter kernel[c][3][3].
 *   output[c][oh][ow] = sum_{kh,kw} input[c][oh-pad+kh][ow-pad+kw] * kernel[c][kh][kw]
 *
 * Parameters:
 *   input     - [C, H, W] row-major, int8
 *   kernel    - [C, 3, 3] row-major, int8 (one 3x3 filter per channel)
 *   output    - [C, outH, outW] row-major, int32
 *   C         - number of channels
 *   H, W      - input spatial dimensions
 *   pad       - symmetric zero-padding
 *   workspace - L1 scratch for padded row buffers
 *
 * outH = H + 2*pad - 2, outW = W + 2*pad - 2
 *
 * Returns MF_OK on success, MF_ERR_* on failure.
 */
static inline int mf_dw_conv2d_3x3_i8(const int8_t *input,
                                        const int8_t *kernel,
                                        int32_t *output,
                                        int C, int H, int W,
                                        int pad,
                                        mf_l1_buf_t workspace) {
    const int kH = 3, kW = 3;
    const int outH = H + 2 * pad - kH + 1;
    const int outW = W + 2 * pad - kW + 1;

    if (outH <= 0 || outW <= 0 || C <= 0)
        return MF_ERR_INVALID_PARAM;

    /*
     * Row-vectorized approach: vectorize across the W (output width) dimension.
     * For each output row, process vl pixels per strip using contiguous loads
     * and scalar-broadcast widening multiply-accumulate (vwmacc_vx).
     *
     * 1. Zero-pad input into workspace (padH × padW per channel)
     * 2. For each output row strip: 9 contiguous loads + 9 vsext + 9 vwmacc_vx
     *    → vl output pixels per iteration.
     *
     * Scales with VLEN: wider vectors process more pixels per strip.
     */
    const int padH = H + 2 * pad;
    const int padW = W + 2 * pad;
    const int padPlane = padH * padW;

    /* Use workspace for zero-padded input (1 channel at a time) */
    int8_t *padded = (int8_t *)workspace.ptr;

    for (int c = 0; c < C; c++) {
        const int8_t *in_c  = input  + c * H * W;
        const int8_t *kr_c  = kernel + c * kH * kW;
        int32_t      *out_c = output + c * outH * outW;

        /* Zero-pad: clear padded buffer then copy input rows */
        mf_memset(padded, 0, (size_t)padPlane);
        for (int h = 0; h < H; h++)
            mf_memcpy(padded + (h + pad) * padW + pad,
                       in_c + h * W, (size_t)W);

        /* Load 9 kernel weights as scalars (hoisted per channel) */
        int16_t kw16[9];
        for (int i = 0; i < 9; i++)
            kw16[i] = (int16_t)kr_c[i];

        /* Process each output row, vectorized across W */
        for (int oh = 0; oh < outH; oh++) {
            int32_t *out_row = out_c + oh * outW;
            int ow = 0;

            while (ow < outW) {
                size_t vl = __riscv_vsetvl_e8m2((size_t)(outW - ow));

                /* Zero accumulator (i32, LMUL=8 for widening from i16m4) */
                vint32m8_t acc = __riscv_vmv_v_x_i32m8(0, vl);

                /* 9 kernel positions: contiguous load + sign-extend + vwmacc */
                for (int kh = 0; kh < kH; kh++) {
                    const int8_t *row = padded + (oh + kh) * padW + ow;
                    for (int kw = 0; kw < kW; kw++) {
                        vint8m2_t vin = __riscv_vle8_v_i8m2(row + kw, vl);
                        vint16m4_t v16 = __riscv_vsext_vf2_i16m4(vin, vl);
                        acc = __riscv_vwmacc_vx_i32m8(acc, kw16[kh * kW + kw], v16, vl);
                    }
                }

                /* Store vl output pixels */
                __riscv_vse32_v_i32m8(out_row + ow, acc, vl);
                ow += (int)vl;
            }
        }
    }

    return MF_OK;
}

/* ========================================================================
 * INT16 Depthwise 3x3 Convolution
 * ======================================================================== */

/*
 * mf_dw_conv2d_3x3_i16 - Depthwise 3x3 Convolution (int16 -> int32)
 *
 * Same algorithm as int8 variant but operates on int16 input/kernel data.
 *
 * Parameters:
 *   input     - [C, H, W] row-major, int16
 *   kernel    - [C, 3, 3] row-major, int16
 *   output    - [C, outH, outW] row-major, int32
 *   C         - number of channels
 *   H, W      - input spatial dimensions
 *   pad       - symmetric zero-padding
 *   workspace - L1 scratch (currently unused)
 *
 * Returns MF_OK on success, MF_ERR_* on failure.
 */
static inline int mf_dw_conv2d_3x3_i16(const int16_t *input,
                                         const int16_t *kernel,
                                         int32_t *output,
                                         int C, int H, int W,
                                         int pad,
                                         mf_l1_buf_t workspace) {
    const int kH = 3, kW = 3;
    const int outH = H + 2 * pad - kH + 1;
    const int outW = W + 2 * pad - kW + 1;

    if (outH <= 0 || outW <= 0 || C <= 0)
        return MF_ERR_INVALID_PARAM;

    (void)workspace;

    for (int c = 0; c < C; c++) {
        const int16_t *in_c  = input  + c * H * W;
        const int16_t *kr_c  = kernel + c * kH * kW;
        int32_t       *out_c = output + c * outH * outW;

        /* Load 9 kernel weights */
        int16_t k00 = kr_c[0], k01 = kr_c[1], k02 = kr_c[2];
        int16_t k10 = kr_c[3], k11 = kr_c[4], k12 = kr_c[5];
        int16_t k20 = kr_c[6], k21 = kr_c[7], k22 = kr_c[8];

        for (int oh = 0; oh < outH; oh++) {
            int ow = 0;
            while (ow < outW) {
                size_t vl = __riscv_vsetvl_e32m4((size_t)(outW - ow));

                vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);

                for (int kh = 0; kh < kH; kh++) {
                    int ih = oh - pad + kh;
                    int16_t kv0, kv1, kv2;
                    switch (kh) {
                    case 0: kv0 = k00; kv1 = k01; kv2 = k02; break;
                    case 1: kv0 = k10; kv1 = k11; kv2 = k12; break;
                    default: kv0 = k20; kv1 = k21; kv2 = k22; break;
                    }

                    if (ih < 0 || ih >= H)
                        continue;

                    const int16_t *in_row = in_c + ih * W;

                    for (int kw_idx = 0; kw_idx < kW; kw_idx++) {
                        int iw_base = ow - pad + kw_idx;
                        int16_t kval;
                        switch (kw_idx) {
                        case 0: kval = kv0; break;
                        case 1: kval = kv1; break;
                        default: kval = kv2; break;
                        }

                        if (kval == 0)
                            continue;

                        if (iw_base >= 0 && iw_base + (int)vl <= W) {
                            /* Fast path: all elements in bounds */
                            vint16m2_t vin = __riscv_vle16_v_i16m2(in_row + iw_base, vl);

                            /* Broadcast kernel value */
                            vint16m2_t kbroad = __riscv_vmv_v_x_i16m2(kval, vl);

                            /* Widening multiply: int16 * int16 -> int32 */
                            vint32m4_t prod = __riscv_vwmul_vv_i32m4(vin, kbroad, vl);

                            acc = __riscv_vadd_vv_i32m4(acc, prod, vl);
                        } else {
                            /* Boundary: use temp buffer */
                            int32_t tmp[64];
                            __riscv_vse32_v_i32m4(tmp, acc, vl);
                            for (size_t vi = 0; vi < vl; vi++) {
                                int iw = iw_base + (int)vi;
                                if (iw >= 0 && iw < W) {
                                    tmp[vi] += (int32_t)in_row[iw] * (int32_t)kval;
                                }
                            }
                            acc = __riscv_vle32_v_i32m4(tmp, vl);
                        }
                    }
                }

                __riscv_vse32_v_i32m4(out_c + oh * outW + ow, acc, vl);
                ow += (int)vl;
            }
        }
    }

    return MF_OK;
}

#endif /* MF_DW_CONV2D_H */

