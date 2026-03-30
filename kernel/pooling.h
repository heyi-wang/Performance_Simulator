/*
 * mf_pool.h - MatrixFlow Pooling Kernels
 *
 * Header-only C file providing pooling operations using RISC-V Vector:
 *   - 2x2 Max Pooling (int8, int16) for U-Net encoder downsampling
 *   - Global Average Pooling (int8, int16) for NAF-Net SCA (Spatial Channel Attention)
 *
 * Data layout: [C, H, W] channels-first, contiguous.
 * All inputs/outputs reside in L1 memory.
 */

#ifndef MF_POOL_H
#define MF_POOL_H

#include "mf_kernel.h"
#include <riscv_vector.h>

/* ========================================================================
 * 2x2 Max Pooling for int8 - U-Net
 *
 * Performs 2x2 max pooling with stride 2 (non-overlapping).
 *   input:  [C, H, W]       int8
 *   output: [C, H/2, W/2]   int8
 *
 * For each channel, for each 2x2 block:
 *   out[c][oh][ow] = max(in[c][2*oh][2*ow],     in[c][2*oh][2*ow+1],
 *                        in[c][2*oh+1][2*ow],   in[c][2*oh+1][2*ow+1])
 *
 * Vectorized across W/2 output elements per output row:
 *   - Load even-indexed elements from row0 and row1
 *   - Load odd-indexed elements from row0 and row1
 *   - Take pairwise max across the 4 sources
 *
 * H and W must be even.
 *
 * Parameters:
 *   input  - [C, H, W] int8, in L1
 *   output - [C, H/2, W/2] int8, in L1
 *   C      - number of channels
 *   H      - input height (must be even)
 *   W      - input width (must be even)
 * ======================================================================== */
static inline void mf_maxpool2x2_i8(const int8_t *input, int8_t *output,
                                     int C, int H, int W) {
    const int oH = H / 2;
    const int oW = W / 2;

    for (int c = 0; c < C; c++) {
        const int8_t *in_ch  = input  + c * H * W;
        int8_t       *out_ch = output + c * oH * oW;

        for (int oh = 0; oh < oH; oh++) {
            const int8_t *row0 = in_ch + (2 * oh)     * W;
            const int8_t *row1 = in_ch + (2 * oh + 1) * W;
            int8_t *out_row = out_ch + oh * oW;

            int remaining = oW;
            int ow = 0;

            while (remaining > 0) {
                size_t vl = __riscv_vsetvl_e8m4(remaining);

                /*
                 * Load strided: pick every other element.
                 * Stride = 2 bytes (sizeof(int8_t) * 2) to get even indices.
                 * row0[2*ow], row0[2*ow+2], row0[2*ow+4], ...
                 */
                vint8m4_t r0_even = __riscv_vlse8_v_i8m4(row0 + 2 * ow, 2, vl);
                vint8m4_t r0_odd  = __riscv_vlse8_v_i8m4(row0 + 2 * ow + 1, 2, vl);
                vint8m4_t r1_even = __riscv_vlse8_v_i8m4(row1 + 2 * ow, 2, vl);
                vint8m4_t r1_odd  = __riscv_vlse8_v_i8m4(row1 + 2 * ow + 1, 2, vl);

                /* Max of all four */
                vint8m4_t m01 = __riscv_vmax_vv_i8m4(r0_even, r0_odd, vl);
                vint8m4_t m23 = __riscv_vmax_vv_i8m4(r1_even, r1_odd, vl);
                vint8m4_t mfinal = __riscv_vmax_vv_i8m4(m01, m23, vl);

                __riscv_vse8_v_i8m4(out_row + ow, mfinal, vl);

                ow += vl;
                remaining -= vl;
            }
        }
    }
}

/* ========================================================================
 * 2x2 Max Pooling for int16 - U-Net
 *
 * Same as int8 variant but for int16 data.
 *   input:  [C, H, W]       int16
 *   output: [C, H/2, W/2]   int16
 *
 * H and W must be even.
 * ======================================================================== */
static inline void mf_maxpool2x2_i16(const int16_t *input, int16_t *output,
                                      int C, int H, int W) {
    const int oH = H / 2;
    const int oW = W / 2;

    for (int c = 0; c < C; c++) {
        const int16_t *in_ch  = input  + c * H * W;
        int16_t       *out_ch = output + c * oH * oW;

        for (int oh = 0; oh < oH; oh++) {
            const int16_t *row0 = in_ch + (2 * oh)     * W;
            const int16_t *row1 = in_ch + (2 * oh + 1) * W;
            int16_t *out_row = out_ch + oh * oW;

            int remaining = oW;
            int ow = 0;

            while (remaining > 0) {
                size_t vl = __riscv_vsetvl_e16m4(remaining);

                /*
                 * Strided load: stride = 4 bytes (2 * sizeof(int16_t))
                 * to pick every other int16 element.
                 */
                vint16m4_t r0_even = __riscv_vlse16_v_i16m4(row0 + 2 * ow,
                                                             2 * sizeof(int16_t), vl);
                vint16m4_t r0_odd  = __riscv_vlse16_v_i16m4(row0 + 2 * ow + 1,
                                                             2 * sizeof(int16_t), vl);
                vint16m4_t r1_even = __riscv_vlse16_v_i16m4(row1 + 2 * ow,
                                                             2 * sizeof(int16_t), vl);
                vint16m4_t r1_odd  = __riscv_vlse16_v_i16m4(row1 + 2 * ow + 1,
                                                             2 * sizeof(int16_t), vl);

                /* Max of all four */
                vint16m4_t m01 = __riscv_vmax_vv_i16m4(r0_even, r0_odd, vl);
                vint16m4_t m23 = __riscv_vmax_vv_i16m4(r1_even, r1_odd, vl);
                vint16m4_t mfinal = __riscv_vmax_vv_i16m4(m01, m23, vl);

                __riscv_vse16_v_i16m4(out_row + ow, mfinal, vl);

                ow += vl;
                remaining -= vl;
            }
        }
    }
}

/* ========================================================================
 * Global Average Pooling for int8 - NAF-Net SCA
 *
 * Computes one output value per channel by averaging all H*W spatial elements.
 *   input:  [C, H, W]   int8
 *   output: [C]          int32 (to avoid overflow; caller can re-quantize)
 *
 * For each channel c:
 *   output[c] = sum(input[c, :, :]) / (H * W)
 *
 * Uses RVV vredsum for efficient reduction per channel.
 *
 * Parameters:
 *   input  - [C, H, W] int8, in L1
 *   output - [C] int32, in L1
 *   C      - number of channels
 *   H      - spatial height
 *   W      - spatial width
 * ======================================================================== */
static inline void mf_global_avgpool_i8(const int8_t *input, int32_t *output,
                                         int C, int H, int W) {
    const int spatial = H * W;

    for (int c = 0; c < C; c++) {
        const int8_t *in_ch = input + c * spatial;

        int32_t total_sum = 0;
        int remaining = spatial;
        const int8_t *p = in_ch;

        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e8m4(remaining);
            vint8m4_t v = __riscv_vle8_v_i8m4(p, vl);

            /* Widen int8 -> int16 to avoid overflow in reduction */
            vint16m8_t vw = __riscv_vwmul_vx_i16m8(v, 1, vl);

            /* Reduce sum */
            vint16m1_t vzero = __riscv_vmv_v_x_i16m1(0, 1);
            vint16m1_t vsum = __riscv_vredsum_vs_i16m8_i16m1(vw, vzero, vl);
            total_sum += (int32_t)__riscv_vmv_x_s_i16m1_i16(vsum);

            p += vl;
            remaining -= vl;
        }

        output[c] = total_sum / spatial;
    }
}

/* ========================================================================
 * Global Average Pooling for int16 - NAF-Net SCA
 *
 * Same as int8 variant but for int16 data.
 *   input:  [C, H, W]   int16
 *   output: [C]          int32
 *
 * For each channel c:
 *   output[c] = sum(input[c, :, :]) / (H * W)
 *
 * Uses RVV vredsum with int32 accumulation for overflow safety.
 *
 * Parameters:
 *   input  - [C, H, W] int16, in L1
 *   output - [C] int32, in L1
 *   C      - number of channels
 *   H      - spatial height
 *   W      - spatial width
 * ======================================================================== */
static inline void mf_global_avgpool_i16(const int16_t *input, int32_t *output,
                                          int C, int H, int W) {
    const int spatial = H * W;

    for (int c = 0; c < C; c++) {
        const int16_t *in_ch = input + c * spatial;

        int64_t total_sum = 0;
        int remaining = spatial;
        const int16_t *p = in_ch;

        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e16m4(remaining);
            vint16m4_t v = __riscv_vle16_v_i16m4(p, vl);

            /* Widen int16 -> int32 to avoid overflow in reduction */
            vint32m8_t vw = __riscv_vwmul_vx_i32m8(v, 1, vl);

            /* Reduce sum */
            vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
            vint32m1_t vsum = __riscv_vredsum_vs_i32m8_i32m1(vw, vzero, vl);
            total_sum += (int64_t)__riscv_vmv_x_s_i32m1_i32(vsum);

            p += vl;
            remaining -= vl;
        }

        output[c] = (int32_t)(total_sum / spatial);
    }
}

#endif /* MF_POOL_H */

