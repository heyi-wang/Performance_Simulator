/*
 * mf_norm.h - MatrixFlow Normalization Kernels
 *
 * Header-only C file providing normalization operations using RISC-V Vector:
 *   - LayerNorm2d (int16, int8) for NAF-Net: per-channel normalization over H*W
 *   - Fused BatchNorm (int8, int16) for U-Net inference: pre-computed scale+bias
 *
 * Data layout: [C, H, W] channels-first, contiguous.
 * All inputs/outputs reside in L1 memory.
 */

#ifndef MF_NORM_H
#define MF_NORM_H

#include "mf_kernel.h"
#include <riscv_vector.h>

/* ========================================================================
 * Internal: Integer square root approximation (Newton's method)
 *
 * Computes floor(sqrt(x)) for unsigned 32-bit input.
 * Used for computing 1/sqrt(variance) in LayerNorm.
 * ======================================================================== */
static inline uint32_t mf_isqrt(uint32_t x) {
    if (x == 0) return 0;
    if (x == 1) return 1;

    /* Initial estimate: find highest set bit, start near sqrt */
    uint32_t r = x;
    uint32_t bit = 1u << 30;

    /* Find the highest power of 4 <= x */
    while (bit > x)
        bit >>= 2;

    r = 0;
    while (bit != 0) {
        if (x >= r + bit) {
            x -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
        bit >>= 2;
    }
    return r;
}

/* ========================================================================
 * Internal: Count leading zeros for shift-based reciprocal approximation
 * ======================================================================== */
static inline int mf_clz32(uint32_t x) {
    if (x == 0) return 32;
    int n = 0;
    if ((x & 0xFFFF0000u) == 0) { n += 16; x <<= 16; }
    if ((x & 0xFF000000u) == 0) { n +=  8; x <<=  8; }
    if ((x & 0xF0000000u) == 0) { n +=  4; x <<=  4; }
    if ((x & 0xC0000000u) == 0) { n +=  2; x <<=  2; }
    if ((x & 0x80000000u) == 0) { n +=  1; }
    return n;
}

/* ========================================================================
 * LayerNorm2d for int16 - NAF-Net
 *
 * For each channel c in [0, C):
 *   1. Compute mean = sum(input[c, :, :]) / (H * W)
 *   2. Compute var  = sum((input[c, :, :] - mean)^2) / (H * W)
 *   3. Normalize: out[c,h,w] = gamma[c] * (in[c,h,w] - mean) / sqrt(var + eps) + beta[c]
 *
 * Fixed-point approximation:
 *   - mean computed as integer division (sum / N)
 *   - variance computed as (sum_sq / N) - mean^2
 *   - 1/sqrt(var) approximated via shift: find s such that 2^s ~ 1/sqrt(var)
 *   - gamma/beta applied as scale+shift in fixed-point
 *
 * Parameters:
 *   input     - [C, H, W] int16, in L1
 *   output    - [C, H, W] int16, in L1
 *   gamma     - [C] int16, per-channel scale (Q8.8 or similar fixed-point)
 *   beta      - [C] int16, per-channel bias
 *   C, H, W   - tensor dimensions
 *   workspace - L1 scratch for intermediate computations
 *
 * Returns: MF_OK or MF_ERR_INVALID_PARAM
 * ======================================================================== */
static inline int mf_layernorm2d_i16(const int16_t *input, int16_t *output,
                                     const int16_t *gamma, const int16_t *beta,
                                     int C, int H, int W,
                                     mf_l1_buf_t workspace) {
    if (!input || !output || !gamma || !beta || C <= 0 || H <= 0 || W <= 0)
        return MF_ERR_INVALID_PARAM;

    const int spatial = H * W;

    /* Process each channel independently */
    for (int c = 0; c < C; c++) {
        const int16_t *in_ch  = input  + c * spatial;
        int16_t       *out_ch = output + c * spatial;
        int16_t g = gamma[c];
        int16_t b = beta[c];

        /* ----- Step 1: Compute sum using RVV vredsum ----- */
        int32_t total_sum = 0;
        int remaining = spatial;
        const int16_t *p = in_ch;

        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e16m4(remaining);
            vint16m4_t v = __riscv_vle16_v_i16m4(p, vl);
            /* Widen to 32-bit for accumulation */
            vint32m8_t vw = __riscv_vwmul_vx_i32m8(v, 1, vl);
            /* Reduce: sum all elements in vector */
            vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
            vint32m1_t vsum = __riscv_vredsum_vs_i32m8_i32m1(vw, vzero, vl);
            total_sum += __riscv_vmv_x_s_i32m1_i32(vsum);
            p += vl;
            remaining -= vl;
        }

        int32_t mean = total_sum / spatial;

        /* ----- Step 2: Compute variance using RVV ----- */
        int64_t total_sq = 0;
        remaining = spatial;
        p = in_ch;

        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e16m4(remaining);
            vint16m4_t v = __riscv_vle16_v_i16m4(p, vl);
            /* Subtract mean (broadcast) */
            vint16m4_t vdiff = __riscv_vsub_vx_i16m4(v, (int16_t)mean, vl);
            /* Widen to 32-bit and square: diff * diff */
            vint32m8_t vsq = __riscv_vwmul_vv_i32m8(vdiff, vdiff, vl);
            /* Reduce sum of squares */
            vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
            vint32m1_t vsum = __riscv_vredsum_vs_i32m8_i32m1(vsq, vzero, vl);
            total_sq += (int64_t)__riscv_vmv_x_s_i32m1_i32(vsum);
            p += vl;
            remaining -= vl;
        }

        /* var = sum_sq / N  (mean of squared deviations) */
        int32_t var = (int32_t)(total_sq / spatial);

        /* Add epsilon (1 in integer domain to avoid division by zero) */
        if (var < 1) var = 1;

        /* ----- Step 3: Approximate 1/sqrt(var) using shifts ----- */
        /*
         * We want: normalized = (x - mean) * gamma / sqrt(var) + beta
         *
         * Approximation: inv_std ~ (1 << shift) / sqrt(var)
         * We use a fixed-point representation:
         *   inv_std_fp = (1 << FRAC_BITS) / isqrt(var)
         * Then: normalized = ((x - mean) * gamma * inv_std_fp) >> FRAC_BITS + beta
         */
        uint32_t std_dev = mf_isqrt((uint32_t)var);
        if (std_dev == 0) std_dev = 1;

        /* Use 14-bit fractional precision for inv_std */
        const int FRAC_BITS = 14;
        int32_t inv_std_fp = (1 << FRAC_BITS) / (int32_t)std_dev;

        /* ----- Step 4: Normalize using RVV ----- */
        remaining = spatial;
        p = in_ch;
        int16_t *q = out_ch;

        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e16m4(remaining);
            vint16m4_t v = __riscv_vle16_v_i16m4(p, vl);

            /* Subtract mean */
            vint16m4_t vdiff = __riscv_vsub_vx_i16m4(v, (int16_t)mean, vl);

            /* Widen to 32-bit for multiplication precision */
            vint32m8_t vw = __riscv_vwmul_vx_i32m8(vdiff, 1, vl);

            /* Multiply by inv_std_fp */
            vw = __riscv_vmul_vx_i32m8(vw, inv_std_fp, vl);

            /* Multiply by gamma (scale) */
            vw = __riscv_vmul_vx_i32m8(vw, (int32_t)g, vl);

            /* Arithmetic right shift to remove fractional bits + gamma Q format
             * Assuming gamma is Q8.8, total shift = FRAC_BITS + 8 = 22
             * But to keep precision, shift by FRAC_BITS first
             */
            vw = __riscv_vsra_vx_i32m8(vw, FRAC_BITS, vl);

            /* Add beta (widen beta to 32-bit) */
            vw = __riscv_vadd_vx_i32m8(vw, (int32_t)b, vl);

            /* Narrow back to int16 with saturation */
            /* Clip to int16 range */
            vw = __riscv_vmax_vx_i32m8(vw, -32768, vl);
            vw = __riscv_vmin_vx_i32m8(vw, 32767, vl);

            /* Narrow: 32->16 */
            vint16m4_t vout = __riscv_vnsra_wx_i16m4(vw, 0, vl);

            __riscv_vse16_v_i16m4(q, vout, vl);

            p += vl;
            q += vl;
            remaining -= vl;
        }
    }

    return MF_OK;
}

/* ========================================================================
 * LayerNorm2d for int8 - NAF-Net
 *
 * Same algorithm as int16 variant but operating on int8 data.
 * Internal accumulation uses int16/int32 to avoid overflow.
 *
 * Parameters:
 *   input     - [C, H, W] int8, in L1
 *   output    - [C, H, W] int8, in L1
 *   gamma     - [C] int8, per-channel scale
 *   beta      - [C] int8, per-channel bias
 *   C, H, W   - tensor dimensions
 *   workspace - L1 scratch
 *
 * Returns: MF_OK or MF_ERR_INVALID_PARAM
 * ======================================================================== */
static inline int mf_layernorm2d_i8(const int8_t *input, int8_t *output,
                                    const int8_t *gamma, const int8_t *beta,
                                    int C, int H, int W,
                                    mf_l1_buf_t workspace) {
    if (!input || !output || !gamma || !beta || C <= 0 || H <= 0 || W <= 0)
        return MF_ERR_INVALID_PARAM;

    const int spatial = H * W;

    for (int c = 0; c < C; c++) {
        const int8_t *in_ch  = input  + c * spatial;
        int8_t       *out_ch = output + c * spatial;
        int8_t g = gamma[c];
        int8_t b = beta[c];

        /* ----- Step 1: Compute sum using RVV vredsum ----- */
        int32_t total_sum = 0;
        int remaining = spatial;
        const int8_t *p = in_ch;

        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e8m4(remaining);
            vint8m4_t v = __riscv_vle8_v_i8m4(p, vl);
            /* Widen int8 -> int16 for safe accumulation */
            vint16m8_t vw = __riscv_vwmul_vx_i16m8(v, 1, vl);
            /* Reduce sum */
            vint16m1_t vzero = __riscv_vmv_v_x_i16m1(0, 1);
            vint16m1_t vsum = __riscv_vredsum_vs_i16m8_i16m1(vw, vzero, vl);
            total_sum += (int32_t)__riscv_vmv_x_s_i16m1_i16(vsum);
            p += vl;
            remaining -= vl;
        }

        int32_t mean = total_sum / spatial;

        /* ----- Step 2: Compute variance ----- */
        int64_t total_sq = 0;
        remaining = spatial;
        p = in_ch;

        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e8m2(remaining);
            vint8m2_t v = __riscv_vle8_v_i8m2(p, vl);
            /* Widen int8 -> int16 (m2 -> m4) */
            vint16m4_t vw = __riscv_vwmul_vx_i16m4(v, 1, vl);
            /* Subtract mean */
            vint16m4_t vdiff = __riscv_vsub_vx_i16m4(vw, (int16_t)mean, vl);
            /* Square: widen int16 m4 -> int32 m8 */
            vint32m8_t vsq = __riscv_vwmul_vv_i32m8(vdiff, vdiff, vl);
            /* Reduce sum of squares */
            vint32m1_t vzero32 = __riscv_vmv_v_x_i32m1(0, 1);
            vint32m1_t vsum = __riscv_vredsum_vs_i32m8_i32m1(vsq, vzero32, vl);
            total_sq += (int64_t)__riscv_vmv_x_s_i32m1_i32(vsum);
            p += vl;
            remaining -= vl;
        }

        int32_t var = (int32_t)(total_sq / spatial);
        if (var < 1) var = 1;

        /* ----- Step 3: Approximate 1/sqrt(var) ----- */
        uint32_t std_dev = mf_isqrt((uint32_t)var);
        if (std_dev == 0) std_dev = 1;

        const int FRAC_BITS = 10;  /* Less precision needed for int8 */
        int32_t inv_std_fp = (1 << FRAC_BITS) / (int32_t)std_dev;

        /* ----- Step 4: Normalize using RVV ----- */
        remaining = spatial;
        p = in_ch;
        int8_t *q = out_ch;

        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e8m2(remaining);
            vint8m2_t v = __riscv_vle8_v_i8m2(p, vl);

            /* Widen int8 -> int16 (m2 -> m4) */
            vint16m4_t vw = __riscv_vwmul_vx_i16m4(v, 1, vl);

            /* Subtract mean */
            vint16m4_t vdiff = __riscv_vsub_vx_i16m4(vw, (int16_t)mean, vl);

            /* Widen to int32 and multiply by inv_std_fp */
            vint32m8_t v32 = __riscv_vwmul_vx_i32m8(vdiff, (int16_t)inv_std_fp, vl);

            /* Multiply by gamma */
            v32 = __riscv_vmul_vx_i32m8(v32, (int32_t)g, vl);

            /* Shift right by fractional bits */
            v32 = __riscv_vsra_vx_i32m8(v32, FRAC_BITS, vl);

            /* Add beta */
            v32 = __riscv_vadd_vx_i32m8(v32, (int32_t)b, vl);

            /* Clip to int8 range [-128, 127] */
            v32 = __riscv_vmax_vx_i32m8(v32, -128, vl);
            v32 = __riscv_vmin_vx_i32m8(v32, 127, vl);

            /* Narrow int32 -> int16 -> int8 */
            vint16m4_t v16 = __riscv_vnsra_wx_i16m4(v32, 0, vl);
            vint8m2_t v8 = __riscv_vnsra_wx_i8m2(v16, 0, vl);

            __riscv_vse8_v_i8m2(q, v8, vl);

            p += vl;
            q += vl;
            remaining -= vl;
        }
    }

    return MF_OK;
}

/* ========================================================================
 * Fused BatchNorm for int8 - U-Net inference
 *
 * Pre-fused BN where running mean/var are folded into scale and bias:
 *   output[c, h, w] = clamp(scale[c] * input[c, h, w] + bias[c], -128, 127)
 *
 * scale and bias are int16 per-channel values pre-computed from:
 *   scale[c] = gamma[c] / sqrt(running_var[c] + eps)   (quantized to int16)
 *   bias[c]  = beta[c] - scale[c] * running_mean[c]    (quantized to int16)
 *
 * Parameters:
 *   input  - [C, H, W] int8, in L1
 *   output - [C, H, W] int8, in L1
 *   scale  - [C] int16, per-channel scale (Q8.8 fixed-point)
 *   bias   - [C] int16, per-channel bias
 *   C, H, W - tensor dimensions
 * ======================================================================== */
static inline void mf_batchnorm_fused_i8(const int8_t *input, int8_t *output,
                                          const int16_t *scale,
                                          const int16_t *bias,
                                          int C, int H, int W) {
    const int spatial = H * W;

    for (int c = 0; c < C; c++) {
        const int8_t *in_ch  = input  + c * spatial;
        int8_t       *out_ch = output + c * spatial;
        int16_t s = scale[c];
        int16_t b = bias[c];

        int remaining = spatial;
        const int8_t *p = in_ch;
        int8_t *q = out_ch;

        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e8m2(remaining);
            vint8m2_t v = __riscv_vle8_v_i8m2(p, vl);

            /* Widen int8 -> int16 */
            vint16m4_t vw = __riscv_vwmul_vx_i16m4(v, 1, vl);

            /* Multiply by scale: int16 * int16 -> int32 */
            vint32m8_t v32 = __riscv_vwmul_vx_i32m8(vw, s, vl);

            /* Scale is Q8.8 fixed-point, shift right by 8 to get integer part */
            v32 = __riscv_vsra_vx_i32m8(v32, 8, vl);

            /* Add bias */
            v32 = __riscv_vadd_vx_i32m8(v32, (int32_t)b, vl);

            /* Clip to int8 range */
            v32 = __riscv_vmax_vx_i32m8(v32, -128, vl);
            v32 = __riscv_vmin_vx_i32m8(v32, 127, vl);

            /* Narrow int32 -> int16 -> int8 */
            vint16m4_t v16 = __riscv_vnsra_wx_i16m4(v32, 0, vl);
            vint8m2_t vout = __riscv_vnsra_wx_i8m2(v16, 0, vl);

            __riscv_vse8_v_i8m2(q, vout, vl);

            p += vl;
            q += vl;
            remaining -= vl;
        }
    }
}

/* ========================================================================
 * Fused BatchNorm for int16 - U-Net inference
 *
 * Same as int8 variant but with int16 input/output.
 *   output[c, h, w] = clamp(scale[c] * input[c, h, w] + bias[c], -32768, 32767)
 *
 * Parameters:
 *   input  - [C, H, W] int16, in L1
 *   output - [C, H, W] int16, in L1
 *   scale  - [C] int16, per-channel scale (Q8.8 fixed-point)
 *   bias   - [C] int16, per-channel bias
 *   C, H, W - tensor dimensions
 * ======================================================================== */
static inline void mf_batchnorm_fused_i16(const int16_t *input, int16_t *output,
                                           const int16_t *scale,
                                           const int16_t *bias,
                                           int C, int H, int W) {
    const int spatial = H * W;

    for (int c = 0; c < C; c++) {
        const int16_t *in_ch  = input  + c * spatial;
        int16_t       *out_ch = output + c * spatial;
        int16_t s = scale[c];
        int16_t b = bias[c];

        int remaining = spatial;
        const int16_t *p = in_ch;
        int16_t *q = out_ch;

        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e16m4(remaining);
            vint16m4_t v = __riscv_vle16_v_i16m4(p, vl);

            /* Widen int16 -> int32 and multiply by scale */
            vint32m8_t v32 = __riscv_vwmul_vx_i32m8(v, s, vl);

            /* Scale is Q8.8 fixed-point, shift right by 8 */
            v32 = __riscv_vsra_vx_i32m8(v32, 8, vl);

            /* Add bias */
            v32 = __riscv_vadd_vx_i32m8(v32, (int32_t)b, vl);

            /* Clip to int16 range */
            v32 = __riscv_vmax_vx_i32m8(v32, -32768, vl);
            v32 = __riscv_vmin_vx_i32m8(v32, 32767, vl);

            /* Narrow int32 -> int16 */
            vint16m4_t vout = __riscv_vnsra_wx_i16m4(v32, 0, vl);

            __riscv_vse16_v_i16m4(q, vout, vl);

            p += vl;
            q += vl;
            remaining -= vl;
        }
    }
}

#endif /* MF_NORM_H */

