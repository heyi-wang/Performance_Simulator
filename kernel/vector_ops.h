/*
 * mf_vector_ops.h - MatrixFlow RVV Element-wise Vector Operations
 *
 * Header-only library providing RISC-V Vector (RVV 1.0) based element-wise
 * operations for neural network inference on bare-metal rv64imacv targets.
 *
 * All functions operate on data already resident in L1 (DTCM) memory.
 * They use RVV stripmining loops for portable, length-agnostic vectorization.
 *
 * Supported operations:
 *   - ReLU (int8, int16)
 *   - Element-wise add (int8, int16, int32)
 *   - Element-wise mul (int8, int16, int32)
 *   - Scalar broadcast mul (int8, int16, int32)
 *   - Quantize int32 -> int8/int16
 *   - Dequantize int8 -> int32
 *   - Per-channel bias add (int32)
 */

#ifndef MF_VECTOR_OPS_H
#define MF_VECTOR_OPS_H

#include "mf_kernel.h"
#include <riscv_vector.h>

/* ========================================================================
 * ReLU: max(0, x)
 * ======================================================================== */

/*
 * mf_relu_i8 - Apply ReLU activation on int8 data.
 *
 * out[i] = max(0, in[i]) for i in [0, n)
 * in and out may alias (in-place operation is safe).
 */
static inline void mf_relu_i8(const int8_t *in, int8_t *out, int n) {
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e8m4(n - i);
        vint8m4_t va = __riscv_vle8_v_i8m4(in + i, vl);
        vint8m4_t vz = __riscv_vmv_v_x_i8m4(0, vl);
        vint8m4_t vr = __riscv_vmax_vv_i8m4(va, vz, vl);
        __riscv_vse8_v_i8m4(out + i, vr, vl);
        i += vl;
    }
}

/*
 * mf_relu_i16 - Apply ReLU activation on int16 data.
 *
 * out[i] = max(0, in[i]) for i in [0, n)
 */
static inline void mf_relu_i16(const int16_t *in, int16_t *out, int n) {
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e16m4(n - i);
        vint16m4_t va = __riscv_vle16_v_i16m4(in + i, vl);
        vint16m4_t vz = __riscv_vmv_v_x_i16m4(0, vl);
        vint16m4_t vr = __riscv_vmax_vv_i16m4(va, vz, vl);
        __riscv_vse16_v_i16m4(out + i, vr, vl);
        i += vl;
    }
}

/* ========================================================================
 * Element-wise Addition: out = a + b
 * ======================================================================== */

/*
 * mf_elemwise_add_i8 - Element-wise addition of two int8 vectors.
 *
 * out[i] = a[i] + b[i] for i in [0, n)
 * Result wraps on overflow (standard C signed int8 behavior).
 */
static inline void mf_elemwise_add_i8(const int8_t *a, const int8_t *b,
                                       int8_t *out, int n) {
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e8m4(n - i);
        vint8m4_t va = __riscv_vle8_v_i8m4(a + i, vl);
        vint8m4_t vb = __riscv_vle8_v_i8m4(b + i, vl);
        vint8m4_t vr = __riscv_vadd_vv_i8m4(va, vb, vl);
        __riscv_vse8_v_i8m4(out + i, vr, vl);
        i += vl;
    }
}

/*
 * mf_elemwise_add_i16 - Element-wise addition of two int16 vectors.
 */
static inline void mf_elemwise_add_i16(const int16_t *a, const int16_t *b,
                                        int16_t *out, int n) {
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e16m4(n - i);
        vint16m4_t va = __riscv_vle16_v_i16m4(a + i, vl);
        vint16m4_t vb = __riscv_vle16_v_i16m4(b + i, vl);
        vint16m4_t vr = __riscv_vadd_vv_i16m4(va, vb, vl);
        __riscv_vse16_v_i16m4(out + i, vr, vl);
        i += vl;
    }
}

/*
 * mf_elemwise_add_i32 - Element-wise addition of two int32 vectors.
 */
static inline void mf_elemwise_add_i32(const int32_t *a, const int32_t *b,
                                        int32_t *out, int n) {
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e32m4(n - i);
        vint32m4_t va = __riscv_vle32_v_i32m4(a + i, vl);
        vint32m4_t vb = __riscv_vle32_v_i32m4(b + i, vl);
        vint32m4_t vr = __riscv_vadd_vv_i32m4(va, vb, vl);
        __riscv_vse32_v_i32m4(out + i, vr, vl);
        i += vl;
    }
}

/* ========================================================================
 * Element-wise Multiplication: out = a * b
 * ======================================================================== */

/*
 * mf_elemwise_mul_i8 - Element-wise multiplication of two int8 vectors.
 *
 * out[i] = (int8_t)(a[i] * b[i]) for i in [0, n)
 * The multiplication is performed at the native element width; the result
 * is the low 8 bits (truncating / wrapping semantics).
 */
static inline void mf_elemwise_mul_i8(const int8_t *a, const int8_t *b,
                                       int8_t *out, int n) {
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e8m4(n - i);
        vint8m4_t va = __riscv_vle8_v_i8m4(a + i, vl);
        vint8m4_t vb = __riscv_vle8_v_i8m4(b + i, vl);
        vint8m4_t vr = __riscv_vmul_vv_i8m4(va, vb, vl);
        __riscv_vse8_v_i8m4(out + i, vr, vl);
        i += vl;
    }
}

/*
 * mf_elemwise_mul_i16 - Element-wise multiplication of two int16 vectors.
 *
 * Truncating: result is the low 16 bits of the full product.
 */
static inline void mf_elemwise_mul_i16(const int16_t *a, const int16_t *b,
                                        int16_t *out, int n) {
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e16m4(n - i);
        vint16m4_t va = __riscv_vle16_v_i16m4(a + i, vl);
        vint16m4_t vb = __riscv_vle16_v_i16m4(b + i, vl);
        vint16m4_t vr = __riscv_vmul_vv_i16m4(va, vb, vl);
        __riscv_vse16_v_i16m4(out + i, vr, vl);
        i += vl;
    }
}

/*
 * mf_elemwise_mul_i32 - Element-wise multiplication of two int32 vectors.
 *
 * Truncating: result is the low 32 bits of the full product.
 */
static inline void mf_elemwise_mul_i32(const int32_t *a, const int32_t *b,
                                        int32_t *out, int n) {
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e32m4(n - i);
        vint32m4_t va = __riscv_vle32_v_i32m4(a + i, vl);
        vint32m4_t vb = __riscv_vle32_v_i32m4(b + i, vl);
        vint32m4_t vr = __riscv_vmul_vv_i32m4(va, vb, vl);
        __riscv_vse32_v_i32m4(out + i, vr, vl);
        i += vl;
    }
}

/* ========================================================================
 * Scalar Broadcast Multiplication: out = a * scalar
 * ======================================================================== */

/*
 * mf_elemwise_mul_scalar_i8 - Multiply int8 vector by a scalar.
 *
 * out[i] = (int8_t)(a[i] * scalar) for i in [0, n)
 * Truncating semantics (low 8 bits).
 */
static inline void mf_elemwise_mul_scalar_i8(const int8_t *a, int8_t scalar,
                                              int8_t *out, int n) {
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e8m4(n - i);
        vint8m4_t va = __riscv_vle8_v_i8m4(a + i, vl);
        vint8m4_t vr = __riscv_vmul_vx_i8m4(va, scalar, vl);
        __riscv_vse8_v_i8m4(out + i, vr, vl);
        i += vl;
    }
}

/*
 * mf_elemwise_mul_scalar_i16 - Multiply int16 vector by a scalar.
 */
static inline void mf_elemwise_mul_scalar_i16(const int16_t *a, int16_t scalar,
                                               int16_t *out, int n) {
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e16m4(n - i);
        vint16m4_t va = __riscv_vle16_v_i16m4(a + i, vl);
        vint16m4_t vr = __riscv_vmul_vx_i16m4(va, scalar, vl);
        __riscv_vse16_v_i16m4(out + i, vr, vl);
        i += vl;
    }
}

/*
 * mf_elemwise_mul_scalar_i32 - Multiply int32 vector by a scalar.
 */
static inline void mf_elemwise_mul_scalar_i32(const int32_t *a, int32_t scalar,
                                               int32_t *out, int n) {
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e32m4(n - i);
        vint32m4_t va = __riscv_vle32_v_i32m4(a + i, vl);
        vint32m4_t vr = __riscv_vmul_vx_i32m4(va, scalar, vl);
        __riscv_vse32_v_i32m4(out + i, vr, vl);
        i += vl;
    }
}

/* ========================================================================
 * Quantization: int32 -> int8 / int16
 *
 * Performs: out = clip((in >> shift) + zero_point, lo, hi)
 *
 * This is a standard post-accumulator quantization step used after
 * matrix multiplication to convert int32 accumulator results back to
 * narrow integer types for the next layer.
 * ======================================================================== */

/*
 * mf_quantize_i32_to_i8 - Quantize int32 to int8.
 *
 * For each element:
 *   shifted = in[i] >> shift        (arithmetic right shift)
 *   biased  = shifted + zero_point
 *   out[i]  = clip(biased, -128, 127)
 *
 * Uses widening/narrowing intrinsics:
 *   1. Load int32 source
 *   2. Arithmetic shift right by 'shift'
 *   3. Add zero_point
 *   4. Clip to [-128, 127]
 *   5. Narrow int32 -> int16 -> int8 via signed narrowing shift (vnsra by 0)
 */
static inline void mf_quantize_i32_to_i8(const int32_t *in, int8_t *out,
                                           int n, int shift, int zero_point) {
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e32m4(n - i);

        /* Load int32 values */
        vint32m4_t vi = __riscv_vle32_v_i32m4(in + i, vl);

        /* Arithmetic right shift */
        vint32m4_t vs = __riscv_vsra_vx_i32m4(vi, (size_t)shift, vl);

        /* Add zero point */
        vint32m4_t vz = __riscv_vadd_vx_i32m4(vs, zero_point, vl);

        /* Clip to int8 range [-128, 127] */
        vint32m4_t vc = __riscv_vmax_vx_i32m4(vz, -128, vl);
        vc = __riscv_vmin_vx_i32m4(vc, 127, vl);

        /* Narrow int32 -> int16 (shift by 0, preserving value) */
        vint16m2_t vn16 = __riscv_vnsra_wx_i16m2(vc, 0, vl);

        /* Narrow int16 -> int8 (shift by 0, preserving value) */
        vint8m1_t vn8 = __riscv_vnsra_wx_i8m1(vn16, 0, vl);

        /* Store result */
        __riscv_vse8_v_i8m1(out + i, vn8, vl);
        i += vl;
    }
}

/*
 * mf_quantize_i32_to_i16 - Quantize int32 to int16.
 *
 * For each element:
 *   shifted = in[i] >> shift        (arithmetic right shift)
 *   biased  = shifted + zero_point
 *   out[i]  = clip(biased, -32768, 32767)
 */
static inline void mf_quantize_i32_to_i16(const int32_t *in, int16_t *out,
                                            int n, int shift, int zero_point) {
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e32m4(n - i);

        /* Load int32 values */
        vint32m4_t vi = __riscv_vle32_v_i32m4(in + i, vl);

        /* Arithmetic right shift */
        vint32m4_t vs = __riscv_vsra_vx_i32m4(vi, (size_t)shift, vl);

        /* Add zero point */
        vint32m4_t vz = __riscv_vadd_vx_i32m4(vs, zero_point, vl);

        /* Clip to int16 range [-32768, 32767] */
        vint32m4_t vc = __riscv_vmax_vx_i32m4(vz, -32768, vl);
        vc = __riscv_vmin_vx_i32m4(vc, 32767, vl);

        /* Narrow int32 -> int16 (shift by 0, preserving value) */
        vint16m2_t vn16 = __riscv_vnsra_wx_i16m2(vc, 0, vl);

        /* Store result */
        __riscv_vse16_v_i16m2(out + i, vn16, vl);
        i += vl;
    }
}

/* ========================================================================
 * Dequantization: int8 -> int32
 *
 * Performs: out = (int32_t)in * scale
 *
 * Used to widen narrow activations/weights back to int32 for accumulation.
 * ======================================================================== */

/*
 * mf_dequantize_i8_to_i32 - Dequantize int8 to int32.
 *
 * For each element:
 *   out[i] = (int32_t)in[i] * scale
 *
 * Uses sign-extension widening:
 *   int8 -> int16 (vwadd with 0) -> int32 (vwadd with 0) -> vmul by scale
 */
static inline void mf_dequantize_i8_to_i32(const int8_t *in, int32_t *out,
                                             int n, int scale) {
    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e8m1(n - i);

        /* Load int8 values */
        vint8m1_t vi8 = __riscv_vle8_v_i8m1(in + i, vl);

        /* Widen int8 -> int16 via sign-extending widening add with 0 */
        vint16m2_t vi16 = __riscv_vwadd_vx_i16m2(vi8, 0, vl);

        /* Widen int16 -> int32 via sign-extending widening add with 0 */
        vint32m4_t vi32 = __riscv_vwadd_vx_i32m4(vi16, 0, vl);

        /* Multiply by scale factor */
        vint32m4_t vr = __riscv_vmul_vx_i32m4(vi32, scale, vl);

        /* Store result */
        __riscv_vse32_v_i32m4(out + i, vr, vl);
        i += vl;
    }
}

/* ========================================================================
 * Per-Channel Bias Addition
 *
 * For a tensor with layout [channels, spatial], adds bias[c] to every
 * spatial position within channel c:
 *   data[c * spatial + s] += bias[c]   for all c in [0, channels), s in [0, spatial)
 *
 * This is the standard bias-add after a convolution or fully-connected layer
 * where the accumulator output is stored as [C, H*W].
 * ======================================================================== */

/*
 * mf_bias_add_i32 - Add per-channel bias to int32 tensor in-place.
 *
 * data:     pointer to int32 tensor of shape [channels, spatial], modified in-place
 * bias:     pointer to int32 bias vector of length 'channels'
 * channels: number of output channels (C)
 * spatial:  number of spatial positions per channel (H * W)
 */
static inline void mf_bias_add_i32(int32_t *data, const int32_t *bias,
                                    int channels, int spatial) {
    for (int c = 0; c < channels; c++) {
        int32_t b = bias[c];
        int32_t *row = data + (size_t)c * spatial;
        for (int s = 0; s < spatial; ) {
            size_t vl = __riscv_vsetvl_e32m4(spatial - s);
            vint32m4_t vd = __riscv_vle32_v_i32m4(row + s, vl);
            vint32m4_t vr = __riscv_vadd_vx_i32m4(vd, b, vl);
            __riscv_vse32_v_i32m4(row + s, vr, vl);
            s += vl;
        }
    }
}

#endif /* MF_VECTOR_OPS_H */

