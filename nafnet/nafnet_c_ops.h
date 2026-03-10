/*
 * nafnet_c_ops.h - Pure C integer-only operations for NAFNet inference
 *
 * All operations use int8/int16/int32 arithmetic only - NO floating point.
 * This ensures identical results on x86 native and RISC-V spike.
 *
 * Data layout: [C, H, W] channels-first (CHW), contiguous.
 */

#ifndef NAFNET_C_OPS_H
#define NAFNET_C_OPS_H

#include <stdint.h>

/* ============================================================
 * Utility: clamp
 * ============================================================ */
static inline int8_t clip_i8(int32_t x) {
    if (x < -128) return -128;
    if (x > 127) return 127;
    return (int8_t)x;
}

static inline int16_t clip_i16(int32_t x) {
    if (x < -32768) return -32768;
    if (x > 32767) return 32767;
    return (int16_t)x;
}

/* ============================================================
 * Integer square root (matches mf_isqrt from mf_norm.h exactly)
 * ============================================================ */
static uint32_t isqrt_u32(uint32_t x) {
    if (x == 0) return 0;
    if (x == 1) return 1;

    uint32_t r = x;
    uint32_t bit = 1u << 30;

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

/* ============================================================
 * Requantize: int32 accumulator -> int8 output
 *
 * out[i] = clip((in[i] * M + round) >> shift, -128, 127)
 * where round = (1 << (shift - 1)) for rounding
 * ============================================================ */
static void requantize_i32_to_i8(const int32_t *in, int8_t *out, int n,
                                  int32_t M, int shift) {
    int32_t round = (shift > 0) ? (1 << (shift - 1)) : 0;
    for (int i = 0; i < n; i++) {
        int64_t val = (int64_t)in[i] * M;
        int32_t result = (int32_t)((val + round) >> shift);
        out[i] = clip_i8(result);
    }
}

/* Simple shift-based requant (matches mf_quantize_i32_to_i8 with zero_point=0) */
static void shift_i32_to_i8(const int32_t *in, int8_t *out, int n, int shift) {
    for (int i = 0; i < n; i++) {
        int32_t val = in[i] >> shift;
        out[i] = clip_i8(val);
    }
}

/* ============================================================
 * Bias addition: add per-channel bias to int32 accumulator
 *
 * data layout: [C, spatial] where spatial = H * W
 * bias: [C]
 * ============================================================ */
static void bias_add_i32(int32_t *data, const int32_t *bias,
                          int channels, int spatial) {
    for (int c = 0; c < channels; c++) {
        int32_t b = bias[c];
        int32_t *row = data + c * spatial;
        for (int s = 0; s < spatial; s++) {
            row[s] += b;
        }
    }
}

/* ============================================================
 * Conv2d 1x1: pointwise convolution
 *
 * input:  [C_in, H, W]  int8
 * weight: [C_out, C_in]  int8 (stored as [C_out * C_in] row-major)
 * output: [C_out, H, W]  int32 (accumulator, before requantization)
 * ============================================================ */
static void conv2d_1x1_i8(const int8_t *input, const int8_t *weight,
                            int32_t *output,
                            int C_in, int C_out, int H, int W) {
    int spatial = H * W;
    for (int co = 0; co < C_out; co++) {
        const int8_t *w_row = weight + co * C_in;
        int32_t *out_row = output + co * spatial;
        for (int s = 0; s < spatial; s++) {
            int32_t acc = 0;
            for (int ci = 0; ci < C_in; ci++) {
                acc += (int32_t)input[ci * spatial + s] * (int32_t)w_row[ci];
            }
            out_row[s] = acc;
        }
    }
}

/* ============================================================
 * Conv2d 3x3: standard 3x3 convolution with padding=1
 *
 * input:  [C_in, H, W]  int8
 * weight: [C_out, C_in, 3, 3]  int8 (row-major)
 * output: [C_out, H, W]  int32
 * ============================================================ */
static void conv2d_3x3_i8(const int8_t *input, const int8_t *weight,
                            int32_t *output,
                            int C_in, int C_out, int H, int W) {
    int spatial = H * W;
    for (int co = 0; co < C_out; co++) {
        int32_t *out_ch = output + co * spatial;
        for (int oh = 0; oh < H; oh++) {
            for (int ow = 0; ow < W; ow++) {
                int32_t acc = 0;
                for (int ci = 0; ci < C_in; ci++) {
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            int ih = oh + kh - 1;  /* padding=1 */
                            int iw = ow + kw - 1;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int8_t iv = input[ci * spatial + ih * W + iw];
                                int8_t wv = weight[((co * C_in + ci) * 3 + kh) * 3 + kw];
                                acc += (int32_t)iv * (int32_t)wv;
                            }
                        }
                    }
                }
                out_ch[oh * W + ow] = acc;
            }
        }
    }
}

/* ============================================================
 * Conv2d 3x3 depthwise: groups=C
 *
 * input:  [C, H, W]  int8
 * weight: [C, 1, 3, 3]  int8 (stored as [C * 9])
 * output: [C, H, W]  int32
 * ============================================================ */
static void conv2d_3x3_dw_i8(const int8_t *input, const int8_t *weight,
                               int32_t *output,
                               int C, int H, int W) {
    int spatial = H * W;
    for (int c = 0; c < C; c++) {
        const int8_t *in_ch = input + c * spatial;
        const int8_t *w_ch = weight + c * 9;
        int32_t *out_ch = output + c * spatial;
        for (int oh = 0; oh < H; oh++) {
            for (int ow = 0; ow < W; ow++) {
                int32_t acc = 0;
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int ih = oh + kh - 1;
                        int iw = ow + kw - 1;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            acc += (int32_t)in_ch[ih * W + iw] * (int32_t)w_ch[kh * 3 + kw];
                        }
                    }
                }
                out_ch[oh * W + ow] = acc;
            }
        }
    }
}

/* ============================================================
 * Conv2d 2x2 stride 2: downsample convolution
 *
 * input:  [C_in, H, W]  int8
 * weight: [C_out, C_in, 2, 2]  int8
 * output: [C_out, H/2, W/2]  int32
 * ============================================================ */
static void conv2d_2x2_s2_i8(const int8_t *input, const int8_t *weight,
                               int32_t *output,
                               int C_in, int C_out, int H, int W) {
    int outH = H / 2;
    int outW = W / 2;
    int in_spatial = H * W;
    int out_spatial = outH * outW;
    for (int co = 0; co < C_out; co++) {
        int32_t *out_ch = output + co * out_spatial;
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                int32_t acc = 0;
                int ih_base = oh * 2;
                int iw_base = ow * 2;
                for (int ci = 0; ci < C_in; ci++) {
                    for (int kh = 0; kh < 2; kh++) {
                        for (int kw = 0; kw < 2; kw++) {
                            int ih = ih_base + kh;
                            int iw = iw_base + kw;
                            int8_t iv = input[ci * in_spatial + ih * W + iw];
                            int8_t wv = weight[((co * C_in + ci) * 2 + kh) * 2 + kw];
                            acc += (int32_t)iv * (int32_t)wv;
                        }
                    }
                }
                out_ch[oh * outW + ow] = acc;
            }
        }
    }
}

/* ============================================================
 * LayerNorm2d for int8
 *
 * NAFNet LayerNorm2d: normalize across CHANNELS at each spatial position.
 *   mean[h,w] = mean_c(input[c,h,w])
 *   var[h,w] = var_c(input[c,h,w])
 *   output[c,h,w] = gamma[c] * (input[c,h,w] - mean[h,w]) / sqrt(var[h,w]) + beta[c]
 *
 * input:  [C, H, W]  int8
 * output: [C, H, W]  int8
 * gamma:  [C] int32 (= round(gamma_float / out_scale * (1 << frac_bits)))
 * beta:   [C] int8  (= round(beta_float / out_scale))
 * ============================================================ */
static void layernorm2d_i8(const int8_t *input, int8_t *output,
                            const int32_t *gamma, const int8_t *beta,
                            int C, int H, int W, int frac_bits) {
    int spatial = H * W;
    const int INV_STD_FRAC = 10;
    int total_shift = INV_STD_FRAC + frac_bits;
    int64_t round_val = (total_shift > 0) ? ((int64_t)1 << (total_shift - 1)) : 0;

    /* For each spatial position, compute mean and variance across channels */
    for (int s = 0; s < spatial; s++) {
        /* Step 1: Compute mean across C channels */
        int32_t total_sum = 0;
        for (int c = 0; c < C; c++) {
            total_sum += (int32_t)input[c * spatial + s];
        }
        int32_t mean = total_sum / C;

        /* Step 2: Compute variance across C channels */
        int64_t total_sq = 0;
        for (int c = 0; c < C; c++) {
            int32_t diff = (int32_t)input[c * spatial + s] - mean;
            total_sq += (int64_t)diff * diff;
        }
        int32_t var = (int32_t)(total_sq / C);
        if (var < 1) var = 1;

        /* Step 3: 1/sqrt(var) */
        uint32_t std_dev = isqrt_u32((uint32_t)var);
        if (std_dev == 0) std_dev = 1;
        int32_t inv_std_fp = (1 << INV_STD_FRAC) / (int32_t)std_dev;

        /* Step 4: Normalize each channel at this spatial position */
        for (int c = 0; c < C; c++) {
            int32_t diff = (int32_t)input[c * spatial + s] - mean;
            int32_t norm = diff * inv_std_fp;  /* Q INV_STD_FRAC */
            int64_t val = (int64_t)norm * (int64_t)gamma[c];
            int32_t result = (int32_t)((val + round_val) >> total_shift) + (int32_t)beta[c];
            output[c * spatial + s] = clip_i8(result);
        }
    }
}

/* ============================================================
 * SimpleGate: split channels in half, element-wise multiply
 *
 * input:  [2*C, H, W]  int8
 * output: [C, H, W]  int8
 *
 * The product of two int8 values (max 127*127=16129) fits in int16.
 * We apply requantization to get back to int8.
 * ============================================================ */
static void simplegate_i8(const int8_t *input, int8_t *output,
                           int C, int H, int W,
                           int32_t M, int shift) {
    int spatial = H * W;
    int half_size = C * spatial;  /* C channels * spatial */
    int32_t round = (shift > 0) ? (1 << (shift - 1)) : 0;

    for (int i = 0; i < half_size; i++) {
        int32_t a = (int32_t)input[i];
        int32_t b = (int32_t)input[half_size + i];
        int64_t prod = (int64_t)(a * b) * M;
        int32_t result = (int32_t)((prod + round) >> shift);
        output[i] = clip_i8(result);
    }
}

/* ============================================================
 * SCA (Simplified Channel Attention)
 *
 * input:  [C, H, W]  int8
 * output: [C, H, W]  int8
 *
 * Steps:
 * 1. Global average pool: pool[c] = sum(input[c,:,:]) / spatial
 * 2. 1x1 conv: attn[c] = sum_j(weight[c,j] * pool[j]) + bias[c]
 * 3. Requantize attn to int8
 * 4. Broadcast multiply: output[c,h,w] = input[c,h,w] * attn[c]
 * 5. Requantize result to int8
 * ============================================================ */
static void sca_i8(const int8_t *input, int8_t *output,
                    const int8_t *conv_weight, const int32_t *conv_bias,
                    int C, int H, int W,
                    int32_t conv_M, int conv_shift,
                    int32_t mul_M, int mul_shift) {
    int spatial = H * W;

    /* Step 1: Global average pool */
    int8_t pool[1024];  /* max channels */
    for (int c = 0; c < C; c++) {
        const int8_t *in_ch = input + c * spatial;
        int32_t sum = 0;
        for (int s = 0; s < spatial; s++) {
            sum += (int32_t)in_ch[s];
        }
        pool[c] = clip_i8(sum / spatial);
    }

    /* Step 2: 1x1 conv on pooled values */
    int8_t attn[1024];
    int32_t conv_round = (conv_shift > 0) ? (1 << (conv_shift - 1)) : 0;
    for (int co = 0; co < C; co++) {
        int32_t acc = 0;
        for (int ci = 0; ci < C; ci++) {
            acc += (int32_t)conv_weight[co * C + ci] * (int32_t)pool[ci];
        }
        acc += conv_bias[co];
        /* Requantize conv output to int8 */
        int64_t val = (int64_t)acc * conv_M;
        attn[co] = clip_i8((int32_t)((val + conv_round) >> conv_shift));
    }

    /* Step 3: Broadcast multiply + requantize */
    int32_t mul_round = (mul_shift > 0) ? (1 << (mul_shift - 1)) : 0;
    for (int c = 0; c < C; c++) {
        const int8_t *in_ch = input + c * spatial;
        int8_t *out_ch = output + c * spatial;
        int32_t a = (int32_t)attn[c];
        for (int s = 0; s < spatial; s++) {
            int32_t prod = (int32_t)in_ch[s] * a;
            int64_t val = (int64_t)prod * mul_M;
            out_ch[s] = clip_i8((int32_t)((val + mul_round) >> mul_shift));
        }
    }
}

/* ============================================================
 * PixelShuffle: sub-pixel rearrangement (r=2)
 *
 * input:  [C*r*r, H, W]  int8
 * output: [C, H*r, W*r]  int8
 *
 * Maps input[c*r*r + (sh*r + sw), h, w] -> output[c, h*r+sh, w*r+sw]
 * ============================================================ */
static void pixelshuffle_i8(const int8_t *input, int8_t *output,
                              int C, int H, int W, int r) {
    int outH = H * r;
    int outW = W * r;
    int in_spatial = H * W;
    int out_spatial = outH * outW;

    for (int c = 0; c < C; c++) {
        for (int sh = 0; sh < r; sh++) {
            for (int sw = 0; sw < r; sw++) {
                int in_c = c * r * r + sh * r + sw;
                const int8_t *in_ch = input + in_c * in_spatial;
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        int oh = h * r + sh;
                        int ow = w * r + sw;
                        output[c * out_spatial + oh * outW + ow] = in_ch[h * W + w];
                    }
                }
            }
        }
    }
}

/* ============================================================
 * Element-wise addition of two int8 tensors (with clipping)
 * output[i] = clip(a[i] + b[i], -128, 127)
 * ============================================================ */
static void elemwise_add_i8(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = clip_i8((int32_t)a[i] + (int32_t)b[i]);
    }
}

/* ============================================================
 * Residual with scaling: output = x + y * beta
 *
 * x, y: [C, H, W] int8
 * beta: [C] int16 (fixed-point with FRAC_BITS)
 * output: [C, H, W] int8
 *
 * For each channel c:
 *   output[c,h,w] = clip(x[c,h,w] + (y[c,h,w] * beta[c]) >> FRAC_BITS)
 * ============================================================ */
static void residual_scale_add_i8(const int8_t *x, const int8_t *y,
                                    const int16_t *beta_scale,
                                    int8_t *output,
                                    int C, int H, int W, int frac_bits) {
    int spatial = H * W;
    for (int c = 0; c < C; c++) {
        int16_t bs = beta_scale[c];
        for (int s = 0; s < spatial; s++) {
            int idx = c * spatial + s;
            int32_t val = (int32_t)x[idx] + (((int32_t)y[idx] * (int32_t)bs) >> frac_bits);
            output[idx] = clip_i8(val);
        }
    }
}

/* ============================================================
 * Conv + bias + requant combo (convenience functions)
 * These combine conv -> bias_add -> requantize in one call
 * ============================================================ */

/* 1x1 conv with bias and requant to int8 */
static void conv1x1_bias_requant(const int8_t *input, const int8_t *weight,
                                   const int32_t *bias,
                                   int8_t *output, int32_t *acc_buf,
                                   int C_in, int C_out, int H, int W,
                                   int32_t M, int shift) {
    conv2d_1x1_i8(input, weight, acc_buf, C_in, C_out, H, W);
    bias_add_i32(acc_buf, bias, C_out, H * W);
    requantize_i32_to_i8(acc_buf, output, C_out * H * W, M, shift);
}

/* 3x3 conv with bias and requant to int8 */
static void conv3x3_bias_requant(const int8_t *input, const int8_t *weight,
                                   const int32_t *bias,
                                   int8_t *output, int32_t *acc_buf,
                                   int C_in, int C_out, int H, int W,
                                   int32_t M, int shift) {
    conv2d_3x3_i8(input, weight, acc_buf, C_in, C_out, H, W);
    bias_add_i32(acc_buf, bias, C_out, H * W);
    requantize_i32_to_i8(acc_buf, output, C_out * H * W, M, shift);
}

/* 3x3 depthwise conv with bias and requant to int8 */
static void dw_conv3x3_bias_requant(const int8_t *input, const int8_t *weight,
                                      const int32_t *bias,
                                      int8_t *output, int32_t *acc_buf,
                                      int C, int H, int W,
                                      int32_t M, int shift) {
    conv2d_3x3_dw_i8(input, weight, acc_buf, C, H, W);
    bias_add_i32(acc_buf, bias, C, H * W);
    requantize_i32_to_i8(acc_buf, output, C * H * W, M, shift);
}

/* 2x2 stride-2 conv with bias and requant to int8 */
static void conv2x2s2_bias_requant(const int8_t *input, const int8_t *weight,
                                     const int32_t *bias,
                                     int8_t *output, int32_t *acc_buf,
                                     int C_in, int C_out, int H, int W,
                                     int32_t M, int shift) {
    int outH = H / 2, outW = W / 2;
    conv2d_2x2_s2_i8(input, weight, acc_buf, C_in, C_out, H, W);
    bias_add_i32(acc_buf, bias, C_out, outH * outW);
    requantize_i32_to_i8(acc_buf, output, C_out * outH * outW, M, shift);
}

/* 1x1 conv without bias, requant to int8 */
static void conv1x1_requant(const int8_t *input, const int8_t *weight,
                              int8_t *output, int32_t *acc_buf,
                              int C_in, int C_out, int H, int W,
                              int32_t M, int shift) {
    conv2d_1x1_i8(input, weight, acc_buf, C_in, C_out, H, W);
    requantize_i32_to_i8(acc_buf, output, C_out * H * W, M, shift);
}

/* ============================================================
 * PSNR computation in integer domain (for output comparison)
 *
 * Computes PSNR between two uint8 images [0, 255].
 * Returns PSNR * 100 as an integer (e.g., 3456 means 34.56 dB).
 * ============================================================ */
static int32_t compute_psnr_int(const uint8_t *img1, const uint8_t *img2, int n) {
    int64_t mse_sum = 0;
    for (int i = 0; i < n; i++) {
        int32_t diff = (int32_t)img1[i] - (int32_t)img2[i];
        mse_sum += diff * diff;
    }
    if (mse_sum == 0) return 9999;  /* "infinite" PSNR */

    /* PSNR = 10 * log10(255^2 * n / mse_sum)
     * = 10 * log10(65025 * n / mse_sum)
     * We use a lookup-table-free integer log10 approximation:
     * log10(x) ≈ (log2(x) * 3) / 10, rough but sufficient for reporting
     *
     * More precisely, we'll compute it with enough precision:
     * PSNR * 100 = 1000 * log10(65025 * n / mse_sum)
     */

    /* Use integer arithmetic to compute approximate PSNR * 100 */
    int64_t peak_sq_n = (int64_t)65025 * n;
    /* Find the ratio * 1000000 for precision */
    int64_t ratio_x1000 = (peak_sq_n * 1000) / mse_sum;

    /* Now compute 10 * log10(ratio_x1000 / 1000) * 100
     * = 1000 * log10(ratio_x1000 / 1000)
     * = 1000 * (log10(ratio_x1000) - 3)
     *
     * Simple integer log10 * 100 approximation using binary search
     */
    int32_t psnr_x100 = 0;

    /* Count digits and interpolate */
    if (ratio_x1000 <= 0) return 0;

    /* log10(ratio_x1000) ~= number_of_digits - 1 + fraction
     * Simpler: we know PSNR is typically 20-50 dB for images
     * Just use a fixed-point log computation
     */
    /* log2 approximation */
    int64_t val = ratio_x1000;
    int log2_int = 0;
    while (val > 1) { val >>= 1; log2_int++; }

    /* log10 = log2 / 3.3219, approximated as log2 * 3010 / 10000 */
    int32_t log10_x10000 = (int32_t)((int64_t)log2_int * 3010);

    /* PSNR*100 = 1000*(log10(ratio_x1000) - 3)
     * = 1000 * (log10_x10000/10000 - 3)
     * = (log10_x10000 - 30000) / 10
     */
    psnr_x100 = (log10_x10000 - 30000) / 10;

    return psnr_x100;
}

#endif /* NAFNET_C_OPS_H */
