/*
 * nafnet_c_ops.h - Pure C integer-only operations for NAFNet inference
 *
 * All operations use int8/int16/int32 arithmetic only - NO floating point.
 * This ensures identical results on x86 native and RISC-V spike.
 *
 * Data layout: [C, H, W] channels-first (CHW), contiguous.
 */

#ifndef NAFNET_C_OPS_PARALLEL_H
#define NAFNET_C_OPS_PARALLEL_H

#define NUM_THREADS 8 /* for conv tiling parallelism */
#define BLOCK_H 4
#define BLOCK_W 4
#include <stdint.h>
#include "conv_tiling/src/conv_worker.h"
// #include "conv_tiling/src/Conv_para_dynamic.h"
#include "conv_tiling/src/conv_worker_cw.h"

/* ============================================================
 * Conv2d 1x1: pointwise convolution
 *
 * input:  [C_in, H, W]  int8
 * weight: [C_out, C_in]  int8 (stored as [C_out * C_in] row-major)
 * output: [C_out, H, W]  int32 (accumulator, before requantization)
 * ============================================================ */
static void conv2d_1x1_i8_para(const int8_t *input, const int8_t *weight,
                               int32_t *output,
                               int C_in, int C_out, int H, int W, void *pool)
{
    // conv2d_tiled_mt_int8(input, weight, output, NULL,
    // 1, C_in, H, W,
    // C_out, 1, 1,
    // 1, 1, 0, 0, 1, 1,
    // H, W, BLOCK_H, BLOCK_W, NUM_THREADS);

    conv2d_tiled_cw_int8(input, weight, output, NULL,
                         1, C_in, H, W,
                         C_out, 1, 1,
                         1, 1, 0, 0, 1, 1,
                         H, W, pool);

    // conv2d_tiled_dynamic_int8(input, weight, output, NULL,
    // 1, C_in, H, W,
    // C_out, 1, 1,
    // 1, 1, 0, 0, 1, 1,
    // H, W, BLOCK_H, BLOCK_W, NUM_THREADS);
}

/* ============================================================
 * Conv2d 3x3: standard 3x3 convolution with padding=1
 *
 * input:  [C_in, H, W]  int8
 * weight: [C_out, C_in, 3, 3]  int8 (row-major)
 * output: [C_out, H, W]  int32
 * ============================================================ */
static void conv2d_3x3_i8_para(const int8_t *input, const int8_t *weight,
                               int32_t *output,
                               int C_in, int C_out, int H, int W, void *pool)
{

    conv2d_tiled_int8(input, weight, output, NULL,
                      1, C_in, H, W,
                      C_out, 3, 3,
                      1, 1, 1, 1, 1, 1,
                      H, W, BLOCK_H, BLOCK_W, pool);

    // conv2d_tiled_cw_int8(input, weight, output, NULL,
    //     N, C_in, H, W,
    //     C_out, 3, 3,
    //     1, 1, 1, 1, 1, 1,
    //     H, W, pool);

    // conv2d_tiled_dynamic_int8(input, weight, output, NULL,
    // 1, C_in, H, W,
    // C_out, 3, 3,
    // 1, 1, 1, 1, 1, 1,
    // H, W, BLOCK_H, BLOCK_W, NUM_THREADS);

    // conv2d_tiled_dynamic_int8(input, weight, output, NULL,
    // 1, C_in, H, W,
    // C_out, 3, 3,
    // 1, 1, 1, 1, 1, 1,
    // H, W, BLOCK_H, BLOCK_W, NUM_THREADS);
}

/* ============================================================
 * Conv2d 2x2 stride 2: downsample convolution
 *
 * input:  [C_in, H, W]  int8
 * weight: [C_out, C_in, 2, 2]  int8
 * output: [C_out, H/2, W/2]  int32
 * ============================================================ */
static void conv2d_2x2_s2_i8_para(const int8_t *input, const int8_t *weight,
                                  int32_t *output,
                                  int C_in, int C_out, int H, int W, void *pool)
{

    // conv2d_tiled_int8(input, weight, output, NULL,
    // 1, C_in, H, W,
    // C_out, 2, 2,
    // 2, 2, 0, 0, 1, 1,
    // H/2, W/2, BLOCK_H, BLOCK_W, pool);

    conv2d_tiled_cw_int8(input, weight, output, NULL,
                         1, C_in, H, W,
                         C_out, 2, 2,
                         2, 2, 0, 0, 1, 1,
                         H / 2, W / 2, pool);

    // conv2d_tiled_dynamic_int8(input, weight, output, NULL,
    // 1, C_in, H, W,
    // C_out, 2, 2,
    // 2, 2, 0, 0, 1, 1,
    // H/2, W/2, BLOCK_H, BLOCK_W, NUM_THREADS);
}

/* ============================================================
 * Conv2d 3x3 depthwise: groups=C
 *
 * input:  [C, H, W]  int8
 * weight: [C, 1, 3, 3]  int8 (stored as [C * 9])
 * output: [C, H, W]  int32
 * ============================================================ */
static void conv2d_3x3_dw_i8_para(const int8_t *input, const int8_t *weight,
                                  int32_t *output,
                                  int C, int H, int W)
{
    // int spatial = H * W;
    // for (int c = 0; c < C; c++) {
    //     const int8_t *in_ch = input + c * spatial;
    //     const int8_t *w_ch = weight + c * 9;
    //     int32_t *out_ch = output + c * spatial;

    //     conv2d_tiled_dynamic_int8(in_ch, w_ch, out_ch, NULL,
    //     1, 1, H, W,
    //     1, 3, 3,
    //     1, 1, 1, 1, 1, 1,
    //     H, W, BLOCK_H, BLOCK_W, NUM_THREADS);
    // }
}

/* ============================================================
 * Conv + bias + requant combo (convenience functions)
 * These combine conv -> bias_add -> requantize in one call
 * ============================================================ */

/* 1x1 conv with bias and requant to int8 */
static void conv1x1_bias_requant_para(const int8_t *input, const int8_t *weight,
                                      const int32_t *bias,
                                      int8_t *output, int32_t *acc_buf,
                                      int C_in, int C_out, int H, int W,
                                      int32_t M, int shift, void *pool)
{
    conv2d_1x1_i8_para(input, weight, acc_buf, C_in, C_out, H, W, pool);
    bias_add_i32(acc_buf, bias, C_out, H * W);
    requantize_i32_to_i8(acc_buf, output, C_out * H * W, M, shift);
}

/* 3x3 conv with bias and requant to int8 */
static void conv3x3_bias_requant_para(const int8_t *input, const int8_t *weight,
                                      const int32_t *bias,
                                      int8_t *output, int32_t *acc_buf,
                                      int C_in, int C_out, int H, int W,
                                      int32_t M, int shift, void *pool)
{
    conv2d_3x3_i8_para(input, weight, acc_buf, C_in, C_out, H, W, pool);
    bias_add_i32(acc_buf, bias, C_out, H * W);
    requantize_i32_to_i8(acc_buf, output, C_out * H * W, M, shift);
}

/* 3x3 depthwise conv with bias and requant to int8 */
static void dw_conv3x3_bias_requant_para(const int8_t *input, const int8_t *weight,
                                         const int32_t *bias,
                                         int8_t *output, int32_t *acc_buf,
                                         int C, int H, int W,
                                         int32_t M, int shift)
{
    conv2d_3x3_dw_i8_para(input, weight, acc_buf, C, H, W);
    bias_add_i32(acc_buf, bias, C, H * W);
    requantize_i32_to_i8(acc_buf, output, C * H * W, M, shift);
}

static void conv2x2s2_bias_requant_para(const int8_t *input, const int8_t *weight,
                                        const int32_t *bias,
                                        int8_t *output, int32_t *acc_buf,
                                        int C_in, int C_out, int H, int W,
                                        int32_t M, int shift, void *pool)
{
    int outH = H / 2, outW = W / 2;
    conv2d_2x2_s2_i8_para(input, weight, acc_buf, C_in, C_out, H, W, pool);
    bias_add_i32(acc_buf, bias, C_out, outH * outW);
    requantize_i32_to_i8(acc_buf, output, C_out * outH * outW, M, shift);
}

static void conv1x1_requant_para(const int8_t *input, const int8_t *weight,
                                 int8_t *output, int32_t *acc_buf,
                                 int C_in, int C_out, int H, int W,
                                 int32_t M, int shift, void *pool)
{
    conv2d_1x1_i8_para(input, weight, acc_buf, C_in, C_out, H, W, pool);
    requantize_i32_to_i8(acc_buf, output, C_out * H * W, M, shift);
}

/* 3x3 depthwise conv with bias and requant to int8 */

#endif /* NAFNET_C_OPS_PARALLEL_H */
