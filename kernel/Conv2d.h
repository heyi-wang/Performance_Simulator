/*
 * mf_conv2d.h - MatrixFlow Conv2D Kernels (Partial Materialization + TMMA)
 *
 * All spatial convolutions use partial materialization: instead of
 * materializing the full [K x N] im2col matrix, B-tiles are constructed
 * on-the-fly from a pre-padded (stride-1) or raw (stride-2) input.
 *
 * Supported convolutions:
 *   - 3x3 convolution with configurable padding (stride-1)
 *   - 1x1 pointwise convolution (direct GeMM, no im2col)
 *   - 2x2 stride-2 convolution for downsampling (NAF-Net)
 *   - Transposed 2x2 stride-2 convolution for upsampling (U-Net)
 *
 * Data types: int4, int8, int16 (signed)
 *
 * All inputs and outputs reside in L1. Caller provides workspace.
 * Matrix layouts:
 *   input:  [C_in, H, W] row-major
 *   weight: row-major for GeMM (see per-function docs)
 *   output: [C_out, outH, outW] row-major
 *
 * Hardware config: v1-4-16-32-64 (M=16, K=32, N=64, 4 accumulators)
 */

#ifndef MF_CONV2D_H
#define MF_CONV2D_H

#include "mf_kernel.h"
#include "mf_gemm.h"
#include "mf_transform.h"

/* ========================================================================
 * Internal helpers
 * ======================================================================== */

static inline mf_l1_buf_t mf_l1_buf_from_alloc(const mf_l1_alloc_t *a) {
    mf_l1_buf_t buf;
    buf.ptr  = a->base + a->offset;
    buf.size = a->capacity - a->offset;
    return buf;
}

/*
 * mf_btile_fill_s1_i8 - Construct one B-tile row for stride-1 conv (int8)
 *
 * Copies actual_n elements from pre-padded input into b_row, handling
 * row wrapping when the N-tile spans multiple output rows.
 * Uses DMA for contiguous row segments.
 */
static inline void mf_btile_fill_s1_i8(int8_t *b_row,
                                         const int8_t *padded,
                                         int ci, int kh, int kw,
                                         int padH, int padW,
                                         int n_start, int actual_n,
                                         int outW) {
    int remaining = actual_n;
    int b_col = 0;
    int cur_oh = n_start / outW;
    int cur_ow = n_start % outW;

    while (remaining > 0) {
        int run = MF_MIN(remaining, outW - cur_ow);
        __builtin_riscv_mf_dma_x(
            b_row + b_col,
            (void *)(padded + ci * padH * padW
                     + (cur_oh + kh) * padW
                     + (cur_ow + kw)),
            (unsigned long)run);
        b_col += run;
        remaining -= run;
        cur_oh++;
        cur_ow = 0;
    }
}

/*
 * mf_btile_fill_s2_i8 - Construct one B-tile row for stride-2 conv (int8)
 *
 * Copies actual_n elements from raw input with stride-2 addressing.
 * Uses scalar copy (stride-2 prevents contiguous DMA).
 */
static inline void mf_btile_fill_s2_i8(int8_t *b_row,
                                         const int8_t *input,
                                         int ci, int kh, int kw,
                                         int H, int W,
                                         int n_start, int actual_n,
                                         int outW) {
    int remaining = actual_n;
    int b_col = 0;
    int cur_oh = n_start / outW;
    int cur_ow = n_start % outW;

    while (remaining > 0) {
        int run = MF_MIN(remaining, outW - cur_ow);
        const int8_t *src = input + ci * H * W
            + (cur_oh * 2 + kh) * W + (cur_ow * 2 + kw);
        for (int j = 0; j < run; j++)
            b_row[b_col + j] = src[j * 2];
        b_col += run;
        remaining -= run;
        cur_oh++;
        cur_ow = 0;
    }
}

/*
 * mf_atile_load - Load A-tile from weight with arbitrary stride
 *
 * Loads [ml x tile_k_bytes] from weight at w_stride byte stride per row.
 * Handles partial K (load_k < tile_k_bytes) and partial M (ml < tile_m)
 * by pre-zeroing and using per-row DMA.
 */
static inline void mf_atile_load(uint8_t *a_pad,
                                   const uint8_t *w_src,
                                   int ml, int tile_m,
                                   int load_k_bytes, int tile_k_bytes,
                                   int w_stride_bytes) {
    if (ml < tile_m || load_k_bytes < tile_k_bytes)
        mf_memset(a_pad, 0, tile_m * tile_k_bytes);

    if (load_k_bytes == tile_k_bytes) {
        /* Full K-tile: efficient 2D DMA */
        mf_dma_load_tile_2d(a_pad, w_src,
                            (size_t)tile_k_bytes,
                            (size_t)ml,
                            (size_t)w_stride_bytes);
        mf_dma_sync();
    } else {
        /* Partial K-tile: per-row DMA to maintain tile_k stride in dst */
        for (int r = 0; r < ml; r++)
            __builtin_riscv_mf_dma_x(
                a_pad + r * tile_k_bytes,
                (void *)(w_src + r * w_stride_bytes),
                (unsigned long)load_k_bytes);
        __builtin_riscv_mf_dma_sync();
    }
}

/* ========================================================================
 * INT8 Conv2D Kernels
 * ======================================================================== */

/*
 * mf_conv2d_3x3_i8 - 3x3 Convolution (int8), partial materialization
 *
 * Pre-pads input, constructs B-tiles on-the-fly via DMA from padded input,
 * transposes and feeds directly to TMMA. Handles any spatial dimensions
 * (no alignment requirement on outW).
 *
 * Parameters:
 *   input     - [C_in, H, W] row-major, int8
 *   weight    - [C_out, K_padded] row-major (pre-padded to tile_k multiple)
 *   output    - [C_out, outH, outW] row-major, int32
 *   C_in, C_out, H, W, pad - conv parameters
 *   workspace - L1 scratch memory
 */
static inline int mf_conv2d_3x3_i8(const int8_t *input,
                                     const int8_t *weight,
                                     int32_t *output,
                                     int C_in, int C_out,
                                     int H, int W, int pad,
                                     mf_l1_buf_t workspace) {
    const int kH = 3, kW = 3;
    const int outH = H + 2 * pad - kH + 1;
    const int outW = W + 2 * pad - kW + 1;
    const int K = C_in * kH * kW;
    const int K_padded = MF_DIV_CEIL(K, MF_TILE_K) * MF_TILE_K;
    const int N = outH * outW;

    if (outH <= 0 || outW <= 0)
        return MF_ERR_INVALID_PARAM;

    const int tile_m = MF_TILE_M;
    const int tile_n = MF_TILE_N;
    const int tile_k = MF_TILE_K;
    const int c_elem = 4;          /* int32 accumulator */
    const int padH = H + 2 * pad;
    const int padW = W + 2 * pad;

    mf_l1_alloc_t alloc;
    mf_l1_alloc_init(&alloc, workspace);

    /* Pre-padded input: [C_in x padH x padW] */
    size_t pad_bytes = (size_t)C_in * padH * padW;
    int8_t *padded = (int8_t *)mf_l1_alloc(&alloc, pad_bytes);
    if (!padded) return MF_ERR_L1_TOO_SMALL;

    /* Tile working buffers */
    uint8_t *a_pad = (uint8_t *)mf_l1_alloc(&alloc, MF_TILE_A_BYTES);
    uint8_t *b_tmp = (uint8_t *)mf_l1_alloc(&alloc, tile_k * tile_n);
    uint8_t *b_pad = (uint8_t *)mf_l1_alloc(&alloc, MF_TILE_B_BYTES);
    uint8_t *c_tmp = (uint8_t *)mf_l1_alloc(&alloc, MF_ACC_OUT_I8);
    if (!a_pad || !b_tmp || !b_pad || !c_tmp)
        return MF_ERR_L1_TOO_SMALL;

    /* Pre-pad input: zero then DMA each row into interior */
    mf_memset(padded, 0, pad_bytes);
    for (int c = 0; c < C_in; c++)
        for (int h = 0; h < H; h++)
            __builtin_riscv_mf_dma_x(
                padded + c * padH * padW + (h + pad) * padW + pad,
                (void *)(input + c * H * W + h * W),
                (unsigned long)W);
    __builtin_riscv_mf_dma_sync();

    /* Tile counts */
    int m_tiles = MF_DIV_CEIL(C_out, tile_m);
    int n_tiles = MF_DIV_CEIL(N, tile_n);
    int k_tiles = K_padded / tile_k;

    const uint8_t *w_base = (const uint8_t *)weight;
    uint8_t *c_base = (uint8_t *)output;

    /* Tiled fused conv: m-outer (multi-acc), n, k-inner */
    for (int mg = 0; mg < m_tiles; mg += MF_NUM_ACC) {
        int m_batch = MF_MIN(MF_NUM_ACC, m_tiles - mg);

        for (int nt = 0; nt < n_tiles; nt++) {
            int n_start = nt * tile_n;
            int actual_n = MF_MIN(tile_n, N - n_start);

            for (int a = 0; a < m_batch; a++)
                mf_tzero(a);

            for (int kt = 0; kt < k_tiles; kt++) {
                int k_start = kt * tile_k;
                int k_valid = MF_MIN(tile_k, K - k_start);

                /* Construct B-tile [tile_k x tile_n] */
                if (k_valid < tile_k || actual_n < tile_n)
                    mf_memset(b_tmp, 0, tile_k * tile_n);

                for (int ki = 0; ki < k_valid; ki++) {
                    int k  = k_start + ki;
                    int ci = k / (kH * kW);
                    int kh = (k % (kH * kW)) / kW;
                    int kw = k % kW;
                    mf_btile_fill_s1_i8((int8_t *)b_tmp + ki * tile_n,
                                        padded, ci, kh, kw,
                                        padH, padW,
                                        n_start, actual_n, outW);
                }
                __builtin_riscv_mf_dma_sync();

                /* Transpose [tile_k x tile_n] -> [tile_n x tile_k] */
                mf_dma_transpose_8(b_pad, b_tmp,
                                   (size_t)tile_n,
                                   (size_t)tile_n,
                                   (size_t)tile_k);

                /* A-tile load + TMMA per accumulator */
                for (int a = 0; a < m_batch; a++) {
                    int ms = (mg + a) * tile_m;
                    int ml = MF_MIN(tile_m, C_out - ms);
                    mf_atile_load(a_pad,
                                  w_base + ms * K_padded + k_start,
                                  ml, tile_m, tile_k, tile_k, K_padded);
                    mf_tmma_ii8(a, a_pad, b_pad);
                }
            }

            /* Store accumulators to output */
            for (int a = 0; a < m_batch; a++) {
                int ms = (mg + a) * tile_m;
                int ml = MF_MIN(tile_m, C_out - ms);
                uint8_t *c_dst = c_base + (ms * N + n_start) * c_elem;

                mf_tstore32(a, c_tmp,
                            (size_t)(tile_m * tile_n * c_elem));
                for (int r = 0; r < ml; r++)
                    __builtin_riscv_mf_dma_x(
                        c_dst + r * N * c_elem,
                        (void *)(c_tmp + r * tile_n * c_elem),
                        (unsigned long)(actual_n * c_elem));
                __builtin_riscv_mf_dma_sync();
            }
        }
    }
    return MF_OK;
}

/*
 * mf_conv2d_1x1_i8 - 1x1 Pointwise Convolution (int8)
 *
 * Direct GeMM, no im2col needed.
 */
static inline int mf_conv2d_1x1_i8(const int8_t *input,
                                     const int8_t *weight,
                                     int32_t *output,
                                     int C_in, int C_out,
                                     int H, int W,
                                     mf_l1_buf_t workspace) {
    const int N = H * W;
    if (C_in <= 0 || C_out <= 0 || N <= 0)
        return MF_ERR_INVALID_PARAM;
    return mf_gemm_ii8(C_out, N, C_in, weight, input, output, workspace);
}

/*
 * mf_conv2d_2x2s2_i8 - 2x2 Stride-2 Downsampling Convolution (int8)
 *
 * Partial materialization with stride-2 B-tile construction from raw input.
 * No pre-padding needed (pad=0 for downsampling).
 *
 * Parameters:
 *   input     - [C_in, H, W] row-major, int8
 *   weight    - [C_out x K] row-major, K = C_in * 4
 *   output    - [C_out, H/2, W/2] row-major, int32
 */
static inline int mf_conv2d_2x2s2_i8(const int8_t *input,
                                       const int8_t *weight,
                                       int32_t *output,
                                       int C_in, int C_out,
                                       int H, int W,
                                       mf_l1_buf_t workspace) {
    const int kH = 2, kW = 2;
    const int outH = H / 2;
    const int outW = W / 2;
    const int K = C_in * kH * kW;
    const int K_padded = MF_DIV_CEIL(K, MF_TILE_K) * MF_TILE_K;
    const int N = outH * outW;

    if (outH <= 0 || outW <= 0 || (H % 2 != 0) || (W % 2 != 0))
        return MF_ERR_INVALID_PARAM;

    const int tile_m = MF_TILE_M;
    const int tile_n = MF_TILE_N;
    const int tile_k = MF_TILE_K;
    const int c_elem = 4;

    mf_l1_alloc_t alloc;
    mf_l1_alloc_init(&alloc, workspace);

    /* Tile working buffers only — no full im2col buffer */
    uint8_t *a_pad = (uint8_t *)mf_l1_alloc(&alloc, MF_TILE_A_BYTES);
    uint8_t *b_tmp = (uint8_t *)mf_l1_alloc(&alloc, tile_k * tile_n);
    uint8_t *b_pad = (uint8_t *)mf_l1_alloc(&alloc, MF_TILE_B_BYTES);
    uint8_t *c_tmp = (uint8_t *)mf_l1_alloc(&alloc, MF_ACC_OUT_I8);
    if (!a_pad || !b_tmp || !b_pad || !c_tmp)
        return MF_ERR_L1_TOO_SMALL;

    int m_tiles = MF_DIV_CEIL(C_out, tile_m);
    int n_tiles = MF_DIV_CEIL(N, tile_n);
    int k_tiles = K_padded / tile_k;

    const uint8_t *w_base = (const uint8_t *)weight;
    uint8_t *c_base = (uint8_t *)output;

    for (int mg = 0; mg < m_tiles; mg += MF_NUM_ACC) {
        int m_batch = MF_MIN(MF_NUM_ACC, m_tiles - mg);

        for (int nt = 0; nt < n_tiles; nt++) {
            int n_start = nt * tile_n;
            int actual_n = MF_MIN(tile_n, N - n_start);

            for (int a = 0; a < m_batch; a++)
                mf_tzero(a);

            for (int kt = 0; kt < k_tiles; kt++) {
                int k_start = kt * tile_k;
                int k_valid = MF_MIN(tile_k, K - k_start);

                /* Construct B-tile with stride-2 access */
                if (k_valid < tile_k || actual_n < tile_n)
                    mf_memset(b_tmp, 0, tile_k * tile_n);

                for (int ki = 0; ki < k_valid; ki++) {
                    int k  = k_start + ki;
                    int ci = k / (kH * kW);
                    int kh = (k % (kH * kW)) / kW;
                    int kw = k % kW;
                    mf_btile_fill_s2_i8((int8_t *)b_tmp + ki * tile_n,
                                        input, ci, kh, kw,
                                        H, W,
                                        n_start, actual_n, outW);
                }

                mf_dma_transpose_8(b_pad, b_tmp,
                                   (size_t)tile_n,
                                   (size_t)tile_n,
                                   (size_t)tile_k);

                for (int a = 0; a < m_batch; a++) {
                    int ms = (mg + a) * tile_m;
                    int ml = MF_MIN(tile_m, C_out - ms);
                    int load_k = MF_MIN(tile_k, K - k_start);
                    mf_atile_load(a_pad,
                                  w_base + ms * K + k_start,
                                  ml, tile_m, load_k, tile_k, K);
                    mf_tmma_ii8(a, a_pad, b_pad);
                }
            }

            for (int a = 0; a < m_batch; a++) {
                int ms = (mg + a) * tile_m;
                int ml = MF_MIN(tile_m, C_out - ms);
                uint8_t *c_dst = c_base + (ms * N + n_start) * c_elem;

                mf_tstore32(a, c_tmp,
                            (size_t)(tile_m * tile_n * c_elem));
                for (int r = 0; r < ml; r++)
                    __builtin_riscv_mf_dma_x(
                        c_dst + r * N * c_elem,
                        (void *)(c_tmp + r * tile_n * c_elem),
                        (unsigned long)(actual_n * c_elem));
                __builtin_riscv_mf_dma_sync();
            }
        }
    }
    return MF_OK;
}

/*
 * mf_transposed_conv2d_2x2_i8 - Transposed 2x2 Stride-2 Upsampling (int8)
 *
 * GEMM + col2im on output. No input im2col to optimize.
 */
static inline int mf_transposed_conv2d_2x2_i8(const int8_t *input,
                                                const int8_t *weight,
                                                int32_t *output,
                                                int C_in, int C_out,
                                                int H, int W,
                                                mf_l1_buf_t workspace) {
    const int kH = 2, kW = 2;
    const int strH = 2, strW = 2;
    const int outH = 2 * H;
    const int outW = 2 * W;
    const int M_gemm = C_out * kH * kW;
    const int N_gemm = H * W;
    const int K_gemm = C_in;

    if (H <= 0 || W <= 0 || C_in <= 0 || C_out <= 0)
        return MF_ERR_INVALID_PARAM;

    mf_l1_alloc_t alloc;
    mf_l1_alloc_init(&alloc, workspace);

    size_t col_bytes = (size_t)M_gemm * (size_t)N_gemm * sizeof(int32_t);
    int32_t *col = (int32_t *)mf_l1_alloc(&alloc, col_bytes);
    if (!col) return MF_ERR_L1_TOO_SMALL;

    mf_l1_buf_t remaining = mf_l1_buf_from_alloc(&alloc);
    int rc = mf_gemm_ii8(M_gemm, N_gemm, K_gemm,
                          weight, input, col, remaining);
    if (rc != MF_OK) return rc;

    mf_memset(output, 0, (size_t)C_out * (size_t)outH * (size_t)outW * sizeof(int32_t));
    mf_col2im_i8((const int8_t *)col, (int8_t *)output,
                  C_out, outH, outW, kH, kW, strH, 0);

    return MF_OK;
}

/* ========================================================================
 * INT4 Conv2D Kernels
 * ======================================================================== */

/*
 * mf_conv2d_3x3_i4 - 3x3 Convolution (int4), partial materialization
 *
 * int4 packed at byte level: im2col and B-tile construction identical to i8.
 * K_nibbles = C_in * 9 * 2, K_bytes = K_nibbles / 2.
 * Output: [C_out, outH, outW] int16.
 */
static inline int mf_conv2d_3x3_i4(const int8_t *input,
                                     const int8_t *weight,
                                     int16_t *output,
                                     int C_in, int C_out,
                                     int H, int W, int pad,
                                     mf_l1_buf_t workspace) {
    const int kH = 3, kW = 3;
    const int outH = H + 2 * pad - kH + 1;
    const int outW = W + 2 * pad - kW + 1;
    const int K_nibbles = C_in * kH * kW * 2;
    const int K_bytes = K_nibbles / 2;
    const int K_bytes_padded = MF_DIV_CEIL(K_bytes, MF_TILE_K) * MF_TILE_K;
    const int N = outH * outW;

    if (outH <= 0 || outW <= 0)
        return MF_ERR_INVALID_PARAM;

    const int tile_m = MF_TILE_M;
    const int tile_n = MF_TILE_N;
    const int tile_k = MF_TILE_K;
    const int c_elem = 2;          /* int16 accumulator */
    const int padH = H + 2 * pad;
    const int padW = W + 2 * pad;

    mf_l1_alloc_t alloc;
    mf_l1_alloc_init(&alloc, workspace);

    /* Pre-padded input (byte-level, same as i8) */
    size_t pad_bytes = (size_t)C_in * padH * padW;
    int8_t *padded = (int8_t *)mf_l1_alloc(&alloc, pad_bytes);
    if (!padded) return MF_ERR_L1_TOO_SMALL;

    uint8_t *a_pad = (uint8_t *)mf_l1_alloc(&alloc, MF_TILE_A_BYTES);
    uint8_t *b_tmp = (uint8_t *)mf_l1_alloc(&alloc, tile_k * tile_n);
    uint8_t *b_pad = (uint8_t *)mf_l1_alloc(&alloc, MF_TILE_B_BYTES);
    uint8_t *c_tmp = (uint8_t *)mf_l1_alloc(&alloc, MF_ACC_OUT_I4);
    if (!a_pad || !b_tmp || !b_pad || !c_tmp)
        return MF_ERR_L1_TOO_SMALL;

    mf_memset(padded, 0, pad_bytes);
    for (int c = 0; c < C_in; c++)
        for (int h = 0; h < H; h++)
            __builtin_riscv_mf_dma_x(
                padded + c * padH * padW + (h + pad) * padW + pad,
                (void *)(input + c * H * W + h * W),
                (unsigned long)W);
    __builtin_riscv_mf_dma_sync();

    int m_tiles = MF_DIV_CEIL(C_out, tile_m);
    int n_tiles = MF_DIV_CEIL(N, tile_n);
    int k_tiles = K_bytes_padded / tile_k;

    const uint8_t *w_base = (const uint8_t *)weight;
    uint8_t *c_base = (uint8_t *)output;

    for (int mg = 0; mg < m_tiles; mg += MF_NUM_ACC) {
        int m_batch = MF_MIN(MF_NUM_ACC, m_tiles - mg);

        for (int nt = 0; nt < n_tiles; nt++) {
            int n_start = nt * tile_n;
            int actual_n = MF_MIN(tile_n, N - n_start);

            for (int a = 0; a < m_batch; a++)
                mf_tzero(a);

            for (int kt = 0; kt < k_tiles; kt++) {
                int k_start = kt * tile_k;
                int k_valid = MF_MIN(tile_k, K_bytes - k_start);

                if (k_valid < tile_k || actual_n < tile_n)
                    mf_memset(b_tmp, 0, tile_k * tile_n);

                /* i4 packed bytes: same im2col as i8 */
                for (int ki = 0; ki < k_valid; ki++) {
                    int k  = k_start + ki;
                    int ci = k / (kH * kW);
                    int kh = (k % (kH * kW)) / kW;
                    int kw = k % kW;
                    mf_btile_fill_s1_i8((int8_t *)b_tmp + ki * tile_n,
                                        padded, ci, kh, kw,
                                        padH, padW,
                                        n_start, actual_n, outW);
                }
                __builtin_riscv_mf_dma_sync();

                mf_dma_transpose_8(b_pad, b_tmp,
                                   (size_t)tile_n,
                                   (size_t)tile_n,
                                   (size_t)tile_k);

                for (int a = 0; a < m_batch; a++) {
                    int ms = (mg + a) * tile_m;
                    int ml = MF_MIN(tile_m, C_out - ms);
                    mf_atile_load(a_pad,
                                  w_base + ms * K_bytes_padded + k_start,
                                  ml, tile_m, tile_k, tile_k,
                                  K_bytes_padded);
                    mf_tmma_ii4(a, a_pad, b_pad);
                }
            }

            for (int a = 0; a < m_batch; a++) {
                int ms = (mg + a) * tile_m;
                int ml = MF_MIN(tile_m, C_out - ms);
                uint8_t *c_dst = c_base + (ms * N + n_start) * c_elem;

                mf_tstore16(a, c_tmp,
                            (size_t)(tile_m * tile_n * c_elem));
                for (int r = 0; r < ml; r++)
                    __builtin_riscv_mf_dma_x(
                        c_dst + r * N * c_elem,
                        (void *)(c_tmp + r * tile_n * c_elem),
                        (unsigned long)(actual_n * c_elem));
                __builtin_riscv_mf_dma_sync();
            }
        }
    }
    return MF_OK;
}

/*
 * mf_conv2d_1x1_i4 - 1x1 Pointwise Convolution (int4)
 */
static inline int mf_conv2d_1x1_i4(const int8_t *input,
                                     const int8_t *weight,
                                     int16_t *output,
                                     int C_in, int C_out,
                                     int H, int W,
                                     mf_l1_buf_t workspace) {
    const int N = H * W;
    const int K_nibbles = C_in * 2;
    if (C_in <= 0 || C_out <= 0 || N <= 0)
        return MF_ERR_INVALID_PARAM;
    return mf_gemm_ii4(C_out, N, K_nibbles, weight, input, output, workspace);
}

/*
 * mf_conv2d_2x2s2_i4 - 2x2 Stride-2 Downsampling Convolution (int4)
 */
static inline int mf_conv2d_2x2s2_i4(const int8_t *input,
                                       const int8_t *weight,
                                       int16_t *output,
                                       int C_in, int C_out,
                                       int H, int W,
                                       mf_l1_buf_t workspace) {
    const int kH = 2, kW = 2;
    const int outH = H / 2;
    const int outW = W / 2;
    const int K_nibbles = C_in * kH * kW * 2;
    const int K_bytes = K_nibbles / 2;
    const int K_bytes_padded = MF_DIV_CEIL(K_bytes, MF_TILE_K) * MF_TILE_K;
    const int N = outH * outW;

    if (outH <= 0 || outW <= 0 || (H % 2 != 0) || (W % 2 != 0))
        return MF_ERR_INVALID_PARAM;

    const int tile_m = MF_TILE_M;
    const int tile_n = MF_TILE_N;
    const int tile_k = MF_TILE_K;
    const int c_elem = 2;

    mf_l1_alloc_t alloc;
    mf_l1_alloc_init(&alloc, workspace);

    uint8_t *a_pad = (uint8_t *)mf_l1_alloc(&alloc, MF_TILE_A_BYTES);
    uint8_t *b_tmp = (uint8_t *)mf_l1_alloc(&alloc, tile_k * tile_n);
    uint8_t *b_pad = (uint8_t *)mf_l1_alloc(&alloc, MF_TILE_B_BYTES);
    uint8_t *c_tmp = (uint8_t *)mf_l1_alloc(&alloc, MF_ACC_OUT_I4);
    if (!a_pad || !b_tmp || !b_pad || !c_tmp)
        return MF_ERR_L1_TOO_SMALL;

    int m_tiles = MF_DIV_CEIL(C_out, tile_m);
    int n_tiles = MF_DIV_CEIL(N, tile_n);
    int k_tiles = K_bytes_padded / tile_k;

    const uint8_t *w_base = (const uint8_t *)weight;
    uint8_t *c_base = (uint8_t *)output;

    for (int mg = 0; mg < m_tiles; mg += MF_NUM_ACC) {
        int m_batch = MF_MIN(MF_NUM_ACC, m_tiles - mg);

        for (int nt = 0; nt < n_tiles; nt++) {
            int n_start = nt * tile_n;
            int actual_n = MF_MIN(tile_n, N - n_start);

            for (int a = 0; a < m_batch; a++)
                mf_tzero(a);

            for (int kt = 0; kt < k_tiles; kt++) {
                int k_start = kt * tile_k;
                int k_valid = MF_MIN(tile_k, K_bytes - k_start);

                if (k_valid < tile_k || actual_n < tile_n)
                    mf_memset(b_tmp, 0, tile_k * tile_n);

                for (int ki = 0; ki < k_valid; ki++) {
                    int k  = k_start + ki;
                    int ci = k / (kH * kW);
                    int kh = (k % (kH * kW)) / kW;
                    int kw = k % kW;
                    mf_btile_fill_s2_i8((int8_t *)b_tmp + ki * tile_n,
                                        input, ci, kh, kw,
                                        H, W,
                                        n_start, actual_n, outW);
                }

                mf_dma_transpose_8(b_pad, b_tmp,
                                   (size_t)tile_n,
                                   (size_t)tile_n,
                                   (size_t)tile_k);

                for (int a = 0; a < m_batch; a++) {
                    int ms = (mg + a) * tile_m;
                    int ml = MF_MIN(tile_m, C_out - ms);
                    int load_k = MF_MIN(tile_k, K_bytes - k_start);
                    mf_atile_load(a_pad,
                                  w_base + ms * K_bytes + k_start,
                                  ml, tile_m, load_k, tile_k, K_bytes);
                    mf_tmma_ii4(a, a_pad, b_pad);
                }
            }

            for (int a = 0; a < m_batch; a++) {
                int ms = (mg + a) * tile_m;
                int ml = MF_MIN(tile_m, C_out - ms);
                uint8_t *c_dst = c_base + (ms * N + n_start) * c_elem;

                mf_tstore16(a, c_tmp,
                            (size_t)(tile_m * tile_n * c_elem));
                for (int r = 0; r < ml; r++)
                    __builtin_riscv_mf_dma_x(
                        c_dst + r * N * c_elem,
                        (void *)(c_tmp + r * tile_n * c_elem),
                        (unsigned long)(actual_n * c_elem));
                __builtin_riscv_mf_dma_sync();
            }
        }
    }
    return MF_OK;
}

/*
 * mf_transposed_conv2d_2x2_i4 - Transposed 2x2 Stride-2 Upsampling (int4)
 */
static inline int mf_transposed_conv2d_2x2_i4(const int8_t *input,
                                                const int8_t *weight,
                                                int16_t *output,
                                                int C_in, int C_out,
                                                int H, int W,
                                                mf_l1_buf_t workspace) {
    const int kH = 2, kW = 2;
    const int outH = 2 * H;
    const int outW = 2 * W;
    const int M_gemm = C_out * kH * kW;
    const int N_gemm = H * W;
    const int K_nibbles = C_in * 2;

    if (H <= 0 || W <= 0 || C_in <= 0 || C_out <= 0)
        return MF_ERR_INVALID_PARAM;

    mf_l1_alloc_t alloc;
    mf_l1_alloc_init(&alloc, workspace);

    size_t col_buf_bytes = (size_t)M_gemm * (size_t)N_gemm * sizeof(int16_t);
    int16_t *col = (int16_t *)mf_l1_alloc(&alloc, col_buf_bytes);
    if (!col) return MF_ERR_L1_TOO_SMALL;

    mf_l1_buf_t remaining = mf_l1_buf_from_alloc(&alloc);
    int rc = mf_gemm_ii4(M_gemm, N_gemm, K_nibbles,
                          weight, input, col, remaining);
    if (rc != MF_OK) return rc;

    mf_memset(output, 0, (size_t)C_out * (size_t)outH * (size_t)outW * sizeof(int16_t));
    mf_col2im_i4((const int8_t *)col, (int8_t *)output,
                  C_out, outH, outW, kH, kW, 2, 0);

    return MF_OK;
}

/* ========================================================================
 * INT16 Conv2D Kernels
 *
 * int16: K dimension is in int16 elements. tile_k_elems = MF_K_ELEMS_I16.
 *        Accumulator output is int64.
 * ======================================================================== */

/*
 * mf_btile_fill_s1_i16 - Construct one B-tile row for stride-1 conv (int16)
 */
static inline void mf_btile_fill_s1_i16(int16_t *b_row,
                                          const int16_t *padded,
                                          int ci, int kh, int kw,
                                          int padH, int padW,
                                          int n_start, int actual_n,
                                          int outW) {
    int remaining = actual_n;
    int b_col = 0;
    int cur_oh = n_start / outW;
    int cur_ow = n_start % outW;

    while (remaining > 0) {
        int run = MF_MIN(remaining, outW - cur_ow);
        __builtin_riscv_mf_dma_x(
            b_row + b_col,
            (void *)(padded + ci * padH * padW
                     + (cur_oh + kh) * padW
                     + (cur_ow + kw)),
            (unsigned long)(run * sizeof(int16_t)));
        b_col += run;
        remaining -= run;
        cur_oh++;
        cur_ow = 0;
    }
}

/*
 * mf_btile_fill_s2_i16 - Construct one B-tile row for stride-2 conv (int16)
 */
static inline void mf_btile_fill_s2_i16(int16_t *b_row,
                                          const int16_t *input,
                                          int ci, int kh, int kw,
                                          int H, int W,
                                          int n_start, int actual_n,
                                          int outW) {
    int remaining = actual_n;
    int b_col = 0;
    int cur_oh = n_start / outW;
    int cur_ow = n_start % outW;

    while (remaining > 0) {
        int run = MF_MIN(remaining, outW - cur_ow);
        const int16_t *src = input + ci * H * W
            + (cur_oh * 2 + kh) * W + (cur_ow * 2 + kw);
        for (int j = 0; j < run; j++)
            b_row[b_col + j] = src[j * 2];
        b_col += run;
        remaining -= run;
        cur_oh++;
        cur_ow = 0;
    }
}

/*
 * mf_conv2d_3x3_i16 - 3x3 Convolution (int16), partial materialization
 */
static inline int mf_conv2d_3x3_i16(const int16_t *input,
                                      const int16_t *weight,
                                      int64_t *output,
                                      int C_in, int C_out,
                                      int H, int W, int pad,
                                      mf_l1_buf_t workspace) {
    const int kH = 3, kW = 3;
    const int outH = H + 2 * pad - kH + 1;
    const int outW = W + 2 * pad - kW + 1;
    const int K_elems = C_in * kH * kW;
    const int tile_k_elems = MF_K_ELEMS_I16;
    const int K_padded = MF_DIV_CEIL(K_elems, tile_k_elems) * tile_k_elems;
    const int N = outH * outW;

    if (outH <= 0 || outW <= 0)
        return MF_ERR_INVALID_PARAM;

    const int tile_m = MF_TILE_M;
    const int tile_n = MF_TILE_N;
    const int tile_k = MF_TILE_K;   /* bytes */
    const int c_elem = 8;           /* int64 accumulator */
    const int padH = H + 2 * pad;
    const int padW = W + 2 * pad;

    mf_l1_alloc_t alloc;
    mf_l1_alloc_init(&alloc, workspace);

    /* Pre-padded input: [C_in x padH x padW] int16 */
    size_t pad_bytes = (size_t)C_in * padH * padW * sizeof(int16_t);
    int16_t *padded = (int16_t *)mf_l1_alloc(&alloc, pad_bytes);
    if (!padded) return MF_ERR_L1_TOO_SMALL;

    uint8_t *a_pad = (uint8_t *)mf_l1_alloc(&alloc, MF_TILE_A_BYTES);
    /* B-tile: tile_k_elems rows of tile_n int16 elements = tile_k * tile_n bytes */
    uint8_t *b_tmp = (uint8_t *)mf_l1_alloc(&alloc, tile_k * tile_n);
    uint8_t *b_pad = (uint8_t *)mf_l1_alloc(&alloc, MF_TILE_B_BYTES);
    uint8_t *c_tmp = (uint8_t *)mf_l1_alloc(&alloc, MF_ACC_OUT_I16);
    if (!a_pad || !b_tmp || !b_pad || !c_tmp)
        return MF_ERR_L1_TOO_SMALL;

    /* Pre-pad input (int16 elements) */
    mf_memset(padded, 0, pad_bytes);
    for (int c = 0; c < C_in; c++)
        for (int h = 0; h < H; h++)
            __builtin_riscv_mf_dma_x(
                padded + c * padH * padW + (h + pad) * padW + pad,
                (void *)(input + c * H * W + h * W),
                (unsigned long)(W * sizeof(int16_t)));
    __builtin_riscv_mf_dma_sync();

    int m_tiles = MF_DIV_CEIL(C_out, tile_m);
    int n_tiles = MF_DIV_CEIL(N, tile_n);
    int k_tiles = K_padded / tile_k_elems;

    const uint8_t *w_base = (const uint8_t *)weight;
    uint8_t *c_base = (uint8_t *)output;

    for (int mg = 0; mg < m_tiles; mg += MF_NUM_ACC) {
        int m_batch = MF_MIN(MF_NUM_ACC, m_tiles - mg);

        for (int nt = 0; nt < n_tiles; nt++) {
            int n_start = nt * tile_n;
            int actual_n = MF_MIN(tile_n, N - n_start);

            for (int a = 0; a < m_batch; a++)
                mf_tzero(a);

            for (int kt = 0; kt < k_tiles; kt++) {
                int k_start_elem = kt * tile_k_elems;
                int k_valid_elems = MF_MIN(tile_k_elems, K_elems - k_start_elem);

                if (k_valid_elems < tile_k_elems || actual_n < tile_n)
                    mf_memset(b_tmp, 0, tile_k * tile_n);

                for (int ki = 0; ki < k_valid_elems; ki++) {
                    int k  = k_start_elem + ki;
                    int ci = k / (kH * kW);
                    int kh = (k % (kH * kW)) / kW;
                    int kw = k % kW;
                    mf_btile_fill_s1_i16(
                        (int16_t *)b_tmp + ki * tile_n,
                        padded, ci, kh, kw,
                        padH, padW,
                        n_start, actual_n, outW);
                }
                __builtin_riscv_mf_dma_sync();

                mf_dma_transpose_16(b_pad, b_tmp,
                                    (size_t)tile_n,
                                    (size_t)tile_n,
                                    (size_t)tile_k_elems);

                for (int a = 0; a < m_batch; a++) {
                    int ms = (mg + a) * tile_m;
                    int ml = MF_MIN(tile_m, C_out - ms);
                    /* Weight: [C_out, K_padded] int16 elements */
                    int w_stride_bytes = K_padded * (int)sizeof(int16_t);
                    int k_start_bytes = k_start_elem * (int)sizeof(int16_t);
                    mf_atile_load(a_pad,
                                  w_base + ms * w_stride_bytes + k_start_bytes,
                                  ml, tile_m, tile_k, tile_k, w_stride_bytes);
                    mf_tmma_ii16(a, a_pad, b_pad);
                }
            }

            for (int a = 0; a < m_batch; a++) {
                int ms = (mg + a) * tile_m;
                int ml = MF_MIN(tile_m, C_out - ms);
                uint8_t *c_dst = c_base + (ms * N + n_start) * c_elem;

                mf_tstore64(a, c_tmp,
                            (size_t)(tile_m * tile_n * c_elem));
                for (int r = 0; r < ml; r++)
                    __builtin_riscv_mf_dma_x(
                        c_dst + r * N * c_elem,
                        (void *)(c_tmp + r * tile_n * c_elem),
                        (unsigned long)(actual_n * c_elem));
                __builtin_riscv_mf_dma_sync();
            }
        }
    }
    return MF_OK;
}

/*
 * mf_conv2d_1x1_i16 - 1x1 Pointwise Convolution (int16)
 */
static inline int mf_conv2d_1x1_i16(const int16_t *input,
                                      const int16_t *weight,
                                      int64_t *output,
                                      int C_in, int C_out,
                                      int H, int W,
                                      mf_l1_buf_t workspace) {
    const int N = H * W;
    if (C_in <= 0 || C_out <= 0 || N <= 0)
        return MF_ERR_INVALID_PARAM;
    return mf_gemm_ii16(C_out, N, C_in, weight, input, output, workspace);
}

/*
 * mf_conv2d_2x2s2_i16 - 2x2 Stride-2 Downsampling Convolution (int16)
 */
static inline int mf_conv2d_2x2s2_i16(const int16_t *input,
                                        const int16_t *weight,
                                        int64_t *output,
                                        int C_in, int C_out,
                                        int H, int W,
                                        mf_l1_buf_t workspace) {
    const int kH = 2, kW = 2;
    const int outH = H / 2;
    const int outW = W / 2;
    const int K_elems = C_in * kH * kW;
    const int tile_k_elems = MF_K_ELEMS_I16;
    const int K_padded = MF_DIV_CEIL(K_elems, tile_k_elems) * tile_k_elems;
    const int N = outH * outW;

    if (outH <= 0 || outW <= 0 || (H % 2 != 0) || (W % 2 != 0))
        return MF_ERR_INVALID_PARAM;

    const int tile_m = MF_TILE_M;
    const int tile_n = MF_TILE_N;
    const int tile_k = MF_TILE_K;
    const int c_elem = 8;

    mf_l1_alloc_t alloc;
    mf_l1_alloc_init(&alloc, workspace);

    uint8_t *a_pad = (uint8_t *)mf_l1_alloc(&alloc, MF_TILE_A_BYTES);
    uint8_t *b_tmp = (uint8_t *)mf_l1_alloc(&alloc, tile_k * tile_n);
    uint8_t *b_pad = (uint8_t *)mf_l1_alloc(&alloc, MF_TILE_B_BYTES);
    uint8_t *c_tmp = (uint8_t *)mf_l1_alloc(&alloc, MF_ACC_OUT_I16);
    if (!a_pad || !b_tmp || !b_pad || !c_tmp)
        return MF_ERR_L1_TOO_SMALL;

    int m_tiles = MF_DIV_CEIL(C_out, tile_m);
    int n_tiles = MF_DIV_CEIL(N, tile_n);
    int k_tiles = K_padded / tile_k_elems;

    const uint8_t *w_base = (const uint8_t *)weight;
    uint8_t *c_base = (uint8_t *)output;

    for (int mg = 0; mg < m_tiles; mg += MF_NUM_ACC) {
        int m_batch = MF_MIN(MF_NUM_ACC, m_tiles - mg);

        for (int nt = 0; nt < n_tiles; nt++) {
            int n_start = nt * tile_n;
            int actual_n = MF_MIN(tile_n, N - n_start);

            for (int a = 0; a < m_batch; a++)
                mf_tzero(a);

            for (int kt = 0; kt < k_tiles; kt++) {
                int k_start_elem = kt * tile_k_elems;
                int k_valid_elems = MF_MIN(tile_k_elems, K_elems - k_start_elem);

                if (k_valid_elems < tile_k_elems || actual_n < tile_n)
                    mf_memset(b_tmp, 0, tile_k * tile_n);

                for (int ki = 0; ki < k_valid_elems; ki++) {
                    int k  = k_start_elem + ki;
                    int ci = k / (kH * kW);
                    int kh = (k % (kH * kW)) / kW;
                    int kw = k % kW;
                    mf_btile_fill_s2_i16(
                        (int16_t *)b_tmp + ki * tile_n,
                        input, ci, kh, kw,
                        H, W,
                        n_start, actual_n, outW);
                }

                mf_dma_transpose_16(b_pad, b_tmp,
                                    (size_t)tile_n,
                                    (size_t)tile_n,
                                    (size_t)tile_k_elems);

                for (int a = 0; a < m_batch; a++) {
                    int ms = (mg + a) * tile_m;
                    int ml = MF_MIN(tile_m, C_out - ms);
                    int w_stride_bytes = K_elems * (int)sizeof(int16_t);
                    int k_start_bytes = k_start_elem * (int)sizeof(int16_t);
                    int load_k_bytes = MF_MIN(tile_k, (K_elems - k_start_elem) * (int)sizeof(int16_t));
                    mf_atile_load(a_pad,
                                  w_base + ms * w_stride_bytes + k_start_bytes,
                                  ml, tile_m, load_k_bytes, tile_k,
                                  w_stride_bytes);
                    mf_tmma_ii16(a, a_pad, b_pad);
                }
            }

            for (int a = 0; a < m_batch; a++) {
                int ms = (mg + a) * tile_m;
                int ml = MF_MIN(tile_m, C_out - ms);
                uint8_t *c_dst = c_base + (ms * N + n_start) * c_elem;

                mf_tstore64(a, c_tmp,
                            (size_t)(tile_m * tile_n * c_elem));
                for (int r = 0; r < ml; r++)
                    __builtin_riscv_mf_dma_x(
                        c_dst + r * N * c_elem,
                        (void *)(c_tmp + r * tile_n * c_elem),
                        (unsigned long)(actual_n * c_elem));
                __builtin_riscv_mf_dma_sync();
            }
        }
    }
    return MF_OK;
}

/*
 * mf_transposed_conv2d_2x2_i16 - Transposed 2x2 Stride-2 Upsampling (int16)
 */
static inline int mf_transposed_conv2d_2x2_i16(const int16_t *input,
                                                 const int16_t *weight,
                                                 int64_t *output,
                                                 int C_in, int C_out,
                                                 int H, int W,
                                                 mf_l1_buf_t workspace) {
    const int kH = 2, kW = 2;
    const int outH = 2 * H;
    const int outW = 2 * W;
    const int M_gemm = C_out * kH * kW;
    const int N_gemm = H * W;
    const int K_gemm = C_in;

    if (H <= 0 || W <= 0 || C_in <= 0 || C_out <= 0)
        return MF_ERR_INVALID_PARAM;

    mf_l1_alloc_t alloc;
    mf_l1_alloc_init(&alloc, workspace);

    size_t col_buf_bytes = (size_t)M_gemm * (size_t)N_gemm * sizeof(int64_t);
    int64_t *col = (int64_t *)mf_l1_alloc(&alloc, col_buf_bytes);
    if (!col) return MF_ERR_L1_TOO_SMALL;

    mf_l1_buf_t remaining = mf_l1_buf_from_alloc(&alloc);
    int rc = mf_gemm_ii16(M_gemm, N_gemm, K_gemm,
                           weight, input, col, remaining);
    if (rc != MF_OK) return rc;

    mf_memset(output, 0, (size_t)C_out * (size_t)outH * (size_t)outW * sizeof(int64_t));
    mf_col2im_i16((const int16_t *)col, (int16_t *)output,
                   C_out, outH, outW, kH, kW, 2, 0);

    return MF_OK;
}

/* Exo-generated Conv2D kernels (all sign variants) */
#include "mf_conv2d_exo.h"

#endif /* MF_CONV2D_H */
