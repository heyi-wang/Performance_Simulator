/*
 * mf_gemm.h - MatrixFlow Tiled GeMM Kernels
 *
 * Implements tiled General Matrix Multiply using MatrixFlow tmma instructions.
 * Supports int4, int8, int16 with signed/unsigned variants (ii, uu, iu).
 *
 * A1: mf_gemm_{ii,uu,iu}{4,8,16} - 9 variants total
 *
 * All inputs/outputs are in L1. Caller provides workspace for tiling.
 * The kernel adapts pipeline depth (1-4 accumulators) based on workspace size.
 *
 * Matrix layout:
 *   A: [M × K_eff] row-major
 *   B: [K_eff × N] row-major
 *   C: [M × N] row-major output
 *
 * Hardware config: M=16, K=32, N=64, 4 accumulators
 */

#ifndef MF_GEMM_H
#define MF_GEMM_H

#include "mf_kernel.h"
#include "mf_dma.h"

/* ========================================================================
 * Internal: Tile size computation helpers
 * ======================================================================== */

/* Bytes per A-tile, B-tile, C-tile for each data type */
#define MF_GEMM_A_TILE_BYTES MF_TILE_A_BYTES /* 512  */
#define MF_GEMM_B_TILE_BYTES MF_TILE_B_BYTES /* 2048 */

/* C output bytes per tile: depends on accumulator type */
#define MF_GEMM_C_TILE_I4 MF_ACC_OUT_I4   /* 2048  M*N*2  int16 */
#define MF_GEMM_C_TILE_I8 MF_ACC_OUT_I8   /* 4096  M*N*4  int32 */
#define MF_GEMM_C_TILE_I16 MF_ACC_OUT_I16 /* 8192  M*N*8  int64 */

/* ========================================================================
 * Internal: Per-tile-step bytes (A + B + C for pipeline staging)
 * ======================================================================== */
#define MF_GEMM_STAGE_I4 (MF_GEMM_A_TILE_BYTES + MF_GEMM_B_TILE_BYTES + MF_GEMM_C_TILE_I4)
#define MF_GEMM_STAGE_I8 (MF_GEMM_A_TILE_BYTES + MF_GEMM_B_TILE_BYTES + MF_GEMM_C_TILE_I8)
#define MF_GEMM_STAGE_I16 (MF_GEMM_A_TILE_BYTES + MF_GEMM_B_TILE_BYTES + MF_GEMM_C_TILE_I16)

/* Minimum workspace: 1 stage (sequential, no pipelining) */
#define MF_GEMM_MIN_WS_I4 MF_GEMM_STAGE_I4
#define MF_GEMM_MIN_WS_I8 MF_GEMM_STAGE_I8
#define MF_GEMM_MIN_WS_I16 MF_GEMM_STAGE_I16

/* ========================================================================
 * Internal: tmma dispatch by type
 *
 * Uses immediate-index variants (tmma_*_i) for known accumulator indices.
 * ======================================================================== */

/* --- int8 tmma wrappers --- */
static inline void mf_tmma_ii8(int acc_idx, const void *A, const void *B)
{
    switch (acc_idx)
    {
    case 0:
        __builtin_riscv_mf_tmma_ii8_i(0, (void *)A, (void *)B);
        break;
    case 1:
        __builtin_riscv_mf_tmma_ii8_i(1, (void *)A, (void *)B);
        break;
    case 2:
        __builtin_riscv_mf_tmma_ii8_i(2, (void *)A, (void *)B);
        break;
    case 3:
        __builtin_riscv_mf_tmma_ii8_i(3, (void *)A, (void *)B);
        break;
    }
}

static inline void mf_tmma_uu8(int acc_idx, const void *A, const void *B)
{
    switch (acc_idx)
    {
    case 0:
        __builtin_riscv_mf_tmma_uu8_i(0, (void *)A, (void *)B);
        break;
    case 1:
        __builtin_riscv_mf_tmma_uu8_i(1, (void *)A, (void *)B);
        break;
    case 2:
        __builtin_riscv_mf_tmma_uu8_i(2, (void *)A, (void *)B);
        break;
    case 3:
        __builtin_riscv_mf_tmma_uu8_i(3, (void *)A, (void *)B);
        break;
    }
}

static inline void mf_tmma_iu8(int acc_idx, const void *A, const void *B)
{
    /* ISA has tmma.iu8.x (GPR) but no tmma.iu8.i; use register variant */
    __builtin_riscv_mf_tmma_iu8_x((size_t)acc_idx, (void *)A, (void *)B);
}

/* --- int4 tmma wrappers --- */
static inline void mf_tmma_ii4(int acc_idx, const void *A, const void *B)
{
    switch (acc_idx)
    {
    case 0:
        __builtin_riscv_mf_tmma_ii4_i(0, (void *)A, (void *)B);
        break;
    case 1:
        __builtin_riscv_mf_tmma_ii4_i(1, (void *)A, (void *)B);
        break;
    case 2:
        __builtin_riscv_mf_tmma_ii4_i(2, (void *)A, (void *)B);
        break;
    case 3:
        __builtin_riscv_mf_tmma_ii4_i(3, (void *)A, (void *)B);
        break;
    }
}

static inline void mf_tmma_uu4(int acc_idx, const void *A, const void *B)
{
    switch (acc_idx)
    {
    case 0:
        __builtin_riscv_mf_tmma_uu4_i(0, (void *)A, (void *)B);
        break;
    case 1:
        __builtin_riscv_mf_tmma_uu4_i(1, (void *)A, (void *)B);
        break;
    case 2:
        __builtin_riscv_mf_tmma_uu4_i(2, (void *)A, (void *)B);
        break;
    case 3:
        __builtin_riscv_mf_tmma_uu4_i(3, (void *)A, (void *)B);
        break;
    }
}

static inline void mf_tmma_iu4(int acc_idx, const void *A, const void *B)
{
    /* ISA has tmma.iu4.x (GPR) but no tmma.iu4.i; use register variant */
    __builtin_riscv_mf_tmma_iu4_x((size_t)acc_idx, (void *)A, (void *)B);
}

/* --- int16 tmma wrappers --- */
static inline void mf_tmma_ii16(int acc_idx, const void *A, const void *B)
{
    switch (acc_idx)
    {
    case 0:
        __builtin_riscv_mf_tmma_ii16_i(0, (void *)A, (void *)B);
        break;
    case 1:
        __builtin_riscv_mf_tmma_ii16_i(1, (void *)A, (void *)B);
        break;
    case 2:
        __builtin_riscv_mf_tmma_ii16_i(2, (void *)A, (void *)B);
        break;
    case 3:
        __builtin_riscv_mf_tmma_ii16_i(3, (void *)A, (void *)B);
        break;
    }
}

static inline void mf_tmma_uu16(int acc_idx, const void *A, const void *B)
{
    switch (acc_idx)
    {
    case 0:
        __builtin_riscv_mf_tmma_uu16_i(0, (void *)A, (void *)B);
        break;
    case 1:
        __builtin_riscv_mf_tmma_uu16_i(1, (void *)A, (void *)B);
        break;
    case 2:
        __builtin_riscv_mf_tmma_uu16_i(2, (void *)A, (void *)B);
        break;
    case 3:
        __builtin_riscv_mf_tmma_uu16_i(3, (void *)A, (void *)B);
        break;
    }
}

static inline void mf_tmma_iu16(int acc_idx, const void *A, const void *B)
{
    /* ISA has tmma.iu16.x (GPR) but no tmma.iu16.i; use register variant */
    __builtin_riscv_mf_tmma_iu16_x((size_t)acc_idx, (void *)A, (void *)B);
}

/* --- tzero wrapper --- */
static inline void mf_tzero(int acc_idx)
{
    switch (acc_idx)
    {
    case 0:
        __builtin_riscv_mf_tzero_i(0);
        break;
    case 1:
        __builtin_riscv_mf_tzero_i(1);
        break;
    case 2:
        __builtin_riscv_mf_tzero_i(2);
        break;
    case 3:
        __builtin_riscv_mf_tzero_i(3);
        break;
    }
}

/* --- tsync wrapper --- */
static inline void mf_tsync(int acc_idx)
{
    switch (acc_idx)
    {
    case 0:
        __builtin_riscv_mf_tsync_i(0);
        break;
    case 1:
        __builtin_riscv_mf_tsync_i(1);
        break;
    case 2:
        __builtin_riscv_mf_tsync_i(2);
        break;
    case 3:
        __builtin_riscv_mf_tsync_i(3);
        break;
    }
}

/* --- tstore wrappers --- */
static inline void mf_tstore16(int acc_idx, void *dst, size_t bytes)
{
    switch (acc_idx)
    {
    case 0:
        __builtin_riscv_mf_tstore16_ix(0, dst, (unsigned long)bytes);
        break;
    case 1:
        __builtin_riscv_mf_tstore16_ix(1, dst, (unsigned long)bytes);
        break;
    case 2:
        __builtin_riscv_mf_tstore16_ix(2, dst, (unsigned long)bytes);
        break;
    case 3:
        __builtin_riscv_mf_tstore16_ix(3, dst, (unsigned long)bytes);
        break;
    }
}

static inline void mf_tstore32(int acc_idx, void *dst, size_t bytes)
{
    switch (acc_idx)
    {
    case 0:
        __builtin_riscv_mf_tstore32_ix(0, dst, (unsigned long)bytes);
        break;
    case 1:
        __builtin_riscv_mf_tstore32_ix(1, dst, (unsigned long)bytes);
        break;
    case 2:
        __builtin_riscv_mf_tstore32_ix(2, dst, (unsigned long)bytes);
        break;
    case 3:
        __builtin_riscv_mf_tstore32_ix(3, dst, (unsigned long)bytes);
        break;
    }
}

static inline void mf_tstore64(int acc_idx, void *dst, size_t bytes)
{
    switch (acc_idx)
    {
    case 0:
        __builtin_riscv_mf_tstore64_ix(0, dst, (unsigned long)bytes);
        break;
    case 1:
        __builtin_riscv_mf_tstore64_ix(1, dst, (unsigned long)bytes);
        break;
    case 2:
        __builtin_riscv_mf_tstore64_ix(2, dst, (unsigned long)bytes);
        break;
    case 3:
        __builtin_riscv_mf_tstore64_ix(3, dst, (unsigned long)bytes);
        break;
    }
}

/* --- tload wrappers --- */
static inline void mf_tload16(int acc_idx, const void *src, size_t bytes)
{
    switch (acc_idx)
    {
    case 0:
        __builtin_riscv_mf_tload16_ix(0, (void *)src, (unsigned long)bytes);
        break;
    case 1:
        __builtin_riscv_mf_tload16_ix(1, (void *)src, (unsigned long)bytes);
        break;
    case 2:
        __builtin_riscv_mf_tload16_ix(2, (void *)src, (unsigned long)bytes);
        break;
    case 3:
        __builtin_riscv_mf_tload16_ix(3, (void *)src, (unsigned long)bytes);
        break;
    }
}

static inline void mf_tload32(int acc_idx, const void *src, size_t bytes)
{
    switch (acc_idx)
    {
    case 0:
        __builtin_riscv_mf_tload32_ix(0, (void *)src, (unsigned long)bytes);
        break;
    case 1:
        __builtin_riscv_mf_tload32_ix(1, (void *)src, (unsigned long)bytes);
        break;
    case 2:
        __builtin_riscv_mf_tload32_ix(2, (void *)src, (unsigned long)bytes);
        break;
    case 3:
        __builtin_riscv_mf_tload32_ix(3, (void *)src, (unsigned long)bytes);
        break;
    }
}

static inline void mf_tload64(int acc_idx, const void *src, size_t bytes)
{
    switch (acc_idx)
    {
    case 0:
        __builtin_riscv_mf_tload64_ix(0, (void *)src, (unsigned long)bytes);
        break;
    case 1:
        __builtin_riscv_mf_tload64_ix(1, (void *)src, (unsigned long)bytes);
        break;
    case 2:
        __builtin_riscv_mf_tload64_ix(2, (void *)src, (unsigned long)bytes);
        break;
    case 3:
        __builtin_riscv_mf_tload64_ix(3, (void *)src, (unsigned long)bytes);
        break;
    }
}

/* ========================================================================
 * Generic tiled GeMM implementation macro
 *
 * Generates mf_gemm_{sign}{width} for each variant.
 *
 * Parameters:
 *   M, N, K    - matrix dimensions (M×K × K×N → M×N)
 *   A          - [M × K] row-major in L1
 *   B          - [K × N] row-major in L1
 *   C          - [M × N] row-major output in L1
 *   workspace  - additional L1 scratch for tiling intermediates
 *
 * Tiling strategy:
 *   - Tile M by MF_TILE_M, N by MF_TILE_N, K by tile_k (type-dependent)
 *   - For each (m_tile, n_tile): zero acc, loop over k_tiles doing tmma, tstore
 *   - Pipeline depth adapts to workspace size (1-4 accumulators)
 *
 * K dimension for one tmma call (in elements):
 *   int4:  2*MF_TILE_K = 64 nibbles
 *   int8:  MF_TILE_K   = 32 bytes
 *   int16: MF_TILE_K/2 = 16 shorts
 *
 * But in bytes, A-tile is always MF_TILE_M * MF_TILE_K = 512 bytes
 * and B-tile is always MF_TILE_N * MF_TILE_K = 2048 bytes.
 * ======================================================================== */

/*
 * mf_gemm_ii8 - Signed int8 × Signed int8 GeMM
 *
 * C[M×N] int32 = A[M×K] int8 × B[K×N] int8
 * A is row-major, B is row-major.
 * B-tile transpose uses DMA strided copy.
 */
static inline int mf_gemm_ii8(int M, int N, int K,
                              const void *A, const void *B, void *C,
                              mf_l1_buf_t workspace)
{
    const int tile_m = MF_TILE_M;
    const int tile_n = MF_TILE_N;
    const int tile_k = MF_TILE_K; /* int8: K elements per tmma */
    const int c_elem = 4;         /* int32 accumulator output */

    /* Check minimum workspace */
    if (workspace.size < (size_t)MF_GEMM_MIN_WS_I8)
        return MF_ERR_L1_TOO_SMALL;

    /* Compute pipeline depth from workspace */
    int pipeline_depth = MF_MIN((int)(workspace.size / MF_GEMM_STAGE_I8),
                                MF_NUM_ACC);
    if (pipeline_depth < 1)
        return MF_ERR_L1_TOO_SMALL;

    /* Reserve pad buffers at the start of workspace for partial tiles */
    uint8_t *a_pad = (uint8_t *)workspace.ptr;
    uint8_t *b_pad = a_pad + MF_TILE_A_BYTES;
    uint8_t *ws_rest = b_pad + MF_TILE_B_BYTES;

    const uint8_t *a_base = (const uint8_t *)A;
    const uint8_t *b_base = (const uint8_t *)B;
    uint8_t *c_base = (uint8_t *)C;

    int m_tiles = MF_DIV_CEIL(M, tile_m);
    int n_tiles = MF_DIV_CEIL(N, tile_n);
    int k_tiles = MF_DIV_CEIL(K, tile_k);

    for (int mt = 0; mt < m_tiles; mt++)
    {
        int m_start = mt * tile_m;
        int m_len = MF_MIN(tile_m, M - m_start);

        for (int nt = 0; nt < n_tiles; nt++)
        {
            int n_start = nt * tile_n;
            int n_len = MF_MIN(tile_n, N - n_start);

            /* Use accumulator 0, zero it */
            mf_tzero(0);

            /* Accumulate over K tiles */
            for (int kt = 0; kt < k_tiles; kt++)
            {
                int k_start = kt * tile_k;
                int k_len = MF_MIN(tile_k, K - k_start);

                /* A tile: row-major [M, K], stride = K bytes */
                const uint8_t *a_tile = a_base + m_start * K + k_start;

                /* Pack A tile into contiguous buffer (DMA) */
                if (k_len < tile_k || m_len < tile_m)
                    mf_memset(a_pad, 0, MF_TILE_A_BYTES);
                if (k_len == tile_k)
                {
                    /* Full K: strided DMA gather (src stride=K, dst contiguous) */
                    mf_dma_load_tile_2d(a_pad, a_tile, (size_t)tile_k,
                                        (size_t)m_len, (size_t)K);
                    mf_dma_sync();
                }
                else
                {
                    /* Partial K: per-row DMA copies */
                    for (int r = 0; r < m_len; r++)
                        __builtin_riscv_mf_dma_x(a_pad + r * tile_k,
                                                 (void *)(a_tile + r * K),
                                                 (unsigned long)k_len);
                    __builtin_riscv_mf_dma_sync();
                }

                /* B tile: row-major [K, N]. tmma needs [tile_n, tile_k].
                 * Transpose sub-block B[k_start..+k_len, n_start..+n_len]
                 * from [k_len × n_len] at stride N into [n_len × k_len]. */
                mf_memset(b_pad, 0, MF_TILE_B_BYTES);
                if (k_len == tile_k)
                {
                    /* Full K tile: DMA transpose writes [n_len × tile_k]
                     * contiguous, rows align with b_pad stride. */
                    const uint8_t *b_src = b_base + k_start * N + n_start;
                    mf_dma_transpose_8(b_pad, b_src, N, n_len, k_len);
                }
                else
                {
                    /* Partial K tile: scalar gather (rows would misalign) */
                    for (int c = 0; c < n_len; c++)
                    {
                        const uint8_t *src_col = b_base + k_start * N + n_start + c;
                        for (int ki = 0; ki < k_len; ki++)
                            b_pad[c * tile_k + ki] = src_col[ki * N];
                    }
                }

                mf_tmma_ii8(0, a_pad, b_pad);
            }

            /* Store accumulator to C: C[m_start:m_start+tile_m][n_start:n_start+tile_n] */
            /* tstore32 writes M×N int32 values contiguously */
            uint8_t *c_tile = c_base + (m_start * N + n_start) * c_elem;

            if (n_len == tile_n && N == tile_n)
            {
                /* Contiguous output: single tstore */
                mf_tstore32(0, c_tile, (size_t)(m_len * n_len * c_elem));
            }
            else
            {
                /* Need to store to temp then scatter rows (DMA) */
                uint8_t *tmp = ws_rest;
                mf_tstore32(0, tmp, (size_t)(tile_m * tile_n * c_elem));
                for (int r = 0; r < m_len; r++)
                    __builtin_riscv_mf_dma_x(c_tile + r * N * c_elem,
                                             (void *)(tmp + r * tile_n * c_elem),
                                             (unsigned long)(n_len * c_elem));
                __builtin_riscv_mf_dma_sync();
            }
        }
    }
    return MF_OK;
}

/*
 * mf_gemm_uu8 - Unsigned int8 × Unsigned int8 GeMM
 */
static inline int mf_gemm_uu8(int M, int N, int K,
                              const void *A, const void *B, void *C,
                              mf_l1_buf_t workspace)
{
    const int tile_m = MF_TILE_M;
    const int tile_n = MF_TILE_N;
    const int tile_k = MF_TILE_K;
    const int c_elem = 4;

    if (workspace.size < (size_t)MF_GEMM_MIN_WS_I8)
        return MF_ERR_L1_TOO_SMALL;

    /* Reserve pad buffers at the start of workspace for partial tiles */
    uint8_t *a_pad = (uint8_t *)workspace.ptr;
    uint8_t *b_pad = a_pad + MF_TILE_A_BYTES;
    uint8_t *ws_rest = b_pad + MF_TILE_B_BYTES;

    const uint8_t *a_base = (const uint8_t *)A;
    const uint8_t *b_base = (const uint8_t *)B;
    uint8_t *c_base = (uint8_t *)C;

    int m_tiles = MF_DIV_CEIL(M, tile_m);
    int n_tiles = MF_DIV_CEIL(N, tile_n);
    int k_tiles = MF_DIV_CEIL(K, tile_k);

    for (int mt = 0; mt < m_tiles; mt++)
    {
        int m_start = mt * tile_m;
        int m_len = MF_MIN(tile_m, M - m_start);
        for (int nt = 0; nt < n_tiles; nt++)
        {
            int n_start = nt * tile_n;
            int n_len = MF_MIN(tile_n, N - n_start);
            mf_tzero(0);
            for (int kt = 0; kt < k_tiles; kt++)
            {
                int k_start = kt * tile_k;
                int k_len = MF_MIN(tile_k, K - k_start);

                const uint8_t *a_tile = a_base + m_start * K + k_start;

                if (k_len < tile_k || m_len < tile_m)
                    mf_memset(a_pad, 0, MF_TILE_A_BYTES);
                if (k_len == tile_k)
                {
                    mf_dma_load_tile_2d(a_pad, a_tile, (size_t)tile_k,
                                        (size_t)m_len, (size_t)K);
                    mf_dma_sync();
                }
                else
                {
                    for (int r = 0; r < m_len; r++)
                        __builtin_riscv_mf_dma_x(a_pad + r * tile_k,
                                                 (void *)(a_tile + r * K),
                                                 (unsigned long)k_len);
                    __builtin_riscv_mf_dma_sync();
                }

                mf_memset(b_pad, 0, MF_TILE_B_BYTES);
                if (k_len == tile_k)
                {
                    const uint8_t *b_src = b_base + k_start * N + n_start;
                    mf_dma_transpose_8(b_pad, b_src, N, n_len, k_len);
                }
                else
                {
                    for (int c = 0; c < n_len; c++)
                    {
                        const uint8_t *src_col = b_base + k_start * N + n_start + c;
                        for (int ki = 0; ki < k_len; ki++)
                            b_pad[c * tile_k + ki] = src_col[ki * N];
                    }
                }

                mf_tmma_uu8(0, a_pad, b_pad);
            }
            uint8_t *c_tile = c_base + (m_start * N + n_start) * c_elem;
            if (n_len == tile_n && N == tile_n)
            {
                mf_tstore32(0, c_tile, (size_t)(m_len * n_len * c_elem));
            }
            else
            {
                uint8_t *tmp = ws_rest;
                mf_tstore32(0, tmp, (size_t)(tile_m * tile_n * c_elem));
                for (int r = 0; r < m_len; r++)
                    __builtin_riscv_mf_dma_x(c_tile + r * N * c_elem,
                                             (void *)(tmp + r * tile_n * c_elem),
                                             (unsigned long)(n_len * c_elem));
                __builtin_riscv_mf_dma_sync();
            }
        }
    }
    return MF_OK;
}

/*
 * mf_gemm_iu8 - Signed int8 × Unsigned int8 GeMM
 */
static inline int mf_gemm_iu8(int M, int N, int K,
                              const void *A, const void *B, void *C,
                              mf_l1_buf_t workspace)
{
    const int tile_m = MF_TILE_M;
    const int tile_n = MF_TILE_N;
    const int tile_k = MF_TILE_K;
    const int c_elem = 4;

    if (workspace.size < (size_t)MF_GEMM_MIN_WS_I8)
        return MF_ERR_L1_TOO_SMALL;

    /* Reserve pad buffers at the start of workspace for partial tiles */
    uint8_t *a_pad = (uint8_t *)workspace.ptr;
    uint8_t *b_pad = a_pad + MF_TILE_A_BYTES;
    uint8_t *ws_rest = b_pad + MF_TILE_B_BYTES;

    const uint8_t *a_base = (const uint8_t *)A;
    const uint8_t *b_base = (const uint8_t *)B;
    uint8_t *c_base = (uint8_t *)C;

    int m_tiles = MF_DIV_CEIL(M, tile_m);
    int n_tiles = MF_DIV_CEIL(N, tile_n);
    int k_tiles = MF_DIV_CEIL(K, tile_k);

    for (int mt = 0; mt < m_tiles; mt++)
    {
        int m_start = mt * tile_m;
        int m_len = MF_MIN(tile_m, M - m_start);
        for (int nt = 0; nt < n_tiles; nt++)
        {
            int n_start = nt * tile_n;
            int n_len = MF_MIN(tile_n, N - n_start);
            mf_tzero(0);
            for (int kt = 0; kt < k_tiles; kt++)
            {
                int k_start = kt * tile_k;
                int k_len = MF_MIN(tile_k, K - k_start);

                const uint8_t *a_tile = a_base + m_start * K + k_start;

                if (k_len < tile_k || m_len < tile_m)
                    mf_memset(a_pad, 0, MF_TILE_A_BYTES);
                if (k_len == tile_k)
                {
                    mf_dma_load_tile_2d(a_pad, a_tile, (size_t)tile_k,
                                        (size_t)m_len, (size_t)K);
                    mf_dma_sync();
                }
                else
                {
                    for (int r = 0; r < m_len; r++)
                        __builtin_riscv_mf_dma_x(a_pad + r * tile_k,
                                                 (void *)(a_tile + r * K),
                                                 (unsigned long)k_len);
                    __builtin_riscv_mf_dma_sync();
                }

                mf_memset(b_pad, 0, MF_TILE_B_BYTES);
                if (k_len == tile_k)
                {
                    const uint8_t *b_src = b_base + k_start * N + n_start;
                    mf_dma_transpose_8(b_pad, b_src, N, n_len, k_len);
                }
                else
                {
                    for (int c = 0; c < n_len; c++)
                    {
                        const uint8_t *src_col = b_base + k_start * N + n_start + c;
                        for (int ki = 0; ki < k_len; ki++)
                            b_pad[c * tile_k + ki] = src_col[ki * N];
                    }
                }

                mf_tmma_iu8(0, a_pad, b_pad);
            }
            uint8_t *c_tile = c_base + (m_start * N + n_start) * c_elem;
            if (n_len == tile_n && N == tile_n)
            {
                mf_tstore32(0, c_tile, (size_t)(m_len * n_len * c_elem));
            }
            else
            {
                uint8_t *tmp = ws_rest;
                mf_tstore32(0, tmp, (size_t)(tile_m * tile_n * c_elem));
                for (int r = 0; r < m_len; r++)
                    __builtin_riscv_mf_dma_x(c_tile + r * N * c_elem,
                                             (void *)(tmp + r * tile_n * c_elem),
                                             (unsigned long)(n_len * c_elem));
                __builtin_riscv_mf_dma_sync();
            }
        }
    }
    return MF_OK;
}

/* ========================================================================
 * INT4 GeMM variants
 *
 * int4: A = [M × 2K] nibbles = [M × K] bytes (row-major)
 *       B = [2K × N] nibbles = [K × N] bytes (row-major)
 *       C = [M × N] int16 accumulator
 *
 * K_elems = 2 * MF_TILE_K = 64 nibbles per tmma call
 * But in bytes: A_tile = M*K = 512, B_tile = N*K = 2048 (same as int8)
 *
 * For tiling larger K: each k-tile covers 2*tile_k nibbles = tile_k bytes
 * ======================================================================== */

static inline int mf_gemm_ii4(int M, int N, int K_nibbles,
                              const void *A, const void *B, void *C,
                              mf_l1_buf_t workspace)
{
    const int tile_m = MF_TILE_M;
    const int tile_n = MF_TILE_N;
    const int tile_k_nibbles = MF_K_ELEMS_I4; /* 64 nibbles per tmma */
    const int tile_k_bytes = MF_TILE_K;       /* 32 bytes per tmma */
    const int c_elem = 2;                     /* int16 accumulator */
    const int K_bytes = K_nibbles / 2;

    if (workspace.size < (size_t)MF_GEMM_MIN_WS_I4)
        return MF_ERR_L1_TOO_SMALL;

    /* Reserve pad buffers at the start of workspace for partial tiles */
    uint8_t *a_pad = (uint8_t *)workspace.ptr;
    uint8_t *b_pad = a_pad + MF_TILE_A_BYTES;
    uint8_t *ws_rest = b_pad + MF_TILE_B_BYTES;

    const uint8_t *a_base = (const uint8_t *)A;
    const uint8_t *b_base = (const uint8_t *)B;
    uint8_t *c_base = (uint8_t *)C;

    int m_tiles = MF_DIV_CEIL(M, tile_m);
    int n_tiles = MF_DIV_CEIL(N, tile_n);
    int k_tiles = MF_DIV_CEIL(K_nibbles, tile_k_nibbles);

    for (int mt = 0; mt < m_tiles; mt++)
    {
        int m_start = mt * tile_m;
        int m_len = MF_MIN(tile_m, M - m_start);
        for (int nt = 0; nt < n_tiles; nt++)
        {
            int n_start = nt * tile_n;
            int n_len = MF_MIN(tile_n, N - n_start);
            mf_tzero(0);
            for (int kt = 0; kt < k_tiles; kt++)
            {
                int k_start = kt * tile_k_bytes;
                int k_len = MF_MIN(tile_k_bytes, K_bytes - k_start);

                const uint8_t *a_tile = a_base + m_start * K_bytes + k_start;

                if (k_len < tile_k_bytes || m_len < tile_m)
                    mf_memset(a_pad, 0, MF_TILE_A_BYTES);
                if (k_len == tile_k_bytes)
                {
                    mf_dma_load_tile_2d(a_pad, a_tile, (size_t)tile_k_bytes,
                                        (size_t)m_len, (size_t)K_bytes);
                    mf_dma_sync();
                }
                else
                {
                    for (int r = 0; r < m_len; r++)
                        __builtin_riscv_mf_dma_x(a_pad + r * tile_k_bytes,
                                                 (void *)(a_tile + r * K_bytes),
                                                 (unsigned long)k_len);
                    __builtin_riscv_mf_dma_sync();
                }

                mf_memset(b_pad, 0, MF_TILE_B_BYTES);
                if (k_len == tile_k_bytes)
                {
                    const uint8_t *b_src = b_base + k_start * N + n_start;
                    mf_dma_transpose_8(b_pad, b_src, N, n_len, k_len);
                }
                else
                {
                    for (int c = 0; c < n_len; c++)
                    {
                        const uint8_t *src_col = b_base + k_start * N + n_start + c;
                        for (int ki = 0; ki < k_len; ki++)
                            b_pad[c * tile_k_bytes + ki] = src_col[ki * N];
                    }
                }

                mf_tmma_ii4(0, a_pad, b_pad);
            }
            uint8_t *c_tile = c_base + (m_start * N + n_start) * c_elem;
            if (n_len == tile_n && N == tile_n)
            {
                mf_tstore16(0, c_tile, (size_t)(m_len * n_len * c_elem));
            }
            else
            {
                uint8_t *tmp = ws_rest;
                mf_tstore16(0, tmp, (size_t)(tile_m * tile_n * c_elem));
                for (int r = 0; r < m_len; r++)
                    __builtin_riscv_mf_dma_x(c_tile + r * N * c_elem,
                                             (void *)(tmp + r * tile_n * c_elem),
                                             (unsigned long)(n_len * c_elem));
                __builtin_riscv_mf_dma_sync();
            }
        }
    }
    return MF_OK;
}

static inline int mf_gemm_uu4(int M, int N, int K_nibbles,
                              const void *A, const void *B, void *C,
                              mf_l1_buf_t workspace)
{
    const int tile_m = MF_TILE_M;
    const int tile_n = MF_TILE_N;
    const int tile_k_nibbles = MF_K_ELEMS_I4;
    const int tile_k_bytes = MF_TILE_K;
    const int c_elem = 2;
    const int K_bytes = K_nibbles / 2;

    if (workspace.size < (size_t)MF_GEMM_MIN_WS_I4)
        return MF_ERR_L1_TOO_SMALL;

    /* Reserve pad buffers at the start of workspace for partial tiles */
    uint8_t *a_pad = (uint8_t *)workspace.ptr;
    uint8_t *b_pad = a_pad + MF_TILE_A_BYTES;
    uint8_t *ws_rest = b_pad + MF_TILE_B_BYTES;

    const uint8_t *a_base = (const uint8_t *)A;
    const uint8_t *b_base = (const uint8_t *)B;
    uint8_t *c_base = (uint8_t *)C;

    int m_tiles = MF_DIV_CEIL(M, tile_m);
    int n_tiles = MF_DIV_CEIL(N, tile_n);
    int k_tiles = MF_DIV_CEIL(K_nibbles, tile_k_nibbles);

    for (int mt = 0; mt < m_tiles; mt++)
    {
        int m_start = mt * tile_m;
        int m_len = MF_MIN(tile_m, M - m_start);
        for (int nt = 0; nt < n_tiles; nt++)
        {
            int n_start = nt * tile_n;
            int n_len = MF_MIN(tile_n, N - n_start);
            mf_tzero(0);
            for (int kt = 0; kt < k_tiles; kt++)
            {
                int k_start = kt * tile_k_bytes;
                int k_len = MF_MIN(tile_k_bytes, K_bytes - k_start);

                const uint8_t *a_tile = a_base + m_start * K_bytes + k_start;

                if (k_len < tile_k_bytes || m_len < tile_m)
                    mf_memset(a_pad, 0, MF_TILE_A_BYTES);
                if (k_len == tile_k_bytes)
                {
                    mf_dma_load_tile_2d(a_pad, a_tile, (size_t)tile_k_bytes,
                                        (size_t)m_len, (size_t)K_bytes);
                    mf_dma_sync();
                }
                else
                {
                    for (int r = 0; r < m_len; r++)
                        __builtin_riscv_mf_dma_x(a_pad + r * tile_k_bytes,
                                                 (void *)(a_tile + r * K_bytes),
                                                 (unsigned long)k_len);
                    __builtin_riscv_mf_dma_sync();
                }

                mf_memset(b_pad, 0, MF_TILE_B_BYTES);
                if (k_len == tile_k_bytes)
                {
                    const uint8_t *b_src = b_base + k_start * N + n_start;
                    mf_dma_transpose_8(b_pad, b_src, N, n_len, k_len);
                }
                else
                {
                    for (int c = 0; c < n_len; c++)
                    {
                        const uint8_t *src_col = b_base + k_start * N + n_start + c;
                        for (int ki = 0; ki < k_len; ki++)
                            b_pad[c * tile_k_bytes + ki] = src_col[ki * N];
                    }
                }

                mf_tmma_uu4(0, a_pad, b_pad);
            }
            uint8_t *c_tile = c_base + (m_start * N + n_start) * c_elem;
            if (n_len == tile_n && N == tile_n)
            {
                mf_tstore16(0, c_tile, (size_t)(m_len * n_len * c_elem));
            }
            else
            {
                uint8_t *tmp = ws_rest;
                mf_tstore16(0, tmp, (size_t)(tile_m * tile_n * c_elem));
                for (int r = 0; r < m_len; r++)
                    __builtin_riscv_mf_dma_x(c_tile + r * N * c_elem,
                                             (void *)(tmp + r * tile_n * c_elem),
                                             (unsigned long)(n_len * c_elem));
                __builtin_riscv_mf_dma_sync();
            }
        }
    }
    return MF_OK;
}

static inline int mf_gemm_iu4(int M, int N, int K_nibbles,
                              const void *A, const void *B, void *C,
                              mf_l1_buf_t workspace)
{
    const int tile_m = MF_TILE_M;
    const int tile_n = MF_TILE_N;
    const int tile_k_nibbles = MF_K_ELEMS_I4;
    const int tile_k_bytes = MF_TILE_K;
    const int c_elem = 2;
    const int K_bytes = K_nibbles / 2;

    if (workspace.size < (size_t)MF_GEMM_MIN_WS_I4)
        return MF_ERR_L1_TOO_SMALL;

    /* Reserve pad buffers at the start of workspace for partial tiles */
    uint8_t *a_pad = (uint8_t *)workspace.ptr;
    uint8_t *b_pad = a_pad + MF_TILE_A_BYTES;
    uint8_t *ws_rest = b_pad + MF_TILE_B_BYTES;

    const uint8_t *a_base = (const uint8_t *)A;
    const uint8_t *b_base = (const uint8_t *)B;
    uint8_t *c_base = (uint8_t *)C;

    int m_tiles = MF_DIV_CEIL(M, tile_m);
    int n_tiles = MF_DIV_CEIL(N, tile_n);
    int k_tiles = MF_DIV_CEIL(K_nibbles, tile_k_nibbles);

    for (int mt = 0; mt < m_tiles; mt++)
    {
        int m_start = mt * tile_m;
        int m_len = MF_MIN(tile_m, M - m_start);
        for (int nt = 0; nt < n_tiles; nt++)
        {
            int n_start = nt * tile_n;
            int n_len = MF_MIN(tile_n, N - n_start);
            mf_tzero(0);
            for (int kt = 0; kt < k_tiles; kt++)
            {
                int k_start = kt * tile_k_bytes;
                int k_len = MF_MIN(tile_k_bytes, K_bytes - k_start);

                const uint8_t *a_tile = a_base + m_start * K_bytes + k_start;

                if (k_len < tile_k_bytes || m_len < tile_m)
                    mf_memset(a_pad, 0, MF_TILE_A_BYTES);
                if (k_len == tile_k_bytes)
                {
                    mf_dma_load_tile_2d(a_pad, a_tile, (size_t)tile_k_bytes,
                                        (size_t)m_len, (size_t)K_bytes);
                    mf_dma_sync();
                }
                else
                {
                    for (int r = 0; r < m_len; r++)
                        __builtin_riscv_mf_dma_x(a_pad + r * tile_k_bytes,
                                                 (void *)(a_tile + r * K_bytes),
                                                 (unsigned long)k_len);
                    __builtin_riscv_mf_dma_sync();
                }

                mf_memset(b_pad, 0, MF_TILE_B_BYTES);
                if (k_len == tile_k_bytes)
                {
                    const uint8_t *b_src = b_base + k_start * N + n_start;
                    mf_dma_transpose_8(b_pad, b_src, N, n_len, k_len);
                }
                else
                {
                    for (int c = 0; c < n_len; c++)
                    {
                        const uint8_t *src_col = b_base + k_start * N + n_start + c;
                        for (int ki = 0; ki < k_len; ki++)
                            b_pad[c * tile_k_bytes + ki] = src_col[ki * N];
                    }
                }

                mf_tmma_iu4(0, a_pad, b_pad);
            }
            uint8_t *c_tile = c_base + (m_start * N + n_start) * c_elem;
            if (n_len == tile_n && N == tile_n)
            {
                mf_tstore16(0, c_tile, (size_t)(m_len * n_len * c_elem));
            }
            else
            {
                uint8_t *tmp = ws_rest;
                mf_tstore16(0, tmp, (size_t)(tile_m * tile_n * c_elem));
                for (int r = 0; r < m_len; r++)
                    __builtin_riscv_mf_dma_x(c_tile + r * N * c_elem,
                                             (void *)(tmp + r * tile_n * c_elem),
                                             (unsigned long)(n_len * c_elem));
                __builtin_riscv_mf_dma_sync();
            }
        }
    }
    return MF_OK;
}

/* ========================================================================
 * INT16 GeMM variants
 *
 * int16: A = [M × K/2] elements = [M × K] bytes (row-major)
 *        B = [K/2 × N] elements = [K/2 × N] (row-major, K bytes = K_elems*2)
 *        C = [M × N] int64 accumulator
 * ======================================================================== */

static inline int mf_gemm_ii16(int M, int N, int K_elems,
                               const void *A, const void *B, void *C,
                               mf_l1_buf_t workspace)
{
    const int tile_m = MF_TILE_M;
    const int tile_n = MF_TILE_N;
    const int tile_k_elems = MF_K_ELEMS_I16; /* 16 int16 elements */
    const int tile_k_bytes = MF_TILE_K;      /* 32 bytes */
    const int c_elem = 8;                    /* int64 accumulator */
    const int K_bytes = K_elems * 2;

    if (workspace.size < (size_t)MF_GEMM_MIN_WS_I16)
        return MF_ERR_L1_TOO_SMALL;

    /* Reserve pad buffers at the start of workspace for partial tiles */
    uint8_t *a_pad = (uint8_t *)workspace.ptr;
    uint8_t *b_pad = a_pad + MF_TILE_A_BYTES;
    uint8_t *ws_rest = b_pad + MF_TILE_B_BYTES;

    const uint8_t *a_base = (const uint8_t *)A;
    const uint8_t *b_base = (const uint8_t *)B;
    uint8_t *c_base = (uint8_t *)C;

    int m_tiles = MF_DIV_CEIL(M, tile_m);
    int n_tiles = MF_DIV_CEIL(N, tile_n);
    int k_tiles = MF_DIV_CEIL(K_elems, tile_k_elems);

    for (int mt = 0; mt < m_tiles; mt++)
    {
        int m_start = mt * tile_m;
        int m_len = MF_MIN(tile_m, M - m_start);
        for (int nt = 0; nt < n_tiles; nt++)
        {
            int n_start = nt * tile_n;
            int n_len = MF_MIN(tile_n, N - n_start);
            mf_tzero(0);
            for (int kt = 0; kt < k_tiles; kt++)
            {
                int k_start = kt * tile_k_bytes;
                int k_len = MF_MIN(tile_k_bytes, K_bytes - k_start);

                const uint8_t *a_tile = a_base + m_start * K_bytes + k_start;

                if (k_len < tile_k_bytes || m_len < tile_m)
                    mf_memset(a_pad, 0, MF_TILE_A_BYTES);
                if (k_len == tile_k_bytes)
                {
                    mf_dma_load_tile_2d(a_pad, a_tile, (size_t)tile_k_bytes,
                                        (size_t)m_len, (size_t)K_bytes);
                    mf_dma_sync();
                }
                else
                {
                    for (int r = 0; r < m_len; r++)
                        __builtin_riscv_mf_dma_x(a_pad + r * tile_k_bytes,
                                                 (void *)(a_tile + r * K_bytes),
                                                 (unsigned long)k_len);
                    __builtin_riscv_mf_dma_sync();
                }

                mf_memset(b_pad, 0, MF_TILE_B_BYTES);
                if (k_len == tile_k_bytes)
                {
                    /* Full K tile: DMA transpose int16 sub-block */
                    const uint8_t *b_src = b_base + k_start * N + n_start * 2;
                    mf_dma_transpose_16(b_pad, b_src, N * 2, n_len, k_len / 2);
                }
                else
                {
                    /* Partial K tile: scalar gather */
                    const uint16_t *bw = (const uint16_t *)b_base;
                    int k_elem_start = k_start / 2;
                    int k_elem_len = k_len / 2;
                    for (int c = 0; c < n_len; c++)
                    {
                        uint16_t *bp = (uint16_t *)(b_pad + c * tile_k_bytes);
                        for (int ki = 0; ki < k_elem_len; ki++)
                            bp[ki] = bw[(k_elem_start + ki) * N + n_start + c];
                    }
                }

                mf_tmma_ii16(0, a_pad, b_pad);
            }
            uint8_t *c_tile = c_base + (m_start * N + n_start) * c_elem;
            if (n_len == tile_n && N == tile_n)
            {
                mf_tstore64(0, c_tile, (size_t)(m_len * n_len * c_elem));
            }
            else
            {
                uint8_t *tmp = ws_rest;
                mf_tstore64(0, tmp, (size_t)(tile_m * tile_n * c_elem));
                for (int r = 0; r < m_len; r++)
                    __builtin_riscv_mf_dma_x(c_tile + r * N * c_elem,
                                             (void *)(tmp + r * tile_n * c_elem),
                                             (unsigned long)(n_len * c_elem));
                __builtin_riscv_mf_dma_sync();
            }
        }
    }
    return MF_OK;
}

static inline int mf_gemm_uu16(int M, int N, int K_elems,
                               const void *A, const void *B, void *C,
                               mf_l1_buf_t workspace)
{
    const int tile_m = MF_TILE_M;
    const int tile_n = MF_TILE_N;
    const int tile_k_elems = MF_K_ELEMS_I16;
    const int tile_k_bytes = MF_TILE_K;
    const int c_elem = 8;
    const int K_bytes = K_elems * 2;

    if (workspace.size < (size_t)MF_GEMM_MIN_WS_I16)
        return MF_ERR_L1_TOO_SMALL;

    /* Reserve pad buffers at the start of workspace for partial tiles */
    uint8_t *a_pad = (uint8_t *)workspace.ptr;
    uint8_t *b_pad = a_pad + MF_TILE_A_BYTES;
    uint8_t *ws_rest = b_pad + MF_TILE_B_BYTES;

    const uint8_t *a_base = (const uint8_t *)A;
    const uint8_t *b_base = (const uint8_t *)B;
    uint8_t *c_base = (uint8_t *)C;

    int m_tiles = MF_DIV_CEIL(M, tile_m);
    int n_tiles = MF_DIV_CEIL(N, tile_n);
    int k_tiles = MF_DIV_CEIL(K_elems, tile_k_elems);

    for (int mt = 0; mt < m_tiles; mt++)
    {
        int m_start = mt * tile_m;
        int m_len = MF_MIN(tile_m, M - m_start);
        for (int nt = 0; nt < n_tiles; nt++)
        {
            int n_start = nt * tile_n;
            int n_len = MF_MIN(tile_n, N - n_start);
            mf_tzero(0);
            for (int kt = 0; kt < k_tiles; kt++)
            {
                int k_start = kt * tile_k_bytes;
                int k_len = MF_MIN(tile_k_bytes, K_bytes - k_start);

                const uint8_t *a_tile = a_base + m_start * K_bytes + k_start;

                if (k_len < tile_k_bytes || m_len < tile_m)
                    mf_memset(a_pad, 0, MF_TILE_A_BYTES);
                if (k_len == tile_k_bytes)
                {
                    mf_dma_load_tile_2d(a_pad, a_tile, (size_t)tile_k_bytes,
                                        (size_t)m_len, (size_t)K_bytes);
                    mf_dma_sync();
                }
                else
                {
                    for (int r = 0; r < m_len; r++)
                        __builtin_riscv_mf_dma_x(a_pad + r * tile_k_bytes,
                                                 (void *)(a_tile + r * K_bytes),
                                                 (unsigned long)k_len);
                    __builtin_riscv_mf_dma_sync();
                }

                mf_memset(b_pad, 0, MF_TILE_B_BYTES);
                if (k_len == tile_k_bytes)
                {
                    /* Full K tile: DMA transpose int16 sub-block */
                    const uint8_t *b_src = b_base + k_start * N + n_start * 2;
                    mf_dma_transpose_16(b_pad, b_src, N * 2, n_len, k_len / 2);
                }
                else
                {
                    /* Partial K tile: scalar gather */
                    const uint16_t *bw = (const uint16_t *)b_base;
                    int k_elem_start = k_start / 2;
                    int k_elem_len = k_len / 2;
                    for (int c = 0; c < n_len; c++)
                    {
                        uint16_t *bp = (uint16_t *)(b_pad + c * tile_k_bytes);
                        for (int ki = 0; ki < k_elem_len; ki++)
                            bp[ki] = bw[(k_elem_start + ki) * N + n_start + c];
                    }
                }

                mf_tmma_uu16(0, a_pad, b_pad);
            }
            uint8_t *c_tile = c_base + (m_start * N + n_start) * c_elem;
            if (n_len == tile_n && N == tile_n)
            {
                mf_tstore64(0, c_tile, (size_t)(m_len * n_len * c_elem));
            }
            else
            {
                uint8_t *tmp = ws_rest;
                mf_tstore64(0, tmp, (size_t)(tile_m * tile_n * c_elem));
                for (int r = 0; r < m_len; r++)
                    __builtin_riscv_mf_dma_x(c_tile + r * N * c_elem,
                                             (void *)(tmp + r * tile_n * c_elem),
                                             (unsigned long)(n_len * c_elem));
                __builtin_riscv_mf_dma_sync();
            }
        }
    }
    return MF_OK;
}

static inline int mf_gemm_iu16(int M, int N, int K_elems,
                               const void *A, const void *B, void *C,
                               mf_l1_buf_t workspace)
{
    const int tile_m = MF_TILE_M;
    const int tile_n = MF_TILE_N;
    const int tile_k_elems = MF_K_ELEMS_I16;
    const int tile_k_bytes = MF_TILE_K;
    const int c_elem = 8;
    const int K_bytes = K_elems * 2;

    if (workspace.size < (size_t)MF_GEMM_MIN_WS_I16)
        return MF_ERR_L1_TOO_SMALL;

    /* Reserve pad buffers at the start of workspace for partial tiles */
    uint8_t *a_pad = (uint8_t *)workspace.ptr;
    uint8_t *b_pad = a_pad + MF_TILE_A_BYTES;
    uint8_t *ws_rest = b_pad + MF_TILE_B_BYTES;

    const uint8_t *a_base = (const uint8_t *)A;
    const uint8_t *b_base = (const uint8_t *)B;
    uint8_t *c_base = (uint8_t *)C;

    int m_tiles = MF_DIV_CEIL(M, tile_m);
    int n_tiles = MF_DIV_CEIL(N, tile_n);
    int k_tiles = MF_DIV_CEIL(K_elems, tile_k_elems);

    for (int mt = 0; mt < m_tiles; mt++)
    {
        int m_start = mt * tile_m;
        int m_len = MF_MIN(tile_m, M - m_start);
        for (int nt = 0; nt < n_tiles; nt++)
        {
            int n_start = nt * tile_n;
            int n_len = MF_MIN(tile_n, N - n_start);
            mf_tzero(0);
            for (int kt = 0; kt < k_tiles; kt++)
            {
                int k_start = kt * tile_k_bytes;
                int k_len = MF_MIN(tile_k_bytes, K_bytes - k_start);

                const uint8_t *a_tile = a_base + m_start * K_bytes + k_start;

                if (k_len < tile_k_bytes || m_len < tile_m)
                    mf_memset(a_pad, 0, MF_TILE_A_BYTES);
                if (k_len == tile_k_bytes)
                {
                    mf_dma_load_tile_2d(a_pad, a_tile, (size_t)tile_k_bytes,
                                        (size_t)m_len, (size_t)K_bytes);
                    mf_dma_sync();
                }
                else
                {
                    for (int r = 0; r < m_len; r++)
                        __builtin_riscv_mf_dma_x(a_pad + r * tile_k_bytes,
                                                 (void *)(a_tile + r * K_bytes),
                                                 (unsigned long)k_len);
                    __builtin_riscv_mf_dma_sync();
                }

                mf_memset(b_pad, 0, MF_TILE_B_BYTES);
                if (k_len == tile_k_bytes)
                {
                    /* Full K tile: DMA transpose int16 sub-block */
                    const uint8_t *b_src = b_base + k_start * N + n_start * 2;
                    mf_dma_transpose_16(b_pad, b_src, N * 2, n_len, k_len / 2);
                }
                else
                {
                    /* Partial K tile: scalar gather */
                    const uint16_t *bw = (const uint16_t *)b_base;
                    int k_elem_start = k_start / 2;
                    int k_elem_len = k_len / 2;
                    for (int c = 0; c < n_len; c++)
                    {
                        uint16_t *bp = (uint16_t *)(b_pad + c * tile_k_bytes);
                        for (int ki = 0; ki < k_elem_len; ki++)
                            bp[ki] = bw[(k_elem_start + ki) * N + n_start + c];
                    }
                }

                mf_tmma_iu16(0, a_pad, b_pad);
            }
            uint8_t *c_tile = c_base + (m_start * N + n_start) * c_elem;
            if (n_len == tile_n && N == tile_n)
            {
                mf_tstore64(0, c_tile, (size_t)(m_len * n_len * c_elem));
            }
            else
            {
                uint8_t *tmp = ws_rest;
                mf_tstore64(0, tmp, (size_t)(tile_m * tile_n * c_elem));
                for (int r = 0; r < m_len; r++)
                    __builtin_riscv_mf_dma_x(c_tile + r * N * c_elem,
                                             (void *)(tmp + r * tile_n * c_elem),
                                             (unsigned long)(n_len * c_elem));
                __builtin_riscv_mf_dma_sync();
            }
        }
    }
    return MF_OK;
}

/* ========================================================================
 * Transposed GeMM for small-N optimization
 *
 * When N < MF_TILE_N (e.g. N=16 with tile_N=64), the standard GeMM wastes
 * 75% of each tmma call. This function computes the same result via:
 *
 *   C[M×N] = A[M×K] × B[K×N]
 *   ≡ transpose( B^T[N×K] × A^T[K×M] )
 *
 * The transposed GEMM has M'=N (small → fits M_tile exactly),
 * N'=M (large → fills N_tile fully), achieving 100% utilization.
 *
 * Cost: 3 DMA transpose instructions (B, A, C^T) — negligible vs tmma savings.
 * ======================================================================== */
static inline int mf_gemm_ii8_transposed(int M, int N, int K,
                                         const void *A, const void *B, void *C,
                                         mf_l1_buf_t workspace)
{
    const int c_elem = 4; /* int32 accumulator */

    mf_l1_alloc_t alloc;
    mf_l1_alloc_init(&alloc, workspace);

    /* Allocate transpose buffers */
    int8_t *bt = (int8_t *)mf_l1_alloc(&alloc, (size_t)N * K);            /* B^T [N×K] */
    int8_t *at = (int8_t *)mf_l1_alloc(&alloc, (size_t)K * M);            /* A^T [K×M] */
    int32_t *ct = (int32_t *)mf_l1_alloc(&alloc, (size_t)N * M * c_elem); /* C^T [N×M] */

    if (!bt || !at || !ct)
        return MF_ERR_L1_TOO_SMALL;

    /* Transpose B [K×N] row-major → B^T [N×K] row-major */
    mf_dma_transpose_load(bt, B, K, N, 1);

    /* Transpose A [M×K] row-major → A^T [K×M] row-major */
    mf_dma_transpose_load(at, A, M, K, 1);

    /* GEMM: C^T[N×M] = B^T[N×K] × A^T[K×M] */
    mf_l1_buf_t remaining;
    remaining.ptr = alloc.base + alloc.offset;
    remaining.size = alloc.capacity - alloc.offset;
    int rc = mf_gemm_ii8(N, M, K, bt, at, ct, remaining);
    if (rc != MF_OK)
        return rc;

    /* Transpose C^T [N×M] int32 → C [M×N] int32 */
    mf_dma_transpose_load(C, ct, N, M, c_elem);

    return MF_OK;
}

/* Exo-generated GeMM kernels (all 12 sign x bit-width variants) */
#include "mf_gemm_exo.h"

#endif /* MF_GEMM_H */
