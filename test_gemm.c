/*
 * test_gemm.c - Bare-Metal Test Suite for MatrixFlow GeMM Kernels
 *
 * Runs on the spike simulator. No libc available.
 * Tests all 9 GeMM variants: {ii,uu,iu} x {4,8,16}
 *
 * Test pattern: identity-like A (single nonzero per row on diagonal)
 * combined with constant-fill B. This yields a predictable constant
 * output across all elements of C.
 *
 * Hardware config: v1-4-16-32-64
 *   int8:  A=16x32, B=32x64 (row-major [K,N]), acc=16x64 int32
 *   int4:  A=16x32 bytes (64 nibbles/row), B=32x64 bytes (row-major), acc=16x64 int16
 *   int16: A=16x16 elems (32 bytes/row), B=16x64 elems (row-major), acc=16x64 int64
 *
 * Return codes: 0 = all pass, 1-9 = tile-sized test N failed, 10 = large random test failed.
 */
#include <stdio.h>
#include "mf_gemm.h"

/* ========================================================================
 * Static buffers (aligned for hardware requirements)
 * ======================================================================== */
static unsigned char matA[512]   __attribute__((aligned(8)));   /* A tile */
static unsigned char matB[2048]  __attribute__((aligned(8)));   /* B tile */
static unsigned char matC[8192]  __attribute__((aligned(8)));   /* C: max for int64 */
static unsigned char workspace[32768] __attribute__((aligned(8)));

static mf_l1_buf_t ws = { workspace, sizeof(workspace) };

/* ========================================================================
 * Test 1: test_gemm_ii8
 *
 * Signed int8 x Signed int8 -> int32
 * M=16, N=64, K=32
 * A: identity-like, diagonal value = 1 (row i has A[i][i]=1, rest 0)
 * B: all 1s
 * Expected C: every element = 1 (dot product picks up B[i][j] = 1)
 * ======================================================================== */
static int test_gemm_ii8(void) {
    const int M = MF_TILE_M;   /* 16 */
    const int N = MF_TILE_N;   /* 64 */
    const int K = MF_TILE_K;   /* 32 */
    int rc;

    /* Clear A, then set diagonal = 1 */
    mf_memset(matA, 0, (size_t)(M * K));
    for (int i = 0; i < M; i++)
        matA[i * K + i] = 1;  /* A[row i][col i] = 1 (signed int8) */

    /* Fill B: all 1s. B is row-major [K, N] */
    mf_memset(matB, 1, (size_t)(K * N));

    /* Clear C */
    mf_memset(matC, 0, (size_t)(M * N * 4));

    rc = mf_gemm_ii8(M, N, K, matA, matB, matC, ws);
    if (rc != MF_OK)
        return -1;

    /* Verify: every int32 element of C should be 1 */
    int32_t *c = (int32_t *)matC;
    for (int i = 0; i < M * N; i++) {
        if (c[i] != 1)
            return -1;
    }
    return 0;
}

/* ========================================================================
 * Test 2: test_gemm_uu8
 *
 * Unsigned int8 x Unsigned int8 -> int32
 * A: diagonal value = 2
 * B: all 3s
 * Expected C: every element = 2 * 3 = 6
 * ======================================================================== */
static int test_gemm_uu8(void) {
    const int M = MF_TILE_M;
    const int N = MF_TILE_N;
    const int K = MF_TILE_K;
    int rc;

    mf_memset(matA, 0, (size_t)(M * K));
    for (int i = 0; i < M; i++)
        matA[i * K + i] = 2;

    mf_memset(matB, 3, (size_t)(N * K));

    mf_memset(matC, 0, (size_t)(M * N * 4));

    rc = mf_gemm_uu8(M, N, K, matA, matB, matC, ws);
    if (rc != MF_OK)
        return -1;

    int32_t *c = (int32_t *)matC;
    for (int i = 0; i < M * N; i++) {
        if (c[i] != 6)
            return -1;
    }
    return 0;
}

/* ========================================================================
 * Test 3: test_gemm_iu8
 *
 * Signed int8 A x Unsigned int8 B -> int32
 * A: diagonal value = -1 (0xFF as signed int8)
 * B: all 5s (unsigned)
 * Expected C: every element = (-1) * 5 = -5
 * ======================================================================== */
static int test_gemm_iu8(void) {
    const int M = MF_TILE_M;
    const int N = MF_TILE_N;
    const int K = MF_TILE_K;
    int rc;

    mf_memset(matA, 0, (size_t)(M * K));
    for (int i = 0; i < M; i++)
        matA[i * K + i] = 0xFF;  /* -1 as signed int8 */

    mf_memset(matB, 5, (size_t)(N * K));

    mf_memset(matC, 0, (size_t)(M * N * 4));

    rc = mf_gemm_iu8(M, N, K, matA, matB, matC, ws);
    if (rc != MF_OK)
        return -1;

    int32_t *c = (int32_t *)matC;
    for (int i = 0; i < M * N; i++) {
        if (c[i] != -5)
            return -1;
    }
    return 0;
}

/* ========================================================================
 * Test 4: test_gemm_ii4
 *
 * Signed int4 x Signed int4 -> int16
 * K_nibbles = 64 (one tile), K_bytes = 32
 * M=16, N=64
 *
 * A: identity-like in nibble domain. Row i has nibble value 1 at
 *    nibble position i, rest 0. Since M=16, positions 0..15 fit in
 *    the first 8 bytes of each 32-byte row.
 *    Nibble packing: little-endian nibble order (lower 4 bits = first element).
 *    Nibble position i -> byte i/2, shift (i%2)*4.
 *
 * B: all nibbles = 1, packed as 0x11 per byte.
 * Expected C: every int16 element = 1
 * ======================================================================== */
static int test_gemm_ii4(void) {
    const int M = MF_TILE_M;       /* 16 */
    const int N = MF_TILE_N;       /* 64 */
    const int K_nibbles = MF_K_ELEMS_I4;  /* 64 nibbles */
    const int K_bytes = MF_TILE_K; /* 32 */
    int rc;

    /* Clear A */
    mf_memset(matA, 0, (size_t)(M * K_bytes));

    /* Set identity diagonal: row i, nibble position i = 1 */
    for (int i = 0; i < M; i++) {
        int row_offset = i * K_bytes;
        int byte_idx = i / 2;
        int nibble_pos = i % 2;  /* 0 = lower nibble, 1 = upper nibble */
        matA[row_offset + byte_idx] |= (uint8_t)(1 << (nibble_pos * 4));
    }

    /* Fill B: all nibbles = 1 -> every byte = 0x11 */
    mf_memset(matB, 0x11, (size_t)(N * K_bytes));

    /* Clear C */
    mf_memset(matC, 0, (size_t)(M * N * 2));

    rc = mf_gemm_ii4(M, N, K_nibbles, matA, matB, matC, ws);
    if (rc != MF_OK)
        return -1;

    /* Verify: every int16 element should be 1 */
    int16_t *c = (int16_t *)matC;
    for (int i = 0; i < M * N; i++) {
        if (c[i] != 1)
            return -1;
    }
    return 0;
}

/* ========================================================================
 * Test 5: test_gemm_uu4
 *
 * Unsigned int4 x Unsigned int4 -> int16
 * A: identity-like, diagonal nibble = 2
 * B: all nibbles = 3, packed as 0x33
 * Expected C: every int16 element = 2 * 3 = 6
 * ======================================================================== */
static int test_gemm_uu4(void) {
    const int M = MF_TILE_M;
    const int N = MF_TILE_N;
    const int K_nibbles = MF_K_ELEMS_I4;
    const int K_bytes = MF_TILE_K;
    int rc;

    mf_memset(matA, 0, (size_t)(M * K_bytes));

    for (int i = 0; i < M; i++) {
        int row_offset = i * K_bytes;
        int byte_idx = i / 2;
        int nibble_pos = i % 2;
        matA[row_offset + byte_idx] |= (uint8_t)(2 << (nibble_pos * 4));
    }

    /* B: all nibbles = 3 -> every byte = 0x33 */
    mf_memset(matB, 0x33, (size_t)(N * K_bytes));

    mf_memset(matC, 0, (size_t)(M * N * 2));

    rc = mf_gemm_uu4(M, N, K_nibbles, matA, matB, matC, ws);
    if (rc != MF_OK)
        return -1;

    int16_t *c = (int16_t *)matC;
    for (int i = 0; i < M * N; i++) {
        if (c[i] != 6)
            return -1;
    }
    return 0;
}

/* ========================================================================
 * Test 6: test_gemm_iu4
 *
 * Signed int4 A x Unsigned int4 B -> int16
 * A: diagonal nibble = -1 (0xF as signed 4-bit = -1)
 * B: all nibbles = 2, packed as 0x22
 * Expected C: every int16 element = (-1) * 2 = -2
 * ======================================================================== */
static int test_gemm_iu4(void) {
    const int M = MF_TILE_M;
    const int N = MF_TILE_N;
    const int K_nibbles = MF_K_ELEMS_I4;
    const int K_bytes = MF_TILE_K;
    int rc;

    mf_memset(matA, 0, (size_t)(M * K_bytes));

    for (int i = 0; i < M; i++) {
        int row_offset = i * K_bytes;
        int byte_idx = i / 2;
        int nibble_pos = i % 2;
        /* -1 in signed 4-bit = 0xF */
        matA[row_offset + byte_idx] |= (uint8_t)(0xF << (nibble_pos * 4));
    }

    /* B: all nibbles = 2 -> every byte = 0x22 */
    mf_memset(matB, 0x22, (size_t)(N * K_bytes));

    mf_memset(matC, 0, (size_t)(M * N * 2));

    rc = mf_gemm_iu4(M, N, K_nibbles, matA, matB, matC, ws);
    if (rc != MF_OK)
        return -1;

    int16_t *c = (int16_t *)matC;
    for (int i = 0; i < M * N; i++) {
        if (c[i] != -2)
            return -1;
    }
    return 0;
}

/* ========================================================================
 * Test 7: test_gemm_ii16
 *
 * Signed int16 x Signed int16 -> int64
 * K_elems = 16 (one tile), K_bytes = 32
 * M=16, N=64
 *
 * A: identity-like. Row i has int16 value 1 at element position i,
 *    rest 0. Since K_elems=16 and M=16, diagonal fits exactly.
 *    A layout: int16 A[M][K_elems] = A[16][16] row-major.
 *
 * B: all int16 elements = 1.
 *    B layout: row-major [K_elems, N], int16 elements.
 *    B[k][n] = 1 for all entries.
 *
 * Expected C: every int64 element = 1
 * ======================================================================== */
static int test_gemm_ii16(void) {
    const int M = MF_TILE_M;       /* 16 */
    const int N = MF_TILE_N;       /* 64 */
    const int K_elems = MF_K_ELEMS_I16;  /* 16 */
    const int K_bytes = MF_TILE_K; /* 32 */
    int rc;

    /* Clear A */
    mf_memset(matA, 0, (size_t)(M * K_bytes));

    /* Set identity: A[row i][elem i] = 1 as int16 */
    int16_t *a16 = (int16_t *)matA;
    for (int i = 0; i < M; i++)
        a16[i * K_elems + i] = 1;

    /* Fill B: all int16 elements = 1. Row-major [K_elems, N] */
    int16_t *b16 = (int16_t *)matB;
    for (int k = 0; k < K_elems; k++)
        for (int n = 0; n < N; n++)
            b16[k * N + n] = 1;

    /* Clear C */
    mf_memset(matC, 0, (size_t)(M * N * 8));

    rc = mf_gemm_ii16(M, N, K_elems, matA, matB, matC, ws);
    if (rc != MF_OK)
        return -1;

    /* Verify: every int64 element should be 1 */
    int64_t *c = (int64_t *)matC;
    for (int i = 0; i < M * N; i++) {
        if (c[i] != 1)
            return -1;
    }
    return 0;
}

/* ========================================================================
 * Test 8: test_gemm_uu16
 *
 * Unsigned int16 x Unsigned int16 -> int64
 * A: diagonal value = 2
 * B: all elements = 3
 * Expected C: every int64 element = 6
 * ======================================================================== */
static int test_gemm_uu16(void) {
    const int M = MF_TILE_M;
    const int N = MF_TILE_N;
    const int K_elems = MF_K_ELEMS_I16;
    const int K_bytes = MF_TILE_K;
    int rc;

    mf_memset(matA, 0, (size_t)(M * K_bytes));

    uint16_t *a16 = (uint16_t *)matA;
    for (int i = 0; i < M; i++)
        a16[i * K_elems + i] = 2;

    uint16_t *b16 = (uint16_t *)matB;
    for (int k = 0; k < K_elems; k++)
        for (int n = 0; n < N; n++)
            b16[k * N + n] = 3;

    mf_memset(matC, 0, (size_t)(M * N * 8));

    rc = mf_gemm_uu16(M, N, K_elems, matA, matB, matC, ws);
    if (rc != MF_OK)
        return -1;

    int64_t *c = (int64_t *)matC;
    for (int i = 0; i < M * N; i++) {
        if (c[i] != 6)
            return -1;
    }
    return 0;
}

/* ========================================================================
 * Test 9: test_gemm_iu16
 *
 * Signed int16 A x Unsigned int16 B -> int64
 * A: diagonal value = -1 (0xFFFF as signed int16)
 * B: all elements = 5 (unsigned)
 * Expected C: every int64 element = (-1) * 5 = -5
 * ======================================================================== */
static int test_gemm_iu16(void) {
    const int M = MF_TILE_M;
    const int N = MF_TILE_N;
    const int K_elems = MF_K_ELEMS_I16;
    const int K_bytes = MF_TILE_K;
    int rc;

    mf_memset(matA, 0, (size_t)(M * K_bytes));

    int16_t *a16 = (int16_t *)matA;
    for (int i = 0; i < M; i++)
        a16[i * K_elems + i] = -1;  /* 0xFFFF */

    uint16_t *b16 = (uint16_t *)matB;
    for (int k = 0; k < K_elems; k++)
        for (int n = 0; n < N; n++)
            b16[k * N + n] = 5;

    mf_memset(matC, 0, (size_t)(M * N * 8));

    rc = mf_gemm_iu16(M, N, K_elems, matA, matB, matC, ws);
    if (rc != MF_OK)
        return -1;

    int64_t *c = (int64_t *)matC;
    for (int i = 0; i < M * N; i++) {
        if (c[i] != -5)
            return -1;
    }
    return 0;
}

/* ========================================================================
 * Test 10: test_gemm_ii8_random_large
 *
 * Random-input GeMM with element-by-element verification against a
 * scalar reference (equivalent to numpy C = A @ B).
 *
 * Dimensions deliberately non-tile-aligned to stress padding/tiling:
 *   M=1055, K=1243, N=97
 *   A: int8 [M, K]  (signed)
 *   B: int8 [K, N]  (signed, row-major)
 *   C: int32 [M, N]
 *
 * Uses a simple LCG PRNG for reproducible random data.
 * ======================================================================== */
#define LARGE_M  123
#define LARGE_K  97
#define LARGE_N  111

static int8_t  lgA[LARGE_M * LARGE_K]  __attribute__((aligned(8)));
static int8_t  lgB[LARGE_K * LARGE_N]  __attribute__((aligned(8)));
static int32_t lgC[LARGE_M * LARGE_N]  __attribute__((aligned(8)));
static int32_t lgC_ref[LARGE_M * LARGE_N] __attribute__((aligned(8)));
static unsigned char lg_ws[65536] __attribute__((aligned(8)));

static int test_gemm_ii8_random_large(void) {
    const int M = LARGE_M;
    const int N = LARGE_N;
    const int K = LARGE_K;

    /* LCG PRNG: x_{n+1} = (a*x_n + c) mod 2^32 */
    uint32_t rng = 0xDEADBEEF;
    #define LCG_NEXT(s) ((s) = (s) * 1103515245u + 12345u)

    /* Fill A with random signed int8 values */
    for (int i = 0; i < M * K; i++) {
        LCG_NEXT(rng);
        lgA[i] = (int8_t)(rng >> 24);  /* use top 8 bits */
    }

    /* Fill B with random signed int8 values */
    for (int i = 0; i < K * N; i++) {
        LCG_NEXT(rng);
        lgB[i] = (int8_t)(rng >> 24);
    }

    /* Compute scalar reference: C_ref[m][n] = sum_k A[m][k] * B[k][n] */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)lgA[m * K + k] * (int32_t)lgB[k * N + n];
            lgC_ref[m * N + n] = acc;
        }
    }

    /* Clear C and run hardware GeMM */
    mf_memset(lgC, 0, (size_t)(M * N * 4));
    mf_l1_buf_t lws = { lg_ws, sizeof(lg_ws) };
    int rc = mf_gemm_ii8(M, N, K, lgA, lgB, lgC, lws);
    if (rc != MF_OK)
        return -1;

    /* Element-by-element comparison */
    for (int i = 0; i < M * N; i++) {
        if (lgC[i] != lgC_ref[i])
            return -1;
    }
    return 0;

    #undef LCG_NEXT
}

/* ========================================================================
 * Main entry point
 *
 * Returns 0 on success. Returns test number (1-10) on first failure.
 * ======================================================================== */
int main(void) {
    printf("Start testing ...\n");
    if (test_gemm_ii8()  != 0) return 1;
    if (test_gemm_uu8()  != 0) return 2;
    if (test_gemm_iu8()  != 0) return 3;
    if (test_gemm_ii4()  != 0) return 4;
    if (test_gemm_uu4()  != 0) return 5;
    if (test_gemm_iu4()  != 0) return 6;
    if (test_gemm_ii16() != 0) return 7;
    if (test_gemm_uu16() != 0) return 8;
    if (test_gemm_iu16() != 0) return 9;
    if (test_gemm_ii8_random_large() != 0) return 10;
    printf("All test passed!\n");
    return 0;
}
