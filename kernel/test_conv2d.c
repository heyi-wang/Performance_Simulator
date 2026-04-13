/*
 * test_conv2d.c - Bare-metal tests for MatrixFlow Conv2D kernels
 *
 * Tests mf_conv2d.h which provides Conv2D operations (im2col + GeMM based).
 * Designed for spike RISC-V simulator, no libc dependencies.
 *
 * Returns 0 on success, non-zero on failure:
 *   1 = test_conv2d_1x1_i8 failed
 *   2 = test_conv2d_3x3_i8 failed
 *   3 = test_conv2d_return_check failed
 */

/* ============================================================================
 * Bare-metal libc stubs
 * ============================================================================ */
typedef __SIZE_TYPE__ size_t;

void *memset(void *dst, int val, size_t n) {
    unsigned char *p = (unsigned char *)dst;
    for (size_t i = 0; i < n; i++)
        p[i] = (unsigned char)val;
    return dst;
}

void *memcpy(void *dst, const void *src, size_t n) {
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    for (size_t i = 0; i < n; i++)
        d[i] = s[i];
    return dst;
}

/* ============================================================================
 * Kernel headers
 * ============================================================================ */
#include "mf_conv2d.h"

/* ============================================================================
 * Shared workspace and test buffers
 * ============================================================================ */
static unsigned char work[65536] __attribute__((aligned(8)));
static mf_l1_buf_t ws = { work, sizeof(work) };

/* ============================================================================
 * Test 1: 1x1 Pointwise Convolution (int8)
 *
 * C_in=32, C_out=16, H=1, W=1 (single pixel, simplest case)
 *
 * Input: all 1s  [C_in x 1 x 1] = 32 bytes, each = 1
 * Weight: [C_out x C_in] row-major
 *   For each output channel co:
 *     weight[co][co] = 1, weight[co][k] = 0 for k != co
 *   This maps output[co] = input[co] = 1 for co < C_out (=16)
 *
 * Expected: each output element should be 1
 *   output[co] = sum_{ci=0}^{31} weight[co][ci] * input[ci]
 *              = weight[co][co] * 1 = 1  (since co < 16 <= 32)
 * ============================================================================ */
#define T1_CIN  32
#define T1_COUT 16
#define T1_H    1
#define T1_W    1

static int8_t  t1_input[T1_CIN * T1_H * T1_W] __attribute__((aligned(8)));
static int8_t  t1_weight[T1_COUT * T1_CIN] __attribute__((aligned(8)));
static int32_t t1_output[T1_COUT * T1_H * T1_W] __attribute__((aligned(8)));

static int test_conv2d_1x1_i8(void) {
    /* Fill input: all 1s */
    for (int i = 0; i < T1_CIN * T1_H * T1_W; i++)
        t1_input[i] = 1;

    /* Fill weight: identity-like [C_out x C_in]
     * weight[co][ci] = 1 if co == ci, else 0
     * This picks out the first 16 channels of the input */
    mf_memset(t1_weight, 0, sizeof(t1_weight));
    for (int co = 0; co < T1_COUT; co++)
        t1_weight[co * T1_CIN + co] = 1;

    /* Clear output */
    mf_memset(t1_output, 0, sizeof(t1_output));

    /* Run 1x1 convolution */
    int rc = mf_conv2d_1x1_i8(t1_input, t1_weight, t1_output,
                                T1_CIN, T1_COUT, T1_H, T1_W, ws);
    if (rc != MF_OK)
        return 1;

    /* Verify: each output channel should be 1 */
    for (int co = 0; co < T1_COUT; co++) {
        if (t1_output[co] != 1)
            return 1;
    }

    return 0;
}

/* ============================================================================
 * Test 2: 3x3 Convolution (int8)
 *
 * C_in=1, C_out=1, H=4, W=4, pad=1
 *
 * Input: all 1s  [1 x 4 x 4]
 * Weight: all 1s [C_out x C_in*9] row-major = [1 x 9], each = 1
 *
 * With pad=1, outH = 4+2*1-3+1 = 4, outW = 4
 * Output is [1 x 4 x 4] int32
 *
 * Expected output (3x3 convolution of all-ones with all-ones kernel):
 *   Corners (0,0), (0,3), (3,0), (3,3): 4 (2x2 overlap with padded input)
 *   Edges (non-corner border): 6 (2x3 or 3x2 overlap)
 *   Center (1,1), (1,2), (2,1), (2,2): 9 (full 3x3 overlap)
 *
 * We verify output has reasonable non-zero values. Exact values depend on
 * im2col + GeMM tiling, so we check known convolution results.
 * ============================================================================ */
#define T2_CIN  1
#define T2_COUT 1
#define T2_H    4
#define T2_W    4
#define T2_PAD  1
#define T2_OUTH (T2_H + 2 * T2_PAD - 3 + 1)  /* 4 */
#define T2_OUTW (T2_W + 2 * T2_PAD - 3 + 1)  /* 4 */

static int8_t  t2_input[T2_CIN * T2_H * T2_W] __attribute__((aligned(8)));
static int8_t  t2_weight[T2_COUT * T2_CIN * 9] __attribute__((aligned(8)));
static int32_t t2_output[T2_COUT * T2_OUTH * T2_OUTW] __attribute__((aligned(8)));

/* Expected output for 3x3 all-ones convolution on 4x4 all-ones with pad=1:
 *   4 6 6 4
 *   6 9 9 6
 *   6 9 9 6
 *   4 6 6 4
 */
static const int32_t t2_expected[T2_OUTH * T2_OUTW] = {
    4, 6, 6, 4,
    6, 9, 9, 6,
    6, 9, 9, 6,
    4, 6, 6, 4
};

static int test_conv2d_3x3_i8(void) {
    /* Fill input: all 1s */
    for (int i = 0; i < T2_CIN * T2_H * T2_W; i++)
        t2_input[i] = 1;

    /* Fill weight: all 1s [C_out x C_in*9] = [1 x 9] */
    for (int i = 0; i < T2_COUT * T2_CIN * 9; i++)
        t2_weight[i] = 1;

    /* Clear output */
    mf_memset(t2_output, 0, sizeof(t2_output));

    /* Run 3x3 convolution */
    int rc = mf_conv2d_3x3_i8(t2_input, t2_weight, t2_output,
                                T2_CIN, T2_COUT, T2_H, T2_W, T2_PAD, ws);
    if (rc != MF_OK)
        return 1;

    /* Verify output against expected values */
    for (int i = 0; i < T2_OUTH * T2_OUTW; i++) {
        if (t2_output[i] != t2_expected[i])
            return 1;
    }

    return 0;
}

/* ============================================================================
 * Test 3: Error return check
 *
 * Pass a tiny workspace that is too small for any convolution.
 * The function should return MF_ERR_L1_TOO_SMALL.
 * ============================================================================ */
static unsigned char tiny_work[16] __attribute__((aligned(8)));

static int test_conv2d_return_check(void) {
    /* Tiny workspace: only 16 bytes, way too small */
    mf_l1_buf_t tiny_ws = { tiny_work, sizeof(tiny_work) };

    /* Minimal input/weight/output arrays (just enough to not segfault) */
    static int8_t  rc_input[32] __attribute__((aligned(8)));
    static int8_t  rc_weight[32 * 16] __attribute__((aligned(8)));
    static int32_t rc_output[16] __attribute__((aligned(8)));

    mf_memset(rc_input, 1, sizeof(rc_input));
    mf_memset(rc_weight, 1, sizeof(rc_weight));
    mf_memset(rc_output, 0, sizeof(rc_output));

    /* 1x1 conv with C_in=32, C_out=16, H=1, W=1 needs workspace for
     * transpose buffer (32 bytes) + gemm workspace. 16 bytes is too small. */
    int rc = mf_conv2d_1x1_i8(rc_input, rc_weight, rc_output,
                                32, 16, 1, 1, tiny_ws);
    if (rc != MF_ERR_L1_TOO_SMALL)
        return 1;

    /* 3x3 conv: even more workspace needed for im2col */
    static int8_t  rc3_input[4 * 4] __attribute__((aligned(8)));
    static int8_t  rc3_weight[9] __attribute__((aligned(8)));
    static int32_t rc3_output[4 * 4] __attribute__((aligned(8)));

    mf_memset(rc3_input, 1, sizeof(rc3_input));
    mf_memset(rc3_weight, 1, sizeof(rc3_weight));
    mf_memset(rc3_output, 0, sizeof(rc3_output));

    rc = mf_conv2d_3x3_i8(rc3_input, rc3_weight, rc3_output,
                            1, 1, 4, 4, 1, tiny_ws);
    if (rc != MF_ERR_L1_TOO_SMALL)
        return 1;

    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */
int main(void) {
    if (test_conv2d_1x1_i8() != 0)     return 1;
    if (test_conv2d_3x3_i8() != 0)     return 2;
    if (test_conv2d_return_check() != 0) return 3;
    return 0;
}
