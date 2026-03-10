/*
 * nafnet_inference.c - Self-contained NAFNet INT8 inference
 *
 * Pure integer arithmetic. No floating point. Identical output on any platform.
 *
 * Processes 10 embedded 64x64 image patches through NAFNet-32.
 * Outputs hex-encoded pixels and PSNR values via printf.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "generated/nafnet_config.h"
#include "generated/nafnet_weights.h"
#include "generated/nafnet_scales.h"
#include "generated/nafnet_images.h"
#include "nafnet_c_ops.h"
#include "nafnet_c_ops_parallel.h"
#include "conv_tiling/src/thread_pool.h"

static inline uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* ============================================================
 * Working buffers (static allocation)
 *
 * Memory budget for 64x64 patches, NAFNet-32:
 * - Two ping-pong int8 buffers for activations
 * - Skip connection storage for encoder outputs
 * - One int32 accumulator buffer
 * ============================================================ */

/* Max activation size: 512ch * 64 * 64 at encoder level 0
 * But after downsampling: 512ch * 4 * 4 at bottleneck
 * The largest is at level 0: 32ch * 64 * 64 = 131072
 * But expanded channels (dw_expand=2): 64ch * 64 * 64 = 262144
 * Actually the largest expanded is at bottleneck before downsample:
 * Level 0: 32ch * 64 * 64, expanded: 64ch * 64 * 64 = 262144
 * Level 1: 64ch * 32 * 32, expanded: 128ch * 32 * 32 = 131072
 * etc.
 * Max is 262144 bytes for int8 activations.
 *
 * For the upsample path, the conv1x1 expands channels by 4x:
 * ups.0: 512 -> 1024 channels at 4x4 = 16384 (small)
 * ups.3: 64 -> 128 channels at 32x32 = 131072
 * After pixelshuffle: 32 * 64 * 64 = 131072
 *
 * So 512KB per buffer is more than enough.
 */
#define BUF_SIZE (512 * 1024 * 8 * 8)
#define ACC_SIZE (512 * 1024 * 8 * 8)  /* int32 accumulator: C_out * H * W * 4 bytes */
#define SKIP_SIZE (256 * 1024 * 8 * 8) /* per-level skip storage */

static int8_t buf_a[BUF_SIZE];
static int8_t buf_b[BUF_SIZE];
static int8_t buf_c[BUF_SIZE]; /* extra buffer for intermediate results */
static int32_t acc_buf[ACC_SIZE];

/* Skip connections: one per encoder level */
static int8_t skip_0[32 * 512 * 512];  /* 131072 */
static int8_t skip_1[64 * 256 * 256];  /*  65536 */
static int8_t skip_2[128 * 128 * 128]; /*  32768 */
static int8_t skip_3[256 * 64 * 64];   /*  16384 */

/* Output image buffer */
static uint8_t output_img[3 * 512 * 512];
static uint8_t output_img_para[3 * 512 * 512];

/* ============================================================
 * NAFBlock: core building block
 *
 * Branch 1: norm1 -> conv1(expand) -> dw_conv2 -> simplegate -> sca -> conv3 -> + x*beta
 * Branch 2: norm2 -> conv4(expand) -> simplegate -> conv5 -> + x*gamma
 * ============================================================ */

/* Macro to get the right weight/bias/scale arrays for a NAFBlock */
/* We pass everything as function parameters to keep it generic */

static void nafblock(
    int8_t *x, /* [C, H, W] input/output (in-place) */
    int C, int H, int W,
    /* Branch 1 */
    const int32_t *norm1_gamma, const int8_t *norm1_beta, int norm1_frac,
    const int8_t *conv1_w, const int32_t *conv1_b, int32_t conv1_M, int conv1_shift,
    const int8_t *conv2_w, const int32_t *conv2_b, int32_t conv2_M, int conv2_shift,
    int32_t sg1_M, int sg1_shift,
    const int8_t *sca_conv_w, const int32_t *sca_conv_b,
    int32_t sca_conv_M, int sca_conv_shift,
    int32_t sca_mul_M, int sca_mul_shift,
    const int8_t *conv3_w, const int32_t *conv3_b, int32_t conv3_M, int conv3_shift,
    const int16_t *beta_param, int beta_frac,
    /* Branch 2 */
    const int32_t *norm2_gamma, const int8_t *norm2_beta, int norm2_frac,
    const int8_t *conv4_w, const int32_t *conv4_b, int32_t conv4_M, int conv4_shift,
    int32_t sg2_M, int sg2_shift,
    const int8_t *conv5_w, const int32_t *conv5_b, int32_t conv5_M, int conv5_shift,
    const int16_t *gamma_param, int gamma_frac)
{
    int spatial = H * W;
    int n = C * spatial;
    int dw_channel = C * 2;  /* dw_expand = 2 */
    int ffn_channel = C * 2; /* ffn_expand = 2 */

    /* CRITICAL: Save block input to buf_c.
     * x may alias buf_a, which gets overwritten by intermediate ops.
     * buf_c holds the residual base for both branch 1 and branch 2. */
    memcpy(buf_c, x, n);

    /* ---- Branch 1 ---- */

    /* norm1(x) -> buf_a */
    layernorm2d_i8(x, buf_a, norm1_gamma, norm1_beta, C, H, W, norm1_frac);

    /* conv1: 1x1, C -> 2C, buf_a -> buf_b */
    conv1x1_bias_requant(buf_a, conv1_w, conv1_b, buf_b, acc_buf,
                         C, dw_channel, H, W, conv1_M, conv1_shift);

    /* conv2: 3x3 depthwise, 2C -> 2C, buf_b -> buf_a */
    dw_conv3x3_bias_requant(buf_b, conv2_w, conv2_b, buf_a, acc_buf,
                            dw_channel, H, W, conv2_M, conv2_shift);

    /* simplegate: 2C -> C, buf_a -> buf_b */
    simplegate_i8(buf_a, buf_b, C, H, W, sg1_M, sg1_shift);

    /* sca: C -> C, buf_b -> buf_a */
    sca_i8(buf_b, buf_a, sca_conv_w, sca_conv_b, C, H, W,
           sca_conv_M, sca_conv_shift, sca_mul_M, sca_mul_shift);

    /* conv3: 1x1, C -> C, buf_a -> buf_b */
    conv1x1_bias_requant(buf_a, conv3_w, conv3_b, buf_b, acc_buf,
                         C, C, H, W, conv3_M, conv3_shift);

    /* residual: buf_c = saved_input + buf_b * beta */
    residual_scale_add_i8(buf_c, buf_b, beta_param, buf_c, C, H, W, beta_frac);

    /* ---- Branch 2 ---- */

    /* norm2(buf_c) -> buf_a  (buf_c = branch 1 result) */
    layernorm2d_i8(buf_c, buf_a, norm2_gamma, norm2_beta, C, H, W, norm2_frac);

    /* conv4: 1x1, C -> 2C, buf_a -> buf_b */
    conv1x1_bias_requant(buf_a, conv4_w, conv4_b, buf_b, acc_buf,
                         C, ffn_channel, H, W, conv4_M, conv4_shift);

    /* simplegate: 2C -> C, buf_b -> buf_a */
    simplegate_i8(buf_b, buf_a, C, H, W, sg2_M, sg2_shift);

    /* conv5: 1x1, C -> C, buf_a -> buf_b */
    conv1x1_bias_requant(buf_a, conv5_w, conv5_b, buf_b, acc_buf,
                         C, C, H, W, conv5_M, conv5_shift);

    /* residual: x = branch1_result + buf_b * gamma */
    residual_scale_add_i8(buf_c, buf_b, gamma_param, x, C, H, W, gamma_frac);
}

static void nafblock_para(
    int8_t *x, /* [C, H, W] input/output (in-place) */
    int C, int H, int W,
    /* Branch 1 */
    const int32_t *norm1_gamma, const int8_t *norm1_beta, int norm1_frac,
    const int8_t *conv1_w, const int32_t *conv1_b, int32_t conv1_M, int conv1_shift,
    const int8_t *conv2_w, const int32_t *conv2_b, int32_t conv2_M, int conv2_shift,
    int32_t sg1_M, int sg1_shift,
    const int8_t *sca_conv_w, const int32_t *sca_conv_b,
    int32_t sca_conv_M, int sca_conv_shift,
    int32_t sca_mul_M, int sca_mul_shift,
    const int8_t *conv3_w, const int32_t *conv3_b, int32_t conv3_M, int conv3_shift,
    const int16_t *beta_param, int beta_frac,
    /* Branch 2 */
    const int32_t *norm2_gamma, const int8_t *norm2_beta, int norm2_frac,
    const int8_t *conv4_w, const int32_t *conv4_b, int32_t conv4_M, int conv4_shift,
    int32_t sg2_M, int sg2_shift,
    const int8_t *conv5_w, const int32_t *conv5_b, int32_t conv5_M, int conv5_shift,
    const int16_t *gamma_param, int gamma_frac, void *pool)
{

    int spatial = H * W;
    int n = C * spatial;
    int dw_channel = C * 2;  /* dw_expand = 2 */
    int ffn_channel = C * 2; /* ffn_expand = 2 */

    /* CRITICAL: Save block input to buf_c.
     * x may alias buf_a, which gets overwritten by intermediate ops.
     * buf_c holds the residual base for both branch 1 and branch 2. */
    memcpy(buf_c, x, n);

    /* ---- Branch 1 ---- */

    /* norm1(x) -> buf_a */
    layernorm2d_i8(x, buf_a, norm1_gamma, norm1_beta, C, H, W, norm1_frac);

    /* conv1: 1x1, C -> 2C, buf_a -> buf_b */
    conv1x1_bias_requant_para(buf_a, conv1_w, conv1_b, buf_b, acc_buf,
                              C, dw_channel, H, W, conv1_M, conv1_shift, &pool);

    /* conv2: 3x3 depthwise, 2C -> 2C, buf_b -> buf_a */
    dw_conv3x3_bias_requant(buf_b, conv2_w, conv2_b, buf_a, acc_buf,
                            dw_channel, H, W, conv2_M, conv2_shift);

    /* simplegate: 2C -> C, buf_a -> buf_b */
    simplegate_i8(buf_a, buf_b, C, H, W, sg1_M, sg1_shift);

    /* sca: C -> C, buf_b -> buf_a */
    sca_i8(buf_b, buf_a, sca_conv_w, sca_conv_b, C, H, W,
           sca_conv_M, sca_conv_shift, sca_mul_M, sca_mul_shift);

    /* conv3: 1x1, C -> C, buf_a -> buf_b */
    conv1x1_bias_requant_para(buf_a, conv3_w, conv3_b, buf_b, acc_buf,
                              C, C, H, W, conv3_M, conv3_shift, &pool);

    /* residual: buf_c = saved_input + buf_b * beta */
    residual_scale_add_i8(buf_c, buf_b, beta_param, buf_c, C, H, W, beta_frac);

    /* ---- Branch 2 ---- */

    /* norm2(buf_c) -> buf_a  (buf_c = branch 1 result) */
    layernorm2d_i8(buf_c, buf_a, norm2_gamma, norm2_beta, C, H, W, norm2_frac);

    /* conv4: 1x1, C -> 2C, buf_a -> buf_b */
    conv1x1_bias_requant_para(buf_a, conv4_w, conv4_b, buf_b, acc_buf,
                              C, ffn_channel, H, W, conv4_M, conv4_shift, &pool);

    /* simplegate: 2C -> C, buf_b -> buf_a */
    simplegate_i8(buf_b, buf_a, C, H, W, sg2_M, sg2_shift);

    /* conv5: 1x1, C -> C, buf_a -> buf_b */
    conv1x1_bias_requant_para(buf_a, conv5_w, conv5_b, buf_b, acc_buf,
                              C, C, H, W, conv5_M, conv5_shift, &pool);

    /* residual: x = branch1_result + buf_b * gamma */
    residual_scale_add_i8(buf_c, buf_b, gamma_param, x, C, H, W, gamma_frac);
}

/* Debug helper: print buffer stats */
static int dbg_block_count = 0;
static void print_stats(const char *label, const int8_t *data, int n)
{
    int32_t mn = 127, mx = -128;
    int64_t sum = 0;
    int sat_lo = 0, sat_hi = 0;
    for (int i = 0; i < n; i++)
    {
        int8_t v = data[i];
        if (v < mn)
            mn = v;
        if (v > mx)
            mx = v;
        sum += v;
        if (v == -128)
            sat_lo++;
        if (v == 127)
            sat_hi++;
    }
    printf("    %s: min=%d max=%d mean=%d sat_lo=%d sat_hi=%d\n",
           label, (int)mn, (int)mx, (int)(sum / n), sat_lo, sat_hi);
}

/* Debug version of nafblock - prints intermediate stats for first N calls */
static void nafblock_debug(
    int8_t *x, int C, int H, int W,
    const int32_t *norm1_gamma, const int8_t *norm1_beta, int norm1_frac,
    const int8_t *conv1_w, const int32_t *conv1_b, int32_t conv1_M, int conv1_shift,
    const int8_t *conv2_w, const int32_t *conv2_b, int32_t conv2_M, int conv2_shift,
    int32_t sg1_M, int sg1_shift,
    const int8_t *sca_conv_w, const int32_t *sca_conv_b,
    int32_t sca_conv_M, int sca_conv_shift,
    int32_t sca_mul_M, int sca_mul_shift,
    const int8_t *conv3_w, const int32_t *conv3_b, int32_t conv3_M, int conv3_shift,
    const int16_t *beta_param, int beta_frac,
    const int32_t *norm2_gamma, const int8_t *norm2_beta, int norm2_frac,
    const int8_t *conv4_w, const int32_t *conv4_b, int32_t conv4_M, int conv4_shift,
    int32_t sg2_M, int sg2_shift,
    const int8_t *conv5_w, const int32_t *conv5_b, int32_t conv5_M, int conv5_shift,
    const int16_t *gamma_param, int gamma_frac)
{
    int spatial = H * W;
    int n = C * spatial;
    int dw_channel = C * 2;
    int ffn_channel = C * 2;
    int debug = (dbg_block_count < 2);

    if (debug)
        printf("  === Block %d: C=%d H=%d W=%d ===\n", dbg_block_count, C, H, W);
    if (debug)
        print_stats("input", x, n);

    /* CRITICAL: Save block input to buf_c for residual connections */
    memcpy(buf_c, x, n);

    layernorm2d_i8(x, buf_a, norm1_gamma, norm1_beta, C, H, W, norm1_frac);
    if (debug)
        print_stats("norm1", buf_a, C * spatial);

    conv1x1_bias_requant(buf_a, conv1_w, conv1_b, buf_b, acc_buf,
                         C, dw_channel, H, W, conv1_M, conv1_shift);
    if (debug)
    {
        print_stats("conv1", buf_b, dw_channel * spatial);
        printf("      conv1 M=%d shift=%d\n", (int)conv1_M, conv1_shift);
    }

    dw_conv3x3_bias_requant(buf_b, conv2_w, conv2_b, buf_a, acc_buf,
                            dw_channel, H, W, conv2_M, conv2_shift);
    if (debug)
        print_stats("conv2", buf_a, dw_channel * spatial);

    simplegate_i8(buf_a, buf_b, C, H, W, sg1_M, sg1_shift);
    if (debug)
    {
        print_stats("sg1", buf_b, C * spatial);
        printf("      sg1 M=%d shift=%d\n", (int)sg1_M, sg1_shift);
    }

    sca_i8(buf_b, buf_a, sca_conv_w, sca_conv_b, C, H, W,
           sca_conv_M, sca_conv_shift, sca_mul_M, sca_mul_shift);
    if (debug)
        print_stats("sca", buf_a, C * spatial);

    conv1x1_bias_requant(buf_a, conv3_w, conv3_b, buf_b, acc_buf,
                         C, C, H, W, conv3_M, conv3_shift);
    if (debug)
        print_stats("conv3", buf_b, C * spatial);

    if (debug)
    {
        int16_t bmin = 32767, bmax = -32768;
        for (int c = 0; c < C; c++)
        {
            if (beta_param[c] < bmin)
                bmin = beta_param[c];
            if (beta_param[c] > bmax)
                bmax = beta_param[c];
        }
        printf("      beta params: min=%d max=%d frac=%d\n", (int)bmin, (int)bmax, beta_frac);
    }

    /* Residual: buf_c = saved_input + conv3*beta */
    residual_scale_add_i8(buf_c, buf_b, beta_param, buf_c, C, H, W, beta_frac);
    if (debug)
        print_stats("after_beta", buf_c, n);

    /* Branch 2: norm2 reads from buf_c (branch 1 result) */
    layernorm2d_i8(buf_c, buf_a, norm2_gamma, norm2_beta, C, H, W, norm2_frac);
    if (debug)
        print_stats("norm2", buf_a, C * spatial);

    conv1x1_bias_requant(buf_a, conv4_w, conv4_b, buf_b, acc_buf,
                         C, ffn_channel, H, W, conv4_M, conv4_shift);
    if (debug)
        print_stats("conv4", buf_b, ffn_channel * spatial);

    simplegate_i8(buf_b, buf_a, C, H, W, sg2_M, sg2_shift);
    if (debug)
        print_stats("sg2", buf_a, C * spatial);

    conv1x1_bias_requant(buf_a, conv5_w, conv5_b, buf_b, acc_buf,
                         C, C, H, W, conv5_M, conv5_shift);
    if (debug)
        print_stats("conv5", buf_b, C * spatial);

    if (debug)
    {
        int16_t gmin = 32767, gmax = -32768;
        for (int c = 0; c < C; c++)
        {
            if (gamma_param[c] < gmin)
                gmin = gamma_param[c];
            if (gamma_param[c] > gmax)
                gmax = gamma_param[c];
        }
        printf("      gamma params: min=%d max=%d frac=%d\n", (int)gmin, (int)gmax, gamma_frac);
    }

    /* Residual: x = branch1_result + conv5*gamma */
    residual_scale_add_i8(buf_c, buf_b, gamma_param, x, C, H, W, gamma_frac);
    if (debug)
        print_stats("after_gamma", x, n);

    dbg_block_count++;
}

/* ============================================================
 * Helper macros to call nafblock with the generated weight names
 *
 * This avoids writing out 30+ arguments for each of the 28+12=40 blocks.
 * The naming convention follows the PyTorch state_dict keys.
 * ============================================================ */

#define NAFBLOCK_CALL(prefix, x_ptr, ch, h, w)                             \
    nafblock(x_ptr, ch, h, w,                                              \
             prefix##_norm1_gamma, prefix##_norm1_beta,                    \
             prefix##_conv1_weight, prefix##_conv1_bias,                   \
             prefix##_CONV1_REQUANT_M, prefix##_CONV1_REQUANT_SHIFT,       \
             prefix##_conv2_weight, prefix##_conv2_bias,                   \
             prefix##_CONV2_REQUANT_M, prefix##_CONV2_REQUANT_SHIFT,       \
             prefix##_SG1_REQUANT_M, prefix##_SG1_REQUANT_SHIFT,           \
             prefix##_sca_conv_weight, prefix##_sca_conv_bias,             \
             prefix##_SCA_CONV_REQUANT_M, prefix##_SCA_CONV_REQUANT_SHIFT, \
             prefix##_SCA_MUL_REQUANT_M, prefix##_SCA_MUL_REQUANT_SHIFT,   \
             prefix##_conv3_weight, prefix##_conv3_bias,                   \
             prefix##_CONV3_REQUANT_M, prefix##_CONV3_REQUANT_SHIFT,       \
             prefix##_beta,                                                \
             prefix##_norm2_gamma, prefix##_norm2_beta,                    \
             prefix##_conv4_weight, prefix##_conv4_bias,                   \
             prefix##_CONV4_REQUANT_M, prefix##_CONV4_REQUANT_SHIFT,       \
             prefix##_SG2_REQUANT_M, prefix##_SG2_REQUANT_SHIFT,           \
             prefix##_conv5_weight, prefix##_conv5_bias,                   \
             prefix##_CONV5_REQUANT_M, prefix##_CONV5_REQUANT_SHIFT,       \
             prefix##_gamma)

#define NAFBLOCK_CALL_PARA(prefix, x_ptr, ch, h, w, pool)                       \
    nafblock_para(x_ptr, ch, h, w,                                              \
                  prefix##_norm1_gamma, prefix##_norm1_beta,                    \
                  prefix##_conv1_weight, prefix##_conv1_bias,                   \
                  prefix##_CONV1_REQUANT_M, prefix##_CONV1_REQUANT_SHIFT,       \
                  prefix##_conv2_weight, prefix##_conv2_bias,                   \
                  prefix##_CONV2_REQUANT_M, prefix##_CONV2_REQUANT_SHIFT,       \
                  prefix##_SG1_REQUANT_M, prefix##_SG1_REQUANT_SHIFT,           \
                  prefix##_sca_conv_weight, prefix##_sca_conv_bias,             \
                  prefix##_SCA_CONV_REQUANT_M, prefix##_SCA_CONV_REQUANT_SHIFT, \
                  prefix##_SCA_MUL_REQUANT_M, prefix##_SCA_MUL_REQUANT_SHIFT,   \
                  prefix##_conv3_weight, prefix##_conv3_bias,                   \
                  prefix##_CONV3_REQUANT_M, prefix##_CONV3_REQUANT_SHIFT,       \
                  prefix##_beta,                                                \
                  prefix##_norm2_gamma, prefix##_norm2_beta,                    \
                  prefix##_conv4_weight, prefix##_conv4_bias,                   \
                  prefix##_CONV4_REQUANT_M, prefix##_CONV4_REQUANT_SHIFT,       \
                  prefix##_SG2_REQUANT_M, prefix##_SG2_REQUANT_SHIFT,           \
                  prefix##_conv5_weight, prefix##_conv5_bias,                   \
                  prefix##_CONV5_REQUANT_M, prefix##_CONV5_REQUANT_SHIFT,       \
                  prefix##_gamma, pool)

/* Define uppercase macro names to bridge weight arrays (lowercase) and scale defines (uppercase) */
/* The weight arrays use names like encoders_0_0_conv1_weight */
/* The scale defines use names like ENCODERS_0_0_CONV1_REQUANT_M */
/* We need macros that resolve both forms from a single prefix */

/* For encoders.L.B -> prefix = encoders_L_B */
/* Scale param frac_bits is always 10 for beta/gamma */
#define SCALE_FRAC 10

#define CALL_ENC_BLOCK(L, B, x_ptr, ch, h, w)                                                      \
    nafblock(x_ptr, ch, h, w,                                                                      \
             encoders_##L##_##B##_norm1_gamma, encoders_##L##_##B##_norm1_beta,                    \
             ENCODERS_##L##_##B##_NORM1_FRAC_BITS,                                                 \
             encoders_##L##_##B##_conv1_weight, encoders_##L##_##B##_conv1_bias,                   \
             ENCODERS_##L##_##B##_CONV1_REQUANT_M, ENCODERS_##L##_##B##_CONV1_REQUANT_SHIFT,       \
             encoders_##L##_##B##_conv2_weight, encoders_##L##_##B##_conv2_bias,                   \
             ENCODERS_##L##_##B##_CONV2_REQUANT_M, ENCODERS_##L##_##B##_CONV2_REQUANT_SHIFT,       \
             ENCODERS_##L##_##B##_SG1_REQUANT_M, ENCODERS_##L##_##B##_SG1_REQUANT_SHIFT,           \
             encoders_##L##_##B##_sca_conv_weight, encoders_##L##_##B##_sca_conv_bias,             \
             ENCODERS_##L##_##B##_SCA_CONV_REQUANT_M, ENCODERS_##L##_##B##_SCA_CONV_REQUANT_SHIFT, \
             ENCODERS_##L##_##B##_SCA_MUL_REQUANT_M, ENCODERS_##L##_##B##_SCA_MUL_REQUANT_SHIFT,   \
             encoders_##L##_##B##_conv3_weight, encoders_##L##_##B##_conv3_bias,                   \
             ENCODERS_##L##_##B##_CONV3_REQUANT_M, ENCODERS_##L##_##B##_CONV3_REQUANT_SHIFT,       \
             encoders_##L##_##B##_beta, SCALE_FRAC,                                                \
             encoders_##L##_##B##_norm2_gamma, encoders_##L##_##B##_norm2_beta,                    \
             ENCODERS_##L##_##B##_NORM2_FRAC_BITS,                                                 \
             encoders_##L##_##B##_conv4_weight, encoders_##L##_##B##_conv4_bias,                   \
             ENCODERS_##L##_##B##_CONV4_REQUANT_M, ENCODERS_##L##_##B##_CONV4_REQUANT_SHIFT,       \
             ENCODERS_##L##_##B##_SG2_REQUANT_M, ENCODERS_##L##_##B##_SG2_REQUANT_SHIFT,           \
             encoders_##L##_##B##_conv5_weight, encoders_##L##_##B##_conv5_bias,                   \
             ENCODERS_##L##_##B##_CONV5_REQUANT_M, ENCODERS_##L##_##B##_CONV5_REQUANT_SHIFT,       \
             encoders_##L##_##B##_gamma, SCALE_FRAC)

#define CALL_ENC_BLOCK_PARA(L, B, x_ptr, ch, h, w, pool)                                                \
    nafblock_para(x_ptr, ch, h, w,                                                                      \
                  encoders_##L##_##B##_norm1_gamma, encoders_##L##_##B##_norm1_beta,                    \
                  ENCODERS_##L##_##B##_NORM1_FRAC_BITS,                                                 \
                  encoders_##L##_##B##_conv1_weight, encoders_##L##_##B##_conv1_bias,                   \
                  ENCODERS_##L##_##B##_CONV1_REQUANT_M, ENCODERS_##L##_##B##_CONV1_REQUANT_SHIFT,       \
                  encoders_##L##_##B##_conv2_weight, encoders_##L##_##B##_conv2_bias,                   \
                  ENCODERS_##L##_##B##_CONV2_REQUANT_M, ENCODERS_##L##_##B##_CONV2_REQUANT_SHIFT,       \
                  ENCODERS_##L##_##B##_SG1_REQUANT_M, ENCODERS_##L##_##B##_SG1_REQUANT_SHIFT,           \
                  encoders_##L##_##B##_sca_conv_weight, encoders_##L##_##B##_sca_conv_bias,             \
                  ENCODERS_##L##_##B##_SCA_CONV_REQUANT_M, ENCODERS_##L##_##B##_SCA_CONV_REQUANT_SHIFT, \
                  ENCODERS_##L##_##B##_SCA_MUL_REQUANT_M, ENCODERS_##L##_##B##_SCA_MUL_REQUANT_SHIFT,   \
                  encoders_##L##_##B##_conv3_weight, encoders_##L##_##B##_conv3_bias,                   \
                  ENCODERS_##L##_##B##_CONV3_REQUANT_M, ENCODERS_##L##_##B##_CONV3_REQUANT_SHIFT,       \
                  encoders_##L##_##B##_beta, SCALE_FRAC,                                                \
                  encoders_##L##_##B##_norm2_gamma, encoders_##L##_##B##_norm2_beta,                    \
                  ENCODERS_##L##_##B##_NORM2_FRAC_BITS,                                                 \
                  encoders_##L##_##B##_conv4_weight, encoders_##L##_##B##_conv4_bias,                   \
                  ENCODERS_##L##_##B##_CONV4_REQUANT_M, ENCODERS_##L##_##B##_CONV4_REQUANT_SHIFT,       \
                  ENCODERS_##L##_##B##_SG2_REQUANT_M, ENCODERS_##L##_##B##_SG2_REQUANT_SHIFT,           \
                  encoders_##L##_##B##_conv5_weight, encoders_##L##_##B##_conv5_bias,                   \
                  ENCODERS_##L##_##B##_CONV5_REQUANT_M, ENCODERS_##L##_##B##_CONV5_REQUANT_SHIFT,       \
                  encoders_##L##_##B##_gamma, SCALE_FRAC, pool)

#define CALL_MID_BLOCK(B, x_ptr, ch, h, w)                                                   \
    nafblock(x_ptr, ch, h, w,                                                                \
             middle_blks_##B##_norm1_gamma, middle_blks_##B##_norm1_beta,                    \
             MIDDLE_BLKS_##B##_NORM1_FRAC_BITS,                                              \
             middle_blks_##B##_conv1_weight, middle_blks_##B##_conv1_bias,                   \
             MIDDLE_BLKS_##B##_CONV1_REQUANT_M, MIDDLE_BLKS_##B##_CONV1_REQUANT_SHIFT,       \
             middle_blks_##B##_conv2_weight, middle_blks_##B##_conv2_bias,                   \
             MIDDLE_BLKS_##B##_CONV2_REQUANT_M, MIDDLE_BLKS_##B##_CONV2_REQUANT_SHIFT,       \
             MIDDLE_BLKS_##B##_SG1_REQUANT_M, MIDDLE_BLKS_##B##_SG1_REQUANT_SHIFT,           \
             middle_blks_##B##_sca_conv_weight, middle_blks_##B##_sca_conv_bias,             \
             MIDDLE_BLKS_##B##_SCA_CONV_REQUANT_M, MIDDLE_BLKS_##B##_SCA_CONV_REQUANT_SHIFT, \
             MIDDLE_BLKS_##B##_SCA_MUL_REQUANT_M, MIDDLE_BLKS_##B##_SCA_MUL_REQUANT_SHIFT,   \
             middle_blks_##B##_conv3_weight, middle_blks_##B##_conv3_bias,                   \
             MIDDLE_BLKS_##B##_CONV3_REQUANT_M, MIDDLE_BLKS_##B##_CONV3_REQUANT_SHIFT,       \
             middle_blks_##B##_beta, SCALE_FRAC,                                             \
             middle_blks_##B##_norm2_gamma, middle_blks_##B##_norm2_beta,                    \
             MIDDLE_BLKS_##B##_NORM2_FRAC_BITS,                                              \
             middle_blks_##B##_conv4_weight, middle_blks_##B##_conv4_bias,                   \
             MIDDLE_BLKS_##B##_CONV4_REQUANT_M, MIDDLE_BLKS_##B##_CONV4_REQUANT_SHIFT,       \
             MIDDLE_BLKS_##B##_SG2_REQUANT_M, MIDDLE_BLKS_##B##_SG2_REQUANT_SHIFT,           \
             middle_blks_##B##_conv5_weight, middle_blks_##B##_conv5_bias,                   \
             MIDDLE_BLKS_##B##_CONV5_REQUANT_M, MIDDLE_BLKS_##B##_CONV5_REQUANT_SHIFT,       \
             middle_blks_##B##_gamma, SCALE_FRAC)

#define CALL_MID_BLOCK_PARA(B, x_ptr, ch, h, w, pool)                                             \
    nafblock_para(x_ptr, ch, h, w,                                                                \
                  middle_blks_##B##_norm1_gamma, middle_blks_##B##_norm1_beta,                    \
                  MIDDLE_BLKS_##B##_NORM1_FRAC_BITS,                                              \
                  middle_blks_##B##_conv1_weight, middle_blks_##B##_conv1_bias,                   \
                  MIDDLE_BLKS_##B##_CONV1_REQUANT_M, MIDDLE_BLKS_##B##_CONV1_REQUANT_SHIFT,       \
                  middle_blks_##B##_conv2_weight, middle_blks_##B##_conv2_bias,                   \
                  MIDDLE_BLKS_##B##_CONV2_REQUANT_M, MIDDLE_BLKS_##B##_CONV2_REQUANT_SHIFT,       \
                  MIDDLE_BLKS_##B##_SG1_REQUANT_M, MIDDLE_BLKS_##B##_SG1_REQUANT_SHIFT,           \
                  middle_blks_##B##_sca_conv_weight, middle_blks_##B##_sca_conv_bias,             \
                  MIDDLE_BLKS_##B##_SCA_CONV_REQUANT_M, MIDDLE_BLKS_##B##_SCA_CONV_REQUANT_SHIFT, \
                  MIDDLE_BLKS_##B##_SCA_MUL_REQUANT_M, MIDDLE_BLKS_##B##_SCA_MUL_REQUANT_SHIFT,   \
                  middle_blks_##B##_conv3_weight, middle_blks_##B##_conv3_bias,                   \
                  MIDDLE_BLKS_##B##_CONV3_REQUANT_M, MIDDLE_BLKS_##B##_CONV3_REQUANT_SHIFT,       \
                  middle_blks_##B##_beta, SCALE_FRAC,                                             \
                  middle_blks_##B##_norm2_gamma, middle_blks_##B##_norm2_beta,                    \
                  MIDDLE_BLKS_##B##_NORM2_FRAC_BITS,                                              \
                  middle_blks_##B##_conv4_weight, middle_blks_##B##_conv4_bias,                   \
                  MIDDLE_BLKS_##B##_CONV4_REQUANT_M, MIDDLE_BLKS_##B##_CONV4_REQUANT_SHIFT,       \
                  MIDDLE_BLKS_##B##_SG2_REQUANT_M, MIDDLE_BLKS_##B##_SG2_REQUANT_SHIFT,           \
                  middle_blks_##B##_conv5_weight, middle_blks_##B##_conv5_bias,                   \
                  MIDDLE_BLKS_##B##_CONV5_REQUANT_M, MIDDLE_BLKS_##B##_CONV5_REQUANT_SHIFT,       \
                  middle_blks_##B##_gamma, SCALE_FRAC, pool)

#define CALL_DEC_BLOCK(L, B, x_ptr, ch, h, w)                                                      \
    nafblock(x_ptr, ch, h, w,                                                                      \
             decoders_##L##_##B##_norm1_gamma, decoders_##L##_##B##_norm1_beta,                    \
             DECODERS_##L##_##B##_NORM1_FRAC_BITS,                                                 \
             decoders_##L##_##B##_conv1_weight, decoders_##L##_##B##_conv1_bias,                   \
             DECODERS_##L##_##B##_CONV1_REQUANT_M, DECODERS_##L##_##B##_CONV1_REQUANT_SHIFT,       \
             decoders_##L##_##B##_conv2_weight, decoders_##L##_##B##_conv2_bias,                   \
             DECODERS_##L##_##B##_CONV2_REQUANT_M, DECODERS_##L##_##B##_CONV2_REQUANT_SHIFT,       \
             DECODERS_##L##_##B##_SG1_REQUANT_M, DECODERS_##L##_##B##_SG1_REQUANT_SHIFT,           \
             decoders_##L##_##B##_sca_conv_weight, decoders_##L##_##B##_sca_conv_bias,             \
             DECODERS_##L##_##B##_SCA_CONV_REQUANT_M, DECODERS_##L##_##B##_SCA_CONV_REQUANT_SHIFT, \
             DECODERS_##L##_##B##_SCA_MUL_REQUANT_M, DECODERS_##L##_##B##_SCA_MUL_REQUANT_SHIFT,   \
             decoders_##L##_##B##_conv3_weight, decoders_##L##_##B##_conv3_bias,                   \
             DECODERS_##L##_##B##_CONV3_REQUANT_M, DECODERS_##L##_##B##_CONV3_REQUANT_SHIFT,       \
             decoders_##L##_##B##_beta, SCALE_FRAC,                                                \
             decoders_##L##_##B##_norm2_gamma, decoders_##L##_##B##_norm2_beta,                    \
             DECODERS_##L##_##B##_NORM2_FRAC_BITS,                                                 \
             decoders_##L##_##B##_conv4_weight, decoders_##L##_##B##_conv4_bias,                   \
             DECODERS_##L##_##B##_CONV4_REQUANT_M, DECODERS_##L##_##B##_CONV4_REQUANT_SHIFT,       \
             DECODERS_##L##_##B##_SG2_REQUANT_M, DECODERS_##L##_##B##_SG2_REQUANT_SHIFT,           \
             decoders_##L##_##B##_conv5_weight, decoders_##L##_##B##_conv5_bias,                   \
             DECODERS_##L##_##B##_CONV5_REQUANT_M, DECODERS_##L##_##B##_CONV5_REQUANT_SHIFT,       \
             decoders_##L##_##B##_gamma, SCALE_FRAC)

#define CALL_DEC_BLOCK_PARA(L, B, x_ptr, ch, h, w, pool)                                                \
    nafblock_para(x_ptr, ch, h, w,                                                                      \
                  decoders_##L##_##B##_norm1_gamma, decoders_##L##_##B##_norm1_beta,                    \
                  DECODERS_##L##_##B##_NORM1_FRAC_BITS,                                                 \
                  decoders_##L##_##B##_conv1_weight, decoders_##L##_##B##_conv1_bias,                   \
                  DECODERS_##L##_##B##_CONV1_REQUANT_M, DECODERS_##L##_##B##_CONV1_REQUANT_SHIFT,       \
                  decoders_##L##_##B##_conv2_weight, decoders_##L##_##B##_conv2_bias,                   \
                  DECODERS_##L##_##B##_CONV2_REQUANT_M, DECODERS_##L##_##B##_CONV2_REQUANT_SHIFT,       \
                  DECODERS_##L##_##B##_SG1_REQUANT_M, DECODERS_##L##_##B##_SG1_REQUANT_SHIFT,           \
                  decoders_##L##_##B##_sca_conv_weight, decoders_##L##_##B##_sca_conv_bias,             \
                  DECODERS_##L##_##B##_SCA_CONV_REQUANT_M, DECODERS_##L##_##B##_SCA_CONV_REQUANT_SHIFT, \
                  DECODERS_##L##_##B##_SCA_MUL_REQUANT_M, DECODERS_##L##_##B##_SCA_MUL_REQUANT_SHIFT,   \
                  decoders_##L##_##B##_conv3_weight, decoders_##L##_##B##_conv3_bias,                   \
                  DECODERS_##L##_##B##_CONV3_REQUANT_M, DECODERS_##L##_##B##_CONV3_REQUANT_SHIFT,       \
                  decoders_##L##_##B##_beta, SCALE_FRAC,                                                \
                  decoders_##L##_##B##_norm2_gamma, decoders_##L##_##B##_norm2_beta,                    \
                  DECODERS_##L##_##B##_NORM2_FRAC_BITS,                                                 \
                  decoders_##L##_##B##_conv4_weight, decoders_##L##_##B##_conv4_bias,                   \
                  DECODERS_##L##_##B##_CONV4_REQUANT_M, DECODERS_##L##_##B##_CONV4_REQUANT_SHIFT,       \
                  DECODERS_##L##_##B##_SG2_REQUANT_M, DECODERS_##L##_##B##_SG2_REQUANT_SHIFT,           \
                  decoders_##L##_##B##_conv5_weight, decoders_##L##_##B##_conv5_bias,                   \
                  DECODERS_##L##_##B##_CONV5_REQUANT_M, DECODERS_##L##_##B##_CONV5_REQUANT_SHIFT,       \
                  decoders_##L##_##B##_gamma, SCALE_FRAC, pool)

/* ============================================================
 * Forward pass for one image
 * ============================================================ */
static void nafnet_forward(const uint8_t *input_rgb, uint8_t *output_rgb)
{
    int H = 512;
    int W = 512;
    int spatial;
    int chan;

    /* Step 1: Convert uint8 [0,255] to int8 [0,127]
     * Model expects float in [0,1]. We use scale = 1/127, so:
     * int8_val = round(float_val * 127) = round(uint8_val * 127 / 255)
     * = (uint8_val * 127 + 128) / 255  (integer rounding)
     */
    spatial = 3 * H * W;
    for (int i = 0; i < spatial; i++)
    {
        buf_a[i] = (int8_t)(((int32_t)input_rgb[i] * 127 + 128) / 255);
    }

    /* Step 2: Intro conv: 3 -> 32 channels, 3x3 */
    /* buf_a [3, 64, 64] -> buf_c [32, 64, 64] */
    uint64_t start = now_ns();
    conv3x3_bias_requant(buf_a, intro_weight, intro_bias, buf_c, acc_buf,
                         3, 32, H, W, INTRO_REQUANT_M, INTRO_REQUANT_SHIFT);

    uint64_t end = now_ns();
    double elapsed_sec = (end - start) / 1e9;
    printf("  single-threaded intro layer 1: %.3f sec\n", elapsed_sec);
    chan = 32;

    /* ============================================================
     * ENCODER
     * ============================================================ */

    /* --- Encoder Level 0: 2 blocks at 32ch, 64x64 --- */
    /* Copy buf_c to working buffer for NAFBlocks */
    memcpy(buf_a, buf_c, chan * H * W);
    start = now_ns();
    CALL_ENC_BLOCK(0, 0, buf_a, 32, 512, 512);
    CALL_ENC_BLOCK(0, 1, buf_a, 32, 512, 512);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  single-threaded encoder layer 0: %.3f sec\n", elapsed_sec);

    /* Save skip connection */
    memcpy(skip_0, buf_a, 32 * 512 * 512);

    /* Downsample: 32 -> 64 ch, 64x64 -> 32x32 */

    start = now_ns();
    conv2x2s2_bias_requant(buf_a, downs_0_weight, downs_0_bias, buf_c, acc_buf,
                           32, 64, 512, 512, DOWNS_0_REQUANT_M, DOWNS_0_REQUANT_SHIFT);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  single-threaded encoder downsample 0: %.3f sec\n", elapsed_sec);
    H = 256;
    W = 256;
    chan = 64;
    memcpy(buf_a, buf_c, chan * H * W);

    /* --- Encoder Level 1: 2 blocks at 64ch, 32x32 --- */
    CALL_ENC_BLOCK(1, 0, buf_a, 64, 256, 256);
    CALL_ENC_BLOCK(1, 1, buf_a, 64, 256, 256);

    memcpy(skip_1, buf_a, 64 * 256 * 256);

    start = now_ns();
    conv2x2s2_bias_requant(buf_a, downs_1_weight, downs_1_bias, buf_c, acc_buf,
                           64, 128, 256, 256, DOWNS_1_REQUANT_M, DOWNS_1_REQUANT_SHIFT);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  single-threaded encoder downsample 1: %.3f sec\n", elapsed_sec);

    H = 128;
    W = 128;
    chan = 128;
    memcpy(buf_a, buf_c, chan * H * W);

    /* --- Encoder Level 2: 4 blocks at 128ch, 16x16 --- */

    start = now_ns();
    CALL_ENC_BLOCK(2, 0, buf_a, 128, 128, 128);
    CALL_ENC_BLOCK(2, 1, buf_a, 128, 128, 128);
    CALL_ENC_BLOCK(2, 2, buf_a, 128, 128, 128);
    CALL_ENC_BLOCK(2, 3, buf_a, 128, 128, 128);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  single-threaded encoder layer 2: %.3f sec\n", elapsed_sec);

    memcpy(skip_2, buf_a, 128 * 128 * 128);

    start = now_ns();
    conv2x2s2_bias_requant(buf_a, downs_2_weight, downs_2_bias, buf_c, acc_buf,
                           128, 256, 64, 64, DOWNS_2_REQUANT_M, DOWNS_2_REQUANT_SHIFT);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  single-threaded encoder downsample 2: %.3f sec\n", elapsed_sec);

    H = 64;
    W = 64;
    chan = 256;
    memcpy(buf_a, buf_c, chan * H * W);

    /* --- Encoder Level 3: 8 blocks at 256ch, 8x8 --- */
    start = now_ns();
    CALL_ENC_BLOCK(3, 0, buf_a, 256, 64, 64);
    CALL_ENC_BLOCK(3, 1, buf_a, 256, 64, 64);
    CALL_ENC_BLOCK(3, 2, buf_a, 256, 64, 64);
    CALL_ENC_BLOCK(3, 3, buf_a, 256, 64, 64);
    CALL_ENC_BLOCK(3, 4, buf_a, 256, 64, 64);
    CALL_ENC_BLOCK(3, 5, buf_a, 256, 64, 64);
    CALL_ENC_BLOCK(3, 6, buf_a, 256, 64, 64);
    CALL_ENC_BLOCK(3, 7, buf_a, 256, 64, 64);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  single-threaded encoder layer 3: %.3f sec\n", elapsed_sec);

    memcpy(skip_3, buf_a, 256 * 64 * 64);

    start = now_ns();
    conv2x2s2_bias_requant(buf_a, downs_3_weight, downs_3_bias, buf_c, acc_buf,
                           256, 512, 64, 64, DOWNS_3_REQUANT_M, DOWNS_3_REQUANT_SHIFT);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  single-threaded encoder downsample 3: %.3f sec\n", elapsed_sec);

    H = 32;
    W = 32;
    chan = 512;
    memcpy(buf_a, buf_c, chan * H * W);

    /* ============================================================
     * MIDDLE: 12 blocks at 512ch, 4x4
     * ============================================================ */
    start = now_ns();
    CALL_MID_BLOCK(0, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(1, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(2, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(3, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(4, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(5, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(6, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(7, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(8, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(9, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(10, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(11, buf_a, 512, 32, 32);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  single-threaded middle blocks: %.3f sec\n", elapsed_sec);

    /* ============================================================
     * DECODER
     * ============================================================ */

    /* --- Decoder Level 0: upsample 512->256, + skip_3, 2 blocks --- */
    /* ups.0: Conv1x1 512->1024 (no bias) + PixelShuffle(2) */
    /* buf_a [512, 4, 4] -> conv1x1 -> buf_b [1024, 4, 4] -> pixelshuffle -> buf_c [256, 8, 8] */

    start = now_ns();
    conv1x1_requant(buf_a, ups_0_0_weight, buf_b, acc_buf,
                    512, 1024, 32, 32, UPS_0_0_REQUANT_M, UPS_0_0_REQUANT_SHIFT);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  single-threaded decoder upsample 1 (conv1x1): %.3f sec\n", elapsed_sec);
    pixelshuffle_i8(buf_b, buf_c, 256, 32, 32, 2);

    /* Add skip connection: buf_c + skip_3 */
    H = 64;
    W = 64;
    chan = 256;
    elemwise_add_i8(buf_c, skip_3, buf_a, chan * H * W);

    /* 2 decoder blocks */
    start = now_ns();
    CALL_DEC_BLOCK(0, 0, buf_a, 256, 64, 64);
    CALL_DEC_BLOCK(0, 1, buf_a, 256, 64, 64);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  single-threaded decoder layer 0: %.3f sec\n", elapsed_sec);

    /* --- Decoder Level 1: upsample 256->128, + skip_2, 2 blocks --- */
    start = now_ns();
    conv1x1_requant(buf_a, ups_1_0_weight, buf_b, acc_buf,
                    256, 512, 64, 64, UPS_1_0_REQUANT_M, UPS_1_0_REQUANT_SHIFT);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  single-threaded decoder upsample 1 (conv1x1): %.3f sec\n", elapsed_sec);
    pixelshuffle_i8(buf_b, buf_c, 128, 64, 64, 2);

    H = 128;
    W = 128;
    chan = 128;
    elemwise_add_i8(buf_c, skip_2, buf_a, chan * H * W);

    start = now_ns();
    CALL_DEC_BLOCK(1, 0, buf_a, 128, 128, 128);
    CALL_DEC_BLOCK(1, 1, buf_a, 128, 128, 128);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  single-threaded decoder layer 1: %.3f sec\n", elapsed_sec);

    /* --- Decoder Level 2: upsample 128->64, + skip_1, 2 blocks --- */
    start = now_ns();
    conv1x1_requant(buf_a, ups_2_0_weight, buf_b, acc_buf,
                    128, 256, 128, 128, UPS_2_0_REQUANT_M, UPS_2_0_REQUANT_SHIFT);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  single-threaded decoder upsample 2 (conv1x1): %.3f sec\n", elapsed_sec);

    pixelshuffle_i8(buf_b, buf_c, 64, 128, 128, 2);

    H = 256;
    W = 256;
    chan = 64;
    elemwise_add_i8(buf_c, skip_1, buf_a, chan * H * W);

    CALL_DEC_BLOCK(2, 0, buf_a, 64, 256, 256);
    CALL_DEC_BLOCK(2, 1, buf_a, 64, 256, 256);

    /* --- Decoder Level 3: upsample 64->32, + skip_0, 2 blocks --- */
    start = now_ns();
    conv1x1_requant(buf_a, ups_3_0_weight, buf_b, acc_buf,
                    64, 128, 256, 256, UPS_3_0_REQUANT_M, UPS_3_0_REQUANT_SHIFT);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  single-threaded decoder upsample 3 (conv1x1): %.3f sec\n", elapsed_sec);
    pixelshuffle_i8(buf_b, buf_c, 32, 256, 256, 2);

    H = 512;
    W = 512;
    chan = 32;
    elemwise_add_i8(buf_c, skip_0, buf_a, chan * H * W);

    CALL_DEC_BLOCK(3, 0, buf_a, 32, 512, 512);
    CALL_DEC_BLOCK(3, 1, buf_a, 32, 512, 512);

    /* ============================================================
     * ENDING: conv 3x3, 32 -> 3 channels
     * ============================================================ */
    conv3x3_bias_requant(buf_a, ending_weight, ending_bias, buf_b, acc_buf,
                         32, 3, 512, 512, ENDING_REQUANT_M, ENDING_REQUANT_SHIFT);
    /* Step 7: Convert int8 output back to uint8 [0, 255]
     * The ending conv output is in int8 domain with output_scale.
     * float_val = int8_val * output_scale
     * uint8_val = clip(float_val * 255, 0, 255)
     *           = clip(int8_val * output_scale * 255, 0, 255)
     *
     * We use: uint8 = clip((int8 * ENDING_DEQUANT_M + (1 << (ENDING_DEQUANT_SHIFT-1)))
     *                       >> ENDING_DEQUANT_SHIFT, 0, 255)
     * where ENDING_DEQUANT_M / 2^ENDING_DEQUANT_SHIFT ≈ output_scale * 255
     */
    spatial = 3 * 512 * 512;
    for (int i = 0; i < spatial; i++)
    {
        int32_t val = (int32_t)buf_b[i] * ENDING_DEQUANT_M;
        val = (val + (1 << (ENDING_DEQUANT_SHIFT - 1))) >> ENDING_DEQUANT_SHIFT;
        if (val < 0)
            val = 0;
        if (val > 255)
            val = 255;
        output_rgb[i] = (uint8_t)val;
    }
}

static void nafnet_forward_para(const uint8_t *input_rgb, uint8_t *output_rgb)
{
    int H = 512;
    int W = 512;
    int spatial;
    int chan;

    /* Step 1: Convert uint8 [0,255] to int8 [0,127]
     * Model expects float in [0,1]. We use scale = 1/127, so:
     * int8_val = round(float_val * 127) = round(uint8_val * 127 / 255)
     * = (uint8_val * 127 + 128) / 255  (integer rounding)
     */
    spatial = 3 * H * W;
    for (int i = 0; i < spatial; i++)
    {
        buf_a[i] = (int8_t)(((int32_t)input_rgb[i] * 127 + 128) / 255);
    }

    thread_pool_t pool;
    thread_pool_init(&pool, 4);
    // if (!thread_pool_init(&pool, 8)) {
    //     fprintf(stderr, "thread pool init failed\n");
    //     return NULL;
    // }

    /* Step 2: Intro conv: 3 -> 32 channels, 3x3 */
    /* buf_a [3, 64, 64] -> buf_c [32, 64, 64] */
    int64_t start = now_ns();
    conv3x3_bias_requant_para(buf_a, intro_weight, intro_bias, buf_c, acc_buf,
                              3, 32, H, W, INTRO_REQUANT_M, INTRO_REQUANT_SHIFT, &pool);
    int64_t end = now_ns();
    double elapsed_sec = (end - start) / 1e9;
    printf("  multi-threaded intro layer 1: %.3f sec\n", elapsed_sec);

    chan = 32;

    /* ============================================================
     * ENCODER
     * ============================================================ */

    /* --- Encoder Level 0: 2 blocks at 32ch, 64x64 --- */
    /* Copy buf_c to working buffer for NAFBlocks */
    memcpy(buf_a, buf_c, chan * H * W);

    start = now_ns();
    CALL_ENC_BLOCK(0, 0, buf_a, 32, 512, 512);
    CALL_ENC_BLOCK(0, 1, buf_a, 32, 512, 512);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  multi-threaded encoder layer 1: %.3f sec\n", elapsed_sec);

    /* Save skip connection */
    memcpy(skip_0, buf_a, 32 * 512 * 512);

    /* Downsample: 32 -> 64 ch, 64x64 -> 32x32 */
    start = now_ns();
    conv2x2s2_bias_requant_para(buf_a, downs_0_weight, downs_0_bias, buf_c, acc_buf,
                                32, 64, 64, 64, DOWNS_0_REQUANT_M, DOWNS_0_REQUANT_SHIFT, &pool);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  multi-threaded encoder downsample 1: %.3f sec\n", elapsed_sec);

    H = 256;
    W = 256;
    chan = 64;
    memcpy(buf_a, buf_c, chan * H * W);

    /* --- Encoder Level 1: 2 blocks at 64ch, 32x32 --- */
    CALL_ENC_BLOCK(1, 0, buf_a, 64, 256, 256);
    CALL_ENC_BLOCK(1, 1, buf_a, 64, 256, 256);

    memcpy(skip_1, buf_a, 64 * 256 * 256);

    start = now_ns();
    conv2x2s2_bias_requant_para(buf_a, downs_1_weight, downs_1_bias, buf_c, acc_buf,
                                64, 128, 256, 256, DOWNS_1_REQUANT_M, DOWNS_1_REQUANT_SHIFT, &pool);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  multi-threaded encoder downsample 2: %.3f sec\n", elapsed_sec);

    H = 128;
    W = 128;
    chan = 128;
    memcpy(buf_a, buf_c, chan * H * W);

    /* --- Encoder Level 2: 4 blocks at 128ch, 16x16 --- */
    CALL_ENC_BLOCK(2, 0, buf_a, 128, 128, 128);
    CALL_ENC_BLOCK(2, 1, buf_a, 128, 128, 128);
    CALL_ENC_BLOCK(2, 2, buf_a, 128, 128, 128);
    CALL_ENC_BLOCK(2, 3, buf_a, 128, 128, 128);

    memcpy(skip_2, buf_a, 128 * 128 * 128);

    start = now_ns();
    conv2x2s2_bias_requant_para(buf_a, downs_2_weight, downs_2_bias, buf_c, acc_buf,
                                128, 256, 64, 64, DOWNS_2_REQUANT_M, DOWNS_2_REQUANT_SHIFT, &pool);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  multi-threaded encoder downsample 3: %.3f sec\n", elapsed_sec);

    H = 64;
    W = 64;
    chan = 256;
    memcpy(buf_a, buf_c, chan * H * W);

    /* --- Encoder Level 3: 8 blocks at 256ch, 8x8 --- */
    CALL_ENC_BLOCK(3, 0, buf_a, 256, 64, 64);
    CALL_ENC_BLOCK(3, 1, buf_a, 256, 64, 64);
    CALL_ENC_BLOCK(3, 2, buf_a, 256, 64, 64);
    CALL_ENC_BLOCK(3, 3, buf_a, 256, 64, 64);
    CALL_ENC_BLOCK(3, 4, buf_a, 256, 64, 64);
    CALL_ENC_BLOCK(3, 5, buf_a, 256, 64, 64);
    CALL_ENC_BLOCK(3, 6, buf_a, 256, 64, 64);
    CALL_ENC_BLOCK(3, 7, buf_a, 256, 64, 64);

    memcpy(skip_3, buf_a, 256 * 64 * 64);

    conv2x2s2_bias_requant_para(buf_a, downs_3_weight, downs_3_bias, buf_c, acc_buf,
                                256, 512, 64, 64, DOWNS_3_REQUANT_M, DOWNS_3_REQUANT_SHIFT, &pool);

    H = 32;
    W = 32;
    chan = 512;
    memcpy(buf_a, buf_c, chan * H * W);

    /* ============================================================
     * MIDDLE: 12 blocks at 512ch, 4x4
     * ============================================================ */
    CALL_MID_BLOCK(0, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(1, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(2, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(3, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(4, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(5, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(6, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(7, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(8, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(9, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(10, buf_a, 512, 32, 32);
    CALL_MID_BLOCK(11, buf_a, 512, 32, 32);

    /* ============================================================
     * DECODER
     * ============================================================ */

    /* --- Decoder Level 0: upsample 512->256, + skip_3, 2 blocks --- */
    /* ups.0: Conv1x1 512->1024 (no bias) + PixelShuffle(2) */
    /* buf_a [512, 4, 4] -> conv1x1 -> buf_b [1024, 4, 4] -> pixelshuffle -> buf_c [256, 8, 8] */
    start = now_ns();
    conv1x1_requant_para(buf_a, ups_0_0_weight, buf_b, acc_buf,
                         512, 1024, 32, 32, UPS_0_0_REQUANT_M, UPS_0_0_REQUANT_SHIFT, &pool);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  multi-threaded decoder upsample 0 (conv1x1): %.3f sec\n", elapsed_sec);

    pixelshuffle_i8(buf_b, buf_c, 256, 32, 32, 2);

    /* Add skip connection: buf_c + skip_3 */
    H = 64;
    W = 64;
    chan = 256;
    elemwise_add_i8(buf_c, skip_3, buf_a, chan * H * W);

    /* 2 decoder blocks */
    CALL_DEC_BLOCK(0, 0, buf_a, 256, 64, 64);
    CALL_DEC_BLOCK(0, 1, buf_a, 256, 64, 64);

    /* --- Decoder Level 1: upsample 256->128, + skip_2, 2 blocks --- */
    start = now_ns();
    conv1x1_requant_para(buf_a, ups_1_0_weight, buf_b, acc_buf,
                         256, 512, 64, 64, UPS_1_0_REQUANT_M, UPS_1_0_REQUANT_SHIFT, &pool);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  multi-threaded decoder upsample 1 (conv1x1): %.3f sec\n", elapsed_sec);
    pixelshuffle_i8(buf_b, buf_c, 128, 64, 64, 2);

    H = 128;
    W = 128;
    chan = 128;
    elemwise_add_i8(buf_c, skip_2, buf_a, chan * H * W);

    CALL_DEC_BLOCK(1, 0, buf_a, 128, 128, 128);
    CALL_DEC_BLOCK(1, 1, buf_a, 128, 128, 128);

    /* --- Decoder Level 2: upsample 128->64, + skip_1, 2 blocks --- */
    start = now_ns();
    conv1x1_requant_para(buf_a, ups_2_0_weight, buf_b, acc_buf,
                         128, 256, 128, 128, UPS_2_0_REQUANT_M, UPS_2_0_REQUANT_SHIFT, &pool);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  multi-threaded decoder upsample 2 (conv1x1): %.3f sec\n", elapsed_sec);

    pixelshuffle_i8(buf_b, buf_c, 64, 128, 128, 2);

    H = 256;
    W = 256;
    chan = 64;
    elemwise_add_i8(buf_c, skip_1, buf_a, chan * H * W);

    CALL_DEC_BLOCK(2, 0, buf_a, 64, 256, 256);
    CALL_DEC_BLOCK(2, 1, buf_a, 64, 256, 256);

    /* --- Decoder Level 3: upsample 64->32, + skip_0, 2 blocks --- */
    start = now_ns();
    conv1x1_requant_para(buf_a, ups_3_0_weight, buf_b, acc_buf,
                         64, 128, 256, 256, UPS_3_0_REQUANT_M, UPS_3_0_REQUANT_SHIFT, &pool);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  multi-threaded decoder upsample 3 (conv1x1): %.3f sec\n", elapsed_sec);

    pixelshuffle_i8(buf_b, buf_c, 32, 256, 256, 2);

    H = 512;
    W = 512;
    chan = 32;
    elemwise_add_i8(buf_c, skip_0, buf_a, chan * H * W);

    CALL_DEC_BLOCK(3, 0, buf_a, 32, 512, 512);
    CALL_DEC_BLOCK(3, 1, buf_a, 32, 512, 512);

    /* ============================================================
     * ENDING: conv 3x3, 32 -> 3 channels
     * ============================================================ */
    conv3x3_bias_requant_para(buf_a, ending_weight, ending_bias, buf_b, acc_buf,
                              32, 3, 512, 512, ENDING_REQUANT_M, ENDING_REQUANT_SHIFT, &pool);
    /* Step 7: Convert int8 output back to uint8 [0, 255]
     * The ending conv output is in int8 domain with output_scale.
     * float_val = int8_val * output_scale
     * uint8_val = clip(float_val * 255, 0, 255)
     *           = clip(int8_val * output_scale * 255, 0, 255)
     *
     * We use: uint8 = clip((int8 * ENDING_DEQUANT_M + (1 << (ENDING_DEQUANT_SHIFT-1)))
     *                       >> ENDING_DEQUANT_SHIFT, 0, 255)
     * where ENDING_DEQUANT_M / 2^ENDING_DEQUANT_SHIFT ≈ output_scale * 255
     */

    thread_pool_destroy(&pool);

    spatial = 3 * 512 * 512;
    for (int i = 0; i < spatial; i++)
    {
        int32_t val = (int32_t)buf_b[i] * ENDING_DEQUANT_M;
        val = (val + (1 << (ENDING_DEQUANT_SHIFT - 1))) >> ENDING_DEQUANT_SHIFT;
        if (val < 0)
            val = 0;
        if (val > 255)
            val = 255;
        output_rgb[i] = (uint8_t)val;
    }
}
int compare_bytes(const void *a, const void *b, size_t nbytes, int max_print)
{
    const uint8_t *pa = (const uint8_t *)a;
    const uint8_t *pb = (const uint8_t *)b;

    int mism = 0;
    for (size_t i = 0; i < nbytes; ++i)
    {
        if (pa[i] != pb[i])
        {
            if (mism < max_print)
                printf("Mismatch at byte %zu: A=%02X  B=%02X\n", i, pa[i], pb[i]);
            mism++;
        }
    }
    return mism;
}
/* ============================================================
 * Main: process all 10 images
 * ============================================================ */
int main(void)
{
    int pixels_per_image = IMG_CHANNELS * CROP_SIZE * CROP_SIZE;

    uint8_t input[3 * 512 * 512];

    srand((unsigned)time(NULL));
    for (size_t i = 0; i < 3 * 512 * 512; i++)
    {
        input[i] = (uint8_t)(rand() % 512);
    }
    // printf("begin: \n");

    uint64_t start = now_ns();
    nafnet_forward(input, output_img);
    uint64_t end = now_ns();
    double elapsed_sec = (end - start) / 1e9;
    printf("  Forward pass (single-threaded): %.3f sec\n", elapsed_sec);

    start = now_ns();
    nafnet_forward_para(input, output_img_para);
    end = now_ns();
    elapsed_sec = (end - start) / 1e9;
    printf("  Forward pass (parallel): %.3f sec\n", elapsed_sec);

    // printf("NAFNet INT8 Inference - %d images, %dx%d patches\n",
    //        NUM_IMAGES, CROP_SIZE, CROP_SIZE);

    // for (int img = 0; img < NUM_IMAGES; img++)
    // {
    //     printf("Processing image %d...\n", img);

    //     double elapsed_sec = (end - start) / 1e9;

    //     start = now_ns();
    //     nafnet_forward_para(input_images[img], output_img_para);
    //     end = now_ns();

    //     printf("  Forward pass (parallel): %.3f sec\n", elapsed_sec);

    //     /* Print output as hex */
    //     // printf("IMG %d:", img);

    //     int mism = compare_bytes(&output_img, &output_img_para, pixels_per_image, 20);

    //     if (mism == 0)
    //     {
    //         printf(" [PASS] Outputs match!\n");
    //     }
    //     else
    //     {
    //         printf(" [FAIL] Total mismatches: %d\n", mism);
    //     }
    //     // for (int i = 0; i < pixels_per_image; i++) {
    //     //     // printf(" %02X", output_img[i]-output_img_para[i]);
    //     //     //

    //     // }
    //     // printf("\n");

    //     /* Compute PSNR vs ground truth */
    //     // int32_t psnr = compute_psnr_int(output_img, gt_images[img], pixels_per_image);
    //     // printf("PSNR %d: %d.%02d dB\n", img, psnr / 100,
    //     //        (psnr >= 0 ? psnr : -psnr) % 100);
    // }

    printf("Done.\n");
    return 0;
}
