#pragma once
#include <cstdint>
#include "hardware_config.h"

// ============================================================
// Hardware and tensor configuration for the LayerNorm2d TLM
// performance simulator (int8 in / int8 out, three-pass model).
//
// Per Layer_Norm.md Requirements v1:
//   - inputs and outputs are int8 (LN_INPUT_ELEM_BYTES = 1)
//   - intermediate reductions live in int32 in vec-unit registers
//   - per channel the kernel runs three passes
//       1) sum  -> scalar mean
//       2) sum-of-squares (about mean) -> scalar isqrt + 1/std
//       3) normalize and store
//   - hardware knobs inherit from config/hardware_config.h so this
//     kernel sees the same L1/L2/accelerator parameters as
//     pooling/dw_conv2d/matmul.
//
// Layout mirrors kernel/dw_conv2d/dw_conv2d_config.h.
// ============================================================

// ------------------------------------------------------------
// Data element sizes (bytes).
// LN_INPUT_ELEM_BYTES   = 1  (int8 input)
// LN_OUTPUT_ELEM_BYTES  = 1  (int8 output; intermediate int32
//                              accumulators live in the pinned
//                              vec unit register, not in memory).
// ------------------------------------------------------------
static const uint64_t LN_INPUT_ELEM_BYTES  = 1;
static const uint64_t LN_OUTPUT_ELEM_BYTES = 1;

// ------------------------------------------------------------
// Input tensor: [C, H, W] int8, channels-first.
// Each channel is normalized independently.
// ------------------------------------------------------------
static const int LN_C = 32;
static const int LN_H = 64;
static const int LN_W = 64;

// ------------------------------------------------------------
// Parallelism: number of worker SC_THREADs.
// Channels distributed evenly:
//   worker tid owns channels [c_start, c_end)
//   where c_start = (tid * LN_C) / LN_NUM_WORKERS.
// ------------------------------------------------------------
static const int LN_NUM_WORKERS = 4;

// ------------------------------------------------------------
// Vector accelerator: processes LN_VEC_ACC_CAP elements per
// call. Service cycles are computed per-pass as
//   n_compute_intrinsics * LN_VEC_INSN_CYCLE
// to mirror matmul's MATMUL_ACCUM_VEC_INSNS / MATMUL_QUANT_VEC_INSNS
// pattern. Pool runs in pinned mode (5-arg ctor) with
// worker i -> unit (i % LN_VEC_ACC_INSTANCES).
// ------------------------------------------------------------
static const uint64_t LN_VEC_ACC_CAP       = VECTOR_ACC_CAP;
static const uint64_t LN_VEC_INSN_CYCLE    = HW_VECTOR_INSN_CYCLE;
static const int      LN_VEC_ACC_INSTANCES = VEC_ACCEL_COUNT;

// ------------------------------------------------------------
// Per-pass vector-instruction count (mf_layernorm2d_i8 in
// kernel/layer_norm.h). Counts arithmetic intrinsics only:
// vle/vse/vsetvl, redsum-identity vmv_v_x, and scalar vmv_x_s
// are excluded.
//   Pass 1 (sum):       vwmul_vx, vredsum_vs                          = 2
//   Pass 2 (variance):  vwmul_vx, vsub_vx, vwmul_vv, vredsum_vs       = 4
//   Pass 3 (normalize): vwmul_vx, vsub_vx, vwmul_vx, vmul_vx, vsra_vx,
//                       vadd_vx, vmax_vx, vmin_vx, vnsra_wx, vnsra_wx = 10
// ------------------------------------------------------------
#ifndef LN_PASS1_INSNS
#define LN_PASS1_INSNS 2
#endif
#ifndef LN_PASS2_INSNS
#define LN_PASS2_INSNS 4
#endif
#ifndef LN_PASS3_INSNS
#define LN_PASS3_INSNS 10
#endif

static const uint64_t LN_PASS1_INSNS_CFG = LN_PASS1_INSNS;
static const uint64_t LN_PASS2_INSNS_CFG = LN_PASS2_INSNS;
static const uint64_t LN_PASS3_INSNS_CFG = LN_PASS3_INSNS;

// Back-compat alias for nafnet/nafnet_layers.h's independent expected-stats
// path, which still references a single flat per-request vec cycle. Use
// pass-3 (the most expensive) so its rough estimate is a worst-case rather
// than an under-count. The bridge's expected-stats logic is independently
// stale (assumes 4-step int16 model) and needs its own update.
static const uint64_t LN_VEC_ACC_CYCLE = LN_VEC_INSN_CYCLE * LN_PASS3_INSNS_CFG;

// ------------------------------------------------------------
// Scalar CPU overhead per vec dispatch call.
// ------------------------------------------------------------
static const uint64_t LN_SCALAR_OVERHEAD = HW_VEC_SCALAR_OVERHEAD;

// ------------------------------------------------------------
// Scalar cost charged between passes:
//   LN_MEAN_CYCLES   - integer divide  sum / spatial   after pass 1
//   LN_INVSTD_CYCLES - var/N + isqrt + 1/std            after pass 2
// Picked to mirror pooling's LN_DIVIDE_CYCLES / mf_isqrt cost.
// ------------------------------------------------------------
#ifndef LN_MEAN_CYCLES
#define LN_MEAN_CYCLES 4
#endif
#ifndef LN_INVSTD_CYCLES
#define LN_INVSTD_CYCLES 16
#endif

static const uint64_t LN_MEAN_CYCLES_CFG   = LN_MEAN_CYCLES;
static const uint64_t LN_INVSTD_CYCLES_CFG = LN_INVSTD_CYCLES;

// Back-compat alias for nafnet/nafnet_layers.h, which still uses a single
// "step3" cycle constant for its independent expected-stats path. Sum of
// both scalar between-pass costs matches the legacy behaviour closely
// enough that the bridge link compiles; the bridge's expected counters
// are independently stale and need their own update.
static const uint64_t LN_STEP3_CYCLES = LN_MEAN_CYCLES_CFG + LN_INVSTD_CYCLES_CFG;

// ------------------------------------------------------------
// Shared memory subsystem - L1/L2 split, aligned with pooling
// and dw_conv2d. L1 bw = VECTOR_ACC_CAP so one vector-wide load
// is delivered in 1 cycle; L2 bw = L1/4 with base latency 10 to
// reflect the slower L2 path.
// ------------------------------------------------------------
#ifndef LN_L1_BASE_LAT
#define LN_L1_BASE_LAT 1
#endif

#ifndef LN_L1_BW
#define LN_L1_BW VECTOR_ACC_CAP
#endif

#ifndef LN_L1_SLOTS
#define LN_L1_SLOTS 8
#endif

#ifndef LN_L2_BASE_LAT
#define LN_L2_BASE_LAT 10
#endif

#ifndef LN_L2_BW
#define LN_L2_BW (LN_L1_BW / 4)
#endif

#ifndef LN_L2_SLOTS
#define LN_L2_SLOTS 16
#endif

static const uint64_t LN_L1_BASE_LAT_CFG = LN_L1_BASE_LAT;
static const uint64_t LN_L1_BW_CFG       = LN_L1_BW;
static const uint64_t LN_L1_SLOTS_CFG    = LN_L1_SLOTS;
static const uint64_t LN_L2_BASE_LAT_CFG = LN_L2_BASE_LAT;
static const uint64_t LN_L2_BW_CFG       = LN_L2_BW;
static const uint64_t LN_L2_SLOTS_CFG    = LN_L2_SLOTS;

// ------------------------------------------------------------
// Accelerator queue depth - per-unit capacity in pinned mode.
// ------------------------------------------------------------
static const size_t LN_ACC_QUEUE_DEPTH =
    HW_ACC_QUEUE_DEPTH > static_cast<size_t>(VEC_ACCEL_COUNT * 4)
        ? HW_ACC_QUEUE_DEPTH
        : static_cast<size_t>(VEC_ACCEL_COUNT * 4);

// ------------------------------------------------------------
// Per-DMA scalar setup overhead (vector-pipe DMA).
// ------------------------------------------------------------
#ifndef LN_DMA_VEC_RD_SCALAR
#define LN_DMA_VEC_RD_SCALAR HW_DMA_VEC_RD_SCALAR
#endif
#ifndef LN_DMA_VEC_WR_SCALAR
#define LN_DMA_VEC_WR_SCALAR HW_DMA_VEC_WR_SCALAR
#endif

static const uint64_t LN_DMA_VEC_RD_SCALAR_CFG = LN_DMA_VEC_RD_SCALAR;
static const uint64_t LN_DMA_VEC_WR_SCALAR_CFG = LN_DMA_VEC_WR_SCALAR;

// ------------------------------------------------------------
// Max outstanding vec-compute requests per worker within one
// pass. Defaults to the per-unit queue capacity so the worker
// can keep its pinned unit fully fed.
// ------------------------------------------------------------
#ifndef LN_MAX_INFLIGHT_VEC_REQS
#define LN_MAX_INFLIGHT_VEC_REQS LN_ACC_QUEUE_DEPTH
#endif

static const uint64_t LN_MAX_INFLIGHT_VEC_REQS_CFG = LN_MAX_INFLIGHT_VEC_REQS;

// ------------------------------------------------------------
// Per-worker bound on outstanding L2 writeback DMAs
// (fire-and-forget channel results). Reaped when window fills,
// fully drained at end of run.
// ------------------------------------------------------------
#ifndef LN_MAX_INFLIGHT_DMA_WRITES
#define LN_MAX_INFLIGHT_DMA_WRITES 2
#endif

static const uint64_t LN_MAX_INFLIGHT_DMA_WRITES_CFG = LN_MAX_INFLIGHT_DMA_WRITES;
