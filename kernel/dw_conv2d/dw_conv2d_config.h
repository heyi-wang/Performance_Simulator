#pragma once
#include <cstdint>
#include "hardware_config.h"

// ============================================================
// Hardware and tensor configuration for the DwConv2d TLM
// performance simulator (int8 input → int32 output, v2 model).
//
// Per Dw_Conv2d.md Requirements v2:
//   - inputs are int8 (DW_INPUT_ELEM_BYTES = 1)
//   - one output strip emits Kh*Kw load-broadcast-compute vec
//     requests; the last sub-request carries the L1 writeback
//   - all hardware-side knobs inherit from config/hardware_config.h
//
// Layout aligned with kernel/pooling/pooling_config.h.
// ============================================================

// ------------------------------------------------------------
// Data element sizes (bytes)
// Change DW_INPUT_ELEM_BYTES here to switch input precision:
//   1  → int8   (default per v2)
//   2  → int16  (mf_dw_conv2d_3x3_i16)
// DW_OUTPUT_ELEM_BYTES is always 4 (int32 accumulator).
// ------------------------------------------------------------
static const uint64_t DW_INPUT_ELEM_BYTES  = 1;   // int8 input & kernel weights
static const uint64_t DW_OUTPUT_ELEM_BYTES = 4;   // int32 accumulator output

// ------------------------------------------------------------
// Input tensor: [C, H, W], channels-first layout.
// Depth-wise conv applies one kernel per channel (C_in == C_out).
// ------------------------------------------------------------
static const int DW_C = 64;   // number of channels
static const int DW_H = 128;  // input spatial height
static const int DW_W = 128;  // input spatial width

// ------------------------------------------------------------
// Convolution kernel geometry (parameterised).
// Default: 3×3 with symmetric padding and stride 1.
// ------------------------------------------------------------
static const int DW_KH     = 3; // kernel height
static const int DW_KW     = 3; // kernel width
static const int DW_PAD    = 1; // symmetric zero-padding
static const int DW_STRIDE = 1; // stride

// Output spatial dimensions (derived).
static const int DW_OUT_H =
    (DW_H + 2 * DW_PAD - DW_KH) / DW_STRIDE + 1;
static const int DW_OUT_W =
    (DW_W + 2 * DW_PAD - DW_KW) / DW_STRIDE + 1;

// ------------------------------------------------------------
// Parallelism: number of worker SC_THREADs.
// Channels are distributed evenly across workers:
//   worker tid owns channels [c_start, c_end)
//   where c_start = (tid * DW_C) / DW_NUM_WORKERS
// ------------------------------------------------------------
static const int DW_NUM_WORKERS = 16;

// ------------------------------------------------------------
// Vector accelerator: processes DW_VEC_ACC_CAP elements per
// call in DW_VEC_ACC_CYCLE cycles. The kernel uses the pinned
// `AcceleratorPool` mode (5-arg ctor) so the per-unit FIFO
// guarantees a strip on a unit cannot start until the unit's
// current strip finishes. The mapping is round-robin
// (worker i → unit (i % DW_VEC_ACC_INSTANCES)) so each unit
// serves DW_NUM_WORKERS / DW_VEC_ACC_INSTANCES workers.
// ------------------------------------------------------------
static const uint64_t DW_VEC_ACC_CAP       = VECTOR_ACC_CAP;    // elements per call
static const uint64_t DW_VEC_ACC_CYCLE     = VECTOR_ACC_CYCLE;  // cycles per call
static const int      DW_VEC_ACC_INSTANCES = VEC_ACCEL_COUNT;   // pinned units

// ------------------------------------------------------------
// Scalar CPU overhead per accelerator tile dispatch call.
// (address computation, loop bookkeeping, tile setup)
// ------------------------------------------------------------
static const uint64_t DW_SCALAR_OVERHEAD = HW_VEC_SCALAR_OVERHEAD;   // cycles per dispatch

// ------------------------------------------------------------
// Shared memory subsystem — L1/L2 split, aligned with pooling
// (see kernel/pooling/pooling_config.h). L1 bandwidth =
// VECTOR_ACC_CAP so one vector-wide load is delivered in 1 cycle.
// DMA bandwidth is L1/4 and DMA base latency is 10 to reflect
// the slower L2 path.
// ------------------------------------------------------------
#ifndef DW_L1_BASE_LAT
#define DW_L1_BASE_LAT 1
#endif

#ifndef DW_L1_BW
#define DW_L1_BW VECTOR_ACC_CAP
#endif

#ifndef DW_L1_SLOTS
#define DW_L1_SLOTS 8
#endif

#ifndef DW_L2_BASE_LAT
#define DW_L2_BASE_LAT 10
#endif

#ifndef DW_L2_BW
#define DW_L2_BW (DW_L1_BW / 4)
#endif

#ifndef DW_L2_SLOTS
#define DW_L2_SLOTS 16
#endif

static const uint64_t DW_L1_BASE_LAT_CFG = DW_L1_BASE_LAT;
static const uint64_t DW_L1_BW_CFG       = DW_L1_BW;
static const uint64_t DW_L1_SLOTS_CFG    = DW_L1_SLOTS;
static const uint64_t DW_L2_BASE_LAT_CFG = DW_L2_BASE_LAT;
static const uint64_t DW_L2_BW_CFG       = DW_L2_BW;
static const uint64_t DW_L2_SLOTS_CFG    = DW_L2_SLOTS;

// ------------------------------------------------------------
// Accelerator queue depth — matches pooling/matmul:
//   max(HW_ACC_QUEUE_DEPTH, VEC_ACCEL_COUNT * 4).
// Per-unit capacity in pinned mode.
// ------------------------------------------------------------
static const size_t DW_ACC_QUEUE_DEPTH =
    HW_ACC_QUEUE_DEPTH > static_cast<size_t>(VEC_ACCEL_COUNT * 4)
        ? HW_ACC_QUEUE_DEPTH
        : static_cast<size_t>(VEC_ACCEL_COUNT * 4);

// ------------------------------------------------------------
// Per-DMA scalar setup overhead (vector-pipe DMA).
// Each dw_conv2d DMA carries one strip's worth of payload, so
// mirrors matmul/pooling's DmaScalarMode::VecPerCall semantics.
// ------------------------------------------------------------
#ifndef DW_DMA_VEC_RD_SCALAR
#define DW_DMA_VEC_RD_SCALAR HW_DMA_VEC_RD_SCALAR
#endif
#ifndef DW_DMA_VEC_WR_SCALAR
#define DW_DMA_VEC_WR_SCALAR HW_DMA_VEC_WR_SCALAR
#endif

static const uint64_t DW_DMA_VEC_RD_SCALAR_CFG = DW_DMA_VEC_RD_SCALAR;
static const uint64_t DW_DMA_VEC_WR_SCALAR_CFG = DW_DMA_VEC_WR_SCALAR;

// ------------------------------------------------------------
// Max outstanding vec-compute requests per worker. Bounds the
// 3-queue (read / compute / write) pipeline depth in
// DwConvPostProcessor. Defaults to the per-unit queue capacity
// so the worker can keep its pinned unit fully fed.
// ------------------------------------------------------------
#ifndef DW_MAX_INFLIGHT_VEC_REQS
#define DW_MAX_INFLIGHT_VEC_REQS DW_ACC_QUEUE_DEPTH
#endif

static const uint64_t DW_MAX_INFLIGHT_VEC_REQS_CFG = DW_MAX_INFLIGHT_VEC_REQS;

// ------------------------------------------------------------
// Per-worker bound on outstanding L2 writeback DMAs
// (fire-and-forget strip results). Reaped when window fills,
// then fully drained at the end of each row.
// ------------------------------------------------------------
#ifndef DW_MAX_INFLIGHT_DMA_WRITES
#define DW_MAX_INFLIGHT_DMA_WRITES 2
#endif

static const uint64_t DW_MAX_INFLIGHT_DMA_WRITES_CFG = DW_MAX_INFLIGHT_DMA_WRITES;
