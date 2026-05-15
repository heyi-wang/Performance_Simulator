#pragma once
#include <cstdint>
#include "hardware_config.h"

// ============================================================
// Hardware and tensor configuration for the Global Average
// Pooling TLM performance simulator.
//
// Data-type parameterisation
// --------------------------
// To switch input precision, change POOL_INPUT_ELEM_BYTES only:
//   1  → int8   (current default, mf_global_avgpool_i8)
//   2  → int16  (mf_global_avgpool_i16)
// POOL_OUTPUT_ELEM_BYTES is always 4 (int32 accumulator).
// ============================================================

// ------------------------------------------------------------
// Data element sizes (bytes).
// Change POOL_INPUT_ELEM_BYTES here to switch input precision.
// ------------------------------------------------------------
static const uint64_t POOL_INPUT_ELEM_BYTES  = 1;   // int8 input
static const uint64_t POOL_OUTPUT_ELEM_BYTES = 4;   // int32 output per channel

// ------------------------------------------------------------
// Input tensor: [C, H, W], channels-first layout.
// Global average pooling reduces each channel to one scalar.
//   output: [C]  int32
// ------------------------------------------------------------
static const int POOL_C = 32;    // number of channels
static const int POOL_H = 64;    // spatial height
static const int POOL_W = 64;    // spatial width

// ------------------------------------------------------------
// Parallelism: number of worker SC_THREADs.
// Channels are distributed evenly across workers:
//   worker tid owns channels [c_start, c_end)
//   where c_start = (tid * POOL_C) / POOL_NUM_WORKERS
// ------------------------------------------------------------
static const int POOL_NUM_WORKERS = 1;

// ------------------------------------------------------------
// Vector accelerator: processes POOL_VEC_ACC_CAP elements per
// call in POOL_VEC_ACC_CYCLE cycles.
// POOL_VEC_ACC_INSTANCES controls the pool size.
// This kernel still uses a flat per-request vector latency model.
// ------------------------------------------------------------
static const uint64_t POOL_VEC_ACC_CAP       = VECTOR_ACC_CAP;    // elements per call
static const uint64_t POOL_VEC_ACC_CYCLE     = VECTOR_ACC_CYCLE;  // cycles per call
static const int      POOL_VEC_ACC_INSTANCES = VEC_ACCEL_COUNT;   // physical units in pool

// ------------------------------------------------------------
// Scalar CPU overhead per accelerator tile dispatch call.
// (address computation, loop bookkeeping, tile setup)
// ------------------------------------------------------------
static const uint64_t POOL_SCALAR_OVERHEAD = HW_VEC_SCALAR_OVERHEAD;  // cycles per dispatch

// ------------------------------------------------------------
// Scalar post-processing per channel.
//
// After all reduction tiles complete, the kernel executes:
//   output[c] = (int32_t)(total_sum / spatial)
//
// This is a scalar integer divide followed by a scalar store.
// It is modelled as a CPU stall of POOL_DIVIDE_CYCLES with
// POOL_OUTPUT_ELEM_BYTES added to total_wr_bytes.
// ------------------------------------------------------------
static const uint64_t POOL_DIVIDE_CYCLES = 4;   // approx cycles for integer divide

// ------------------------------------------------------------
// Shared memory subsystem — aligned with matmul defaults
// (see MatmulRuntimeConfig in kernel/matmul/matmul_top.h).
// L1 bandwidth = VECTOR_ACC_CAP: one vector-wide load is delivered
// to the vector accumulator in 1 cycle. DMA bandwidth is L1/4 and
// DMA base latency is 10 to reflect the slower L2 path.
// ------------------------------------------------------------
#ifndef POOL_L1_BASE_LAT
#define POOL_L1_BASE_LAT 1
#endif

#ifndef POOL_L1_BW
#define POOL_L1_BW VECTOR_ACC_CAP
#endif

#ifndef POOL_L1_SLOTS
#define POOL_L1_SLOTS 8
#endif

#ifndef POOL_L2_BASE_LAT
#define POOL_L2_BASE_LAT 10
#endif

#ifndef POOL_L2_BW
#define POOL_L2_BW (POOL_L1_BW / 4)
#endif

#ifndef POOL_L2_SLOTS
#define POOL_L2_SLOTS 16
#endif

// ------------------------------------------------------------
// Accelerator queue depth — matches matmul: max(HW_ACC_QUEUE_DEPTH,
// VEC_ACCEL_COUNT * 4). Maximum number of admitted requests
// (including the one being serviced); workers stall on backpressure
// when this limit is reached.
// ------------------------------------------------------------
static const size_t POOL_ACC_QUEUE_DEPTH =
    HW_ACC_QUEUE_DEPTH > static_cast<size_t>(VEC_ACCEL_COUNT * 4)
        ? HW_ACC_QUEUE_DEPTH
        : static_cast<size_t>(VEC_ACCEL_COUNT * 4);

// ------------------------------------------------------------
// L1 tile-buffer window. Matmul's Worker::issue_stream does not
// impose a separate tile-buffer cap; its read/compute/write stream
// depth is the same queue-cap window passed as max_inflight_*_reqs.
// Default to the same queue capacity so pooling has the same setup.
// ------------------------------------------------------------
#ifndef POOL_L1_TILE_BUFFERS
#define POOL_L1_TILE_BUFFERS POOL_ACC_QUEUE_DEPTH
#endif

static const uint64_t POOL_L1_BASE_LAT_CFG = POOL_L1_BASE_LAT;
static const uint64_t POOL_L1_BW_CFG = POOL_L1_BW;
static const uint64_t POOL_L1_SLOTS_CFG = POOL_L1_SLOTS;
static const uint64_t POOL_L2_BASE_LAT_CFG = POOL_L2_BASE_LAT;
static const uint64_t POOL_L2_BW_CFG = POOL_L2_BW;
static const uint64_t POOL_L2_SLOTS_CFG = POOL_L2_SLOTS;
static const int      POOL_L1_TILE_BUFFERS_CFG =
    static_cast<int>(POOL_L1_TILE_BUFFERS);

// ------------------------------------------------------------
// Per-DMA scalar setup overhead (vector-pipe DMA).
// Mirrors matmul's DMA_VEC_*_SCALAR semantics: each pooling DMA
// carries one tile of int8 input or one int32 output scalar, so
// the cost is charged once per call. Defaults match matmul.
// ------------------------------------------------------------
#ifndef POOL_DMA_VEC_RD_SCALAR
#define POOL_DMA_VEC_RD_SCALAR HW_DMA_VEC_RD_SCALAR
#endif
#ifndef POOL_DMA_VEC_WR_SCALAR
#define POOL_DMA_VEC_WR_SCALAR HW_DMA_VEC_WR_SCALAR
#endif

static const uint64_t POOL_DMA_VEC_RD_SCALAR_CFG = POOL_DMA_VEC_RD_SCALAR;
static const uint64_t POOL_DMA_VEC_WR_SCALAR_CFG = POOL_DMA_VEC_WR_SCALAR;

// ------------------------------------------------------------
// Max outstanding vec-compute requests per worker. Aligns with
// matmul's Worker setup: workers receive the vector accelerator
// queue capacity as their `max_inflight_vec_reqs` stream window,
// so the default is the same queue-cap expression used above.
// ------------------------------------------------------------
#ifndef POOL_MAX_INFLIGHT_VEC_REQS
#define POOL_MAX_INFLIGHT_VEC_REQS POOL_ACC_QUEUE_DEPTH
#endif

static const uint64_t POOL_MAX_INFLIGHT_VEC_REQS_CFG = POOL_MAX_INFLIGHT_VEC_REQS;
