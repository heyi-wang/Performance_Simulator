#pragma once
#include <cstdint>
#include "hardware_config.h"
// ============================================================
// Hardware and tensor configuration for the LayerNorm2d
// TLM performance simulator (int16, NAFNet).
//
// All cycle/byte parameters are tunable.  Derived quantities
// (e.g. n_tiles per channel) are computed at runtime from
// the values below.
// ============================================================

// ------------------------------------------------------------
// Input tensor: [C, H, W] int16, channels-first layout.
// Each channel is processed independently (per mf_layernorm2d_i16).
// ------------------------------------------------------------
static const int LN_C = 32;    // number of channels
static const int LN_H = 64;    // spatial height
static const int LN_W = 64;    // spatial width

// ------------------------------------------------------------
// Parallelism: number of worker SC_THREADs.
// Channels are distributed evenly across workers:
//   worker tid owns channels [tid*C/N, (tid+1)*C/N).
// ------------------------------------------------------------
static const int LN_NUM_WORKERS = 4;

// ------------------------------------------------------------
// Vector accelerator: processes LN_VEC_ACC_CAP elements/call.
// Each call costs LN_VEC_ACC_CYCLE cycles.
// LN_VEC_ACC_INSTANCES controls the pool size.
// This kernel still uses a flat per-request vector latency model.
// ------------------------------------------------------------
static const uint64_t LN_VEC_ACC_CAP       = VECTOR_ACC_CAP;   // elements per call
static const uint64_t LN_VEC_ACC_CYCLE     = VECTOR_ACC_CYCLE;    // cycles per call
static const int      LN_VEC_ACC_INSTANCES = VEC_ACCEL_COUNT;    // physical accelerator units in pool

// ------------------------------------------------------------
// Scalar CPU overhead per accelerator tile dispatch call.
// (address computation, loop bookkeeping, tile setup)
// ------------------------------------------------------------
static const uint64_t LN_SCALAR_OVERHEAD = HW_VEC_SCALAR_OVERHEAD;  // cycles per dispatch

// ------------------------------------------------------------
// Step 3 (inv_std_fp): isqrt + integer divide — pure scalar.
// No accelerator request; modelled as a CPU stall.
// ------------------------------------------------------------
static const uint64_t LN_STEP3_CYCLES = 16;   // approx cycles for isqrt + divide

// ------------------------------------------------------------
// Shared memory subsystem.
// Latency model: cycles = LN_MEM_BASE_LAT + ceil(bytes / LN_MEM_BW)
// ------------------------------------------------------------
static const uint64_t LN_MEM_BASE_LAT = HW_MEMORY_BASE_LAT;      // fixed base latency (cycles)
static const uint64_t LN_MEM_BW       = HW_MEMORY_BYTES_PER_CYCLE; // bandwidth: bytes per cycle

// ------------------------------------------------------------
// Accelerator queue depth.
// Maximum number of admitted requests (including the one being
// serviced).  Workers stall when this limit is reached.
// ------------------------------------------------------------
static const size_t LN_ACC_QUEUE_DEPTH = HW_ACC_QUEUE_DEPTH;
