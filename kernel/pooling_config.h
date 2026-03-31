#pragma once
#include <cstdint>
#include "../config.h"

// ============================================================
// Hardware and tensor configuration for the Global Average
// Pooling TLM performance simulator.
//
// Data-type parameterisation
// --------------------------
// To switch input precision, change POOL_INPUT_ELEM_BYTES only:
//   2  → int16  (current default, mf_global_avgpool_i16)
//   1  → int8   (mf_global_avgpool_i8)
// POOL_OUTPUT_ELEM_BYTES is always 4 (int32 accumulator).
// ============================================================

// ------------------------------------------------------------
// Data element sizes (bytes).
// Change POOL_INPUT_ELEM_BYTES here to switch input precision.
// ------------------------------------------------------------
static const uint64_t POOL_INPUT_ELEM_BYTES  = 2;   // int16 input
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
static const int POOL_NUM_WORKERS = 16;

// ------------------------------------------------------------
// Vector accelerator: processes POOL_VEC_ACC_CAP elements per
// call in POOL_VEC_ACC_CYCLE cycles.
// POOL_VEC_ACC_INSTANCES controls the pool size.
// ------------------------------------------------------------
static const uint64_t POOL_VEC_ACC_CAP       = VECTOR_ACC_CAP;    // elements per call
static const uint64_t POOL_VEC_ACC_CYCLE     = VECTOR_ACC_CYCLE;  // cycles per call
static const int      POOL_VEC_ACC_INSTANCES = VEC_ACCEL_COUNT;   // physical units in pool

// ------------------------------------------------------------
// Scalar CPU overhead per accelerator tile dispatch call.
// (address computation, loop bookkeeping, tile setup)
// ------------------------------------------------------------
static const uint64_t POOL_SCALAR_OVERHEAD = SCALAR_OVERHEAD;  // cycles per dispatch

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
// Shared memory subsystem.
// Latency model: cycles = POOL_MEM_BASE_LAT + ceil(bytes / POOL_MEM_BW)
// ------------------------------------------------------------
static const uint64_t POOL_MEM_BASE_LAT = 1;    // fixed base latency (cycles)
static const uint64_t POOL_MEM_BW       = 64;   // bandwidth: bytes per cycle

// ------------------------------------------------------------
// Accelerator queue depth.
// Maximum number of admitted requests (including the one being
// serviced).  Workers stall (back-pressure) when this limit is
// reached.
// ------------------------------------------------------------
static const size_t POOL_ACC_QUEUE_DEPTH = 32;
