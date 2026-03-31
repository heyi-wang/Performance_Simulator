#pragma once
#include <cstdint>
#include "../config.h"

// ============================================================
// Hardware and tensor configuration for the DwConv2d
// TLM performance simulator (int16 input → int32 output).
//
// Data-type parameterisation
// --------------------------
// To switch input precision, change DW_INPUT_ELEM_BYTES only:
//   2  → int16  (current default, mf_dw_conv2d_3x3_i16)
//   1  → int8   (mf_dw_conv2d_3x3_i8)
// DW_OUTPUT_ELEM_BYTES is always 4 (int32 accumulator).
// ============================================================

// ------------------------------------------------------------
// Data element sizes (bytes)
// Change DW_INPUT_ELEM_BYTES here to switch input precision.
// ------------------------------------------------------------
static const uint64_t DW_INPUT_ELEM_BYTES  = 2;   // int16 input & kernel weights
static const uint64_t DW_OUTPUT_ELEM_BYTES = 4;   // int32 accumulator output

// ------------------------------------------------------------
// Input tensor: [C, H, W], channels-first layout.
// Depth-wise conv applies one kernel per channel (C_in == C_out).
// ------------------------------------------------------------
static const int DW_C = static_cast<int>(CONV_C_IN);   // number of channels
static const int DW_H = static_cast<int>(CONV_H_IN);   // input spatial height
static const int DW_W = static_cast<int>(CONV_W_IN);   // input spatial width

// ------------------------------------------------------------
// Convolution kernel geometry (parameterised).
// Default: 3×3 with symmetric padding and stride 1.
// ------------------------------------------------------------
static const int DW_KH     = static_cast<int>(CONV_KH);     // kernel height
static const int DW_KW     = static_cast<int>(CONV_KW);     // kernel width
static const int DW_PAD    = static_cast<int>(CONV_PAD);    // symmetric zero-padding
static const int DW_STRIDE = static_cast<int>(CONV_STRIDE); // stride

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
// call in DW_VEC_ACC_CYCLE cycles.
// DW_VEC_ACC_INSTANCES controls the pool size.
// ------------------------------------------------------------
static const uint64_t DW_VEC_ACC_CAP       = VECTOR_ACC_CAP;    // elements per call
static const uint64_t DW_VEC_ACC_CYCLE     = VECTOR_ACC_CYCLE;  // cycles per call
static const int      DW_VEC_ACC_INSTANCES = VEC_ACCEL_COUNT;   // physical units in pool

// ------------------------------------------------------------
// Scalar CPU overhead per accelerator tile dispatch call.
// (address computation, loop bookkeeping, tile setup)
// ------------------------------------------------------------
static const uint64_t DW_SCALAR_OVERHEAD = SCALAR_OVERHEAD;   // cycles per dispatch

// ------------------------------------------------------------
// Shared memory subsystem.
// Latency model: cycles = DW_MEM_BASE_LAT + ceil(bytes / DW_MEM_BW)
// ------------------------------------------------------------
static const uint64_t DW_MEM_BASE_LAT = 1;    // fixed base latency (cycles)
static const uint64_t DW_MEM_BW       = 256;   // bandwidth: bytes per cycle

// ------------------------------------------------------------
// Accelerator queue depth.
// Maximum number of admitted requests (including the one being
// serviced).  Workers stall (back-pressure) when this limit is
// reached.
// ------------------------------------------------------------
static const size_t DW_ACC_QUEUE_DEPTH = 32;
