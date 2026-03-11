#pragma once
#include <cstdint>

// ============================================================
// Hardware configuration for NAFNet performance simulator
//
// This file defines the simulated hardware parameters used to
// estimate cycle counts for each layer in the NAFNet network.
// All values are tunable to model different hardware targets.
// ============================================================

// ------------------------------------------------------------
// Thread / parallelism model
//
// NAFNet inference uses a thread pool.  Parallelisable layers
// (conv1x1, conv3x3, DW conv) are split across N_WORKERS
// independent compute threads that all share one accelerator.
// ------------------------------------------------------------
static const int N_WORKERS = 4;

// ------------------------------------------------------------
// Matrix accelerator tile dimensions
//
// Parallelisable convolutions are rewritten as GEMM and broken
// into M×K×N tiles:
//   M  = rows of the output (spatial positions per worker)
//   K  = depth (C_in × Kh × Kw)
//   N  = output channels (C_out)
//
// Each tile costs MATMUL_ACC_CYCLE cycles.
// Example: 8×8×8 = 512 MACs, 64 MAC units → 8 cycles/tile
// ------------------------------------------------------------
static const uint64_t MATMUL_M          = 8;
static const uint64_t MATMUL_K          = 8;
static const uint64_t MATMUL_N          = 8;
static const uint64_t MATMUL_ACC_CYCLE  = 8;  // cycles per tile

// ------------------------------------------------------------
// Depth-wise convolution accelerator
//
// DW conv cannot be expressed as a dense GEMM.  Instead it is
// modelled as a vector operation:  DWCONV_CAP output elements
// are produced every DWCONV_ACC_CYCLE cycles.
// ------------------------------------------------------------
static const uint64_t DWCONV_CAP        = 8;  // elements per batch
static const uint64_t DWCONV_ACC_CYCLE  = 4;  // cycles per batch

// ------------------------------------------------------------
// CPU scalar throughput
//
// Sequential layers (LayerNorm, SimpleGate, SCA pool/multiply,
// residual add, PixelShuffle) run on the scalar CPU pipeline.
// CPU_THROUGHPUT is the number of arithmetic ops per cycle.
// ------------------------------------------------------------
static const uint64_t CPU_THROUGHPUT    = 4;  // scalar ops / cycle

// ------------------------------------------------------------
// Shared memory subsystem
//
// Every accelerator request triggers memory traffic.
// Latency model:  cycles = MEM_BASE_LAT + ceil(bytes / MEM_BW)
// ------------------------------------------------------------
static const uint64_t MEM_BASE_LAT      = 10; // fixed base latency (cycles)
static const uint64_t MEM_BW            = 32; // bytes per cycle

// ------------------------------------------------------------
// Scalar overhead per accelerator call
//
// Accounts for address computation, tile dispatch bookkeeping,
// and any loop overhead that happens on the scalar CPU before
// and after each accelerator invocation.
// ------------------------------------------------------------
static const uint64_t SCALAR_OVERHEAD   = 8;  // cycles per call

// ------------------------------------------------------------
// Accelerator request queue depth
//
// Each accelerator (matrix and vector) holds at most
// ACC_QUEUE_DEPTH admitted requests at one time (including
// the one currently being serviced and those waiting in the
// internal FIFO queue).  Workers that try to issue a request
// when the queue is full are stalled until a slot opens.
// ------------------------------------------------------------
static const size_t ACC_QUEUE_DEPTH     = 4;  // max admitted requests per accelerator
