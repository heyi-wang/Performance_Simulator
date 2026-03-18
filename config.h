#pragma once
#include <cstdint>

// ============================================================
// Convolution layer parameters
// Example: ResNet-50-style 3×3 conv
//   Input  [N, C_in, H_in, W_in]
//   Filter [C_out, C_in, KH, KW]
//   Output [N, C_out, H_out, W_out]
// ============================================================
static const uint64_t CONV_N      = 1;
static const uint64_t CONV_C_IN   = 64;
static const uint64_t CONV_H_IN   = 56;
static const uint64_t CONV_W_IN   = 56;
static const uint64_t CONV_C_OUT  = 64;
static const uint64_t CONV_KH     = 3;
static const uint64_t CONV_KW     = 3;
static const uint64_t CONV_STRIDE = 1;
static const uint64_t CONV_PAD    = 1;

// Derived output spatial dimensions
static const uint64_t CONV_H_OUT =
    (CONV_H_IN + 2 * CONV_PAD - CONV_KH) / CONV_STRIDE + 1;
static const uint64_t CONV_W_OUT =
    (CONV_W_IN + 2 * CONV_PAD - CONV_KW) / CONV_STRIDE + 1;

// Total MACs for the full layer (for throughput reporting)
static const uint64_t CONV_TOTAL_MACS =
    CONV_N * CONV_H_OUT * CONV_W_OUT *
    CONV_C_IN * CONV_KH * CONV_KW * CONV_C_OUT;

// ============================================================
// Thread count
// Each thread handles an equal slice of output spatial positions
//   (N × H_out × W_out) / NUM_THREADS rows
// ============================================================
static const int NUM_THREADS = 12;

// Number of physical accelerator instances per class.
// Requests still enter through one shared queue per class.
#ifndef MAT_ACCEL_COUNT
#define MAT_ACCEL_COUNT 2
#endif

#ifndef VEC_ACCEL_COUNT
#define VEC_ACCEL_COUNT 4
#endif

static const int MAT_ACCEL_COUNT_CFG = MAT_ACCEL_COUNT;
static const int VEC_ACCEL_COUNT_CFG = VEC_ACCEL_COUNT;

// ============================================================
// GEMM mapping via im2col
//   Convolution rewritten as: A [M_t × K] × B [K × C_out]
//     A_M = per-thread output rows  = N × H_out × W_out / T
//     A_K = filter depth            = C_in × KH × KW
//     B_N = output channels         = C_out
// ============================================================
static const uint64_t A_M =
    CONV_N * CONV_H_OUT * CONV_W_OUT / (uint64_t)NUM_THREADS;
static const uint64_t A_K = CONV_C_IN * CONV_KH * CONV_KW;
static const uint64_t B_N = CONV_C_OUT;

// ============================================================
// Hardware accelerator parameters
// ============================================================

// Matrix accelerator tile dimensions (must divide evenly into A_M, A_K, B_N
// for best efficiency; ceil-div is applied automatically in top.cpp)
static const uint64_t MATMUL_M = 8;
static const uint64_t MATMUL_K = 8;
static const uint64_t MATMUL_N = 8;

// Cycles to complete one M×K×N tile (= M*K*N MACs / mac_units_per_cycle)
// Example: 8×8×8 = 512 MACs, 64 MAC units → 8 cycles
static const uint64_t MATMUL_ACC_CYCLE = 8;

// Vector accelerator (bias add + activation, e.g. ReLU):
//   processes VECTOR_ACC_CAP elements per call in VECTOR_ACC_CYCLE cycles
static const uint64_t VECTOR_ACC_CAP   = 8;
static const uint64_t VECTOR_ACC_CYCLE = 4;

// Scalar overhead cycles per accelerator call
// (loop bookkeeping, address computation, tile dispatch)
static const uint64_t SCALAR_OVERHEAD = 1000;
