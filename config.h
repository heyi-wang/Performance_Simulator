#pragma once
#include <cstdint>

// ============================================================
// Simulation configuration parameters
// ============================================================

// Number of parallel worker threads
static const int NUM_THREADS = 16;

// Overall problem: A[A_M x A_K] * B[A_K x B_N] = C[A_M x B_N]
static const uint64_t A_M = 1024;
static const uint64_t A_K = 1024;
static const uint64_t B_N =  512;

// Matrix accelerator tile dimensions
static const uint64_t MATMUL_M = 8;
static const uint64_t MATMUL_K = 8;
static const uint64_t MATMUL_N = 8;

// Vector accelerator: number of elements processed per invocation
static const uint64_t VECTOR_ACC_CAP = 16;

// Cycles consumed by one matrix tile accelerator operation
static const uint64_t MATMUL_ACC_CYCLE = 16;

// Cycles consumed by one vector accelerator invocation
static const uint64_t VECTOR_ACC_CYCLE = 16;

// Scalar (host-side) overhead cycles between successive accelerator calls
static const uint64_t SCALAR_OVERHEAD = 16;
