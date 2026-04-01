#pragma once
#include <cstdint>
#include "../config.h"

// ============================================================
// Hardware and tensor configuration for the Vector Operations
// TLM performance simulator.
//
// Supports multiple element-wise vector operations from
// vector_ops.h.  Select the operation to simulate by setting
// VOP_SELECTED_OP below.
//
// Default input precision is int8 (VOP_ELEM_BYTES = 1) per
// Vec_Ops.md.  Change VOP_ELEM_BYTES to switch precision.
// ============================================================

// ------------------------------------------------------------
// Operation type enumeration.
// Each value maps to a kernel function in vector_ops.h with a
// distinct memory access pattern (different rd/wr bytes per tile).
// NAFNet does not use activation functions, so ReLU is omitted.
// ------------------------------------------------------------
enum VopType
{
    VOP_ELEMWISE_ADD,        // mf_elemwise_add_i8:       rd = 2*vl*elem, wr = vl*elem
    VOP_ELEMWISE_MUL,        // mf_elemwise_mul_i8:       rd = 2*vl*elem, wr = vl*elem
    VOP_SCALAR_MUL,          // mf_elemwise_mul_scalar_i8: rd = vl*elem,  wr = vl*elem
    VOP_QUANTIZE_I32_TO_I8,  // mf_quantize_i32_to_i8:   rd = vl*4,     wr = vl*1
    VOP_DEQUANTIZE_I8_TO_I32,// mf_dequantize_i8_to_i32: rd = vl*1,     wr = vl*4
    VOP_BIAS_ADD_I32,        // mf_bias_add_i32:          rd = vl*4,     wr = vl*4
};

// ------------------------------------------------------------
// Active operation.  Change this to simulate a different kernel.
// Default: ELEMWISE_MUL (used in NAFNet SCA attention module).
// ------------------------------------------------------------
static const VopType VOP_SELECTED_OP = VOP_QUANTIZE_I32_TO_I8;

// ------------------------------------------------------------
// Base element size (bytes).
// For int8 operations:  VOP_ELEM_BYTES = 1
// For int16 operations: VOP_ELEM_BYTES = 2
// ------------------------------------------------------------
static const uint64_t VOP_ELEM_BYTES = 1;   // int8

// ------------------------------------------------------------
// Input tensor: [C, H, W], channels-first layout.
// Element-wise ops produce an output of the same spatial shape
// (possibly at a different precision for quantize/dequantize).
// ------------------------------------------------------------
static const int VOP_C = 32;    // number of channels
static const int VOP_H = 64;    // spatial height
static const int VOP_W = 64;    // spatial width

// ------------------------------------------------------------
// Parallelism: number of worker SC_THREADs.
// Channels are distributed evenly across workers:
//   worker tid owns channels [c_start, c_end)
//   where c_start = (tid * VOP_C) / VOP_NUM_WORKERS
// ------------------------------------------------------------
static const int VOP_NUM_WORKERS = 16;

// ------------------------------------------------------------
// Vector accelerator parameters.
// ------------------------------------------------------------
static const uint64_t VOP_VEC_ACC_CAP       = VECTOR_ACC_CAP;
static const uint64_t VOP_VEC_ACC_CYCLE     = VECTOR_ACC_CYCLE;
static const int      VOP_VEC_ACC_INSTANCES = VEC_ACCEL_COUNT;

// ------------------------------------------------------------
// Scalar CPU overhead per accelerator tile dispatch call.
// ------------------------------------------------------------
static const uint64_t VOP_SCALAR_OVERHEAD = SCALAR_OVERHEAD;

// ------------------------------------------------------------
// Shared memory subsystem.
// Latency model: cycles = VOP_MEM_BASE_LAT + ceil(bytes / VOP_MEM_BW)
// ------------------------------------------------------------
static const uint64_t VOP_MEM_BASE_LAT = 1;
static const uint64_t VOP_MEM_BW       = 64;

// ------------------------------------------------------------
// Accelerator queue depth.
// ------------------------------------------------------------
static const size_t VOP_ACC_QUEUE_DEPTH = 32;

// ============================================================
// RVV stripmining shape per operation.
//
// VECTOR_ACC_CAP is treated as the e8m4 baseline tile capacity:
//   e8m4  -> 1x baseline
//   e16m4 -> 1/2 baseline
//   e32m4 -> 1/4 baseline
//   e8m1  -> 1/4 baseline
// ============================================================

enum class VopVecShape
{
    E8M4,
    E16M4,
    E32M4,
    E8M1,
};

static inline VopVecShape vop_vec_shape(VopType op)
{
    switch (op)
    {
    case VOP_ELEMWISE_ADD:         return VopVecShape::E8M4;
    case VOP_ELEMWISE_MUL:         return VopVecShape::E8M4;
    case VOP_SCALAR_MUL:           return VopVecShape::E8M4;
    case VOP_QUANTIZE_I32_TO_I8:   return VopVecShape::E32M4;
    case VOP_DEQUANTIZE_I8_TO_I32: return VopVecShape::E8M1;
    case VOP_BIAS_ADD_I32:         return VopVecShape::E32M4;
    }
    return VopVecShape::E8M4;
}

static inline uint64_t vop_tile_cap_elems(VopType op)
{
    switch (vop_vec_shape(op))
    {
    case VopVecShape::E8M4:  return VOP_VEC_ACC_CAP;
    case VopVecShape::E16M4: return (VOP_VEC_ACC_CAP >= 2) ? (VOP_VEC_ACC_CAP / 2) : 1;
    case VopVecShape::E32M4: return (VOP_VEC_ACC_CAP >= 4) ? (VOP_VEC_ACC_CAP / 4) : 1;
    case VopVecShape::E8M1:  return (VOP_VEC_ACC_CAP >= 4) ? (VOP_VEC_ACC_CAP / 4) : 1;
    }
    return VOP_VEC_ACC_CAP;
}

// ============================================================
// Per-tile read/write byte computation.
//
// Each RVV stripmining iteration processes `vl` elements.
// The number of bytes read/written depends on the operation:
//
//   ELEMWISE_ADD/MUL:  two input vectors + one output vector
//   SCALAR_MUL:        one input vector  + one output vector
//   QUANTIZE i32→i8:   read int32, write int8
//   DEQUANTIZE i8→i32: read int8, write int32
//   BIAS_ADD i32:      read+write int32 (in-place with broadcast scalar)
// ============================================================

static inline uint64_t vop_rd_bytes(VopType op, uint64_t vl)
{
    switch (op)
    {
    case VOP_ELEMWISE_ADD:         return vl * VOP_ELEM_BYTES * 2;
    case VOP_ELEMWISE_MUL:         return vl * VOP_ELEM_BYTES * 2;
    case VOP_SCALAR_MUL:           return vl * VOP_ELEM_BYTES;
    case VOP_QUANTIZE_I32_TO_I8:   return vl * 4;
    case VOP_DEQUANTIZE_I8_TO_I32: return vl * 1;
    case VOP_BIAS_ADD_I32:         return vl * 4;
    }
    return 0;
}

// Extra scalar-side memory reads performed once per channel
// outside the vector tile loop.
static inline uint64_t vop_extra_rd_bytes_per_channel(VopType op)
{
    switch (op)
    {
    case VOP_BIAS_ADD_I32:         return 4;
    case VOP_ELEMWISE_ADD:
    case VOP_ELEMWISE_MUL:
    case VOP_SCALAR_MUL:
    case VOP_QUANTIZE_I32_TO_I8:
    case VOP_DEQUANTIZE_I8_TO_I32:
        return 0;
    }
    return 0;
}

static inline uint64_t vop_wr_bytes(VopType op, uint64_t vl)
{
    switch (op)
    {
    case VOP_ELEMWISE_ADD:         return vl * VOP_ELEM_BYTES;
    case VOP_ELEMWISE_MUL:         return vl * VOP_ELEM_BYTES;
    case VOP_SCALAR_MUL:           return vl * VOP_ELEM_BYTES;
    case VOP_QUANTIZE_I32_TO_I8:   return vl * 1;
    case VOP_DEQUANTIZE_I8_TO_I32: return vl * 4;
    case VOP_BIAS_ADD_I32:         return vl * 4;
    }
    return 0;
}

// Human-readable name for the selected operation (used in report).
static inline const char *vop_name(VopType op)
{
    switch (op)
    {
    case VOP_ELEMWISE_ADD:         return "mf_elemwise_add_i8";
    case VOP_ELEMWISE_MUL:         return "mf_elemwise_mul_i8";
    case VOP_SCALAR_MUL:           return "mf_elemwise_mul_scalar_i8";
    case VOP_QUANTIZE_I32_TO_I8:   return "mf_quantize_i32_to_i8";
    case VOP_DEQUANTIZE_I8_TO_I32: return "mf_dequantize_i8_to_i32";
    case VOP_BIAS_ADD_I32:         return "mf_bias_add_i32";
    }
    return "unknown";
}
