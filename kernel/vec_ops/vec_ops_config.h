#pragma once
#include <cstdint>
#include "hardware_config.h"

// ============================================================
// Hardware and tensor configuration for the Vector Operations
// TLM performance simulator.
//
// Supports the element-wise vector operations from vector_ops.h
// that NAFNet actually exercises:
//   - element-wise add  (mf_elemwise_add_i8)
//   - element-wise mul  (int8 in -> int16 out, widening vwmul_vv)
//   - scalar-broadcast mul (mf_elemwise_mul_scalar_i8)
//   - quantize int32 -> int8
//   - quantize int16 -> int8
//   - dequantize int8 -> int32
//
// Per Vec_Ops.md v1, ReLU and bias-add are deliberately excluded.
// ============================================================

// ------------------------------------------------------------
// Operation type enumeration.
// Each value maps to a kernel function in vector_ops.h with a
// distinct memory access pattern (different rd/wr bytes per tile)
// AND a distinct arithmetic-instruction count per RVV stripmining
// iteration.
// ------------------------------------------------------------
enum VopType
{
    VOP_ELEMWISE_ADD,           // mf_elemwise_add_i8           (rd=2*vl, wr=vl;  1 insn)
    VOP_ELEMWISE_MUL,           // widening int8 in -> int16 out (rd=2*vl, wr=2*vl; 1 insn)
    VOP_SCALAR_MUL,             // mf_elemwise_mul_scalar_i8    (rd=vl,   wr=vl;  1 insn)
    VOP_QUANTIZE_I32_TO_I8,     // mf_quantize_i32_to_i8        (rd=4*vl, wr=vl;  6 insns)
    VOP_QUANTIZE_I16_TO_I8,     // quant i16->i8                (rd=2*vl, wr=vl;  5 insns)
    VOP_DEQUANTIZE_I8_TO_I32,   // mf_dequantize_i8_to_i32      (rd=vl,   wr=4*vl;3 insns)
};

// ------------------------------------------------------------
// Active operation. Change this to simulate a different kernel.
// Default: ELEMWISE_MUL (used in NAFNet SCA attention module).
// ------------------------------------------------------------
static const VopType VOP_SELECTED_OP = VOP_ELEMWISE_MUL;

// ------------------------------------------------------------
// Input tensor: [C, H, W], channels-first layout. Element-wise
// ops produce an output of the same spatial shape (the precision
// of the output may differ from the input; see vop_*_elem_bytes).
// ------------------------------------------------------------
static const int VOP_C = 32;    // number of channels
static const int VOP_H = 64;    // spatial height
static const int VOP_W = 64;    // spatial width

// ------------------------------------------------------------
// Parallelism: number of worker SC_THREADs.
// Channels are distributed evenly across workers:
//   worker tid owns channels [c_start, c_end)
//   where c_start = (tid * VOP_C) / VOP_NUM_WORKERS.
// ------------------------------------------------------------
static const int VOP_NUM_WORKERS = 16;

// ------------------------------------------------------------
// Vector accelerator parameters. Per-request service cycles are
// derived from `vop_insn_count(op) * VOP_VEC_INSN_CYCLE` so that
// each operation's compute overhead matches the arithmetic
// intrinsic count in vector_ops.h (mirrors the layer_norm sim's
// LN_PASSk_INSNS_CFG pattern).
// ------------------------------------------------------------
static const uint64_t VOP_VEC_ACC_CAP       = VECTOR_ACC_CAP;
static const uint64_t VOP_VEC_INSN_CYCLE    = HW_VECTOR_INSN_CYCLE;
static const int      VOP_VEC_ACC_INSTANCES = VEC_ACCEL_COUNT;

// ------------------------------------------------------------
// Scalar CPU overhead per accelerator tile dispatch call.
// ------------------------------------------------------------
static const uint64_t VOP_SCALAR_OVERHEAD = HW_VEC_SCALAR_OVERHEAD;

// ------------------------------------------------------------
// L1 / L2 memory subsystem - aligned with pooling / layer_norm /
// dw_conv2d. L1 bw = VECTOR_ACC_CAP so one vector-wide load is
// delivered in 1 cycle; L2 bw = L1 / 4 with base latency 10 to
// reflect the slower L2 path. Worker prefetches one input channel
// from L2 -> L1 before issuing vec calls, then fires-and-forgets
// one L2 writeback per channel.
// ------------------------------------------------------------
#ifndef VOP_L1_BASE_LAT
#define VOP_L1_BASE_LAT 1
#endif
#ifndef VOP_L1_BW
#define VOP_L1_BW VECTOR_ACC_CAP
#endif
#ifndef VOP_L1_SLOTS
#define VOP_L1_SLOTS 8
#endif
#ifndef VOP_L2_BASE_LAT
#define VOP_L2_BASE_LAT 10
#endif
#ifndef VOP_L2_BW
#define VOP_L2_BW (VOP_L1_BW / 4)
#endif
#ifndef VOP_L2_SLOTS
#define VOP_L2_SLOTS 16
#endif

static const uint64_t VOP_L1_BASE_LAT_CFG = VOP_L1_BASE_LAT;
static const uint64_t VOP_L1_BW_CFG       = VOP_L1_BW;
static const uint64_t VOP_L1_SLOTS_CFG    = VOP_L1_SLOTS;
static const uint64_t VOP_L2_BASE_LAT_CFG = VOP_L2_BASE_LAT;
static const uint64_t VOP_L2_BW_CFG       = VOP_L2_BW;
static const uint64_t VOP_L2_SLOTS_CFG    = VOP_L2_SLOTS;

// ------------------------------------------------------------
// Accelerator queue depth - matches the other vec-only kernels.
// ------------------------------------------------------------
static const size_t VOP_ACC_QUEUE_DEPTH =
    HW_ACC_QUEUE_DEPTH > static_cast<size_t>(VEC_ACCEL_COUNT * 4)
        ? HW_ACC_QUEUE_DEPTH
        : static_cast<size_t>(VEC_ACCEL_COUNT * 4);

// ------------------------------------------------------------
// Per-DMA scalar setup overhead (vector-pipe DMA).
// ------------------------------------------------------------
#ifndef VOP_DMA_VEC_RD_SCALAR
#define VOP_DMA_VEC_RD_SCALAR HW_DMA_VEC_RD_SCALAR
#endif
#ifndef VOP_DMA_VEC_WR_SCALAR
#define VOP_DMA_VEC_WR_SCALAR HW_DMA_VEC_WR_SCALAR
#endif

static const uint64_t VOP_DMA_VEC_RD_SCALAR_CFG = VOP_DMA_VEC_RD_SCALAR;
static const uint64_t VOP_DMA_VEC_WR_SCALAR_CFG = VOP_DMA_VEC_WR_SCALAR;

// ------------------------------------------------------------
// Max outstanding L2 writeback DMAs per worker (fire-and-forget
// channel results). Reaped when full, fully drained at run end.
// ------------------------------------------------------------
#ifndef VOP_MAX_INFLIGHT_DMA_WRITES
#define VOP_MAX_INFLIGHT_DMA_WRITES 2
#endif

static const uint64_t VOP_MAX_INFLIGHT_DMA_WRITES_CFG = VOP_MAX_INFLIGHT_DMA_WRITES;

// ============================================================
// Per-op input / output element widths (bytes).
//
// For widening (ELEMWISE_MUL i8->i16) and narrowing/widening
// (quantize / dequantize) ops the L2 prefetch and writeback byte
// counts must use the *real* input vs. output width, not a single
// flat element size.
// ============================================================

static inline uint64_t vop_input_elem_bytes(VopType op)
{
    switch (op)
    {
    case VOP_ELEMWISE_ADD:         return 1;  // int8
    case VOP_ELEMWISE_MUL:         return 1;  // int8 inputs (widening)
    case VOP_SCALAR_MUL:           return 1;  // int8
    case VOP_QUANTIZE_I32_TO_I8:   return 4;  // int32
    case VOP_QUANTIZE_I16_TO_I8:   return 2;  // int16
    case VOP_DEQUANTIZE_I8_TO_I32: return 1;  // int8
    }
    return 1;
}

static inline uint64_t vop_output_elem_bytes(VopType op)
{
    switch (op)
    {
    case VOP_ELEMWISE_ADD:         return 1;  // int8
    case VOP_ELEMWISE_MUL:         return 2;  // int16 (widening result)
    case VOP_SCALAR_MUL:           return 1;  // int8
    case VOP_QUANTIZE_I32_TO_I8:   return 1;  // int8
    case VOP_QUANTIZE_I16_TO_I8:   return 1;  // int8
    case VOP_DEQUANTIZE_I8_TO_I32: return 4;  // int32
    }
    return 1;
}

// Element-wise add and mul read two input vectors per tile;
// every other op reads one input vector per tile.
static inline uint64_t vop_input_operand_count(VopType op)
{
    switch (op)
    {
    case VOP_ELEMWISE_ADD: return 2;
    case VOP_ELEMWISE_MUL: return 2;
    default:               return 1;
    }
}

// ============================================================
// RVV stripmining tile capacity per operation.
//
// VECTOR_ACC_CAP is treated as the e8m4 baseline tile capacity
// in *bytes*. The per-call element count `vl` is bounded by the
// widest operand: an int16 output halves the cap, an int32 input
// or output quarters it.
// ============================================================

static inline uint64_t vop_tile_cap_elems(VopType op)
{
    const uint64_t baseline = VOP_VEC_ACC_CAP;
    uint64_t max_width = vop_input_elem_bytes(op);
    if (vop_output_elem_bytes(op) > max_width)
        max_width = vop_output_elem_bytes(op);
    if (max_width <= 1)
        return baseline;
    return (baseline >= max_width) ? (baseline / max_width) : 1;
}

// ============================================================
// Per-op arithmetic instruction count (one RVV stripmining
// iteration). Counted from vector_ops.h, excluding vle / vse /
// vsetvl / identity vmv. Used to derive per-request service
// cycles via vop_service_cycles().
// ============================================================
static inline uint64_t vop_insn_count(VopType op)
{
    switch (op)
    {
    case VOP_ELEMWISE_ADD:         return 1;  // vadd_vv
    case VOP_ELEMWISE_MUL:         return 1;  // vwmul_vv (widening)
    case VOP_SCALAR_MUL:           return 1;  // vmul_vx
    case VOP_QUANTIZE_I32_TO_I8:   return 6;  // vsra,vadd,vmax,vmin,vnsra,vnsra
    case VOP_QUANTIZE_I16_TO_I8:   return 5;  // vsra,vadd,vmax,vmin,vnsra
    case VOP_DEQUANTIZE_I8_TO_I32: return 3;  // vwadd,vwadd,vmul_vx
    }
    return 1;
}

static inline uint64_t vop_service_cycles(VopType op)
{
    return vop_insn_count(op) * VOP_VEC_INSN_CYCLE;
}

// Back-compat alias for the nafnet bridge's independent
// expected-stats path ([nafnet/nafnet_layers.h]). The bridge
// still references a single flat per-request vec cycle; pick the
// most expensive op so its rough estimate is a worst-case rather
// than an under-count. The bridge's expected-stats logic is
// independently stale and needs its own update.
static const uint64_t VOP_VEC_ACC_CYCLE = 6 * VOP_VEC_INSN_CYCLE;

// ============================================================
// Per-tile rd / wr byte tables. `vl` is the per-call element
// count (bounded by vop_tile_cap_elems).
// ============================================================
static inline uint64_t vop_rd_bytes(VopType op, uint64_t vl)
{
    return vop_input_operand_count(op) * vl * vop_input_elem_bytes(op);
}

static inline uint64_t vop_wr_bytes(VopType op, uint64_t vl)
{
    return vl * vop_output_elem_bytes(op);
}

// Human-readable name for the selected operation (used in
// report + CLI --op flag).
static inline const char *vop_name(VopType op)
{
    switch (op)
    {
    case VOP_ELEMWISE_ADD:         return "mf_elemwise_add_i8";
    case VOP_ELEMWISE_MUL:         return "mf_elemwise_mul_i8_to_i16";
    case VOP_SCALAR_MUL:           return "mf_elemwise_mul_scalar_i8";
    case VOP_QUANTIZE_I32_TO_I8:   return "mf_quantize_i32_to_i8";
    case VOP_QUANTIZE_I16_TO_I8:   return "mf_quantize_i16_to_i8";
    case VOP_DEQUANTIZE_I8_TO_I32: return "mf_dequantize_i8_to_i32";
    }
    return "unknown";
}
