# CLAUDE.md

## Project goal

Build a **functionally abstract performance simulator** in **C** for a specific **NAFNet** network that is already implemented in C.

The simulator should **not** execute the real neural network numerics for correctness. Instead, it should estimate performance by modeling:

- layer-by-layer compute cost
- memory access cost
- data movement cost
- latency accumulation
- optional cycle breakdown by layer type

The target network contains:

- standard convolution layers
- depth-wise convolution layers

The simulator must be designed specifically for this network style, and the code structure should make it easy to extend later with:

- pointwise convolution
- activation
- elementwise add
- layer normalization
- pooling
- upsampling / pixel shuffle related operators

---

## High-level requirements

When generating code for this project, always follow these principles:

1. **Use C, not C++**
   - Do not use classes, templates, STL, or C++ syntax.
   - Use plain C structs, enums, arrays, and functions.

2. **Beginner-first coding style**
   - Write the first version in a very clear and explicit way.
   - Prefer readable loops and direct formulas over overly clever abstractions.
   - Comments should explain why each step exists.

3. **Performance simulator, not full inference engine**
   - Do not implement real convolution math unless explicitly requested.
   - The simulator only needs metadata such as tensor sizes, kernel sizes, channels, stride, padding, etc.
   - The purpose is cycle estimation and bottleneck analysis.

4. **Network-specific but structured**
   - The simulator should work well for this NAFNet-style network.
   - Still keep a modular structure so new operators can be added later.

5. **Deterministic and inspectable**
   - Every cycle contribution should come from a clear formula.
   - The program should be able to print a per-layer report and a final summary.

---

## What the simulator should model

The simulator should estimate the latency of each layer using a simplified hardware-aware model.

For each layer, model at least:

- input tensor shape
- output tensor shape
- kernel size
- stride
- padding
- number of channels
- operation type
- total MACs / ops
- estimated memory reads
- estimated memory writes
- estimated cycle count

For the whole network, report at least:

- total cycles
- total MACs
- total bytes read
- total bytes written
- per-layer cycle table
- percentage of time spent in:
  - normal convolution
  - depth-wise convolution
  - memory traffic

---

## Supported layer types

Implement these layer types first:

### 1. Standard convolution
A normal convolution with:

- `Cin` input channels
- `Cout` output channels
- kernel `Kh x Kw`
- output feature map `Hout x Wout`

Basic operation count:

- MACs = `Hout * Wout * Cout * Cin * Kh * Kw`

### 2. Depth-wise convolution
A depth-wise convolution with:

- one spatial kernel per input channel
- usually `groups = Cin = Cout`

Basic operation count:

- MACs = `Hout * Wout * Cin * Kh * Kw`

This must be modeled separately because its compute density and memory behavior differ from standard convolution.

---

## Performance model assumptions

Unless the user provides a different hardware model, assume a simple configurable accelerator model.

Use a parameter struct such as:

```c
typedef struct {
    unsigned long macs_per_cycle_conv;
    unsigned long macs_per_cycle_dwconv;
    unsigned long bytes_per_cycle_mem;
    unsigned long cache_line_bytes;
    unsigned long memory_latency_cycles;
    unsigned long kernel_launch_overhead_cycles;
} HardwareConfig;
