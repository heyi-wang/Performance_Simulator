# CLAUDE.md

## Project Goal

Build a **SystemC TLM-2.0 performance simulator** for a specific **NAFNet network**.

The simulator is based on an existing architecture skeleton that already models:

- multiple worker threads
- one shared matrix accelerator
- one shared vector accelerator
- one shared memory system

The simulator does **not** need to perform full neural-network numerical inference for correctness.  
Its main purpose is to **estimate execution cycles** of the NAFNet workload on the modeled hardware architecture.

The simulator should map NAFNet layers into hardware requests and simulate:

- compute latency
- memory latency
- accelerator contention
- worker waiting time
- total execution cycles
- per-layer timing breakdown

---

## Base Architecture

The simulator is built on the following hardware model:

Top
 ├── Workers (N)  
 │     Simulating CPU threads generating accelerator requests from layer tasks  
 │
 ├── MatrixAccelerator  
 │     Processes matrix-related operations  
 │
 ├── VectorAccelerator  
 │     Processes vector-related operations  
 │
 └── Memory  
       Handles shared memory access latency

This base simulator skeleton already provides the shared-resource architecture.  
The NAFNet simulator must be built **on top of this structure**, not as a completely separate design.

---

## Communication Model

Use **TLM-2.0 sockets** for communication.


### Transport style

Use the transport style already chosen in the base simulator skeleton.

If the base simulator already uses:

- `nb_transport_fw` / `nb_transport_bw`, then preserve the non-blocking architecture

Do **not** rewrite the simulator into a different communication style unless explicitly requested.

---

## Timing Model

Simulation is cycle-based.

- `CYCLE = 1 ns`

All latencies must be expressed in cycles and converted into `sc_time`.

The simulator must include:

- accelerator compute cycles
- memory access cycles
- waiting time when accelerators are busy
- total completion time of all worker tasks

The model is performance-oriented, so timing behavior is more important than actual arithmetic results.

---

## Purpose of the NAFNet Simulator

The NAFNet simulator should represent the execution of a NAFNet network as a sequence of layer tasks.

The simulator must estimate how the network runs on the hardware architecture by modeling:

- which accelerator each layer uses
- how many requests each layer generates
- how long each request takes
- how much memory traffic each layer causes
- how workers contend for shared resources
- how much total execution time the network needs

The simulator should support at least the following NAFNet-relevant operators first:

1. standard/pointwise convolution 
2. depth-wise convolution

Later it should be easy to extend to:

- elementwise add
- activation-like vector operations
- layer normalization
- pixel shuffle / reshape-like steps

---

## Mapping NAFNet Layers to Hardware

The simulator must map NAFNet layers to the base hardware as follows.

### 1. Standard convolution / 
Standard convolution is mainly modeled as a **matrix accelerator workload**.
After the matrix task, a vector task as **vector accelerator workload is followed to quantize the output back to smaller datatype

For each standard convolution layer, the simulator should estimate:

- total compute cycles on matrix accelerator
- memory reads for input feature map
- memory reads for weights
- memory writes for output feature map

The layer may be split into multiple accelerator requests if needed.

### 2. Depth-wise convolution
Depth-wise convolution should be modeled separately from normal convolution.

In the first version, depth-wise convolution should be mapped to the **vector accelerator** unless the user later specifies a different mapping.

This is because depth-wise convolution has lower cross-channel reuse and should be treated differently from dense convolution in the performance model.

For each depth-wise convolution layer, estimate:

- compute cycles on vector accelerator
- input memory reads
- weight memory reads
- output memory writes

### 3. Optional future mapping
If later requested, the simulator may support more detailed mappings, for example:

- pointwise conv → matrix accelerator
- activation / scale / add → vector accelerator
- normalization → vector accelerator
- reshape / shuffle → memory-only or lightweight vector operations

But the first implementation should focus on:

- standard conv
- depth-wise conv

---

## Design Rules

Always follow these rules:

1. Accelerators are shared resources.
2. Only one request can be processed on an accelerator at a time.
3. If multiple workers request the same accelerator, later requests must wait.
4. Waiting time contributes to the worker’s total cycle count.
5. Memory latency must be included in total layer latency.
6. Each accelerator may generate memory transactions while processing a request.
7. The simulator must preserve timing causality of the base SystemC/TLM model.
8. each nb_transport_fw call from the worker generates certain scalar overhead. 

---

## Worker Role in the NAFNet Simulator

Workers do not execute real convolution numerics.

Instead, workers act as **task generators** for the NAFNet workload.

Each worker should process assigned portions of the network workload and send requests to the correct accelerator.

Possible workload partitioning strategies:

- layer-by-layer assignment
- tile-based assignment
- output-channel partitioning
- spatial partitioning

For the first implementation, prefer the simplest clear strategy.

Recommended default:
- represent the NAFNet as an ordered layer list
- each worker issues requests corresponding to its assigned work units
- requests carry metadata describing the work

The simulator must be structured so that the partitioning strategy can be changed later without redesigning the whole system.

---

## Required Metadata in Transactions

Use `tlm_extension` to attach request metadata.

Each request extension should contain at least:

- source worker id
- layer id
- layer type
- computation cycles
- memory read bytes
- memory write bytes

Optional fields that may be added later:

- tile id
- input tensor dimensions
- output tensor dimensions
- kernel dimensions
- request start cycle
- queue waiting cycles
- memory service cycles

Example intent of metadata:
- identify which worker sent the request
- identify which layer the request belongs to
- allow accelerator and memory modules to compute timing correctly
- support reporting later

---

## NAFNet Layer Representation

Represent the NAFNet network explicitly as a list of layer descriptors.

Do **not** try to automatically parse arbitrary C source in the first version.

The first version should use a hardcoded structural description of the network.

Each NAFNet layer descriptor should contain at least:

- layer name
- layer type
- input height / width / channels
- output height / width / channels
- kernel size
- stride
- padding
- groups
- estimated compute cycles
- estimated memory traffic

Suggested layer types:

- `LAYER_CONV`
- `LAYER_DWCONV`

Future layer types may include:

- `LAYER_PWCONV`
- `LAYER_ADD`
- `LAYER_ACT`
- `LAYER_NORM`
- `LAYER_SHUFFLE`

The first task is to build a correct simulator for the first two.

---

## Performance Model

The simulator is cycle-based and uses predefined cycle assumptions.

For each NAFNet layer or tile, estimate:

- accelerator compute cycles
- memory read latency
- memory write latency
- waiting time due to accelerator contention

### Standard convolution
Treat as matrix accelerator workload.

The compute cycles may be estimated from convolution shape information such as:

- `Hout`
- `Wout`
- `Cin`
- `Cout`
- `Kh`
- `Kw`

A simple first model is acceptable, for example using total MAC count and accelerator throughput.

### Depth-wise convolution
Treat as vector accelerator workload in the first version.

Its cycle model should be separate from standard convolution, because its computation and memory behavior differ.

### Memory
Memory latency must be included whenever an accelerator fetches input/weights or writes back output.

The memory system is shared.  
If the existing base simulator models memory as a simple fixed-latency target, keep that behavior first.

Later extensions may model:

- bandwidth
- burst count
- read/write separation
- contention

But first keep it simple and consistent with the base simulator.

---

## Layer Latency Composition

For each request, total timing should conceptually include:

- time waiting for accelerator availability
- compute time on accelerator
- memory access time triggered by accelerator

For each worker, total cycles should accumulate:

- local scalar overhead if applicable
- waiting cycles
- accelerator service cycles
- memory cycles

For the whole network, the simulator should report:

- total simulated time
- per-worker accumulated cycles
- per-layer accumulated cycles
- per-accelerator utilization estimate

---

## Reporting Requirements

The simulator must produce a readable report.

At minimum, report:

### Global summary
- total simulation time
- total cycles
- number of workers
- number of requests sent to matrix accelerator
- number of requests sent to vector accelerator
- total memory transactions

### Per-layer summary
For each NAFNet layer, print:

- layer id
- layer name
- layer type
- assigned accelerator
- number of requests
- compute cycles
- memory bytes read
- memory bytes written
- estimated total cycles

### Per-worker summary
For each worker, print:

- worker id
- total accumulated cycles
- waiting cycles
- number of issued requests

### Optional later reports
- matrix accelerator busy timeline
- vector accelerator busy timeline
- memory access timeline
- CSV export

---

## Coding Style

Always follow these coding rules:

1. Use `struct` for SystemC modules.
2. Keep the existing base simulator style consistent.
3. Use explicit and readable code first.
4. Prefer beginner-friendly logic over over-engineered abstractions.
5. Keep comments in English.
6. Separate module responsibilities clearly.
7. Use clear names for layers, transactions, and timing fields.
8. Preserve the SystemC/TLM structure instead of hiding behavior in overly generic utilities.

---

