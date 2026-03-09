# Project Overview

This project is a performance simulator written in SystemC TLM-2.0.

The simulator models a hardware architecture consisting of:

- Multiple worker threads
- One matrix accelerator
- One vector accelerator
- A shared memory system

The purpose of the simulator is to estimate execution cycles of a workload.

---

# Architecture

Top Level Modules

Top
 ├── Workers (N)
 │     Generate accelerator requests
 │
 ├── MatrixAccelerator
 │     Processes matrix operations
 │
 ├── VectorAccelerator
 │     Processes vector operations
 │
 └── Memory
       Handles memory access latency

---

# Communication Model

The modules communicate via TLM sockets.

Workers → Accelerator
    initiator_socket

Accelerator → Memory
    initiator_socket

Memory → Accelerator
    target_socket

---

# Timing Model

Simulation is cycle-based.

CYCLE = 1ns

Accelerator operations take predefined cycles:
- Matrix op: X cycles
- Vector op: Y cycles
- Memory access: Z cycles

Workers must wait if the accelerator is busy.

---

# Design Rules

1. Accelerators are shared resources.
2. Request are sent in a non-blocking style.
3. Waiting threads accumulate cycle counts.
4. Memory latency must be included in total execution time.

---

# Coding Style

- Use SystemC modules defined as `struct`
- TLM communication uses:
  - `nb_transport` (non-blocking)

Extensions (`tlm_extension`) carry metadata such as:

- source thread id
- computation cycles
- memory traffic

# Reference Document

A implementation has been provided in @perf_sim.cpp. Check the correctness of this file. Rewrite it if necessary.
