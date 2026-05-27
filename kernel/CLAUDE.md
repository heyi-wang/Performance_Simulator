## Objective: Build SystemC TLM simulation of different kernels in a NafNet with existing building blocks

# Kernels to simulate:
- depth-wise convolution
- layer normalization
- pooling 
- vector operations like element-wise multiplication

# Rules
- Use the current building blocks in the @src folder
- Keep the communication, synchronization and concurrency model consistent to @matmul_main.cpp
- The simulator of each kernels should be organized so that it will be easily integrated into the whole networks in the future
- Always ask me when you want to modify the files in @src folder

# Perfetto tracing
The kernel build defines `-DPERFETTO_TRACE` ([Makefile](Makefile)), so every
kernel sim can emit a Chrome-Trace-Event JSON timeline for
https://ui.perfetto.dev/ . Tracing is **opt-in per run** — pass `--trace-out
<file.json>`; with no flag, recording is gated off (one cheap branch per span
site, no file written), so the sweep scripts are unaffected. The trace sink is
[src/perfetto_trace.h](../src/perfetto_trace.h); spans come from the shared
`Worker` / `AcceleratorTLM` / `Memory` (see
[nafblock/CLAUDE.md](../nafblock/CLAUDE.md) for the group/lane scheme).

```
./kernel/build/matmul_sim --threads 4 --trace-out matmul.json
./kernel/build/layer_norm_sim --channels 32 --trace-out ln.json
```

Lanes per run: **Scalar Unit `<tid>`** (worker threads) with `scalar` +
`stall (matrix/vector/DMA FIFO full)` lanes, **Matrix/Vector Unit `<n>`**
(accelerator instances) with `load`/`compute`/`write`/`stall`, and **DMA
Engine** with `read`/`write`.

- matmul / layer_norm / dw_conv2d emit Scalar Unit lanes from the shared
  `src/Worker`. **pooling** (`PoolWorker`) and **vec_ops** (`VecOpsWorker`) run
  their own worker SC_THREADs; their `do_scalar` / `issue_begin` are
  instrumented directly (same `Scalar Unit` group + lanes), so they emit scalar
  and vector-FIFO-stall spans too. These two only issue to the vector pool, so
  their `stall (matrix FIFO full)` lane is always empty (declared for layout
  parity, alongside the always-empty `stall (DMA FIFO full)`).