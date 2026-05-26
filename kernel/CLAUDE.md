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

Lanes per run: **Scalar Unit `<tid>`** (worker threads), **Matrix/Vector Unit
`<n>`** (accelerator instances), **DMA Engine**.

- Scalar Unit lanes appear only for kernels that use the shared `src/Worker`
  (matmul, layer_norm, dw_conv2d). **pooling** (`PoolWorker`) and **vec_ops**
  (`VecOpsWorker`) run custom worker SC_THREADs that bypass the instrumented
  `Worker::do_scalar`/`issue_begin`, so they show accelerator + DMA lanes but no
  Scalar Unit lanes. Instrumenting those custom workers would require editing
  kernel execution code (ask first).