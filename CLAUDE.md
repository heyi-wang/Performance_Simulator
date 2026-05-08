# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SystemC TLM-2.0 performance simulator that estimates execution cycles for NAFNet-style workloads on a modeled accelerator architecture. The goal is cycle/timing estimation, not numerical inference correctness.

## Build & Run

Top-level [Makefile](Makefile) dispatches to per-area makefiles; it does not build a single monolithic binary.

- `make kernels` — build every standalone kernel simulator under [kernel/build/](kernel/build/) (`matmul_sim`, `dw_conv2d_sim`, `layer_norm_sim`, `pooling_sim`, `vec_ops_sim`).
- `make kernel-matmul` / `kernel-dwconv` / `kernel-layernorm` / `kernel-pooling` / `kernel-vecops` — build one kernel.
- `make nafnet` — build full network simulator [nafnet/build/nafnet_perf_sim](nafnet/build/nafnet_perf_sim) plus bridge test `nafnet_kernel_bridge_test`.
- `make -C nafnet run` / `make -C nafnet test` — execute the network sim or the bridge test.
- `make clean` — cleans `kernel/build/` only; `nafnet/build/` and the stray root `build/` are cleaned via their own makefiles.

Matmul sim accepts `--threads N` and `--accum-registers N` ([kernel/matmul/main.cpp:16-50](kernel/matmul/main.cpp#L16-L50)). Exit code 2 indicates verification/req-count mismatch. Parameter sweeps live in [kernel/matmul/sweep_workers.py](kernel/matmul/sweep_workers.py).

Build flags: `g++ -std=c++17`, links `-lsystemc -lpthread`, expects SystemC headers at `/usr/include` and lib at `/usr/lib/x86_64-linux-gnu`. Override via `EXTRA_CXXFLAGS`.

## Architecture

Shared hardware building blocks live in [src/](src/) and are linked into every simulator. Kernel-specific top modules and workload geometry live under [kernel/<op>/](kernel/). The full-network sim lives in [nafnet/](nafnet/) and reuses the same kernel tops via a bridge.

Data path: `Worker` → `Interconnect` → `AcceleratorPool` (shared FIFO queue in front of N identical `AcceleratorTLM` units) → `Memory`. One pool per accelerator class — matrix and vector. Memory models bandwidth via a bounded parallel-slot queue.

Communication: TLM-2.0 non-blocking (`nb_transport_fw` / `nb_transport_bw`) only. Do not convert to blocking. Each request carries two extensions defined in [src/extensions.h](src/extensions.h):
- `ReqExt` — compute cycles, read/write byte counts, queue-wait accounting.
- `TxnExt` — routing context (`src_worker`, `upstream_id`) and `done_ev`/`admit_ev` event pointers used to wake the worker on BEGIN_RESP and deferred END_REQ.

Backpressure protocol: if the pool queue is full, `nb_transport_fw` returns `TLM_ACCEPTED` instead of `TLM_UPDATED`; the worker blocks on `admit_ev` until the pool sends a deferred END_REQ to grant the slot. Stall cycles are tracked separately from queue-wait cycles — see [src/worker.h:55-66](src/worker.h#L55-L66).

Worker pipeline is per-accelerator-class: `issue_begin[i] → do_scalar (∥ accel services i) → issue_end[i] → issue_begin[i+1]`. Inflight depth is controlled by `max_inflight_{mat,vec}_reqs`. DMA requests go directly to memory via `issue_dma_begin` / `finish_dma`.

## Configuration

Shared hardware knobs are in [config/hardware_config.h](config/hardware_config.h) — accelerator counts, queue depth, memory bandwidth, tile sizes (`MATMUL_M/K/N`), vector lane width, scalar dispatch overhead. They're `#ifndef`-guarded so each kernel or sweep run can override via `-D` on the compile line.

Per-kernel workload geometry (e.g. GEMM dims, conv shape) lives in `kernel/<op>/<op>_config.h`. NAFNet network descriptors live in [nafnet/nafnet_layers.h](nafnet/nafnet_layers.h) and [nafnet/nafnet_hw_config.h](nafnet/nafnet_hw_config.h).

## Reporting

`KernelWorkerInfo` and `AccelInstanceStats` in [src/common.h](src/common.h) are the canonical per-worker / per-accelerator stat structs. All sims emit a report with global summary + per-worker + per-accelerator + (for nafnet) per-layer breakdown through [src/report_formatter.h](src/report_formatter.h).

## Scoped Instructions

These subdirectories have their own `CLAUDE.md` with stricter rules that apply when editing within them — read them before making changes:

- [kernel/CLAUDE.md](kernel/CLAUDE.md) — never modify `src/` without asking; keep communication model consistent with matmul.
- [nafnet/CLAUDE.md](nafnet/CLAUDE.md) — layer-by-layer generation workflow, required report fields, do not rewrite the comm style, do not modify kernel execution logic without being asked.
- [matmul_sim/CLAUDE.md](matmul_sim/CLAUDE.md) — matmul report formatting requirements (per-thread mat/vec-reduction/vec-quant call counts, stall cycles, slowest-worker elapsed).

## Conventions

- 1 simulation cycle = 1 ns (`CYCLE` in [src/common.h](src/common.h#L11)).
- SystemC modules are `struct` with `SC_HAS_PROCESS`; use `SC_THREAD` for worker/service loops and `peq_with_get` for backward-path event demux.
- For multi-worker layers in nafnet, the next layer starts only after all workers finish the current one.
- The root `perf_sim` binary and `build/` directory are stale leftovers from an earlier flat layout — the active sources are under `src/`, `kernel/`, and `nafnet/`.

## Requirements
- **Do Not** change the structure/moeling strategy of the simulator unless explicitly asked.
