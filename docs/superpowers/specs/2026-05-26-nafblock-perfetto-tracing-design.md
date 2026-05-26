# NafBlock Perfetto Tracing — Design

**Date:** 2026-05-26
**Status:** Approved design, pending implementation plan
**Scope:** Standalone nafblock simulator (`nafblock/`) and its private build of the
shared `src/` + `kernel/*_top` objects. Other kernel sims and `nafnet/` are unaffected.

## Goal

Produce a timeline trace of one nafblock simulation that can be opened in
[ui.perfetto.dev](https://ui.perfetto.dev/), showing per-worker and per-accelerator
activity across the 14 sub-layers. The trace is for visualizing contention, idle
gaps, stalls, and pipeline overlap — it does not change any timing/modeling.

Non-goal: the legacy `nafnet/waveform.*` / VCD path is out of scope and untouched.

## Output format

Chrome Trace Event JSON — a flat top-level array of event objects, the simplest
format Perfetto ingests by drag-and-drop. Each activity interval is a complete
event:

```json
{"ph":"X","name":"compute","cat":"accel","pid":2,"tid":5,"ts":12.345,"dur":3.000}
```

- Simulation time is nanoseconds (`CYCLE` = 1 ns). Chrome-JSON `ts`/`dur` are
  microseconds, so values are emitted as `ns / 1000.0` (fractional doubles).
- The top-level object form is used so `"displayTimeUnit":"ns"` can be set:
  `{"displayTimeUnit":"ns","traceEvents":[ ... ]}`. Perfetto then labels the
  timeline in ns.

## Gating (compile-time)

All tracing is wrapped in `#ifdef PERFETTO_TRACE`.

- The nafblock [Makefile](../../../nafblock/Makefile) adds `-DPERFETTO_TRACE` to
  `CXXFLAGS`. Because nafblock compiles its own object copies into
  `nafblock/build/` (it does **not** share `kernel/build/`), this flag is fully
  isolated to nafblock.
- No other makefile defines the macro. Kernel sims and `nafnet/` compile
  byte-identical and incur zero runtime cost.
- Call sites use a macro that expands to nothing when the flag is off, so no
  `#ifdef` clutter is needed at each site.

## New file: `src/perfetto_trace.h`

A self-contained, header-only global trace sink (mirrors how the legacy
`waveform.h` was header-only, but is independent of it).

Contents:

- `struct PerfSpan { const char *group; std::string track; std::string name;
  uint64_t ts_ns; uint64_t dur_ns; };`
- A singleton accessor `std::vector<PerfSpan> &perf_spans();` (inline, function-local
  static).
- A recording function `inline void perf_trace_record(const char *group,
  std::string track, std::string name, uint64_t ts_ns, uint64_t dur_ns);` that
  appends a span.
- A macro:
  ```c++
  #ifdef PERFETTO_TRACE
  #  define PERF_TRACE_SPAN(group, track, name, ts_ns, dur_ns) \
            perf_trace_record((group), (track), (name), (ts_ns), (dur_ns))
  #else
  #  define PERF_TRACE_SPAN(group, track, name, ts_ns, dur_ns) ((void)0)
  #endif
  ```
- `inline void perf_trace_write_json(const char *path);`
  - Assigns a stable integer `pid` per distinct `group` (first-seen order) and a
    stable `tid` per distinct `track` within everything (a global track→tid map is
    fine; Perfetto groups by pid).
  - Emits `process_name` metadata events (`ph:"M"`) for each group and
    `thread_name` metadata events for each track.
  - Emits one `ph:"X"` event per span with `ts = ts_ns/1000.0`,
    `dur = dur_ns/1000.0`.
  - Writes `{"displayTimeUnit":"ns","traceEvents":[...]}`.

The whole header is always includable; only the macro's expansion is gated, so
modules can `#include "perfetto_trace.h"` unconditionally.

## Instrumentation sites

All calls use `PERF_TRACE_SPAN(...)` (no-op when the macro is off). Spans only read
`sc_time_stamp()`; they never call `wait()` or change control flow.

### Worker (`src/worker.cpp`) — group `"Workers"`, track `"worker_<tid>"`

Worker tags spans by `tid`, so worker N is one continuous lane across all 14
layers. The layer name is encoded in the span `name` where available.

1. **`run` span** — in `Worker::run()`: capture `start` (already at line ~600) and
   `end` (already at line ~667). On completion, emit a span
   `[start_ns, end_ns)` named `"run"`. This is the worker's active window for the
   current layer.
2. **`scalar` spans** — in `Worker::do_scalar(cyc)`: capture `t0 =
   sc_time_stamp()` before `wait(cyc*CYCLE)`, emit `[t0, t0+cyc)` named
   `"scalar"`. Covers DMA-setup / scalar overhead intervals.
3. **`stall` spans** — in `Worker::issue_begin(...)` backpressure branch
   (lines ~224-227): the stall window `[t_stall_start, now)` is already computed;
   emit a span named `"stall"`.

### AcceleratorTLM (`src/accelerator.cpp`) — group `"Accelerators"`, track = module `name()`

The module's hierarchical `name()` already encodes mat-vs-vec and the unit index,
and is unique per layer instance (`sc_gen_unique_name`), so each layer's
accelerators appear as their own lanes.

- **Serial mode** (`service_thread`): emit one `"service"` span per request,
  `[t_start, now)`, where `t_start` is line ~152 and the end is at completion
  (line ~179/183). This is the occupied interval (queue dequeue → response).
- **Pipeline mode**: the `Entry` already captures `t_load_start/_end`,
  `t_compute_start/_end`, `t_write_start/_end`. Emit three sub-spans per request
  on the same track: `"load"`, `"compute"`, `"write"`, using those timestamps.
  Emitted from `write_thread` at completion (all six timestamps known there).

### Memory (`src/memory.cpp`) — group `"Memory"`, track = `name()` (+ `:L1` / `:DMA`)

Both `Memory::dispatch_thread` and `L1L2Memory::{l1,dma}_dispatch_thread` compute
the request latency at dispatch, so a span's start and duration are both known
immediately:

- `Memory::dispatch_thread` (line ~84): emit span `[t_start, t_start+mem_lat)`
  named by command (`"read"`/`"write"`), track = `name()`.
- `L1L2Memory::l1_dispatch_thread` (line ~227): emit `[now, now+lat)`, track =
  `name() + ":L1"`, name `"read"`/`"write"`.
- `L1L2Memory::dma_dispatch_thread` (line ~265): emit `[now, now+lat)`, track =
  `name() + ":DMA"`, name `"read"`/`"write"`.

## Emission

In `nafblock/nafblock_sim.cpp` `sc_main`, after `sc_start()` returns:

```c++
#ifdef PERFETTO_TRACE
    perf_trace_write_json(opts.trace_out.c_str());
#endif
```

CLI: add `--trace-out <path>` to `BlockOptions` / `parse_args`, default
`"nafblock_trace.json"`. The flag is parsed unconditionally (harmless in an
untraced build) but only consumed under the macro; to avoid an "unknown arg"
surprise in untraced builds, the flag is accepted in both builds and ignored when
the macro is off. Default behavior of a traced run: write `nafblock_trace.json` in
the cwd.

## Files touched

- **New:** `src/perfetto_trace.h`
- **Edit:** `src/worker.cpp` (3 span sites), `src/accelerator.cpp` (serial + pipeline
  span sites), `src/memory.cpp` (3 dispatch span sites) — all macro-guarded,
  `#include "perfetto_trace.h"` added.
- **Edit:** `nafblock/nafblock_sim.cpp` (CLI flag + write call).
- **Edit:** `nafblock/Makefile` (add `-DPERFETTO_TRACE`; add `perfetto_trace.h`
  to relevant object dependency lists).
- **Doc:** update `nafblock/CLAUDE.md` Status/Gotchas with the trace build + how to
  open the JSON.

## Verification

1. Build nafblock (`make nafblock`) — compiles clean with the macro.
2. Build a kernel sim (e.g. `make kernel-matmul`) — confirm it still compiles and
   its binary is unchanged in behavior (no macro, no trace output).
3. Run `./nafblock/build/nafblock_perf_sim --trace-out /tmp/nb.json` — exit 0,
   report unchanged vs. a baseline run (timing must be identical: tracing is
   observation-only).
4. Validate `/tmp/nb.json` is well-formed JSON (`python3 -m json.tool`).
5. Sanity-check span counts/extents: total trace span over all `run` spans should
   cover `[0, total_cycles]`; spot-check one accelerator lane has the expected
   request count for a known layer.
6. Open in ui.perfetto.dev and confirm worker lanes are continuous and accelerator
   lanes appear per layer.

## Risks / notes

- `src/` is shared infra; the rule is "don't modify `src/` without asking." This
  change is explicitly requested, additive, and fully macro-gated, so non-nafblock
  builds are unaffected.
- Span volume: with `scalar` spans + split pipeline sub-spans, large shapes can
  produce many spans. Acceptable for typical block shapes; if a run is huge the
  user can shrink `--block-*`. No cap is imposed in v1.
- Memory spans use dispatch-time latency, which is the modeled busy interval; this
  matches `busy_cycles` accounting and needs no end-time pairing.
