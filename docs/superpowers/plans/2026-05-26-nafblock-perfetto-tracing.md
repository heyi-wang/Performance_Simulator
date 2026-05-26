# NafBlock Perfetto Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit a Chrome-Trace-Event JSON timeline from the nafblock simulator showing per-worker, per-accelerator, and per-memory activity across the 14 sub-layers, openable in ui.perfetto.dev.

**Architecture:** A header-only global span sink (`src/perfetto_trace.h`) records `(group, track, name, ts_ns, dur_ns)` spans. Shared modules (`Worker`, `AcceleratorTLM`, `Memory`/`L1L2Memory`) record spans at points where begin/end timestamps already exist, via a `PERF_TRACE_SPAN` macro that is a no-op unless `PERFETTO_TRACE` is defined. Only the nafblock Makefile defines that macro, so all other sims compile byte-identical with zero cost. After `sc_stop()`, nafblock writes the JSON.

**Tech Stack:** C++17, SystemC TLM-2.0, g++, Make. Chrome Trace Event JSON output. Python3 `json.tool` for validation.

**Spec:** [docs/superpowers/specs/2026-05-26-nafblock-perfetto-tracing-design.md](../specs/2026-05-26-nafblock-perfetto-tracing-design.md)

---

## File Structure

- **Create** `src/perfetto_trace.h` — global span buffer, `PERF_TRACE_SPAN` macro, `perf_trace_write_json()`. No SystemC dependency (pure C++), so it is unit-testable on its own.
- **Create** `src/perfetto_trace_test.cpp` — standalone unit test for the JSON writer (compiled directly with g++, no SystemC, no Make rule needed).
- **Modify** `src/worker.cpp` — add `#include "perfetto_trace.h"` + 3 span sites (run, scalar, stall).
- **Modify** `src/accelerator.cpp` — add include + service span (serial) and load/compute/write spans (pipeline).
- **Modify** `src/memory.cpp` — add include + dispatch spans for `Memory` and `L1L2Memory` (L1/DMA).
- **Modify** `nafblock/nafblock_sim.cpp` — `--trace-out` CLI flag + macro-guarded `perf_trace_write_json` call.
- **Modify** `nafblock/Makefile` — add `-DPERFETTO_TRACE`; add `perfetto_trace.h` to dependency lists of the touched objects.
- **Modify** `nafblock/CLAUDE.md` — document the trace build, the `--trace-out` flag, and how to open the JSON.

---

### Task 1: Trace sink header + JSON writer (TDD)

**Files:**
- Create: `src/perfetto_trace.h`
- Test: `src/perfetto_trace_test.cpp`

- [ ] **Step 1: Write the failing test**

Create `src/perfetto_trace_test.cpp`:

```cpp
// Standalone unit test for perfetto_trace.h JSON writer.
// Build: g++ -std=c++17 -DPERFETTO_TRACE -I src src/perfetto_trace_test.cpp -o /tmp/pt_test
#include "perfetto_trace.h"

#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

static std::string slurp(const char *path)
{
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main()
{
    perf_spans().clear();

    // Two groups, two tracks, ns->us conversion (1500 ns -> 1.5 us).
    PERF_TRACE_SPAN("Workers", "worker_0", "run", 0, 1500);
    PERF_TRACE_SPAN("Accelerators", "nb_mat.accel_0", "service", 1000, 2000);

    assert(perf_spans().size() == 2);

    const char *out = "/tmp/pt_test_out.json";
    perf_trace_write_json(out);
    const std::string j = slurp(out);

    // Format expectations.
    assert(j.find("\"displayTimeUnit\":\"ns\"") != std::string::npos);
    assert(j.find("\"traceEvents\"") != std::string::npos);
    // Metadata names present.
    assert(j.find("\"Workers\"") != std::string::npos);
    assert(j.find("\"worker_0\"") != std::string::npos);
    assert(j.find("\"nb_mat.accel_0\"") != std::string::npos);
    // Span names present.
    assert(j.find("\"run\"") != std::string::npos);
    assert(j.find("\"service\"") != std::string::npos);
    // ns->us conversion: 1500ns -> ts/dur 1.5 ; 2000ns -> 2
    assert(j.find("1.5") != std::string::npos);
    // Two distinct pids (one per group): 0 and 1 used in process_name metadata.
    assert(j.find("\"pid\":0") != std::string::npos);
    assert(j.find("\"pid\":1") != std::string::npos);

    std::cout << "perfetto_trace_test PASSED\n";
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `g++ -std=c++17 -DPERFETTO_TRACE -I src src/perfetto_trace_test.cpp -o /tmp/pt_test`
Expected: FAIL — compile error, `perfetto_trace.h` does not exist (`fatal error: perfetto_trace.h: No such file or directory`).

- [ ] **Step 3: Write minimal implementation**

Create `src/perfetto_trace.h`:

```cpp
#pragma once

// ============================================================
// perfetto_trace.h — header-only global span sink that emits a
// Chrome Trace Event JSON file openable in ui.perfetto.dev.
//
// All recording goes through PERF_TRACE_SPAN(...), which expands
// to nothing unless PERFETTO_TRACE is defined at compile time.
// The writer (perf_trace_write_json) is always compiled so a test
// harness can exercise it directly.
//
// No SystemC dependency: callers convert sc_time to ns themselves.
// ============================================================

#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct PerfSpan
{
    const char *group;   // process lane label (string literal; stable lifetime)
    std::string track;   // thread lane label within the group
    std::string name;    // span label
    uint64_t    ts_ns;   // start time, nanoseconds
    uint64_t    dur_ns;  // duration, nanoseconds
};

// Singleton span buffer (inline so the header is self-contained).
inline std::vector<PerfSpan> &perf_spans()
{
    static std::vector<PerfSpan> v;
    return v;
}

inline void perf_trace_record(const char *group,
                              std::string track,
                              std::string name,
                              uint64_t    ts_ns,
                              uint64_t    dur_ns)
{
    perf_spans().push_back(
        PerfSpan{group, std::move(track), std::move(name), ts_ns, dur_ns});
}

#ifdef PERFETTO_TRACE
#  define PERF_TRACE_SPAN(group, track, name, ts_ns, dur_ns) \
        perf_trace_record((group), (track), (name), (ts_ns), (dur_ns))
#else
#  define PERF_TRACE_SPAN(group, track, name, ts_ns, dur_ns) ((void)0)
#endif

// Emit nanoseconds as fractional microseconds (Chrome-JSON ts/dur unit),
// trimming trailing zeros so 1500ns -> "1.5", 2000ns -> "2".
inline std::string perf_ns_to_us(uint64_t ns)
{
    uint64_t whole = ns / 1000;
    uint64_t frac  = ns % 1000;        // 0..999
    std::string s = std::to_string(whole);
    if (frac == 0)
        return s;
    char buf[4] = {
        static_cast<char>('0' + (frac / 100) % 10),
        static_cast<char>('0' + (frac / 10) % 10),
        static_cast<char>('0' + frac % 10),
        '\0'};
    std::string f = buf;
    while (!f.empty() && f.back() == '0')
        f.pop_back();
    return s + "." + f;
}

inline void perf_trace_write_json(const char *path)
{
    std::ofstream f(path);
    if (!f)
        return;

    // Stable pid per group (first-seen order); stable tid per (group,track).
    std::vector<const char *>     group_order;
    std::map<std::string, int>    group_pid;
    std::map<std::string, int>    track_tid;   // key: "<pid>\0<track>"
    int next_tid = 0;

    auto pid_for = [&](const char *g) -> int {
        auto it = group_pid.find(g);
        if (it != group_pid.end())
            return it->second;
        int pid = static_cast<int>(group_order.size());
        group_pid[g] = pid;
        group_order.push_back(g);
        return pid;
    };
    auto tid_for = [&](int pid, const std::string &track) -> int {
        std::string key = std::to_string(pid) + "\x1f" + track;
        auto it = track_tid.find(key);
        if (it != track_tid.end())
            return it->second;
        int tid = next_tid++;
        track_tid[key] = tid;
        return tid;
    };

    f << "{\"displayTimeUnit\":\"ns\",\"traceEvents\":[\n";
    bool first = true;
    auto comma = [&]() { if (!first) f << ",\n"; first = false; };

    // 1. process_name + thread_name metadata.
    for (const auto &s : perf_spans())
    {
        int pid = pid_for(s.group);
        (void)tid_for(pid, s.track);
    }
    for (const char *g : group_order)
    {
        comma();
        f << "{\"ph\":\"M\",\"name\":\"process_name\",\"pid\":"
          << group_pid[g] << ",\"args\":{\"name\":\"" << g << "\"}}";
    }
    for (const auto &kv : track_tid)
    {
        // key = "<pid>\x1f<track>"
        const std::string &key = kv.first;
        size_t sep = key.find('\x1f');
        int pid = std::stoi(key.substr(0, sep));
        std::string track = key.substr(sep + 1);
        comma();
        f << "{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":" << pid
          << ",\"tid\":" << kv.second
          << ",\"args\":{\"name\":\"" << track << "\"}}";
    }

    // 2. Spans.
    for (const auto &s : perf_spans())
    {
        int pid = group_pid[s.group];
        int tid = track_tid[std::to_string(pid) + "\x1f" + s.track];
        comma();
        f << "{\"ph\":\"X\",\"name\":\"" << s.name << "\",\"pid\":" << pid
          << ",\"tid\":" << tid
          << ",\"ts\":" << perf_ns_to_us(s.ts_ns)
          << ",\"dur\":" << perf_ns_to_us(s.dur_ns) << "}";
    }

    f << "\n]}\n";
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
g++ -std=c++17 -DPERFETTO_TRACE -I src src/perfetto_trace_test.cpp -o /tmp/pt_test && /tmp/pt_test && python3 -m json.tool /tmp/pt_test_out.json > /dev/null && echo "JSON VALID"
```
Expected: `perfetto_trace_test PASSED` then `JSON VALID`.

- [ ] **Step 5: Verify no-op build compiles (macro off)**

Run: `g++ -std=c++17 -I src -fsyntax-only -x c++ - <<'EOF'
#include "perfetto_trace.h"
int main(){ PERF_TRACE_SPAN("g","t","n",1,2); return (int)perf_spans().size(); }
EOF`
Expected: no output, exit 0 (the macro compiled to `((void)0)`, buffer stays empty).

- [ ] **Step 6: Commit**

```bash
git add src/perfetto_trace.h src/perfetto_trace_test.cpp
git commit -m "feat: add header-only Perfetto trace sink + JSON writer

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Instrument Worker spans

**Files:**
- Modify: `src/worker.cpp` (add include; sites in `do_scalar`, `issue_begin`, `run`)

- [ ] **Step 1: Add the include**

At the top of `src/worker.cpp`, after the existing includes (after `#include <iostream>`), add:

```cpp
#include "perfetto_trace.h"
```

- [ ] **Step 2: Scalar span in `do_scalar`**

Replace the body of `Worker::do_scalar`:

```cpp
void Worker::do_scalar(uint64_t cyc)
{
    compute_cycles += cyc;
    const uint64_t t0 = static_cast<uint64_t>(sc_time_stamp() / CYCLE);
    wait(cyc * CYCLE);
    PERF_TRACE_SPAN("Workers", "worker_" + std::to_string(tid), "scalar",
                    t0, cyc);
}
```

- [ ] **Step 3: Stall span in `issue_begin`**

In `Worker::issue_begin(addr, svc_cycles, rd, wr)`, the backpressure branch currently reads:

```cpp
    if (status == TLM_ACCEPTED)
    {
        // Queue was full: stall until the accelerator grants a slot.
        sc_time t_stall_start = sc_time_stamp();
        if (!p.done_entry->admit_fired)
            wait(p.done_entry->admit_ev);
        p.stall_cycles = (uint64_t)((sc_time_stamp() - t_stall_start) / CYCLE);
    }
```

Replace it with (adds a span using the already-computed window):

```cpp
    if (status == TLM_ACCEPTED)
    {
        // Queue was full: stall until the accelerator grants a slot.
        sc_time t_stall_start = sc_time_stamp();
        if (!p.done_entry->admit_fired)
            wait(p.done_entry->admit_ev);
        p.stall_cycles = (uint64_t)((sc_time_stamp() - t_stall_start) / CYCLE);
        PERF_TRACE_SPAN("Workers", "worker_" + std::to_string(tid), "stall",
                        static_cast<uint64_t>(t_stall_start / CYCLE),
                        p.stall_cycles);
    }
```

- [ ] **Step 4: Run span in `run`**

At the end of `Worker::run()`, the tail currently reads:

```cpp
    sc_time end        = sc_time_stamp();
    elapsed_cycles     = (uint64_t)((end - start) / CYCLE);
    if (completion_fifo)
        completion_fifo->write(tid);
```

Replace with:

```cpp
    sc_time end        = sc_time_stamp();
    elapsed_cycles     = (uint64_t)((end - start) / CYCLE);
    PERF_TRACE_SPAN("Workers", "worker_" + std::to_string(tid), "run",
                    static_cast<uint64_t>(start / CYCLE),
                    static_cast<uint64_t>((end - start) / CYCLE));
    if (completion_fifo)
        completion_fifo->write(tid);
```

- [ ] **Step 5: Verify it compiles both ways**

Run (macro off — default kernel build path):
```bash
g++ -std=c++17 -DSC_INCLUDE_DYNAMIC_PROCESSES -I src -I config -c src/worker.cpp -o /tmp/worker_off.o && echo "OFF OK"
```
Run (macro on):
```bash
g++ -std=c++17 -DSC_INCLUDE_DYNAMIC_PROCESSES -DPERFETTO_TRACE -I src -I config -c src/worker.cpp -o /tmp/worker_on.o && echo "ON OK"
```
Expected: `OFF OK` then `ON OK` (both compile).

- [ ] **Step 6: Commit**

```bash
git add src/worker.cpp
git commit -m "feat: emit Perfetto run/scalar/stall spans from Worker

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Instrument AcceleratorTLM spans

**Files:**
- Modify: `src/accelerator.cpp` (include; service span in `service_thread`; load/compute/write spans in `write_thread`)

- [ ] **Step 1: Add the include**

At the top of `src/accelerator.cpp`, after `#include "memory.h"`, add:

```cpp
#include "perfetto_trace.h"
```

- [ ] **Step 2: Service span (serial mode)**

In `AcceleratorTLM::service_thread`, the busy-end section currently reads:

```cpp
        occupied_cycles += (uint64_t)((sc_time_stamp() - t_start) / CYCLE);

        // Signal busy end (compute finished, about to send response)
        if (busy_cb)
            busy_cb((uint64_t)(sc_time_stamp() / CYCLE), false);

        complete_request(e);
```

Replace with:

```cpp
        occupied_cycles += (uint64_t)((sc_time_stamp() - t_start) / CYCLE);

        // Signal busy end (compute finished, about to send response)
        if (busy_cb)
            busy_cb((uint64_t)(sc_time_stamp() / CYCLE), false);

        PERF_TRACE_SPAN("Accelerators", name(), "service",
                        static_cast<uint64_t>(t_start / CYCLE),
                        static_cast<uint64_t>((sc_time_stamp() - t_start) / CYCLE));

        complete_request(e);
```

- [ ] **Step 3: load/compute/write spans (pipeline mode)**

In `AcceleratorTLM::write_thread`, the tail currently reads:

```cpp
        if (busy_cb)
            busy_cb(static_cast<uint64_t>(sc_time_stamp() / CYCLE), false);
        stage_exit();

        complete_request(e);
```

Replace with (all six timestamps are populated on `e` by this point):

```cpp
        if (busy_cb)
            busy_cb(static_cast<uint64_t>(sc_time_stamp() / CYCLE), false);
        stage_exit();

        PERF_TRACE_SPAN("Accelerators", name(), "load",
                        static_cast<uint64_t>(e.t_load_start / CYCLE),
                        static_cast<uint64_t>((e.t_load_end - e.t_load_start) / CYCLE));
        PERF_TRACE_SPAN("Accelerators", name(), "compute",
                        static_cast<uint64_t>(e.t_compute_start / CYCLE),
                        static_cast<uint64_t>((e.t_compute_end - e.t_compute_start) / CYCLE));
        PERF_TRACE_SPAN("Accelerators", name(), "write",
                        static_cast<uint64_t>(e.t_write_start / CYCLE),
                        static_cast<uint64_t>((e.t_write_end - e.t_write_start) / CYCLE));

        complete_request(e);
```

- [ ] **Step 4: Verify it compiles both ways**

Run:
```bash
g++ -std=c++17 -DSC_INCLUDE_DYNAMIC_PROCESSES -I src -I config -c src/accelerator.cpp -o /tmp/accel_off.o && echo "OFF OK"
g++ -std=c++17 -DSC_INCLUDE_DYNAMIC_PROCESSES -DPERFETTO_TRACE -I src -I config -c src/accelerator.cpp -o /tmp/accel_on.o && echo "ON OK"
```
Expected: `OFF OK` then `ON OK`.

- [ ] **Step 5: Commit**

```bash
git add src/accelerator.cpp
git commit -m "feat: emit Perfetto service/load/compute/write spans from AcceleratorTLM

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: Instrument Memory spans

**Files:**
- Modify: `src/memory.cpp` (include; span in `Memory::dispatch_thread`, `L1L2Memory::l1_dispatch_thread`, `L1L2Memory::dma_dispatch_thread`)

- [ ] **Step 1: Add the include**

At the top of `src/memory.cpp`, after `#include "memory.h"`, add:

```cpp
#include "perfetto_trace.h"
```

- [ ] **Step 2: `Memory::dispatch_thread` span**

In `Memory::dispatch_thread`, the section that computes latency currently reads:

```cpp
            uint64_t mem_lat = base_lat_cycles + xfer;

            reqs += 1;
            busy_cycles += mem_lat;
            active_reqs += 1;
            resp_peq.notify(*e.gp, mem_lat * CYCLE);
```

Replace with:

```cpp
            uint64_t mem_lat = base_lat_cycles + xfer;

            reqs += 1;
            busy_cycles += mem_lat;
            active_reqs += 1;
            PERF_TRACE_SPAN("Memory", name(),
                            e.gp->is_write() ? "write" : "read",
                            static_cast<uint64_t>(t_start / CYCLE), mem_lat);
            resp_peq.notify(*e.gp, mem_lat * CYCLE);
```

- [ ] **Step 3: `L1L2Memory::l1_dispatch_thread` span**

In `L1L2Memory::l1_dispatch_thread`, the tail of the inner loop currently reads:

```cpp
            l1_active_reqs += 1;
            resp_peq.notify(*e.gp, lat * CYCLE);
```

Replace with:

```cpp
            l1_active_reqs += 1;
            PERF_TRACE_SPAN("Memory", std::string(name()) + ":L1",
                            e.gp->is_write() ? "write" : "read",
                            static_cast<uint64_t>(sc_time_stamp() / CYCLE), lat);
            resp_peq.notify(*e.gp, lat * CYCLE);
```

- [ ] **Step 4: `L1L2Memory::dma_dispatch_thread` span**

In `L1L2Memory::dma_dispatch_thread`, the tail of the inner loop currently reads:

```cpp
            dma_active_reqs += 1;
            resp_peq.notify(*e.gp, lat * CYCLE);
```

Replace with:

```cpp
            dma_active_reqs += 1;
            PERF_TRACE_SPAN("Memory", std::string(name()) + ":DMA",
                            e.gp->is_write() ? "write" : "read",
                            static_cast<uint64_t>(sc_time_stamp() / CYCLE), lat);
            resp_peq.notify(*e.gp, lat * CYCLE);
```

- [ ] **Step 5: Verify it compiles both ways**

Run:
```bash
g++ -std=c++17 -DSC_INCLUDE_DYNAMIC_PROCESSES -I src -I config -c src/memory.cpp -o /tmp/mem_off.o && echo "OFF OK"
g++ -std=c++17 -DSC_INCLUDE_DYNAMIC_PROCESSES -DPERFETTO_TRACE -I src -I config -c src/memory.cpp -o /tmp/mem_on.o && echo "ON OK"
```
Expected: `OFF OK` then `ON OK`.

- [ ] **Step 6: Commit**

```bash
git add src/memory.cpp
git commit -m "feat: emit Perfetto dispatch spans from Memory and L1L2Memory

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: nafblock CLI flag + JSON emission

**Files:**
- Modify: `nafblock/nafblock_sim.cpp` (include; `BlockOptions::trace_out`; `--trace-out` parse; write call)

- [ ] **Step 1: Add the include**

In `nafblock/nafblock_sim.cpp`, after `#include "report_formatter.h"`, add:

```cpp
#include "perfetto_trace.h"
```

- [ ] **Step 2: Add `trace_out` to `BlockOptions`**

In the `BlockOptions` struct, after the `dma_base_lat` field, add:

```cpp
    // Perfetto trace output path (only consumed when built with PERFETTO_TRACE).
    std::string trace_out = "nafblock_trace.json";
```

- [ ] **Step 3: Parse `--trace-out`**

In `parse_args`, the first `if` currently matches the int-valued flags:

```cpp
        if (arg == "--block-c" || arg == "--block-h" || arg == "--block-w"
            || arg == "--dma-base-lat")
        {
```

Immediately BEFORE that `if`, add a new branch that consumes a string value:

```cpp
        if (arg == "--trace-out")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for " << arg << "\n";
                return false;
            }
            opts.trace_out = argv[++i];
            continue;
        }
```

- [ ] **Step 4: Mention the flag in `--help`**

In `parse_args`, the help `std::cout` block currently ends with:

```cpp
                << " [--dma-base-lat N]\n"
```

Change that line to:

```cpp
                << " [--dma-base-lat N] [--trace-out FILE]\n"
```

- [ ] **Step 5: Write the trace after `sc_start()`**

In `sc_main`, immediately after the `sc_start();` call and before the
`const uint64_t total_cycles = ...` line, add:

```cpp
#ifdef PERFETTO_TRACE
    perf_trace_write_json(opts.trace_out.c_str());
    std::cerr << "Perfetto trace written to " << opts.trace_out << "\n";
#endif
```

- [ ] **Step 6: Verify it compiles (macro off, via existing build)**

Run: `make nafblock`
Expected: builds `nafblock/build/nafblock_perf_sim` with no errors. (Macro is still off at this point — Task 6 turns it on. This confirms the CLI changes compile in the untraced build.)

- [ ] **Step 7: Confirm `--trace-out` is accepted but inert without the macro**

Run: `./nafblock/build/nafblock_perf_sim --trace-out /tmp/should_not_exist.json --block-c 16 --block-h 16 --block-w 16 ; ls /tmp/should_not_exist.json 2>&1 | head -1`
Expected: simulator runs to a PASS report (exit 0), and `ls` reports the file does NOT exist (no trace written because macro is off).

- [ ] **Step 8: Commit**

```bash
git add nafblock/nafblock_sim.cpp
git commit -m "feat: add --trace-out flag and macro-guarded trace emission to nafblock

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: Enable the macro in the nafblock Makefile

**Files:**
- Modify: `nafblock/Makefile` (define `-DPERFETTO_TRACE`; add header dep to touched objects)

- [ ] **Step 1: Define the macro**

In `nafblock/Makefile`, the `CXXFLAGS` block currently reads:

```make
CXXFLAGS := -std=c++17 -O2 -Wall -Wno-unused-parameter \
            -DSC_INCLUDE_DYNAMIC_PROCESSES \
            -MMD -MP
```

Change it to:

```make
CXXFLAGS := -std=c++17 -O2 -Wall -Wno-unused-parameter \
            -DSC_INCLUDE_DYNAMIC_PROCESSES \
            -DPERFETTO_TRACE \
            -MMD -MP
```

- [ ] **Step 2: Add header dependency to the touched object rules**

The Makefile uses `-MMD -MP`, so header dependencies are auto-tracked after the
first build. No manual edit to the per-object recipes is required for correctness.
Verify the four touched objects (`memory.o`, `accelerator.o`, `worker.o`,
`nafblock_sim.o`) exist as rules already — they do (see the `OBJS` list). No change
needed in this step beyond confirming.

- [ ] **Step 3: Clean rebuild with the macro on**

Run: `make -C nafblock clean 2>/dev/null; rm -rf nafblock/build; make nafblock`
Expected: clean build of `nafblock/build/nafblock_perf_sim` with `-DPERFETTO_TRACE` on every compile line (no errors).

- [ ] **Step 4: Confirm other sims are unaffected**

Run: `make kernel-matmul && ./kernel/build/matmul_sim --threads 2 >/dev/null; echo "exit=$?"`
Expected: builds and runs; `exit=0`. (matmul build does not define the macro, so its `src/*.o` are untraced — confirms isolation.)

- [ ] **Step 5: Commit**

```bash
git add nafblock/Makefile
git commit -m "build: enable PERFETTO_TRACE for the nafblock simulator

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 7: End-to-end verification + docs

**Files:**
- Modify: `nafblock/CLAUDE.md` (Status/Gotchas: trace build + usage)

- [ ] **Step 1: Baseline report (no trace) for timing-equivalence check**

Run:
```bash
git stash list >/dev/null 2>&1
./nafblock/build/nafblock_perf_sim --block-c 16 --block-h 16 --block-w 16 | grep "Total Elapsed Cycles" | tee /tmp/nb_traced_cycles.txt
```
Expected: prints `Total Elapsed Cycles [cycles] : <N>` (record N). Because tracing is observation-only, this must equal the pre-tracing baseline. To confirm against an untraced binary, build one explicitly:
```bash
make -C nafblock clean; rm -rf nafblock/build
make -C nafblock CXXFLAGS="-std=c++17 -O2 -Wall -Wno-unused-parameter -DSC_INCLUDE_DYNAMIC_PROCESSES -MMD -MP"
./nafblock/build/nafblock_perf_sim --block-c 16 --block-h 16 --block-w 16 | grep "Total Elapsed Cycles"
```
Expected: same `<N>` as the traced build. Then rebuild traced: `rm -rf nafblock/build; make nafblock`.

- [ ] **Step 2: Run traced and validate JSON**

Run:
```bash
./nafblock/build/nafblock_perf_sim --block-c 16 --block-h 16 --block-w 16 --trace-out /tmp/nb.json
python3 -m json.tool /tmp/nb.json > /dev/null && echo "JSON VALID"
```
Expected: simulator exits 0 with `Perfetto trace written to /tmp/nb.json` on stderr; `JSON VALID`.

- [ ] **Step 3: Sanity-check span content**

Run:
```bash
python3 - <<'EOF'
import json
d = json.load(open("/tmp/nb.json"))
ev = d["traceEvents"]
spans = [e for e in ev if e.get("ph") == "X"]
groups = {e["args"]["name"] for e in ev if e.get("name") == "process_name"}
names  = {s["name"] for s in spans}
print("groups:", sorted(groups))
print("span kinds:", sorted(names))
print("span count:", len(spans))
assert groups == {"Workers", "Accelerators", "Memory"}, groups
assert "run" in names and "scalar" in names
assert any(k in names for k in ("service", "compute"))
assert any(k in names for k in ("read", "write"))
assert len(spans) > 0
print("SANITY OK")
EOF
```
Expected: prints the groups/kinds/counts then `SANITY OK`.

- [ ] **Step 4: Manual Perfetto check (informational)**

Open https://ui.perfetto.dev/ and drag `/tmp/nb.json` in. Confirm three process
lanes (Workers / Accelerators / Memory), worker lanes span the full run, and
accelerator lanes appear per layer. (This step is a human visual check; record the
outcome but it does not block automated verification.)

- [ ] **Step 5: Document in `nafblock/CLAUDE.md`**

Under the `### Build / run` section in `nafblock/CLAUDE.md`, after the existing
fenced command block, add:

```markdown
#### Perfetto timeline trace
The nafblock build compiles with `-DPERFETTO_TRACE` (set in [Makefile](Makefile)),
so every run writes a Chrome-Trace-Event JSON timeline:
```
./nafblock/build/nafblock_perf_sim --trace-out nafblock_trace.json
```
Drag the JSON into https://ui.perfetto.dev/ . Lanes:
- **Workers** — one continuous lane per worker tid (across all 14 sub-layers),
  with `run` / `scalar` / `stall` spans.
- **Accelerators** — one lane per accelerator instance per layer (unique SystemC
  name), with `service` (serial) or `load`/`compute`/`write` (pipeline) spans.
- **Memory** — one lane per memory port (`:L1` / `:DMA`), with `read` / `write` spans.

Tracing is observation-only (reads `sc_time_stamp()`, never waits); total cycles
are identical to an untraced build. Only the nafblock Makefile defines
`PERFETTO_TRACE`; kernel sims and `nafnet/` are unaffected. See
[src/perfetto_trace.h](../src/perfetto_trace.h).
```

Also add to the `### Gotchas` list:

```markdown
- **Perfetto tracing is nafblock-only**: `-DPERFETTO_TRACE` lives in this
  subproject's Makefile and gates all span code in `src/{worker,accelerator,memory}.cpp`
  via `PERF_TRACE_SPAN`. Don't define it for other sims — their `src/*.o` must stay
  untraced. The macro-off expansion is a no-op, so the same sources compile both ways.
```

- [ ] **Step 6: Commit**

```bash
git add nafblock/CLAUDE.md
git commit -m "docs: document nafblock Perfetto tracing build and usage

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage:** output format (Task 1), gating macro (Tasks 1/6), `src/perfetto_trace.h` (Task 1), worker run+scalar+stall (Task 2), accelerator serial+pipeline (Task 3), memory L1/DMA (Task 4), emission + `--trace-out` (Task 5), Makefile (Task 6), verification + docs (Task 7). All spec sections mapped.
- **Type consistency:** `perf_trace_record` / `PERF_TRACE_SPAN` / `perf_spans()` / `perf_trace_write_json` / `perf_ns_to_us` signatures defined in Task 1 are used verbatim in Tasks 2-5. Group labels are exactly `"Workers"`, `"Accelerators"`, `"Memory"` everywhere (asserted in Task 7).
- **Observation-only invariant:** every span site reads timestamps already present in the surrounding code; no `wait()` added. Verified by the timing-equivalence check in Task 7 Step 1.
- **`name()` availability:** `AcceleratorTLM` and `Memory`/`L1L2Memory` are `sc_module`s, so `name()` returns the hierarchical instance name — valid in the thread bodies where spans are emitted.
