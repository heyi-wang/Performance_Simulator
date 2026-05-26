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
    std::map<std::string, int>    track_tid;   // key: "<pid>\x1f<track>"
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
