#pragma once

// ============================================================
// perfetto_trace.h — header-only global span sink that emits a
// Perfetto protobuf TrackEvent trace, openable in ui.perfetto.dev.
//
// All recording goes through PERF_TRACE_SPAN(...), which expands
// to nothing unless PERFETTO_TRACE is defined at compile time.
// The writer (perf_trace_write_json — name retained for caller
// stability) is always compiled so a test harness can exercise
// it directly.
//
// Output format: Perfetto native protobuf (Trace message with
// TrackDescriptor + TrackEvent packets). The file is binary; the
// extension is whatever the caller picks — ui.perfetto.dev sniffs
// the format. Compared to Chrome JSON, this lets us label tracks
// with free-form names (no pid/tid suffix appended by the UI) and
// pick the group order explicitly.
//
// Group sort order in the UI: Scalar Unit < DMA Engine <
// Matrix Unit < Vector Unit < anything else.
//
// No SystemC dependency: callers convert sc_time to ns themselves.
// ============================================================

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

struct PerfSpan
{
    std::string group;   // group track label (e.g. "Vector Unit 0")
    std::string track;   // lane label within the group (e.g. "compute")
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

// Tracks declared up-front so an (always-)empty lane still appears in the UI.
// Order of first appearance fixes the lane order within a group.
inline std::vector<std::pair<std::string, std::string>> &perf_declared_tracks()
{
    static std::vector<std::pair<std::string, std::string>> v;
    return v;
}

inline void perf_trace_record(std::string group,
                              std::string track,
                              std::string name,
                              uint64_t    ts_ns,
                              uint64_t    dur_ns)
{
    perf_spans().push_back(
        PerfSpan{std::move(group), std::move(track), std::move(name),
                 ts_ns, dur_ns});
}

inline void perf_trace_declare(std::string group, std::string track)
{
    perf_declared_tracks().emplace_back(std::move(group), std::move(track));
}

// Runtime gate. Recording is skipped (and span-argument expressions are not
// evaluated) unless this is set true — typically when a sim is given
// --trace-out. This keeps the cost of a traced build to one branch per span
// site when tracing is not requested (so sweeps pay ~nothing).
inline bool &perf_trace_enabled()
{
    static bool enabled = false;
    return enabled;
}

#ifdef PERFETTO_TRACE
#  define PERF_TRACE_SPAN(group, track, name, ts_ns, dur_ns)                  \
        do {                                                                  \
            if (perf_trace_enabled())                                         \
                perf_trace_record((group), (track), (name), (ts_ns), (dur_ns)); \
        } while (0)
#  define PERF_TRACE_DECLARE(group, track)                                    \
        do {                                                                  \
            if (perf_trace_enabled())                                         \
                perf_trace_declare((group), (track));                         \
        } while (0)
#else
#  define PERF_TRACE_SPAN(group, track, name, ts_ns, dur_ns) ((void)0)
#  define PERF_TRACE_DECLARE(group, track) ((void)0)
#endif

// Extract (and remove) a "--trace-out <path>" pair from argv so each sim's own
// argument parser never sees it. Returns the path, or "" if not present.
// Always compiled (independent of PERFETTO_TRACE) so the flag is accepted and
// stripped uniformly; callers decide whether to act on it.
inline std::string perf_take_trace_out_arg(int &argc, char **argv)
{
    std::string out;
    int w = 1;
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--trace-out" && i + 1 < argc)
            out = argv[++i];
        else
            argv[w++] = argv[i];
    }
    argc = w;
    return out;
}

// ----------------------------------------------------------
// Protobuf wire-format helpers (just enough for what we emit).
//
// Encoding reference: tag = (field << 3) | wire_type, varint-encoded.
//   wire 0 = varint, wire 2 = length-delimited (string / submessage).
// ----------------------------------------------------------
namespace perf_pb {
inline void varint(std::string &out, uint64_t v)
{
    while (v >= 0x80) { out.push_back(static_cast<char>((v & 0x7F) | 0x80)); v >>= 7; }
    out.push_back(static_cast<char>(v));
}
inline void tag(std::string &out, uint32_t field, uint32_t wire)
{
    varint(out, (static_cast<uint64_t>(field) << 3) | wire);
}
inline void u64(std::string &out, uint32_t field, uint64_t v)
{
    tag(out, field, 0); varint(out, v);
}
inline void str(std::string &out, uint32_t field, const std::string &s)
{
    tag(out, field, 2); varint(out, s.size()); out.append(s);
}
inline void submsg(std::string &out, uint32_t field, const std::string &m)
{
    tag(out, field, 2); varint(out, m.size()); out.append(m);
}
}  // namespace perf_pb

// Write the recorded spans as a Perfetto protobuf Trace.
//
// Function name is "_json" for historical reasons (callers haven't changed);
// the on-disk format is Perfetto protobuf TrackEvent regardless of the
// caller-chosen extension.
inline void perf_trace_write_json(const char *path)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) return;

    using namespace perf_pb;

    // 1. Group sort order in the UI:
    //    Scalar Unit < DMA Engine < Matrix Unit < Vector Unit < anything else.
    //    Within a rank, natural-sort by trailing index (Scalar Unit 2 < 10).
    auto rank_of = [](const std::string &g) {
        if (g.rfind("Scalar Unit", 0) == 0) return 0;
        if (g == "DMA Engine")              return 1;
        if (g.rfind("Matrix Unit", 0) == 0) return 2;
        if (g.rfind("Vector Unit", 0) == 0) return 3;
        return 4;
    };
    auto natural_key = [](const std::string &s) {
        size_t i = s.size();
        while (i > 0 && std::isdigit(static_cast<unsigned char>(s[i-1]))) --i;
        uint64_t n = (i < s.size()) ? std::stoull(s.substr(i)) : 0;
        return std::pair<std::string, uint64_t>{s.substr(0, i), n};
    };

    // 2. Collect groups (sorted) and lanes per group (declared first, then
    //    first-seen from spans; preserves the user's declared lane order).
    std::vector<std::string> groups;
    std::set<std::string> group_set;
    std::map<std::string, std::vector<std::string>> lanes_of;
    std::set<std::string> lane_set;
    auto add_lane = [&](const std::string &g, const std::string &t) {
        std::string key = g + "\x1f" + t;
        if (!lane_set.insert(key).second) return;
        if (group_set.insert(g).second) groups.push_back(g);
        lanes_of[g].push_back(t);
    };
    for (const auto &d : perf_declared_tracks()) add_lane(d.first, d.second);
    for (const auto &s : perf_spans())           add_lane(s.group, s.track);

    std::sort(groups.begin(), groups.end(),
              [&](const std::string &a, const std::string &b) {
                  int ra = rank_of(a), rb = rank_of(b);
                  if (ra != rb) return ra < rb;
                  return natural_key(a) < natural_key(b);
              });

    // 3. Assign track UUIDs and pid/tid pairs.
    //    Each group is a ProcessDescriptor (pid=N); each lane is a
    //    ThreadDescriptor (pid=N, tid=M) of that process. This is Perfetto's
    //    native process/thread rendering — one clean row per thread, no
    //    depth-aggregator row added on top for high-density tracks (which is
    //    what the plain "name + parent_uuid" pattern triggers).
    std::map<std::string, uint64_t> guuid;
    std::map<std::string, uint64_t> tuuid;       // key "group\x1ftrack"
    std::map<std::string, int32_t>  gpid;        // group -> pid
    std::map<std::string, int32_t>  ttid;        // key "group\x1ftrack" -> tid
    uint64_t next_uuid = 1;
    int32_t  next_pid  = 1;
    int32_t  next_tid  = 1;
    for (const auto &g : groups) {
        guuid[g] = next_uuid++;
        gpid[g]  = next_pid++;
    }
    for (const auto &g : groups)
        for (const auto &t : lanes_of[g]) {
            const std::string key = g + "\x1f" + t;
            tuuid[key] = next_uuid++;
            ttid[key]  = next_tid++;
        }

    const uint32_t SEQ_ID = 1;
    bool first_packet = true;

    auto emit_packet = [&](const std::string &pkt_body) {
        // Trace.packet (field 1, length-delimited).
        std::string outer;
        submsg(outer, 1, pkt_body);
        f.write(outer.data(), static_cast<std::streamsize>(outer.size()));
    };
    auto wrap_descriptor = [&](const std::string &td_body) {
        std::string pkt;
        submsg(pkt, 60, td_body);   // TracePacket.track_descriptor
        u64(pkt, 10, SEQ_ID);        // trusted_packet_sequence_id
        if (first_packet) {
            // sequence_flags = SEQ_INCREMENTAL_STATE_CLEARED (1) on first pkt.
            u64(pkt, 87, 1);
            first_packet = false;
        }
        emit_packet(pkt);
    };

    // 4. Group TrackDescriptors as ProcessDescriptors, in sorted order.
    //    ProcessDescriptor fields: pid (1), process_name (6).
    for (const auto &g : groups) {
        std::string proc;
        u64(proc, 1, static_cast<uint64_t>(gpid[g]));   // pid
        str(proc, 6, g);                                  // process_name
        std::string td;
        u64(td, 1, guuid[g]);                             // uuid
        submsg(td, 3, proc);                              // process
        wrap_descriptor(td);
    }
    // 5. Lane TrackDescriptors as ThreadDescriptors of their group's process.
    //    ThreadDescriptor fields: pid (1), tid (2), thread_name (5).
    for (const auto &g : groups) {
        for (const auto &t : lanes_of[g]) {
            const std::string key = g + "\x1f" + t;
            std::string thr;
            u64(thr, 1, static_cast<uint64_t>(gpid[g]));  // pid (links to process)
            u64(thr, 2, static_cast<uint64_t>(ttid[key])); // tid
            str(thr, 5, t);                                // thread_name
            std::string td;
            u64(td, 1, tuuid[key]);                        // uuid
            submsg(td, 4, thr);                            // thread
            wrap_descriptor(td);
        }
    }

    // 6. Emit each span as a SLICE_BEGIN + SLICE_END pair, in chronological
    //    order so the per-track slice stack stays correctly nested.
    struct Evt { uint64_t ts; bool end; uint64_t track; size_t span_index; };
    std::vector<Evt> events;
    const auto &spans_v = perf_spans();
    events.reserve(spans_v.size() * 2);
    for (size_t i = 0; i < spans_v.size(); ++i) {
        const auto &s = spans_v[i];
        auto it = tuuid.find(s.group + "\x1f" + s.track);
        if (it == tuuid.end()) continue;
        events.push_back({s.ts_ns, false, it->second, i});
        events.push_back({s.ts_ns + s.dur_ns, true, it->second, i});
    }
    std::sort(events.begin(), events.end(), [](const Evt &a, const Evt &b) {
        if (a.ts != b.ts) return a.ts < b.ts;
        return a.end && !b.end;  // END before BEGIN at the same timestamp
    });

    for (const auto &e : events) {
        std::string te;
        u64(te, 11, e.track);              // TrackEvent.track_uuid
        u64(te, 9, e.end ? 2 : 1);         // .type = SLICE_END (2) / SLICE_BEGIN (1)
        if (!e.end) str(te, 23, spans_v[e.span_index].name);  // .name

        std::string pkt;
        u64(pkt, 8, e.ts);                  // TracePacket.timestamp
        submsg(pkt, 11, te);                // .track_event
        u64(pkt, 10, SEQ_ID);               // .trusted_packet_sequence_id
        emit_packet(pkt);
    }
}
