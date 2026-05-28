// Standalone unit test for perfetto_trace.h Perfetto protobuf writer.
// Build: g++ -std=c++17 -DPERFETTO_TRACE -I src src/perfetto_trace_test.cpp -o /tmp/pt_test
#include "perfetto_trace.h"

#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

static std::string slurp(const char *path)
{
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Minimal protobuf varint reader for verifying packet structure.
static uint64_t rd_varint(const std::string &b, size_t &i)
{
    uint64_t v = 0;
    int sh = 0;
    while (i < b.size())
    {
        unsigned char c = static_cast<unsigned char>(b[i++]);
        v |= uint64_t(c & 0x7F) << sh;
        if (!(c & 0x80))
            break;
        sh += 7;
    }
    return v;
}

int main()
{
    perf_spans().clear();
    perf_trace_enabled() = true;

    PERF_TRACE_DECLARE("Workers", "worker_0");
    PERF_TRACE_SPAN("Workers", "worker_0", "run", 0, 1500);
    PERF_TRACE_SPAN("Accelerators", "nb_mat.accel_0", "service", 1000, 2000);

    assert(perf_spans().size() == 2);

    const char *out = "/tmp/pt_test_out.pftrace";
    perf_trace_write_json(out);
    const std::string buf = slurp(out);

    // First byte: tag for Trace.packet (field 1, wire-type 2) = (1<<3)|2 = 0x0a.
    assert(!buf.empty());
    assert(static_cast<unsigned char>(buf[0]) == 0x0a);

    // Names should appear verbatim in the binary somewhere (no pid/tid suffix).
    assert(buf.find("Workers") != std::string::npos);
    assert(buf.find("worker_0") != std::string::npos);
    assert(buf.find("Accelerators") != std::string::npos);
    assert(buf.find("nb_mat.accel_0") != std::string::npos);
    assert(buf.find("run") != std::string::npos);
    assert(buf.find("service") != std::string::npos);

    // Walk the top-level Trace.packet records: each is 0x0a + varint(len) + bytes.
    // Count them to make sure the stream is well-formed.
    size_t i = 0;
    size_t packets = 0;
    while (i < buf.size())
    {
        unsigned char tag = static_cast<unsigned char>(buf[i++]);
        assert(tag == 0x0a);
        uint64_t len = rd_varint(buf, i);
        assert(i + len <= buf.size());
        i += len;
        ++packets;
    }
    // Expect at least: 2 group descriptors + 2 lane descriptors + 4 begin/end events.
    assert(packets >= 8);

    std::cout << "perfetto_trace_test PASSED (" << packets << " packets, "
              << buf.size() << " bytes)\n";
    return 0;
}
