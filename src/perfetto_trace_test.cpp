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
    perf_trace_enabled() = true;   // recording is gated on this at runtime

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
