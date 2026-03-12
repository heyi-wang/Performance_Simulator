#pragma once

#include <cstdint>
#include <ctime>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "waveform.h"

// Assign a VCD identifier string to each index.
// VCD identifiers are printable ASCII chars except space (0x20) and '$' (0x24).
// We start at '!' (0x21) and skip '$'.
static inline std::string vcd_id(int idx)
{
    // Printable ASCII: 0x21–0x7E, excluding 0x24 ('$')
    // Build a base-93 encoding (94 printable chars minus '$' = 93 usable chars)
    static const char *chars =
        "!\"#%&'()*+,-./:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";
    // 93 characters
    constexpr int BASE = 93;
    std::string id;
    do {
        id += chars[idx % BASE];
        idx  = idx / BASE - 1;
    } while (idx >= 0);
    return id;
}

inline void write_vcd(const std::vector<WaveEvent> &evts, const char *path)
{
    std::ofstream f(path);
    if (!f) return;

    // ------------------------------------------------------------------
    // 1. Collect unique signal names in first-appearance order
    // ------------------------------------------------------------------
    std::vector<std::string> signal_order;
    std::map<std::string, std::string> sig_id; // signal name → VCD identifier

    for (const auto &e : evts) {
        if (sig_id.find(e.name) == sig_id.end()) {
            std::string id = vcd_id((int)signal_order.size());
            sig_id[e.name] = id;
            signal_order.push_back(e.name);
        }
    }

    // ------------------------------------------------------------------
    // 2. Header
    // ------------------------------------------------------------------
    {
        time_t now = time(nullptr);
        char buf[64];
        strftime(buf, sizeof(buf), "%b %d %Y", localtime(&now));
        f << "$date " << buf << " $end\n";
    }
    f << "$version NAFNet Performance Simulator $end\n";
    f << "$timescale 1 ns $end\n";

    // ------------------------------------------------------------------
    // 3. Scope + variable declarations
    // ------------------------------------------------------------------
    f << "$scope module nafnet_sim $end\n";
    for (const auto &name : signal_order)
        f << "$var wire 1 " << sig_id[name] << " " << name << " $end\n";
    f << "$upscope $end\n";
    f << "$enddefinitions $end\n";

    // ------------------------------------------------------------------
    // 4. Initial values at time 0
    // ------------------------------------------------------------------
    f << "#0\n";
    f << "$dumpvars\n";
    for (const auto &name : signal_order)
        f << "0" << sig_id[name] << "\n";
    f << "$end\n";

    // ------------------------------------------------------------------
    // 5. Value changes — iterate sorted events, group by cycle
    // ------------------------------------------------------------------
    uint64_t cur_cycle = UINT64_MAX;
    for (const auto &e : evts) {
        if (e.cycle != cur_cycle) {
            f << "#" << e.cycle << "\n";
            cur_cycle = e.cycle;
        }
        f << e.val << sig_id[e.name] << "\n";
    }
}
