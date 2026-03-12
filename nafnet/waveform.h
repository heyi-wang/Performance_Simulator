#pragma once

#include <cstdint>
#include <string>
#include <vector>

// ============================================================
// WaveformLogger — lightweight global event log for waveform
// visualization.
//
// Each hardware unit calls wave_log() at every busy/idle
// transition.  sc_main dumps the collected events to CSV
// after simulation ends; plot_waveform.py renders the plot.
// ============================================================

struct WaveEvent
{
    uint64_t    cycle;
    std::string name;
    int         val;   // 1 = busy, 0 = idle
};

// Singleton event buffer (inline so the header is self-contained)
inline std::vector<WaveEvent> &wave_events()
{
    static std::vector<WaveEvent> v;
    return v;
}

inline void wave_log(uint64_t cycle, const std::string &name, int val)
{
    wave_events().push_back({cycle, name, val});
}
