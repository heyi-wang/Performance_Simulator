#pragma once

#include <systemc>
#include <tlm>
#include <cstdint>

using namespace sc_core;
using namespace tlm;

// Shared by all modules: 1 cycle = 1 ns
inline const sc_time CYCLE{1, SC_NS};

static inline uint64_t ceil_div_u64(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}
