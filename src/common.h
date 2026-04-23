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

static inline uint64_t vec_request_cycles(uint64_t base_cycle,
                                          uint64_t instruction_count)
{
    return base_cycle * instruction_count;
}

struct KernelWorkerInfo
{
    int      tid = -1;
    uint64_t mat_reqs = 0;
    uint64_t vec_reqs = 0;
    uint64_t elapsed_cycles = 0;
    uint64_t scalar_cycles = 0;
    uint64_t stall_cycles = 0;
    uint64_t mem_cycles = 0;
    uint64_t rd_bytes = 0;
    uint64_t wr_bytes = 0;
};

struct AccelInstanceStats
{
    int      instance_id = -1;
    uint64_t req_count = 0;
    uint64_t busy_cycles = 0;
    uint64_t occupied_cycles = 0;
    uint64_t queue_wait_cycles = 0;
};
