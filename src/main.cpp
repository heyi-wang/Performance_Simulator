#include "top.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

int sc_main(int /*argc*/, char * /*argv*/[])
{
    Top top("top");
    sc_start();

    // -------------------------------------------------------
    // Accelerator and memory utilization
    // -------------------------------------------------------
    std::cout << "\n=== Accelerator stats ===\n";
    std::cout << "mat_acc pool (" << top.mat_acc.instance_count() << " units):"
              << " reqs="                << top.mat_acc.req_count_total()
              << " busy_cycles="         << top.mat_acc.busy_cycles_total()
              << " occupied_cycles="     << top.mat_acc.occupied_cycles_total()
              << " queue_wait_cycles="   << top.mat_acc.queue_wait_cycles_total()
              << "\n";

    std::cout << "vec_acc pool (" << top.vec_acc.instance_count() << " units):"
              << " reqs="                << top.vec_acc.req_count_total()
              << " busy_cycles="         << top.vec_acc.busy_cycles_total()
              << " occupied_cycles="     << top.vec_acc.occupied_cycles_total()
              << " queue_wait_cycles="   << top.vec_acc.queue_wait_cycles_total()
              << "\n";

    std::cout << "memory : reqs="          << top.memory.reqs
              << " busy_cycles="           << top.memory.busy_cycles
              << " queue_wait_cycles="     << top.memory.qwait_cycles
              << "\n";

    // -------------------------------------------------------
    // Throughput and bottleneck analysis
    // -------------------------------------------------------

    // Wall-clock cycles = max elapsed across all worker threads
    uint64_t max_elapsed = 0;
    for (const auto *w : top.workers)
        max_elapsed = std::max(max_elapsed, w->elapsed_cycles);

    // GFLOPS: each MAC = 1 multiply + 1 add = 2 FLOPs
    //   CYCLE = 1 ns  →  frequency = 1 GHz
    //   GFLOPS = (total_MACs × 2) / (elapsed_cycles × 1e-9) / 1e9
    //          = (total_MACs × 2) / elapsed_cycles   (units cancel)
    double total_flops = static_cast<double>(CONV_TOTAL_MACS) * 2.0;
    double gflops      = total_flops / static_cast<double>(max_elapsed);

    // mat_acc utilization = busy / (busy + queue_wait)
    double mat_busy    = static_cast<double>(top.mat_acc.busy_cycles_total());
    double mat_total   = mat_busy + static_cast<double>(top.mat_acc.queue_wait_cycles_total());
    double mat_util    = (mat_total > 0) ? mat_busy / mat_total * 100.0 : 0.0;

    // Memory bandwidth used:
    //   memory.busy_cycles × bytes_per_cycle (default 32) / max_elapsed = bytes/cycle
    double mem_bw_used = static_cast<double>(top.memory.busy_cycles) * 32.0
                         / static_cast<double>(max_elapsed);

    std::cout << "\n=== Performance summary ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Conv layer     : [" << CONV_N << ", " << CONV_C_IN
              << ", " << CONV_H_IN << ", " << CONV_W_IN << "]"
              << " × filter [" << CONV_C_OUT << ", " << CONV_C_IN
              << ", " << CONV_KH << ", " << CONV_KW << "]"
              << "  stride=" << CONV_STRIDE << " pad=" << CONV_PAD
              << "\n";
    std::cout << "Output shape   : [" << CONV_N << ", " << CONV_C_OUT
              << ", " << CONV_H_OUT << ", " << CONV_W_OUT << "]\n";
    std::cout << "Total MACs     : " << CONV_TOTAL_MACS << "\n";
    std::cout << "Threads        : " << NUM_THREADS << "\n";
    std::cout << "Mat accels     : " << MAT_ACCEL_COUNT_CFG << "\n";
    std::cout << "Vec accels     : " << VEC_ACCEL_COUNT_CFG << "\n";
    std::cout << "Max elapsed    : " << max_elapsed << " cycles\n";
    std::cout << "Throughput     : " << gflops << " GFLOPS  (at 1 GHz)\n";
    std::cout << "mat_acc util   : " << mat_util << "%"
              << "  (busy=" << top.mat_acc.busy_cycles_total()
              << "  stall=" << top.mat_acc.queue_wait_cycles_total() << ")\n";
    std::cout << "Mem BW avg use : " << mem_bw_used << " bytes/cycle\n";

    if (mat_util < 50.0)
        std::cout << "Note: mat_acc utilization is low — "
                     "consider reducing SCALAR_OVERHEAD or increasing NUM_THREADS.\n";
    else if (mat_util > 90.0)
        std::cout << "Note: mat_acc is heavily contended — "
                     "this accelerator is the bottleneck.\n";

    return 0;
}
