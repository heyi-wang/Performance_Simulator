#include "matmul_top.h"
#include "matmul_config.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

// ============================================================
// Verification helpers
// ============================================================

static bool check(bool cond, const char *desc)
{
    if (cond)
        std::cout << "  [PASS] " << desc << "\n";
    else
        std::cout << "  [FAIL] " << desc << "\n";
    return cond;
}

// ============================================================
// sc_main
// ============================================================
int sc_main(int /*argc*/, char * /*argv*/[])
{
    std::cout << "=== K-split GEMM Performance Simulator ===\n";
    std::cout << "Threads      : " << NUM_THREADS          << "\n";
    std::cout << "GEMM shape   : [" << GEMM_M << " x " << GEMM_K << " x " << GEMM_N << "]\n";
    std::cout << "K per thread : " << GEMM_K_PER_THREAD    << "\n";
    std::cout << "Mat tiles/w  : " << GEMM_ACCESS_MAT      << "\n";
    std::cout << "Accum calls  : " << GEMM_ACCUM_VEC_CALLS << " vec_acc calls per pair\n";
    std::cout << "\n";

    MatmulTop top("top");
    sc_start();

    // -------------------------------------------------------
    // Raw stats
    // -------------------------------------------------------
    std::cout << "\n=== Accelerator stats ===\n";
    std::cout << "mat_acc : reqs="      << top.mat_acc.req_count
              << " busy_cycles="        << top.mat_acc.busy_cycles
              << " queue_wait_cycles="  << top.mat_acc.queue_wait_cycles
              << "\n";
    std::cout << "vec_acc : reqs="      << top.vec_acc.req_count
              << " busy_cycles="        << top.vec_acc.busy_cycles
              << " queue_wait_cycles="  << top.vec_acc.queue_wait_cycles
              << "\n";
    std::cout << "memory  : reqs="      << top.memory.reqs
              << " busy_cycles="        << top.memory.busy_cycles
              << " queue_wait_cycles="  << top.memory.qwait_cycles
              << "\n";

    // -------------------------------------------------------
    // Wall-clock timing
    // -------------------------------------------------------
    // max worker mat-phase elapsed (per-thread independent)
    uint64_t max_mat_elapsed = 0;
    for (const auto *w : top.workers)
        max_mat_elapsed = std::max(max_mat_elapsed, w->mat_elapsed_cycles);

    // Total simulation time = when the final accumulated result is ready.
    // With early-start accumulation this is <= max_mat_elapsed + serial_accum_time.
    uint64_t total_elapsed =
        (uint64_t)(top.coordinator->accum_end_time / CYCLE);

    // Accumulation overhead beyond the mat-phase critical path.
    uint64_t accum_overhead =
        (total_elapsed > max_mat_elapsed) ? (total_elapsed - max_mat_elapsed) : 0;

    // GFLOPS: 2 FLOPs per MAC, 1 cycle = 1 ns (1 GHz)
    double total_macs  = static_cast<double>(GEMM_M) * GEMM_K * GEMM_N;
    double total_flops = total_macs * 2.0;
    double gflops      = (total_elapsed > 0)
                         ? total_flops / static_cast<double>(total_elapsed)
                         : 0.0;

    // mat_acc utilization = busy / (busy + queue_wait)
    double mat_busy  = static_cast<double>(top.mat_acc.busy_cycles);
    double mat_total = mat_busy + static_cast<double>(top.mat_acc.queue_wait_cycles);
    double mat_util  = (mat_total > 0) ? mat_busy / mat_total * 100.0 : 0.0;

    // vec_acc utilization
    double vec_busy  = static_cast<double>(top.vec_acc.busy_cycles);
    double vec_total = vec_busy + static_cast<double>(top.vec_acc.queue_wait_cycles);
    double vec_util  = (vec_total > 0) ? vec_busy / vec_total * 100.0 : 0.0;

    // Memory bandwidth averaged over total elapsed
    double mem_bw = (total_elapsed > 0)
                    ? static_cast<double>(top.memory.busy_cycles) * 32.0
                      / static_cast<double>(total_elapsed)
                    : 0.0;

    std::cout << "\n=== Performance summary ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "GEMM           : [" << GEMM_M << " x " << GEMM_K << " x " << GEMM_N << "]\n";
    std::cout << "Threads        : " << NUM_THREADS << "\n";
    std::cout << "Total MACs     : " << (uint64_t)total_macs << "\n";
    std::cout << "Mat phase      : " << max_mat_elapsed << " cycles  (critical-path worker)\n";
    std::cout << "Accum overhead : " << accum_overhead
              << " cycles  (beyond mat critical path)\n";
    std::cout << "Total elapsed  : " << total_elapsed << " cycles\n";
    std::cout << "Throughput     : " << gflops << " GFLOPS  (at 1 GHz)\n";
    std::cout << "mat_acc util   : " << mat_util << "%\n";
    std::cout << "vec_acc util   : " << vec_util << "%\n";
    std::cout << "Mem BW avg     : " << mem_bw << " bytes/cycle\n";

    // Per-pair timing breakdown (sorted by start time = tree order)
    const auto &pst = top.coordinator->pair_start_times;
    const auto &pet = top.coordinator->pair_end_times;
    if (!pst.empty())
    {
        std::cout << "\n=== Pairwise accumulation breakdown ===\n";
        for (size_t i = 0; i < pst.size(); i++)
        {
            sc_time dur = pet[i] - pst[i];
            std::cout << "  Pair " << i
                      << " : start=" << pst[i]
                      << "  end=" << pet[i]
                      << "  dur=" << dur
                      << "\n";
        }
        // Check for pipelined pairs: find the earliest pair start
        sc_time earliest = pst[0];
        for (auto &t : pst)
            if (t < earliest) earliest = t;
        sc_time latest_mat = max_mat_elapsed * CYCLE;
        if (earliest < latest_mat)
            std::cout << "  ** Early-start: first pair began at " << earliest
                      << " (before slowest worker finished at " << latest_mat << ")\n";
        else
            std::cout << "  All workers finished before any pair started"
                         " (no early-start benefit in this run).\n";
    }

    // -------------------------------------------------------
    // Verification checks
    // -------------------------------------------------------
    std::cout << "\n=== Verification ===\n";
    bool all_pass = true;

    // Test 1: single-thread — no accumulation
    if (NUM_THREADS == 1)
    {
        all_pass &= check(top.coordinator->vec_calls_total == 0,
                          "Single thread: no vec_acc calls for accumulation");
    }

    // Test 2: two threads — exactly one pairwise accumulation
    if (NUM_THREADS == 2)
    {
        all_pass &= check(pst.size() == 1,
                          "Two threads: exactly 1 pairwise accumulation");
    }

    // Test 3: each pair starts only after both its inputs are ready.
    // For leaf pairs (both inputs are workers), start time >= min(w_i, w_j) mat end.
    // We verify the weaker condition: every pair starts >= min worker mat end.
    if (!pst.empty())
    {
        uint64_t min_mat_elapsed = UINT64_MAX;
        for (const auto *w : top.workers)
            min_mat_elapsed = std::min(min_mat_elapsed, w->mat_elapsed_cycles);
        sc_time min_mat_time = min_mat_elapsed * CYCLE;

        bool pairs_respect_deps = true;
        for (const auto &t : pst)
            if (t < min_mat_time)
                pairs_respect_deps = false;

        all_pass &= check(pairs_respect_deps,
                          "All pairs start after at least one worker has finished");
    }

    // Test 4: total number of pairwise accumulations = T-1 (binary tree property)
    if (NUM_THREADS > 1)
    {
        uint64_t expected_pairs = (uint64_t)(NUM_THREADS - 1);
        all_pass &= check(pst.size() == expected_pairs,
                          "Total pairs = T-1 (complete binary reduction tree)");
    }

    // Test 5: total vec_acc calls = (T-1) * GEMM_ACCUM_VEC_CALLS
    if (NUM_THREADS > 1)
    {
        uint64_t expected_vec = (uint64_t)(NUM_THREADS - 1) * GEMM_ACCUM_VEC_CALLS;
        all_pass &= check(top.coordinator->vec_calls_total == expected_vec,
                          "Total vec_acc calls = (T-1) * GEMM_ACCUM_VEC_CALLS");
    }

    // Test 6: final result is ready after both the mat critical path and
    //         after all accumulations have completed (sanity)
    {
        sc_time accum_end = top.coordinator->accum_end_time;
        all_pass &= check(accum_end >= max_mat_elapsed * CYCLE,
                          "Final result ready after mat critical path");
        if (!pet.empty())
        {
            sc_time last_pair_end = pet[0];
            for (const auto &t : pet)
                if (t > last_pair_end) last_pair_end = t;
            all_pass &= check(accum_end >= last_pair_end,
                              "Final result ready after last pairwise accumulation");
        }
    }

    std::cout << (all_pass ? "\nAll checks passed.\n" : "\nSome checks FAILED.\n");

    return 0;
}
