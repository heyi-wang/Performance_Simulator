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
    MatmulTop top("top");

    std::cout << "=== K-split GEMM Performance Simulator ===\n";
    std::cout << "Threads          : " << NUM_THREADS           << "\n";
    std::cout << "GEMM shape       : [" << GEMM_M << " x " << GEMM_K << " x " << GEMM_N << "]\n";
    std::cout << "K per thread     : " << GEMM_K_PER_THREAD     << "\n";
    uint64_t min_mat_tiles = UINT64_MAX;
    uint64_t max_mat_tiles = 0;
    uint64_t total_mat_tiles = 0;
    for (const auto *w : top.workers)
    {
        min_mat_tiles = std::min(min_mat_tiles, w->access_mat);
        max_mat_tiles = std::max(max_mat_tiles, w->access_mat);
        total_mat_tiles += w->access_mat;
    }
    std::cout << "Mat tiles/worker : " << min_mat_tiles;
    if (min_mat_tiles != max_mat_tiles)
        std::cout << " .. " << max_mat_tiles;
    std::cout << "  (total=" << total_mat_tiles << ")\n";
    std::cout << "Quant tiles/w    : " << GEMM_QUANT_VEC_CALLS  << " vec_acc calls (int32→int8)\n";
    std::cout << "Accum tiles/pair : " << GEMM_ACCUM_VEC_CALLS  << " vec_acc calls per pair\n";
    std::cout << "mat_acc queue    : " << MAT_ACC_QUEUE_CAP      << " slots\n";
    std::cout << "vec_acc queue    : " << VEC_ACC_QUEUE_CAP      << " slots";
    if (VEC_ACC_QUEUE_CAP < (size_t)NUM_THREADS)
        std::cout << "  (< NUM_THREADS → backpressure expected during quant)";
    std::cout << "\n\n";

    sc_start();

    // -------------------------------------------------------
    // Raw accelerator & memory stats
    // -------------------------------------------------------
    std::cout << "\n=== Accelerator stats ===\n";
    std::cout << "mat_acc : reqs="         << top.mat_acc.req_count
              << "  busy_cycles="          << top.mat_acc.busy_cycles
              << "  occupied_cycles="      << top.mat_acc.occupied_cycles
              << "  queue_wait_cycles="    << top.mat_acc.queue_wait_cycles
              << "\n";
    std::cout << "vec_acc : reqs="         << top.vec_acc.req_count
              << "  busy_cycles="          << top.vec_acc.busy_cycles
              << "  occupied_cycles="      << top.vec_acc.occupied_cycles
              << "  queue_wait_cycles="    << top.vec_acc.queue_wait_cycles
              << "\n";
    std::cout << "memory  : reqs="         << top.memory.reqs
              << "  busy_cycles="          << top.memory.busy_cycles
              << "  queue_wait_cycles="    << top.memory.qwait_cycles
              << "\n";

    // -------------------------------------------------------
    // Per-worker breakdown
    // -------------------------------------------------------
    std::cout << "\n=== Per-worker stats ===\n";
    for (const auto *w : top.workers)
    {
        std::cout << "  [T" << w->tid << "]"
                  << " mat_calls="  << w->mat_calls
                  << " vec_calls="  << w->vec_calls
                  << " compute="    << w->compute_cycles
                  << " wait="       << w->wait_cycles
                  << " stall="      << w->stall_cycles
                  << " mem="        << w->mem_cycles_accum
                  << " elapsed="    << w->mat_elapsed_cycles
                  << "\n";
    }

    // -------------------------------------------------------
    // Wall-clock timing
    // mat_elapsed_cycles now represents mat + quant combined
    // (the time until the quantized partial result is ready).
    // -------------------------------------------------------
    uint64_t max_mat_elapsed = 0;
    for (const auto *w : top.workers)
        max_mat_elapsed = std::max(max_mat_elapsed, w->mat_elapsed_cycles);

    uint64_t total_elapsed =
        (uint64_t)(top.coordinator->accum_end_time / CYCLE);

    uint64_t accum_overhead =
        (total_elapsed > max_mat_elapsed) ? (total_elapsed - max_mat_elapsed) : 0;

    // GFLOPS: 2 FLOPs per MAC, CYCLE = 1 ns → 1 GHz clock
    double total_macs  = static_cast<double>(GEMM_M) * GEMM_K * GEMM_N;
    double total_flops = total_macs * 2.0;
    double gflops      = (total_elapsed > 0)
                         ? total_flops / static_cast<double>(total_elapsed)
                         : 0.0;

    // Accelerator occupancy
    double mat_busy     = static_cast<double>(top.mat_acc.busy_cycles);
    double mat_occupied = static_cast<double>(top.mat_acc.occupied_cycles);
    double mat_occ      = (total_elapsed > 0)
                          ? mat_occupied / static_cast<double>(total_elapsed) * 100.0
                          : 0.0;
    double mat_compute_util = (total_elapsed > 0)
                              ? mat_busy / static_cast<double>(total_elapsed) * 100.0
                              : 0.0;

    double vec_busy     = static_cast<double>(top.vec_acc.busy_cycles);
    double vec_occupied = static_cast<double>(top.vec_acc.occupied_cycles);
    double vec_occ      = (total_elapsed > 0)
                          ? vec_occupied / static_cast<double>(total_elapsed) * 100.0
                          : 0.0;
    double vec_compute_util = (total_elapsed > 0)
                              ? vec_busy / static_cast<double>(total_elapsed) * 100.0
                              : 0.0;

    double mem_bw = (total_elapsed > 0)
                    ? static_cast<double>(top.memory.busy_cycles) * 32.0
                      / static_cast<double>(total_elapsed)
                    : 0.0;

    // Total backpressure stall across all workers
    uint64_t total_worker_stall = 0;
    for (const auto *w : top.workers)
        total_worker_stall += w->stall_cycles;

    std::cout << "\n=== Performance summary ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "GEMM               : [" << GEMM_M << " x " << GEMM_K << " x " << GEMM_N << "]\n";
    std::cout << "Threads            : " << NUM_THREADS << "\n";
    std::cout << "Total MACs         : " << (uint64_t)total_macs << "\n";
    std::cout << "Mat+quant phase    : " << max_mat_elapsed
              << " cycles  (critical-path worker)\n";
    std::cout << "Accum overhead     : " << accum_overhead
              << " cycles  (beyond mat+quant critical path)\n";
    std::cout << "Total elapsed      : " << total_elapsed << " cycles\n";
    std::cout << "Throughput         : " << gflops << " GFLOPS  (at 1 GHz)\n";
    std::cout << "mat_acc occupancy  : " << mat_occ << "%\n";
    std::cout << "mat_acc compute    : " << mat_compute_util << "%\n";
    std::cout << "vec_acc occupancy  : " << vec_occ << "%\n";
    std::cout << "vec_acc compute    : " << vec_compute_util << "%\n";
    std::cout << "Mem BW avg         : " << mem_bw << " bytes/cycle\n";
    std::cout << "Worker stall total : " << total_worker_stall
              << " cycles  (backpressure on vec_acc)\n";
    std::cout << "Coord  stall total : " << top.coordinator->stall_cycles
              << " cycles  (backpressure on vec_acc)\n";
    std::cout << "Coord  compute     : " << top.coordinator->compute_cycles
              << " cycles  (scalar overhead during accum)\n";

    // Per-pair timing breakdown
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
                      << "  end="   << pet[i]
                      << "  dur="   << dur
                      << "\n";
        }
        sc_time earliest = pst[0];
        for (const auto &t : pst)
            if (t < earliest) earliest = t;
        sc_time latest_mat = max_mat_elapsed * CYCLE;
        if (earliest < latest_mat)
            std::cout << "  ** Early-start: first pair began at " << earliest
                      << " (before slowest worker finished at " << latest_mat << ")\n";
        else
            std::cout << "  All workers finished before any pair started.\n";
    }

    // -------------------------------------------------------
    // Verification checks
    // -------------------------------------------------------
    std::cout << "\n=== Verification ===\n";
    bool all_pass = true;

    // Test 1: single thread → no accumulation
    if (NUM_THREADS == 1)
    {
        all_pass &= check(top.coordinator->vec_calls_total == 0,
                          "Single thread: no accum vec_acc calls");
    }

    // Test 2: two threads → exactly 1 pairwise accumulation
    if (NUM_THREADS == 2)
    {
        all_pass &= check(pst.size() == 1,
                          "Two threads: exactly 1 pairwise accumulation");
    }

    // Test 3: single thread result time matches worker completion
    if (NUM_THREADS == 1)
    {
        all_pass &= check(top.coordinator->accum_end_time == top.workers[0]->mat_done_time,
                          "Single thread: final result time matches worker mat+quant completion");
    }

    // Test 4: each pair starts only after both inputs are ready
    if (!pst.empty())
    {
        bool deps_ok = true;
        for (size_t i = 0; i < pst.size(); i++)
        {
            if (pst[i] < top.coordinator->pair_left_ready_times[i] ||
                pst[i] < top.coordinator->pair_right_ready_times[i])
                deps_ok = false;
        }

        all_pass &= check(deps_ok,
                          "All pairs start after both dependency inputs are ready");
    }

    // Test 5: total pairwise accumulations = T-1 (binary tree property)
    if (NUM_THREADS > 1)
    {
        all_pass &= check(pst.size() == (size_t)(NUM_THREADS - 1),
                          "Total pairs = T-1 (complete binary reduction tree)");
    }

    // Test 6: total vec_acc calls = quant + accum
    {
        uint64_t expected_quant = (uint64_t)NUM_THREADS * GEMM_QUANT_VEC_CALLS;
        uint64_t expected_accum = (uint64_t)(NUM_THREADS > 1 ? NUM_THREADS - 1 : 0)
                                  * GEMM_ACCUM_VEC_CALLS;
        uint64_t expected_total = expected_quant + expected_accum;

        uint64_t actual_worker_vec = 0;
        for (const auto *w : top.workers)
            actual_worker_vec += w->vec_calls;
        uint64_t actual_total = actual_worker_vec + top.coordinator->vec_calls_total;

        all_pass &= check(actual_total == expected_total,
                          "Total vec_acc calls = T*QUANT + (T-1)*ACCUM");
    }

    // Test 7: final result is ready after mat+quant critical path
    {
        all_pass &= check(top.coordinator->accum_end_time >= max_mat_elapsed * CYCLE,
                          "Final result ready after mat+quant critical path");
    }

    // Test 8: coordinator scalar overhead matches pipelined accumulation model
    {
        uint64_t expected_coord_compute = 0;
        if (NUM_THREADS > 1 && GEMM_ACCUM_VEC_CALLS > 0)
            expected_coord_compute =
                (uint64_t)(NUM_THREADS - 1) * (GEMM_ACCUM_VEC_CALLS - 1) * SCALAR_OVERHEAD;

        all_pass &= check(top.coordinator->compute_cycles == expected_coord_compute,
                          "Coordinator compute cycles match pipelined scalar-overhead model");
    }

    // Test 9: occupancy is time-based and bounded by elapsed time
    {
        bool occ_ok = top.mat_acc.occupied_cycles <= total_elapsed &&
                      top.vec_acc.occupied_cycles <= total_elapsed;
        all_pass &= check(occ_ok,
                          "Accelerator occupancy is bounded by total elapsed time");
    }

    // Test 10: accumulation traffic consumes quantized inputs
    {
        bool precision_ok =
            GEMM_ACCUM_RD_BYTES == 2 * VECTOR_ACC_CAP * GEMM_QUANT_OUT_ELEM_BYTES &&
            GEMM_ACCUM_WR_BYTES == VECTOR_ACC_CAP * GEMM_ACCUM_OUT_ELEM_BYTES;
        all_pass &= check(precision_ok,
                          "Accumulation traffic matches quantized precision configuration");
    }

    // Test 11: backpressure triggers when VEC_ACC_QUEUE_CAP < NUM_THREADS
    if (VEC_ACC_QUEUE_CAP < (size_t)NUM_THREADS && NUM_THREADS > 1)
    {
        all_pass &= check(total_worker_stall > 0,
                          "Backpressure triggered (stall > 0) with tight vec_acc queue");
    }

    std::cout << (all_pass ? "\nAll checks passed.\n" : "\nSome checks FAILED.\n");

    return 0;
}
