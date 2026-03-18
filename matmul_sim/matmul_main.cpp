#include "matmul_top.h"
#include "matmul_config.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

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

static bool parse_args(int argc, char *argv[], int &thread_count)
{
    thread_count = NUM_THREADS;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--threads")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for --threads\n";
                return false;
            }

            std::string value = argv[++i];
            try
            {
                size_t pos = 0;
                int parsed = std::stoi(value, &pos);
                if (pos != value.size() || parsed < 1)
                {
                    std::cerr << "Invalid --threads value: " << value << "\n";
                    return false;
                }
                thread_count = parsed;
            }
            catch (const std::exception &)
            {
                std::cerr << "Invalid --threads value: " << value << "\n";
                return false;
            }
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: " << argv[0] << " [--threads N]\n";
            std::cout << "  --threads N   Runtime worker count for the K-split GEMM simulator\n";
            return false;
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }

    return true;
}

// ============================================================
// sc_main
// ============================================================
int sc_main(int argc, char *argv[])
{
    int thread_count = NUM_THREADS;
    if (!parse_args(argc, argv, thread_count))
        return (argc > 1) ? 1 : 0;

    MatmulConfig cfg(thread_count);
    MatmulTop top("top", cfg);

    std::cout << "=== K-split GEMM Performance Simulator ===\n";
    std::cout << "Threads          : " << cfg.thread_count << "\n";
    std::cout << "Mat accels       : " << cfg.mat_accel_count << "\n";
    std::cout << "Vec accels       : " << cfg.vec_accel_count << "\n";
    std::cout << "GEMM shape       : [" << MatmulConfig::gemm_m << " x "
              << MatmulConfig::gemm_k << " x " << MatmulConfig::gemm_n << "]\n";
    std::cout << "K per thread     : " << cfg.gemm_k_per_thread() << "\n";
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
    std::cout << "Final quant      : " << MatmulConfig::gemm_quant_vec_calls()
              << " vec_acc calls (int32→int8, once after reduction)\n";
    std::cout << "Accum tiles/pair : " << MatmulConfig::gemm_accum_vec_calls()
              << " vec_acc calls per pair\n";
    std::cout << "mat_acc queue    : " << cfg.mat_acc_queue_cap()
              << " shared slots across " << cfg.mat_accel_count << " units\n";
    std::cout << "vec_acc queue    : " << cfg.vec_acc_queue_cap()
              << " shared slots across " << cfg.vec_accel_count << " units";
    if (cfg.vec_acc_queue_cap() < static_cast<size_t>(cfg.thread_count))
        std::cout << "  (< NUM_THREADS → backpressure expected during reduction/final quant)";
    std::cout << "\n\n";

    sc_start();

    // -------------------------------------------------------
    // Raw accelerator & memory stats
    // -------------------------------------------------------
    uint64_t mat_req_total = top.mat_acc.req_count_total();
    uint64_t mat_busy_total = top.mat_acc.busy_cycles_total();
    uint64_t mat_occupied_total = top.mat_acc.occupied_cycles_total();
    uint64_t mat_qwait_total = top.mat_acc.queue_wait_cycles_total();
    uint64_t vec_req_total = top.vec_acc.req_count_total();
    uint64_t vec_busy_total = top.vec_acc.busy_cycles_total();
    uint64_t vec_occupied_total = top.vec_acc.occupied_cycles_total();
    uint64_t vec_qwait_total = top.vec_acc.queue_wait_cycles_total();

    std::cout << "\n=== Accelerator stats ===\n";
    std::cout << "mat_acc pool : units="    << top.mat_acc.instance_count()
              << "  reqs="                  << mat_req_total
              << "  busy_cycles="           << mat_busy_total
              << "  occupied_cycles="       << mat_occupied_total
              << "  shared_qwait_cycles="   << mat_qwait_total
              << "\n";
    std::cout << "vec_acc pool : units="    << top.vec_acc.instance_count()
              << "  reqs="                  << vec_req_total
              << "  busy_cycles="           << vec_busy_total
              << "  occupied_cycles="       << vec_occupied_total
              << "  shared_qwait_cycles="   << vec_qwait_total
              << "\n";
    std::cout << "memory  : reqs="         << top.memory.reqs
              << "  busy_cycles="          << top.memory.busy_cycles
              << "  queue_wait_cycles="    << top.memory.qwait_cycles
              << "\n";
    std::cout << "  Note: pool qwait is shared-queue waiting before dispatch; "
                 "per-unit qwait is not reported because units are dispatched only when free.\n";

    std::cout << "\n=== Accelerator instances ===\n";
    for (size_t i = 0; i < top.mat_acc.units.size(); ++i)
    {
        const auto &unit = top.mat_acc.units[i];
        std::cout << "  mat[" << i << "]"
                  << " reqs=" << unit->req_count
                  << " busy=" << unit->busy_cycles
                  << " occupied=" << unit->occupied_cycles
                  << "\n";
    }
    for (size_t i = 0; i < top.vec_acc.units.size(); ++i)
    {
        const auto &unit = top.vec_acc.units[i];
        std::cout << "  vec[" << i << "]"
                  << " reqs=" << unit->req_count
                  << " busy=" << unit->busy_cycles
                  << " occupied=" << unit->occupied_cycles
                  << "\n";
    }

    // -------------------------------------------------------
    // Per-worker breakdown
    // -------------------------------------------------------
    std::cout << "\n=== Per-worker stats ===\n";
    for (const auto *w : top.workers)
    {
        std::cout << "  [T" << w->tid << "]"
                  << " mat_calls="  << w->mat_calls
                  << " vec_calls="  << w->vec_calls
                  << " accum_vec="  << w->accum_vec_calls
                  << " quant_vec="  << w->quant_vec_calls
                  << " pairs="      << w->reduction_pairs
                  << " compute="    << w->compute_cycles
                  << " wait="       << w->wait_cycles
                  << " stall="      << w->stall_cycles
                  << " mem="        << w->mem_cycles_accum
                  << " mat_elapsed=" << w->mat_elapsed_cycles
                  << " elapsed="    << w->elapsed_cycles
                  << "\n";
    }

    // -------------------------------------------------------
    // Wall-clock timing
    // total elapsed is now the slowest worker's full lifetime.
    // -------------------------------------------------------
    uint64_t max_mat_elapsed = 0;
    uint64_t max_elapsed = 0;
    for (const auto *w : top.workers)
    {
        max_mat_elapsed = std::max(max_mat_elapsed, w->mat_elapsed_cycles);
        max_elapsed = std::max(max_elapsed, w->elapsed_cycles);
    }

    uint64_t total_elapsed = max_elapsed;

    uint64_t accum_overhead =
        (total_elapsed > max_mat_elapsed) ? (total_elapsed - max_mat_elapsed) : 0;

    // GFLOPS: 2 FLOPs per MAC, CYCLE = 1 ns → 1 GHz clock
    double total_macs =
        static_cast<double>(MatmulConfig::gemm_m) * MatmulConfig::gemm_k * MatmulConfig::gemm_n;
    double total_flops = total_macs * 2.0;
    double gflops = (total_elapsed > 0)
                        ? total_flops / static_cast<double>(total_elapsed)
                        : 0.0;

    // Accelerator occupancy
    double mat_busy     = static_cast<double>(mat_busy_total);
    double mat_occupied = static_cast<double>(mat_occupied_total);
    double mat_capacity = static_cast<double>(total_elapsed) *
                          static_cast<double>(top.mat_acc.instance_count());
    double mat_occ      = (mat_capacity > 0.0)
                              ? mat_occupied / mat_capacity * 100.0
                              : 0.0;
    double mat_compute_util = (mat_capacity > 0.0)
                                  ? mat_busy / mat_capacity * 100.0
                                  : 0.0;

    double vec_busy     = static_cast<double>(vec_busy_total);
    double vec_occupied = static_cast<double>(vec_occupied_total);
    double vec_capacity = static_cast<double>(total_elapsed) *
                          static_cast<double>(top.vec_acc.instance_count());
    double vec_occ      = (vec_capacity > 0.0)
                              ? vec_occupied / vec_capacity * 100.0
                              : 0.0;
    double vec_compute_util = (vec_capacity > 0.0)
                                  ? vec_busy / vec_capacity * 100.0
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
    std::cout << "GEMM               : [" << MatmulConfig::gemm_m << " x "
              << MatmulConfig::gemm_k << " x " << MatmulConfig::gemm_n << "]\n";
    std::cout << "Threads            : " << cfg.thread_count << "\n";
    std::cout << "Total MACs         : " << static_cast<uint64_t>(total_macs) << "\n";
    std::cout << "Mat phase          : " << max_mat_elapsed
              << " cycles  (critical-path worker)\n";
    std::cout << "Accum overhead     : " << accum_overhead
              << " cycles  (beyond worker mat critical path)\n";
    std::cout << "Total elapsed      : " << total_elapsed << " cycles\n";
    std::cout << "Throughput         : " << gflops << " GFLOPS  (at 1 GHz)\n";
    std::cout << "mat_acc occupancy  : " << mat_occ << "%\n";
    std::cout << "mat_acc compute    : " << mat_compute_util << "%\n";
    std::cout << "vec_acc occupancy  : " << vec_occ << "%\n";
    std::cout << "vec_acc compute    : " << vec_compute_util << "%\n";
    std::cout << "Mem BW avg         : " << mem_bw << " bytes/cycle\n";
    std::cout << "Worker mat stall   : " << total_worker_stall
              << " cycles  (all worker-side backpressure across mat/reduction/quant)\n";
    std::cout << "Post-mat stall     : " << top.coordinator->stall_cycles
              << " cycles  (worker-driven reduction/final-quant vec backpressure)\n";
    std::cout << "Post-mat compute   : " << top.coordinator->compute_cycles
              << " cycles  (worker-driven reduction/final-quant scalar+service time)\n";

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
            if (t < earliest)
                earliest = t;
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

    // Test 1: single thread → no accumulation calls
    if (cfg.thread_count == 1)
    {
        all_pass &= check(top.coordinator->accum_vec_calls_total == 0,
                          "Single thread: no accumulation vec_acc calls");
    }

    // Test 2: two threads → exactly 1 pairwise accumulation
    if (cfg.thread_count == 2)
    {
        all_pass &= check(pst.size() == 1,
                          "Two threads: exactly 1 pairwise accumulation");
    }

    // Test 3: single thread still performs one final quantization pass
    if (cfg.thread_count == 1)
    {
        all_pass &= check(top.coordinator->final_quant_calls_total ==
                              MatmulConfig::gemm_quant_vec_calls(),
                          "Single thread: final quantization still runs once");
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
    if (cfg.thread_count > 1)
    {
        all_pass &= check(pst.size() == static_cast<size_t>(cfg.thread_count - 1),
                          "Total pairs = T-1 (complete binary reduction tree)");
    }

    // Test 6: total vec_acc calls = final quant + accum
    {
        uint64_t expected_quant = MatmulConfig::gemm_quant_vec_calls();
        uint64_t expected_accum =
            static_cast<uint64_t>(cfg.thread_count > 1 ? cfg.thread_count - 1 : 0) *
            MatmulConfig::gemm_accum_vec_calls();
        uint64_t expected_total = expected_quant + expected_accum;

        uint64_t actual_worker_vec = 0;
        for (const auto *w : top.workers)
            actual_worker_vec += w->vec_calls;

        all_pass &= check(actual_worker_vec == expected_total &&
                              top.coordinator->vec_calls_total == expected_total,
                          "Worker vec_acc calls match FINAL_QUANT + (T-1)*ACCUM");
    }

    // Test 7: full elapsed is determined by the slowest worker
    {
        all_pass &= check(total_elapsed == max_elapsed &&
                              total_elapsed >= max_mat_elapsed,
                          "Total elapsed equals the slowest worker completion time");
    }

    // Test 8: worker-driven post-mat compute matches reduction + quant model
    {
        auto expected_worker_stream_compute = [&](uint64_t call_count,
                                                  uint64_t service_cycles) -> uint64_t {
            if (call_count == 0)
                return 0;
            return call_count * service_cycles +
                   (call_count - 1) * SCALAR_OVERHEAD;
        };

        uint64_t expected_coord_compute = 0;
        if (cfg.thread_count > 1)
        {
            expected_coord_compute =
                static_cast<uint64_t>(cfg.thread_count - 1) *
                expected_worker_stream_compute(MatmulConfig::gemm_accum_vec_calls(),
                                               VECTOR_ACC_CYCLE);
        }
        for (int tid = 0; tid < cfg.thread_count; ++tid)
        {
            uint64_t base = MatmulConfig::gemm_quant_vec_calls() /
                            static_cast<uint64_t>(cfg.thread_count);
            uint64_t rem = MatmulConfig::gemm_quant_vec_calls() %
                           static_cast<uint64_t>(cfg.thread_count);
            uint64_t quant_calls = base + ((static_cast<uint64_t>(tid) < rem) ? 1 : 0);
            expected_coord_compute +=
                expected_worker_stream_compute(quant_calls, VECTOR_ACC_CYCLE);
        }

        all_pass &= check(top.coordinator->compute_cycles == expected_coord_compute,
                          "Worker-driven post-mat compute matches reduction-plus-final-quant model");
    }

    // Test 9: occupancy is time-based and bounded by elapsed time
    {
        bool occ_ok = true;
        for (const auto &unit : top.mat_acc.units)
            occ_ok &= unit->occupied_cycles <= total_elapsed;
        for (const auto &unit : top.vec_acc.units)
            occ_ok &= unit->occupied_cycles <= total_elapsed;
        all_pass &= check(occ_ok,
                          "Accelerator occupancy is bounded by total elapsed time");
    }

    // Test 10: accumulation stays high precision until final quantization
    {
        bool precision_ok =
            MatmulConfig::gemm_accum_rd_bytes ==
                2 * VECTOR_ACC_CAP * MatmulConfig::gemm_quant_in_elem_bytes &&
            MatmulConfig::gemm_accum_wr_bytes ==
                VECTOR_ACC_CAP * MatmulConfig::gemm_accum_out_elem_bytes;
        all_pass &= check(precision_ok,
                          "Accumulation traffic matches high-precision reduction configuration");
    }

    // Test 11: post-mat quantization is split across workers exactly once
    {
        uint64_t total_quant = 0;
        bool static_partition_ok = true;
        for (int tid = 0; tid < cfg.thread_count; ++tid)
        {
            uint64_t base = MatmulConfig::gemm_quant_vec_calls() /
                            static_cast<uint64_t>(cfg.thread_count);
            uint64_t rem = MatmulConfig::gemm_quant_vec_calls() %
                           static_cast<uint64_t>(cfg.thread_count);
            uint64_t expected = base + ((static_cast<uint64_t>(tid) < rem) ? 1 : 0);
            total_quant += top.workers[static_cast<size_t>(tid)]->quant_vec_calls;
            static_partition_ok &=
                top.workers[static_cast<size_t>(tid)]->quant_vec_calls == expected;
        }

        all_pass &= check(static_partition_ok &&
                              total_quant == MatmulConfig::gemm_quant_vec_calls(),
                          "Final quantization is partitioned across workers exactly once");
    }

    // Test 12: quantization starts after full reduction completes
    {
        all_pass &= check(top.coordinator->quant_start_time >= top.coordinator->accum_end_time,
                          "Final quantization starts after the root reduction completes");
    }

    // Test 13: backpressure triggers when post-mat vec pressure is high enough
    if (cfg.vec_acc_queue_cap() <= static_cast<size_t>(cfg.thread_count) &&
        cfg.thread_count > 1 &&
        cfg.vec_accel_count == 1)
    {
        all_pass &= check(top.coordinator->stall_cycles > 0 || total_worker_stall > 0,
                          "Backpressure triggered (stall > 0) with tight vec_acc queue");
    }

    // Test 14: multiple accelerator instances are used when configured
    if (cfg.mat_accel_count > 1 && cfg.thread_count > 1 && mat_req_total > 0)
    {
        size_t active_units = 0;
        for (const auto &unit : top.mat_acc.units)
            active_units += (unit->req_count > 0) ? 1 : 0;
        all_pass &= check(active_units >= 2,
                          "Multiple mat accelerators receive work when enabled");
    }

    if (cfg.vec_accel_count > 1 && cfg.thread_count > 2 && vec_req_total > 0)
    {
        size_t active_units = 0;
        for (const auto &unit : top.vec_acc.units)
            active_units += (unit->req_count > 0) ? 1 : 0;
        all_pass &= check(active_units >= 2,
                          "Multiple vec accelerators receive work when enabled");
    }

    std::cout << (all_pass ? "\nAll checks passed.\n" : "\nSome checks FAILED.\n");

    return all_pass ? 0 : 2;
}
