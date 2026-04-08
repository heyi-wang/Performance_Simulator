#include "matmul_top.h"

#include <iostream>
#include <string>

static bool parse_args(int argc, char *argv[], int &thread_count)
{
    thread_count = MatmulConfig::default_thread_count;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--threads")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for --threads\n";
                return false;
            }
            thread_count = std::max(std::stoi(argv[++i]), 1);
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: " << argv[0] << " [--threads N]\n";
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

int sc_main(int argc, char *argv[])
{
    int thread_count = MatmulConfig::default_thread_count;
    if (!parse_args(argc, argv, thread_count))
        return (argc > 1) ? 1 : 0;

    MatmulRuntimeConfig cfg = MatmulRuntimeConfig::defaults(thread_count);
    MatmulTop top("top", cfg);

    std::cout << "=== K-split GEMM Performance Simulator ===\n";
    std::cout << "Threads              : " << cfg.thread_count << "\n";
    std::cout << "Active threads       : " << cfg.active_thread_count() << "\n";
    std::cout << "Mat accels           : " << cfg.mat_accel_count << "\n";
    std::cout << "Vec accels           : " << cfg.vec_accel_count << "\n";
    std::cout << "GEMM shape           : [" << cfg.gemm_m() << " x "
              << cfg.gemm_k() << " x " << cfg.gemm_n() << "]\n";
    std::cout << "K per thread         : " << cfg.gemm_k_per_thread() << "\n";

    sc_start();

    const MatmulSimulationStats stats = top.collect_stats();
    top.print_report(std::cout);

    bool pass =
        stats.verification_passed &&
        stats.total_elapsed >= stats.max_mat_elapsed &&
        stats.vec_req_total == stats.expected_vec_req_total &&
        stats.mat_req_total == stats.expected_mat_req_total;

    std::cout << "\n=== Verification ===\n";
    std::cout << "mat reqs             : " << stats.mat_req_total
              << " (expected " << stats.expected_mat_req_total << ")\n";
    std::cout << "vec reqs             : " << stats.vec_req_total
              << " (expected " << stats.expected_vec_req_total << ")\n";
    std::cout << "final status         : " << (pass ? "PASS" : "FAIL") << "\n";
    return pass ? 0 : 2;
}
