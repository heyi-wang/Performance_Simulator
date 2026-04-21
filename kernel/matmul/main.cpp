#include "matmul_top.h"

#include <algorithm>
#include <iostream>
#include <string>

static bool parse_args(int argc,
                       char *argv[],
                       int &thread_count,
                       uint64_t &accumulator_register_count,
                       uint64_t &gemm_m,
                       uint64_t &gemm_k,
                       uint64_t &gemm_n)
{
    thread_count = MatmulConfig::default_thread_count;
    accumulator_register_count =
        MatmulConfig::default_accumulator_register_count;
    gemm_m = MatmulConfig::gemm_m;
    gemm_k = MatmulConfig::gemm_k;
    gemm_n = MatmulConfig::gemm_n;

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
        else if (arg == "--accum-registers")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for --accum-registers\n";
                return false;
            }
            accumulator_register_count =
                std::max<uint64_t>(std::stoull(argv[++i]), 1);
        }
        else if (arg == "--gemm-m")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for --gemm-m\n";
                return false;
            }
            gemm_m = std::max<uint64_t>(std::stoull(argv[++i]), 1);
        }
        else if (arg == "--gemm-k")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for --gemm-k\n";
                return false;
            }
            gemm_k = std::max<uint64_t>(std::stoull(argv[++i]), 1);
        }
        else if (arg == "--gemm-n")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for --gemm-n\n";
                return false;
            }
            gemm_n = std::max<uint64_t>(std::stoull(argv[++i]), 1);
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: " << argv[0]
                      << " [--threads N] [--accum-registers N]"
                      << " [--gemm-m M] [--gemm-k K] [--gemm-n N]\n";
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
    uint64_t accumulator_register_count =
        MatmulConfig::default_accumulator_register_count;
    uint64_t gemm_m = MatmulConfig::gemm_m;
    uint64_t gemm_k = MatmulConfig::gemm_k;
    uint64_t gemm_n = MatmulConfig::gemm_n;
    if (!parse_args(argc,
                    argv,
                    thread_count,
                    accumulator_register_count,
                    gemm_m,
                    gemm_k,
                    gemm_n))
        return (argc > 1) ? 1 : 0;

    MatmulRuntimeConfig cfg =
        MatmulRuntimeConfig::defaults(thread_count,
                                      MAT_ACCEL_COUNT,
                                      VEC_ACCEL_COUNT,
                                      accumulator_register_count);
    cfg.workload_n = 1;
    cfg.workload_h = gemm_m;
    cfg.workload_w = 1;
    cfg.workload_c_in = gemm_k;
    cfg.workload_kh = 1;
    cfg.workload_kw = 1;
    cfg.workload_c_out = gemm_n;
    MatmulTop top("top", cfg);

    sc_start();

    const MatmulSimulationStats stats = top.collect_stats();
    top.print_report(std::cout);

    bool pass =
        stats.verification_passed &&
        stats.total_elapsed >= stats.max_mat_elapsed &&
        stats.vec_req_total == stats.expected_vec_req_total &&
        stats.mat_req_total == stats.expected_mat_req_total;

    return pass ? 0 : 2;
}
