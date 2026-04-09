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
