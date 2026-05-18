#include "pooling_top.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>

static bool parse_args(int argc,
                       char *argv[],
                       int &workers,
                       int &channels,
                       int &height,
                       int &width,
                       int64_t &max_inflight,
                       int64_t &max_inflight_dma_writes,
                       int64_t &l2_base_lat)
{
    workers = -1;
    channels = -1;
    height = -1;
    width = -1;
    max_inflight = -1;
    max_inflight_dma_writes = -1;
    l2_base_lat = -1;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto need_value = [&](const char *flag) {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for " << flag << "\n";
                return false;
            }
            return true;
        };

        if (arg == "--workers")
        {
            if (!need_value("--workers")) return false;
            workers = std::max(std::stoi(argv[++i]), 1);
        }
        else if (arg == "--channels")
        {
            if (!need_value("--channels")) return false;
            channels = std::max(std::stoi(argv[++i]), 1);
        }
        else if (arg == "--height")
        {
            if (!need_value("--height")) return false;
            height = std::max(std::stoi(argv[++i]), 1);
        }
        else if (arg == "--width")
        {
            if (!need_value("--width")) return false;
            width = std::max(std::stoi(argv[++i]), 1);
        }
        else if (arg == "--max-inflight-vec")
        {
            if (!need_value("--max-inflight-vec")) return false;
            max_inflight = std::max<int64_t>(std::stoll(argv[++i]), 1);
        }
        else if (arg == "--max-inflight-dma-writes")
        {
            if (!need_value("--max-inflight-dma-writes")) return false;
            max_inflight_dma_writes = std::max<int64_t>(std::stoll(argv[++i]), 1);
        }
        else if (arg == "--dma-base-lat")
        {
            if (!need_value("--dma-base-lat")) return false;
            l2_base_lat = std::max<int64_t>(std::stoll(argv[++i]), 0);
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: " << argv[0]
                      << " [--workers N] [--channels C]"
                      << " [--height H] [--width W]"
                      << " [--max-inflight-vec N]"
                      << " [--max-inflight-dma-writes N]"
                      << " [--dma-base-lat N]\n";
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
    int workers = -1;
    int channels = -1;
    int height = -1;
    int width = -1;
    int64_t max_inflight = -1;
    int64_t max_inflight_dma_writes = -1;
    int64_t l2_base_lat = -1;
    if (!parse_args(argc, argv,
                    workers, channels, height, width,
                    max_inflight, max_inflight_dma_writes, l2_base_lat))
        return (argc > 1) ? 1 : 0;

    PoolRuntimeConfig cfg = PoolRuntimeConfig::defaults();
    if (workers > 0)      cfg.worker_count = workers;
    if (channels > 0)     cfg.channels = channels;
    if (height > 0)       cfg.height = height;
    if (width > 0)        cfg.width = width;
    if (max_inflight > 0) cfg.max_inflight_vec_reqs = static_cast<uint64_t>(max_inflight);
    if (max_inflight_dma_writes > 0)
        cfg.max_inflight_dma_writes = static_cast<uint64_t>(max_inflight_dma_writes);
    if (l2_base_lat >= 0) cfg.l2_base_lat = static_cast<uint64_t>(l2_base_lat);

    PoolTop top("pool_top", cfg);
    sc_start();

    top.print_report(std::cout);

    const PoolSimulationStats stats = top.collect_stats();
    return stats.verification_passed ? 0 : 2;
}
