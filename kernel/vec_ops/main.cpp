#include "vec_ops_top.h"
#include "perfetto_trace.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>

static bool parse_op(const std::string &name, VopType &op)
{
    if (name == "elemwise_add" || name == "add")        { op = VOP_ELEMWISE_ADD;         return true; }
    if (name == "elemwise_mul" || name == "mul")        { op = VOP_ELEMWISE_MUL;         return true; }
    if (name == "scalar_mul")                            { op = VOP_SCALAR_MUL;           return true; }
    if (name == "quantize_i32_to_i8" || name == "qi32") { op = VOP_QUANTIZE_I32_TO_I8;   return true; }
    if (name == "quantize_i16_to_i8" || name == "qi16") { op = VOP_QUANTIZE_I16_TO_I8;   return true; }
    if (name == "dequantize_i8_to_i32" || name == "deq"){ op = VOP_DEQUANTIZE_I8_TO_I32; return true; }
    if (name == "dot_product_i8"      || name == "dot") { op = VOP_DOT_PRODUCT_I8;       return true; }
    if (name == "scale_requant_i8"    || name == "srq") { op = VOP_SCALE_REQUANT_I8;     return true; }
    return false;
}

static void print_usage(const char *prog)
{
    std::cout << "Usage: " << prog
              << " [--workers N] [--channels C] [--height H] [--width W]"
              << " [--vec-instances N] [--op NAME]\n"
              << "  --op NAME           one of: elemwise_add, elemwise_mul,\n"
              << "                              scalar_mul, quantize_i32_to_i8,\n"
              << "                              quantize_i16_to_i8, dequantize_i8_to_i32,\n"
              << "                              dot_product_i8, scale_requant_i8\n";
}

static bool parse_args(int argc,
                       char *argv[],
                       int &workers,
                       int &channels,
                       int &height,
                       int &width,
                       int &vec_instances,
                       VopType &op,
                       bool &op_set)
{
    workers = -1;
    channels = -1;
    height = -1;
    width = -1;
    vec_instances = -1;
    op_set = false;

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
        else if (arg == "--vec-instances")
        {
            if (!need_value("--vec-instances")) return false;
            vec_instances = std::max(std::stoi(argv[++i]), 1);
        }
        else if (arg == "--op")
        {
            if (!need_value("--op")) return false;
            if (!parse_op(argv[++i], op))
            {
                std::cerr << "Unknown op: " << argv[i] << "\n";
                print_usage(argv[0]);
                return false;
            }
            op_set = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            return false;
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return false;
        }
    }

    return true;
}

int sc_main(int argc, char *argv[])
{
    [[maybe_unused]] std::string trace_out = perf_take_trace_out_arg(argc, argv);
#ifdef PERFETTO_TRACE
    if (!trace_out.empty()) perf_trace_enabled() = true;
#endif
    int workers = -1;
    int channels = -1;
    int height = -1;
    int width = -1;
    int vec_instances = -1;
    VopType op = VOP_SELECTED_OP;
    bool op_set = false;
    if (!parse_args(argc, argv, workers, channels, height, width,
                    vec_instances, op, op_set))
        return (argc > 1) ? 1 : 0;

    VecOpsRuntimeConfig cfg = VecOpsRuntimeConfig::defaults();
    if (workers > 0)       cfg.worker_count = workers;
    if (channels > 0)      cfg.channels = channels;
    if (height > 0)        cfg.height = height;
    if (width > 0)         cfg.width = width;
    if (vec_instances > 0) cfg.vec_acc_instances = vec_instances;
    if (op_set)            cfg.op = op;

    VecOpsTop top("vec_ops_top", cfg);
    sc_start();

#ifdef PERFETTO_TRACE
    if (!trace_out.empty())
    {
        perf_trace_write_json(trace_out.c_str());
        std::cerr << "Perfetto trace written to " << trace_out << "\n";
    }
#endif

    top.print_report(std::cout);

    const VecOpsSimulationStats stats = top.collect_stats();
    return stats.verification_passed ? 0 : 2;
}
