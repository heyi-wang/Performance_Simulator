#include <systemc>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nafnet_hw_config.h"
#include "nafnet_kernel_bridge.h"
#include "nafnet_layers.h"

using namespace sc_core;

struct NafLayerStats
{
    uint64_t mat_reqs = 0;
    uint64_t vec_reqs = 0;
    uint64_t mem_reqs = 0;
    uint64_t accel_cycles = 0;
    uint64_t cpu_cycles = 0;
    uint64_t wait_cycles = 0;
    uint64_t mem_cycles = 0;
    uint64_t rd_bytes = 0;
    uint64_t wr_bytes = 0;
};

struct LayerRunResult
{
    NafLayerStats stats;
    bool verification_passed = false;
};

struct NafOptions
{
    bool intro_only = false;
    bool nafblock_only = false;
    int block_c = 32;
    int block_h = 64;
    int block_w = 64;
};

static NafLayerStats convert_stats(const MatmulSimulationStats &stats)
{
    NafLayerStats out;
    out.mat_reqs = stats.mat_req_total;
    out.vec_reqs = stats.vec_req_total;
    out.mem_reqs = stats.memory_reqs;
    out.accel_cycles = stats.mat_busy_total + stats.vec_busy_total;
    out.cpu_cycles = stats.coordinator_compute;
    out.wait_cycles = stats.total_worker_stall + stats.coordinator_stall;
    out.mem_cycles = stats.memory_busy_cycles;
    out.rd_bytes = stats.total_rd_bytes;
    out.wr_bytes = stats.total_wr_bytes;
    return out;
}

static NafLayerStats convert_stats(const DwConvSimulationStats &stats)
{
    NafLayerStats out;
    out.vec_reqs = stats.total_vec_calls;
    out.mem_reqs = stats.memory_reqs;
    out.accel_cycles = stats.vec_acc_busy_cycles;
    out.wait_cycles = stats.total_wait_cycles;
    out.mem_cycles = stats.total_mem_cycles;
    out.rd_bytes = stats.total_rd_bytes;
    out.wr_bytes = stats.total_wr_bytes;
    return out;
}

static NafLayerStats convert_stats(const LayerNormSimulationStats &stats)
{
    NafLayerStats out;
    out.vec_reqs = stats.total_vec_reqs;
    out.mem_reqs = stats.memory_reqs;
    out.mem_cycles = stats.memory_busy_cycles;
    out.rd_bytes = stats.total_rd_bytes;
    out.wr_bytes = stats.total_wr_bytes;
    for (const auto &step : stats.steps)
    {
        out.accel_cycles += step.accel_cycles;
        out.cpu_cycles += step.scalar_cycles;
        out.wait_cycles += step.wait_cycles;
    }
    return out;
}

static NafLayerStats convert_stats(const PoolSimulationStats &stats)
{
    NafLayerStats out;
    out.vec_reqs = stats.total_vec_calls;
    out.mem_reqs = stats.memory_reqs;
    out.accel_cycles = stats.vec_acc_busy_cycles;
    out.wait_cycles = stats.total_wait_cycles;
    out.mem_cycles = stats.total_mem_cycles;
    out.rd_bytes = stats.total_rd_bytes;
    out.wr_bytes = stats.total_wr_bytes;
    return out;
}

static NafLayerStats convert_stats(const VecOpsSimulationStats &stats)
{
    NafLayerStats out;
    out.vec_reqs = stats.total_vec_calls;
    out.mem_reqs = stats.memory_reqs;
    out.accel_cycles = stats.vec_acc_busy_cycles;
    out.wait_cycles = stats.total_wait_cycles;
    out.mem_cycles = stats.total_mem_cycles;
    out.rd_bytes = stats.total_rd_bytes;
    out.wr_bytes = stats.total_wr_bytes;
    return out;
}

struct LayerRunnerBase
{
    LayerDesc layer;

    explicit LayerRunnerBase(const LayerDesc &layer_) : layer(layer_) {}
    virtual ~LayerRunnerBase() = default;
    virtual LayerRunResult run() = 0;
};

struct MatmulLayerRunner : LayerRunnerBase
{
    sc_event start_ev;
    sc_event done_ev;
    std::unique_ptr<MatmulTop> top;

    explicit MatmulLayerRunner(const LayerDesc &layer_)
        : LayerRunnerBase(layer_)
    {
        top = std::make_unique<MatmulTop>(sc_gen_unique_name("naf_matmul_top"),
                                          make_matmul_runtime_config(layer),
                                          &start_ev,
                                          &done_ev);
    }

    LayerRunResult run() override
    {
        start_ev.notify(SC_ZERO_TIME);
        wait(done_ev);
        const MatmulSimulationStats stats = top->collect_stats();
        return {convert_stats(stats), stats.verification_passed};
    }
};

struct DwConvLayerRunner : LayerRunnerBase
{
    sc_event start_ev;
    sc_event done_ev;
    std::unique_ptr<DwConvTop> top;

    explicit DwConvLayerRunner(const LayerDesc &layer_)
        : LayerRunnerBase(layer_)
    {
        top = std::make_unique<DwConvTop>(sc_gen_unique_name("naf_dw_top"),
                                          make_dwconv_runtime_config(layer),
                                          &start_ev,
                                          &done_ev);
    }

    LayerRunResult run() override
    {
        start_ev.notify(SC_ZERO_TIME);
        wait(done_ev);
        const DwConvSimulationStats stats = top->collect_stats();
        return {convert_stats(stats), stats.verification_passed};
    }
};

struct LayerNormLayerRunner : LayerRunnerBase
{
    sc_event start_ev;
    sc_event done_ev;
    std::unique_ptr<LayerNormTop> top;

    explicit LayerNormLayerRunner(const LayerDesc &layer_)
        : LayerRunnerBase(layer_)
    {
        top = std::make_unique<LayerNormTop>(sc_gen_unique_name("naf_ln_top"),
                                             make_layernorm_runtime_config(layer),
                                             &start_ev,
                                             &done_ev);
    }

    LayerRunResult run() override
    {
        start_ev.notify(SC_ZERO_TIME);
        wait(done_ev);
        const LayerNormSimulationStats stats = top->collect_stats();
        return {convert_stats(stats), stats.verification_passed};
    }
};

struct PoolLayerRunner : LayerRunnerBase
{
    sc_event start_ev;
    sc_event done_ev;
    std::unique_ptr<PoolTop> top;

    explicit PoolLayerRunner(const LayerDesc &layer_)
        : LayerRunnerBase(layer_)
    {
        top = std::make_unique<PoolTop>(sc_gen_unique_name("naf_pool_top"),
                                        make_pool_runtime_config(layer),
                                        &start_ev,
                                        &done_ev);
    }

    LayerRunResult run() override
    {
        start_ev.notify(SC_ZERO_TIME);
        wait(done_ev);
        const PoolSimulationStats stats = top->collect_stats();
        return {convert_stats(stats), stats.verification_passed};
    }
};

struct VecOpsLayerRunner : LayerRunnerBase
{
    struct Phase
    {
        VopType op = VOP_ELEMWISE_MUL;
        sc_event start_ev;
        sc_event done_ev;
        std::unique_ptr<VecOpsTop> top;
    };

    std::vector<std::unique_ptr<Phase>> phases;

    explicit VecOpsLayerRunner(const LayerDesc &layer_)
        : LayerRunnerBase(layer_)
    {
        for (VopType op : vecops_phases_for_layer(layer))
        {
            auto phase = std::make_unique<Phase>();
            phase->op = op;
            phase->top = std::make_unique<VecOpsTop>(
                sc_gen_unique_name("naf_vec_top"),
                make_vecops_runtime_config(layer, op),
                &phase->start_ev,
                &phase->done_ev);
            phases.push_back(std::move(phase));
        }
    }

    LayerRunResult run() override
    {
        LayerRunResult result;
        result.verification_passed = true;

        for (auto &phase : phases)
        {
            phase->start_ev.notify(SC_ZERO_TIME);
            wait(phase->done_ev);
            const VecOpsSimulationStats stats = phase->top->collect_stats();
            NafLayerStats converted = convert_stats(stats);
            result.stats.vec_reqs += converted.vec_reqs;
            result.stats.mem_reqs += converted.mem_reqs;
            result.stats.accel_cycles += converted.accel_cycles;
            result.stats.wait_cycles += converted.wait_cycles;
            result.stats.mem_cycles += converted.mem_cycles;
            result.stats.rd_bytes += converted.rd_bytes;
            result.stats.wr_bytes += converted.wr_bytes;
            result.verification_passed &= stats.verification_passed;
        }

        return result;
    }
};

static std::unique_ptr<LayerRunnerBase> make_runner(const LayerDesc &layer)
{
    switch (layer.backend)
    {
    case BACKEND_MATMUL:
        return std::make_unique<MatmulLayerRunner>(layer);
    case BACKEND_DWCONV:
        return std::make_unique<DwConvLayerRunner>(layer);
    case BACKEND_LAYERNORM:
        return std::make_unique<LayerNormLayerRunner>(layer);
    case BACKEND_POOLING:
        return std::make_unique<PoolLayerRunner>(layer);
    case BACKEND_VECOPS:
        return std::make_unique<VecOpsLayerRunner>(layer);
    }
    return nullptr;
}

struct NafTop : sc_module
{
    NafOptions opts;
    std::vector<LayerDesc> layers;
    std::vector<std::unique_ptr<LayerRunnerBase>> runners;
    std::vector<LayerRunResult> results;

    SC_HAS_PROCESS(NafTop);

    NafTop(sc_module_name name, const NafOptions &opts_)
        : sc_module(name),
          opts(opts_)
    {
        if (opts.nafblock_only)
            layers = build_nafblock_only_layers(opts.block_c, opts.block_h, opts.block_w);
        else
            layers = build_nafnet32_layers();

        if (opts.intro_only && !opts.nafblock_only)
            layers.resize(1);

        results.resize(layers.size());
        runners.reserve(layers.size());
        for (const auto &layer : layers)
            runners.push_back(make_runner(layer));

        SC_THREAD(run_all);
    }

    void run_all()
    {
        for (size_t i = 0; i < runners.size(); ++i)
            results[i] = runners[i]->run();
        sc_stop();
    }
};

static bool parse_args(int argc, char *argv[], NafOptions &opts)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--intro-only")
        {
            opts.intro_only = true;
        }
        else if (arg == "--nafblock-only")
        {
            opts.nafblock_only = true;
        }
        else if (arg == "--block-c" || arg == "--block-h" || arg == "--block-w")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for " << arg << "\n";
                return false;
            }
            int value = std::stoi(argv[++i]);
            if (value <= 0)
            {
                std::cerr << "Invalid value for " << arg << ": " << value << "\n";
                return false;
            }
            if (arg == "--block-c") opts.block_c = value;
            if (arg == "--block-h") opts.block_h = value;
            if (arg == "--block-w") opts.block_w = value;
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout
                << "Usage: " << argv[0]
                << " [--intro-only] [--nafblock-only] [--block-c N] [--block-h N] [--block-w N]\n";
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
    NafOptions opts;
    if (!parse_args(argc, argv, opts))
        return (argc > 1) ? 1 : 0;

    if (opts.nafblock_only)
    {
        std::cout << "[Mode] nafblock-only: C=" << opts.block_c
                  << " H=" << opts.block_h
                  << " W=" << opts.block_w << "\n";
    }
    else if (opts.intro_only)
    {
        std::cout << "[Mode] intro-only: simulating only the intro CONV layer\n";
    }

    NafTop top("nafnet_top", opts);

    if (opts.nafblock_only)
    {
        std::string err;
        if (!validate_nafblock_manifest(top.layers, "nafblock",
                                        opts.block_c, opts.block_h, opts.block_w, &err))
        {
            std::cerr << "NafBlock manifest validation failed before simulation: "
                      << err << "\n";
            return 2;
        }
    }

    sc_start();

    std::cout << "\n--- Per-Layer Summary ---\n";
    std::cout << std::left
              << std::setw(22) << "Layer"
              << std::setw(12) << "Backend"
              << std::setw(12) << "MatReqs"
              << std::setw(12) << "VecReqs"
              << std::setw(12) << "MemReqs"
              << std::setw(14) << "RdBytes"
              << std::setw(14) << "WrBytes"
              << std::setw(8) << "Check"
              << "\n";
    std::cout << std::string(106, '-') << "\n";

    NafLayerStats totals;
    bool all_layers_ok = true;
    for (size_t i = 0; i < top.layers.size(); ++i)
    {
        const LayerDesc &layer = top.layers[i];
        const LayerRunResult &result = top.results[i];

        totals.mat_reqs += result.stats.mat_reqs;
        totals.vec_reqs += result.stats.vec_reqs;
        totals.mem_reqs += result.stats.mem_reqs;
        totals.accel_cycles += result.stats.accel_cycles;
        totals.cpu_cycles += result.stats.cpu_cycles;
        totals.wait_cycles += result.stats.wait_cycles;
        totals.mem_cycles += result.stats.mem_cycles;
        totals.rd_bytes += result.stats.rd_bytes;
        totals.wr_bytes += result.stats.wr_bytes;
        all_layers_ok &= result.verification_passed;

        std::cout << std::left
                  << std::setw(22) << layer.name
                  << std::setw(12) << layer_backend_str(layer.backend)
                  << std::setw(12) << result.stats.mat_reqs
                  << std::setw(12) << result.stats.vec_reqs
                  << std::setw(12) << result.stats.mem_reqs
                  << std::setw(14) << result.stats.rd_bytes
                  << std::setw(14) << result.stats.wr_bytes
                  << std::setw(8) << (result.verification_passed ? "PASS" : "FAIL")
                  << "\n";
    }

    std::cout << "\n--- Aggregate Summary ---\n";
    std::cout << "Layers                  : " << top.layers.size() << "\n";
    std::cout << "Total mat requests      : " << totals.mat_reqs << "\n";
    std::cout << "Total vec requests      : " << totals.vec_reqs << "\n";
    std::cout << "Total memory requests   : " << totals.mem_reqs << "\n";
    std::cout << "Total accelerator cycles: " << totals.accel_cycles << "\n";
    std::cout << "Total cpu cycles        : " << totals.cpu_cycles << "\n";
    std::cout << "Total wait cycles       : " << totals.wait_cycles << "\n";
    std::cout << "Total memory cycles     : " << totals.mem_cycles << "\n";
    std::cout << "Total read bytes        : " << totals.rd_bytes << "\n";
    std::cout << "Total write bytes       : " << totals.wr_bytes << "\n";

    bool manifest_ok = true;
    if (opts.nafblock_only)
    {
        std::string err;
        manifest_ok = validate_nafblock_manifest(top.layers, "nafblock",
                                                 opts.block_c, opts.block_h, opts.block_w, &err);
        std::cout << "\n--- NafBlock Verification ---\n";
        std::cout << "[" << (manifest_ok ? "PASS" : "FAIL") << "] manifest";
        if (!manifest_ok)
            std::cout << ": " << err;
        std::cout << "\n";
    }

    bool pass = manifest_ok && all_layers_ok;
    std::cout << "[" << (pass ? "PASS" : "FAIL")
              << "] delegated kernel backend verification\n";
    std::cout << (pass ? "\nAll checks passed.\n" : "\nSome checks FAILED.\n");
    return pass ? 0 : 2;
}
