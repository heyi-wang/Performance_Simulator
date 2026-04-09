#include <systemc>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "report_formatter.h"
#include "nafnet_hw_config.h"
#include "nafnet_kernel_bridge.h"
#include "nafnet_layers.h"

using namespace sc_core;

struct NafResourceStats
{
    bool present = false;
    int instances = 0;
    uint64_t requests = 0;
    uint64_t queue_wait_cycles = 0;
    uint64_t busy_cycles = 0;
    uint64_t occupied_cycles = 0;
};

struct NafLayerStats
{
    uint64_t mat_reqs = 0;
    uint64_t vec_reqs = 0;
    uint64_t mem_reqs = 0;
    uint64_t scalar_cycles = 0;
    uint64_t stall_cycles = 0;
    uint64_t mem_cycles = 0;
    uint64_t rd_bytes = 0;
    uint64_t wr_bytes = 0;
    uint64_t elapsed_cycles = 0;
    NafResourceStats mat_pool;
    NafResourceStats vec_pool;
    NafResourceStats memory;
};

struct LayerRunResult
{
    NafLayerStats stats;
    std::vector<KernelWorkerInfo> worker_stats;
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

static void accumulate_worker_info(KernelWorkerInfo &dst, const KernelWorkerInfo &src)
{
    dst.tid = src.tid;
    dst.mat_reqs += src.mat_reqs;
    dst.vec_reqs += src.vec_reqs;
    dst.elapsed_cycles += src.elapsed_cycles;
    dst.scalar_cycles += src.scalar_cycles;
    dst.stall_cycles += src.stall_cycles;
    dst.mem_cycles += src.mem_cycles;
    dst.rd_bytes += src.rd_bytes;
    dst.wr_bytes += src.wr_bytes;
}

static void finalize_layer_result(LayerRunResult &result)
{
    result.stats.scalar_cycles = 0;
    result.stats.stall_cycles = 0;
    result.stats.mem_cycles = 0;
    for (const auto &worker : result.worker_stats)
    {
        result.stats.scalar_cycles += worker.scalar_cycles;
        result.stats.stall_cycles += worker.stall_cycles;
        result.stats.mem_cycles += worker.mem_cycles;
    }
}

static NafLayerStats convert_stats(const MatmulSimulationStats &stats)
{
    NafLayerStats out;
    out.mat_reqs = stats.mat_req_total;
    out.vec_reqs = stats.vec_req_total;
    out.mem_reqs = stats.memory_reqs;
    out.rd_bytes = stats.total_rd_bytes;
    out.wr_bytes = stats.total_wr_bytes;
    out.mat_pool.present = true;
    out.mat_pool.requests = stats.mat_req_total;
    out.mat_pool.queue_wait_cycles = stats.mat_qwait_total;
    out.mat_pool.busy_cycles = stats.mat_busy_total;
    out.mat_pool.occupied_cycles = stats.mat_occupied_total;
    out.vec_pool.present = true;
    out.vec_pool.requests = stats.vec_req_total;
    out.vec_pool.queue_wait_cycles = stats.vec_qwait_total;
    out.vec_pool.busy_cycles = stats.vec_busy_total;
    out.vec_pool.occupied_cycles = stats.vec_occupied_total;
    out.memory.present = true;
    out.memory.requests = stats.memory_reqs;
    out.memory.queue_wait_cycles = stats.memory_queue_wait_cycles;
    out.memory.busy_cycles = stats.memory_busy_cycles;
    return out;
}

static NafLayerStats convert_stats(const DwConvSimulationStats &stats)
{
    NafLayerStats out;
    out.vec_reqs = stats.total_vec_calls;
    out.mem_reqs = stats.memory_reqs;
    out.rd_bytes = stats.total_rd_bytes;
    out.wr_bytes = stats.total_wr_bytes;
    out.vec_pool.present = true;
    out.vec_pool.requests = stats.vec_acc_reqs;
    out.vec_pool.queue_wait_cycles = stats.vec_acc_queue_wait_cycles;
    out.vec_pool.busy_cycles = stats.vec_acc_busy_cycles;
    out.vec_pool.occupied_cycles = stats.vec_acc_occupied_cycles;
    out.memory.present = true;
    out.memory.requests = stats.memory_reqs;
    out.memory.queue_wait_cycles = stats.memory_queue_wait_cycles;
    out.memory.busy_cycles = stats.memory_busy_cycles;
    return out;
}

static NafLayerStats convert_stats(const LayerNormSimulationStats &stats)
{
    NafLayerStats out;
    out.vec_reqs = stats.total_vec_reqs;
    out.mem_reqs = stats.memory_reqs;
    out.rd_bytes = stats.total_rd_bytes;
    out.wr_bytes = stats.total_wr_bytes;
    out.vec_pool.present = true;
    out.vec_pool.requests = stats.vec_acc_reqs;
    out.vec_pool.queue_wait_cycles = stats.vec_acc_queue_wait_cycles;
    out.vec_pool.busy_cycles = stats.vec_acc_busy_cycles;
    out.vec_pool.occupied_cycles = stats.vec_acc_occupied_cycles;
    out.memory.present = true;
    out.memory.requests = stats.memory_reqs;
    out.memory.queue_wait_cycles = stats.memory_queue_wait_cycles;
    out.memory.busy_cycles = stats.memory_busy_cycles;
    return out;
}

static NafLayerStats convert_stats(const PoolSimulationStats &stats)
{
    NafLayerStats out;
    out.vec_reqs = stats.total_vec_calls;
    out.mem_reqs = stats.memory_reqs;
    out.rd_bytes = stats.total_rd_bytes;
    out.wr_bytes = stats.total_wr_bytes;
    out.vec_pool.present = true;
    out.vec_pool.requests = stats.vec_acc_reqs;
    out.vec_pool.queue_wait_cycles = stats.vec_acc_queue_wait_cycles;
    out.vec_pool.busy_cycles = stats.vec_acc_busy_cycles;
    out.vec_pool.occupied_cycles = stats.vec_acc_occupied_cycles;
    out.memory.present = true;
    out.memory.requests = stats.memory_reqs;
    out.memory.queue_wait_cycles = stats.memory_queue_wait_cycles;
    out.memory.busy_cycles = stats.memory_busy_cycles;
    return out;
}

static NafLayerStats convert_stats(const VecOpsSimulationStats &stats)
{
    NafLayerStats out;
    out.vec_reqs = stats.total_vec_calls;
    out.mem_reqs = stats.memory_reqs;
    out.rd_bytes = stats.total_rd_bytes;
    out.wr_bytes = stats.total_wr_bytes;
    out.vec_pool.present = true;
    out.vec_pool.requests = stats.vec_acc_reqs;
    out.vec_pool.queue_wait_cycles = stats.vec_acc_queue_wait_cycles;
    out.vec_pool.busy_cycles = stats.vec_acc_busy_cycles;
    out.vec_pool.occupied_cycles = stats.vec_acc_occupied_cycles;
    out.memory.present = true;
    out.memory.requests = stats.memory_reqs;
    out.memory.queue_wait_cycles = stats.memory_queue_wait_cycles;
    out.memory.busy_cycles = stats.memory_busy_cycles;
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
        const sc_time layer_start = sc_time_stamp();
        start_ev.notify(SC_ZERO_TIME);
        wait(done_ev);
        const MatmulSimulationStats stats = top->collect_stats();
        LayerRunResult result;
        result.stats = convert_stats(stats);
        result.stats.elapsed_cycles =
            static_cast<uint64_t>((sc_time_stamp() - layer_start) / CYCLE);
        result.stats.mat_pool.instances = top->cfg.mat_accel_count;
        result.stats.vec_pool.instances = top->cfg.vec_accel_count;
        result.stats.memory.instances = 1;
        result.verification_passed = stats.verification_passed;
        result.worker_stats = top->collect_worker_info();
        finalize_layer_result(result);
        return result;
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
        const sc_time layer_start = sc_time_stamp();
        start_ev.notify(SC_ZERO_TIME);
        wait(done_ev);
        const DwConvSimulationStats stats = top->collect_stats();
        LayerRunResult result;
        result.stats = convert_stats(stats);
        result.stats.elapsed_cycles =
            static_cast<uint64_t>((sc_time_stamp() - layer_start) / CYCLE);
        result.stats.vec_pool.instances = top->cfg.vec_acc_instances;
        result.stats.memory.instances = 1;
        result.verification_passed = stats.verification_passed;
        result.worker_stats = top->collect_worker_info();
        finalize_layer_result(result);
        return result;
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
        const sc_time layer_start = sc_time_stamp();
        start_ev.notify(SC_ZERO_TIME);
        wait(done_ev);
        const LayerNormSimulationStats stats = top->collect_stats();
        LayerRunResult result;
        result.stats = convert_stats(stats);
        result.stats.elapsed_cycles =
            static_cast<uint64_t>((sc_time_stamp() - layer_start) / CYCLE);
        result.stats.vec_pool.instances = top->cfg.vec_acc_instances;
        result.stats.memory.instances = 1;
        result.verification_passed = stats.verification_passed;
        result.worker_stats = top->collect_worker_info();
        finalize_layer_result(result);
        return result;
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
        const sc_time layer_start = sc_time_stamp();
        start_ev.notify(SC_ZERO_TIME);
        wait(done_ev);
        const PoolSimulationStats stats = top->collect_stats();
        LayerRunResult result;
        result.stats = convert_stats(stats);
        result.stats.elapsed_cycles =
            static_cast<uint64_t>((sc_time_stamp() - layer_start) / CYCLE);
        result.stats.vec_pool.instances = top->cfg.vec_acc_instances;
        result.stats.memory.instances = 1;
        result.verification_passed = stats.verification_passed;
        result.worker_stats = top->collect_worker_info();
        finalize_layer_result(result);
        return result;
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
        const sc_time layer_start = sc_time_stamp();

        for (auto &phase : phases)
        {
            phase->start_ev.notify(SC_ZERO_TIME);
            wait(phase->done_ev);
            const VecOpsSimulationStats stats = phase->top->collect_stats();
            NafLayerStats converted = convert_stats(stats);
            result.stats.vec_reqs += converted.vec_reqs;
            result.stats.mem_reqs += converted.mem_reqs;
            result.stats.rd_bytes += converted.rd_bytes;
            result.stats.wr_bytes += converted.wr_bytes;
            result.stats.vec_pool.present = result.stats.vec_pool.present || converted.vec_pool.present;
            result.stats.vec_pool.requests += converted.vec_pool.requests;
            result.stats.vec_pool.queue_wait_cycles += converted.vec_pool.queue_wait_cycles;
            result.stats.vec_pool.busy_cycles += converted.vec_pool.busy_cycles;
            result.stats.vec_pool.occupied_cycles += converted.vec_pool.occupied_cycles;
            result.stats.vec_pool.instances = phase->top->cfg.vec_acc_instances;
            result.stats.memory.present = result.stats.memory.present || converted.memory.present;
            result.stats.memory.requests += converted.memory.requests;
            result.stats.memory.queue_wait_cycles += converted.memory.queue_wait_cycles;
            result.stats.memory.busy_cycles += converted.memory.busy_cycles;
            result.stats.memory.instances = 1;
            result.verification_passed &= stats.verification_passed;

            for (const auto &wi : phase->top->collect_worker_info())
            {
                if (result.worker_stats.size() <= static_cast<size_t>(wi.tid))
                    result.worker_stats.resize(static_cast<size_t>(wi.tid) + 1);
                accumulate_worker_info(result.worker_stats[wi.tid], wi);
            }
        }

        result.stats.elapsed_cycles =
            static_cast<uint64_t>((sc_time_stamp() - layer_start) / CYCLE);
        finalize_layer_result(result);
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
    const uint64_t total_elapsed_cycles =
        static_cast<uint64_t>(sc_time_stamp() / CYCLE);
    NafLayerStats totals;
    totals.mat_pool.instances = MAT_ACCEL_COUNT_CFG;
    totals.vec_pool.instances = VEC_ACCEL_COUNT_CFG;
    totals.memory.instances = 1;
    bool all_layers_ok = true;
    report::Table layer_summary;
    layer_summary.headers = {
        "Layer",
        "Backend",
        "Elapsed Cycles [cycles]",
        "Matrix Requests [requests]",
        "Vector Requests [requests]",
        "Memory Requests [requests]",
        "Read Bytes [bytes]",
        "Write Bytes [bytes]",
        "Verification",
    };

    std::vector<KernelWorkerInfo> global_worker(N_WORKERS);
    for (int worker_id = 0; worker_id < N_WORKERS; ++worker_id)
        global_worker[worker_id].tid = worker_id;

    auto accumulate_resource = [](NafResourceStats &dst, const NafResourceStats &src) {
        dst.present = dst.present || src.present;
        dst.instances = std::max(dst.instances, src.instances);
        dst.requests += src.requests;
        dst.queue_wait_cycles += src.queue_wait_cycles;
        dst.busy_cycles += src.busy_cycles;
        dst.occupied_cycles += src.occupied_cycles;
    };

    for (size_t i = 0; i < top.layers.size(); ++i)
    {
        const LayerDesc &layer = top.layers[i];
        const LayerRunResult &result = top.results[i];

        totals.mat_reqs += result.stats.mat_reqs;
        totals.vec_reqs += result.stats.vec_reqs;
        totals.mem_reqs += result.stats.mem_reqs;
        totals.scalar_cycles += result.stats.scalar_cycles;
        totals.stall_cycles += result.stats.stall_cycles;
        totals.mem_cycles += result.stats.mem_cycles;
        totals.rd_bytes += result.stats.rd_bytes;
        totals.wr_bytes += result.stats.wr_bytes;
        totals.elapsed_cycles += result.stats.elapsed_cycles;
        accumulate_resource(totals.mat_pool, result.stats.mat_pool);
        accumulate_resource(totals.vec_pool, result.stats.vec_pool);
        accumulate_resource(totals.memory, result.stats.memory);
        all_layers_ok &= result.verification_passed;

        layer_summary.rows.push_back({
            layer.name,
            layer_backend_str(layer.backend),
            report::fmt_u64(result.stats.elapsed_cycles),
            report::fmt_u64(result.stats.mat_reqs),
            report::fmt_u64(result.stats.vec_reqs),
            report::fmt_u64(result.stats.mem_reqs),
            report::fmt_u64(result.stats.rd_bytes),
            report::fmt_u64(result.stats.wr_bytes),
            result.verification_passed ? "PASS" : "FAIL",
        });

        for (const auto &worker : result.worker_stats)
        {
            if (worker.tid >= 0 && worker.tid < N_WORKERS)
                accumulate_worker_info(global_worker[worker.tid], worker);
        }
    }

    bool manifest_ok = true;
    std::string manifest_err;
    if (opts.nafblock_only)
    {
        manifest_ok = validate_nafblock_manifest(top.layers, "nafblock",
                                                 opts.block_c, opts.block_h, opts.block_w,
                                                 &manifest_err);
    }

    bool pass = manifest_ok && all_layers_ok;

    auto utilization = [total_elapsed_cycles](uint64_t busy_cycles, int instances) {
        const double capacity =
            static_cast<double>(total_elapsed_cycles) * static_cast<double>(instances);
        return (capacity > 0.0)
            ? static_cast<double>(busy_cycles) / capacity * 100.0
            : 0.0;
    };
    auto occupancy = [total_elapsed_cycles](uint64_t occupied_cycles, int instances) {
        const double capacity =
            static_cast<double>(total_elapsed_cycles) * static_cast<double>(instances);
        return (capacity > 0.0)
            ? static_cast<double>(occupied_cycles) / capacity * 100.0
            : 0.0;
    };

    report::print_section_title(std::cout, "Simulation Info");
    report::print_fields(std::cout, {
        {"Run Mode", opts.nafblock_only ? "NafBlock only" :
                      (opts.intro_only ? "Intro layer only" : "Full NafNet pipeline")},
        {"Layer Count [count]", report::fmt_u64(top.layers.size())},
        {"NafBlock Tensor Shape", opts.nafblock_only
            ? ("[C=" + report::fmt_int(opts.block_c) +
               ", H=" + report::fmt_int(opts.block_h) +
               ", W=" + report::fmt_int(opts.block_w) + "]")
            : report::na()},
    });

    report::print_section_title(std::cout, "Hardware Configuration");
    report::print_fields(std::cout, {
        {"Workers [count]", report::fmt_int(N_WORKERS)},
        {"Matrix Accelerators [count]", report::fmt_int(MAT_ACCEL_COUNT_CFG)},
        {"Vector Accelerators [count]", report::fmt_int(VEC_ACCEL_COUNT_CFG)},
        {"Matrix Accelerator Tile", "[" + report::fmt_u64(MATMUL_M) + " x " +
                                     report::fmt_u64(MATMUL_K) + " x " +
                                     report::fmt_u64(MATMUL_N) + "]"},
        {"Vector Accelerator Capacity [elements/request]", report::fmt_u64(VECTOR_ACC_CAP)},
        {"Accelerator Queue Depth [requests]", report::fmt_u64(HW_ACC_QUEUE_DEPTH)},
        {"Memory Bandwidth [bytes/cycle]",
         "matrix=" + report::fmt_u64(HW_MATMUL_MEMORY_BYTES_PER_CYCLE) +
         ", depth-wise convolution=" + report::fmt_u64(HW_DW_MEMORY_BYTES_PER_CYCLE) +
         ", other kernels=" + report::fmt_u64(HW_MEMORY_BYTES_PER_CYCLE)},
        {"Memory Base Latency [cycles]", report::fmt_u64(HW_MEMORY_BASE_LAT)},
    });

    report::print_section_title(std::cout, "Worker Summary");
    for (size_t i = 0; i < top.layers.size(); ++i)
    {
        const LayerDesc &layer = top.layers[i];
        const LayerRunResult &result = top.results[i];
        report::print_subtitle(
            std::cout,
            std::string("Per-layer Worker Summary: ") + layer.name +
            " (" + layer_backend_str(layer.backend) + ")");
        report::print_table(std::cout, report::make_worker_summary_table(result.worker_stats));
    }
    report::print_subtitle(std::cout, "Aggregate Per-Worker Summary Across All Layers");
    report::print_table(std::cout, report::make_worker_summary_table(global_worker));

    report::print_section_title(std::cout, "Accelerator Summary");
    report::print_table(std::cout, report::make_accelerator_summary_table({
        {
            "Matrix Accelerator",
            "pool-level",
            totals.mat_pool.present ? report::fmt_int(totals.mat_pool.instances) : report::na(),
            totals.mat_pool.present ? report::fmt_u64(totals.mat_pool.requests) : report::na(),
            totals.mat_pool.present ? report::fmt_u64(totals.mat_pool.queue_wait_cycles) : report::na(),
            totals.mat_pool.present ? report::fmt_u64(totals.mat_pool.busy_cycles) : report::na(),
            totals.mat_pool.present ? report::fmt_u64(totals.mat_pool.occupied_cycles) : report::na(),
            totals.mat_pool.present ? report::fmt_percent(
                utilization(totals.mat_pool.busy_cycles, totals.mat_pool.instances)) : report::na(),
            totals.mat_pool.present ? report::fmt_percent(
                occupancy(totals.mat_pool.occupied_cycles, totals.mat_pool.instances)) : report::na(),
            report::na(),
            report::na(),
        },
        {
            "Vector Accelerator",
            "pool-level",
            totals.vec_pool.present ? report::fmt_int(totals.vec_pool.instances) : report::na(),
            totals.vec_pool.present ? report::fmt_u64(totals.vec_pool.requests) : report::na(),
            totals.vec_pool.present ? report::fmt_u64(totals.vec_pool.queue_wait_cycles) : report::na(),
            totals.vec_pool.present ? report::fmt_u64(totals.vec_pool.busy_cycles) : report::na(),
            totals.vec_pool.present ? report::fmt_u64(totals.vec_pool.occupied_cycles) : report::na(),
            totals.vec_pool.present ? report::fmt_percent(
                utilization(totals.vec_pool.busy_cycles, totals.vec_pool.instances)) : report::na(),
            totals.vec_pool.present ? report::fmt_percent(
                occupancy(totals.vec_pool.occupied_cycles, totals.vec_pool.instances)) : report::na(),
            report::na(),
            report::na(),
        },
        {
            "Memory",
            "shared resource",
            "1",
            report::fmt_u64(totals.memory.requests),
            report::fmt_u64(totals.memory.queue_wait_cycles),
            report::fmt_u64(totals.memory.busy_cycles),
            report::na(),
            report::na(),
            report::na(),
            report::fmt_u64(totals.rd_bytes),
            report::fmt_u64(totals.wr_bytes),
        },
    }));

    report::print_section_title(std::cout, "Overall Summary");
    report::print_subtitle(std::cout, "Per-Layer Summary");
    report::print_table(std::cout, layer_summary);
    report::print_subtitle(std::cout, "Aggregate Summary");
    report::print_fields(std::cout, {
        {"Layer Count [count]", report::fmt_u64(top.layers.size())},
        {"Total Elapsed Cycles [cycles]", report::fmt_u64(total_elapsed_cycles)},
        {"Total Matrix Accelerator Requests [requests]", report::fmt_u64(totals.mat_reqs)},
        {"Total Vector Accelerator Requests [requests]", report::fmt_u64(totals.vec_reqs)},
        {"Total Memory Requests [requests]", report::fmt_u64(totals.mem_reqs)},
        {"Total Read Bytes [bytes]", report::fmt_u64(totals.rd_bytes)},
        {"Total Write Bytes [bytes]", report::fmt_u64(totals.wr_bytes)},
        {"Total Stall Cycles [cycles]", report::fmt_u64(totals.stall_cycles)},
        {"Total Memory Cycles [cycles]", report::fmt_u64(totals.mem_cycles)},
        {"Total Scalar Cycles [cycles]", report::fmt_u64(totals.scalar_cycles)},
    });

    report::print_section_title(std::cout, "Verification");
    report::print_fields(std::cout, {
        {"Delegated Kernel Verification", all_layers_ok ? "PASS" : "FAIL"},
        {"NafBlock Manifest Verification", opts.nafblock_only
            ? (manifest_ok ? "PASS" : ("FAIL: " + manifest_err))
            : report::na()},
        {"Overall Verification Status", pass ? "PASS" : "FAIL"},
    });

    return pass ? 0 : 2;
}
