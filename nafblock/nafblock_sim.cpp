// SystemC TLM-2.0 performance simulator for a single NafBlock.
//
// Drives the 14 sub-layers of one NafBlock sequentially:
//   MBConv  : LN -> 1x1 -> DW3x3 -> SimpleGate -> GAP -> 1x1 -> Scale -> 1x1 -> beta-residual
//   FFN     : LN -> 1x1 -> SimpleGate -> 1x1 -> gamma-residual
//
// Reuses src/ HW infrastructure and kernel/*_top.{h,cpp} unchanged. The
// 14-layer composition (append_nafblock_layers) is also the public entry
// point used to embed this block in a future NafNet simulator.

#include <systemc>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "hardware_config.h"
#include "report_formatter.h"
#include "perfetto_trace.h"

#include "../kernel/dw_conv2d/dw_conv2d_top.h"
#include "../kernel/layer_norm/layer_norm_top.h"
#include "../kernel/matmul/matmul_top.h"
#include "../kernel/pooling/pooling_top.h"
#include "../kernel/vec_ops/vec_ops_top.h"

#include "nafblock_config.h"
#include "nafblock_kernel_bridge.h"
#include "nafblock_layers.h"

using namespace sc_core;

// ============================================================
// Per-layer aggregation types
// ============================================================
struct ResourceStats
{
    bool     present = false;
    int      instances = 0;
    uint64_t requests = 0;
    uint64_t queue_wait_cycles = 0;
    uint64_t busy_cycles = 0;
    uint64_t occupied_cycles = 0;
};

struct LayerStats
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
    ResourceStats mat_pool;
    ResourceStats vec_pool;
    ResourceStats memory;
};

struct LayerRunResult
{
    LayerStats stats;
    std::vector<KernelWorkerInfo> worker_stats;
    bool verification_passed = false;
};

static void accumulate_worker(KernelWorkerInfo &dst, const KernelWorkerInfo &src)
{
    dst.tid = src.tid;
    dst.mat_reqs       += src.mat_reqs;
    dst.vec_reqs       += src.vec_reqs;
    dst.elapsed_cycles += src.elapsed_cycles;
    dst.scalar_cycles  += src.scalar_cycles;
    dst.stall_cycles   += src.stall_cycles;
    dst.mem_cycles     += src.mem_cycles;
    dst.rd_bytes       += src.rd_bytes;
    dst.wr_bytes       += src.wr_bytes;
}

static void finalize_layer(LayerRunResult &r)
{
    r.stats.scalar_cycles = 0;
    r.stats.stall_cycles  = 0;
    r.stats.mem_cycles    = 0;
    for (const auto &w : r.worker_stats)
    {
        r.stats.scalar_cycles += w.scalar_cycles;
        r.stats.stall_cycles  += w.stall_cycles;
        r.stats.mem_cycles    += w.mem_cycles;
    }
}

// ============================================================
// Kernel-stats → LayerStats conversion (one overload per kernel)
// ============================================================
static LayerStats from_kernel(const MatmulSimulationStats &s)
{
    LayerStats o;
    o.mat_reqs = s.mat_req_total;
    o.vec_reqs = s.vec_req_total;
    o.mem_reqs = s.memory_reqs;
    o.rd_bytes = s.total_rd_bytes;
    o.wr_bytes = s.total_wr_bytes;
    o.mat_pool.present         = true;
    o.mat_pool.requests        = s.mat_req_total;
    o.mat_pool.queue_wait_cycles = s.mat_qwait_total;
    o.mat_pool.busy_cycles     = s.mat_busy_total;
    o.mat_pool.occupied_cycles = s.mat_occupied_total;
    o.vec_pool.present         = true;
    o.vec_pool.requests        = s.vec_req_total;
    o.vec_pool.queue_wait_cycles = s.vec_qwait_total;
    o.vec_pool.busy_cycles     = s.vec_busy_total;
    o.vec_pool.occupied_cycles = s.vec_occupied_total;
    o.memory.present           = true;
    o.memory.requests          = s.memory_reqs;
    o.memory.queue_wait_cycles = s.memory_queue_wait_cycles;
    o.memory.busy_cycles       = s.memory_busy_cycles;
    return o;
}

static LayerStats from_kernel(const DwConvSimulationStats &s)
{
    LayerStats o;
    o.vec_reqs = s.total_vec_calls;
    o.mem_reqs = s.memory_reqs;
    o.rd_bytes = s.total_rd_bytes;
    o.wr_bytes = s.total_wr_bytes;
    o.vec_pool.present         = true;
    o.vec_pool.requests        = s.vec_acc_reqs;
    o.vec_pool.queue_wait_cycles = s.vec_acc_queue_wait_cycles;
    o.vec_pool.busy_cycles     = s.vec_acc_busy_cycles;
    o.vec_pool.occupied_cycles = s.vec_acc_occupied_cycles;
    o.memory.present           = true;
    o.memory.requests          = s.memory_reqs;
    o.memory.queue_wait_cycles = s.memory_queue_wait_cycles;
    o.memory.busy_cycles       = s.memory_busy_cycles;
    return o;
}

static LayerStats from_kernel(const LayerNormSimulationStats &s)
{
    LayerStats o;
    o.vec_reqs = s.total_vec_reqs;
    o.mem_reqs = s.memory_reqs;
    o.rd_bytes = s.total_rd_bytes;
    o.wr_bytes = s.total_wr_bytes;
    o.vec_pool.present         = true;
    o.vec_pool.requests        = s.vec_acc_reqs;
    o.vec_pool.queue_wait_cycles = s.vec_acc_queue_wait_cycles;
    o.vec_pool.busy_cycles     = s.vec_acc_busy_cycles;
    o.vec_pool.occupied_cycles = s.vec_acc_occupied_cycles;
    o.memory.present           = true;
    o.memory.requests          = s.memory_reqs;
    o.memory.queue_wait_cycles = s.memory_queue_wait_cycles;
    o.memory.busy_cycles       = s.memory_busy_cycles;
    return o;
}

static LayerStats from_kernel(const PoolSimulationStats &s)
{
    LayerStats o;
    o.vec_reqs = s.total_vec_calls;
    o.mem_reqs = s.l2_dma_reqs;
    o.rd_bytes = s.l2_dma_read_bytes;
    o.wr_bytes = s.l2_dma_write_bytes;
    o.vec_pool.present         = true;
    o.vec_pool.requests        = s.vec_acc_reqs;
    o.vec_pool.queue_wait_cycles = s.vec_acc_queue_wait_cycles;
    o.vec_pool.busy_cycles     = s.vec_acc_busy_cycles;
    o.vec_pool.occupied_cycles = s.vec_acc_occupied_cycles;
    o.memory.present           = true;
    o.memory.requests          = s.l2_dma_reqs;
    o.memory.queue_wait_cycles = s.l2_dma_queue_wait_cycles;
    o.memory.busy_cycles       = s.l2_dma_busy_cycles;
    return o;
}

static LayerStats from_kernel(const VecOpsSimulationStats &s)
{
    LayerStats o;
    o.vec_reqs = s.total_vec_calls;
    o.mem_reqs = s.memory_reqs;
    o.rd_bytes = s.total_rd_bytes;
    o.wr_bytes = s.total_wr_bytes;
    o.vec_pool.present         = true;
    o.vec_pool.requests        = s.vec_acc_reqs;
    o.vec_pool.queue_wait_cycles = s.vec_acc_queue_wait_cycles;
    o.vec_pool.busy_cycles     = s.vec_acc_busy_cycles;
    o.vec_pool.occupied_cycles = s.vec_acc_occupied_cycles;
    o.memory.present           = true;
    o.memory.requests          = s.memory_reqs;
    o.memory.queue_wait_cycles = s.memory_queue_wait_cycles;
    o.memory.busy_cycles       = s.memory_busy_cycles;
    return o;
}

// ============================================================
// LayerRunner polymorphic hierarchy
// ============================================================
struct LayerRunnerBase
{
    NafBlockLayerDesc layer;
    explicit LayerRunnerBase(const NafBlockLayerDesc &l) : layer(l) {}
    virtual ~LayerRunnerBase() = default;
    virtual LayerRunResult run() = 0;
};

struct MatmulRunner : LayerRunnerBase
{
    sc_event start_ev;
    sc_event done_ev;
    std::unique_ptr<MatmulTop> top;

    explicit MatmulRunner(const NafBlockLayerDesc &l, int dma_base_lat)
        : LayerRunnerBase(l)
    {
        top = std::make_unique<MatmulTop>(
            sc_gen_unique_name("nb_matmul_top"),
            nb_make_matmul_cfg(layer, nafblock_cfg::N_WORKERS, dma_base_lat),
            &start_ev,
            &done_ev);
    }

    LayerRunResult run() override
    {
        const sc_time t0 = sc_time_stamp();
        start_ev.notify(SC_ZERO_TIME);
        wait(done_ev);
        const auto s = top->collect_stats();
        LayerRunResult r;
        r.stats = from_kernel(s);
        r.stats.elapsed_cycles =
            static_cast<uint64_t>((sc_time_stamp() - t0) / CYCLE);
        r.stats.mat_pool.instances = top->cfg.mat_accel_count;
        r.stats.vec_pool.instances = top->cfg.vec_accel_count;
        r.stats.memory.instances   = 1;
        r.verification_passed      = s.verification_passed;
        r.worker_stats             = top->collect_worker_info();
        finalize_layer(r);
        return r;
    }
};

struct DwConvRunner : LayerRunnerBase
{
    sc_event start_ev;
    sc_event done_ev;
    std::unique_ptr<DwConvTop> top;

    explicit DwConvRunner(const NafBlockLayerDesc &l, int dma_base_lat)
        : LayerRunnerBase(l)
    {
        top = std::make_unique<DwConvTop>(
            sc_gen_unique_name("nb_dw_top"),
            nb_make_dwconv_cfg(layer, dma_base_lat),
            &start_ev,
            &done_ev);
    }

    LayerRunResult run() override
    {
        const sc_time t0 = sc_time_stamp();
        start_ev.notify(SC_ZERO_TIME);
        wait(done_ev);
        const auto s = top->collect_stats();
        LayerRunResult r;
        r.stats = from_kernel(s);
        r.stats.elapsed_cycles =
            static_cast<uint64_t>((sc_time_stamp() - t0) / CYCLE);
        r.stats.vec_pool.instances = top->cfg.vec_acc_instances;
        r.stats.memory.instances   = 1;
        r.verification_passed      = s.verification_passed;
        r.worker_stats             = top->collect_worker_info();
        finalize_layer(r);
        return r;
    }
};

struct LayerNormRunner : LayerRunnerBase
{
    sc_event start_ev;
    sc_event done_ev;
    std::unique_ptr<LayerNormTop> top;

    explicit LayerNormRunner(const NafBlockLayerDesc &l, int dma_base_lat)
        : LayerRunnerBase(l)
    {
        top = std::make_unique<LayerNormTop>(
            sc_gen_unique_name("nb_ln_top"),
            nb_make_layernorm_cfg(layer, dma_base_lat),
            &start_ev,
            &done_ev);
    }

    LayerRunResult run() override
    {
        const sc_time t0 = sc_time_stamp();
        start_ev.notify(SC_ZERO_TIME);
        wait(done_ev);
        const auto s = top->collect_stats();
        LayerRunResult r;
        r.stats = from_kernel(s);
        r.stats.elapsed_cycles =
            static_cast<uint64_t>((sc_time_stamp() - t0) / CYCLE);
        r.stats.vec_pool.instances = top->cfg.vec_acc_instances;
        r.stats.memory.instances   = 1;
        r.verification_passed      = s.verification_passed;
        r.worker_stats             = top->collect_worker_info();
        finalize_layer(r);
        return r;
    }
};

struct PoolRunner : LayerRunnerBase
{
    sc_event start_ev;
    sc_event done_ev;
    std::unique_ptr<PoolTop> top;

    explicit PoolRunner(const NafBlockLayerDesc &l, int dma_base_lat)
        : LayerRunnerBase(l)
    {
        top = std::make_unique<PoolTop>(
            sc_gen_unique_name("nb_pool_top"),
            nb_make_pool_cfg(layer, dma_base_lat),
            &start_ev,
            &done_ev);
    }

    LayerRunResult run() override
    {
        const sc_time t0 = sc_time_stamp();
        start_ev.notify(SC_ZERO_TIME);
        wait(done_ev);
        const auto s = top->collect_stats();
        LayerRunResult r;
        r.stats = from_kernel(s);
        r.stats.elapsed_cycles =
            static_cast<uint64_t>((sc_time_stamp() - t0) / CYCLE);
        r.stats.vec_pool.instances = top->cfg.vec_acc_instances;
        r.stats.memory.instances   = 1;
        r.verification_passed      = s.verification_passed;
        r.worker_stats             = top->collect_worker_info();
        finalize_layer(r);
        return r;
    }
};

// Multi-phase: e.g. residual = scale (phase 0) then add (phase 1).
struct VecOpsRunner : LayerRunnerBase
{
    struct Phase
    {
        VopType op = VOP_ELEMWISE_MUL;
        sc_event start_ev;
        sc_event done_ev;
        std::unique_ptr<VecOpsTop> top;
    };

    std::vector<std::unique_ptr<Phase>> phases;

    explicit VecOpsRunner(const NafBlockLayerDesc &l, int dma_base_lat)
        : LayerRunnerBase(l)
    {
        const auto phase_ops = nb_vecops_phases(layer);
        for (size_t i = 0; i < phase_ops.size(); ++i)
        {
            auto p = std::make_unique<Phase>();
            p->op = phase_ops[i];
            p->top = std::make_unique<VecOpsTop>(
                sc_gen_unique_name("nb_vec_top"),
                nb_make_vecops_cfg(layer, phase_ops[i],
                                   static_cast<int>(i), dma_base_lat),
                &p->start_ev,
                &p->done_ev);
            phases.push_back(std::move(p));
        }
    }

    LayerRunResult run() override
    {
        LayerRunResult r;
        r.verification_passed = true;
        const sc_time t0 = sc_time_stamp();

        for (auto &p : phases)
        {
            p->start_ev.notify(SC_ZERO_TIME);
            wait(p->done_ev);
            const auto s = p->top->collect_stats();
            const LayerStats cur = from_kernel(s);
            r.stats.vec_reqs              += cur.vec_reqs;
            r.stats.mem_reqs              += cur.mem_reqs;
            r.stats.rd_bytes              += cur.rd_bytes;
            r.stats.wr_bytes              += cur.wr_bytes;
            r.stats.vec_pool.present       = r.stats.vec_pool.present || cur.vec_pool.present;
            r.stats.vec_pool.requests     += cur.vec_pool.requests;
            r.stats.vec_pool.queue_wait_cycles += cur.vec_pool.queue_wait_cycles;
            r.stats.vec_pool.busy_cycles  += cur.vec_pool.busy_cycles;
            r.stats.vec_pool.occupied_cycles += cur.vec_pool.occupied_cycles;
            r.stats.vec_pool.instances     = p->top->cfg.vec_acc_instances;
            r.stats.memory.present         = r.stats.memory.present || cur.memory.present;
            r.stats.memory.requests       += cur.memory.requests;
            r.stats.memory.queue_wait_cycles += cur.memory.queue_wait_cycles;
            r.stats.memory.busy_cycles    += cur.memory.busy_cycles;
            r.stats.memory.instances       = 1;
            r.verification_passed         &= s.verification_passed;

            for (const auto &wi : p->top->collect_worker_info())
            {
                if (r.worker_stats.size() <= static_cast<size_t>(wi.tid))
                    r.worker_stats.resize(static_cast<size_t>(wi.tid) + 1);
                accumulate_worker(r.worker_stats[wi.tid], wi);
            }
        }

        r.stats.elapsed_cycles =
            static_cast<uint64_t>((sc_time_stamp() - t0) / CYCLE);
        finalize_layer(r);
        return r;
    }
};

static std::unique_ptr<LayerRunnerBase>
make_runner(const NafBlockLayerDesc &layer, int dma_base_lat)
{
    switch (layer.backend)
    {
    case NB_BACKEND_MATMUL:    return std::make_unique<MatmulRunner>(layer, dma_base_lat);
    case NB_BACKEND_DWCONV:    return std::make_unique<DwConvRunner>(layer, dma_base_lat);
    case NB_BACKEND_LAYERNORM: return std::make_unique<LayerNormRunner>(layer, dma_base_lat);
    case NB_BACKEND_POOLING:   return std::make_unique<PoolRunner>(layer, dma_base_lat);
    case NB_BACKEND_VECOPS:    return std::make_unique<VecOpsRunner>(layer, dma_base_lat);
    }
    return nullptr;
}

// ============================================================
// Top-level orchestration
// ============================================================
struct BlockOptions
{
    int block_c = nafblock_cfg::DEFAULT_C;
    int block_h = nafblock_cfg::DEFAULT_H;
    int block_w = nafblock_cfg::DEFAULT_W;
    // L2 / DMA base latency override (cycles). -1 keeps per-kernel defaults.
    int dma_base_lat = -1;
    // Perfetto trace output path (only consumed when built with PERFETTO_TRACE).
    std::string trace_out = "nafblock_trace.json";
};

struct NafBlockTop : sc_module
{
    BlockOptions opts;
    std::vector<NafBlockLayerDesc> layers;
    std::vector<std::unique_ptr<LayerRunnerBase>> runners;
    std::vector<LayerRunResult> results;

    SC_HAS_PROCESS(NafBlockTop);
    NafBlockTop(sc_module_name nm, const BlockOptions &o)
        : sc_module(nm), opts(o)
    {
        layers = build_nafblock_layers(opts.block_c, opts.block_h, opts.block_w);
        results.resize(layers.size());
        runners.reserve(layers.size());
        for (const auto &layer : layers)
            runners.push_back(make_runner(layer, opts.dma_base_lat));

        SC_THREAD(run_all);
    }

    void run_all()
    {
        for (size_t i = 0; i < runners.size(); ++i)
            results[i] = runners[i]->run();
        sc_stop();
    }
};

// ============================================================
// CLI parsing
// ============================================================
static bool parse_args(int argc, char *argv[], BlockOptions &opts)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--trace-out")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for " << arg << "\n";
                return false;
            }
            opts.trace_out = argv[++i];
            continue;
        }
        if (arg == "--block-c" || arg == "--block-h" || arg == "--block-w"
            || arg == "--dma-base-lat")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for " << arg << "\n";
                return false;
            }
            int v = std::stoi(argv[++i]);
            if (arg == "--dma-base-lat")
            {
                if (v < 0)
                {
                    std::cerr << "Invalid value for " << arg << ": " << v << "\n";
                    return false;
                }
                opts.dma_base_lat = v;
            }
            else
            {
                if (v <= 0)
                {
                    std::cerr << "Invalid value for " << arg << ": " << v << "\n";
                    return false;
                }
                if (arg == "--block-c") opts.block_c = v;
                if (arg == "--block-h") opts.block_h = v;
                if (arg == "--block-w") opts.block_w = v;
            }
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout
                << "Usage: " << argv[0]
                << " [--block-c N] [--block-h N] [--block-w N]"
                << " [--dma-base-lat N] [--trace-out FILE]\n"
                << "Defaults: C=" << nafblock_cfg::DEFAULT_C
                << ", H=" << nafblock_cfg::DEFAULT_H
                << ", W=" << nafblock_cfg::DEFAULT_W
                << ", dma_base_lat=<per-kernel default>\n";
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
    BlockOptions opts;
    if (!parse_args(argc, argv, opts))
        return (argc > 1) ? 1 : 0;

    NafBlockTop top("nafblock_top", opts);

    // Pre-sim manifest sanity check (canonical 14 entries, in order).
    std::string manifest_err;
    if (!validate_nafblock_manifest(top.layers, "nafblock", &manifest_err))
    {
        std::cerr << "NafBlock manifest validation failed: "
                  << manifest_err << "\n";
        return 2;
    }

#ifdef PERFETTO_TRACE
    // nafblock traces by default (single-run usage); recording must be on
    // before the worker/accelerator SC_THREADs run during sc_start().
    perf_trace_enabled() = true;
#endif

    sc_start();

#ifdef PERFETTO_TRACE
    perf_trace_write_json(opts.trace_out.c_str());
    std::cerr << "Perfetto trace written to " << opts.trace_out << "\n";
#endif

    const uint64_t total_cycles =
        static_cast<uint64_t>(sc_time_stamp() / CYCLE);

    // Aggregate per-layer results -> totals + per-worker.
    LayerStats totals;
    totals.mat_pool.instances = MAT_ACCEL_COUNT_CFG;
    totals.vec_pool.instances = VEC_ACCEL_COUNT_CFG;
    totals.memory.instances   = 1;
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

    std::vector<KernelWorkerInfo> global_worker(nafblock_cfg::N_WORKERS);
    for (int wid = 0; wid < nafblock_cfg::N_WORKERS; ++wid)
        global_worker[wid].tid = wid;

    auto accum_resource = [](ResourceStats &dst, const ResourceStats &src) {
        dst.present   = dst.present || src.present;
        dst.instances = std::max(dst.instances, src.instances);
        dst.requests          += src.requests;
        dst.queue_wait_cycles += src.queue_wait_cycles;
        dst.busy_cycles       += src.busy_cycles;
        dst.occupied_cycles   += src.occupied_cycles;
    };

    for (size_t i = 0; i < top.layers.size(); ++i)
    {
        const auto &layer = top.layers[i];
        const auto &res   = top.results[i];

        totals.mat_reqs       += res.stats.mat_reqs;
        totals.vec_reqs       += res.stats.vec_reqs;
        totals.mem_reqs       += res.stats.mem_reqs;
        totals.scalar_cycles  += res.stats.scalar_cycles;
        totals.stall_cycles   += res.stats.stall_cycles;
        totals.mem_cycles     += res.stats.mem_cycles;
        totals.rd_bytes       += res.stats.rd_bytes;
        totals.wr_bytes       += res.stats.wr_bytes;
        totals.elapsed_cycles += res.stats.elapsed_cycles;
        accum_resource(totals.mat_pool, res.stats.mat_pool);
        accum_resource(totals.vec_pool, res.stats.vec_pool);
        accum_resource(totals.memory,   res.stats.memory);
        all_layers_ok &= res.verification_passed;

        layer_summary.rows.push_back({
            layer.name,
            nb_backend_str(layer.backend),
            report::fmt_u64(res.stats.elapsed_cycles),
            report::fmt_u64(res.stats.mat_reqs),
            report::fmt_u64(res.stats.vec_reqs),
            report::fmt_u64(res.stats.mem_reqs),
            report::fmt_u64(res.stats.rd_bytes),
            report::fmt_u64(res.stats.wr_bytes),
            res.verification_passed ? "PASS" : "FAIL",
        });

        for (const auto &w : res.worker_stats)
        {
            if (w.tid >= 0 && w.tid < nafblock_cfg::N_WORKERS)
                accumulate_worker(global_worker[w.tid], w);
        }
    }

    const bool overall_pass = all_layers_ok;

    auto utilization = [total_cycles](uint64_t busy, int instances) {
        const double cap =
            static_cast<double>(total_cycles) * static_cast<double>(instances);
        return (cap > 0.0) ? static_cast<double>(busy) / cap * 100.0 : 0.0;
    };
    auto occupancy = [total_cycles](uint64_t occ, int instances) {
        const double cap =
            static_cast<double>(total_cycles) * static_cast<double>(instances);
        return (cap > 0.0) ? static_cast<double>(occ) / cap * 100.0 : 0.0;
    };

    // ---- Report ----
    report::print_section_title(std::cout, "Simulation Info");
    report::print_fields(std::cout, {
        {"Run Mode", "Single NafBlock"},
        {"Layer Count [count]", report::fmt_u64(top.layers.size())},
        {"NafBlock Tensor Shape",
            "[C=" + report::fmt_int(opts.block_c) +
            ", H=" + report::fmt_int(opts.block_h) +
            ", W=" + report::fmt_int(opts.block_w) + "]"},
    });

    report::print_section_title(std::cout, "Hardware Configuration");
    report::print_fields(std::cout, {
        {"Workers [count]",                  report::fmt_int(nafblock_cfg::N_WORKERS)},
        {"Matrix Accelerators [count]",      report::fmt_int(MAT_ACCEL_COUNT_CFG)},
        {"Vector Accelerators [count]",      report::fmt_int(VEC_ACCEL_COUNT_CFG)},
        {"Matrix Accelerator Tile",
            "[" + report::fmt_u64(MATMUL_M) + " x " +
            report::fmt_u64(MATMUL_K) + " x " +
            report::fmt_u64(MATMUL_N) + "]"},
        {"Vector Accelerator Capacity [elements/request]",
            report::fmt_u64(VECTOR_ACC_CAP)},
        {"Accelerator Queue Depth [requests]",
            report::fmt_u64(HW_ACC_QUEUE_DEPTH)},
        {"Memory Bandwidth [bytes/cycle]",
            "matrix=" + report::fmt_u64(HW_MATMUL_MEMORY_BYTES_PER_CYCLE) +
            ", depth-wise convolution=" + report::fmt_u64(HW_DW_MEMORY_BYTES_PER_CYCLE) +
            ", other kernels=" + report::fmt_u64(HW_MEMORY_BYTES_PER_CYCLE)},
        {"Memory Base Latency [cycles]",
            report::fmt_u64(HW_MEMORY_BASE_LAT)},
    });

    report::print_section_title(std::cout, "Worker Summary");
    for (size_t i = 0; i < top.layers.size(); ++i)
    {
        report::print_subtitle(
            std::cout,
            std::string("Per-layer Worker Summary: ") + top.layers[i].name +
            " (" + nb_backend_str(top.layers[i].backend) + ")");
        report::print_table(std::cout,
            report::make_worker_summary_table(top.results[i].worker_stats));
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
            report::na(), report::na(), report::na(),
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
            report::na(), report::na(), report::na(),
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
            report::na(),
        },
    }));

    report::print_section_title(std::cout, "Overall Summary");
    report::print_subtitle(std::cout, "Per-Layer Summary");
    report::print_table(std::cout, layer_summary);
    report::print_subtitle(std::cout, "Aggregate Summary");
    report::print_fields(std::cout, {
        {"Layer Count [count]", report::fmt_u64(top.layers.size())},
        {"Total Elapsed Cycles [cycles]", report::fmt_u64(total_cycles)},
        {"Total Matrix Accelerator Requests [requests]", report::fmt_u64(totals.mat_reqs)},
        {"Total Vector Accelerator Requests [requests]", report::fmt_u64(totals.vec_reqs)},
        {"Total Memory Requests [requests]", report::fmt_u64(totals.mem_reqs)},
        {"Total Read Bytes [bytes]",  report::fmt_u64(totals.rd_bytes)},
        {"Total Write Bytes [bytes]", report::fmt_u64(totals.wr_bytes)},
        {"Total Stall Cycles [cycles]",  report::fmt_u64(totals.stall_cycles)},
        {"Total Memory Cycles [cycles]", report::fmt_u64(totals.mem_cycles)},
        {"Total Scalar Cycles [cycles]", report::fmt_u64(totals.scalar_cycles)},
    });

    report::print_section_title(std::cout, "Verification");
    report::print_fields(std::cout, {
        {"Delegated Kernel Verification", all_layers_ok ? "PASS" : "FAIL"},
        {"NafBlock Manifest Verification", "PASS"},
        {"Overall Verification Status", overall_pass ? "PASS" : "FAIL"},
    });

    return overall_pass ? 0 : 2;
}
