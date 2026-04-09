#pragma once

#include "common.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iosfwd>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace report
{

struct Field
{
    std::string label;
    std::string value;
};

struct Table
{
    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> rows;
};

struct AcceleratorSummaryRow
{
    std::string accelerator_type;
    std::string scope;
    std::string instances;
    std::string requests;
    std::string queue_wait_cycles;
    std::string busy_cycles;
    std::string occupied_cycles;
    std::string compute_utilization;
    std::string occupancy_utilization;
    std::string read_bytes;
    std::string write_bytes;
};

inline std::string na()
{
    return "N/A";
}

inline std::string fmt_u64(uint64_t value)
{
    return std::to_string(value);
}

inline std::string fmt_int(int value)
{
    return std::to_string(value);
}

inline std::string fmt_double(double value, int precision = 2)
{
    std::ostringstream os;
    os << std::fixed << std::setprecision(precision) << value;
    return os.str();
}

inline std::string fmt_percent(double value)
{
    return fmt_double(value) + " %";
}

inline std::string fmt_rate(double value, const std::string &unit)
{
    return fmt_double(value) + " " + unit;
}

inline void print_section_title(std::ostream &os, const std::string &title)
{
    os << "\n=== " << title << " ===\n";
}

inline void print_subtitle(std::ostream &os, const std::string &title)
{
    os << "\n" << title << "\n";
}

inline void print_fields(std::ostream &os, const std::vector<Field> &fields)
{
    size_t label_width = 0;
    for (const auto &field : fields)
        label_width = std::max(label_width, field.label.size());

    for (const auto &field : fields)
    {
        os << std::left << std::setw(static_cast<int>(label_width))
           << field.label << " : " << field.value << "\n";
    }
}

inline void print_table(std::ostream &os, const Table &table)
{
    if (table.headers.empty())
        return;

    if (table.rows.empty())
    {
        os << "(no rows)\n";
        return;
    }

    std::vector<size_t> widths(table.headers.size(), 0);
    for (size_t i = 0; i < table.headers.size(); ++i)
        widths[i] = table.headers[i].size();

    for (const auto &row : table.rows)
    {
        for (size_t i = 0; i < row.size() && i < widths.size(); ++i)
            widths[i] = std::max(widths[i], row[i].size());
    }

    for (size_t i = 0; i < table.headers.size(); ++i)
    {
        if (i > 0)
            os << "  ";
        os << std::left << std::setw(static_cast<int>(widths[i]))
           << table.headers[i];
    }
    os << "\n";

    size_t total_width = 0;
    for (size_t width : widths)
        total_width += width;
    total_width += (table.headers.size() - 1) * 2;
    os << std::string(total_width, '-') << "\n";

    for (const auto &row : table.rows)
    {
        for (size_t i = 0; i < table.headers.size(); ++i)
        {
            const std::string cell = (i < row.size()) ? row[i] : "";
            if (i > 0)
                os << "  ";
            os << std::left << std::setw(static_cast<int>(widths[i])) << cell;
        }
        os << "\n";
    }
}

inline Table make_worker_summary_table(const std::vector<KernelWorkerInfo> &workers)
{
    Table table;
    table.headers = {
        "Worker ID",
        "Matrix Requests [requests]",
        "Vector Requests [requests]",
        "Scalar Cycles [cycles]",
        "Stall Cycles [cycles]",
        "Memory Cycles [cycles]",
        "Elapsed Cycles [cycles]",
        "Read Bytes [bytes]",
        "Write Bytes [bytes]",
    };

    for (const auto &worker : workers)
    {
        table.rows.push_back({
            fmt_int(worker.tid),
            fmt_u64(worker.mat_reqs),
            fmt_u64(worker.vec_reqs),
            fmt_u64(worker.scalar_cycles),
            fmt_u64(worker.stall_cycles),
            fmt_u64(worker.mem_cycles),
            fmt_u64(worker.elapsed_cycles),
            fmt_u64(worker.rd_bytes),
            fmt_u64(worker.wr_bytes),
        });
    }

    return table;
}

inline Table make_accelerator_summary_table(
    const std::vector<AcceleratorSummaryRow> &rows)
{
    Table table;
    table.headers = {
        "Accelerator Type",
        "Summary Scope",
        "Instances [count]",
        "Requests [requests]",
        "Queue Wait Cycles [cycles]",
        "Busy Cycles [cycles]",
        "Occupied Cycles [cycles]",
        "Compute Utilization [%]",
        "Occupancy Utilization [%]",
        "Read Bytes [bytes]",
        "Write Bytes [bytes]",
    };

    for (const auto &row : rows)
    {
        table.rows.push_back({
            row.accelerator_type,
            row.scope,
            row.instances,
            row.requests,
            row.queue_wait_cycles,
            row.busy_cycles,
            row.occupied_cycles,
            row.compute_utilization,
            row.occupancy_utilization,
            row.read_bytes,
            row.write_bytes,
        });
    }

    return table;
}

}  // namespace report
