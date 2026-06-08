#include "app/bench/benchmark_report.h"

#include <cuda_runtime.h>

#include <cmath>
#include <fstream>
#include <numeric>

namespace app::bench
{
namespace
{
double average(const std::vector<double>& values)
{
    if(values.empty())
        return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double stddev(const std::vector<double>& values, double mean)
{
    if(values.empty())
        return 0.0;

    double accum = 0.0;
    for(double value : values)
    {
        double delta = value - mean;
        accum += delta * delta;
    }
    return std::sqrt(accum / values.size());
}
}  // namespace

BenchmarkSummaryRow summarize_frames(const BenchmarkRunConfig& run_config,
                                     const gipc::Json& frames_json)
{
    std::vector<double> hess_values;
    std::vector<double> lsolver_values;
    std::vector<double> lines_values;
    std::vector<double> misc_values;
    std::vector<double> total_values;
    std::vector<double> newton_values;
    std::vector<double> cg_values;
    std::vector<double> contact_values;
    std::vector<double> lsolver_subsystem_assemble_values;
    std::vector<double> lsolver_triplet_ops_values;
    std::vector<double> lsolver_preconditioner_assemble_values;
    std::vector<double> lsolver_pcg_values;
    std::vector<double> lsolver_solution_distribute_values;

    for(const auto& frame : frames_json)
    {
        hess_values.push_back(frame.value("time_hess_ms", 0.0));
        lsolver_values.push_back(frame.value("time_lsolver_ms", 0.0));
        lines_values.push_back(frame.value("time_lines_ms", 0.0));
        misc_values.push_back(frame.value("time_misc_ms", 0.0));
        total_values.push_back(frame.value("time_tot_ms", 0.0));
        newton_values.push_back(frame.value("newton_count", 0.0));
        cg_values.push_back(frame.value("cg_total", 0.0));
        contact_values.push_back(frame.value("contact_pairs_avg", 0.0));
        lsolver_subsystem_assemble_values.push_back(
            frame.value("time_lsolver_subsystem_assemble_ms", 0.0));
        lsolver_triplet_ops_values.push_back(frame.value("time_lsolver_triplet_ops_ms", 0.0));
        lsolver_preconditioner_assemble_values.push_back(
            frame.value("time_lsolver_preconditioner_assemble_ms", 0.0));
        lsolver_pcg_values.push_back(frame.value("time_lsolver_pcg_ms", 0.0));
        lsolver_solution_distribute_values.push_back(
            frame.value("time_lsolver_solution_distribute_ms", 0.0));
    }

    BenchmarkSummaryRow row;
    row.scene             = run_config.scene;
    row.baseline          = baseline_display_name(run_config.baseline);
    row.dataset           = run_config.dataset;
    row.task_id           = run_config.task_id;
    row.asset_root        = run_config.asset_root;
    row.notes             = run_config.notes;
    row.frames            = static_cast<int>(frames_json.size());
    row.warmup            = run_config.warmup;
    row.avg_hess_ms       = average(hess_values);
    row.avg_lsolver_ms    = average(lsolver_values);
    row.avg_lines_ms      = average(lines_values);
    row.avg_misc_ms       = average(misc_values);
    row.avg_time_tot_ms   = average(total_values);
    row.avg_newton        = average(newton_values);
    row.avg_cg            = average(cg_values);
    row.avg_contact_pairs = average(contact_values);
    row.std_time_tot_ms   = stddev(total_values, row.avg_time_tot_ms);
    row.avg_lsolver_subsystem_assemble_ms = average(lsolver_subsystem_assemble_values);
    row.avg_lsolver_triplet_ops_ms = average(lsolver_triplet_ops_values);
    row.avg_lsolver_preconditioner_assemble_ms =
        average(lsolver_preconditioner_assemble_values);
    row.avg_lsolver_pcg_ms = average(lsolver_pcg_values);
    row.avg_lsolver_solution_distribute_ms =
        average(lsolver_solution_distribute_values);
    return row;
}

gipc::Json make_raw_report(const BenchmarkRunConfig& run_config,
                           const gipc::Json& frames_json)
{
    gipc::Json report;
    report["scene"]      = run_config.scene;
    report["baseline"]   = baseline_display_name(run_config.baseline);
    report["frames"]     = frames_json;
    report["frame_start"] = run_config.frame_start;
    report["warmup"]     = run_config.warmup;
    report["requested_frames"] = run_config.frames;
    report["settings"]   = run_config.settings_path;
    report["task_id"]    = run_config.task_id;
    report["dataset"]    = run_config.dataset;
    report["asset_root"] = run_config.asset_root;
    report["notes"]      = run_config.notes;
    return report;
}

gipc::Json make_meta_report(const BenchmarkRunConfig& run_config,
                            const std::vector<std::string>& command_args)
{
    gipc::Json meta;
    meta["scene"]      = run_config.scene;
    meta["baseline"]   = baseline_display_name(run_config.baseline);
    meta["git_hash"]   = GIPC_GIT_HASH;
    meta["build_type"] = GIPC_BUILD_TYPE;
    meta["command_args"] = command_args;

    int runtime_version = 0;
    if(cudaRuntimeGetVersion(&runtime_version) == cudaSuccess)
        meta["cuda_runtime_version"] = runtime_version;

    cudaDeviceProp device_prop{};
    if(cudaGetDeviceProperties(&device_prop, 0) == cudaSuccess)
        meta["gpu_name"] = std::string(device_prop.name);

    return meta;
}

void write_raw_report(const std::string& path, const gipc::Json& raw_report)
{
    std::ofstream output(path);
    output << raw_report.dump(2);
}

void write_meta_report(const std::string& path, const gipc::Json& meta_report)
{
    std::ofstream output(path);
    output << meta_report.dump(2);
}

void write_summary_csv(const std::string& path,
                       const std::vector<BenchmarkSummaryRow>& rows)
{
    // Helper to quote CSV fields that may contain commas or quotes
    auto csv_field = [](const std::string& s) -> std::string {
        if(s.find(',') != std::string::npos || s.find('"') != std::string::npos
           || s.find('\n') != std::string::npos)
        {
            std::string escaped = s;
            // Escape double quotes by doubling
            for(size_t pos = escaped.find('"'); pos != std::string::npos;
                pos = escaped.find('"', pos + 2))
                escaped.insert(pos, 1, '"');
            return '"' + escaped + '"';
        }
        return s;
    };

    std::ofstream output(path);
    output << "dataset,task_id,scene,baseline,frames,warmup,avg_Hess_ms,avg_LSolver_ms,avg_LineS_ms,avg_Misc_ms,avg_TimeTot_ms,avg_newton,avg_cg,avg_contact_pairs,std_TimeTot_ms,asset_root,notes,avg_LSolver_SubsystemAssemble_ms,avg_LSolver_TripletOps_ms,avg_LSolver_PreconditionerAssemble_ms,avg_LSolver_PCG_ms,avg_LSolver_SolutionDistribute_ms\n";
    for(const auto& row : rows)
    {
        output << csv_field(row.dataset) << ','
               << csv_field(row.task_id) << ','
               << csv_field(row.scene) << ','
               << csv_field(row.baseline) << ','
               << row.frames << ',' << row.warmup
               << ',' << row.avg_hess_ms << ',' << row.avg_lsolver_ms << ','
               << row.avg_lines_ms << ',' << row.avg_misc_ms << ',' << row.avg_time_tot_ms
               << ',' << row.avg_newton << ',' << row.avg_cg << ','
               << row.avg_contact_pairs << ',' << row.std_time_tot_ms
               << ',' << csv_field(row.asset_root)
               << ',' << csv_field(row.notes)
               << ',' << row.avg_lsolver_subsystem_assemble_ms
               << ',' << row.avg_lsolver_triplet_ops_ms
               << ',' << row.avg_lsolver_preconditioner_assemble_ms
               << ',' << row.avg_lsolver_pcg_ms
               << ',' << row.avg_lsolver_solution_distribute_ms << '\n';
    }
}
}  // namespace app::bench
