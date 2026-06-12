#include "app/bench/benchmark_runner.h"

#include "app/common/sim_bootstrap.h"

#include <GIPC.cuh>
#include <device_fem_data.cuh>
#include <gipc/statistics.h>
#include <gipc_path.h>
#include <load_mesh.h>

#include <algorithm>
#include <filesystem>
#include <iostream>

namespace app::bench
{
namespace
{
struct BenchmarkSimulationState
{
    tetrahedra_obj   tet_mesh;
    device_TetraData device_tet_mesh;
    GIPC             ipc;
    double           collision_detection_buff_scale = 1.0;
    double           motion_rate                    = 1.0;
    double           linear_system_buff_scale       = 1.0;
};

std::string run_output_directory(const BenchmarkRunConfig& run_config)
{
    std::filesystem::path output_root = run_config.output_root;
    std::filesystem::path output_dir;

    // When dataset + task_id are available, organize by dataset/task_id/baseline
    if(!run_config.dataset.empty() && !run_config.task_id.empty())
        output_dir = output_root / run_config.dataset / run_config.task_id
                     / baseline_directory_name(run_config.baseline);
    else
        output_dir = output_root / run_config.scene / baseline_directory_name(run_config.baseline);

    if(!run_config.output_tag.empty())
        output_dir /= run_config.output_tag;

    return output_dir.string();
}

gipc::Json measured_frames_json()
{
    auto& stats = gipc::Statistics::instance();
    if(stats.json().contains("frames"))
        return stats.json()["frames"];
    return gipc::Json::array();
}
}  // namespace

BenchmarkSummaryRow BenchmarkRunner::run_single(const BenchmarkRunConfig& run_config)
{
    BenchmarkSimulationState state;
    state.ipc.benchmark_mode              = true;
    state.ipc.verbose_output              = false;
    state.ipc.write_legacy_time_cost_file = false;
    state.ipc.write_statistics_file       = false;

    app::common::initialize_cuda();

    app::common::SimulationContext context{state.ipc,
                                           state.device_tet_mesh,
                                           state.tet_mesh,
                                           state.collision_detection_buff_scale,
                                           state.motion_rate,
                                           state.linear_system_buff_scale,
                                           std::string{gipc::assets_dir()},
                                           std::string{gipc::assets_dir()} + "sorted_mesh/"};

    app::common::SimulationBootstrapOptions options;
    options.scene_name             = run_config.scene;
    options.settings_path          = run_config.settings_path;
    options.manifest_path          = run_config.manifest_path;
    options.dataset                = run_config.dataset;
    options.task_id                = run_config.task_id;
    options.asset_root             = run_config.asset_root;
    options.runtime_backend_config = gipc::resolve_runtime_backend_config(run_config.baseline);
    app::common::bootstrap_simulation(context, options);

    gipc::Statistics::instance().reset();
    for(int i = 0; i < run_config.frame_start; ++i)
        state.ipc.IPC_Solver(state.device_tet_mesh);
    for(int i = 0; i < run_config.warmup; ++i)
        state.ipc.IPC_Solver(state.device_tet_mesh);

    gipc::Statistics::instance().reset();
    int measured_frames = run_config.frame_cap > 0
                              ? std::min(run_config.frames, run_config.frame_cap)
                              : run_config.frames;
    for(int i = 0; i < measured_frames; ++i)
        state.ipc.IPC_Solver(state.device_tet_mesh);

    auto frames = measured_frames_json();
    auto row    = summarize_frames(run_config, frames);

    auto output_dir = run_output_directory(run_config);
    std::filesystem::create_directories(output_dir);
    write_raw_report(output_dir + "/raw.json", make_raw_report(run_config, frames));
    write_meta_report(output_dir + "/meta.json", make_meta_report(run_config, m_command_args));
    return row;
}

std::vector<BenchmarkSummaryRow> BenchmarkRunner::run_suite(const BenchmarkSuiteConfig& suite_config)
{
    std::vector<BenchmarkSummaryRow> rows;
    rows.reserve(suite_config.runs.size());
    std::filesystem::create_directories(suite_config.output_root);
    std::cout << "Running benchmark suite with " << suite_config.runs.size()
              << " run(s)" << std::endl;
    for(size_t i = 0; i < suite_config.runs.size(); ++i)
    {
        auto run = suite_config.runs[i];
        run.output_root = suite_config.output_root;

        std::cout << "[" << (i + 1) << "/" << suite_config.runs.size() << "] "
                  << (run.dataset.empty() ? run.scene : run.dataset + "/" + run.task_id)
                  << " baseline=" << baseline_directory_name(run.baseline) << std::endl;
        rows.push_back(run_single(run));
        write_summary_csv(suite_config.output_root + "/summary.csv", rows);
    }
    std::cout << "Suite complete. Summary: " << suite_config.output_root + "/summary.csv"
              << std::endl;
    return rows;
}
}  // namespace app::bench
