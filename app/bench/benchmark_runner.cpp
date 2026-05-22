#include "app/bench/benchmark_runner.h"

#include "app/common/sim_bootstrap.h"

#include <GIPC.cuh>
#include <device_fem_data.cuh>
#include <gipc/statistics.h>
#include <gipc_path.h>
#include <load_mesh.h>

#include <algorithm>
#include <filesystem>

namespace app::bench
{
namespace
{
struct BenchmarkSimulationState
{
    GIPC             ipc;
    device_TetraData device_tet_mesh;
    tetrahedra_obj   tet_mesh;
    double           collision_detection_buff_scale = 1.0;
    double           motion_rate                    = 1.0;
    double           linear_system_buff_scale       = 1.0;
};

std::string run_output_directory(const BenchmarkRunConfig& run_config)
{
    std::filesystem::path output_root = run_config.output_root;
    return (output_root / run_config.scene / baseline_directory_name(run_config.baseline)).string();
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
    write_summary_csv(run_config.output_root + "/summary.csv", {row});
    return row;
}

std::vector<BenchmarkSummaryRow> BenchmarkRunner::run_suite(const BenchmarkSuiteConfig& suite_config)
{
    std::vector<BenchmarkSummaryRow> rows;
    rows.reserve(suite_config.runs.size());
    std::filesystem::create_directories(suite_config.output_root);
    for(auto run : suite_config.runs)
    {
        run.output_root = suite_config.output_root;
        rows.push_back(run_single(run));
    }
    write_summary_csv(suite_config.output_root + "/summary.csv", rows);
    return rows;
}
}  // namespace app::bench
