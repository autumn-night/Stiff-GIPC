#include "app/bench/benchmark_config.h"

#include <gipc/utils/json.h>

#include <fstream>
#include <stdexcept>

namespace app::bench
{
namespace
{
void apply_json_to_run(const gipc::Json& json, BenchmarkRunConfig& run)
{
    if(json.contains("scene"))
        run.scene = json["scene"].get<std::string>();
    if(json.contains("baseline"))
    {
        gipc::BenchmarkBaseline baseline;
        if(!gipc::try_parse_benchmark_baseline(json["baseline"].get<std::string>(), baseline))
            throw std::runtime_error("invalid baseline in benchmark suite");
        run.baseline = baseline;
    }
    if(json.contains("frames"))
        run.frames = json["frames"].get<int>();
    if(json.contains("warmup"))
        run.warmup = json["warmup"].get<int>();
    if(json.contains("settings"))
        run.settings_path = json["settings"].get<std::string>();
    if(json.contains("output_root"))
        run.output_root = json["output_root"].get<std::string>();
    if(json.contains("seed"))
        run.seed = json["seed"].get<int>();
    if(json.contains("frame_start"))
        run.frame_start = json["frame_start"].get<int>();
    if(json.contains("frame_cap"))
        run.frame_cap = json["frame_cap"].get<int>();
    if(json.contains("no_save_surface"))
        run.save_surface = !json["no_save_surface"].get<bool>();
    if(json.contains("task_id"))
        run.task_id = json["task_id"].get<std::string>();
    if(json.contains("dataset"))
        run.dataset = json["dataset"].get<std::string>();
    if(json.contains("asset_root"))
        run.asset_root = json["asset_root"].get<std::string>();
    if(json.contains("notes"))
        run.notes = json["notes"].get<std::string>();
}
}  // namespace

bool load_benchmark_suite(const std::string& path,
                          BenchmarkSuiteConfig& suite,
                          std::string& error_message)
{
    try
    {
        auto json = gipc::Json::parse(std::ifstream(path));
        if(json.contains("output_root"))
            suite.output_root = json["output_root"].get<std::string>();

        suite.defaults.output_root = suite.output_root;
        if(json.contains("defaults"))
            apply_json_to_run(json["defaults"], suite.defaults);

        suite.runs.clear();
        if(json.contains("runs"))
        {
            for(const auto& run_json : json["runs"])
            {
                auto run = suite.defaults;
                run.output_root = suite.output_root;
                apply_json_to_run(run_json, run);
                suite.runs.push_back(run);
            }
        }
        return true;
    }
    catch(const std::exception& e)
    {
        error_message = e.what();
        return false;
    }
}

std::string baseline_directory_name(gipc::BenchmarkBaseline baseline)
{
    return std::string(gipc::to_string(baseline));
}

std::string baseline_display_name(gipc::BenchmarkBaseline baseline)
{
    switch(baseline)
    {
        case gipc::BenchmarkBaseline::GIPC:
            return "GIPC";
        case gipc::BenchmarkBaseline::StiffGIPC_SRBK:
            return "StiffGIPC(SRBK)";
        case gipc::BenchmarkBaseline::StiffGIPC_CEMAS_SRBK:
        default:
            return "StiffGIPC(CEMAS+SRBK)";
    }
}
}  // namespace app::bench
