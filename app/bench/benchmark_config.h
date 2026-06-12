#pragma once

#include <gipc/runtime_config.h>

#include <string>
#include <vector>

namespace app::bench
{
struct BenchmarkRunConfig
{
    std::string scene;
    gipc::BenchmarkBaseline baseline = gipc::BenchmarkBaseline::StiffGIPC_CEMAS_SRBK;
    int         frames               = 100;
    int         warmup               = 10;
    std::string settings_path;
    std::string output_root          = "Output/benchmarks";
    int         seed                 = 0;
    int         frame_start          = 0;
    int         frame_cap            = 0;
    bool        save_surface         = false;
    std::string task_id;
    std::string dataset;
    std::string asset_root;
    std::string output_tag;
    std::string notes;
    std::string manifest_path;
};

struct BenchmarkSuiteConfig
{
    std::string                 output_root   = "Output/benchmarks";
    std::string                 datasets_root = "datasets";
    std::string                 dataset_selector;
    std::string                 task_selector;
    bool                        auto_discover = false;
    BenchmarkRunConfig          defaults;
    std::vector<BenchmarkRunConfig> runs;
};

bool load_benchmark_suite(const std::string& path,
                          BenchmarkSuiteConfig& suite,
                          std::string& error_message);

bool load_manifest_run(const std::string& manifest_path,
                       BenchmarkRunConfig& run,
                       std::string& error_message);

std::string baseline_directory_name(gipc::BenchmarkBaseline baseline);
std::string baseline_display_name(gipc::BenchmarkBaseline baseline);
}  // namespace app::bench
