#include "app/bench/benchmark_config.h"

#include <gipc/utils/json.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace app::bench
{
namespace
{
std::string normalize_path_string(const std::filesystem::path& path)
{
    return path.lexically_normal().string();
}

std::string resolve_suite_relative_path(const std::filesystem::path& suite_dir,
                                       const std::string&           value)
{
    if(value.empty())
        return value;

    const std::filesystem::path path_value{value};
    if(path_value.is_absolute())
        return normalize_path_string(path_value);

    const auto first_component = path_value.begin();
    if(first_component != path_value.end())
    {
        const auto& root_relative_hint = first_component->string();
        if(root_relative_hint == "Assets" || root_relative_hint == "datasets"
           || root_relative_hint == "Output")
        {
            return normalize_path_string(std::filesystem::current_path() / path_value);
        }
    }

    return normalize_path_string(suite_dir / path_value);
}

bool matches_selector(const std::string& value, const std::string& selector)
{
    return selector.empty() || value == selector;
}

bool is_supported_ipc_sim_example(const std::filesystem::path& example_path)
{
    try
    {
        std::ifstream file(example_path);
        if(!file.is_open())
            return false;

        std::string line;
        int         remaining_shape_lines = 0;
        bool        in_section            = false;
        while(std::getline(file, line))
        {
            auto comment_pos = line.find('#');
            if(comment_pos != std::string::npos)
                line.erase(comment_pos);

            std::istringstream stream(line);
            std::vector<std::string> tokens;
            for(std::string token; stream >> token;)
                tokens.push_back(token);
            if(tokens.empty())
                continue;

            if(in_section)
            {
                if(tokens[0] == "section" && tokens.size() >= 2 && tokens[1] == "end")
                    in_section = false;
                continue;
            }

            if(remaining_shape_lines > 0)
            {
                --remaining_shape_lines;
                if(tokens.size() != 10)
                    return false;

                auto extension = std::filesystem::path(tokens[0]).extension().string();
                if(extension != ".msh" && extension != ".obj")
                    return false;
                continue;
            }

            if(tokens[0] == "section")
            {
                in_section = true;
                continue;
            }
            if(tokens[0] == "shapes")
            {
                if(tokens.size() != 3 || tokens[1] != "input")
                    return false;
                remaining_shape_lines = std::stoi(tokens[2]);
                continue;
            }
            if(tokens[0] == "meshCO")
                return false;
        }

        return remaining_shape_lines == 0;
    }
    catch(const std::exception&)
    {
        return false;
    }
}

void discover_local_dataset_runs(const BenchmarkSuiteConfig&       suite,
                                 const std::filesystem::path&      dataset_path,
                                 const std::string&                dataset_name,
                                 std::vector<BenchmarkRunConfig>& runs)
{
    namespace fs = std::filesystem;

    if(dataset_name == "stanford")
    {
        for(const auto& mesh_entry : fs::directory_iterator(dataset_path))
        {
            if(!mesh_entry.is_regular_file() || mesh_entry.path().extension() != ".obj")
                continue;

            const std::string task_id = mesh_entry.path().stem().string();
            if(!matches_selector(task_id, suite.task_selector))
                continue;

            BenchmarkRunConfig run = suite.defaults;
            run.dataset            = dataset_name;
            run.task_id            = task_id;
            run.scene              = "local_dataset";
            run.asset_root         = fs::absolute(dataset_path).lexically_normal().string();
            runs.push_back(std::move(run));
        }
        return;
    }

    if(dataset_name == "ipc-sim")
    {
        for(const auto& entry : fs::recursive_directory_iterator(dataset_path))
        {
            if(!entry.is_regular_file() || entry.path().extension() != ".txt")
                continue;
            if(entry.path().string().find("/input/") != std::string::npos)
                continue;
            if(!is_supported_ipc_sim_example(entry.path()))
                continue;

            auto relative_path = entry.path().lexically_relative(dataset_path);
            relative_path.replace_extension("");
            const std::string task_id = relative_path.generic_string();
            if(!matches_selector(task_id, suite.task_selector))
                continue;

            BenchmarkRunConfig run = suite.defaults;
            run.dataset            = dataset_name;
            run.task_id            = task_id;
            run.scene              = "local_dataset";
            run.asset_root         = fs::absolute(dataset_path).lexically_normal().string();
            runs.push_back(std::move(run));
        }
    }
}

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
    if(json.contains("manifest"))
        run.manifest_path = json["manifest"].get<std::string>();
}

void discover_manifest_runs(const BenchmarkSuiteConfig& suite,
                            std::vector<BenchmarkRunConfig>& runs)
{
    namespace fs = std::filesystem;
    fs::path datasets_path = suite.datasets_root;

    if(!fs::exists(datasets_path) || !fs::is_directory(datasets_path))
        return;

    for(const auto& dataset_entry : fs::directory_iterator(datasets_path))
    {
        if(!dataset_entry.is_directory())
            continue;
        std::string dataset_name = dataset_entry.path().filename().string();

        // Skip if dataset_selector is set and doesn't match
        if(!matches_selector(dataset_name, suite.dataset_selector))
            continue;

        if(dataset_name == "stanford" || dataset_name == "ipc-sim")
        {
            discover_local_dataset_runs(suite, dataset_entry.path(), dataset_name, runs);
            continue;
        }

        for(const auto& task_entry : fs::directory_iterator(dataset_entry.path()))
        {
            if(!task_entry.is_directory())
                continue;
            std::string task_name = task_entry.path().filename().string();

            // Skip if task_selector is set and doesn't match
            if(!matches_selector(task_name, suite.task_selector))
                continue;

            fs::path manifest_path =
                task_entry.path() / "benchmark.json";
            if(!fs::exists(manifest_path))
                continue;

            BenchmarkRunConfig run = suite.defaults;
            std::string        error;
            if(load_manifest_run(manifest_path.string(), run, error))
            {
                // Auto-set dataset from directory name if not already set
                if(run.dataset.empty())
                    run.dataset = dataset_name;
                if(run.task_id.empty())
                    run.task_id = task_name;
                runs.push_back(std::move(run));
            }
        }
    }
}
}  // namespace

bool load_benchmark_suite(const std::string& path,
                          BenchmarkSuiteConfig& suite,
                          std::string& error_message)
{
    try
    {
        namespace fs = std::filesystem;
        const fs::path suite_path = fs::absolute(path).lexically_normal();
        const fs::path suite_dir  = suite_path.parent_path();

        auto json = gipc::Json::parse(std::ifstream(path));
        if(json.contains("output_root"))
            suite.output_root = json["output_root"].get<std::string>();
        if(json.contains("datasets_root"))
            suite.datasets_root = json["datasets_root"].get<std::string>();
        if(json.contains("dataset_selector"))
            suite.dataset_selector = json["dataset_selector"].get<std::string>();
        if(json.contains("task_selector"))
            suite.task_selector = json["task_selector"].get<std::string>();
        if(json.contains("auto_discover"))
            suite.auto_discover = json["auto_discover"].get<bool>();

        suite.output_root  = resolve_suite_relative_path(suite_dir, suite.output_root);
        suite.datasets_root = resolve_suite_relative_path(suite_dir, suite.datasets_root);

        suite.defaults.output_root = suite.output_root;
        if(json.contains("defaults"))
            apply_json_to_run(json["defaults"], suite.defaults);
        suite.defaults.output_root = suite.output_root;
        suite.defaults.settings_path =
            resolve_suite_relative_path(suite_dir, suite.defaults.settings_path);
        suite.defaults.asset_root =
            resolve_suite_relative_path(suite_dir, suite.defaults.asset_root);
        suite.defaults.manifest_path =
            resolve_suite_relative_path(suite_dir, suite.defaults.manifest_path);

        suite.runs.clear();
        if(json.contains("runs"))
        {
            for(const auto& run_json : json["runs"])
            {
                auto run = suite.defaults;
                run.output_root = suite.output_root;
                apply_json_to_run(run_json, run);
                run.settings_path =
                    resolve_suite_relative_path(suite_dir, run.settings_path);
                run.asset_root = resolve_suite_relative_path(suite_dir, run.asset_root);
                run.manifest_path =
                    resolve_suite_relative_path(suite_dir, run.manifest_path);
                suite.runs.push_back(run);
            }
        }

        if(suite.auto_discover)
            discover_manifest_runs(suite, suite.runs);

        if(suite.runs.empty())
        {
            std::ostringstream oss;
            oss << "benchmark suite resolved to zero runs"
                << " (suite=" << suite_path.string()
                << ", datasets_root=" << suite.datasets_root;
            if(!suite.dataset_selector.empty())
                oss << ", dataset_selector=" << suite.dataset_selector;
            if(!suite.task_selector.empty())
                oss << ", task_selector=" << suite.task_selector;
            oss << ")";
            error_message = oss.str();
            return false;
        }

        return true;
    }
    catch(const std::exception& e)
    {
        error_message = e.what();
        return false;
    }
}

bool load_manifest_run(const std::string& manifest_path,
                       BenchmarkRunConfig& run,
                       std::string& error_message)
{
    try
    {
        auto json = gipc::Json::parse(std::ifstream(manifest_path));
        apply_json_to_run(json, run);

        // Store manifest path for later use by scene setup
        if(run.manifest_path.empty())
            run.manifest_path = manifest_path;

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
