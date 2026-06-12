#include "app/bench/benchmark_config.h"
#include "app/bench/benchmark_runner.h"

#include <iostream>
#include <stdexcept>

namespace
{
std::vector<std::string> collect_args(int argc, char** argv)
{
    std::vector<std::string> args;
    args.reserve(argc);
    for(int i = 0; i < argc; ++i)
        args.emplace_back(argv[i]);
    return args;
}

app::bench::BenchmarkRunConfig parse_single_run(int argc, char** argv)
{
    app::bench::BenchmarkRunConfig run;
    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto require_value = [&](const std::string& flag) -> std::string {
            if(i + 1 >= argc)
                throw std::runtime_error("missing value for `" + flag + "`");
            return argv[++i];
        };

        if(arg == "--baseline")
        {
            gipc::BenchmarkBaseline baseline;
            auto value = require_value(arg);
            if(!gipc::try_parse_benchmark_baseline(value, baseline))
                throw std::runtime_error("unsupported baseline: " + value);
            run.baseline = baseline;
        }
        else if(arg == "--scene")
            run.scene = require_value(arg);
        else if(arg == "--frames")
            run.frames = std::stoi(require_value(arg));
        else if(arg == "--warmup")
            run.warmup = std::stoi(require_value(arg));
        else if(arg == "--output")
            run.output_root = require_value(arg);
        else if(arg == "--settings")
            run.settings_path = require_value(arg);
        else if(arg == "--seed")
            run.seed = std::stoi(require_value(arg));
        else if(arg == "--frame-start")
            run.frame_start = std::stoi(require_value(arg));
        else if(arg == "--frame-cap")
            run.frame_cap = std::stoi(require_value(arg));
        else if(arg == "--no-save-surface")
            run.save_surface = false;
        else if(arg == "--manifest")
            run.manifest_path = require_value(arg);
        else if(arg == "--dataset")
            run.dataset = require_value(arg);
        else if(arg == "--task-id")
            run.task_id = require_value(arg);
        else if(arg == "--asset-root")
            run.asset_root = require_value(arg);
        else if(arg == "--notes")
            run.notes = require_value(arg);
        else if(arg == "--suite")
            throw std::runtime_error("`--suite` should be handled before single-run parsing");
        else
            throw std::runtime_error("unknown argument: " + arg);
    }

    if(run.scene.empty() && run.manifest_path.empty())
        throw std::runtime_error("`--scene` or `--manifest` is required for single benchmark run");
    return run;
}
}  // namespace

int main(int argc, char** argv)
{
    try
    {
        std::vector<std::string> args = collect_args(argc, argv);
        app::bench::BenchmarkRunner runner(args);

        for(int i = 1; i < argc; ++i)
        {
            if(std::string_view(argv[i]) == "--suite")
            {
                if(i + 1 >= argc)
                    throw std::runtime_error("missing value for `--suite`");
                app::bench::BenchmarkSuiteConfig suite;
                std::string                     error_message;
                if(!app::bench::load_benchmark_suite(argv[i + 1], suite, error_message))
                    throw std::runtime_error("failed to load suite: " + error_message);
                std::cout << "Loaded suite: " << argv[i + 1] << '\n'
                          << "  output_root: " << suite.output_root << '\n'
                          << "  datasets_root: " << suite.datasets_root << '\n'
                          << "  run_count: " << suite.runs.size() << std::endl;
                runner.run_suite(suite);
                return 0;
            }
        }

        auto run_config = parse_single_run(argc, argv);
        // If --manifest was given, load it and merge with CLI overrides
        if(!run_config.manifest_path.empty())
        {
            app::bench::BenchmarkRunConfig manifest_run;
            std::string                   error_message;
            if(!app::bench::load_manifest_run(run_config.manifest_path,
                                              manifest_run,
                                              error_message))
                throw std::runtime_error("failed to load manifest: " + error_message);

            // CLI arguments take precedence over manifest values.
            // Only fill in fields that were NOT explicitly provided via CLI.
            if(run_config.scene.empty())
                run_config.scene = manifest_run.scene;
            // baseline: CLI default is always set, so we always use CLI value
            if(run_config.settings_path.empty())
                run_config.settings_path = manifest_run.settings_path;
            if(run_config.task_id.empty())
                run_config.task_id = manifest_run.task_id;
            if(run_config.dataset.empty())
                run_config.dataset = manifest_run.dataset;
            if(run_config.asset_root.empty())
                run_config.asset_root = manifest_run.asset_root;
            if(run_config.output_tag.empty())
                run_config.output_tag = manifest_run.output_tag;
            if(run_config.notes.empty())
                run_config.notes = manifest_run.notes;
            if(run_config.manifest_path.empty())
                run_config.manifest_path = manifest_run.manifest_path;
        }
        runner.run_single(run_config);
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "gipc_bench error: " << e.what() << std::endl;
        return 1;
    }
}
