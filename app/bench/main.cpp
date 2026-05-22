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
        else if(arg == "--suite")
            throw std::runtime_error("`--suite` should be handled before single-run parsing");
        else
            throw std::runtime_error("unknown argument: " + arg);
    }

    if(run.scene.empty())
        throw std::runtime_error("`--scene` is required for single benchmark run");
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
                runner.run_suite(suite);
                return 0;
            }
        }

        runner.run_single(parse_single_run(argc, argv));
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "gipc_bench error: " << e.what() << std::endl;
        return 1;
    }
}
