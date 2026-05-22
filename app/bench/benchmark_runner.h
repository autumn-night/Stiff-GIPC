#pragma once

#include "app/bench/benchmark_config.h"
#include "app/bench/benchmark_report.h"

#include <string>
#include <utility>
#include <vector>

namespace app::bench
{
class BenchmarkRunner
{
  public:
    explicit BenchmarkRunner(std::vector<std::string> command_args)
        : m_command_args(std::move(command_args))
    {
    }

    BenchmarkSummaryRow run_single(const BenchmarkRunConfig& run_config);
    std::vector<BenchmarkSummaryRow> run_suite(const BenchmarkSuiteConfig& suite_config);

  private:
    std::vector<std::string> m_command_args;
};
}  // namespace app::bench
