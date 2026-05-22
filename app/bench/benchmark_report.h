#pragma once

#include "app/bench/benchmark_config.h"

#include <gipc/utils/json.h>

#include <string>
#include <vector>

namespace app::bench
{
struct BenchmarkSummaryRow
{
    std::string scene;
    std::string baseline;
    int         frames = 0;
    int         warmup = 0;
    double avg_hess_ms          = 0.0;
    double avg_lsolver_ms       = 0.0;
    double avg_lines_ms         = 0.0;
    double avg_misc_ms          = 0.0;
    double avg_time_tot_ms      = 0.0;
    double avg_newton           = 0.0;
    double avg_cg               = 0.0;
    double avg_contact_pairs    = 0.0;
    double std_time_tot_ms      = 0.0;
};

BenchmarkSummaryRow summarize_frames(const BenchmarkRunConfig& run_config,
                                     const gipc::Json& frames_json);

gipc::Json make_raw_report(const BenchmarkRunConfig& run_config,
                           const gipc::Json& frames_json);

gipc::Json make_meta_report(const BenchmarkRunConfig& run_config,
                            const std::vector<std::string>& command_args);

void write_raw_report(const std::string& path, const gipc::Json& raw_report);
void write_meta_report(const std::string& path, const gipc::Json& meta_report);
void write_summary_csv(const std::string& path,
                       const std::vector<BenchmarkSummaryRow>& rows);
}  // namespace app::bench
