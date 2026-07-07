#pragma once

#include <gipc/utils/json.h>
#include <string>
#include <string_view>

namespace gipc
{
enum class BenchmarkBaseline
{
    GIPC = 0,
    StiffGIPC_SRBK,
    StiffGIPC_CEMAS_SRBK
};

enum class LinearAssemblyBackend
{
    LegacyGipc = 0,
    SRBK
};

enum class SpmvBackend
{
    LegacyGipc = 0,
    SRBK
};

enum class MasBackend
{
    GpuMas = 0,
    CEMAS
};

struct RuntimeBackendConfig
{
    BenchmarkBaseline     baseline         = BenchmarkBaseline::StiffGIPC_CEMAS_SRBK;
    LinearAssemblyBackend assembly_backend = LinearAssemblyBackend::SRBK;
    SpmvBackend           spmv_backend     = SpmvBackend::SRBK;
    MasBackend            mas_backend      = MasBackend::CEMAS;

    // Preconditioner improvement flags
    bool   bend_aware_neighbor      = false;  // Step 1: add hinge opposite vertices as neighbors
    bool   contact_aware_precond    = false;  // Step 2: Schur complement enhancement for cross-cluster contacts
    bool   inexact_newton           = false;  // Step 3: dynamic CG tolerance per Newton step
    double inexact_eta_early        = 1e-2;   // Step 3: CG tolerance for early Newton steps
    double inexact_eta_mid          = 1e-3;   // Step 3: CG tolerance for mid Newton steps
    int    inexact_early_steps      = 5;      // Step 3: number of "early" Newton steps
    int    inexact_mid_steps        = 15;     // Step 3: number of "mid" Newton steps
    bool   precond_reuse            = false;  // Step 4: reuse aggregation structure
    int    precond_reuse_interval   = 0;      // Step 4: reaggregate every N steps (0 = disabled, only cpnum threshold triggers)
    double precond_reuse_cpnum_threshold = 0.1; // Step 4: reaggregation trigger threshold (relative cpNum change)
    bool   diag_cluster_stats       = false;  // Step 0: collect cross-cluster triplet diagnostics
};

RuntimeBackendConfig resolve_runtime_backend_config(BenchmarkBaseline baseline);

std::string_view to_string(BenchmarkBaseline baseline);
std::string_view to_string(LinearAssemblyBackend backend);
std::string_view to_string(SpmvBackend backend);
std::string_view to_string(MasBackend backend);

bool try_parse_benchmark_baseline(std::string_view value, BenchmarkBaseline& baseline);
bool try_parse_mas_backend(std::string_view value, MasBackend& backend);

// Apply preconditioner improvement flags from a JSON object into RuntimeBackendConfig.
// The JSON object should contain keys matching the field names (e.g. "bend_aware_neighbor": true).
// Missing keys are left at their current values.
void apply_preconditioner_config_from_json(const Json& json, RuntimeBackendConfig& config);
}  // namespace gipc
