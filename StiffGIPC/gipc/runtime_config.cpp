#include <gipc/runtime_config.h>

#include <algorithm>
#include <cctype>

namespace gipc
{
namespace
{
std::string normalize(std::string_view value)
{
    std::string result;
    result.reserve(value.size());
    for(char ch : value)
    {
        if(ch == '(' || ch == ')' || ch == '+' || ch == '-' || ch == ' ' || ch == '_')
            continue;
        result.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
    }
    return result;
}
}  // namespace

RuntimeBackendConfig resolve_runtime_backend_config(BenchmarkBaseline baseline)
{
    switch(baseline)
    {
        case BenchmarkBaseline::GIPC:
            return RuntimeBackendConfig{baseline,
                                        LinearAssemblyBackend::LegacyGipc,
                                        SpmvBackend::LegacyGipc,
                                        MasBackend::GpuMas};
        case BenchmarkBaseline::StiffGIPC_SRBK:
            return RuntimeBackendConfig{baseline,
                                        LinearAssemblyBackend::SRBK,
                                        SpmvBackend::SRBK,
                                        MasBackend::GpuMas};
        case BenchmarkBaseline::StiffGIPC_CEMAS_SRBK:
        default:
            return RuntimeBackendConfig{baseline,
                                        LinearAssemblyBackend::SRBK,
                                        SpmvBackend::SRBK,
                                        MasBackend::CEMAS};
    }
}

std::string_view to_string(BenchmarkBaseline baseline)
{
    switch(baseline)
    {
        case BenchmarkBaseline::GIPC:
            return "GIPC";
        case BenchmarkBaseline::StiffGIPC_SRBK:
            return "StiffGIPC_SRBK";
        case BenchmarkBaseline::StiffGIPC_CEMAS_SRBK:
        default:
            return "StiffGIPC_CEMAS_SRBK";
    }
}

std::string_view to_string(LinearAssemblyBackend backend)
{
    switch(backend)
    {
        case LinearAssemblyBackend::LegacyGipc:
            return "LegacyGipc";
        case LinearAssemblyBackend::SRBK:
        default:
            return "SRBK";
    }
}

std::string_view to_string(SpmvBackend backend)
{
    switch(backend)
    {
        case SpmvBackend::LegacyGipc:
            return "LegacyGipc";
        case SpmvBackend::SRBK:
        default:
            return "SRBK";
    }
}

std::string_view to_string(MasBackend backend)
{
    switch(backend)
    {
        case MasBackend::GpuMas:
            return "GpuMas";
        case MasBackend::CEMAS:
        default:
            return "CEMAS";
    }
}

bool try_parse_benchmark_baseline(std::string_view value, BenchmarkBaseline& baseline)
{
    const auto normalized = normalize(value);
    if(normalized == "GIPC")
    {
        baseline = BenchmarkBaseline::GIPC;
        return true;
    }
    if(normalized == "STIFFGIPCSRBK")
    {
        baseline = BenchmarkBaseline::StiffGIPC_SRBK;
        return true;
    }
    if(normalized == "STIFFGIPCCEMASSRBK")
    {
        baseline = BenchmarkBaseline::StiffGIPC_CEMAS_SRBK;
        return true;
    }
    return false;
}

bool try_parse_mas_backend(std::string_view value, MasBackend& backend)
{
    const auto normalized = normalize(value);
    if(normalized == "GPUMAS" || normalized == "MAS")
    {
        backend = MasBackend::GpuMas;
        return true;
    }
    if(normalized == "CEMAS")
    {
        backend = MasBackend::CEMAS;
        return true;
    }
    return false;
}

void apply_preconditioner_config_from_json(const Json& json, RuntimeBackendConfig& config)
{
    if(json.contains("bend_aware_neighbor"))
        config.bend_aware_neighbor = json["bend_aware_neighbor"].get<bool>();
    if(json.contains("contact_aware_precond"))
        config.contact_aware_precond = json["contact_aware_precond"].get<bool>();
    if(json.contains("inexact_newton"))
        config.inexact_newton = json["inexact_newton"].get<bool>();
    if(json.contains("inexact_eta_early"))
        config.inexact_eta_early = json["inexact_eta_early"].get<double>();
    if(json.contains("inexact_eta_mid"))
        config.inexact_eta_mid = json["inexact_eta_mid"].get<double>();
    if(json.contains("inexact_early_steps"))
        config.inexact_early_steps = json["inexact_early_steps"].get<int>();
    if(json.contains("inexact_mid_steps"))
        config.inexact_mid_steps = json["inexact_mid_steps"].get<int>();
    if(json.contains("precond_reuse"))
        config.precond_reuse = json["precond_reuse"].get<bool>();
    if(json.contains("precond_reuse_interval"))
        config.precond_reuse_interval = json["precond_reuse_interval"].get<int>();
    if(json.contains("precond_reuse_cpnum_threshold"))
        config.precond_reuse_cpnum_threshold = json["precond_reuse_cpnum_threshold"].get<double>();
    if(json.contains("diag_cluster_stats"))
        config.diag_cluster_stats = json["diag_cluster_stats"].get<bool>();
}
}  // namespace gipc
