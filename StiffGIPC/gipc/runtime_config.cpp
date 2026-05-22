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
}  // namespace gipc
