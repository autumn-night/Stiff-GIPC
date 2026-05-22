#pragma once

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
};

RuntimeBackendConfig resolve_runtime_backend_config(BenchmarkBaseline baseline);

std::string_view to_string(BenchmarkBaseline baseline);
std::string_view to_string(LinearAssemblyBackend backend);
std::string_view to_string(SpmvBackend backend);
std::string_view to_string(MasBackend backend);

bool try_parse_benchmark_baseline(std::string_view value, BenchmarkBaseline& baseline);
bool try_parse_mas_backend(std::string_view value, MasBackend& backend);
}  // namespace gipc
