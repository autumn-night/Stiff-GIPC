#pragma once

#include <gipc/runtime_config.h>

namespace gipc
{
class GlobalLinearSystemOptions
{
  public:
    LinearAssemblyBackend assembly_backend = LinearAssemblyBackend::SRBK;
    SpmvBackend           spmv_backend     = SpmvBackend::SRBK;
};
}  // namespace gipc
