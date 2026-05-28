#pragma once

#include <string>

#include <gipc/runtime_config.h>

class GIPC;
class device_TetraData;
class tetrahedra_obj;

namespace app::common
{
struct SimulationContext
{
    GIPC&             ipc;
    device_TetraData& device_tet_mesh;
    tetrahedra_obj&   tet_mesh;
    double&           collision_detection_buff_scale;
    double&           motion_rate;
    double&           linear_system_buff_scale;
    std::string       assets_dir;
    std::string       metis_dir;
};

struct SimulationBootstrapOptions
{
    std::string                scene_name;
    std::string                settings_path;
    std::string                manifest_path;
    std::string                dataset;
    std::string                task_id;
    std::string                asset_root;
    gipc::RuntimeBackendConfig runtime_backend_config{};
};

void initialize_cuda(int device_id = 0);
void apply_runtime_backend_config(SimulationContext& context,
                                  const gipc::RuntimeBackendConfig& config);
void bootstrap_simulation(SimulationContext& context,
                          const SimulationBootstrapOptions& options);
}  // namespace app::common
