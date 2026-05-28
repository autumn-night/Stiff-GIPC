#pragma once

#include <string>
#include <string_view>
#include <vector>

class GIPC;
class device_TetraData;
class tetrahedra_obj;

namespace app::common
{
struct SceneSetupContext
{
    GIPC&            ipc;
    device_TetraData& device_tet_mesh;
    tetrahedra_obj&  tet_mesh;
    std::string      assets_dir;
    double&          linear_system_buff_scale;
};

bool is_builtin_scene(std::string_view scene_name);
std::vector<std::string> builtin_scene_names();
void configure_builtin_scene(std::string_view scene_name, SceneSetupContext& context);
void configure_local_dataset_scene(const std::string& dataset,
                                   const std::string& task_id,
                                   const std::string& asset_root,
                                   SceneSetupContext& context);

// External manifest-based scene loading for dataset benchmarking.
// When a run config has a manifest_path, this function loads the scene
// using the manifest's scene_json or fallback scene_template specification.
void configure_external_manifest_scene(const std::string& manifest_path,
                                       SceneSetupContext& context);
}  // namespace app::common
