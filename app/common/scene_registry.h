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
}  // namespace app::common
