#include "app/common/scene_registry.h"

#include <GIPC.cuh>
#include <device_fem_data.cuh>
#include <gipc/type_define.h>
#include <gipc/utils/simple_scene_importer.h>
#include <load_mesh.h>

#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace app::common
{
namespace
{
void set_case1(SceneSetupContext& context)
{
    auto& ipc      = context.ipc;
    auto& tet_mesh = context.tet_mesh;
    context.linear_system_buff_scale = 1.0;

    double                    dist       = 0.2;
    int                       count      = 4;
    int                       count_y    = 4;
    double                    fem_height = -0.8;
    double                    abd_height = -0.6;
    gipc::SimpleSceneImporter importer;
    double                    young_modulus = 1e4;

    for(int k = 0; k < count_y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {
                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;

                double3 position_offset =
                    make_double3(-pos.x(), -abd_height - 2 * dist * k, -pos.y());
                double          scale     = 0.4;
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);

                importer.load_geometry(tet_mesh,
                                       3,
                                       gipc::BodyType::ABD,
                                       transform,
                                       1e5,
                                       context.assets_dir + "tetMesh/cube.msh",
                                       ipc.pcg_data.P_type);
            }
        }
    }

    for(int k = 0; k < count_y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {
                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;

                double3 position_offset =
                    make_double3(-pos.x(), -fem_height - 2 * dist * k, -pos.y());
                double          scale     = 0.4;
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);

                importer.load_geometry(tet_mesh,
                                       3,
                                       gipc::BodyType::FEM,
                                       transform,
                                       young_modulus,
                                       context.assets_dir + "tetMesh/cube.msh",
                                       ipc.pcg_data.P_type);
            }
        }
    }
}

void set_case2(SceneSetupContext& context)
{
    auto& ipc      = context.ipc;
    auto& tet_mesh = context.tet_mesh;

    gipc::SimpleSceneImporter importer;
    double                    scale           = 0.2;
    double3                   position_offset = make_double3(0, -0.5, 0);
    Eigen::Matrix4d           transform       = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
    transform.block<3, 1>(0, 3) =
        -Eigen::Vector3d(position_offset.x, position_offset.y, position_offset.z);

    context.linear_system_buff_scale = 1.0;
    double young_modulus = 1e4;
    auto   mesh0_path    = context.assets_dir + "tetMesh/bunny2.msh";
    importer.load_geometry(tet_mesh,
                           3,
                           gipc::BodyType::ABD,
                           transform,
                           young_modulus,
                           mesh0_path,
                           ipc.pcg_data.P_type);

    position_offset             = make_double3(0, 0.65, 0);
    transform                   = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
    transform.block<3, 1>(0, 3) =
        -Eigen::Vector3d(position_offset.x, position_offset.y, position_offset.z);

    importer.load_geometry(tet_mesh,
                           3,
                           gipc::BodyType::FEM,
                           transform,
                           young_modulus,
                           mesh0_path,
                           ipc.pcg_data.P_type);

    position_offset             = make_double3(0, 0, 0);
    transform                   = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    transform.block<3, 1>(0, 3) =
        -Eigen::Vector3d(position_offset.x, position_offset.y, position_offset.z);
    importer.load_geometry(tet_mesh,
                           2,
                           gipc::BodyType::FEM,
                           transform,
                           1e4,
                           context.assets_dir + "triMesh/cloth_high.obj",
                           ipc.pcg_data.P_type);
}

void set_case3(SceneSetupContext& context)
{
    context.linear_system_buff_scale = 1.0;
    gipc::SimpleSceneImporter importer{context.assets_dir + "scene/json/wrecking-ball-simple.json",
                                       context.assets_dir + "tetMesh/wrecking-ball-mesh/",
                                       gipc::BodyType::ABD};
    importer.import_scene(context.tet_mesh);
}

void set_case4(SceneSetupContext& context)
{
    auto& ipc      = context.ipc;
    auto& tet_mesh = context.tet_mesh;

    ipc.pcg_data.P_type = 1;
    context.linear_system_buff_scale = 1.0;

    gipc::SimpleSceneImporter importer;
    double                    scale = 0.6;

    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    Transform transform = Transform::Identity();
    transform.translate(Eigen::Vector3d{0, 1.0, 0});
    transform.scale(scale);
    transform.rotate(Eigen::AngleAxisd(3.1415926 / 2, Eigen::Vector3d::UnitX()));

    importer.load_geometry(tet_mesh,
                           2,
                           gipc::BodyType::FEM,
                           transform.matrix(),
                           1e4,
                           context.assets_dir + "triMesh/cloth_high.obj",
                           ipc.pcg_data.P_type);

    int          fixed_vertex_num = 0;
    const double eps              = 1e-4;
    double       max_y            = tet_mesh.maxTConer.y;
    double       min_x            = tet_mesh.minTConer.x;
    double       max_x            = tet_mesh.maxTConer.x;
    for(int i = 0; i < tet_mesh.vertexNum; i++)
    {
        if(tet_mesh.vertexes[i].y > max_y - eps
           && (tet_mesh.vertexes[i].x < min_x + eps || tet_mesh.vertexes[i].x > max_x - eps))
        {
            tet_mesh.boundaryTypies[i] = 1;
            fixed_vertex_num++;
        }
    }
    std::cout << "fixed vertex num: " << fixed_vertex_num << std::endl;
}

void set_case5(SceneSetupContext& context)
{
    auto& ipc             = context.ipc;
    auto& tet_mesh        = context.tet_mesh;
    auto& device_tet_mesh = context.device_tet_mesh;

    ipc.pcg_data.P_type = 1;
    context.linear_system_buff_scale = 2.0;

    gipc::SimpleSceneImporter importer;
    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    Transform transform = Transform::Identity();
    transform.scale(1.0);

    ipc.PoissonRate = 0.48;
    importer.load_geometry(tet_mesh,
                           3,
                           gipc::BodyType::FEM,
                           transform.matrix(),
                           1e4,
                           context.assets_dir + "tetMesh/high_mat.msh",
                           ipc.pcg_data.P_type);

    for(size_t i = 0; i < tet_mesh.vertexes.size(); i++)
    {
        tet_mesh.apply_gravity[i] = 0;
    }

    const double eps = 1e-4;
    for(int i = 0; i < tet_mesh.vertexNum; i++)
    {
        if(tet_mesh.vertexes[i].x < -0.5 + eps || tet_mesh.vertexes[i].x > 0.5 - eps)
        {
            tet_mesh.targetIndex.push_back(i);
            tet_mesh.targetPos.push_back(tet_mesh.vertexes[i]);
        }
    }
    tet_mesh.softNum = tet_mesh.targetIndex.size();
    std::cout << "soft constraint num: " << tet_mesh.softNum << std::endl;
    ipc.softMotionRate = 1;

    const double angular_vel = 3.14159265358979323846 / 5;
    device_tet_mesh.update_soft_constraint_functor =
        [angular_vel](double3 vertex, int, double ipc_dt) -> double3
    {
        double3 rotated_vertex = vertex;
        if(vertex.x < 0)
        {
            rotated_vertex = {vertex.x,
                              vertex.y * std::cos(angular_vel * ipc_dt)
                                  - vertex.z * std::sin(angular_vel * ipc_dt),
                              vertex.y * std::sin(angular_vel * ipc_dt)
                                  + vertex.z * std::cos(angular_vel * ipc_dt)};
        }
        if(vertex.x > 0)
        {
            rotated_vertex = {vertex.x,
                              vertex.y * std::cos(-angular_vel * ipc_dt)
                                  - vertex.z * std::sin(-angular_vel * ipc_dt),
                              vertex.y * std::sin(-angular_vel * ipc_dt)
                                  + vertex.z * std::cos(-angular_vel * ipc_dt)};
        }
        return rotated_vertex;
    };
}

void set_case6(SceneSetupContext& context)
{
    auto& ipc      = context.ipc;
    auto& tet_mesh = context.tet_mesh;

    context.linear_system_buff_scale = 2.0;
    ipc.pcg_data.P_type              = 1;

    double scale      = 0.3;
    double dist       = scale / 2;
    int    count      = 8;
    int    count_y    = 15;
    double global_offset = 1.0;
    double fem_height = global_offset + 1 - 0.8;
    double abd_height = fem_height - dist;

    for(int k = 0; k < count_y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {
                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;

                double3 position_offset =
                    double3{-pos.x(), -abd_height - 2 * dist * k, -pos.y()};
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);
                tet_mesh.load_tetrahedraMesh(context.assets_dir + "tetMesh/cube.msh",
                                             transform,
                                             1e6,
                                             gipc::BodyType::ABD);
            }
        }
    }

    for(int k = 0; k < count_y; ++k)
    {
        for(int i = 0; i < count; i++)
        {
            for(int j = 0; j < count; j++)
            {
                gipc::Vector2 ij{i, j};
                gipc::Vector2 pos =
                    ij * dist - gipc::Vector2::Ones() * dist * (count - 1) / 2.0;

                double3 position_offset =
                    double3{-pos.x(), -fem_height - 2 * dist * k, -pos.y()};
                Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
                transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
                transform.block<3, 1>(0, 3) = -Eigen::Vector3d(
                    position_offset.x, position_offset.y, position_offset.z);
                tet_mesh.load_tetrahedraMesh(context.assets_dir + "tetMesh/cube.msh",
                                             transform,
                                             5e4,
                                             gipc::BodyType::ABD);
            }
        }
    }

    gipc::SimpleSceneImporter importer;
    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    Transform transform = Transform::Identity();
    transform.scale(1.5);
    transform.translate(Eigen::Vector3d(0, 0.35, 0));
    importer.load_geometry(tet_mesh,
                           2,
                           gipc::BodyType::FEM,
                           transform.matrix(),
                           1e4,
                           context.assets_dir + "triMesh/cloth_high.obj",
                           ipc.pcg_data.P_type);

    const double eps = 1e-4;
    for(int i = 0; i < tet_mesh.vertexNum; i++)
    {
        if(tet_mesh.vertexes[i].x < -1.5 + eps || tet_mesh.vertexes[i].x > 1.5 - eps)
        {
            tet_mesh.boundaryTypies[i] = 1;
        }
    }

    ipc.relative_dhat = 1e-3;
    ipc.strainRate    = 1e6;
}
}  // namespace

bool is_builtin_scene(std::string_view scene_name)
{
    return scene_name == "case1" || scene_name == "box_pipe"
           || scene_name == "cloth_bunny" || scene_name == "case2"
           || scene_name == "wrecking_ball" || scene_name == "case3"
           || scene_name == "fixed_cloth" || scene_name == "case4"
           || scene_name == "mat_twist" || scene_name == "case5"
           || scene_name == "box_pile" || scene_name == "case6";
}

std::vector<std::string> builtin_scene_names()
{
    return {"cloth_bunny", "wrecking_ball", "mat_twist", "box_pile"};
}

void configure_builtin_scene(std::string_view scene_name, SceneSetupContext& context)
{
    if(scene_name == "case1" || scene_name == "box_pipe")
        return set_case1(context);
    if(scene_name == "cloth_bunny" || scene_name == "case2")
        return set_case2(context);
    if(scene_name == "wrecking_ball" || scene_name == "case3")
        return set_case3(context);
    if(scene_name == "fixed_cloth" || scene_name == "case4")
        return set_case4(context);
    if(scene_name == "mat_twist" || scene_name == "case5")
        return set_case5(context);
    if(scene_name == "box_pile" || scene_name == "case6")
        return set_case6(context);

    throw std::runtime_error("Unsupported benchmark scene: " + std::string(scene_name));
}
}  // namespace app::common
