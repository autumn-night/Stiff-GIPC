#include "app/common/scene_registry.h"

#include <GIPC.cuh>
#include <device_fem_data.cuh>
#include <gipc/type_define.h>
#include <gipc/utils/json.h>
#include <gipc/utils/simple_scene_importer.h>
#include <load_mesh.h>

#include <Eigen/Geometry>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace app::common
{
namespace
{
struct LocalSceneObject
{
    std::string     mesh_path;
    int             dimensions = 3;
    Eigen::Matrix4d transform  = Eigen::Matrix4d::Identity();
};

struct IpcSimSceneSpec
{
    std::vector<LocalSceneObject> objects;
    bool                          has_density         = false;
    bool                          has_stiffness       = false;
    bool                          has_self_friction   = false;
    bool                          has_ground_friction = false;
    bool                          has_time_step       = false;
    double                        density             = 0.0;
    double                        young_modulus       = 0.0;
    double                        poisson_rate        = 0.0;
    double                        self_friction       = 0.0;
    double                        ground_friction     = 0.0;
    double                        ipc_dt              = 0.0;
};

std::string trim_copy(const std::string& value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    if(begin == std::string::npos)
        return {};

    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::vector<std::string> split_whitespace(const std::string& line)
{
    std::istringstream          stream(line);
    std::vector<std::string> tokens;
    for(std::string token; stream >> token;)
        tokens.push_back(token);
    return tokens;
}

Eigen::Matrix4d build_transform(double tx,
                                double ty,
                                double tz,
                                double rx_deg,
                                double ry_deg,
                                double rz_deg,
                                double sx,
                                double sy,
                                double sz)
{
    using Transform = Eigen::Transform<double, 3, Eigen::Affine>;
    constexpr double degrees_to_radians = 3.14159265358979323846 / 180.0;

    Transform transform = Transform::Identity();
    transform.translate(Eigen::Vector3d(tx, ty, tz));
    transform.rotate(Eigen::AngleAxisd(rx_deg * degrees_to_radians, Eigen::Vector3d::UnitX()));
    transform.rotate(Eigen::AngleAxisd(ry_deg * degrees_to_radians, Eigen::Vector3d::UnitY()));
    transform.rotate(Eigen::AngleAxisd(rz_deg * degrees_to_radians, Eigen::Vector3d::UnitZ()));
    transform.scale(Eigen::Vector3d(sx, sy, sz));
    return transform.matrix();
}

std::string resolve_ipc_sim_asset_path(const std::filesystem::path& asset_root,
                                       const std::string&           asset_path)
{
    std::filesystem::path relative_path = asset_path;
    if(relative_path.empty())
        throw std::runtime_error("empty ipc-sim asset path");

    if(relative_path.begin() != relative_path.end() && *relative_path.begin() == "input")
        relative_path = relative_path.lexically_relative("input");

    return (asset_root / relative_path).lexically_normal().string();
}

void configure_stanford_scene(const std::string& asset_root,
                              const std::string& task_id,
                              SceneSetupContext& context)
{
    auto mesh_relative_path = std::filesystem::path(task_id);
    if(mesh_relative_path.extension() != ".obj")
        mesh_relative_path += ".obj";

    const auto mesh_path = (std::filesystem::path(asset_root) / mesh_relative_path)
                               .lexically_normal()
                               .string();

    if(!std::filesystem::exists(mesh_path))
        throw std::runtime_error("stanford mesh not found: " + mesh_path);

    gipc::SimpleSceneImporter importer;
    context.linear_system_buff_scale = 1.0;
    importer.load_geometry(context.tet_mesh,
                           2,
                           gipc::BodyType::FEM,
                           build_transform(0.0, 1.0, 0.0, -90.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                           context.ipc.clothYoungModulus,
                           mesh_path,
                           context.ipc.pcg_data.P_type);
}

IpcSimSceneSpec parse_ipc_sim_scene(const std::filesystem::path& scene_path,
                                    const std::filesystem::path& asset_root)
{
    IpcSimSceneSpec spec;

    std::ifstream file(scene_path);
    if(!file.is_open())
        throw std::runtime_error("failed to open ipc-sim example: " + scene_path.string());

    std::string line;
    int         remaining_shape_lines = 0;
    bool        in_section            = false;
    while(std::getline(file, line))
    {
        const auto comment_pos = line.find('#');
        if(comment_pos != std::string::npos)
            line.erase(comment_pos);

        line = trim_copy(line);
        if(line.empty())
            continue;

        auto tokens = split_whitespace(line);
        if(tokens.empty())
            continue;

        if(in_section)
        {
            if(tokens[0] == "section" && tokens.size() >= 2 && tokens[1] == "end")
                in_section = false;
            continue;
        }

        if(remaining_shape_lines > 0)
        {
            --remaining_shape_lines;
            if(tokens.size() != 10)
            {
                throw std::runtime_error(
                    "unsupported ipc-sim shape line in " + scene_path.string() + ": " + line);
            }

            const auto extension = std::filesystem::path(tokens[0]).extension().string();
            int        dimensions = 0;
            if(extension == ".msh")
                dimensions = 3;
            else if(extension == ".obj")
                dimensions = 2;
            else
            {
                throw std::runtime_error(
                    "unsupported ipc-sim mesh extension in " + scene_path.string() + ": "
                    + tokens[0]);
            }

            const auto mesh_path = resolve_ipc_sim_asset_path(asset_root, tokens[0]);
            if(!std::filesystem::exists(mesh_path))
                throw std::runtime_error("ipc-sim mesh not found: " + mesh_path);

            spec.objects.push_back(LocalSceneObject{
                mesh_path,
                dimensions,
                build_transform(std::stod(tokens[1]),
                                std::stod(tokens[2]),
                                std::stod(tokens[3]),
                                std::stod(tokens[4]),
                                std::stod(tokens[5]),
                                std::stod(tokens[6]),
                                std::stod(tokens[7]),
                                std::stod(tokens[8]),
                                std::stod(tokens[9]))});
            continue;
        }

        if(tokens[0] == "section")
        {
            in_section = true;
            continue;
        }
        if(tokens[0] == "shapes")
        {
            if(tokens.size() != 3 || tokens[1] != "input")
                throw std::runtime_error("unsupported ipc-sim shapes directive in "
                                         + scene_path.string() + ": " + line);
            remaining_shape_lines = std::stoi(tokens[2]);
            continue;
        }
        if(tokens[0] == "density" && tokens.size() >= 2)
        {
            spec.has_density = true;
            spec.density     = std::stod(tokens[1]);
            continue;
        }
        if(tokens[0] == "stiffness" && tokens.size() >= 3)
        {
            spec.has_stiffness = true;
            spec.young_modulus = std::stod(tokens[1]);
            spec.poisson_rate  = std::stod(tokens[2]);
            continue;
        }
        if(tokens[0] == "selfFric" && tokens.size() >= 2)
        {
            spec.has_self_friction = true;
            spec.self_friction     = std::stod(tokens[1]);
            continue;
        }
        if(tokens[0] == "ground" && tokens.size() >= 2)
        {
            spec.has_ground_friction = true;
            spec.ground_friction     = std::stod(tokens[1]);
            continue;
        }
        if(tokens[0] == "time" && tokens.size() >= 3)
        {
            spec.has_time_step = true;
            spec.ipc_dt        = std::stod(tokens[2]);
            continue;
        }
        if(tokens[0] == "meshCO")
        {
            throw std::runtime_error("unsupported ipc-sim directive in "
                                     + scene_path.string() + ": " + line);
        }
    }

    if(remaining_shape_lines != 0)
        throw std::runtime_error("truncated ipc-sim shapes block in " + scene_path.string());
    if(spec.objects.empty())
        throw std::runtime_error("ipc-sim example contains no supported shapes: "
                                 + scene_path.string());

    return spec;
}

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

void configure_local_dataset_scene(const std::string& dataset,
                                   const std::string& task_id,
                                   const std::string& asset_root,
                                   SceneSetupContext& context)
{
    if(asset_root.empty())
        throw std::runtime_error("local dataset benchmark requires asset_root");

    if(dataset == "stanford")
        return configure_stanford_scene(asset_root, task_id, context);

    if(dataset == "ipc-sim")
    {
        auto scene_relative_path = std::filesystem::path(task_id);
        if(scene_relative_path.extension() != ".txt")
            scene_relative_path += ".txt";

        const auto scene_path = (std::filesystem::path(asset_root) / scene_relative_path)
                                    .lexically_normal();
        const auto spec = parse_ipc_sim_scene(scene_path, std::filesystem::path(asset_root));

        if(spec.has_density)
            context.ipc.density = spec.density;
        if(spec.has_stiffness)
        {
            context.ipc.YoungModulus = spec.young_modulus;
            context.ipc.PoissonRate  = spec.poisson_rate;
        }
        if(spec.has_self_friction)
            context.ipc.frictionRate = spec.self_friction;
        if(spec.has_ground_friction)
            context.ipc.gd_frictionRate = spec.ground_friction;
        if(spec.has_time_step)
            context.ipc.IPC_dt = spec.ipc_dt;

        context.linear_system_buff_scale = 1.0;
        gipc::SimpleSceneImporter importer;
        for(const auto& object : spec.objects)
        {
            importer.load_geometry(context.tet_mesh,
                                   object.dimensions,
                                   gipc::BodyType::FEM,
                                   object.transform,
                                   context.ipc.YoungModulus,
                                   object.mesh_path,
                                   context.ipc.pcg_data.P_type);
        }
        return;
    }

    throw std::runtime_error("Unsupported local dataset benchmark: " + dataset);
}

void configure_external_manifest_scene(const std::string& manifest_path,
                                       SceneSetupContext& context)
{
    auto json = gipc::Json::parse(std::ifstream(manifest_path));

    // If the manifest provides a scene_json path, use SimpleSceneImporter
    // like case3 (wrecking_ball) does.
    if(json.contains("scene_json"))
    {
        std::string scene_json = json["scene_json"].get<std::string>();
        std::string mesh_dir;
        if(json.contains("mesh_dir"))
            mesh_dir = json["mesh_dir"].get<std::string>();

        // Default body type from manifest, or default to FEM
        gipc::BodyType body_type = gipc::BodyType::FEM;
        if(json.contains("body_type"))
        {
            std::string bt = json["body_type"].get<std::string>();
            if(bt == "ABD")
                body_type = gipc::BodyType::ABD;
            else if(bt == "FEM")
                body_type = gipc::BodyType::FEM;
        }

        if(json.contains("linear_system_buff_scale"))
            context.linear_system_buff_scale = json["linear_system_buff_scale"].get<double>();

        gipc::SimpleSceneImporter importer{scene_json, mesh_dir, body_type};
        importer.import_scene(context.tet_mesh);
        return;
    }

    // If the manifest provides a scene_template (builtin scene name), delegate
    if(json.contains("scene_template"))
    {
        std::string scene_name = json["scene_template"].get<std::string>();
        return configure_builtin_scene(scene_name, context);
    }

    throw std::runtime_error(
        "Manifest does not specify 'scene_json' or 'scene_template': "
        + manifest_path);
}
}  // namespace app::common
