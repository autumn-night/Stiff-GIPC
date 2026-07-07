#include "app/common/sim_bootstrap.h"

#include "app/common/scene_registry.h"
#include "app/common/sim_settings.h"

#include <GIPC.cuh>
#include <device_fem_data.cuh>
#include <cuda_tools/cuda_tools.h>
#include <femEnergy.cuh>
#include <load_mesh.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

namespace app::common
{
namespace
{
void init_fem(GIPC& ipc, tetrahedra_obj& mesh)
{
    double mass_sum   = 0;
    double volume_sum = 0;

    ipc.lengthRateLame = ipc.YoungModulus / (2 * (1 + ipc.PoissonRate));
    ipc.volumeRateLame = ipc.YoungModulus * ipc.PoissonRate
                         / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
    ipc.lengthRate   = 4 * ipc.lengthRateLame / 3;
    ipc.volumeRate   = ipc.volumeRateLame + 5 * ipc.lengthRateLame / 6;
    ipc.stretchStiff = ipc.clothYoungModulus / (2 * (1 + ipc.PoissonRate));
    ipc.bendStiff    = ipc.bendYoungModulus * pow(ipc.clothThickness, 3)
                    / (24 * (1 - ipc.PoissonRate * ipc.PoissonRate));
    ipc.shearStiff   = 0.03 * ipc.stretchStiff * ipc.strainRate;

    for(int i = 0; i < mesh.tetrahedraNum; i++)
    {
        __GEIGEN__::Matrix3x3d dm;
        __calculateDms3D_double(mesh.vertexes.data(), mesh.tetrahedras[i], dm);

        __GEIGEN__::Matrix3x3d dm_inverse;
        __GEIGEN__::__Inverse(dm, dm_inverse);

        double volume = calculateVolum(mesh.vertexes.data(), mesh.tetrahedras[i]);

        mesh.masses[mesh.tetrahedras[i].x] += volume * ipc.density / 4;
        mesh.masses[mesh.tetrahedras[i].y] += volume * ipc.density / 4;
        mesh.masses[mesh.tetrahedras[i].z] += volume * ipc.density / 4;
        mesh.masses[mesh.tetrahedras[i].w] += volume * ipc.density / 4;

        mass_sum += volume * ipc.density;
        volume_sum += volume;
        mesh.DM_inverse.push_back(dm_inverse);
        mesh.volum.push_back(volume);

        double length_rate_lame =
            mesh.vert_youngth_modules[i] / (2 * (1 + ipc.PoissonRate));
        double volume_rate_lame = mesh.vert_youngth_modules[i] * ipc.PoissonRate
                                  / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
        mesh.lengthRate.push_back(4 * length_rate_lame / 3);
        mesh.volumeRate.push_back(volume_rate_lame + 5 * length_rate_lame / 6);
    }

    for(int i = 0; i < mesh.triangles.size(); i++)
    {
        __GEIGEN__::Matrix2x2d dm;
        __calculateDm2D_double(mesh.vertexes.data(), mesh.triangles[i], dm);

        __GEIGEN__::Matrix2x2d dm_inverse;
        __GEIGEN__::__Inverse2x2(dm, dm_inverse);

        double area = calculateArea(mesh.vertexes.data(), mesh.triangles[i]);
        area *= ipc.clothThickness;
        mesh.area.push_back(area);

        mesh.masses[mesh.triangles[i].x] += ipc.clothDensity * area / 3;
        mesh.masses[mesh.triangles[i].y] += ipc.clothDensity * area / 3;
        mesh.masses[mesh.triangles[i].z] += ipc.clothDensity * area / 3;

        mass_sum += area * ipc.clothDensity;
        volume_sum += area;
        mesh.tri_DM_inverse.push_back(dm_inverse);
    }

    mesh.meanMass  = mass_sum / mesh.vertexNum;
    mesh.meanVolum = volume_sum / mesh.vertexNum;
}

void set_mas_partition(tetrahedra_obj& tet_mesh)
{
    tet_mesh.partId_map_real.resize(tet_mesh.part_offset * BANKSIZE, -1);
    tet_mesh.real_map_partId.resize(tet_mesh.partId.size());
    int index = 0;
    for(int i = 0; i < tet_mesh.partId.size(); i++)
    {
        tet_mesh.partId_map_real[BANKSIZE * tet_mesh.partId[i] + index] = i;
        index++;
        if(i <= tet_mesh.partId.size() - 2 && tet_mesh.partId[i + 1] != tet_mesh.partId[i])
        {
            index = 0;
        }
    }
    index = 0;
    for(int i = 0; i < tet_mesh.partId_map_real.size(); i++)
    {
        if(tet_mesh.partId_map_real[i] == index)
        {
            tet_mesh.real_map_partId[index] = i;
            index++;
        }
    }
}

void upload_host_mesh_to_device(SimulationContext& context)
{
    auto& ipc             = context.ipc;
    auto& tet_mesh        = context.tet_mesh;
    auto& device_tet_mesh = context.device_tet_mesh;

    const double relative_dhat_scale =
        ipc.relative_dhat > 0.0 ? std::max(1.0, ipc.relative_dhat / 1e-3) : 1.0;
    const double time_step_scale =
        ipc.IPC_dt > 0.0 ? std::max(1.0, ipc.IPC_dt / 1e-2) : 1.0;
    const double contact_complexity_scale =
        std::max(relative_dhat_scale, time_step_scale);

    // Use contact_complexity_scale directly to ensure buffers scale proportionally
    // with the actual increase in collision pairs. With dhat 10x larger, the
    // effective barrier radius grows by sqrt(10)≈3.16x, and the number of
    // collision pairs within that radius grows roughly with the surface area
    // (quadratic) or volume (cubic), so sqrt-scaling is insufficient.
    // The collision pair buffer scales linearly (capped at 8x) and the triplet
    // buffer scales at sqrt rate (capped at 4x) to keep GPU memory manageable.
    const double collision_buffer_growth =
        std::min(8.0, contact_complexity_scale);
    const double linear_system_buffer_growth =
        std::min(4.0, std::max(1.0, std::sqrt(contact_complexity_scale)));

    const double collision_buffer_scale =
        context.collision_detection_buff_scale * collision_buffer_growth;

    context.linear_system_buff_scale =
        std::max(context.linear_system_buff_scale, linear_system_buffer_growth);

    device_tet_mesh.Malloc_DEVICE_MEM(tet_mesh.vertexNum,
                                      tet_mesh.tetrahedraNum,
                                      tet_mesh.triangleNum,
                                      tet_mesh.softNum,
                                      tet_mesh.tri_edges.size(),
                                      tet_mesh.abd_fem_count_info.total_body_num());

    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.masses,
                              tet_mesh.masses.data(),
                              tet_mesh.vertexNum * sizeof(double),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.apply_gravity,
                              tet_mesh.apply_gravity.data(),
                              tet_mesh.vertexNum * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.lengthRate,
                              tet_mesh.lengthRate.data(),
                              tet_mesh.tetrahedraNum * sizeof(double),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.volumeRate,
                              tet_mesh.volumeRate.data(),
                              tet_mesh.tetrahedraNum * sizeof(double),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.volum,
                              tet_mesh.volum.data(),
                              tet_mesh.tetrahedraNum * sizeof(double),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.vertexes,
                              tet_mesh.vertexes.data(),
                              tet_mesh.vertexNum * sizeof(double3),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.o_vertexes,
                              tet_mesh.vertexes.data(),
                              tet_mesh.vertexNum * sizeof(double3),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.tetrahedras,
                              tet_mesh.tetrahedras.data(),
                              tet_mesh.tetrahedraNum * sizeof(uint4),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.DmInverses,
                              tet_mesh.DM_inverse.data(),
                              tet_mesh.tetrahedraNum * sizeof(__GEIGEN__::Matrix3x3d),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.BoundaryType,
                              tet_mesh.boundaryTypies.data(),
                              tet_mesh.vertexNum * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.velocities,
                              tet_mesh.velocities.data(),
                              tet_mesh.vertexNum * sizeof(double3),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.targetIndex,
                              tet_mesh.targetIndex.data(),
                              tet_mesh.softNum * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.targetVert,
                              tet_mesh.targetPos.data(),
                              tet_mesh.softNum * sizeof(double3),
                              cudaMemcpyHostToDevice));

    device_tet_mesh.host_target_indices  = tet_mesh.targetIndex;
    device_tet_mesh.host_target_vertices = tet_mesh.targetPos;

    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.triDmInverses,
                              tet_mesh.tri_DM_inverse.data(),
                              tet_mesh.triangleNum * sizeof(__GEIGEN__::Matrix2x2d),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.area,
                              tet_mesh.area.data(),
                              tet_mesh.triangleNum * sizeof(double),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.triangles,
                              tet_mesh.triangles.data(),
                              tet_mesh.triangleNum * sizeof(uint3),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.tri_edges,
                              tet_mesh.tri_edges.data(),
                              tet_mesh.tri_edges.size() * sizeof(uint2),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.tri_edge_adj_vertex,
                              tet_mesh.tri_edges_adj_points.data(),
                              tet_mesh.tri_edges.size() * sizeof(uint2),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.body_id_to_boundary_type,
                              tet_mesh.body_id_to_is_fixed.data(),
                              tet_mesh.body_id_to_is_fixed.size() * sizeof(BodyBoundaryType),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.point_id_to_body_id,
                              tet_mesh.point_id_to_body_id.data(),
                              tet_mesh.point_id_to_body_id.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.tet_id_to_body_id,
                              tet_mesh.tet_id_to_body_id.data(),
                              tet_mesh.tet_id_to_body_id.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    ipc.vertexNum      = tet_mesh.vertexNum;
    ipc.tetrahedraNum  = tet_mesh.tetrahedraNum;
    ipc._vertexes      = device_tet_mesh.vertexes;
    ipc._rest_vertexes = device_tet_mesh.rest_vertexes;
    ipc.surf_vertexNum = tet_mesh.surfVerts.size();
    ipc.surface_Num    = tet_mesh.surface.size();
    ipc.edge_Num       = tet_mesh.surfEdges.size();
    ipc.tri_edge_num   = tet_mesh.tri_edges.size();
    ipc.triangleNum    = tet_mesh.triangleNum;
    ipc.targetVert     = device_tet_mesh.targetVert;
    ipc.targetInd      = device_tet_mesh.targetIndex;
    ipc.softNum        = tet_mesh.softNum;
    ipc.abd_fem_count_info = tet_mesh.abd_fem_count_info;

    ipc.MAX_CCD_COLLITION_PAIRS_NUM =
        1 * collision_buffer_scale
        * (((double)(ipc.surface_Num * 15 + ipc.edge_Num * 10))
           * std::max((ipc.IPC_dt / 0.01), 2.0));
    ipc.MAX_COLLITION_PAIRS_NUM =
        (ipc.surf_vertexNum * 3 + ipc.edge_Num * 2) * 3 * collision_buffer_scale;

    ipc.MALLOC_DEVICE_MEM();

    CUDA_SAFE_CALL(cudaMemcpy(
        ipc._faces, tet_mesh.surface.data(), ipc.surface_Num * sizeof(uint3), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(
        ipc._edges, tet_mesh.surfEdges.data(), ipc.edge_Num * sizeof(uint2), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(ipc._surfVerts,
                              tet_mesh.surfVerts.data(),
                              ipc.surf_vertexNum * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    ipc.initBVH(device_tet_mesh.BoundaryType, device_tet_mesh.point_id_to_body_id);

    if(ipc.pcg_data.P_type)
    {
        int neighbor_list_size = tet_mesh.getVertNeighbors();
        ipc.pcg_data.MP.initPreconditioner_Neighbor(ipc.vertexNum - tet_mesh.abd_vertexOffset,
                                                    tet_mesh.abd_vertexOffset,
                                                    neighbor_list_size,
                                                    ipc._collisonPairs,
                                                    tet_mesh.part_offset * BANKSIZE);

        // Pass runtime config to MAS preconditioner for preconditioner improvement flags
        ipc.pcg_data.MP.set_runtime_config(&ipc.runtime_backend_config);

        ipc.pcg_data.MP.neighborListSize = neighbor_list_size;
        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborListInit,
                                  tet_mesh.neighborList.data(),
                                  neighbor_list_size * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborStart,
                                  tet_mesh.neighborStart.data(),
                                  (ipc.vertexNum - tet_mesh.abd_vertexOffset) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborNumInit,
                                  tet_mesh.neighborNum.data(),
                                  (ipc.vertexNum - tet_mesh.abd_vertexOffset) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_partId_map_real,
                                  tet_mesh.partId_map_real.data(),
                                  tet_mesh.part_offset * BANKSIZE * sizeof(int),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_real_map_partId,
                                  tet_mesh.real_map_partId.data(),
                                  tet_mesh.real_map_partId.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
        ipc.pcg_data.MP.initPreconditioner_Matrix();
    }

    CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.rest_vertexes,
                              device_tet_mesh.o_vertexes,
                              ipc.vertexNum * sizeof(double3),
                              cudaMemcpyDeviceToDevice));

#ifdef USE_QUADRATIC_BENDING
    if(!tet_mesh.tri_edges.empty())
    {
        std::vector<Eigen::Matrix4d> q_host(tet_mesh.tri_edges.size());
        std::vector<double3>         rest_verts_host(ipc.vertexNum);
        CUDA_SAFE_CALL(cudaMemcpy(rest_verts_host.data(),
                                  device_tet_mesh.rest_vertexes,
                                  ipc.vertexNum * sizeof(double3),
                                  cudaMemcpyDeviceToHost));
        PrepareQuadBendingQ(rest_verts_host.data(),
                            tet_mesh.tri_edges.data(),
                            tet_mesh.tri_edges_adj_points.data(),
                            tet_mesh.tri_edges.size(),
                            q_host.data());
        CUDA_SAFE_CALL(cudaMemcpy(device_tet_mesh.quad_bending_Q,
                                  q_host.data(),
                                  tet_mesh.tri_edges.size() * sizeof(Eigen::Matrix4d),
                                  cudaMemcpyHostToDevice));
    }
#endif

    ipc.buildBVH();
    ipc.init(tet_mesh.meanMass,
             tet_mesh.meanVolum,
             tet_mesh.minConer,
             tet_mesh.maxConer,
             context.linear_system_buff_scale);
    ipc.buildCP();
    ipc._moveDir          = ipc.pcg_data.dx;
    ipc.animation_subRate = 1.0 / context.motion_rate;
    ipc.computeXTilta(device_tet_mesh, 1);
    ipc.create_LinearSystem(device_tet_mesh);
}
}  // namespace

void initialize_cuda(int device_id)
{
    cudaError_t cuda_status = cudaSetDevice(device_id);
    if(cuda_status != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        throw std::runtime_error("failed to initialize CUDA device");
    }
}

void apply_runtime_backend_config(SimulationContext& context,
                                  const gipc::RuntimeBackendConfig& config)
{
    context.ipc.runtime_backend_config = config;
    context.ipc.pcg_data.P_type        = 1;

    // Set static pointer so tetrahedra_obj::getVertNeighbors() can access runtime config
    tetrahedra_obj::s_runtime_backend_config = &context.ipc.runtime_backend_config;
}

void bootstrap_simulation(SimulationContext& context,
                          const SimulationBootstrapOptions& options)
{
    std::filesystem::exists(context.metis_dir)
        || std::filesystem::create_directory(context.metis_dir);

    apply_runtime_backend_config(context, options.runtime_backend_config);

    auto settings_path = options.settings_path.empty()
                             ? context.assets_dir + "scene/parameterSetting.txt"
                             : options.settings_path;
    auto settings = load_simulation_settings(settings_path);
    apply_simulation_settings(context.ipc,
                              context.collision_detection_buff_scale,
                              context.motion_rate,
                              settings);

    context.ipc.build_gipc_system(context.device_tet_mesh);

    SceneSetupContext scene_context{context.ipc,
                                    context.device_tet_mesh,
                                    context.tet_mesh,
                                    context.assets_dir,
                                    context.linear_system_buff_scale};

    if(!options.manifest_path.empty())
        configure_external_manifest_scene(options.manifest_path, scene_context);
    else if(!options.dataset.empty() && !options.task_id.empty())
        configure_local_dataset_scene(
            options.dataset, options.task_id, options.asset_root, scene_context);
    else
        configure_builtin_scene(options.scene_name, scene_context);

    set_mas_partition(context.tet_mesh);
    context.tet_mesh.getSurface();
    init_fem(context.ipc, context.tet_mesh);
    upload_host_mesh_to_device(context);
}
}  // namespace app::common
