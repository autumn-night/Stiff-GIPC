//
// device_fem_data.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "device_fem_data.cuh"
#include "cuda_tools/cuda_tools.h"


void device_TetraData::Malloc_DEVICE_MEM(const int& vertex_num,
                                         const int& tetradedra_num,
                                         const int& triangle_num,
                                         const int& softNum,
                                         const int& tri_edgeNum,
                                         const int& bodyNum)
{
    m_vertex_num   = vertex_num;
    int maxNumbers = vertex_num > tetradedra_num ? vertex_num : tetradedra_num;
    CUDA_SAFE_CALL(cudaMalloc((void**)&vertexes, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&o_vertexes, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&velocities, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&rest_vertexes, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&temp_double3Mem, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&xTilta, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&fb, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&totalForce, vertex_num * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&shape_grads, vertex_num * sizeof(double3)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&tetrahedras, tetradedra_num * sizeof(uint4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&tempTetrahedras, tetradedra_num * sizeof(uint4)));


    CUDA_SAFE_CALL(cudaMalloc((void**)&tri_edges, tri_edgeNum * sizeof(uint2)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&tri_edge_adj_vertex, tri_edgeNum * sizeof(uint2)));

#ifdef USE_QUADRATIC_BENDING
    CUDA_SAFE_CALL(cudaMalloc((void**)&quad_bending_Q, tri_edgeNum * sizeof(Eigen::Matrix4d)));
#endif

    CUDA_SAFE_CALL(cudaMalloc((void**)&volum, tetradedra_num * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&masses, vertex_num * sizeof(double)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&lengthRate, tetradedra_num * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&volumeRate, tetradedra_num * sizeof(double)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&apply_gravity, vertex_num * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&tempDouble, maxNumbers * sizeof(double)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&BoundaryType, vertex_num * sizeof(int)));

    CUDA_SAFE_CALL(cudaMemset(BoundaryType, 0, vertex_num * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&DmInverses,
                              tetradedra_num * sizeof(__GEIGEN__::Matrix3x3d)));

    m_soft_num = softNum;
    CUDA_SAFE_CALL(cudaMalloc((void**)&targetIndex, softNum * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&targetVert, softNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&triDmInverses,
                              triangle_num * sizeof(__GEIGEN__::Matrix2x2d)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&area, triangle_num * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&triangles, triangle_num * sizeof(uint4)));

    
    CUDA_SAFE_CALL(cudaMalloc((void**)&body_id_to_boundary_type, bodyNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&point_id_to_body_id, vertex_num * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&tet_id_to_body_id, tetradedra_num * sizeof(int)));
}

device_TetraData::~device_TetraData()
{
    FREE_DEVICE_MEM();
}

void device_TetraData::FREE_DEVICE_MEM()
{
    if(vertexes)
        CUDA_SAFE_CALL(cudaFree(vertexes));
    if(o_vertexes)
        CUDA_SAFE_CALL(cudaFree(o_vertexes));
    if(temp_double3Mem)
        CUDA_SAFE_CALL(cudaFree(temp_double3Mem));
    if(velocities)
        CUDA_SAFE_CALL(cudaFree(velocities));
    if(rest_vertexes)
        CUDA_SAFE_CALL(cudaFree(rest_vertexes));
    if(xTilta)
        CUDA_SAFE_CALL(cudaFree(xTilta));
    if(fb)
        CUDA_SAFE_CALL(cudaFree(fb));
    if(apply_gravity)
        CUDA_SAFE_CALL(cudaFree(apply_gravity));
    if(shape_grads)
        CUDA_SAFE_CALL(cudaFree(shape_grads));
    if(tetrahedras)
        CUDA_SAFE_CALL(cudaFree(tetrahedras));
    if(tempTetrahedras)
        CUDA_SAFE_CALL(cudaFree(tempTetrahedras));
    if(volum)
        CUDA_SAFE_CALL(cudaFree(volum));
    if(masses)
        CUDA_SAFE_CALL(cudaFree(masses));
    if(lengthRate)
        CUDA_SAFE_CALL(cudaFree(lengthRate));
    if(volumeRate)
        CUDA_SAFE_CALL(cudaFree(volumeRate));
    if(DmInverses)
        CUDA_SAFE_CALL(cudaFree(DmInverses));
    if(tempDouble)
        CUDA_SAFE_CALL(cudaFree(tempDouble));
    if(BoundaryType)
        CUDA_SAFE_CALL(cudaFree(BoundaryType));

    if(totalForce)
        CUDA_SAFE_CALL(cudaFree(totalForce));
    if(targetIndex)
        CUDA_SAFE_CALL(cudaFree(targetIndex));
    if(targetVert)
        CUDA_SAFE_CALL(cudaFree(targetVert));
    if(triDmInverses)
        CUDA_SAFE_CALL(cudaFree(triDmInverses));
    if(area)
        CUDA_SAFE_CALL(cudaFree(area));
    if(triangles)
        CUDA_SAFE_CALL(cudaFree(triangles));

    if(tri_edges)
        CUDA_SAFE_CALL(cudaFree(tri_edges));
    if(tri_edge_adj_vertex)
        CUDA_SAFE_CALL(cudaFree(tri_edge_adj_vertex));

#ifdef USE_QUADRATIC_BENDING
    if(quad_bending_Q)
        CUDA_SAFE_CALL(cudaFree(quad_bending_Q));
#endif

    if(body_id_to_boundary_type)
        CUDA_SAFE_CALL(cudaFree(body_id_to_boundary_type));
    if(point_id_to_body_id)
        CUDA_SAFE_CALL(cudaFree(point_id_to_body_id));
    if(tet_id_to_body_id)
        CUDA_SAFE_CALL(cudaFree(tet_id_to_body_id));

    // Reset all pointers to nullptr
    vertexes                  = nullptr;
    o_vertexes                = nullptr;
    temp_double3Mem           = nullptr;
    velocities                = nullptr;
    rest_vertexes             = nullptr;
    xTilta                    = nullptr;
    fb                        = nullptr;
    apply_gravity             = nullptr;
    shape_grads               = nullptr;
    tetrahedras               = nullptr;
    tempTetrahedras           = nullptr;
    volum                     = nullptr;
    masses                    = nullptr;
    lengthRate                = nullptr;
    volumeRate                = nullptr;
    DmInverses                = nullptr;
    tempDouble                = nullptr;
    BoundaryType              = nullptr;
    totalForce                = nullptr;
    targetIndex               = nullptr;
    targetVert                = nullptr;
    triDmInverses             = nullptr;
    area                      = nullptr;
    triangles                 = nullptr;
    tri_edges                 = nullptr;
    tri_edge_adj_vertex       = nullptr;
#ifdef USE_QUADRATIC_BENDING
    quad_bending_Q            = nullptr;
#endif
    body_id_to_boundary_type  = nullptr;
    point_id_to_body_id       = nullptr;
    tet_id_to_body_id         = nullptr;
}

void device_TetraData::update_soft_constraint_target_position(int step_id, double ipc_dt)
{
    if(m_soft_num < 1)
        return;

    std::vector<double3> host_vertexes(m_vertex_num);
    CUDA_SAFE_CALL(cudaMemcpy(
        host_vertexes.data(), vertexes, m_vertex_num * sizeof(double3), cudaMemcpyDeviceToHost));

    for(int i = 0; i < m_soft_num; i++)
    {
        if(update_soft_constraint_functor == nullptr)
            host_target_vertices[i] = host_vertexes[host_target_indices[i]];
        else
            host_target_vertices[i] = update_soft_constraint_functor(
                host_vertexes[host_target_indices[i]], step_id, ipc_dt);
    }

    CUDA_SAFE_CALL(cudaMemcpy(targetVert,
                              host_target_vertices.data(),
                              m_soft_num * sizeof(double3),
                              cudaMemcpyHostToDevice));
}
