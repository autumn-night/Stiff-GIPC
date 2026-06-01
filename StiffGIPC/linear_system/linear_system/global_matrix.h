#pragma once

#include"cuda_tools/cuda_device_buffer.h"
#include"Eigen/Eigen"

//#define SymGH
#ifdef SymGH
#define M12_Off 10
#define M9_Off 6
#define M6_Off 3
#else
#define M12_Off 16
#define M9_Off 9
#define M6_Off 4
#endif

#ifdef __CUDACC__
static __global__ void _gipc_triplet_matrix_set_hash_value(const int* row_ids,
                                                           const int* col_ids,
                                                           uint32_t*  index,
                                                           uint64_t*  hashValue,
                                                           int        abd_vert_num,
                                                           int        number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    index[idx] = idx;

    uint64_t self_hash;
    if(row_ids[idx] < abd_vert_num && col_ids[idx] < abd_vert_num)
    {
        self_hash = 3;
    }
    else if(row_ids[idx] < abd_vert_num && col_ids[idx] >= abd_vert_num)
    {
        self_hash = 1;
    }
    else if(row_ids[idx] >= abd_vert_num && col_ids[idx] < abd_vert_num)
    {
        self_hash = 2;
    }
    else
    {
        self_hash = 0;
    }
    hashValue[idx] = self_hash;
}
#endif


class GIPCTripletMatrix
{
  public:
    using BlockMatrix = Eigen::Matrix<double, 3, 3>;
    //using EntryValueType = T;
    //int Dimenstion              = M;
  public:
    cudatool::CudaDeviceBuffer<BlockMatrix> m_block_values;
    cudatool::CudaDeviceBuffer<int>         m_block_row_indices;
    cudatool::CudaDeviceBuffer<int>         m_block_col_indices;
    cudatool::CudaDeviceBuffer<uint64_t>    m_block_hash_value;
    cudatool::CudaDeviceBuffer<uint64_t>    m_block_sort_hash_value;
    cudatool::CudaDeviceBuffer<uint32_t>    m_block_index;
    cudatool::CudaDeviceBuffer<uint32_t>    m_block_sort_index;
    cudatool::CudaDeviceBuffer<uint32_t>    m_block_temp_buffer;
    int                                     m_block_rows = 0;
    int                                     m_block_cols = 0;

  public:
    GIPCTripletMatrix()                                    = default;
    ~GIPCTripletMatrix() { free_var(); }
    GIPCTripletMatrix(const GIPCTripletMatrix&)            = delete;
    GIPCTripletMatrix(GIPCTripletMatrix&&)                 = delete;
    GIPCTripletMatrix& operator=(const GIPCTripletMatrix&) = delete;
    GIPCTripletMatrix& operator=(GIPCTripletMatrix&&)      = delete;

    void reshape(int row, int col)
    {
        m_block_rows = row;
        m_block_cols = col;
    }

    void resize_triplets(size_t nonzero_count)
    {
        m_block_values.resize(nonzero_count);
        m_block_row_indices.resize(nonzero_count);
        m_block_col_indices.resize(nonzero_count);
    }

    void reserve_triplets(size_t nonzero_count)
    {
        m_block_values.reserve(nonzero_count);
        m_block_row_indices.reserve(nonzero_count);
        m_block_col_indices.reserve(nonzero_count);
    }

    void resize(int row, int col, size_t nonzero_count)
    {
        reshape(row, col);
        resize_triplets(nonzero_count);
    }

    void resize_collision_hash_size(size_t nonzero_count)
    {
        m_block_hash_value.resize(nonzero_count);
        m_block_sort_hash_value.resize(nonzero_count);
        m_block_index.resize(nonzero_count);
        m_block_sort_index.resize(nonzero_count);
        m_block_temp_buffer.resize(nonzero_count);
    }

    void reset_zero()
    {
        m_block_values.reset_zero();
        m_block_row_indices.reset_zero();
        m_block_col_indices.reset_zero();
    }

    void update_hash_value(int fem_offset);

    auto block_values(int offset = 0) { return m_block_values.data() + offset; }
    auto block_values(int offset = 0) const
    {
        return m_block_values.data() + offset;
    }
    auto block_row_indices(int offset = 0)
    {
        return m_block_row_indices.data() + offset;
    }
    auto block_row_indices(int offset = 0) const
    {
        return m_block_row_indices.data() + offset;
    }
    auto block_col_indices(int offset = 0)
    {
        return m_block_col_indices.data() + offset;
    }
    auto block_col_indices(int offset = 0) const
    {
        return m_block_col_indices.data() + offset;
    }
    auto block_hash_value(int offset = 0)
    {
        return m_block_hash_value.data() + offset;
    }
    auto block_hash_value(int offset = 0) const
    {
        return m_block_hash_value.data() + offset;
    }

    auto block_sort_hash_value(int offset = 0)
    {
        return m_block_sort_hash_value.data() + offset;
    }
    auto block_sort_hash_value(int offset = 0) const
    {
        return m_block_sort_hash_value.data() + offset;
    }

    auto block_temp_buffer(int offset = 0)
    {
        return m_block_temp_buffer.data() + offset;
    }
    auto block_temp_buffer(int offset = 0) const
    {
        return m_block_temp_buffer.data() + offset;
    }

    auto block_index(int offset = 0) { return m_block_index.data() + offset; }
    auto block_index(int offset = 0) const
    {
        return m_block_index.data() + offset;
    }

    auto block_sort_index(int offset = 0)
    {
        return m_block_sort_index.data() + offset;
    }
    auto block_sort_index(int offset = 0) const
    {
        return m_block_sort_index.data() + offset;
    }

    auto block_rows() const { return m_block_rows; }
    auto block_cols() const { return m_block_cols; }
    auto triplet_count() const { return m_block_values.size(); }
    auto triplet_capacity() const { return m_block_values.capacity(); }

    void clear()
    {
        m_block_rows = 0;
        m_block_cols = 0;
        m_block_values.clear();
        m_block_row_indices.clear();
        m_block_col_indices.clear();
    }
    int global_triplet_offset           = 0;
    int global_collision_triplet_offset = 0;
    int global_external_max_capcity     = 0;
    int global_internal_capcity         = 0;

    int* d_abd_abd_contact_start_id = nullptr;
    int* d_abd_fem_contact_start_id = nullptr;
    int* d_fem_abd_contact_start_id = nullptr;
    int* d_fem_fem_contact_start_id = nullptr;
    int* d_unique_key_number        = nullptr;

    void init_var()
    {
        if(!d_abd_abd_contact_start_id)
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_abd_abd_contact_start_id, sizeof(int)));
        if(!d_abd_fem_contact_start_id)
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_abd_fem_contact_start_id, sizeof(int)));
        if(!d_fem_abd_contact_start_id)
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_fem_abd_contact_start_id, sizeof(int)));
        if(!d_fem_fem_contact_start_id)
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_fem_fem_contact_start_id, sizeof(int)));
        if(!d_unique_key_number)
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_unique_key_number, sizeof(int)));
    }

    void free_var()
    {
        if(d_abd_abd_contact_start_id)
            CUDA_SAFE_CALL(cudaFree(d_abd_abd_contact_start_id));
        if(d_abd_fem_contact_start_id)
            CUDA_SAFE_CALL(cudaFree(d_abd_fem_contact_start_id));
        if(d_fem_abd_contact_start_id)
            CUDA_SAFE_CALL(cudaFree(d_fem_abd_contact_start_id));
        if(d_fem_fem_contact_start_id)
            CUDA_SAFE_CALL(cudaFree(d_fem_fem_contact_start_id));
        if(d_unique_key_number)
            CUDA_SAFE_CALL(cudaFree(d_unique_key_number));

        d_abd_abd_contact_start_id = nullptr;
        d_abd_fem_contact_start_id = nullptr;
        d_fem_abd_contact_start_id = nullptr;
        d_fem_fem_contact_start_id = nullptr;
        d_unique_key_number        = nullptr;
    }

    int h_abd_abd_contact_start_id = -1;
    int h_abd_fem_contact_start_id = -1;
    int h_fem_abd_contact_start_id = -1;
    int h_fem_fem_contact_start_id = -1;
    int h_unique_key_number        = 0;

    uint32_t abd_abd_contact_num = 0;
    uint32_t abd_fem_contact_num = 0;
    uint32_t fem_fem_contact_num = 0;
    uint32_t fem_abd_contact_num = 0;
};

#ifdef __CUDACC__
inline void GIPCTripletMatrix::update_hash_value(int fem_offset)
{
    int threadNum = 256;
    int blockNum  = (global_collision_triplet_offset + threadNum - 1) / threadNum;

    if(global_collision_triplet_offset > global_external_max_capcity)
    {
        global_external_max_capcity = global_collision_triplet_offset;
        resize_collision_hash_size(global_collision_triplet_offset);
    }

    LaunchCudaKernal(blockNum,
                     threadNum,
                     0,
                     _gipc_triplet_matrix_set_hash_value,
                     (const int*)m_block_row_indices.data(),
                     (const int*)m_block_col_indices.data(),
                     m_block_index.data(),
                     m_block_hash_value.data(),
                     fem_offset,
                     global_collision_triplet_offset);
}
#endif
