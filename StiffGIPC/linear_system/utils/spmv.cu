#include <linear_system/utils/spmv.h>
#include <muda/launch/launch.h>
#include <cub/warp/warp_reduce.cuh>
#include <cub/block/block_shuffle.cuh>
#include <cub/util_math.cuh>

namespace gipc
{

__host__ __device__ constexpr int b2i(bool b)
{
    return b ? 1 : 0;
}

struct Flags
{
    union
    {
        struct
        {
            unsigned char is_head;
            unsigned char is_cross_warp;
            unsigned char is_valid;
        };
        unsigned int flags;
    };

    __host__ __device__ void normalize()
    {
        is_head       = is_head ? 1 : 0;
        is_cross_warp = is_cross_warp ? 1 : 0;
        is_valid      = is_valid ? 1 : 0;
    }
};

__device__ __forceinline__ unsigned fns(unsigned mask, unsigned base, int offset)
{
    return __fns(mask, base, offset);
}

void Spmv::legacy_gipc_spmv(Float                           a,
                            muda::CBCOOMatrixView<Float, 3> A,
                            muda::CDenseVectorView<Float>   x,
                            Float                           b,
                            muda::DenseVectorView<Float>    y)
{
    using namespace muda;
    constexpr int N = 3;
    using T         = Float;

    if(b != 0)
    {
        muda::ParallelFor()
            .kernel_name(__FUNCTION__)
            .apply(y.size(),
                   [b = b, y = y.viewer().name("y")] __device__(int i) mutable
                   { y(i) = b * y(i); });
    }
    else
    {
        muda::BufferLaunch().fill<Float>(y.buffer_view(), 0);
    }

    constexpr int          warp_size = 32;
    constexpr unsigned int warp_mask = ~0u;
    constexpr int          block_dim = 128;
    int block_count = (A.triplet_count() + block_dim - 1) / block_dim;

    muda::Launch(block_count, block_dim)
        .kernel_name(__FUNCTION__)
        .apply(
            [a = a,
             A = A.viewer().name("A"),
             x = x.viewer().name("x"),
             y = y.viewer().name("y")] __device__() mutable
            {
                using WarpReduceInt   = cub::WarpReduce<int, warp_size>;
                using WarpReduceFloat = cub::WarpReduce<Float, warp_size>;

                auto global_thread_id   = blockDim.x * blockIdx.x + threadIdx.x;
                auto thread_id_in_block = threadIdx.x;
                auto warp_id            = thread_id_in_block / warp_size;
                auto lane_id            = thread_id_in_block & (warp_size - 1);

                __shared__ union
                {
                    typename WarpReduceInt::TempStorage temp_storage_int[block_dim / warp_size];
                    typename WarpReduceFloat::TempStorage temp_storage_float[block_dim / warp_size];
                };

                int prev_i = -1;
                int next_i = -1;
                int i      = -1;

                Flags   flags;
                Vector3 vec;
                flags.is_cross_warp = 0;

                if(global_thread_id > 0 && global_thread_id < A.triplet_count())
                {
                    auto prev_triplet = A(global_thread_id - 1);
                    prev_i            = prev_triplet.block_row_index;
                }

                if(global_thread_id < A.triplet_count() - 1)
                {
                    auto next_triplet = A(global_thread_id + 1);
                    next_i            = next_triplet.block_row_index;
                }

                if(global_thread_id < A.triplet_count())
                {
                    auto triplet = A(global_thread_id);
                    i            = triplet.block_row_index;
                    auto j       = triplet.block_col_index;
                    vec          = triplet.block_value * x.segment<N>(j * N).as_eigen();
                    flags.is_valid = 1;
                }
                else
                {
                    i = -1;
                    vec.setZero();
                    flags.is_valid      = 0;
                    flags.is_cross_warp = 0;
                }

                if(lane_id == 0)
                {
                    flags.is_head       = 1;
                    flags.is_cross_warp = b2i(prev_i == i);
                }
                else
                {
                    flags.is_head = b2i(prev_i != i);
                    if(lane_id == warp_size - 1)
                        flags.is_cross_warp = b2i(next_i == i);
                }

                flags.flags = WarpReduceInt(temp_storage_int[warp_id])
                                  .HeadSegmentedReduce(flags.flags,
                                                       flags.is_head,
                                                       [](uint32_t lhs, uint32_t rhs)
                                                       { return lhs + rhs; });

                vec.x() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.x(),
                                                   flags.is_head,
                                                   [](Float lhs, Float rhs)
                                                   { return lhs + rhs; });
                vec.y() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.y(),
                                                   flags.is_head,
                                                   [](Float lhs, Float rhs)
                                                   { return lhs + rhs; });
                vec.z() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.z(),
                                                   flags.is_head,
                                                   [](Float lhs, Float rhs)
                                                   { return lhs + rhs; });

                flags.is_head = b2i(flags.is_head && flags.is_valid);
                flags.normalize();

                int      is_head_mask = __ballot_sync(warp_mask, flags.is_head);
                uint32_t offset       = fns(is_head_mask, 0, lane_id + 1);
                int      valid_bit    = (offset != ~0u);
                int      shuffle_mask = __ballot_sync(warp_mask, valid_bit);

                i           = cub::ShuffleIndex<32>(i, offset, shuffle_mask);
                flags.flags = cub::ShuffleIndex<32>(flags.flags, offset, shuffle_mask);
                vec.x()     = cub::ShuffleIndex<32>(vec.x(), offset, shuffle_mask);
                vec.y()     = cub::ShuffleIndex<32>(vec.y(), offset, shuffle_mask);
                vec.z()     = cub::ShuffleIndex<32>(vec.z(), offset, shuffle_mask);

                if(valid_bit && flags.is_head && flags.is_valid)
                {
                    auto seg_y  = y.segment<N>(i * N);
                    auto result = a * vec;

                    if(flags.is_cross_warp)
                        seg_y.atomic_add(result.eval());
                    else
                        seg_y.as_eigen() += result.eval();
                }
            });
}

void Spmv::srbk_spmv(Float                         a,
                     Eigen::Matrix3d*              triplet_values,
                     int*                          row_ids,
                     int*                          col_ids,
                     int                           triplet_count,
                     muda::CDenseVectorView<Float> x,
                     Float                         b,
                     muda::DenseVectorView<Float>  y)

{
    using namespace muda;
    constexpr int N = 3;
    using T         = Float;

    if(b != 0)
    {
        muda::ParallelFor()
            .kernel_name(__FUNCTION__)
            .apply(y.size(),
                   [b = b, y = y.viewer().name("y")] __device__(int i) mutable
                   { y(i) = b * y(i); });
    }
    else
    {
        muda::BufferLaunch().fill<Float>(y.buffer_view(), 0);
    }

    constexpr int          warp_size = 32;
    constexpr unsigned int warp_mask = ~0u;
    constexpr int          block_dim = 256;
    int block_count = (triplet_count + block_dim - 1) / block_dim;

    muda::Launch(block_count, block_dim)
        .kernel_name(__FUNCTION__)
        .apply(
            [a     = a,
             Mats3 = triplet_values,
             rows  = row_ids,
             cols  = col_ids,
             triplet_count,
             x = x.viewer().name("x"),
             b = b,
             y = y.viewer().name("y")] __device__() mutable
            {
                using WarpReduceFloat = cub::WarpReduce<Float, warp_size>;
                auto global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
                if(global_thread_id >= triplet_count)
                    return;
                auto thread_id_in_block = threadIdx.x;
                auto warp_id            = thread_id_in_block / warp_size;
                auto lane_id            = thread_id_in_block & (warp_size - 1);

                __shared__ WarpReduceFloat::TempStorage temp_storage_float[block_dim / warp_size];

                int     prev_i = -1;
                int     i      = -1;
                char    flags;
                Vector3 vec;

                // set the previous row index
                if(global_thread_id > 0)
                {
                    //auto prev_triplet = A(global_thread_id - 1);
                    prev_i = rows[global_thread_id - 1];
                }


                {
                    //auto Triplet = Mats3[];
                    i                = rows[global_thread_id];
                    auto j           = cols[global_thread_id];
                    auto block_value = Mats3[global_thread_id];
                    vec = block_value * x.segment<N>(j * N).as_eigen();

                    //flags.is_valid = 1;

                    if(i != j)  // process lower triangle
                    {
                        Vector3 vec_ = a * block_value.transpose()
                                       * x.segment<N>(i * N).as_eigen();

                        y.segment<N>(j * N).atomic_add(vec_);
                    }
                }


                if((lane_id == 0) || (prev_i != i))
                {
                    flags = 1;
                }
                else
                {
                    flags = 0;
                }

                vec.x() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.x(),
                                                   flags,
                                                   [](Float a, Float b)
                                                   { return a + b; });

                vec.y() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.y(),
                                                   flags,
                                                   [](Float a, Float b)
                                                   { return a + b; });

                vec.z() = WarpReduceFloat(temp_storage_float[warp_id])
                              .HeadSegmentedReduce(vec.z(),
                                                   flags,
                                                   [](Float a, Float b)
                                                   { return a + b; });
                // ----------------------------------- warp reduce -----------------------------------------------


                if(flags)
                {
                    auto seg_y  = y.segment<N>(i * N);
                    auto result = a * vec;

                    // Must use atomic add!
                    // Because the same row may be processed by different warps
                    seg_y.atomic_add(result.eval());
                }
            });
}
}  // namespace gipc
