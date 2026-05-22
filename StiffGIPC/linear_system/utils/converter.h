#pragma once
#include <cuda_runtime.h>
#include <gipc/type_define.h>
#include <muda/buffer/device_buffer.h>
#include <muda/buffer/device_var.h>
#include <muda/ext/linear_system/device_bcoo_matrix.h>
#include "linear_system/linear_system/global_matrix.h"
namespace gipc
{
class Converter
{
    using T                = Float;
    constexpr static int N = 3;
    using BlockMatrix      = Eigen::Matrix<T, N, N>;

    muda::DeviceBuffer<int>      offsets;
    muda::DeviceVar<int>         count;
    muda::DeviceBuffer<int>      sort_index;
    muda::DeviceBuffer<int>      sort_index_input;
    muda::DeviceBuffer<int2>     ij_pairs;
    muda::DeviceBuffer<uint64_t> ij_hash;
    muda::DeviceBuffer<uint64_t> ij_hash_input;
    muda::DeviceBuffer<BlockMatrix> blocks_sorted;

    template <typename BufferT>
    void loose_resize(BufferT& buf, size_t new_size)
    {
        if(buf.capacity() < new_size)
            buf.reserve(static_cast<size_t>(new_size * 1.5) + 1);
        buf.resize(new_size);
    }

  public:
    // Triplet -> BCOO
    void convert(GIPCTripletMatrix& global_triplets,
                 const int&         start,
                 const int&         length,
                 const int&         out_start_id);

    void srbk_convert(GIPCTripletMatrix& global_triplets,
                      const int&         start,
                      const int&         length,
                      const int&         out_start_id);

    void legacy_gipc_convert(GIPCTripletMatrix&                global_triplets,
                             muda::DeviceBCOOMatrix<Float, 3>& symmetric_bcoo,
                             muda::DeviceBCOOMatrix<Float, 3>& legacy_bcoo);


    void _radix_sort_indices_and_blocks(GIPCTripletMatrix& global_triplets,
                                        const int&         start,
                                        const int&         length,
                                        const int&         out_start_id);


    void _make_unique_indices(GIPCTripletMatrix& global_triplets,
                              const int&         start,
                              const int&         length,
                              const int&         out_start_id);


    void _make_unique_block_warp_reduction(GIPCTripletMatrix& global_triplets,
                                           const int&         start,
                                           const int&         length,
                                           const int&         out_start_id);


    void ge2sym(GIPCTripletMatrix& global_triplets);

    void _radix_sort_indices_and_blocks(muda::DeviceBCOOMatrix<T, N>& to);

    void sym2ge(const muda::DeviceBCOOMatrix<T, N>& from,
                muda::DeviceBCOOMatrix<T, N>&       to);
};
}  // namespace gipc
