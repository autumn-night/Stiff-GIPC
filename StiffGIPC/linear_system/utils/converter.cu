#include <linear_system/utils/converter.h>
#include <muda/cub/device/device_run_length_encode.h>
#include <muda/cub/device/device_scan.h>
#include <muda/cub/device/device_radix_sort.h>
#include <muda/cub/device/device_partition.h>
#include <gipc/utils/timer.h>
#include <gipc/utils/parallel_algorithm/fast_segmental_reduce.h>

namespace gipc
{

template <typename T>
__global__ inline void moveMemory_2(T* data, int output_start, int input_start, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= length)
        return;
    data[output_start + idx] = data[input_start + idx];
}

constexpr bool UseRadixSort   = true;
constexpr bool UseReduceByKey = false;

void Converter::convert(GIPCTripletMatrix& global_triplets,
                        const int&         start,
                        const int&         length,
                        const int&         out_start_id)
{
    srbk_convert(global_triplets, start, length, out_start_id);
}

void Converter::srbk_convert(GIPCTripletMatrix& global_triplets,
                             const int&         start,
                             const int&         length,
                             const int&         out_start_id)
{
    gipc::Timer timer("convert3x3");
    if(length < 1)
        return;
    _radix_sort_indices_and_blocks(global_triplets, start, length, out_start_id);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());


    //_make_unique_indices(global_triplets, start, length, out_start_id);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());


    _make_unique_block_warp_reduction(global_triplets, start, length, out_start_id);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

void Converter::legacy_gipc_convert(GIPCTripletMatrix&                global_triplets,
                                    muda::DeviceBCOOMatrix<Float, 3>& symmetric_bcoo,
                                    muda::DeviceBCOOMatrix<Float, 3>& legacy_bcoo)
{
    const auto triplet_count = global_triplets.h_unique_key_number;

    symmetric_bcoo.reshape(global_triplets.block_rows(), global_triplets.block_cols());
    symmetric_bcoo.resize_triplets(triplet_count);

    if(triplet_count == 0)
    {
        legacy_bcoo.reshape(global_triplets.block_rows(), global_triplets.block_cols());
        legacy_bcoo.resize_triplets(0);
        return;
    }

    auto symmetric_view = symmetric_bcoo.view();
    CUDA_SAFE_CALL(cudaMemcpy(symmetric_view.block_row_indices(),
                              global_triplets.block_row_indices(),
                              triplet_count * sizeof(int),
                              cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(symmetric_view.block_col_indices(),
                              global_triplets.block_col_indices(),
                              triplet_count * sizeof(int),
                              cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(symmetric_view.block_values(),
                              global_triplets.block_values(),
                              triplet_count * sizeof(BlockMatrix),
                              cudaMemcpyDeviceToDevice));

    sym2ge(symmetric_bcoo, legacy_bcoo);
}



void Converter::_radix_sort_indices_and_blocks(GIPCTripletMatrix& global_triplets,
                                               const int& start,
                                               const int& length,
                                               const int& out_start_id)
{
    using namespace muda;

    auto src_row_indices = global_triplets.block_row_indices(start);
    auto src_col_indices = global_triplets.block_col_indices(start);
    auto src_blocks      = global_triplets.block_values(start);
    auto index_input   = global_triplets.block_index();
    auto ij_hash_input = global_triplets.block_hash_value();

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(length,
               [row_indices = src_row_indices,
                col_indices = src_col_indices,
                ij_hash_input,
                index_input] __device__(int i) mutable
               {
                   ij_hash_input[i] =
                       (uint64_t{row_indices[i]} << 32) + uint64_t{col_indices[i]};
                   index_input[i] = i;
               });

    DeviceRadixSort().SortPairs(ij_hash_input,
                                global_triplets.block_sort_hash_value(),
                                index_input,
                                global_triplets.block_sort_index(),
                                length);

    auto dst_val = global_triplets.block_values() + out_start_id;
    ParallelFor(256)
        .kernel_name("set col row indices")
        .apply(length,
               [sort_index = global_triplets.block_sort_index(),
                src_blocks,
                dst_val] __device__(int i) mutable
               {
                   dst_val[i] = src_blocks[sort_index[i]];

               });
}


void Converter::_make_unique_indices(GIPCTripletMatrix& global_triplets,
                                     const int&         start,
                                     const int&         length,
                                     const int&         out_start_id)
{
    auto row_indices = global_triplets.block_row_indices(start);
    auto col_indices = global_triplets.block_col_indices(start);

    auto unique_key = global_triplets.block_hash_value();
    auto sort_key   = global_triplets.block_sort_hash_value();

    muda::DeviceRunLengthEncode().Encode(sort_key,
                                         unique_key,
                                         global_triplets.block_temp_buffer(),
                                         global_triplets.d_unique_key_number,
                                         length);

    CUDA_SAFE_CALL(cudaMemcpy(&(global_triplets.h_unique_key_number),
                              global_triplets.d_unique_key_number,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));

    muda::ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(global_triplets.h_unique_key_number,

               [row_indices, col_indices, unique_key] __device__(int i) mutable
               {
                   row_indices[i] = unique_key[i] >> 32;
                   col_indices[i] = unique_key[i] & 0xffffffff;
               });
}





void Converter::_make_unique_block_warp_reduction(GIPCTripletMatrix& global_triplets,
                                                  const int& start, const int& length, const int& out_start_id)
{
    using namespace muda;

    auto sorted_partition_input = global_triplets.block_temp_buffer();
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(length - 1,
               [sorted_partition_input,
                ij_hash = global_triplets.block_sort_hash_value()] __device__(int i) mutable
               {
                   sorted_partition_input[i] = ij_hash[i] != ij_hash[i + 1] ? 1 : 0;
               });
    auto sorted_partition_output = global_triplets.block_index();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // scatter
    DeviceScan().ExclusiveSum(sorted_partition_input, sorted_partition_output, length);

    auto row_indices = global_triplets.block_row_indices(start);
    auto col_indices = global_triplets.block_col_indices(start);


    muda::ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(length,
               [row_indices,
                col_indices,
                ij_hash = global_triplets.block_sort_hash_value(),
                sorted_partition_output] __device__(int i) mutable
               {
                   int index = sorted_partition_output[i];
                   if(i == 0)
                   {

                       auto key           = ij_hash[i];
                       row_indices[index] = key >> 32;
                       col_indices[index] = key & 0xffffffff;
                   }
                   else
                   {
                       if(index != sorted_partition_output[i - 1])
                       {
                           auto key           = ij_hash[i];
                           row_indices[index] = key >> 32;
                           col_indices[index] = key & 0xffffffff;
                       }
                   }
               });


    CUDA_SAFE_CALL(cudaMemcpy(&(global_triplets.h_unique_key_number),
                              sorted_partition_output + length - 1,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));
    global_triplets.h_unique_key_number += 1;

    CUDA_SAFE_CALL(cudaMemset(global_triplets.block_values(start),
                              0,
                              global_triplets.h_unique_key_number * sizeof(Eigen::Matrix3d)));

    FastSegmentalReduce()
        .kernel_name(__FUNCTION__)
        .reduce(length,
                sorted_partition_output,
                global_triplets.block_values(out_start_id),
                global_triplets.block_values(start));
}

void Converter::ge2sym(GIPCTripletMatrix& global_triplets)
{
    using namespace muda;

    auto counts  = global_triplets.block_index();
    auto offsets = global_triplets.block_sort_index();
    auto block_temp = global_triplets.block_values(global_triplets.h_unique_key_number);
    auto blocks      = global_triplets.block_values();
    auto ij_hash     = global_triplets.block_hash_value();
    auto row_indices = global_triplets.block_row_indices();
    auto col_indices = global_triplets.block_col_indices();

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(global_triplets.h_unique_key_number,
               [row_indices, col_indices, ij_hash, blocks, block_temp, counts] __device__(int i) mutable
               {
                   counts[i] = row_indices[i] <= col_indices[i] ? 1 : 0;
                   ij_hash[i] =
                       (uint64_t{row_indices[i]} << 32) + uint64_t{col_indices[i]};
                   block_temp[i] = blocks[i];
               });

    // exclusive sum
    DeviceScan().ExclusiveSum(counts, offsets, global_triplets.h_unique_key_number);

    // set the values
    auto dst_blocks = global_triplets.block_values();

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(global_triplets.h_unique_key_number,
               [dst_blocks,
                block_temp,
                ij_hash,
                row_indices,
                col_indices,
                counts,
                offsets,
                total_count = global_triplets.d_unique_key_number,
                number = global_triplets.h_unique_key_number] __device__(int i) mutable
               {
                   auto count  = counts[i];
                   auto offset = offsets[i];

                   if(count != 0)
                   {
                       dst_blocks[offset]  = block_temp[i];
                       auto ij             = ij_hash[i];
                       row_indices[offset] = ij >> 32;
                       col_indices[offset] = ij & 0xffffffff;
                   }

                   if(i == number - 1)
                   {
                       *total_count = offsets[i] + counts[i];
                   }
               });


    CUDA_SAFE_CALL(cudaMemcpy(&(global_triplets.h_unique_key_number),
                              global_triplets.d_unique_key_number,
                              sizeof(int),
                              cudaMemcpyDeviceToHost));
}

void Converter::_radix_sort_indices_and_blocks(muda::DeviceBCOOMatrix<T, N>& to)
{
    using namespace muda;

    auto src_row_indices = to.block_row_indices();
    auto src_col_indices = to.block_col_indices();
    auto src_blocks      = to.block_values();

    loose_resize(ij_hash_input, src_row_indices.size());
    loose_resize(sort_index_input, src_row_indices.size());

    loose_resize(ij_hash, src_row_indices.size());
    loose_resize(sort_index, src_row_indices.size());
    loose_resize(ij_pairs, src_row_indices.size());

    ParallelFor(256)
        .file_line(__FILE__, __LINE__)
        .apply(src_row_indices.size(),
               [row_indices = src_row_indices.cviewer().name("row_indices"),
                col_indices = src_col_indices.cviewer().name("col_indices"),
                ij_hash     = ij_hash_input.viewer().name("ij_hash"),
                sort_index = sort_index_input.viewer().name("sort_index")] __device__(int i) mutable
               {
                   ij_hash(i) =
                       (uint64_t{row_indices(i)} << 32) + uint64_t{col_indices(i)};
                   sort_index(i) = i;
               });

    DeviceRadixSort().SortPairs(ij_hash_input.data(),
                                ij_hash.data(),
                                sort_index_input.data(),
                                sort_index.data(),
                                ij_hash.size());

    ParallelFor(256)
        .kernel_name("set col row indices")
        .apply(src_row_indices.size(),
               [ij_hash = ij_hash.viewer().name("ij_hash"),
                ij_pairs = ij_pairs.viewer().name("ij_pairs")] __device__(int i) mutable
               {
                   auto hash      = ij_hash(i);
                   auto row_index = int{hash >> 32};
                   auto col_index = int{hash & 0xFFFFFFFF};
                   ij_pairs(i).x  = row_index;
                   ij_pairs(i).y  = col_index;
               });

    loose_resize(blocks_sorted, src_blocks.size());
    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(src_blocks.size(),
               [src_blocks = src_blocks.cviewer().name("blocks"),
                sort_index = sort_index.cviewer().name("sort_index"),
                ij_pairs   = ij_pairs.cviewer().name("ij_pairs"),
                dst_row = to.block_row_indices().viewer().name("row_indices"),
                dst_col = to.block_col_indices().viewer().name("col_indices"),
                dst_blocks = blocks_sorted.viewer().name("block_values")] __device__(int i) mutable
               {
                   dst_blocks(i) = src_blocks(sort_index(i));
                   dst_row(i)    = ij_pairs(i).x;
                   dst_col(i)    = ij_pairs(i).y;
               });

    to.block_values().copy_from(blocks_sorted);
}

void Converter::sym2ge(const muda::DeviceBCOOMatrix<T, N>& from,
                       muda::DeviceBCOOMatrix<T, N>&       to)
{
    using namespace muda;

    auto sym_size   = from.non_zero_blocks();
    auto diag_count = from.block_rows();

    auto& flags                 = offsets;
    auto& partitioned           = blocks_sorted;
    auto& partition_index_input = sort_index_input;
    auto& partition_index       = sort_index;
    auto& selected_count        = count;

    loose_resize(flags, sym_size);
    loose_resize(partitioned, sym_size);
    loose_resize(partition_index_input, sym_size);
    loose_resize(partition_index, sym_size);

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(sym_size,
               [flags = flags.viewer().name("flags"),
                row_indices = from.block_row_indices().cviewer().name("row_indices"),
                col_indices = from.block_col_indices().cviewer().name("col_indices"),
                partition_index = partition_index_input.viewer().name("partitioned")] __device__(int i) mutable
               {
                   flags(i) = (row_indices(i) == col_indices(i)) ? 1 : 0;
                   partition_index(i) = i;
               });

    muda::DevicePartition().Flagged(partition_index_input.data(),
                                    flags.data(),
                                    partition_index.data(),
                                    selected_count.data(),
                                    sym_size);

    auto general_bcoo_size = 2 * (sym_size - diag_count) + diag_count;
    to.resize(from.block_rows(), from.block_cols(), general_bcoo_size);

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(sym_size,
               [to   = to.viewer().name("to"),
                from = from.cviewer().name("from"),
                partition_index = partition_index.cviewer().name("partition_index"),
                diag_count = diag_count,
                sym_size   = sym_size] __device__(int i) mutable
               {
                   auto index = partition_index(i);
                   auto f     = from(index);
                   to(i).write(f.block_row_index, f.block_col_index, f.block_value);
                   if(i >= diag_count)
                   {
                       to(i + sym_size - diag_count)
                           .write(f.block_col_index,
                                  f.block_row_index,
                                  f.block_value.transpose());
                   }
               });

    _radix_sort_indices_and_blocks(to);
}

}  // namespace gipc
