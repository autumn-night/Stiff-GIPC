#pragma once
#include <gipc/type_define.h>

#include <muda/ext/linear_system/bcoo_matrix_view.h>
#include <muda/ext/linear_system/dense_vector_view.h>
#include <muda/ext/linear_system/device_dense_vector.h>

namespace gipc
{
class Spmv
{
  public:
    void legacy_gipc_spmv(Float                           a,
                          muda::CBCOOMatrixView<Float, 3> A,
                          muda::CDenseVectorView<Float>   x,
                          Float                           b,
                          muda::DenseVectorView<Float>    y);

    void srbk_spmv(Float                         a,
                   Eigen::Matrix3d*              triplet_values,
                   int*                          row_ids,
                   int*                          col_ids,
                   int                           triplet_count,
                   muda::CDenseVectorView<Float> x,
                   Float                         b,
                   muda::DenseVectorView<Float>  y);

    void warp_reduce_sym_spmv(Float                         a,
                              Eigen::Matrix3d*              triplet_values,
                              int*                          row_ids,
                              int*                          col_ids,
                              int                           triplet_count,
                              muda::CDenseVectorView<Float> x,
                              Float                         b,
                              muda::DenseVectorView<Float>  y)
    {
        srbk_spmv(a, triplet_values, row_ids, col_ids, triplet_count, x, b, y);
    }
};
}  // namespace gipc
