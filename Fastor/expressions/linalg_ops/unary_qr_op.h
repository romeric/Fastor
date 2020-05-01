#ifndef UNARY_QR_OP_H
#define UNARY_QR_OP_H

#include "Fastor/meta/meta.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/Aliasing.h"
#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/Ranges.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/expressions/expression_traits.h"
#include "Fastor/expressions/linalg_ops/linalg_computation_types.h"


namespace Fastor {

namespace internal {

/* Modified Gram-Schmidt Row-wise [MGSR] QR factorisation
   that provides numerical stabilitiy.
   This simple implementation provided for now is just for
   convenience and not tuned for performance. Given that Tensor
   types have fixed dimensions the compiler typically does a good
   job

   See:
        Thomas Jakobs et. al. "Performance and energy consumption of the SIMD
                                Gram–Schmidt process for vector orthogonalization"
        for implementation details including AVX/AVX512 implementation. A straight-
        forward explicit vectorisation of this does not yield good performance for
        row-major tensors, as step 1-2 need gather instructions and step 3-4 require
        dynamic masking to gain performance
*/
template<typename T, size_t M, size_t N>
FASTOR_INLINE void qr_mgsr_dispatcher(const Tensor<T,M,N> &A0, Tensor<T,M,N>& Q, Tensor<T,M,N>& R) {

    // copy incoming tensor as
    Tensor<T,M,N> A(A0);
    // Zero out
    R.fill(0);

    for (size_t i=0; i< N; ++i) {
        // step 1
        T R_ii = 0;
        for (size_t k=0; k< M; ++k) {
            R_ii += A(k,i)*A(k,i);
        }
        R_ii = sqrts(R_ii);
        R(i,i) = R_ii;

        // step 2
        const T rR_ii = T(1) / R_ii;
        for (size_t k=0; k< M; ++k) {
            // Q(k,i) = A(k,i) / R_ii;
            Q(k,i) = A(k,i) * rR_ii;
        }

        // step 3
        for (size_t k=0; k< M; ++k) {
            for (size_t j=i+1; j<N; ++j) {
                R(i,j) += Q(k,i) * A(k,j);
            }
        }

        // step 4
        for (size_t k=0; k< M; ++k) {
            for (size_t j=i+1; j<N; ++j) {
                A(k,j) -= Q(k,i) * R(i,j);
            }
        }
    }
}

} // internal


template<QRCompType QRType = QRCompType::MGSR, typename Expr, size_t DIM0,
    enable_if_t_<is_tensor_v<Expr> && QRType == QRCompType::MGSR,bool> = false>
FASTOR_INLINE
std::tuple<
    Tensor<
        typename scalar_type_finder<Expr>::type,
        get_tensor_dimension_v<0,typename Expr::result_type>,
        get_tensor_dimension_v<1,typename Expr::result_type>
    >,
    Tensor<
        typename scalar_type_finder<Expr>::type,
        get_tensor_dimension_v<1,typename Expr::result_type>,
        get_tensor_dimension_v<0,typename Expr::result_type>
    >
>
qr(const AbstractTensor<Expr,DIM0> &src) {
    using T = typename scalar_type_finder<Expr>::type;
    constexpr size_t M = get_tensor_dimension_v<0,typename Expr::result_type>;
    constexpr size_t N = get_tensor_dimension_v<1,typename Expr::result_type>;

    Tensor<T,M,N> Q;
    Tensor<T,N,M> R;
    internal::qr_mgsr_dispatcher(src.self(),Q,R);

    return std::make_tuple(Q,R);
}


template<QRCompType QRType = QRCompType::MGSR, typename Expr, size_t DIM0,
    enable_if_t_<!is_tensor_v<Expr> && QRType == QRCompType::MGSR,bool> = false>
FASTOR_INLINE
std::tuple<
    Tensor<
        typename scalar_type_finder<Expr>::type,
        get_tensor_dimension_v<0,typename Expr::result_type>,
        get_tensor_dimension_v<1,typename Expr::result_type>
    >,
    Tensor<
        typename scalar_type_finder<Expr>::type,
        get_tensor_dimension_v<1,typename Expr::result_type>,
        get_tensor_dimension_v<0,typename Expr::result_type>
    >
>
qr(const AbstractTensor<Expr,DIM0> &src) {
    using T = typename scalar_type_finder<Expr>::type;
    constexpr size_t M = get_tensor_dimension_v<0,typename Expr::result_type>;
    constexpr size_t N = get_tensor_dimension_v<1,typename Expr::result_type>;

    Tensor<T,M,N> tmp(src.self());
    Tensor<T,M,N> Q;
    Tensor<T,N,M> R;
    internal::qr_mgsr_dispatcher(tmp,Q,R);

    return std::make_tuple(Q,R);
}


template<QRCompType QRType = QRCompType::MGSR, typename Expr, size_t DIM0,
    enable_if_t_<QRType != QRCompType::MGSR,bool> = false>
FASTOR_INLINE
void
qr(const AbstractTensor<Expr,DIM0> &src) {
    static_assert(QRType==QRCompType::MGSR, "QR FACTORISATION USING HOUSEHOLDER REFLECTIONS NOT IMPLEMENETED YET");
}




} // end of namespace Fastor


#endif // UNARY_QR_OP_H