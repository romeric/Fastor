#ifndef BINARY_SOLVE_OP_H
#define BINARY_SOLVE_OP_H

#include "Fastor/meta/meta.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/tensor/Aliasing.h"
#include "Fastor/expressions/linalg_ops/linalg_computation_types.h"
#include "Fastor/expressions/linalg_ops/binary_matmul_op.h"
#include "Fastor/expressions/linalg_ops/unary_inv_op.h"
#include "Fastor/expressions/expression_traits.h"
#include "Fastor/expressions/linalg_ops/linalg_computation_types.h"
#include "Fastor/expressions/linalg_ops/unary_piv_op.h"


namespace Fastor {

// Solving using LU decomposition is in the LU module [unary_lu_op]
// For tensors
// Solve - no pivot
template<SolveCompType SType = SolveCompType::SimpleInv, typename T, size_t I,
    enable_if_t_< SType == SolveCompType::SimpleInv, bool> = false>
FASTOR_INLINE Tensor<T,I> solve(const Tensor<T,I,I> &A, const Tensor<T,I> &b) {
    return matmul(inverse<InvCompType::SimpleInv>(A),b);
}
// Solve - pivot
template<SolveCompType SType = SolveCompType::SimpleInv, typename T, size_t I,
    enable_if_t_< SType == SolveCompType::SimpleInvPiv, bool> = false>
FASTOR_INLINE Tensor<T,I> solve(const Tensor<T,I,I> &A, const Tensor<T,I> &b) {
    // We need to post multiply - swap columns using row permutation vector
    Tensor<size_t,I> p;
    pivot_inplace(A,p);
    auto tmp(apply_pivot(A,p));
    Tensor<T,I,I> invA = inverse<InvCompType::SimpleInv>(tmp);
    return matmul(reconstruct_colwise(invA,p),b);

    // // matrix version
    // Tensor<T,I,I> p;
    // pivot_inplace(A,p);
    // auto tmp(apply_pivot(A,p));
    // Tensor<T,I,I> invA = inverse<InvCompType::SimpleInv>(tmp);
    // return matmul(matmul(invA,p),b);
}

// Multiple right hand sides
template<SolveCompType SType = SolveCompType::SimpleInv, typename T, size_t I, size_t J,
    enable_if_t_< SType == SolveCompType::SimpleInv, bool> = false>
FASTOR_INLINE Tensor<T,I,J> solve(const Tensor<T,I,I> &A, const Tensor<T,I,J> &B) {
    return matmul(inverse(A),B);
}

// Solve - pivot
template<SolveCompType SType = SolveCompType::SimpleInv, typename T, size_t I, size_t J,
    enable_if_t_< SType == SolveCompType::SimpleInvPiv, bool> = false>
FASTOR_INLINE Tensor<T,I,J> solve(const Tensor<T,I,I> &A, const Tensor<T,I,J> &B) {
    Tensor<size_t,I> p;
    pivot_inplace(A,p);
    auto tmp(apply_pivot(A,p));
    Tensor<T,I,I> invA = inverse<InvCompType::SimpleInv>(tmp);
    return matmul(reconstruct(invA,p),B);
}



// For expressions
template<SolveCompType SType = SolveCompType::SimpleInv,
    typename TLhs, typename TRhs, size_t DIM0, size_t DIM1,
        enable_if_t_< is_tensor_v<TLhs> && is_tensor_v<TRhs> && DIM0==2,bool> = false>
FASTOR_INLINE
conditional_t_<TRhs::result_type::dimension_t::value == 1,
    Tensor<
        typename TLhs::scalar_type,
        get_tensor_dimension_v<0,typename TLhs::result_type>
    >,
    Tensor<
        typename TLhs::scalar_type,
        get_tensor_dimension_v<0,typename TLhs::result_type>,
        get_tensor_dimension_v<1,typename TRhs::result_type>
    >
>
solve(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM1> &rhs) {

    using lhs_type = typename TLhs::result_type;
    using rhs_type = typename TRhs::result_type;
    constexpr FASTOR_INDEX M = get_tensor_dimension_v<0,lhs_type>;
    constexpr FASTOR_INDEX N = get_tensor_dimension_v<1,lhs_type>;
    constexpr FASTOR_INDEX M_other = get_tensor_dimension_v<0,rhs_type>;

    static_assert(M==N, "LHS EXPRESSION FOR SOLVE HAS TO BE A SQUARE MATRIX");
    static_assert(M == M_other, "INVALID SOLVE OPERANDS. IN Ax=b ROWS(A)!=ROWS(b)");

    return solve<SType>(lhs.self(),rhs.self());
}


template<SolveCompType SType = SolveCompType::SimpleInv,
    typename TLhs, typename TRhs, size_t DIM0, size_t DIM1,
        enable_if_t_< !is_tensor_v<TLhs> && is_tensor_v<TRhs> && DIM0==2,bool> = false>
FASTOR_INLINE
conditional_t_<TRhs::result_type::dimension_t::value == 1,
    Tensor<
        typename TLhs::scalar_type,
        get_tensor_dimension_v<0,typename TLhs::result_type>
    >,
    Tensor<
        typename TLhs::scalar_type,
        get_tensor_dimension_v<0,typename TLhs::result_type>,
        get_tensor_dimension_v<1,typename TRhs::result_type>
    >
>
solve(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM1> &rhs) {

    using lhs_type = typename TLhs::result_type;
    using rhs_type = typename TRhs::result_type;
    constexpr FASTOR_INDEX M = get_tensor_dimension_v<0,lhs_type>;
    constexpr FASTOR_INDEX N = get_tensor_dimension_v<1,lhs_type>;
    constexpr FASTOR_INDEX M_other = get_tensor_dimension_v<0,rhs_type>;

    static_assert(M==N, "LHS EXPRESSION FOR SOLVE HAS TO BE A SQUARE MATRIX");
    static_assert(M == M_other, "INVALID SOLVE OPERANDS. IN Ax=b ROWS(A)!=ROWS(b)");

    const lhs_type A(lhs.self());

    return solve<SType>(A,rhs.self());
}


template<SolveCompType SType = SolveCompType::SimpleInv,
    typename TLhs, typename TRhs, size_t DIM0, size_t DIM1,
        enable_if_t_< is_tensor_v<TLhs> && !is_tensor_v<TRhs> && DIM0==2,bool> = false>
FASTOR_INLINE
conditional_t_<TRhs::result_type::dimension_t::value == 1,
    Tensor<
        typename TLhs::scalar_type,
        get_tensor_dimension_v<0,typename TLhs::result_type>
    >,
    Tensor<
        typename TLhs::scalar_type,
        get_tensor_dimension_v<0,typename TLhs::result_type>,
        get_tensor_dimension_v<1,typename TRhs::result_type>
    >
>
solve(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM1> &rhs) {

    using lhs_type = typename TLhs::result_type;
    using rhs_type = typename TRhs::result_type;
    constexpr FASTOR_INDEX M = get_tensor_dimension_v<0,lhs_type>;
    constexpr FASTOR_INDEX N = get_tensor_dimension_v<1,lhs_type>;
    constexpr FASTOR_INDEX M_other = get_tensor_dimension_v<0,rhs_type>;

    static_assert(M==N, "LHS EXPRESSION FOR SOLVE HAS TO BE A SQUARE MATRIX");
    static_assert(M == M_other, "INVALID SOLVE OPERANDS. IN Ax=b ROWS(A)!=ROWS(b)");

    const rhs_type b(rhs.self());

    return solve<SType>(lhs.self(),b);
}


template<SolveCompType SType = SolveCompType::SimpleInv,
    typename TLhs, typename TRhs, size_t DIM0, size_t DIM1,
        enable_if_t_< !is_tensor_v<TLhs> && !is_tensor_v<TRhs> && DIM0==2,bool> = false>
FASTOR_INLINE
conditional_t_<TRhs::result_type::dimension_t::value == 1,
    Tensor<
        typename TLhs::scalar_type,
        get_tensor_dimension_v<0,typename TLhs::result_type>
    >,
    Tensor<
        typename TLhs::scalar_type,
        get_tensor_dimension_v<0,typename TLhs::result_type>,
        get_tensor_dimension_v<1,typename TRhs::result_type>
    >
>
solve(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM1> &rhs) {

    using lhs_type = typename TLhs::result_type;
    using rhs_type = typename TRhs::result_type;
    constexpr FASTOR_INDEX M = get_tensor_dimension_v<0,lhs_type>;
    constexpr FASTOR_INDEX N = get_tensor_dimension_v<1,lhs_type>;
    constexpr FASTOR_INDEX M_other = get_tensor_dimension_v<0,rhs_type>;

    static_assert(M==N, "LHS EXPRESSION FOR SOLVE HAS TO BE A SQUARE MATRIX");
    static_assert(M == M_other, "INVALID SOLVE OPERANDS. IN Ax=b ROWS(A)!=ROWS(b)");

    const lhs_type A(lhs.self());
    const rhs_type b(rhs.self());

    return solve<SType>(A,b);
}


} // end of namespace Fastor


#endif // BINARY_SOLVE_OP_H
