#ifndef BINARY_SOLVE_OP_H
#define BINARY_SOLVE_OP_H

#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/TensorFunctions.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/tensor/Aliasing.h"
#include "Fastor/expressions/linalg_ops/binary_matmul_op.h"
#include "Fastor/expressions/linalg_ops/unary_inv_op.h"
#include "Fastor/expressions/expression_traits.h"

namespace Fastor {

template<typename TLhs, typename TRhs, size_t DIM0>
struct BinarySolveOp: public AbstractTensor<BinarySolveOp<TLhs, TRhs, DIM0>,DIM0> {
    using lhs_expr_type = expression_t<TLhs>;
    using rhs_expr_type = expression_t<TRhs>;
    using lhs_type = typename TLhs::result_type;
    using rhs_type = typename TRhs::result_type;
    static constexpr FASTOR_INDEX lhs_rank = lhs_type::Dimension_t::value;
    static constexpr FASTOR_INDEX rhs_rank = rhs_type::Dimension_t::value;
    static constexpr FASTOR_INDEX M = get_tensor_dimension_v<0,lhs_type>;
    static constexpr FASTOR_INDEX N = get_tensor_dimension_v<1,lhs_type>;
    /* Rows of RHS */
    static constexpr FASTOR_INDEX M_other = get_tensor_dimension_v<0,rhs_type>;
    /* Columns of RHS */
    static constexpr FASTOR_INDEX N_other = get_tensor_dimension_v<1,rhs_type>;
    /* This is not the actual flop cost, but an estimate used for expressions reordering */
    static constexpr FASTOR_INDEX flop_count = M*M*M*N_other;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    using scalar_type = typename scalar_type_finder<BinarySolveOp<TLhs, TRhs, DIM0>>::type;
    using result_type = conditional_t_<rhs_rank == 1,
                                        Tensor<scalar_type,M>,
                                        Tensor<scalar_type,M,N_other>
                                    >;

    FASTOR_INLINE BinarySolveOp(lhs_expr_type inlhs, rhs_expr_type inrhs) : _lhs(inlhs), _rhs(inrhs) {
        static_assert(M==N, "LHS EXPRESSION FOR SOLVE HAS TO BE A SQUARE MATRIX");
        static_assert(M == M_other, "INVALID SOLVE OPERANDS. IN Ax=b ROWS(A)!=ROWS(b)");
    }

    /* It is possible to solve for multiple right hand sides */
    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return M * N_other;}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return i==0 ? M : N_other ;}

    constexpr FASTOR_INLINE lhs_expr_type lhs() const {return _lhs;}
    constexpr FASTOR_INLINE rhs_expr_type rhs() const {return _rhs;}

private:
    lhs_expr_type _lhs;
    rhs_expr_type _rhs;
};

template<typename TLhs, typename TRhs, size_t DIM0, size_t DIM1,
         enable_if_t_<!is_arithmetic_v_<TLhs> &&!is_arithmetic_v_<TRhs> && DIM0==2,bool> = 0 >
constexpr FASTOR_INLINE BinarySolveOp<TLhs, TRhs, DIM0>
solve(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM1> &rhs) {
  return BinarySolveOp<TLhs, TRhs, DIM0>(lhs.self(), rhs.self());
}



namespace internal {
template<typename T, size_t I>
FASTOR_INLINE void solve_dispatcher(const Tensor<T,I,I> &a, const Tensor<T,I> &b, Tensor<T,I> &out) {
    Tensor<T,I,I> _inva;
    internal::inverse_dispatcher(a,_inva);
    _matmul<T,I,I,1>(_inva.data(),b.data(),out.data());
}
// Special case when b is Mx1 and the destination tensor is M
template<typename T, size_t I>
FASTOR_INLINE void solve_dispatcher(const Tensor<T,I,I> &a, const Tensor<T,I,1> &b, Tensor<T,I> &out) {
    Tensor<T,I,I> _inva;
    internal::inverse_dispatcher(a,_inva);
    _matmul<T,I,I,1>(_inva.data(),b.data(),out.data());
}
// Multiple right hand sides
template<typename T, size_t I, size_t J>
FASTOR_INLINE void solve_dispatcher(const Tensor<T,I,I> &a, const Tensor<T,I,J> &b, Tensor<T,I,J> &out) {
    Tensor<T,I,I> _inva;
    internal::inverse_dispatcher(a,_inva);
    _matmul<T,I,I,J>(_inva.data(),b.data(),out.data());
}

template<typename T, size_t I>
FASTOR_INLINE void solve_dispatcher(const T alpha, const Tensor<T,I,I> &a, const Tensor<T,I> &b, const T beta, Tensor<T,I> &out) {
    Tensor<T,I,I> _inva;
    internal::inverse_dispatcher(a,_inva);
    _gemm<T,I,I,1>(alpha,_inva.data(),b.data(),beta,out.data());
}
// Special case when b is Mx1 and the destination tensor is M
template<typename T, size_t I>
FASTOR_INLINE void solve_dispatcher(const T alpha, const Tensor<T,I,I> &a, const Tensor<T,I,1> &b, const T beta, Tensor<T,I> &out) {
    Tensor<T,I,I> _inva;
    internal::inverse_dispatcher(a,_inva);
    _gemm<T,I,I,1>(alpha,_inva.data(),b.data(),beta,out.data());
}
// Multiple right hand sides
template<typename T, size_t I, size_t J>
FASTOR_INLINE void solve_dispatcher(const T alpha, const Tensor<T,I,I> &a, const Tensor<T,I,J> &b, const T beta, Tensor<T,I,J> &out) {
    Tensor<T,I,I> _inva;
    internal::inverse_dispatcher(a,_inva);
    _gemm<T,I,I,J>(alpha,_inva.data(),b.data(),beta,out.data());
}

template<typename T, size_t I>
FASTOR_INLINE void solve_dispatcher_mul(const Tensor<T,I,I> &a, const Tensor<T,I> &b, Tensor<T,I> &out) {
    Tensor<T,I,I> _inva;
    internal::inverse_dispatcher(a,_inva);
    _gemm_mul<T,I,I,1>(_inva.data(),b.data(),out.data());
}
// Special case when b is Mx1 and the destination tensor is M
template<typename T, size_t I>
FASTOR_INLINE void solve_dispatcher_mul(const Tensor<T,I,I> &a, const Tensor<T,I,1> &b, Tensor<T,I> &out) {
    Tensor<T,I,I> _inva;
    internal::inverse_dispatcher(a,_inva);
    _gemm_mul<T,I,I,1>(_inva.data(),b.data(),out.data());
}
// Multiple right hand sides
template<typename T, size_t I, size_t J>
FASTOR_INLINE void solve_dispatcher_mul(const Tensor<T,I,I> &a, const Tensor<T,I,J> &b, Tensor<T,I,J> &out) {
    Tensor<T,I,I> _inva;
    internal::inverse_dispatcher(a,_inva);
    _gemm_mul<T,I,I,J>(_inva.data(),b.data(),out.data());
}

template<typename T, size_t I>
FASTOR_INLINE void solve_dispatcher_div(const Tensor<T,I,I> &a, const Tensor<T,I> &b, Tensor<T,I> &out) {
    Tensor<T,I,I> _inva;
    internal::inverse_dispatcher(a,_inva);
    _gemm_div<T,I,I,1>(_inva.data(),b.data(),out.data());
}
// Special case when b is Mx1 and the destination tensor is M
template<typename T, size_t I>
FASTOR_INLINE void solve_dispatcher_div(const Tensor<T,I,I> &a, const Tensor<T,I,1> &b, Tensor<T,I> &out) {
    Tensor<T,I,I> _inva;
    internal::inverse_dispatcher(a,_inva);
    _gemm_div<T,I,I,1>(_inva.data(),b.data(),out.data());
}
// Multiple right hand sides
template<typename T, size_t I, size_t J>
FASTOR_INLINE void solve_dispatcher_div(const Tensor<T,I,I> &a, const Tensor<T,I,J> &b, Tensor<T,I,J> &out) {
    Tensor<T,I,I> _inva;
    internal::inverse_dispatcher(a,_inva);
    _gemm_div<T,I,I,J>(_inva.data(),b.data(),out.data());
}
} // internal



// assignments
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<TLhs> && is_tensor_v<TRhs>, bool >::type = false>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    internal::solve_dispatcher(src.lhs().self(),src.rhs().self(),dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<TLhs> && is_tensor_v<TRhs>, bool >::type = false>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    using lhs_type = typename TLhs::result_type;
    const lhs_type A(src.lhs().self());
    internal::solve_dispatcher(A,src.rhs().self(),dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    enable_if_t_<is_tensor_v<TLhs> && !is_tensor_v<TRhs>, bool > = false>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    using rhs_type = typename TRhs::result_type;
    const rhs_type b(src.rhs().self());
    internal::solve_dispatcher(src.lhs().self(),b,dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<TLhs> && !is_tensor_v<TRhs>, bool >::type = false>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    using lhs_type = typename TLhs::result_type;
    using rhs_type = typename TRhs::result_type;
    const lhs_type A(src.lhs().self());
    const rhs_type b(src.rhs().self());
    internal::solve_dispatcher(A,b,dst.self());
}

// assign_add
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<TLhs> && is_tensor_v<TRhs>, bool >::type = false>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    using T = typename BinarySolveOp<TLhs, TRhs, OtherDIM>::scalar_type;
    internal::solve_dispatcher(T(1),src.lhs().self(),src.rhs().self(),T(1),dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<TLhs> && is_tensor_v<TRhs>, bool >::type = false>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    using T = typename BinarySolveOp<TLhs, TRhs, OtherDIM>::scalar_type;
    using lhs_type = typename TLhs::result_type;
    const lhs_type A(src.lhs().self());
    internal::solve_dispatcher(T(1),A,src.rhs().self(),T(1),dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    enable_if_t_<is_tensor_v<TLhs> && !is_tensor_v<TRhs>, bool > = false>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    using T = typename BinarySolveOp<TLhs, TRhs, OtherDIM>::scalar_type;
    using rhs_type = typename TRhs::result_type;
    const rhs_type b(src.rhs().self());
    internal::solve_dispatcher(T(1),src.lhs().self(),b,T(1),dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<TLhs> && !is_tensor_v<TRhs>, bool >::type = false>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    using T = typename BinarySolveOp<TLhs, TRhs, OtherDIM>::scalar_type;
    using lhs_type = typename TLhs::result_type;
    using rhs_type = typename TRhs::result_type;
    const lhs_type A(src.lhs().self());
    const rhs_type b(src.rhs().self());
    internal::solve_dispatcher(T(1),A,b,T(1),dst.self());
}

// assign_sub
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<TLhs> && is_tensor_v<TRhs>, bool >::type = false>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    using T = typename BinarySolveOp<TLhs, TRhs, OtherDIM>::scalar_type;
    internal::solve_dispatcher(T(-1),src.lhs().self(),src.rhs().self(),T(1),dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<TLhs> && is_tensor_v<TRhs>, bool >::type = false>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    using T = typename BinarySolveOp<TLhs, TRhs, OtherDIM>::scalar_type;
    using lhs_type = typename TLhs::result_type;
    const lhs_type A(src.lhs().self());
    internal::solve_dispatcher(T(-1),A,src.rhs().self(),T(1),dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    enable_if_t_<is_tensor_v<TLhs> && !is_tensor_v<TRhs>, bool > = false>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    using T = typename BinarySolveOp<TLhs, TRhs, OtherDIM>::scalar_type;
    using rhs_type = typename TRhs::result_type;
    const rhs_type b(src.rhs().self());
    internal::solve_dispatcher(T(-1),src.lhs().self(),b,T(1),dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<TLhs> && !is_tensor_v<TRhs>, bool >::type = false>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    using T = typename BinarySolveOp<TLhs, TRhs, OtherDIM>::scalar_type;
    using lhs_type = typename TLhs::result_type;
    using rhs_type = typename TRhs::result_type;
    const lhs_type A(src.lhs().self());
    const rhs_type b(src.rhs().self());
    internal::solve_dispatcher(T(-1),A,b,T(1),dst.self());
}

// assign_mul
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<TLhs> && is_tensor_v<TRhs>, bool >::type = false>
FASTOR_INLINE void assign_mul(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    internal::solve_dispatcher_mul(src.lhs().self(),src.rhs().self(),dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<TLhs> && is_tensor_v<TRhs>, bool >::type = false>
FASTOR_INLINE void assign_mul(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    using lhs_type = typename TLhs::result_type;
    const lhs_type A(src.lhs().self());
    internal::solve_dispatcher_mul(A,src.rhs().self(),dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    enable_if_t_<is_tensor_v<TLhs> && !is_tensor_v<TRhs>, bool > = false>
FASTOR_INLINE void assign_mul(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    using rhs_type = typename TRhs::result_type;
    const rhs_type b(src.rhs().self());
    internal::solve_dispatcher_mul(src.lhs().self(),b,dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<TLhs> && !is_tensor_v<TRhs>, bool >::type = false>
FASTOR_INLINE void assign_mul(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    using lhs_type = typename TLhs::result_type;
    using rhs_type = typename TRhs::result_type;
    const lhs_type A(src.lhs().self());
    const rhs_type b(src.rhs().self());
    internal::solve_dispatcher_mul(A,b,dst.self());
}

// assign_div
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<TLhs> && is_tensor_v<TRhs>, bool >::type = false>
FASTOR_INLINE void assign_div(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    internal::solve_dispatcher_div(src.lhs().self(),src.rhs().self(),dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<TLhs> && is_tensor_v<TRhs>, bool >::type = false>
FASTOR_INLINE void assign_div(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    using lhs_type = typename TLhs::result_type;
    const lhs_type A(src.lhs().self());
    internal::solve_dispatcher_div(A,src.rhs().self(),dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    enable_if_t_<is_tensor_v<TLhs> && !is_tensor_v<TRhs>, bool > = false>
FASTOR_INLINE void assign_div(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    using rhs_type = typename TRhs::result_type;
    const rhs_type b(src.rhs().self());
    internal::solve_dispatcher_div(src.lhs().self(),b,dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<TLhs> && !is_tensor_v<TRhs>, bool >::type = false>
FASTOR_INLINE void assign_div(AbstractTensor<Derived,DIM> &dst, const BinarySolveOp<TLhs, TRhs, OtherDIM> &src) {
    using lhs_type = typename TLhs::result_type;
    using rhs_type = typename TRhs::result_type;
    const lhs_type A(src.lhs().self());
    const rhs_type b(src.rhs().self());
    internal::solve_dispatcher_div(A,b,dst.self());
}

} // end of namespace Fastor


#endif // BINARY_SOLVE_OP_H