#ifndef BINARY_MATMUL_OP_H
#define BINARY_MATMUL_OP_H

#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/Aliasing.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/expressions/expression_traits.h"


namespace Fastor {

template<typename TLhs, typename TRhs, size_t DIM0>
struct BinaryMatMulOp: public AbstractTensor<BinaryMatMulOp<TLhs, TRhs, DIM0>,DIM0> {
    using lhs_expr_type = expression_t<TLhs>;
    using rhs_expr_type = expression_t<TRhs>;
    using lhs_type = typename TLhs::result_type;
    using rhs_type = typename TRhs::result_type;
    static constexpr FASTOR_INDEX lhs_rank = lhs_type::Dimension_t::value;
    static constexpr FASTOR_INDEX rhs_rank = rhs_type::Dimension_t::value;
    static constexpr FASTOR_INDEX M = lhs_rank == 2 ? get_tensor_dimensions<lhs_type>::dims[0] : 1;
    static constexpr FASTOR_INDEX K = lhs_rank == 2 ? get_tensor_dimensions<lhs_type>::dims[1] : 1;
    static constexpr FASTOR_INDEX N = rhs_rank == 2 ? get_tensor_dimensions<rhs_type>::dims[1] : 1;
    static constexpr FASTOR_INDEX flop_count = M*N*K;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    using scalar_type = typename scalar_type_finder<BinaryMatMulOp<TLhs, TRhs, DIM0>>::type;
    using result_type = conditional_t_<lhs_rank == 1,   // vector-matrix
                                        Tensor<scalar_type,N> ,
                                        conditional_t_<rhs_rank == 1, // matrix-vector
                                            Tensor<scalar_type,M>,
                                            Tensor<scalar_type,M,N>   // matrix-matrix
                                        >
                                    >;

    FASTOR_INLINE BinaryMatMulOp(lhs_expr_type inlhs, rhs_expr_type inrhs) : _lhs(inlhs), _rhs(inrhs) {}

    FASTOR_INLINE FASTOR_INDEX size() const {return M*N;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return i==0 ? M : N;}

    constexpr FASTOR_INLINE lhs_expr_type lhs() const {return _lhs;}
    constexpr FASTOR_INLINE rhs_expr_type rhs() const {return _rhs;}

private:
    lhs_expr_type _lhs;
    rhs_expr_type _rhs;
};

template<typename TLhs, typename TRhs, size_t DIM0, size_t DIM1,
         enable_if_t_<!is_arithmetic_v_<TLhs> &&!is_arithmetic_v_<TRhs>,bool> = 0 >
FASTOR_INLINE BinaryMatMulOp<TLhs, TRhs, meta_min<DIM0,DIM1>::value>
operator %(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM1> &rhs) {
  return BinaryMatMulOp<TLhs, TRhs, meta_min<DIM0,DIM1>::value>(lhs.self(), rhs.self());
}




// helper dispatcher functions
namespace internal {
// till streaming gemm is implemented
template<typename T, size_t M, size_t K, size_t N>
FASTOR_INLINE
void _gemm(const T alpha, const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, const T beta, T * FASTOR_RESTRICT c) {

    T FASTOR_ALIGN tmp[M*N];
    if (beta == 0) {
        // non-streaming
        _matmul<T,M,K,N>(a,b,tmp);
        for (size_t i = 0; i<M*N; ++i)
            c[i] = alpha * tmp[i];
    }
    else {
        // streaming
        _matmul<T,M,K,N>(a,b,tmp);
        for (size_t i = 0; i<M*N; ++i)
            c[i] = alpha * tmp[i] + beta*c[i];
    }
}
template<typename T, size_t M, size_t K, size_t N>
FASTOR_INLINE
void _gemm_mul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {
    T FASTOR_ALIGN tmp[M*N];
    _matmul<T,M,K,N>(a,b,tmp);
    for (size_t i = 0; i<M*N; ++i)
        c[i] *= tmp[i];
}
template<typename T, size_t M, size_t K, size_t N>
FASTOR_INLINE
void _gemm_div(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {
    T FASTOR_ALIGN tmp[M*N];
    _matmul<T,M,K,N>(a,b,tmp);
    for (size_t i = 0; i<M*N; ++i)
        c[i] /= tmp[i];
}



// matmul - matvec overloads
template<typename T, size_t I, size_t J, size_t K>
FASTOR_INLINE void matmul_dispatcher(const Tensor<T,I,J> &a, const Tensor<T,J,K> &b, Tensor<T,I,K> &out) {
    FASTOR_IF_CONSTEXPR(J==1) {
        _dyadic<T,I,K>(a.data(),b.data(),out.data());
    }
    else FASTOR_IF_CONSTEXPR(I==1 && J!=1 && K==1) {
        out = _doublecontract<T,J,1>(a.data(),b.data());
    }
    else {
        _matmul<T,I,J,K>(a.data(),b.data(),out.data());
    }
}
template<typename T, size_t I, size_t J>
FASTOR_INLINE void matmul_dispatcher(const Tensor<T,I,J> &a, const Tensor<T,J> &b, Tensor<T,I> &out) {
    _matmul<T,I,J,1>(a.data(),b.data(),out.data());
}
template<typename T, size_t J, size_t K>
FASTOR_INLINE void matmul_dispatcher(const Tensor<T,J> &a, const Tensor<T,J,K> &b, Tensor<T,K> &out) {
    _matmul<T,1,J,K>(a.data(),b.data(),out.data());
}

template<typename T, size_t I, size_t J, size_t K>
FASTOR_INLINE void matmul_dispatcher(const T alpha, const Tensor<T,I,J> &a, const Tensor<T,J,K> &b, const T beta, Tensor<T,I,K> &out) {
    _gemm<T,I,J,K>(alpha,a.data(),b.data(),beta,out.data());
}
template<typename T, size_t I, size_t J>
FASTOR_INLINE void matmul_dispatcher(const T alpha, const Tensor<T,I,J> &a, const Tensor<T,J> &b, const T beta, Tensor<T,I> &out) {
    _gemm<T,I,J,1>(alpha,a.data(),b.data(),beta,out.data());
}
template<typename T, size_t J, size_t K>
FASTOR_INLINE void matmul_dispatcher(const T alpha, const Tensor<T,J> &a, const Tensor<T,J,K> &b, const T beta, Tensor<T,K> &out) {
    _gemm<T,1,J,K>(alpha,a.data(),b.data(),beta,out.data());
}

template<typename T, size_t I, size_t J, size_t K>
FASTOR_INLINE void matmul_dispatcher_mul(const Tensor<T,I,J> &a, const Tensor<T,J,K> &b, Tensor<T,I,K> &out) {
    _gemm_mul<T,I,J,K>(a.data(),b.data(),out.data());
}
template<typename T, size_t I, size_t J>
FASTOR_INLINE void matmul_dispatcher_mul(const Tensor<T,I,J> &a, const Tensor<T,J> &b, Tensor<T,I> &out) {
    _gemm_mul<T,I,J,1>(a.data(),b.data(),out.data());
}
template<typename T, size_t J, size_t K>
FASTOR_INLINE void matmul_dispatcher_mul(const Tensor<T,J> &a, const Tensor<T,J,K> &b, Tensor<T,K> &out) {
    _gemm_mul<T,1,J,K>(a.data(),b.data(),out.data());
}

template<typename T, size_t I, size_t J, size_t K>
FASTOR_INLINE void matmul_dispatcher_div(const Tensor<T,I,J> &a, const Tensor<T,J,K> &b, Tensor<T,I,K> &out) {
    _gemm_div<T,I,J,K>(a.data(),b.data(),out.data());
}
template<typename T, size_t I, size_t J>
FASTOR_INLINE void matmul_dispatcher_div(const Tensor<T,I,J> &a, const Tensor<T,J> &b, Tensor<T,I> &out) {
    _gemm_div<T,I,J,1>(a.data(),b.data(),out.data());
}
template<typename T, size_t J, size_t K>
FASTOR_INLINE void matmul_dispatcher_div(const Tensor<T,J> &a, const Tensor<T,J,K> &b, Tensor<T,K> &out) {
    _gemm_div<T,1,J,K>(a.data(),b.data(),out.data());
}

} // internal



// assignments
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    // dst = matmul(src.lhs().self(),src.rhs().self()); // this makes a copy for dst, compiler emits a memcpy
    internal::matmul_dispatcher(src.lhs().self(),src.rhs().self(),dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using lhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_type;
    lhs_t a(src.lhs().self());
    // dst = matmul(a,src.rhs().self()); // this makes a copy for dst, compiler emits a memcpy
    internal::matmul_dispatcher(a,src.rhs().self(),dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            !is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using rhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_type;
    rhs_t b(src.rhs().self());
    // dst = matmul(src.lhs().self(),b); // this makes a copy for dst, compiler emits a memcpy
    internal::matmul_dispatcher(src.lhs().self(),b,dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            !is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using lhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_type;
    using rhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_type;
    lhs_t a(src.lhs().self());
    rhs_t b(src.rhs().self());
    // dst = matmul(a,b); // this makes a copy for dst, compiler emits a memcpy
    internal::matmul_dispatcher(a,b,dst.self());
}



// assignments add
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    internal::matmul_dispatcher((T)1,src.lhs().self(),src.rhs().self(),(T)1,dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    using lhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_type;
    lhs_t a(src.lhs().self());
    internal::matmul_dispatcher((T)1,a,src.rhs().self(),(T)1,dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            !is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    using rhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_type;
    rhs_t b(src.rhs().self());
    internal::matmul_dispatcher((T)1,src.lhs().self(),b,(T)1,dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            !is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    using lhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_type;
    using rhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_type;
    lhs_t a(src.lhs().self());
    rhs_t b(src.rhs().self());
    internal::matmul_dispatcher((T)1,a,b,(T)1,dst.self());
}


// assignments sub
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    internal::matmul_dispatcher((T)-1,src.lhs().self(),src.rhs().self(),(T)1,dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    using lhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_type;
    lhs_t a(src.lhs().self());
    internal::matmul_dispatcher((T)-1,a,src.rhs().self(),(T)1,dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            !is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    using rhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_type;
    rhs_t b(src.rhs().self());
    internal::matmul_dispatcher((T)-1,src.lhs().self(),b,(T)1,dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            !is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    using lhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_type;
    using rhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_type;
    lhs_t a(src.lhs().self());
    rhs_t b(src.rhs().self());
    internal::matmul_dispatcher((T)-1,a,b,(T)1,dst.self());
}


// assignments mul
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_mul(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    internal::matmul_dispatcher_mul(src.lhs().self(),src.rhs().self(),dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_mul(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using lhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_type;
    lhs_t a(src.lhs().self());
    internal::matmul_dispatcher_mul(a,src.rhs().self(),dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            !is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_mul(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using rhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_type;
    rhs_t b(src.rhs().self());
    internal::matmul_dispatcher_mul(src.lhs().self(),b,dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            !is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_mul(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using lhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_type;
    using rhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_type;
    lhs_t a(src.lhs().self());
    rhs_t b(src.rhs().self());
    internal::matmul_dispatcher_mul(a,b,dst.self());
}


// assignments div
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_div(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    internal::matmul_dispatcher_div(src.lhs().self(),src.rhs().self(),dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_div(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using lhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_type;
    lhs_t a(src.lhs().self());
    internal::matmul_dispatcher_div(a,src.rhs().self(),dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            !is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_div(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using rhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_type;
    rhs_t b(src.rhs().self());
    internal::matmul_dispatcher_div(src.lhs().self(),b,dst.self());
}
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            !is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_div(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using lhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_type;
    using rhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_type;
    lhs_t a(src.lhs().self());
    rhs_t b(src.rhs().self());
    internal::matmul_dispatcher_div(a,b,dst.self());
}







// recursive greedy-like
template<typename Derived, size_t DIM, typename TLhs, typename TRhs0, typename TRhs1, size_t OtherDIM>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<BinaryMatMulOp<TLhs, TRhs0, OtherDIM>, TRhs1, OtherDIM> &src) {
    // using T = typename BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::scalar_type;
    FASTOR_IF_CONSTEXPR(BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::flop_count > BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>::flop_count)
    {
        using result_t = typename BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>::result_type;
        result_t tmp;
        assign(tmp, BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>(src.lhs().rhs().self(),src.rhs().self()));
        assign(dst, BinaryMatMulOp<TLhs, result_t, OtherDIM>(src.lhs().lhs().self(),tmp));
    }
    else
    {
        using result_t = typename BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::result_type;
        result_t tmp;
        assign(tmp, BinaryMatMulOp<TLhs, TRhs0, OtherDIM>(src.lhs().lhs().self(),src.lhs().rhs().self()));
        assign(dst, BinaryMatMulOp<result_t, TRhs1, OtherDIM>(tmp,src.rhs().self()));
    }
}

template<typename Derived, size_t DIM, typename TLhs, typename TRhs0, typename TRhs1, size_t OtherDIM>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<BinaryMatMulOp<TLhs, TRhs0, OtherDIM>, TRhs1, OtherDIM> &src) {
    // using T = typename BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::scalar_type;
    FASTOR_IF_CONSTEXPR(BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::flop_count > BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>::flop_count)
    {
        using result_t = typename BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>::result_type;
        result_t tmp;
        assign(tmp, BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>(src.lhs().rhs().self(),src.rhs().self()));
        assign_add(dst, BinaryMatMulOp<TLhs, result_t, OtherDIM>(src.lhs().lhs().self(),tmp));
    }
    else
    {
        using result_t = typename BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::result_type;
        result_t tmp;
        assign(tmp, BinaryMatMulOp<TLhs, TRhs0, OtherDIM>(src.lhs().lhs().self(),src.lhs().rhs().self()));
        assign_add(dst, BinaryMatMulOp<result_t, TRhs1, OtherDIM>(tmp,src.rhs().self()));
    }
}

template<typename Derived, size_t DIM, typename TLhs, typename TRhs0, typename TRhs1, size_t OtherDIM>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<BinaryMatMulOp<TLhs, TRhs0, OtherDIM>, TRhs1, OtherDIM> &src) {
    // using T = typename BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::scalar_type;
    FASTOR_IF_CONSTEXPR(BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::flop_count > BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>::flop_count)
    {
        using result_t = typename BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>::result_type;
        result_t tmp;
        assign(tmp, BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>(src.lhs().rhs().self(),src.rhs().self()));
        assign_sub(dst, BinaryMatMulOp<TLhs, result_t, OtherDIM>(src.lhs().lhs().self(),tmp));
    }
    else
    {
        using result_t = typename BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::result_type;
        result_t tmp;
        assign(tmp, BinaryMatMulOp<TLhs, TRhs0, OtherDIM>(src.lhs().lhs().self(),src.lhs().rhs().self()));
        assign_sub(dst, BinaryMatMulOp<result_t, TRhs1, OtherDIM>(tmp,src.rhs().self()));
    }
}

template<typename Derived, size_t DIM, typename TLhs, typename TRhs0, typename TRhs1, size_t OtherDIM>
FASTOR_INLINE void assign_mul(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<BinaryMatMulOp<TLhs, TRhs0, OtherDIM>, TRhs1, OtherDIM> &src) {
    // using T = typename BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::scalar_type;
    FASTOR_IF_CONSTEXPR(BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::flop_count > BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>::flop_count)
    {
        using result_t = typename BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>::result_type;
        result_t tmp;
        assign(tmp, BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>(src.lhs().rhs().self(),src.rhs().self()));
        assign_mul(dst, BinaryMatMulOp<TLhs, result_t, OtherDIM>(src.lhs().lhs().self(),tmp));
    }
    else
    {
        using result_t = typename BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::result_type;
        result_t tmp;
        assign(tmp, BinaryMatMulOp<TLhs, TRhs0, OtherDIM>(src.lhs().lhs().self(),src.lhs().rhs().self()));
        assign_mul(dst, BinaryMatMulOp<result_t, TRhs1, OtherDIM>(tmp,src.rhs().self()));
    }
}

template<typename Derived, size_t DIM, typename TLhs, typename TRhs0, typename TRhs1, size_t OtherDIM>
FASTOR_INLINE void assign_div(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<BinaryMatMulOp<TLhs, TRhs0, OtherDIM>, TRhs1, OtherDIM> &src) {
    // using T = typename BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::scalar_type;
    FASTOR_IF_CONSTEXPR(BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::flop_count > BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>::flop_count)
    {
        using result_t = typename BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>::result_type;
        result_t tmp;
        assign(tmp, BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>(src.lhs().rhs().self(),src.rhs().self()));
        assign_div(dst, BinaryMatMulOp<TLhs, result_t, OtherDIM>(src.lhs().lhs().self(),tmp));
    }
    else
    {
        using result_t = typename BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::result_type;
        result_t tmp;
        assign(tmp, BinaryMatMulOp<TLhs, TRhs0, OtherDIM>(src.lhs().lhs().self(),src.lhs().rhs().self()));
        assign_div(dst, BinaryMatMulOp<result_t, TRhs1, OtherDIM>(tmp,src.rhs().self()));
    }
}


} // end of namespace Fastor


#endif // BINARY_MATMUL_OP_H