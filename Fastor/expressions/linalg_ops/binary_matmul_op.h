#ifndef BINARY_MATMUL_OP_H
#define BINARY_MATMUL_OP_H

#include "Fastor/meta/meta.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/backend/inner.h"
#include "Fastor/backend/matmul/matmul.h"
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
    static constexpr FASTOR_INDEX lhs_rank = lhs_type::dimension_t::value;
    static constexpr FASTOR_INDEX rhs_rank = rhs_type::dimension_t::value;
    static constexpr FASTOR_INDEX M = get_tensor_dimension_v<0,lhs_type>;
    static constexpr FASTOR_INDEX K = get_tensor_dimension_v<1,lhs_type>;
    static constexpr FASTOR_INDEX N = get_tensor_dimension_v<1,rhs_type>;
    static constexpr FASTOR_INDEX K_other = get_tensor_dimension_v<0,rhs_type>;
    static constexpr FASTOR_INDEX flop_count = M*N*K;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    using scalar_type = typename scalar_type_finder<BinaryMatMulOp<TLhs, TRhs, DIM0>>::type;
    // does not matter which one as matmul bypasses this
    using simd_vector_type = typename TLhs::simd_vector_type;
    using simd_abi_type = typename simd_vector_type::abi_type;
    using result_type = conditional_t_<lhs_rank == 1,   // vector-matrix
                                        Tensor<scalar_type,N> ,
                                        conditional_t_<rhs_rank == 1, // matrix-vector
                                            Tensor<scalar_type,M>,
                                            Tensor<scalar_type,M,N>   // matrix-matrix
                                        >
                                    >;

    FASTOR_INLINE BinaryMatMulOp(lhs_expr_type inlhs, rhs_expr_type inrhs) : _lhs(inlhs), _rhs(inrhs) {
        static_assert(lhs_rank <=2 && rhs_rank <=2, "EXPRESSIONS FOR MATRIX MULTIPLICATION HAVE TO BE 2-DIMENSIONAL");
        static_assert(K == K_other, "INVALID MATMUL OPERANDS. COLUMNS(A)!=ROWS(B)");
    }

    constexpr FASTOR_INLINE FASTOR_INDEX size() const {return M*N;}
    constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return i==0 ? M : N;}

    constexpr FASTOR_INLINE lhs_expr_type lhs() const {return _lhs;}
    constexpr FASTOR_INLINE rhs_expr_type rhs() const {return _rhs;}

private:
    lhs_expr_type _lhs;
    rhs_expr_type _rhs;
};

template<typename TLhs, typename TRhs, size_t DIM0, size_t DIM1,
         enable_if_t_<!is_arithmetic_v_<TLhs> &&!is_arithmetic_v_<TRhs>,bool> = 0 >
constexpr FASTOR_INLINE BinaryMatMulOp<TLhs, TRhs, meta_min<DIM0,DIM1>::value>
operator %(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM1> &rhs) {
  return BinaryMatMulOp<TLhs, TRhs, meta_min<DIM0,DIM1>::value>(lhs.self(), rhs.self());
}




// helper dispatcher functions
namespace internal {
// till streaming gemm is implemented
template<typename T, size_t M, size_t K, size_t N>
FASTOR_INLINE
void _gemm(const T alpha, const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, const T beta, T * FASTOR_RESTRICT c) {

    FASTOR_ARCH_ALIGN T tmp[M*N];
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
    FASTOR_ARCH_ALIGN T tmp[M*N];
    _matmul<T,M,K,N>(a,b,tmp);
    for (size_t i = 0; i<M*N; ++i)
        c[i] *= tmp[i];
}
template<typename T, size_t M, size_t K, size_t N>
FASTOR_INLINE
void _gemm_div(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {
    FASTOR_ARCH_ALIGN T tmp[M*N];
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
        out.data()[0] = _inner<T,J>(a.data(),b.data());
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


// For tensors
template<typename T, size_t I, size_t J, size_t K>
FASTOR_INLINE Tensor<T,I,K> matmul(const Tensor<T,I,J> &a, const Tensor<T,J,K> &b) {
    Tensor<T,I,K> out;
    FASTOR_IF_CONSTEXPR(J==1) {
        _dyadic<T,I,K>(a.data(),b.data(),out.data());
    }
    else FASTOR_IF_CONSTEXPR(I==1 && J!=1 && K==1) {
        out.data()[0] = _inner<T,J>(a.data(),b.data());
    }
    else {
        _matmul<T,I,J,K>(a.data(),b.data(),out.data());
    }
    return out;
}
template<typename T, size_t I, size_t J>
FASTOR_INLINE Tensor<T,I> matmul(const Tensor<T,I,J> &a, const Tensor<T,J> &b) {
    Tensor<T,I> out;
    _matmul<T,I,J,1>(a.data(),b.data(),out.data());
    return out;
}
template<typename T, size_t J, size_t K>
FASTOR_INLINE Tensor<T,K> matmul(const Tensor<T,J> &a, const Tensor<T,J,K> &b) {
    Tensor<T,K> out;
    _matmul<T,1,J,K>(a.data(),b.data(),out.data());
    return out;
}


// Generic matmul function for AbstractTensor types are provided here
template<typename Derived0, size_t DIM0, typename Derived1, size_t DIM1,
    enable_if_t_<is_less_equal_v_<DIM0,2> && is_less_equal_v_<DIM1,2>
    && !is_tensor_v<Derived0> && !is_tensor_v<Derived1>,bool> = 0 >
FASTOR_INLINE
conditional_t_<Derived0::result_type::dimension_t::value == 1,
    Tensor<typename Derived0::scalar_type,
        get_tensor_dimension_v<1,typename Derived1::result_type> >,
    conditional_t_<Derived1::result_type::dimension_t::value == 1,
        Tensor<typename Derived0::scalar_type,
            get_tensor_dimension_v<0,typename Derived0::result_type> >,
        Tensor<typename Derived0::scalar_type,
            get_tensor_dimension_v<0,typename Derived0::result_type> ,
            get_tensor_dimension_v<1,typename Derived1::result_type> >
    >
>
matmul(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b) {
    using lhs_type = typename Derived0::result_type;
    using rhs_type = typename Derived1::result_type;
    const lhs_type tmp_a(a);
    const rhs_type tmp_b(b);
    return matmul(tmp_a,tmp_b);
}

template<typename Derived0, size_t DIM0, typename Derived1, size_t DIM1,
    enable_if_t_<is_less_equal_v_<DIM0,2> && is_less_equal_v_<DIM1,2>
    && !is_tensor_v<Derived0> && is_tensor_v<Derived1>,bool> = 0 >
FASTOR_INLINE
conditional_t_<Derived0::result_type::dimension_t::value == 1,
    Tensor<typename Derived0::scalar_type,
        get_tensor_dimension_v<1,typename Derived1::result_type> >,
    conditional_t_<Derived1::result_type::dimension_t::value == 1,
        Tensor<typename Derived0::scalar_type,
            get_tensor_dimension_v<0,typename Derived0::result_type> >,
        Tensor<typename Derived0::scalar_type,
            get_tensor_dimension_v<0,typename Derived0::result_type> ,
            get_tensor_dimension_v<1,typename Derived1::result_type> >
    >
>
matmul(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b) {
    using lhs_type = typename Derived0::result_type;
    const lhs_type tmp_a(a);
    return matmul(tmp_a,b.self());
}

template<typename Derived0, size_t DIM0, typename Derived1, size_t DIM1,
    enable_if_t_<is_less_equal_v_<DIM0,2> && is_less_equal_v_<DIM1,2>
    && is_tensor_v<Derived0> && !is_tensor_v<Derived1>,bool> = 0 >
FASTOR_INLINE
conditional_t_<Derived0::result_type::dimension_t::value == 1,
    Tensor<typename Derived0::scalar_type,
        get_tensor_dimension_v<1,typename Derived1::result_type> >,
    conditional_t_<Derived1::result_type::dimension_t::value == 1,
        Tensor<typename Derived0::scalar_type,
            get_tensor_dimension_v<0,typename Derived0::result_type> >,
        Tensor<typename Derived0::scalar_type,
            get_tensor_dimension_v<0,typename Derived0::result_type> ,
            get_tensor_dimension_v<1,typename Derived1::result_type> >
    >
>
matmul(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b) {
    using rhs_type = typename Derived1::result_type;
    const rhs_type tmp_b(b);
    return matmul(a.self(),tmp_b);
}



// triangular matmul functions
template<typename LhsType = UpLoType::General, typename RhsType = UpLoType::General, typename T, size_t M, size_t K, size_t N>
Tensor<T,M,N> tmatmul(const Tensor<T,M,K> &a, const Tensor<T,K,N> &b) {
    Tensor<T,M,N> out;
    _tmatmul<T,M,K,N,LhsType,RhsType>(a.data(),b.data(),out.data());
    return out;
}
template<typename LhsType = UpLoType::General, typename RhsType = UpLoType::General, typename T, size_t I, size_t J>
FASTOR_INLINE Tensor<T,I> tmatmul(const Tensor<T,I,J> &a, const Tensor<T,J> &b) {
    Tensor<T,I> out;
    _tmatmul<T,I,J,1,LhsType,RhsType>(a.data(),b.data(),out.data());
    return out;
}
template<typename LhsType = UpLoType::General, typename RhsType = UpLoType::General, typename T, size_t J, size_t K>
FASTOR_INLINE Tensor<T,K> tmatmul(const Tensor<T,J> &a, const Tensor<T,J,K> &b) {
    Tensor<T,K> out;
    _tmatmul<T,1,J,K,LhsType,RhsType>(a.data(),b.data(),out.data());
    return out;
}

#if FASTOR_CXX_VERSION >= 2014
// tmatmul for generic expressions
template<typename LhsType = UpLoType::General, typename RhsType = UpLoType::General,
    typename Derived0, size_t DIM0, typename Derived1, size_t DIM1,
    enable_if_t_<is_less_equal_v_<DIM0,2> && is_less_equal_v_<DIM1,2>
    && !is_tensor_v<Derived0> && !is_tensor_v<Derived1>,bool> = 0 >
FASTOR_INLINE
decltype(auto)
tmatmul(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b) {
    using lhs_type = typename Derived0::result_type;
    using rhs_type = typename Derived1::result_type;
    const lhs_type tmp_a(a);
    const rhs_type tmp_b(b);
    return tmatmul<LhsType,RhsType>(tmp_a,tmp_b);
}

template<typename LhsType = UpLoType::General, typename RhsType = UpLoType::General,
    typename Derived0, size_t DIM0, typename Derived1, size_t DIM1,
    enable_if_t_<is_less_equal_v_<DIM0,2> && is_less_equal_v_<DIM1,2>
    && !is_tensor_v<Derived0> && is_tensor_v<Derived1>,bool> = 0 >
FASTOR_INLINE
decltype(auto)
tmatmul(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b) {
    using lhs_type = typename Derived0::result_type;
    const lhs_type tmp_a(a);
    return tmatmul<LhsType,RhsType>(tmp_a,b.self());
}

template<typename LhsType = UpLoType::General, typename RhsType = UpLoType::General,
    typename Derived0, size_t DIM0, typename Derived1, size_t DIM1,
    enable_if_t_<is_less_equal_v_<DIM0,2> && is_less_equal_v_<DIM1,2>
    && is_tensor_v<Derived0> && !is_tensor_v<Derived1>,bool> = 0 >
FASTOR_INLINE
decltype(auto)
tmatmul(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b) {
    using rhs_type = typename Derived1::result_type;
    const rhs_type tmp_b(b);
    return tmatmul<LhsType,RhsType>(a.self(),tmp_b);
}
#endif









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
