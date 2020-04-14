#ifndef BINARY_MATMUL_OP_H
#define BINARY_MATMUL_OP_H

#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/expressions/expression_traits.h"


namespace Fastor {

template<typename TLhs, typename TRhs, size_t DIM0>
struct BinaryMatMulOp: public AbstractTensor<BinaryMatMulOp<TLhs, TRhs, DIM0>,DIM0> {
    using lhs_expr_type = expression_t<TLhs>;
    using rhs_expr_type = expression_t<TRhs>;
    using lhs_type = typename TLhs::result_type;
    using rhs_type = typename TRhs::result_type;
    static constexpr FASTOR_INDEX M = put_dims_in_Index<lhs_type>::type::_IndexHolder[0];
    static constexpr FASTOR_INDEX K = put_dims_in_Index<lhs_type>::type::_IndexHolder[1];
    static constexpr FASTOR_INDEX N = put_dims_in_Index<rhs_type>::type::_IndexHolder[1];
    static constexpr FASTOR_INDEX flop_count = M*N*K;

    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    using scalar_type = typename scalar_type_finder<BinaryMatMulOp<TLhs, TRhs, DIM0>>::type;
    using result_type = Tensor<scalar_type,M,N>;

    FASTOR_INLINE BinaryMatMulOp(lhs_expr_type inlhs, rhs_expr_type inrhs) : _lhs(inlhs), _rhs(inrhs) {}

    FASTOR_INLINE FASTOR_INDEX size() const {return M*N;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return i==0 ? M : N;}

    constexpr FASTOR_INLINE lhs_expr_type lhs() const {return _lhs;}
    constexpr FASTOR_INLINE rhs_expr_type rhs() const {return _rhs;}

private:
    lhs_expr_type _lhs;
    rhs_expr_type _rhs;
};


template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&
                                 !std::is_arithmetic<TRhs>::value,bool>::type = 0 >
FASTOR_INLINE BinaryMatMulOp<TLhs, TRhs, DIM0>
operator %(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM0> &rhs) {
  return BinaryMatMulOp<TLhs, TRhs, DIM0>(lhs.self(), rhs.self());
}

template<typename TLhs, typename TRhs, size_t DIM0, size_t DIM1,
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&
                                 !std::is_arithmetic<TRhs>::value &&
                                 DIM0!=DIM1,bool>::type = 0 >
FASTOR_INLINE BinaryMatMulOp<TLhs, TRhs, meta_min<DIM0,DIM1>::value>
operator %(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM1> &rhs) {
  return BinaryMatMulOp<TLhs, TRhs, meta_min<DIM0,DIM1>::value>(lhs.self(), rhs.self());
}



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



// assignments
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    // dst = matmul(src.lhs().self(),src.rhs().self()); // this makes a copy for dst, compiler emits a memcpy
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    constexpr FASTOR_INDEX M = BinaryMatMulOp<TLhs, TRhs, OtherDIM>::M;
    constexpr FASTOR_INDEX K = BinaryMatMulOp<TLhs, TRhs, OtherDIM>::K;
    constexpr FASTOR_INDEX N = BinaryMatMulOp<TLhs, TRhs, OtherDIM>::N;
    _matmul<T,M,K,N>(src.lhs().self().data(),src.rhs().self().data(),dst.self().data());
}

template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using lhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_type;
    lhs_t a(src.lhs().self());
    // dst = matmul(a,src.rhs().self()); // this makes a copy for dst, compiler emits a memcpy
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    constexpr FASTOR_INDEX M = BinaryMatMulOp<lhs_t, TRhs, OtherDIM>::M;
    constexpr FASTOR_INDEX K = BinaryMatMulOp<lhs_t, TRhs, OtherDIM>::K;
    constexpr FASTOR_INDEX N = BinaryMatMulOp<lhs_t, TRhs, OtherDIM>::N;
    _matmul<T,M,K,N>(a.data(),src.rhs().self().data(),dst.self().data());
}

template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            !is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using rhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_type;
    rhs_t b(src.rhs().self());
    // dst = matmul(src.lhs().self(),b); // this makes a copy for dst, compiler emits a memcpy
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    constexpr FASTOR_INDEX M = BinaryMatMulOp<TLhs, rhs_t, OtherDIM>::M;
    constexpr FASTOR_INDEX K = BinaryMatMulOp<TLhs, rhs_t, OtherDIM>::K;
    constexpr FASTOR_INDEX N = BinaryMatMulOp<TLhs, rhs_t, OtherDIM>::N;
    _matmul<T,M,K,N>(src.lhs().self().data(),b.data(),dst.self().data());
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
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    constexpr FASTOR_INDEX M = BinaryMatMulOp<lhs_t, rhs_t, OtherDIM>::M;
    constexpr FASTOR_INDEX K = BinaryMatMulOp<lhs_t, rhs_t, OtherDIM>::K;
    constexpr FASTOR_INDEX N = BinaryMatMulOp<lhs_t, rhs_t, OtherDIM>::N;
    _matmul<T,M,K,N>(a.data(),b.data(),dst.self().data());
}



// assignments add
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    // dst = matmul(src.lhs().self(),src.rhs().self()); // this makes a copy for dst, compiler emits a memcpy
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    constexpr FASTOR_INDEX M = BinaryMatMulOp<TLhs, TRhs, OtherDIM>::M;
    constexpr FASTOR_INDEX K = BinaryMatMulOp<TLhs, TRhs, OtherDIM>::K;
    constexpr FASTOR_INDEX N = BinaryMatMulOp<TLhs, TRhs, OtherDIM>::N;
    _gemm<T,M,K,N>(0,src.lhs().self().data(),src.rhs().self().data(),1,dst.self().data());
}

template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using lhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_type;
    lhs_t a(src.lhs().self());
    // dst = matmul(a,src.rhs().self()); // this makes a copy for dst, compiler emits a memcpy
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    constexpr FASTOR_INDEX M = BinaryMatMulOp<lhs_t, TRhs, OtherDIM>::M;
    constexpr FASTOR_INDEX K = BinaryMatMulOp<lhs_t, TRhs, OtherDIM>::K;
    constexpr FASTOR_INDEX N = BinaryMatMulOp<lhs_t, TRhs, OtherDIM>::N;
    _gemm<T,M,K,N>(0,a.data(),src.rhs().self().data(),1,dst.self().data());
}

template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            !is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using rhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_type;
    rhs_t b(src.rhs().self());
    // dst = matmul(src.lhs().self(),b); // this makes a copy for dst, compiler emits a memcpy
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    constexpr FASTOR_INDEX M = BinaryMatMulOp<TLhs, rhs_t, OtherDIM>::M;
    constexpr FASTOR_INDEX K = BinaryMatMulOp<TLhs, rhs_t, OtherDIM>::K;
    constexpr FASTOR_INDEX N = BinaryMatMulOp<TLhs, rhs_t, OtherDIM>::N;
    _gemm<T,M,K,N>(0,src.lhs().self().data(),b.data(),1,dst.self().data());
}

template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            !is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using lhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_type;
    using rhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_type;
    lhs_t a(src.lhs().self());
    rhs_t b(src.rhs().self());
    // dst = matmul(a,b); // this makes a copy for dst, compiler emits a memcpy
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    constexpr FASTOR_INDEX M = BinaryMatMulOp<lhs_t, rhs_t, OtherDIM>::M;
    constexpr FASTOR_INDEX K = BinaryMatMulOp<lhs_t, rhs_t, OtherDIM>::K;
    constexpr FASTOR_INDEX N = BinaryMatMulOp<lhs_t, rhs_t, OtherDIM>::N;
    _gemm<T,M,K,N>(0,a.data(),b.data(),1,dst.self().data());
}


// assignments sub
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    // dst = matmul(src.lhs().self(),src.rhs().self()); // this makes a copy for dst, compiler emits a memcpy
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    constexpr FASTOR_INDEX M = BinaryMatMulOp<TLhs, TRhs, OtherDIM>::M;
    constexpr FASTOR_INDEX K = BinaryMatMulOp<TLhs, TRhs, OtherDIM>::K;
    constexpr FASTOR_INDEX N = BinaryMatMulOp<TLhs, TRhs, OtherDIM>::N;
    _gemm<T,M,K,N>(-1,src.lhs().self().data(),src.rhs().self().data(),1,dst.self().data());
}

template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using lhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_type;
    lhs_t a(src.lhs().self());
    // dst = matmul(a,src.rhs().self()); // this makes a copy for dst, compiler emits a memcpy
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    constexpr FASTOR_INDEX M = BinaryMatMulOp<lhs_t, TRhs, OtherDIM>::M;
    constexpr FASTOR_INDEX K = BinaryMatMulOp<lhs_t, TRhs, OtherDIM>::K;
    constexpr FASTOR_INDEX N = BinaryMatMulOp<lhs_t, TRhs, OtherDIM>::N;
    _gemm<T,M,K,N>(-1,a.data(),src.rhs().self().data(),1,dst.self().data());
}

template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            !is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using rhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_type;
    rhs_t b(src.rhs().self());
    // dst = matmul(src.lhs().self(),b); // this makes a copy for dst, compiler emits a memcpy
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    constexpr FASTOR_INDEX M = BinaryMatMulOp<TLhs, rhs_t, OtherDIM>::M;
    constexpr FASTOR_INDEX K = BinaryMatMulOp<TLhs, rhs_t, OtherDIM>::K;
    constexpr FASTOR_INDEX N = BinaryMatMulOp<TLhs, rhs_t, OtherDIM>::N;
    _gemm<T,M,K,N>(-1,src.lhs().self().data(),b.data(),1,dst.self().data());
}

template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,
    typename std::enable_if<!is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_expr_type>> &&
                            !is_tensor_v<remove_all_t<typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_expr_type>>, bool >::type = false>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<TLhs, TRhs, OtherDIM> &src) {
    using lhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::lhs_type;
    using rhs_t = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::rhs_type;
    lhs_t a(src.lhs().self());
    rhs_t b(src.rhs().self());
    // dst = matmul(a,b); // this makes a copy for dst, compiler emits a memcpy
    using T = typename BinaryMatMulOp<TLhs, TRhs, OtherDIM>::scalar_type;
    constexpr FASTOR_INDEX M = BinaryMatMulOp<lhs_t, rhs_t, OtherDIM>::M;
    constexpr FASTOR_INDEX K = BinaryMatMulOp<lhs_t, rhs_t, OtherDIM>::K;
    constexpr FASTOR_INDEX N = BinaryMatMulOp<lhs_t, rhs_t, OtherDIM>::N;
    _gemm<T,M,K,N>(-1,a.data(),b.data(),1,dst.self().data());
}






// recursive greedy-like
template<typename Derived, size_t DIM, typename TLhs, typename TRhs0, typename TRhs1, size_t OtherDIM>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<BinaryMatMulOp<TLhs, TRhs0, OtherDIM>, TRhs1, OtherDIM> &src) {
    using T = typename BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::scalar_type;
    FASTOR_IF_CONSTEXPR(BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::flop_count > BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>::flop_count)
    {
        constexpr FASTOR_INDEX M = BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>::M;
        constexpr FASTOR_INDEX N = BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>::N;
        using result_t = Tensor<T,M,N>;
        result_t tmp;
        assign(tmp, BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>(src.lhs().rhs().self(),src.rhs().self()));
        assign(dst, BinaryMatMulOp<TLhs, result_t, OtherDIM>(src.lhs().lhs().self(),tmp));
    }
    else
    {
        constexpr FASTOR_INDEX M = BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::M;
        constexpr FASTOR_INDEX N = BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::N;
        using result_t = Tensor<T,M,N>;
        result_t tmp;
        assign(tmp, BinaryMatMulOp<TLhs, TRhs0, OtherDIM>(src.lhs().lhs().self(),src.lhs().rhs().self()));
        assign(dst, BinaryMatMulOp<result_t, TRhs1, OtherDIM>(tmp,src.rhs().self()));
    }
}

template<typename Derived, size_t DIM, typename TLhs, typename TRhs0, typename TRhs1, size_t OtherDIM>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<BinaryMatMulOp<TLhs, TRhs0, OtherDIM>, TRhs1, OtherDIM> &src) {
    using T = typename BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::scalar_type;
    FASTOR_IF_CONSTEXPR(BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::flop_count > BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>::flop_count)
    {
        constexpr FASTOR_INDEX M = BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>::M;
        constexpr FASTOR_INDEX N = BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>::N;
        using result_t = Tensor<T,M,N>;
        result_t tmp;
        assign(tmp, BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>(src.lhs().rhs().self(),src.rhs().self()));
        assign_add(dst, BinaryMatMulOp<TLhs, result_t, OtherDIM>(src.lhs().lhs().self(),tmp));
    }
    else
    {
        constexpr FASTOR_INDEX M = BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::M;
        constexpr FASTOR_INDEX N = BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::N;
        using result_t = Tensor<T,M,N>;
        result_t tmp;
        assign(tmp, BinaryMatMulOp<TLhs, TRhs0, OtherDIM>(src.lhs().lhs().self(),src.lhs().rhs().self()));
        assign_add(dst, BinaryMatMulOp<result_t, TRhs1, OtherDIM>(tmp,src.rhs().self()));
    }
}

template<typename Derived, size_t DIM, typename TLhs, typename TRhs0, typename TRhs1, size_t OtherDIM>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, const BinaryMatMulOp<BinaryMatMulOp<TLhs, TRhs0, OtherDIM>, TRhs1, OtherDIM> &src) {
    using T = typename BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::scalar_type;
    FASTOR_IF_CONSTEXPR(BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::flop_count > BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>::flop_count)
    {
        constexpr FASTOR_INDEX M = BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>::M;
        constexpr FASTOR_INDEX N = BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>::N;
        using result_t = Tensor<T,M,N>;
        result_t tmp;
        assign(tmp, BinaryMatMulOp<TRhs0, TRhs1, OtherDIM>(src.lhs().rhs().self(),src.rhs().self()));
        assign_sub(dst, BinaryMatMulOp<TLhs, result_t, OtherDIM>(src.lhs().lhs().self(),tmp));
    }
    else
    {
        constexpr FASTOR_INDEX M = BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::M;
        constexpr FASTOR_INDEX N = BinaryMatMulOp<TLhs, TRhs0, OtherDIM>::N;
        using result_t = Tensor<T,M,N>;
        result_t tmp;
        assign(tmp, BinaryMatMulOp<TLhs, TRhs0, OtherDIM>(src.lhs().lhs().self(),src.lhs().rhs().self()));
        assign_sub(dst, BinaryMatMulOp<result_t, TRhs1, OtherDIM>(tmp,src.rhs().self()));
    }
}


} // end of namespace Fastor


#endif // BINARY_MATMUL_OP_H