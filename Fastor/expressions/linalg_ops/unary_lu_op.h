#ifndef UNARY_LU_OP_H
#define UNARY_LU_OP_H

#include "Fastor/meta/meta.h"
#include "Fastor/backend/inner.h"
#include "Fastor/backend/lufact.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/Aliasing.h"
#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/Ranges.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/expressions/expression_traits.h"
#include "Fastor/expressions/linalg_ops/linalg_computation_types.h"
#include "Fastor/expressions/linalg_ops/unary_piv_op.h"


namespace Fastor {

namespace internal {

/* Compile time recursive loop with inner for forward substitution of b/B given the lower unitriangular matrix L.
    The following meta functions implements L * y = b for single or multiple right sides
*/
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
template <size_t from, size_t to>
struct forward_subs_impl {

    template<typename T, size_t M>
    static FASTOR_INLINE void do_single_rhs(const Tensor<T,M,M> &L, const Tensor<T,M> &b, Tensor<T,M> &y) {
        y(from) = b(from) -  _inner<T,from>(&L.data()[from*M],y.data());
        forward_subs_impl<from+1,to>::do_single_rhs(L, b, y);
    }
    template<typename T, size_t M>
    static FASTOR_INLINE void do_single_rhs_pivot(const Tensor<T,M,M> &L, const Tensor<T,M> &b, const Tensor<size_t,M> &p, Tensor<T,M> &y) {
        y(from) = b(p(from)) -  _inner<T,from>(&L.data()[from*M],y.data());
        forward_subs_impl<from+1,to>::do_single_rhs_pivot(L, b, p, y);
    }

    template<typename T, size_t M, size_t N>
    static FASTOR_INLINE void do_multi_rhs(const size_t j, const Tensor<T,M,M> &L, const Tensor<T,M,N> &B, Tensor<T,M> &y, Tensor<T,M,N> &X) {
        y(from) = B(from,j) - _inner<T,from>(&L.data()[from*M],y.data());
        X(from,j) = y(from);
        forward_subs_impl<from+1,to>::do_multi_rhs(j, L, B, y, X);
    }

    template<typename T, size_t M, size_t N>
    static FASTOR_INLINE void do_multi_rhs_pivot(const size_t j, const Tensor<T,M,M> &L, const Tensor<T,M,N> &B,
            const Tensor<size_t,M> &p, Tensor<T,M> &y, Tensor<T,M,N> &X) {
        y(from) = B(p(from),j) - _inner<T,from>(&L.data()[from*M],y.data());
        X(from,j) = y(from);
        forward_subs_impl<from+1,to>::do_multi_rhs_pivot(j, L, B, p, y, X);
    }
};
template <size_t from>
struct forward_subs_impl<from,from> {

    template<typename T, size_t M>
    static FASTOR_INLINE void do_single_rhs(const Tensor<T,M,M> &L, const Tensor<T,M> &b, Tensor<T,M> &y) {
        y(from) = b(from) -  _inner<T,from>(&L.data()[from*M],y.data());
    }
    template<typename T, size_t M>
    static FASTOR_INLINE void do_single_rhs_pivot(const Tensor<T,M,M> &L, const Tensor<T,M> &b, const Tensor<size_t,M> &p, Tensor<T,M> &y) {
        y(from) = b(p(from)) -  _inner<T,from>(&L.data()[from*M],y.data());
    }

    template<typename T, size_t M, size_t N>
    static FASTOR_INLINE void do_multi_rhs(const size_t j, const Tensor<T,M,M> &L, const Tensor<T,M,N> &B, Tensor<T,M> &y, Tensor<T,M,N> &X) {
        y(from) = B(from,j) - _inner<T,from>(&L.data()[from*M],y.data());
        X(from,j) = y(from);
    }
    template<typename T, size_t M, size_t N>
    static FASTOR_INLINE void do_multi_rhs_pivot(const size_t j, const Tensor<T,M,M> &L, const Tensor<T,M,N> &B,
            const Tensor<size_t,M> &p, Tensor<T,M> &y, Tensor<T,M,N> &X) {
        y(from) = B(p(from),j) - _inner<T,from>(&L.data()[from*M],y.data());
        X(from,j) = y(from);
    }
};

template<typename T, size_t M>
FASTOR_INLINE Tensor<T,M> forward_subs(const Tensor<T,M,M> &L, const Tensor<T,M> &b) {

    Tensor<T,M> y(0);

    forward_subs_impl<0,M-1>::do_single_rhs(L, b, y);
#if 0
    // The run-time loop version
    // Solve for L * y = b
    for (size_t i=0; i< M; ++i) {
        T value = 0;
        for (size_t k=0; k<i; ++k) {
            value += L(i,k)*y(k);
        }
        y(i) = b(i) - value;
    }
#endif
    return y;
}
template<typename T, size_t M, size_t N>
FASTOR_INLINE Tensor<T,M,N> forward_subs(const Tensor<T,M,M> &L, const Tensor<T,M,N> &B) {

    // We keep a separate output tensor X from y [y is columns of X]
    // to avoid strided access in X for the inner product
    Tensor<T,M,N> X;

    for (size_t j=0; j < N; ++j) {
        Tensor<T,M> y(0);
        forward_subs_impl<0,M-1>::do_multi_rhs(j, L, B, y, X);
    }
#if 0
    // The run-time loop version - X needs to be zeroed out for this version
    for (size_t j=0; j < N; ++j) {
        Tensor<T,M> y(0);
        // Solve for L * y = b
        for (size_t i=0; i< M; ++i) {
            T value = 0;
            for (size_t k=0; k<i; ++k) {
                value += L(i,k)*y(k);
            }
            y(i) = B(i,j) - value;
            X(i,j) = y(i);
        }
    }
#endif
    return X;
}
template<typename T, size_t M>
FASTOR_INLINE Tensor<T,M> forward_subs(const Tensor<T,M,M> &L, const Tensor<size_t,M> &p, const Tensor<T,M> &b) {

    Tensor<T,M> y(0);

    forward_subs_impl<0,M-1>::do_single_rhs_pivot(L, b, p, y);
#if 0
    // The run-time loop version
    // Solve for L * y = b
    for (size_t i=0; i< M; ++i) {
        T value = 0;
        for (size_t k=0; k<i; ++k) {
            value += L(i,k)*y(k);
        }
        y(i) = b(p(i)) - value;
    }
#endif
    return y;
}
template<typename T, size_t M, size_t N>
FASTOR_INLINE Tensor<T,M,N> forward_subs(const Tensor<T,M,M> &L, const Tensor<size_t,M> &p, const Tensor<T,M,N> &B) {

    // We keep a separate output tensor X from y [y is columns of X]
    // to avoid strided access in X for the inner product
    Tensor<T,M,N> X;

    for (size_t j=0; j < N; ++j) {
        Tensor<T,M> y(0);
        forward_subs_impl<0,M-1>::do_multi_rhs_pivot(j, L, B, p, y, X);
    }
#if 0
    // The run-time loop version - X needs to be zeroed out for this version
    for (size_t j=0; j < N; ++j) {
        Tensor<T,M> y(0);
        // Solve for L * y = b
        for (size_t i=0; i< M; ++i) {
            T value = 0;
            for (size_t k=0; k<i; ++k) {
                value += L(i,k)*y(k);
            }
            y(i) = B(p(i),j) - value;
            X(i,j) = y(i);
        }
    }
#endif
    return X;
}
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//




/* Compile time recursive loop with inner for backward substitution of y/Y given the upper triangular matrix U.
    The following meta functions implements U * x = y for single or multiple right sides
*/
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
template <int from, int to>
struct backward_subs_impl {

    template<typename T, size_t M>
    static FASTOR_INLINE void do_single_rhs(const Tensor<T,M,M> &U, const Tensor<T,M> &y, Tensor<T,M> &x) {
        constexpr int idx = (int)M - from;
        const T value = _inner<T,idx>(&U.data()[from*M+from],&x.data()[from]);
        x(from) = ( y(from) - value) / U(from, from);
        backward_subs_impl<from-1,to>::do_single_rhs(U, y, x);
    }

    template<typename T, size_t M, size_t N>
    static FASTOR_INLINE void do_multi_rhs(const size_t j, const Tensor<T,M,M> &U, const Tensor<T,M,N> &Y, Tensor<T,M> &x, Tensor<T,M,N> &X) {
        constexpr int idx = (int)M - from;
        const T value = _inner<T,idx>(&U.data()[from*M+from],&x.data()[from]);
        x(from) = ( Y(from,j) - value ) / U(from, from);
        X(from,j) = x(from);
        backward_subs_impl<from-1,to>::do_multi_rhs(j, U, Y, x, X);
    }
};

template <>
struct backward_subs_impl<0,0> {

    template<typename T, size_t M>
    static FASTOR_INLINE void do_single_rhs(const Tensor<T,M,M> &U, const Tensor<T,M> &y, Tensor<T,M> &x) {
        constexpr int idx = (int)M;
        const T value = _inner<T,idx>(&U.data()[0*M+0],&x.data()[0]);
        x(0) = ( y(0) - value) / U(0, 0);
    }

    template<typename T, size_t M, size_t N>
    static FASTOR_INLINE void do_multi_rhs(const size_t j, const Tensor<T,M,M> &U, const Tensor<T,M,N> &Y, Tensor<T,M> &x, Tensor<T,M,N> &X) {
        constexpr int idx = (int)M;
        const T value = _inner<T,idx>(&U.data()[0*M+0],&x.data()[0]);
        x(0) = ( Y(0,j) - value ) / U(0, 0);
        X(0,j) = x(0);
    }
};

template<typename T, size_t M>
FASTOR_INLINE Tensor<T,M> backward_subs(const Tensor<T,M,M> &U, const Tensor<T,M> &y) {

    // We keep a separate output tensor X from x [x is columns of X]
    // to avoid strided access in X for the inner product
    Tensor<T,M> x(0);
    backward_subs_impl<int(M)-1,0>::do_single_rhs(U, y, x);
#if 0
    // The run-time loop version
    // Solve for of U * x = y
    for (int i= int(M) - 1; i>=0; --i) {
        T value = 0;
        for (int k=i; k<int(M); ++k) {
            value += U(i,k)*x(k);
        }
        x(i) = (y(i) - value) / U(i, i);
    }
#endif
    return x;
}

template<typename T, size_t M, size_t N>
FASTOR_INLINE Tensor<T,M,N> backward_subs(const Tensor<T,M,M> &U, const Tensor<T,M,N> &Y) {

    // We keep a separate output tensor X from x [x is columns of X]
    // to avoid strided access in X for the inner product
    Tensor<T,M,N> X;

    for (size_t j=0; j < N; ++j) {
        Tensor<T,M> x(0);
        backward_subs_impl<int(M)-1,0>::do_multi_rhs(j, U, Y, x, X);
    }
#if 0
    // The run-time loop version - X needs to be zeroed out for this version
    Tensor<T,M,N> X(0);
    for (size_t j=0; j < 1; ++j) {
        // Solve for of U * x = y
        for (int i= int(M) - 1; i>=0; --i) {
            T value = 0;
            for (int k=i; k<int(M); ++k) {
                value += U(i,k)*X(k,j);
            }
            X(i,j) = (Y(i,j) - value) / U(i, i);
        }
    }
#endif
    return X;
}

//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//



/* Simple LU factorisation without pivoting */
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
template <typename T, size_t M, enable_if_t_<is_greater_v_<M,0> && is_less_equal_v_<M,8>,bool> = false>
FASTOR_INLINE void lu_simple_dispatcher(const Tensor<T,M,M>& A, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {
    _lufact<T,M>(A.data(),L.data(),U.data());
}

template <typename T, size_t M, enable_if_t_<is_greater_v_<M,8>,bool> = false>
FASTOR_INLINE void lu_simple_dispatcher(const Tensor<T,M,M>& A1, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {
    L.fill(0);
    U.fill(0);
    for (size_t j = 0; j < M; ++j) {
        L(j, j) = 1;
        for (size_t i = 0; i <= j; ++i) {
            T value = A1(i, j);
            for (size_t k = 0; k < i; ++k) {
                value -= L(i, k) * U(k, j);
            }
            U(i, j) = value;
        }
        for (size_t i = j; i < M; ++i) {
            T value = A1(i, j);
            for (size_t k = 0; k < j; ++k) {
                value -= L(i, k) * U(k, j);
            }
            value /= U(j, j);
            L(i, j) = value;
        }
    }
}
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//


// Recursive non-modifying LU using matmul/outer product
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
/* This in essence implements the following loop but with static views
    as dynamic views cannot be dispatched to matmul or a tensor cannot be
    constructed to be sent to matmul

    for(size_t j=0; j<M-1; ++j) {
        L(seq(j+1,M),j) /= L(j,j);
        L(seq(j+1,M),seq(j+1,M)) -= L(seq(j+1,M),j) % L(j,seq(j+1,M));
    }

    This is a non-modifying version that fills L and U directly and avoids
    the need for extracting L and U later
    The pre-requisite is that L should be Identity and U be a copy of A before
    the recursive routine starts

*/
template<size_t from, size_t to>
struct recursive_lu_impl {
    template<typename T, size_t M, size_t N>
    static FASTOR_INLINE void Do(Tensor<T,M,N> &L, Tensor<T,M,N> &U) {
        L(fseq<from+1,M>(),fix<from>) = U(fseq<from+1,M>(),fix<from>) / U.data()[from*N+from];
        U(fseq<from+1,M>(),fix<from>) = 0;
        U(fseq<from+1,M>(),fseq<from+1,M>()) -= matmul(L(fseq<from+1,M>(),fix<from>), U(fix<from>,fseq<from+1,M>()));
        recursive_lu_impl<from+1,to>::Do(L, U);
    }
};
template<size_t from>
struct recursive_lu_impl<from,from> {
    template<typename T, size_t M, size_t N>
    static FASTOR_INLINE void Do(Tensor<T,M,N> &L, Tensor<T,M,N> &U) {
        L(fseq<from+1,M>(),fix<from>) = U(fseq<from+1,M>(),fix<from>) / U.data()[from*N+from];
        U(fseq<from+1,M>(),fix<from>) = 0;
        U(fseq<from+1,M>(),fseq<from+1,M>()) -= matmul(L(fseq<from+1,M>(),fix<from>), U(fix<from>,fseq<from+1,M>()));
    }
};

template <typename T, size_t M, enable_if_t_<is_equal_v_<M,1>,bool> = false>
FASTOR_INLINE void recursive_lu_dispatcher(const Tensor<T,M,M>& A, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {
    L(0,0) = 1; U(0,0) = A(0,0);
}
template <typename T, size_t M, enable_if_t_<is_greater_v_<M,1>,bool> = false>
FASTOR_INLINE void recursive_lu_dispatcher(const Tensor<T,M,M>& A, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {
    // Requires M >=2
    U = A;
    L.eye2();
    recursive_lu_impl<0,M-2>::Do(L, U);
}


#if 0
// Recursive in-place LU using matmul/outer product
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
/* This in essence implements the following loop but with static views
    as dynamic views cannot be dispatched to matmul or a tensor cannot be
    constructed to be sent to matmul

    for(size_t j=0; j<M-1; ++j) {
        L(seq(j+1,M),j) /= L(j,j);
        L(seq(j+1,M),seq(j+1,M)) -= L(seq(j+1,M),j) % L(j,seq(j+1,M));
    }

    This is a self-modifying [in-place] version - if instead of a copy of
    (LU in this case) we pass the matrix itself it will decompose it in to
    an LU. Extracting L and U after this recursive decomposition is done
    almost beats the purpose performance wise

*/
template<size_t from, size_t to>
struct recursive_lu_impl_inplace {
    template<typename T, size_t M, size_t N>
    static FASTOR_INLINE void Do(Tensor<T,M,N> &LU) {
        LU(fseq<from+1,M>(),fix<from>) /= LU.data()[from*N+from];
        LU(fseq<from+1,M>(),fseq<from+1,M>()) -= matmul(LU(fseq<from+1,M>(),fix<from>), LU(fix<from>,fseq<from+1,M>()));
        recursive_lu_impl_inplace<from+1,to>::Do(LU);
    }
};
template<size_t from>
struct recursive_lu_impl_inplace<from,from> {
    template<typename T, size_t M, size_t N>
    static FASTOR_INLINE void Do(Tensor<T,M,N> &LU) {
        LU(fseq<from+1,M>(),fix<from>) /= LU.data()[from*N+from];
        LU(fseq<from+1,M>(),fseq<from+1,M>()) -= matmul(LU(fseq<from+1,M>(),fix<from>), LU(fix<from>,fseq<from+1,M>()));
    }
};


template <typename T, size_t M, enable_if_t_<is_equal_v_<M,1>,bool> = false>
FASTOR_INLINE void recursive_inplace_lu_dispatcher(Tensor<T,M,M>& A) {
    return;
}
template <typename T, size_t M, enable_if_t_<is_greater_v_<M,1>,bool> = false>
FASTOR_INLINE void recursive_inplace_lu_dispatcher(Tensor<T,M,M>& A) {
    // Requires M >=2
    recursive_lu_impl_inplace<0,M-2>::Do(A);
}

template <typename T, size_t M, enable_if_t_<is_equal_v_<M,1>,bool> = false>
FASTOR_INLINE void recursive_inplace_lu_dispatcher(const Tensor<T,M,M>& A, Tensor<T,M,M>& LU) {
    LU(0,0) = A(0,0);
}
template <typename T, size_t M, enable_if_t_<is_greater_v_<M,1>,bool> = false>
FASTOR_INLINE void recursive_inplace_lu_dispatcher(const Tensor<T,M,M>& A, Tensor<T,M,M>& LU) {
    // Requires M >=2
    LU =A;
    recursive_lu_impl_inplace<0,M-2>::Do(LU);
}

template <typename T, size_t M, enable_if_t_<is_equal_v_<M,1>,bool> = false>
FASTOR_INLINE void recursive_inplace_lu_dispatcher(const Tensor<T,M,M>& A, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {
    L(0,0) = 1; U(0,0) = A(0,0);
}
template <typename T, size_t M, enable_if_t_<is_greater_v_<M,1>,bool> = false>
FASTOR_INLINE void recursive_inplace_lu_dispatcher(const Tensor<T,M,M>& A, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {
    // Requires M >=2
    Tensor<T,M,M> LU(A);
    recursive_lu_impl_inplace<0,M-2>::Do(LU);

    // Extracting L and U after this recursive decomposition is done
    // almost beats the purpose performance wise
    for (size_t i=0; i<M; ++i) {
        L(i,i) = 1;
    }
    for (size_t i=0; i<M; ++i) {
        for (size_t j=0; j<i; ++j) {
            L(i,j) = LU(i,j);
        }
    }

    for (size_t i=0; i<M; ++i) {
        for (size_t j=i; j<M; ++j) {
            U(i,j) = LU(i,j);
        }
    }
}
#endif
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//


/* Block LU factorisation without pivoting */
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
template <typename T, size_t M, enable_if_t_<is_greater_v_<M,0> && is_less_equal_v_<M,8>,bool> = false>
FASTOR_INLINE void lu_block_dispatcher(const Tensor<T,M,M>& A, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {
    _lufact<T,M>(A.data(),L.data(),U.data());
}

template <typename T, size_t M, enable_if_t_<is_greater_v_<M,8> && is_less_equal_v_<M,32>,bool> = false>
FASTOR_INLINE void lu_block_dispatcher(const Tensor<T,M,M>& A, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {
    recursive_lu_dispatcher(A, L, U);
}

template <typename T, size_t M, enable_if_t_<is_greater_v_<M,32> && is_less_equal_v_<M,64>,bool> = false>
FASTOR_INLINE void lu_block_dispatcher(const Tensor<T,M,M>& A, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {

    // We will compute LU decomposition block-wise assuming that A11 and A22 are invertible
    //
    // [A11 A12]   [L11    0] [U11 U12]
    // [A21 A22]   [L21  L22] [0   U22]
    //
    // This results in
    //
    // A11 = L11 * U11
    // A12 = L11 * U12
    // A21 = L21 * U11
    // A22 = L21 * U12 - L22 * U22
    //
    // Hence we need to do LU factorisation once for A11 and once for A22

    // This is to avoid odd sizes for instance for size 35 we would
    // want to do 35 = 16 + 19 rather than 35 = 32 + 3 if the start size was 32
    constexpr size_t N = (M / 8UL * 8UL) / 2UL; // start size
    Tensor<T,N  ,N  > A11 = A(fseq<0,N>(),fseq<0,N>());
    Tensor<T,N  ,M-N> A12 = A(fseq<0,N>(),fseq<N,M>());
    Tensor<T,M-N,  N> A21 = A(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> A22 = A(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> L11(0), U11(0);
    lu_block_dispatcher(A11, L11, U11);

    // Solve for U12 = {L11}^(-1)*A12
    Tensor<T,N  ,M-N> U12 = tmatmul<UpLoType::Lower,UpLoType::General>(tinverse<InvCompType::SimpleInv, UpLoType::UniLower>(L11),A12);
    // Solve for L21 = A21*{U11}^(-1)
    Tensor<T,M-N,  N> L21 = tmatmul<UpLoType::General,UpLoType::Upper>(A21,tinverse<InvCompType::SimpleInv, UpLoType::Upper>(U11));

    Tensor<T,M-N,M-N> S   = A22 - matmul(L21,U12);

    Tensor<T,M-N,M-N> L22(0), U22(0);
    lu_block_dispatcher(S, L22, U22);

    L(fseq<0,N>(),fseq<0,N>()) = L11;
    // L(fseq<0,N>(),fseq<N,M>()) = 0;
    L(fseq<N,M>(),fseq<0,N>()) = L21;
    L(fseq<N,M>(),fseq<N,M>()) = L22;

    U(fseq<0,N>(),fseq<0,N>()) = U11;
    U(fseq<0,N>(),fseq<N,M>()) = U12;
    // U(fseq<N,M>(),fseq<0,N>()) = 0;
    U(fseq<N,M>(),fseq<N,M>()) = U22;
}

// Conditional dispatch
namespace useless {
template <typename T, size_t M, enable_if_t_<is_greater_v_<M,0> && is_less_equal_v_<M,64>,bool> = false>
FASTOR_INLINE void lu_block_simple_dispatcher(const Tensor<T,M,M>& A, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {
    lu_block_dispatcher(A, L, U);
    return;
}
template <typename T, size_t M, enable_if_t_<is_greater_v_<M,64>,bool> = false>
FASTOR_INLINE void lu_block_simple_dispatcher(const Tensor<T,M,M>& A, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {
    // lu_simple_dispatcher(A, L, U);
    recursive_lu_dispatcher(A, L, U);
    return;
}
} // useless

/* For sizes greater than 64 we tile differently to avoid too many recursions
*/
template <typename T, size_t M, enable_if_t_<is_greater_v_<M,64>,bool> = false>
FASTOR_INLINE void lu_block_dispatcher(const Tensor<T,M,M>& A, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {

    // We will compute LU decomposition block-wise assuming that A11 and A22 are invertible
    //
    // [A11 A12]   [L11    0] [U11 U12]
    // [A21 A22]   [L21  L22] [0   U22]
    //
    // This results in
    //
    // A11 = L11 * U11
    // A12 = L11 * U12
    // A21 = L21 * U11
    // A22 = L21 * U12 - L22 * U22
    //
    // Hence we need to do LU factorisation once for A11 and once for A22

    // This is to avoid odd sizes for instance for size 65 we would
    // want to do 65 = 32 + 33 rather than 65 = 64 + 1 if the start size was 64
    constexpr size_t N = (M / 16UL * 16UL) / 2UL; // start size
    Tensor<T,N  ,N  > A11 = A(fseq<0,N>(),fseq<0,N>());
    Tensor<T,N  ,M-N> A12 = A(fseq<0,N>(),fseq<N,M>());
    Tensor<T,M-N,  N> A21 = A(fseq<N,M>(),fseq<0,N>());
    Tensor<T,M-N,M-N> A22 = A(fseq<N,M>(),fseq<N,M>());

    Tensor<T,N,N> L11(0), U11(0);
    useless::lu_block_simple_dispatcher(A11, L11, U11);

    // Solve for U12 = {L11}^(-1)*A12
    Tensor<T,N  ,M-N> U12 = tmatmul<UpLoType::Lower,UpLoType::General>(tinverse<InvCompType::SimpleInv, UpLoType::UniLower>(L11),A12);
    // Ideally use forward_subs but its iterative nature makes it less efficient than tmatmul
    // Tensor<T,N  ,M-N> U12 = forward_subs(L11,A12);
    // Solve for L21 = A21*{U11}^(-1)
    Tensor<T,M-N,  N> L21 = tmatmul<UpLoType::General,UpLoType::Upper>(A21,tinverse<InvCompType::SimpleInv, UpLoType::Upper>(U11));
    // Not quite performant as we can't avoid the matmul here
    // Tensor<T,N  ,N  > I; I.eye2();
    // Tensor<T,M-N,  N> L21 = matmul(A21, backward_subs(U11, I));

    Tensor<T,M-N,M-N> S   = A22 - matmul(L21,U12);

    Tensor<T,M-N,M-N> L22(0), U22(0);
    useless::lu_block_simple_dispatcher(S, L22, U22);

    L(fseq<0,N>(),fseq<0,N>()) = L11;
    // L(fseq<0,N>(),fseq<N,M>()) = 0;
    L(fseq<N,M>(),fseq<0,N>()) = L21;
    L(fseq<N,M>(),fseq<N,M>()) = L22;

    U(fseq<0,N>(),fseq<0,N>()) = U11;
    U(fseq<0,N>(),fseq<N,M>()) = U12;
    // U(fseq<N,M>(),fseq<0,N>()) = 0;
    U(fseq<N,M>(),fseq<N,M>()) = U22;
}
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
} // internal


/* Block LU factorisation overloads */
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
// BlockLU - no pivot
template<LUCompType LUType = LUCompType::BlockLU, typename Expr, size_t DIM0, typename T, size_t M,
    enable_if_t_<is_tensor_v<Expr> && LUType == LUCompType::BlockLU,bool> = false>
FASTOR_INLINE
void
lu(const AbstractTensor<Expr,DIM0> &src, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {
    L.fill(0);
    U.fill(0);
    internal::lu_block_dispatcher(src.self(),L,U);
}
template<LUCompType LUType = LUCompType::BlockLU, typename Expr, size_t DIM0, typename T, size_t M,
    enable_if_t_<!is_tensor_v<Expr> && LUType == LUCompType::BlockLU,bool> = false>
FASTOR_INLINE
void
lu(const AbstractTensor<Expr,DIM0> &src, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {
    L.fill(0);
    U.fill(0);
    typename Expr::result_type tmp(src.self());
    internal::lu_block_dispatcher(tmp,L,U);
}

// BlockLU - vector pivot
template<LUCompType LUType = LUCompType::BlockLU, typename Expr, size_t DIM0, typename T, size_t M,
    enable_if_t_<is_tensor_v<Expr> && LUType == LUCompType::BlockLUPiv,bool> = false>
FASTOR_INLINE
void
lu(const AbstractTensor<Expr,DIM0> &src, Tensor<T,M,M>& L, Tensor<T,M,M>& U, Tensor<size_t,M>& P) {
    L.fill(0);
    U.fill(0);
    pivot_inplace(src.self(),P);
    auto A(apply_pivot(src.self(),P));
    internal::lu_block_dispatcher(A,L,U);
}
template<LUCompType LUType = LUCompType::BlockLU, typename Expr, size_t DIM0, typename T, size_t M,
    enable_if_t_<!is_tensor_v<Expr> && LUType == LUCompType::BlockLUPiv,bool> = false>
FASTOR_INLINE
void
lu(const AbstractTensor<Expr,DIM0> &src, Tensor<T,M,M>& L, Tensor<T,M,M>& U, Tensor<size_t,M>& P) {
    L.fill(0);
    U.fill(0);
    typename Expr::result_type A(src.self());
    pivot_inplace(A,P);
    // Modify A as A is a temporary anyway
    apply_pivot_inplace(A,P);
    internal::lu_block_dispatcher(A,L,U);
}

// BlockLU - matrix pivot
template<LUCompType LUType = LUCompType::BlockLU, typename Expr, size_t DIM0, typename T, size_t M,
    enable_if_t_<is_tensor_v<Expr> && LUType == LUCompType::BlockLUPiv,bool> = false>
FASTOR_INLINE
void
lu(const AbstractTensor<Expr,DIM0> &src, Tensor<T,M,M>& L, Tensor<T,M,M>& U, Tensor<T,M,M>& P) {
    L.fill(0);
    U.fill(0);
    pivot_inplace(src.self(),P);
    auto A(apply_pivot(src.self(),P));
    internal::lu_block_dispatcher(A,L,U);
}
template<LUCompType LUType = LUCompType::BlockLU, typename Expr, size_t DIM0, typename T, size_t M,
    enable_if_t_<!is_tensor_v<Expr> && LUType == LUCompType::BlockLUPiv,bool> = false>
FASTOR_INLINE
void
lu(const AbstractTensor<Expr,DIM0> &src, Tensor<T,M,M>& L, Tensor<T,M,M>& U, Tensor<T,M,M>& P) {
    L.fill(0);
    U.fill(0);
    typename Expr::result_type A(src.self());
    pivot_inplace(A,P);
    // Modify A as A is a temporary anyway
    apply_pivot_inplace(A,P);
    internal::lu_block_dispatcher(A,L,U);
}
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//



/* Simple LU factorisation overloads */
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
// SimpleLU - no pivot
template<LUCompType LUType = LUCompType::BlockLU, typename Expr, size_t DIM0, typename T, size_t M,
    enable_if_t_<is_tensor_v<Expr> && LUType == LUCompType::SimpleLU,bool> = false>
FASTOR_INLINE
void
lu(const AbstractTensor<Expr,DIM0> &src, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {
    internal::lu_simple_dispatcher(src.self(),L,U);
}
template<LUCompType LUType = LUCompType::BlockLU, typename Expr, size_t DIM0, typename T, size_t M,
    enable_if_t_<!is_tensor_v<Expr> && LUType == LUCompType::SimpleLU,bool> = false>
FASTOR_INLINE
void
lu(const AbstractTensor<Expr,DIM0> &src, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {
    typename Expr::result_type tmp(src.self());
    internal::lu_simple_dispatcher(tmp,L,U);
}

// SimpleLU - vector pivot
template<LUCompType LUType = LUCompType::BlockLU, typename Expr, size_t DIM0, typename T, size_t M,
    enable_if_t_<is_tensor_v<Expr> && LUType == LUCompType::SimpleLUPiv,bool> = false>
FASTOR_INLINE
void
lu(const AbstractTensor<Expr,DIM0> &src, Tensor<T,M,M>& L, Tensor<T,M,M>& U, Tensor<size_t,M>& P) {
    pivot_inplace(src.self(),P);
    auto A(apply_pivot(src.self(),P));
    internal::lu_simple_dispatcher(A,L,U);
}
template<LUCompType LUType = LUCompType::BlockLU, typename Expr, size_t DIM0, typename T, size_t M,
    enable_if_t_<!is_tensor_v<Expr> && LUType == LUCompType::SimpleLUPiv,bool> = false>
FASTOR_INLINE
void
lu(const AbstractTensor<Expr,DIM0> &src, Tensor<T,M,M>& L, Tensor<T,M,M>& U, Tensor<size_t,M>& P) {
    typename Expr::result_type A(src.self());
    pivot_inplace(A,P);
    // Modify A as A is a temporary anyway
    apply_pivot_inplace(A,P);
    internal::lu_simple_dispatcher(A,L,U);
}

// SimpleLU - matrix pivot
template<LUCompType LUType = LUCompType::BlockLU, typename Expr, size_t DIM0, typename T, size_t M,
    enable_if_t_<is_tensor_v<Expr> && LUType == LUCompType::SimpleLUPiv,bool> = false>
FASTOR_INLINE
void
lu(const AbstractTensor<Expr,DIM0> &src, Tensor<T,M,M>& L, Tensor<T,M,M>& U, Tensor<T,M,M>& P) {
    pivot_inplace(src.self(),P);
    auto A(apply_pivot(src.self(),P));
    internal::lu_simple_dispatcher(A,L,U);
}
template<LUCompType LUType = LUCompType::BlockLU, typename Expr, size_t DIM0, typename T, size_t M,
    enable_if_t_<!is_tensor_v<Expr> && LUType == LUCompType::SimpleLUPiv,bool> = false>
FASTOR_INLINE
void
lu(const AbstractTensor<Expr,DIM0> &src, Tensor<T,M,M>& L, Tensor<T,M,M>& U, Tensor<T,M,M>& P) {
    typename Expr::result_type A(src.self());
    pivot_inplace(A,P);
    // Modify A as A is a temporary anyway
    apply_pivot_inplace(A,P);
    internal::lu_simple_dispatcher(A,L,U);
}
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//








// Inversion using LU
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
namespace internal {

template<typename T, size_t M>
FASTOR_INLINE Tensor<T,M,M> get_lu_inverse(const Tensor<T,M,M> &L, const Tensor<T,M,M> &U) {
    // We will solve for multiple RHS [B = I]
    // Loop over columns of B = I
    Tensor<T,M,M> I; I.eye2();
    Tensor<T,M,M> Y = forward_subs(L, I);
    Tensor<T,M,M> X = backward_subs(U, Y);
    return X;
}

template<typename T, size_t M>
FASTOR_INLINE Tensor<T,M,M> get_lu_inverse(Tensor<T,M,M> &L, const Tensor<T,M,M> &U, const Tensor<size_t,M> &p) {
    // We will solve for multiple RHS [B = I]
    // Loop over columns of B = I
    Tensor<T,M,M> I; I.eye2();
    Tensor<T,M,M> Y = forward_subs(L, p, I);
    Tensor<T,M,M> X = backward_subs(U, Y);
    return X;
}
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
} // internal


// SimpleLU
template<InvCompType InvType = InvCompType::SimpleInv,
    typename T, size_t M, enable_if_t_<InvType == InvCompType::SimpleLU,bool> = false>
FASTOR_INLINE Tensor<T,M,M> inverse(const Tensor<T,M,M> &A) {
    Tensor<T,M,M> L, U;
    lu<LUCompType::SimpleLU>(A, L, U);
    return internal::get_lu_inverse(L, U);
}

// BlockLU
template<InvCompType InvType = InvCompType::SimpleInv,
    typename T, size_t M, enable_if_t_<InvType == InvCompType::BlockLU,bool> = false>
FASTOR_INLINE Tensor<T,M,M> inverse(const Tensor<T,M,M> &A) {
    Tensor<T,M,M> L, U;
    lu<LUCompType::BlockLU>(A, L, U);
    return internal::get_lu_inverse(L, U);
}

// SimpleLUPiv
template<InvCompType InvType = InvCompType::SimpleInv,
    typename T, size_t M, enable_if_t_<InvType == InvCompType::SimpleLUPiv,bool> = false>
FASTOR_INLINE Tensor<T,M,M> inverse(const Tensor<T,M,M> &A) {
    Tensor<T,M,M> L, U;
    Tensor<size_t,M> p;
    lu<LUCompType::SimpleLUPiv>(A, L, U, p);
    return internal::get_lu_inverse(L, U, p);
}

// BlockLUPiv
template<InvCompType InvType = InvCompType::SimpleInv,
    typename T, size_t M, enable_if_t_<InvType == InvCompType::BlockLUPiv,bool> = false>
FASTOR_INLINE Tensor<T,M,M> inverse(const Tensor<T,M,M> &A) {
    Tensor<T,M,M> L, U;
    Tensor<size_t,M> p;
    lu<LUCompType::BlockLUPiv>(A, L, U, p);
    return internal::get_lu_inverse(L, U, p);
}
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//









// Solving linear system of equations using LU
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
namespace internal {

template<typename T, size_t M>
FASTOR_INLINE Tensor<T,M> get_lu_solve(const Tensor<T,M,M> &L, const Tensor<T,M,M> &U, const Tensor<T,M> &b) {
    Tensor<T,M> y = forward_subs(L, b);
    Tensor<T,M> x = backward_subs(U, y);
    return x;
}

template<typename T, size_t M>
FASTOR_INLINE Tensor<T,M> get_lu_solve(Tensor<T,M,M> &L, const Tensor<T,M,M> &U, const Tensor<size_t,M> &p, const Tensor<T,M> &b) {
    Tensor<T,M> y = forward_subs(L, p, b);
    Tensor<T,M> x = backward_subs(U, y);
    return x;
}

// Multiple RHS
template<typename T, size_t M, size_t N>
FASTOR_INLINE Tensor<T,M,N> get_lu_solve(const Tensor<T,M,M> &L, const Tensor<T,M,M> &U, const Tensor<T,M,N> &B) {
    Tensor<T,M,N> Y = forward_subs(L, B);
    Tensor<T,M,N> X = backward_subs(U, Y);
    return X;
}

template<typename T, size_t M, size_t N>
FASTOR_INLINE Tensor<T,M,N> get_lu_solve(const Tensor<T,M,M> &L, const Tensor<T,M,M> &U, const Tensor<size_t,M> &p, const Tensor<T,M,N> &B) {
    Tensor<T,M,N> Y = forward_subs(L, p, B);
    Tensor<T,M,N> X = backward_subs(U, Y);
    return X;
}
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
} // internal


// Single RHS
// SimpleLU - no pivot
template<SolveCompType SType = SolveCompType::SimpleInv, typename T, size_t M,
    enable_if_t_< SType == SolveCompType::SimpleLU, bool> = false>
FASTOR_INLINE Tensor<T,M> solve(const Tensor<T,M,M> &A, const Tensor<T,M> &b) {
    Tensor<T,M,M> L, U;
    lu<LUCompType::SimpleLU>(A, L, U);
    return internal::get_lu_solve(L, U, b);
}

// SimpleLU - pivot
template<SolveCompType SType = SolveCompType::SimpleInv, typename T, size_t M,
    enable_if_t_< SType == SolveCompType::SimpleLUPiv, bool> = false>
FASTOR_INLINE Tensor<T,M> solve(const Tensor<T,M,M> &A, const Tensor<T,M> &b) {
    Tensor<T,M,M> L, U;
    Tensor<size_t,M> p;
    lu<LUCompType::SimpleLUPiv>(A, L, U, p);
    return internal::get_lu_solve(L, U, p, b);
}

// BlockLU - no pivot
template<SolveCompType SType = SolveCompType::SimpleInv, typename T, size_t M,
    enable_if_t_< SType == SolveCompType::BlockLU, bool> = false>
FASTOR_INLINE Tensor<T,M> solve(const Tensor<T,M,M> &A, const Tensor<T,M> &b) {
    Tensor<T,M,M> L, U;
    lu<LUCompType::BlockLU>(A, L, U);
    return internal::get_lu_solve(L, U, b);
}

// BlockLU - pivot
template<SolveCompType SType = SolveCompType::SimpleInv, typename T, size_t M,
    enable_if_t_< SType == SolveCompType::BlockLUPiv, bool> = false>
FASTOR_INLINE Tensor<T,M> solve(const Tensor<T,M,M> &A, const Tensor<T,M> &b) {
    Tensor<T,M,M> L, U;
    Tensor<size_t,M> p;
    lu<LUCompType::BlockLUPiv>(A, L, U, p);
    return internal::get_lu_solve(L, U, p, b);
}

// Multiple RHS
// SimpleLU - no pivot
template<SolveCompType SType = SolveCompType::SimpleInv, typename T, size_t M, size_t N,
    enable_if_t_< SType == SolveCompType::SimpleLU, bool> = false>
FASTOR_INLINE Tensor<T,M,N> solve(const Tensor<T,M,M> &A, const Tensor<T,M,N> &B) {
    Tensor<T,M,M> L, U;
    lu<LUCompType::SimpleLU>(A, L, U);
    return internal::get_lu_solve(L, U, B);
}

// SimpleLU - pivot
template<SolveCompType SType = SolveCompType::SimpleInv, typename T, size_t M, size_t N,
    enable_if_t_< SType == SolveCompType::SimpleLUPiv, bool> = false>
FASTOR_INLINE Tensor<T,M,N> solve(const Tensor<T,M,M> &A, const Tensor<T,M,N> &B) {
    Tensor<T,M,M> L, U;
    Tensor<size_t,M> p;
    lu<LUCompType::SimpleLUPiv>(A, L, U, p);
    return internal::get_lu_solve(L, U, p, B);
}

// SimpleLU - no pivot
template<SolveCompType SType = SolveCompType::SimpleInv, typename T, size_t M, size_t N,
    enable_if_t_< SType == SolveCompType::BlockLU, bool> = false>
FASTOR_INLINE Tensor<T,M,N> solve(const Tensor<T,M,M> &A, const Tensor<T,M,N> &B) {
    Tensor<T,M,M> L, U;
    lu<LUCompType::BlockLU>(A, L, U);
    return internal::get_lu_solve(L, U, B);
}

// SimpleLU - pivot
template<SolveCompType SType = SolveCompType::SimpleInv, typename T, size_t M, size_t N,
    enable_if_t_< SType == SolveCompType::BlockLUPiv, bool> = false>
FASTOR_INLINE Tensor<T,M,N> solve(const Tensor<T,M,M> &A, const Tensor<T,M,N> &B) {
    Tensor<T,M,M> L, U;
    Tensor<size_t,M> p;
    lu<LUCompType::BlockLUPiv>(A, L, U, p);
    return internal::get_lu_solve(L, U, p, B);
}
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//










// Computing determinant using LU
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
template<DetCompType DetType = DetCompType::Simple, typename T, size_t M,
    enable_if_t_<DetType == DetCompType::LU,bool> = false>
FASTOR_INLINE T determinant(const Tensor<T,M,M> &A) {
    int nswaps = internal::count_swaps(A) % 2UL == 0 ? 1 : -1;
    Tensor<T,M,M> L, U;
    Tensor<size_t,M> p;
    lu<LUCompType::BlockLUPiv>(A, L, U, p);
    return product(diag(U)) * nswaps;
}
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//


} // end of namespace Fastor


#endif // UNARY_LU_OP_H
