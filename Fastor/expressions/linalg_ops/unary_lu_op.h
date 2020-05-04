#ifndef UNARY_LU_OP_H
#define UNARY_LU_OP_H

#include "Fastor/meta/meta.h"
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

template<typename T, size_t M, size_t N>
FASTOR_INLINE Tensor<T,M,N> forward_subs(const Tensor<T,M,M> &L, const Tensor<T,M,N> &B) {
    Tensor<T,M,N> X(0);
    for (size_t j=0; j < N; ++j) {
        // Solve for L * y = b
        for (size_t i=0; i< M; ++i) {
            T value = 0;
            for (size_t k=0; k<i; ++k) {
                value += L(i,k)*X(k,j);
            }
            X(i,j) = B(i,j) - value;
        }
    }
    return X;

    // Tensor<T,M,N> X;
    // for (size_t j=0; j < N; ++j) {
    //     // Solve for L * y = b
    //     Tensor<T,M> y(0); y(j) = 1;
    //     X(all,j) = B(all,j) - matmul(L,y);
    // }
    // return X;
}

template<typename T, size_t M, size_t N>
FASTOR_INLINE Tensor<T,M,N> backward_subs(const Tensor<T,M,M> &U, const Tensor<T,M,N> &B) {
    Tensor<T,M,N> X(0);
    for (size_t j=0; j < N; ++j) {
        // Solve for of U * x = y
        for (int i= int(M) - 1; i>=0; --i) {
            T value = 0;
            for (int k=i; k<int(M); ++k) {
                value += U(i,k)*X(k,j);
            }
            X(i,j) = (B(i,j) - value) / U(i, i);
        }
    }
    return X;

    // Tensor<T,M,N> X;
    // for (size_t j=0; j < N; ++j) {
    //     // Solve for L * y = b
    //     Tensor<T,M> y(0); y(j) = 1;
    //     X(all,j) = ( B(all,j) - matmul(U,y) ) / U(j,j);
    // }
    // return X;
}

// template<typename T, size_t M, size_t N>
// FASTOR_INLINE Tensor<T,M> forward_subs(const Tensor<T,M,M> &L, const Tensor<T,M> &b) {

//     Tensor<T,M> X;
//     // Solve for L * y = b
//     Tensor<T,M> y(0); y(0) = 1;
//     X = b - matmul(L,y);
//     return X;
// }

// template<typename T, size_t M, size_t N>
// FASTOR_INLINE Tensor<T,M> backward_subs(const Tensor<T,M,M> &U, const Tensor<T,M> &b) {

//     Tensor<T,M> X;
//     Tensor<T,M> y(0); y(0) = 1;
//     X = ( b - matmul(U,y) ) / U(0,0);
//     return X;
// }

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



// Recursive LU using matmul
//-----------------------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------------------//
/* This in essence implements the following loop but with static views
    as dynamic views cannot be dispatched to matmul or a tensor cannot be
    constructed to be sent to matmul

    for(size_t j=0; j<M-1; ++j) {
        L(seq(j+1,M),j) /= L(j,j);
        L(seq(j+1,M),seq(j+1,M)) -= L(seq(j+1,M),j) % L(j,seq(j+1,M));
    }

*/
template<size_t from, size_t to>
struct recursive_lu_impl {
    template<typename T, size_t M, size_t N>
    static FASTOR_INLINE void Do(Tensor<T,M,N> &LU) {
        LU(fseq<from+1,M>(),fix<from>) /= LU.data()[from*N+from];
        LU(fseq<from+1,M>(),fseq<from+1,M>()) -= LU(fseq<from+1,M>(),fix<from>) % LU(fix<from>,fseq<from+1,M>());
        recursive_lu_impl<from+1,to>::Do(LU);
    }
};
template<size_t from>
struct recursive_lu_impl<from,from> {
    template<typename T, size_t M, size_t N>
    static FASTOR_INLINE void Do(Tensor<T,M,N> &LU) {
        LU(fseq<from+1,M>(),fix<from>) /= LU.data()[from*N+from];
        LU(fseq<from+1,M>(),fseq<from+1,M>()) -= LU(fseq<from+1,M>(),fix<from>) % LU(fix<from>,fseq<from+1,M>());
    }
};


template <typename T, size_t M, enable_if_t_<is_equal_v_<M,1>,bool> = false>
FASTOR_INLINE void recursive_lu_dispatcher(Tensor<T,M,M>& A) {
    return;
}
template <typename T, size_t M, enable_if_t_<is_greater_v_<M,1>,bool> = false>
FASTOR_INLINE void recursive_lu_dispatcher(Tensor<T,M,M>& A) {
    // For this M >=2
    recursive_lu_impl<0,M-2>::Do(A);
}

template <typename T, size_t M, enable_if_t_<is_equal_v_<M,1>,bool> = false>
FASTOR_INLINE void recursive_lu_dispatcher(const Tensor<T,M,M>& A, Tensor<T,M,M>& LU) {
    LU(0,0) = A(0,0);
}
template <typename T, size_t M, enable_if_t_<is_greater_v_<M,1>,bool> = false>
FASTOR_INLINE void recursive_lu_dispatcher(const Tensor<T,M,M>& A, Tensor<T,M,M>& LU) {
    // For this M >=2
    LU(A);
    recursive_lu_impl<0,M-2>::Do(LU);
}

template <typename T, size_t M, enable_if_t_<is_equal_v_<M,1>,bool> = false>
FASTOR_INLINE void recursive_lu_dispatcher(const Tensor<T,M,M>& A, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {
    L(0,0) = 1; U(0,0) = A(0,0);
}
template <typename T, size_t M, enable_if_t_<is_greater_v_<M,1>,bool> = false>
FASTOR_INLINE void recursive_lu_dispatcher(const Tensor<T,M,M>& A, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {
    // For this M >=2
    Tensor<T,M,M> LU(A);
    recursive_lu_impl<0,M-2>::Do(LU);

    // Creating L & U impacts the performance for small sizes
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

// template <typename T, size_t M, enable_if_t_<is_greater_v_<M,8> && is_less_equal_v_<M,16>,bool> = false>
// FASTOR_INLINE void lu_block_dispatcher(const Tensor<T,M,M>& A, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {

//     // We will compute LU decomposition block-wise assuming that A11 and A22 are invertible
//     //
//     // [A11 A12]   [L11    0] [U11 U12]
//     // [A21 A22]   [L21  L22] [0   U22]
//     //
//     // This results in
//     //
//     // A11 = L11 * U11
//     // A12 = L11 * U12
//     // A21 = L21 * U11
//     // A22 = L21 * U12 - L22 * U22
//     //
//     // Hence we need to LU factorisation one for A11 and one for A22

//     constexpr size_t N = 8UL; // start size
//     Tensor<T,N  ,N  > A11 = A(fseq<0,N>(),fseq<0,N>());
//     Tensor<T,N  ,M-N> A12 = A(fseq<0,N>(),fseq<N,M>());
//     Tensor<T,M-N,  N> A21 = A(fseq<N,M>(),fseq<0,N>());
//     Tensor<T,M-N,M-N> A22 = A(fseq<N,M>(),fseq<N,M>());

//     // Don't zero out in the first block recursion as this dipatches to _lufact
//     // which zeros out the tensors anyway
//     Tensor<T,N,N> L11, U11;
//     lu_block_dispatcher(A11, L11, U11);

//     // Solve for U12 = {L11}^(-1)*A12
//     Tensor<T,N  ,M-N> U12 = matmul(inverse<InvCompType::SimpleInv>(L11),A12);
//     // Solve for L21 = A21*{U11}^(-1)
//     Tensor<T,M-N,  N> L21 = matmul(A21,inverse<InvCompType::SimpleInv>(U11));

//     Tensor<T,M-N,M-N> S   = A22 - matmul(L21,U12);

//     Tensor<T,M-N,M-N> L22, U22;
//     lu_block_dispatcher(S, L22, U22);

//     L(fseq<0,N>(),fseq<0,N>()) = L11;
//     // L(fseq<0,N>(),fseq<N,M>()) = 0;
//     L(fseq<N,M>(),fseq<0,N>()) = L21;
//     L(fseq<N,M>(),fseq<N,M>()) = L22;

//     U(fseq<0,N>(),fseq<0,N>()) = U11;
//     U(fseq<0,N>(),fseq<N,M>()) = U12;
//     // U(fseq<N,M>(),fseq<0,N>()) = 0;
//     U(fseq<N,M>(),fseq<N,M>()) = U22;
// }

// template <typename T, size_t M, enable_if_t_<is_greater_v_<M,16> && is_less_equal_v_<M,32>,bool> = false>
// FASTOR_INLINE void lu_block_dispatcher(const Tensor<T,M,M>& A, Tensor<T,M,M>& L, Tensor<T,M,M>& U) {

//     // We will compute LU decomposition block-wise assuming that A11 and A22 are invertible
//     //
//     // [A11 A12]   [L11    0] [U11 U12]
//     // [A21 A22]   [L21  L22] [0   U22]
//     //
//     // This results in
//     //
//     // A11 = L11 * U11
//     // A12 = L11 * U12
//     // A21 = L21 * U11
//     // A22 = L21 * U12 - L22 * U22
//     //
//     // Hence we need to LU factorisation one for A11 and one for A22

//     constexpr size_t N = 16UL; // start size
//     Tensor<T,N  ,N  > A11 = A(fseq<0,N>(),fseq<0,N>());
//     Tensor<T,N  ,M-N> A12 = A(fseq<0,N>(),fseq<N,M>());
//     Tensor<T,M-N,  N> A21 = A(fseq<N,M>(),fseq<0,N>());
//     Tensor<T,M-N,M-N> A22 = A(fseq<N,M>(),fseq<N,M>());

//     Tensor<T,N,N> L11(0), U11(0);
//     lu_block_dispatcher(A11, L11, U11);

//     // Solve for U12 = {L11}^(-1)*A12
//     Tensor<T,N  ,M-N> U12 = matmul(inverse<InvCompType::SimpleInv>(L11),A12);
//     // Solve for L21 = A21*{U11}^(-1)
//     Tensor<T,M-N,  N> L21 = matmul(A21,inverse<InvCompType::SimpleInv>(U11));

//     Tensor<T,M-N,M-N> S   = A22 - matmul(L21,U12);

//     Tensor<T,M-N,M-N> L22(0), U22(0);
//     lu_block_dispatcher(S, L22, U22);

//     L(fseq<0,N>(),fseq<0,N>()) = L11;
//     // L(fseq<0,N>(),fseq<N,M>()) = 0;
//     L(fseq<N,M>(),fseq<0,N>()) = L21;
//     L(fseq<N,M>(),fseq<N,M>()) = L22;

//     U(fseq<0,N>(),fseq<0,N>()) = U11;
//     U(fseq<0,N>(),fseq<N,M>()) = U12;
//     // U(fseq<N,M>(),fseq<0,N>()) = 0;
//     U(fseq<N,M>(),fseq<N,M>()) = U22;
// }

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
    // Hence we need to LU factorisation one for A11 and one for A22

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
    Tensor<T,N  ,M-N> U12 = matmul(inverse<InvCompType::SimpleInv>(L11),A12);
    // Solve for L21 = A21*{U11}^(-1)
    Tensor<T,M-N,  N> L21 = matmul(A21,inverse<InvCompType::SimpleInv>(U11));

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
    We tile based on half of the matrix size so for a 35x35 panel we will get
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
    // Hence we need to LU factorisation one for A11 and one for A22

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
    Tensor<T,N  ,M-N> U12 = matmul(inverse<InvCompType::SimpleInv>(L11),A12);
    // Solve for L21 = A21*{U11}^(-1)
    Tensor<T,M-N,  N> L21 = matmul(A21,inverse<InvCompType::SimpleInv>(U11));

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

    Tensor<T,M,M> out(0);
    // We will solve for multiple RHS [B = I]
    // Loop over columns of B = I
    for (size_t j = 0; j<M; ++j) {
        // Solve for L * y = b
        Tensor<T,M> y(0); y(j) = 1;
        for (size_t i=0; i< M; ++i) {
            // if (i==j) y(i) = 1;
            T value = 0;
            for (size_t k=0; k<i; ++k) {
                value += L(i,k)*y(k);
            }
            y(i) -= value;
        }
        // Solve for of U * x = y
        // Each x is a column of out
        for (int i = int(M) - 1; i>=0; --i) {
            T value = 0;
            for (int k=i; k<int(M); ++k) {
                value += U(i,k)*out(k,j);
            }
            out(i,j) = (y(i) - value) / U(i, i);
        }
    }

    return out;
}

template<typename T, size_t M>
FASTOR_INLINE Tensor<T,M,M> get_lu_inverse(Tensor<T,M,M> &L, const Tensor<T,M,M> &U, const Tensor<size_t,M> &P) {

    Tensor<T,M,M> out(0);
    // We will solve for multiple RHS [B = I]
    // Loop over columns of B = I
    for (size_t j = 0; j<M; ++j) {
        // Solve for L * y = P*b
        Tensor<T,M> y(0);
        for (size_t i=0; i< M; ++i) {
            if (P(i)==j) y(i) = 1;
            T value = 0;
            for (size_t k=0; k<i; ++k) {
                value += L(i,k)*y(k);
            }
            y(i) -= value;
        }
        // Solve for of U * x = y
        // Each x is a column of out
        for (int i = int(M) - 1; i>=0; --i) {
            T value = 0;
            for (int k=i; k<int(M); ++k) {
                value += U(i,k)*out(k,j);
            }
            out(i,j) = (y(i) - value) / U(i, i);
        }
    }

    return out;
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

    Tensor<T,M> x(0), y(0);
    // Solve for L * y = b
    for (size_t i=0; i< M; ++i) {
        T value = 0;
        for (size_t k=0; k<i; ++k) {
            value += L(i,k)*y(k);
        }
        y(i) = b(i) - value;
    }
    // Solve for of U * x = y
    for (int i= int(M) - 1; i>=0; --i) {
        T value = 0;
        for (int k=i; k<int(M); ++k) {
            value += U(i,k)*x(k);
        }
        x(i) = (y(i) - value) / U(i, i);
    }

    return x;
}

template<typename T, size_t M>
FASTOR_INLINE Tensor<T,M> get_lu_solve(Tensor<T,M,M> &L, const Tensor<T,M,M> &U, const Tensor<size_t,M> &P, const Tensor<T,M> &b) {

    Tensor<T,M> x(0), y(0);
    // Solve for L * y = b
    for (size_t i=0; i< M; ++i) {
        T value = 0;
        for (size_t k=0; k<i; ++k) {
            value += L(i,k)*y(k);
        }
        y(i) = b(P(i)) - value;
    }
    // Solve for of U * x = y
    for (int i= int(M) - 1; i>=0; --i) {
        T value = 0;
        for (int k=i; k<int(M); ++k) {
            value += U(i,k)*x(k);
        }
        x(i) = (y(i) - value) / U(i, i);
    }

    return x;
}

// Multiple RHS
template<typename T, size_t M, size_t N>
FASTOR_INLINE Tensor<T,M,N> get_lu_solve(const Tensor<T,M,M> &L, const Tensor<T,M,M> &U, const Tensor<T,M,N> &B) {

    Tensor<T,M,N> X(0);

    for (size_t j=0; j < N; ++j) {
        Tensor<T,M> y(0);
        // Solve for L * y = b
        for (size_t i=0; i< M; ++i) {
            T value = 0;
            for (size_t k=0; k<i; ++k) {
                value += L(i,k)*y(k);
            }
            y(i) = B(i,j) - value;
        }
        // Solve for of U * x = y
        for (int i= int(M) - 1; i>=0; --i) {
            T value = 0;
            for (int k=i; k<int(M); ++k) {
                value += U(i,k)*X(k,j);
            }
            X(i,j) = (y(i) - value) / U(i, i);
        }
    }

    return X;
}

template<typename T, size_t M, size_t N>
FASTOR_INLINE Tensor<T,M,N> get_lu_solve(const Tensor<T,M,M> &L, const Tensor<T,M,M> &U, const Tensor<size_t,M> &P, const Tensor<T,M,N> &B) {

    Tensor<T,M,N> X(0);

    for (size_t j=0; j < N; ++j) {
        Tensor<T,M> y(0);
        // Solve for L * y = b
        for (size_t i=0; i< M; ++i) {
            T value = 0;
            for (size_t k=0; k<i; ++k) {
                value += L(i,k)*y(k);
            }
            y(i) = B(P(i),j) - value;
        }
        // Solve for of U * x = y
        for (int i= int(M) - 1; i>=0; --i) {
            T value = 0;
            for (int k=i; k<int(M); ++k) {
                value += U(i,k)*X(k,j);
            }
            X(i,j) = (y(i) - value) / U(i, i);
        }
    }

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

} // end of namespace Fastor


#endif // UNARY_LU_OP_H