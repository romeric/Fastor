#ifndef UNARY_PIV_OP_H
#define UNARY_PIV_OP_H

#include "Fastor/meta/meta.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/expressions/expression_traits.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"
#include "Fastor/expressions/linalg_ops/linalg_computation_types.h"

#include <algorithm>


namespace Fastor {

namespace internal {

template<typename T, size_t M, size_t N>
FASTOR_INLINE size_t count_swaps(const Tensor<T,M,N>& A) {
    size_t count = 0;
    for (size_t j = 0; j < M; ++j) {
        size_t max_index = j;
        for (size_t i = j; i < M; ++i) {
            // std::abs is necessary only for complex valued numbers
            if (std::abs(A(i, j)) > std::abs(A(max_index, j)))
                max_index = i;
        }
        if (j != max_index)
            count++;
    }
    return count;
}

}


template<PivType PType = PivType::V, typename T, size_t M, size_t N,
    enable_if_t_<PType == PivType::M, bool> = false>
FASTOR_INLINE Tensor<T,M,N> pivot(const Tensor<T,M,N>& A) {
    Tensor<size_t,M> perm;
    perm.iota();
    for (size_t j = 0; j < M; ++j) {
        size_t max_index = j;
        for (size_t i = j; i < M; ++i) {
            // std::abs is necessary only for complex valued numbers
            if (std::abs(A(i, j)) > std::abs(A(max_index, j)))
                max_index = i;
        }
        if (j != max_index)
            std::swap(perm(j), perm(max_index));
    }
    Tensor<T,M,N> P(0);
    for (size_t i = 0; i < M; ++i)
        P(i, perm(i)) = 1;
    return P;
}

template<PivType PType = PivType::V, typename T, size_t M, size_t N,
    enable_if_t_<PType == PivType::V, bool> = false>
FASTOR_INLINE Tensor<size_t,M> pivot(const Tensor<T,M,N>& A) {
    Tensor<size_t,M> perm;
    perm.iota();
    for (size_t j = 0; j < M; ++j) {
        size_t max_index = j;
        for (size_t i = j; i < M; ++i) {
            // std::abs is necessary only for complex valued numbers
            if (std::abs(A(i, j)) > std::abs(A(max_index, j)))
                max_index = i;
        }
        if (j != max_index)
            std::swap(perm(j), perm(max_index));
    }
    return perm;
}


// For generic expressions
template<PivType PType = PivType::V, typename Derived, size_t DIM,
    enable_if_t_<PType == PivType::M && requires_evaluation_v<Derived>, bool> = false>
FASTOR_INLINE
Tensor<typename Derived::scalar_type,
    get_tensor_dimension_v<0,typename Derived::result_type>,
    get_tensor_dimension_v<1,typename Derived::result_type>
>
pivot(const AbstractTensor<Derived,DIM>& src) {
    typename Derived::result_type tmp(src.self());
    return pivot<PType>(tmp);
}
template<PivType PType = PivType::M, typename Derived, size_t DIM,
    enable_if_t_<PType == PivType::M && !requires_evaluation_v<Derived>, bool> = false>
FASTOR_INLINE
Tensor<typename Derived::scalar_type,
    get_tensor_dimension_v<0,typename Derived::result_type>,
    get_tensor_dimension_v<1,typename Derived::result_type>
>
pivot(const AbstractTensor<Derived,DIM>& src) {

    static_assert(DIM==2, "TENSOR MUST BE SQUARE FOR PIVOT COMPUTATION");

    using T = typename Derived::scalar_type;
    using result_type = typename Derived::result_type;
    constexpr FASTOR_INDEX M = get_tensor_dimension_v<0,result_type>;
    constexpr FASTOR_INDEX N = get_tensor_dimension_v<1,result_type>;

    const Derived &A = src.self();

    Tensor<size_t,M> perm;
    perm.iota();
    for (size_t j = 0; j < M; ++j) {
        size_t max_index = j;
        for (size_t i = j; i < M; ++i) {
            // std::abs is necessary only for complex valued numbers
            if (std::abs(A.template eval_s<T>(i, j)) > std::abs(A.template eval_s<T>(max_index, j)))
                max_index = i;
        }
        if (j != max_index)
            std::swap(perm(j), perm(max_index));
    }
    Tensor<T,M,N> P(0);
    for (size_t i = 0; i < M; ++i)
        P(i, perm(i)) = 1;
    return P;
}

template<PivType PType = PivType::V, typename Derived, size_t DIM,
    enable_if_t_<PType == PivType::V && requires_evaluation_v<Derived>, bool> = false>
FASTOR_INLINE
Tensor<size_t,
    get_tensor_dimension_v<0,typename Derived::result_type>
>
pivot(const AbstractTensor<Derived,DIM>& src) {
    typename Derived::result_type tmp(src.self());
    return pivot<PType>(tmp);
}
template<PivType PType = PivType::M, typename Derived, size_t DIM,
    enable_if_t_<PType == PivType::V && !requires_evaluation_v<Derived>, bool> = false>
FASTOR_INLINE
Tensor<size_t,
    get_tensor_dimension_v<0,typename Derived::result_type>
>
pivot(const AbstractTensor<Derived,DIM>& src) {

    static_assert(DIM==2, "TENSOR MUST BE SQUARE FOR PIVOT COMPUTATION");

    using T = typename Derived::scalar_type;
    using result_type = typename Derived::result_type;
    constexpr FASTOR_INDEX M = get_tensor_dimension_v<0,result_type>;
    constexpr FASTOR_INDEX N = get_tensor_dimension_v<1,result_type>;

    const Derived &A = src.self();

    Tensor<size_t,M> perm;
    perm.iota();
    for (size_t j = 0; j < M; ++j) {
        size_t max_index = j;
        for (size_t i = j; i < M; ++i) {
            // std::abs is necessary only for complex valued numbers
            if (std::abs(A.template eval_s<T>(i, j)) > std::abs(A.template eval_s<T>(max_index, j)))
                max_index = i;
        }
        if (j != max_index)
            std::swap(perm(j), perm(max_index));
    }
    return perm;
}

/* In place versions - Given an evaluated pivot tensor populates it
*/
template<typename T, size_t M, size_t N>
FASTOR_INLINE void pivot_inplace(const Tensor<T,M,N>& A, Tensor<size_t,M>& perm) {
    perm.iota();
    for (size_t j = 0; j < M; ++j) {
        size_t max_index = j;
        for (size_t i = j; i < M; ++i) {
            // std::abs is necessary only for complex valued numbers
            if (std::abs(A(i, j)) > std::abs(A(max_index, j)))
                max_index = i;
        }
        if (j != max_index)
            std::swap(perm(j), perm(max_index));
    }
}

template<typename T, size_t M, size_t N>
FASTOR_INLINE void pivot_inplace(const Tensor<T,M,N>& A, Tensor<T,M,N> &P) {
    Tensor<size_t,M> perm;
    perm.iota();
    for (size_t j = 0; j < M; ++j) {
        size_t max_index = j;
        for (size_t i = j; i < M; ++i) {
            // std::abs is necessary only for complex valued numbers
            if (std::abs(A(i, j)) > std::abs(A(max_index, j)))
                max_index = i;
        }
        if (j != max_index)
            std::swap(perm(j), perm(max_index));
    }
    P.fill(0);
    for (size_t i = 0; i < M; ++i)
        P(i, perm(i)) = 1;
}

template<typename Derived, size_t DIM, size_t M,
    enable_if_t_<requires_evaluation_v<Derived>, bool> = false>
FASTOR_INLINE
void
pivot_inplace(const AbstractTensor<Derived,DIM>& src, Tensor<size_t,M> &perm) {
    typename Derived::result_type tmp(src.self());
    pivot_inplace(tmp,perm);
}
template<typename Derived, size_t DIM, size_t M,
    enable_if_t_<!requires_evaluation_v<Derived>, bool> = false>
FASTOR_INLINE
void
pivot_inplace(const AbstractTensor<Derived,DIM>& src, Tensor<size_t,M> &perm) {

    static_assert(DIM==2, "TENSOR MUST BE 2D FOR PIVOT COMPUTATION");

    using T = typename Derived::scalar_type;

    const Derived &A = src.self();

    perm.iota();
    for (size_t j = 0; j < M; ++j) {
        size_t max_index = j;
        for (size_t i = j; i < M; ++i) {
            // std::abs is necessary only for complex valued numbers
            if (std::abs(A.template eval_s<T>(i, j)) > std::abs(A.template eval_s<T>(max_index, j)))
                max_index = i;
        }
        if (j != max_index)
            std::swap(perm(j), perm(max_index));
    }
}

template<typename Derived, size_t DIM, typename T, size_t M, size_t N,
    enable_if_t_<requires_evaluation_v<Derived>, bool> = false>
FASTOR_INLINE
void
pivot_inplace(const AbstractTensor<Derived,DIM>& src, Tensor<T,M,N> &P) {
    typename Derived::result_type tmp(src.self());
    pivot_inplace(tmp,P);
}
template<typename Derived, size_t DIM, typename T, size_t M, size_t N,
    enable_if_t_<!requires_evaluation_v<Derived>, bool> = false>
FASTOR_INLINE
void
pivot_inplace(const AbstractTensor<Derived,DIM>& src, Tensor<T,M,N> &P) {

    static_assert(DIM==2, "TENSOR MUST BE 2D FOR PIVOT COMPUTATION");

    const Derived &A = src.self();

    Tensor<size_t,M> perm;
    perm.iota();
    for (size_t j = 0; j < M; ++j) {
        size_t max_index = j;
        for (size_t i = j; i < M; ++i) {
            // std::abs is necessary only for complex valued numbers
            if (std::abs(A.template eval_s<T>(i, j)) > std::abs(A.template eval_s<T>(max_index, j)))
                max_index = i;
        }
        if (j != max_index)
            std::swap(perm(j), perm(max_index));
    }
    for (size_t i = 0; i < M; ++i)
        P(i, perm(i)) = 1;
}




/* Apply a pivot on a tensor/matrix
    Applying pivot can only work on evaluated tensors and not expression
    as non-evaluated expression do not have storage/data
*/
template <typename T, size_t M, size_t N>
FASTOR_INLINE Tensor<T,M,N> apply_pivot(const Tensor<T,M,N>& A, const Tensor<size_t,M>& P) {
    Tensor<T,M,N> copyA(A);
    for (size_t i=0; i< M; ++i) {
        if (P(i) != i) {
            std::copy_n(&A.data()[P(i)*N],N,&copyA.data()[i*N]);
        }
    }
    return copyA;
}
template <typename T, size_t M, size_t N>
FASTOR_INLINE Tensor<T,M,N> apply_pivot(const Tensor<T,M,N>& A, const Tensor<T,M,N>& P) {
    // The output tensor is just matmul(P,A), but we are going to avoid
    // the call to matmul
    Tensor<T,M,N> copyA(A);
    for (size_t i=0; i< M; ++i) {
        auto it = std::find(&P.data()[i*N],&P.data()[i*N+N],T(1));
        size_t p = std::distance(&P.data()[i*N],it);
        // Swap row p and i
        if (p != i) {
            std::copy_n(&A.data()[p*N],N,&copyA.data()[i*N]);
        }
    }
    return copyA;
}

template <typename T, size_t M, size_t N>
FASTOR_INLINE void apply_pivot_inplace(Tensor<T,M,N>& A, const Tensor<size_t,M>& P) {
    Tensor<T,M,N> copyA(A);
    for (size_t i=0; i< M; ++i) {
        if (P(i) != i) {
            std::copy_n(&copyA.data()[P(i)*N],N,&A.data()[i*N]);
        }
    }
}
template <typename T, size_t M, size_t N>
FASTOR_INLINE void apply_pivot_inplace(Tensor<T,M,N>& A, const Tensor<T,M,N>& P) {
    // The output tensor is just matmul(P,A), but we are going to avoid
    // the call to matmul
    Tensor<T,M,N> copyA(A);
    for (size_t i=0; i< M; ++i) {
        auto it = std::find(&P.data()[i*N],&P.data()[i*N+N],T(1));
        size_t p = std::distance(&P.data()[i*N],it);
        // Swap row p and i
        if (p != i) {
            std::copy_n(&copyA.data()[p*N],N,&A.data()[i*N]);
        }
    }
}




/* Reconstructing the matrix back from a pivoted factorisation
    Reconstructs from LU/QR/QL etc
*/
template <typename T, size_t M, size_t N>
FASTOR_INLINE Tensor<T,M,N> reconstruct(const Tensor<T,M,N>& L, Tensor<T,M,N>& U) {
    return matmul(L,U);
}

/* Reconstructing the matrix back from a pivoted factorisation
    Reconstructs from an PLU decomposition given P, L, U where P is
    integral permutation vector
    A = {P}^(-1)*L*U
*/
template <typename T, size_t M, size_t N>
FASTOR_INLINE Tensor<T,M,N> reconstruct(const Tensor<T,M,N>& L, Tensor<T,M,N>& U, const Tensor<size_t,M>& P) {
    Tensor<T,M,N> A = matmul(L,U);
    Tensor<T,M,N> copyA(A);
    for (size_t i=0; i< M; ++i) {
        if (P(i) != i) {
            std::copy_n(&copyA.data()[i*N],N,&A.data()[P(i)*N]);
        }
    }
    return A;
}

/* Reconstructing the matrix back from a pivoted factorisation
    Reconstructs from an PLU decomposition given P, L, U
    A = {P}^(-1)*L*U
*/
template <typename T, size_t M, size_t N>
FASTOR_INLINE Tensor<T,M,N> reconstruct(const Tensor<T,M,N>& L, Tensor<T,M,N>& U, const Tensor<T,M,N>& P) {
    // To avoid computing the inverse of P in {P}^(-1)*L*U
    Tensor<T,M,N> A = matmul(L,U);
    Tensor<T,M,N> copyA(A);
    for (size_t i=0; i< M; ++i) {
        auto it = std::find(&P.data()[i*N],&P.data()[i*N+N],T(1));
        size_t p = std::distance(&P.data()[i*N],it);
        if (p != i) {
            std::copy_n(&copyA.data()[i*N],N,&A.data()[p*N]);
        }
    }
    return A;
}


/* Reconstructing the matrix back from a pivot only
*/
template <typename T, size_t M, size_t N>
FASTOR_INLINE Tensor<T,M,N> reconstruct(const Tensor<T,M,N>& A, const Tensor<size_t,M>& P) {
    Tensor<T,M,N> copyA(A);
    for (size_t i=0; i< M; ++i) {
        if (P(i) != i) {
            std::copy_n(&A.data()[i*N],N,&copyA.data()[P(i)*N]);
        }
    }
    return copyA;
}

/* Reconstructing the matrix back from a pivot only - column-wise or rather post multiplication
    used for computing inverse
*/
template <typename T, size_t M, size_t N>
FASTOR_INLINE Tensor<T,M,N> reconstruct_colwise(const Tensor<T,M,N>& A, const Tensor<size_t,M>& P) {
    Tensor<T,M,N> copyA(A);
    for (size_t i=0; i< M; ++i) {
        if (P(i) != i) {
            copyA(all,P(i)) = A(all,i);
        }
    }
    return copyA;
}



} // end of namespace Fastor


#endif // UNARY_PIV_OP_H
