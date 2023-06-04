#ifndef UNARY_SVD_OP_H
#define UNARY_SVD_OP_H

#include "Fastor/meta/meta.h"
#include "Fastor/backend/inner.h"
#include "Fastor/backend/lufact.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/Aliasing.h"
#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/expressions/expression_traits.h"
#include "Fastor/expressions/linalg_ops/linalg_computation_types.h"
#include "Fastor/expressions/linalg_ops/unary_det_op.h"
#include "Fastor/expressions/linalg_ops/unary_trans_op.h"


namespace Fastor {

// SVD
template<typename T, size_t M, enable_if_t_<M==2, bool> = false >
FASTOR_INLINE void svd(const Tensor<T,M,M> &A, Tensor<T,M,M> &U, Tensor<T,M,M> &S, Tensor<T,M,M> &V) {

    constexpr T Epsilon_v = std::numeric_limits<T>::epsilon();

    const T f00 = A(0, 0);
    const T f01 = A(0, 1);
    const T f10 = A(1, 0);
    const T f11 = A(1, 1);

    // If matrix is diagonal, SVD is trivial
    if (std::abs(f01 - f10) < Epsilon_v && std::abs(f01) < Epsilon_v)
    {
        // Compute U
        U(0,0) = f00 < 0 ? -1. : 1.;
        U(0,1) = 0.;
        U(1,0) = 0.;
        U(1,1) = f11 < 0. ? -1. : 1.;

        // Compute S
        S(0,0) = std::abs(f00);
        S(0,1) = 0;
        S(1,0) = 0;
        S(1,1) = std::abs(f11);

        // Compute V
        V.eye2();
    }
    // Otherwise, we need to compute A^T*A
    else
    {
        T j    = f00 * f00 + f01 * f01;
        T k    = f10 * f10 + f11 * f11;
        T v_c  = f00 * f10 + f01 * f11;
        // Check to see if A^T*A is diagonal
        if (std::abs(v_c) < Epsilon_v)
        {
            // Compute S
            T s1 = std::sqrt(j);
            T s2 = std::abs(j - k) < Epsilon_v ? s1 : std::sqrt(k);
            S(0,0) = s1;
            S(0,1) = 0;
            S(1,0) = 0;
            S(1,1) = s2;

            // Compute U
            U.eye2();

            // Compute V
            V(0,0) = f00 / s1;
            V(0,1) = f10 / s2;
            V(1,0) = f01 / s1;
            V(1,1) = f11 / s2;
        }
        // Otherwise, solve quadratic equation for eigenvalues
        else
        {
            T jmk    = j - k;
            T jpk    = j + k;
            T root   = std::sqrt(jmk * jmk + 4. * v_c * v_c);
            T eig1   = (jpk + root) * 0.5;
            T eig2   = (jpk - root) * 0.5;

            // Compute S
            T s1     = std::sqrt(eig1);
            T s2     = std::abs(root) < Epsilon_v ? s1 : ( eig2 > 0 ? std::sqrt(eig2) : Epsilon_v);
            S(0,0) = s1;
            S(0,1) = 0;
            S(1,0) = 0;
            S(1,1) = s2;

            // Compute U - use eigenvectors of A^T*A as U
            T v_s = eig1 - j;
            T len = std::max(std::sqrt(v_s * v_s + v_c * v_c), Epsilon_v);
            v_c /= len;
            v_s /= len;

            U(0,0) =  v_c;
            U(0,1) = -v_s;
            U(1,0) =  v_s;
            U(1,1) =  v_c;

            // Compute V - as A * U / s
            const T cc = (f00 * v_c + f10 * v_s) / s1;
            const T cs = (f01 * v_c + f11 * v_s) / s1;
            if (std::abs(s2) > Epsilon_v)
            {
                V(0,0) =  cc;
                V(0,1) =  (f10* v_c - f00 * v_s) / s2;
                V(1,0) =  cs;
                V(1,1) =  (f11 * v_c - f01 * v_s) / s2;
            }
            else
            {
                V(0,0) =  cc;
                V(0,1) =  cs;
                V(1,0) =  cs;
                V(1,1) = -cc;
            }
        }
    }
}



// Signed SVD
template<typename T, size_t M>
FASTOR_INLINE void ssvd(const Tensor<T,M,M> &A, Tensor<T,M,M> &U, Tensor<T,M,M> &S, Tensor<T,M,M> &V) {

    // Same as above but avoiding the L matrix
    svd(A, U, S, V);

    // See where to pull the reflection out of
    const T detU = determinant(U);
    const T detV = determinant(V);

    if (detU >= 0 && detV >= 0)
    {
        // No reflection svd == svd_rv, return
        return;
    }

    Tensor<T, M, M> L = matmul(U, transpose(V));
    const T lastColumn = determinant(L);

    if (detU < 0 && detV > 0)
    {
        U(all, M - 1) *= lastColumn;
    }
    else if (detU > 0 && detV < 0)
    {
        V(all, M - 1) *= lastColumn;
    }

    // Push the reflection to the diagonal
    S(M - 1, M - 1) *= lastColumn;
}
//-----------------------------------------------------------------------------------------------------------//

} // end of namespace Fastor


#endif // UNARY_SVD_OP_H
