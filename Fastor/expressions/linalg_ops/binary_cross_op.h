#ifndef BINARY_CROSS_OP_H
#define BINARY_CROSS_OP_H

#include "Fastor/meta/meta.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/backend/tensor_cross.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/tensor/Aliasing.h"
#include "Fastor/expressions/expression_traits.h"

namespace Fastor {

// Tensor cross product of two 2nd order tensors
template<typename T, size_t I, size_t J, typename std::enable_if<I==3 && J==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J> cross(const Tensor<T,I,J> &b, const Tensor<T,I,J> &a) {
    Tensor<T,I,J> out;
    _crossproduct<T,I,I,J>(a.data(),b.data(),out.data());
    return out;
}

template<typename T, size_t I, size_t J, typename std::enable_if<I==2 && J==2,bool>::type=0>
FASTOR_INLINE Tensor<T,I+1,J+1> cross(const Tensor<T,I,J> &b, const Tensor<T,I,J> &a) {
    Tensor<T,I+1,J+1> out;
    _crossproduct<T,I,I,J>(a.data(),b.data(),out.data());
    return out;
}

template<int Plane, typename T, size_t I, size_t J, typename std::enable_if<I==2 && J==2 && Plane==FASTOR_PlaneStrain,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J> cross(const Tensor<T,I,J> &b, const Tensor<T,I,J> &a) {
    // Plane strain case
    Tensor<T,I,J> out;
    Tensor<T,I+1,J+1> a3, b3;
    a3(0,0) = a(0,0);
    a3(0,1) = a(0,1);
    a3(1,0) = a(1,0);
    a3(1,1) = a(1,1);
    a3(2,2) = 1;
    b3(0,0) = b(0,0);
    b3(0,1) = b(0,1);
    b3(1,0) = b(1,0);
    b3(1,1) = b(1,1);
    b3(2,2) = 1;
    _crossproduct<T,FASTOR_PlaneStrain>(a.data(),b.data(),out.data());
    return out;
}

// Tensor cross product of a vector with 2nd order tensor
template<typename T, size_t I, size_t J, typename std::enable_if<I==3 && J==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J> cross(const Tensor<T,I> &b, const Tensor<T,I,J> &a) {
    Tensor<T,I,J> out;
    _crossproduct<T,I,1,J>(a.data(),b.data(),out.data());
    return out;
}

template<typename T, size_t I, size_t J, typename std::enable_if<I==2 && J==2,bool>::type=0>
FASTOR_INLINE Tensor<T,I+1,J+1> cross(const Tensor<T,I> &b, const Tensor<T,I,J> &a) {
    Tensor<T,I+1,J+1> out;
    _crossproduct<T,I,1,J>(a.data(),b.data(),out.data());
    return out;
}

// Tensor cross product of a 2nd order tensor with a vector
template<typename T, size_t I, size_t J, typename std::enable_if<I==3 && J==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J> cross(const Tensor<T,I,J> &b, const Tensor<T,J> &a) {
    Tensor<T,I,J> out;
    _crossproduct<T,I,J,1>(a.data(),b.data(),out.data());
    return out;
}

template<typename T, size_t I, size_t J, typename std::enable_if<I==2 && J==2,bool>::type=0>
FASTOR_INLINE Tensor<T,I+1,J+1> cross(const Tensor<T,I,J> &b, const Tensor<T,J> &a) {
    Tensor<T,I+1,J+1> out;
    _crossproduct<T,I,J,1>(a.data(),b.data(),out.data());
    return out;
}


// Tensor cross product of a 3rd order tensors
template<typename T, size_t I, size_t J, size_t K, size_t L, size_t M, size_t N,
         typename std::enable_if<I==3 && J==3 && K==3 && L==3 && M==3 && N==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,3,3,N> cross(const Tensor<T,I,J,K> &A, const Tensor<T,L,M,N> &B) {
    Tensor<T,I,3,3,N> C;
    _crossproduct<T,I,J,K,L,M,N>(A.data(),B.data(),C.data());
    return C;
}


// Tensor cross product of a 4th order tensors
template<typename T, size_t I, size_t J, size_t K, size_t L, size_t M, size_t N, size_t O, size_t P,
         typename std::enable_if<I==3 && J==3 && K==3 && L==3 && M==3 && N==3 && O==3 && P==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J,3,3,O,P> cross(const Tensor<T,I,J,K,L> &A, const Tensor<T,M,N,O,P> &B) {
    Tensor<T,I,J,3,3,O,P> C;
    _crossproduct<T,I,J,K,L,M,N,O,P>(A.data(),B.data(),C.data());
    return C;
}


// Tensor cross product of a 4th order tensor with 2nd order tensor
template<typename T, size_t I, size_t J, size_t K, size_t L, size_t M, size_t N,
         typename std::enable_if<I==3 && J==3 && K==3 && L==3 && M==3 && N==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J,K,L> cross(const Tensor<T,I,J,K,L> &A, const Tensor<T,M,N> &B) {
    Tensor<T,I,J,K,L> C;
    _crossproduct42<T,I,J,K,L,M,N>(A.data(),B.data(),C.data());
    return C;
}


// Tensor cross product of a 2nd order tensor with 4th order tensor
template<typename T, size_t I, size_t J, size_t K, size_t L, size_t M, size_t N,
         typename std::enable_if<I==3 && J==3 && K==3 && L==3 && M==3 && N==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J,K,L> cross(const Tensor<T,M,N> &A, const Tensor<T,I,J,K,L> &B) {
    Tensor<T,I,J,K,L> C;
    _crossproduct24<T,I,J,K,L,M,N>(B.data(),A.data(),C.data());
    return C;
}


// Tensor cross product of a 3rd order tensor with 2nd order tensor
template<typename T, size_t I, size_t J, size_t K, size_t L, size_t M,
         typename std::enable_if<I==3 && J==3 && K==3 && L==3 && M==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J,K> cross(const Tensor<T,I,J,K> &A, const Tensor<T,L,M> &B) {
    Tensor<T,I,J,K> C;
    _crossproduct32<T,I,J,K,L,M>(A.data(),B.data(),C.data());
    return C;
}


// Tensor cross product of a 2nd order tensor with 3rd order tensor
template<typename T, size_t I, size_t J, size_t K, size_t L, size_t M,
         typename std::enable_if<I==3 && J==3 && K==3 && L==3 && M==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J,K> cross(const Tensor<T,L,M> &A, const Tensor<T,I,J,K> &B) {
    Tensor<T,I,J,K> C;
    _crossproduct23<T,I,J,K,L,M>(B.data(),A.data(),C.data());
    return C;
}

} // end of namespace Fastor


#endif // BINARY_CROSS_OP_H