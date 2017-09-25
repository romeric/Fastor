#ifndef _HELPER__H
#define _HELPER__H

#include <Fastor.h>
#include <tuple>
using namespace Fastor;

typedef double Real;

template<typename T, size_t ndim>
static FASTOR_INLINE
Tensor<T,ndim,ndim> EYE2() {
    Tensor<T,ndim,ndim> I; I.zeros();
    for (size_t i=0; i<ndim; ++i)
        I(i,i) = T(1);

    return I;
}

enum
{
    i,j,k,l,m,n,o,p,q,r
};

template<typename T, size_t ... Dims>
FASTOR_INLINE
void copy_numpy(Tensor<T,Dims...> &A, const T* A_np, size_t offset=0) {
    std::copy(A_np,A_np+A.size(),A.data());
}


template<typename T, size_t ... Dims>
FASTOR_INLINE
void copy_fastor(T* A_np, const Tensor<T,Dims...> &A, size_t offset=0) {
    std::copy(A.data(),A.data()+A.size(),A_np+offset);
}


// For electro-elasticity
template<typename T, size_t N>
struct ElectroMechanicsHessianType;
template<typename T>
struct ElectroMechanicsHessianType<T,2> {
    using return_type = Tensor<T,5,5>;
};
template<typename T>
struct ElectroMechanicsHessianType<T,3> {
    using return_type = Tensor<T,9,9>;
};

// For mechanics
template<typename T, size_t N>
struct MechanicsHessianType;
template<typename T>
struct MechanicsHessianType<T,2> {
    using return_type = Tensor<T,3,3>;
};
template<typename T>
struct MechanicsHessianType<T,3> {
    using return_type = Tensor<T,6,6>;
};

// For possion
template<typename T, size_t N>
struct PoissonHessianType;
template<typename T>
struct PoissonHessianType<T,2> {
    using return_type = Tensor<T,2,2>;
};
template<typename T>
struct PoissonHessianType<T,3> {
    using return_type = Tensor<T,3,3>;
};




template<typename T, size_t M, size_t N>
FASTOR_INLINE
typename ElectroMechanicsHessianType<T,N>::return_type
make_electromechanical_hessian(Tensor<T,M,M> elasticity, Tensor<T,M,N> coupling, Tensor<T,N,N> dielectric) {

    using ret_type = typename ElectroMechanicsHessianType<T,N>::return_type;
    ret_type hessian;

    for (size_t i=0; i<M; ++i) {
        for (size_t j=0; j<M; ++j) {
            hessian(i,j) = elasticity(i,j);
        }
    }

    for (size_t i=0; i<M; ++i) {
        for (size_t j=0; j<N; ++j) {
            hessian(i,j+M) = -coupling(i,j);
        }
    }

    for (size_t i=0; i<N; ++i) {
        for (size_t j=0; j<M; ++j) {
            hessian(i+M,j) = -coupling(j,i);
        }
    }

    for (size_t i=0; i<N; ++i) {
        for (size_t j=0; j<N; ++j) {
            hessian(i+M,j+M) = dielectric(i,j);
        }
    }

    /*
    // View based NumPy style vectorised version
    // The above for loop style code is more efficient as
    // the compiler completely optimises that away, whereas
    // in the following transpose makes a copy
    hessian(fseq<0,M>(),fseq<0,M>()) = elasticity;
    hessian(fseq<0,M>(),fseq<M+1,N>()) = -coupling;
    hessian(fseq<M+1,N>(),fseq<0,M>()) = -transpose(coupling);
    hessian(fseq<M+1,N>(),fseq<M+1,N>()) = dielectric;
    */

    return hessian;
}


#endif