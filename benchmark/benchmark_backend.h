#ifndef BENCHMARK_BACKEND_H
#define BENCHMARK_BACKEND_H

#define HAS_SSE
#define HAS_AVX

#include <commons/utils.h>
#include <simd_vector/SIMDVector.h>
#include <tensor/Tensor.h>
#include <tensor/tensor_print.h>
#include <tensor/tensor_funcs.h>
#include <tensor_algebra/einsum.h>
#include "expressions/expressions.h"
#include <backend/voigt.h>

enum VERSIONS {
    SCALAR,
    SSE,
    AVX,
    SIMD,
    FORLOOPS
};

using namespace Fastor;

using std::size_t;
//#define SIZE 3
using real = float;

enum {
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P
};


//static const double levi_civita[27] = {0,0,0,0,0,-1,0,1,0,0,0,1,0,0,0,-1,0,0,0,-1,0,1,0,0,0,0,0};
static const real levi_civita[27] = { 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,
                                        0., -1.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.};

//__attribute__((optimize("no-tree-vectorize")))
template<size_t N>
void tensor_cross_scalar(const real *__restrict__ a, const real *__restrict__ b, real *__restrict__ c) {
    constexpr size_t size = N;

//#define CACHE_FRIENDLY
#ifndef CACHE_FRIENDLY
//    for (size_t i=0; i<size; ++i)
//        for (size_t j=0; j<size; ++j)
//            for (size_t k=0; k<size; ++k)
//                for (size_t I=0; I<size; ++I)
//                    for (size_t J=0; J<size; ++J)
//                        for (size_t K=0; K<size; ++K)
//                            c[i*size+I] += levi_civita[i*size*size+j*size+k]*levi_civita[I*size*size+J*size+K]*b[j*size+J]*a[k*size+K];

    for (size_t i=0; i<size; ++i)
        for (size_t j=0; j<size; ++j)
            for (size_t k=0; k<size; ++k)
                for (size_t I=0; I<size; ++I)
                    for (size_t J=0; J<size; ++J)
                        for (size_t K=0; K<size; ++K)
//                            if (std::fabs(levi_civita[i*size*size+j*size+k])>PRECI_TOL && std::fabs(levi_civita[I*size*size+J*size+K])>PRECI_TOL)
//                            if (levi_civita[i*size*size+j*size+k]!=PRECI_TOL && levi_civita[I*size*size+J*size+K]!=0)
                                c[i*size+I] += levi_civita[i*size*size+j*size+k]*levi_civita[I*size*size+J*size+K]*b[j*size+J]*a[k*size+K];

#else
//    for (size_t K=0; K<size; ++K)
//        for (size_t J=0; J<size; ++J)
//            for (size_t I=0; I<size; ++I)
//                for (size_t k=0; k<size; ++k)
//                    for (size_t j=0; j<size; ++j)
//                        for (size_t i=0; i<size; ++i)
//                            c[i*size+I] += levi_civita[i*size*size+j*size+k]*levi_civita[I*size*size+J*size+K]*b[j*size+J]*a[k*size+K];

    // fast for clang // slow for gcc
    for (size_t K=0; K<size; ++K) {
        for (size_t J=0; J<size; ++J) {
            for (size_t k=0; k<size; ++k) {
                for (size_t j=0; j<size; ++j) {
                    for (size_t I=0; I<size; ++I) {
                        for (size_t i=0; i<size; ++i) {
                            c[i*size+I]  += levi_civita[i*size*size+j*size+k]*levi_civita[I*size*size+J*size+K]*a[j*size+J]*b[k*size+K];
                        }
                    }
                }
            }
        }
    }
#endif
}

template<size_t N>
void AilBjk(const real *__restrict__ a, const real *__restrict__ b, real *__restrict__ c) {
    constexpr size_t size = N;
    for (size_t i=0; i<size; ++i)
        for (size_t j=0; j<size; ++j)
            for (size_t k=0; k<size; ++k)
                c[i*size+j] += b[i*size+j]*a[j*size+k];
}

template<size_t N>
void AijBkl(const real *__restrict__ a, const real *__restrict__ b, real *__restrict__ c) {
    constexpr size_t size = N;
    for (size_t i=0; i<size; ++i) {
        for (size_t j=0; j<size; ++j) {
            for (size_t k=0; k<size; ++k) {
                for (size_t l=0; l<size; ++l) {
                    c[i*size*size*size+j*size*size+k*size+l] += a[i*size+j]*b[k*size+l];
                }
            }
        }
    }
}

template<size_t N>
void AikBjl(const real *__restrict__ a, const real *__restrict__ b, real *__restrict__ c) {
    constexpr size_t size = N;
    for (size_t i=0; i<size; ++i) {
        for (size_t j=0; j<size; ++j) {
            for (size_t k=0; k<size; ++k) {
                for (size_t l=0; l<size; ++l) {
                    c[i*size*size*size+j*size*size+k*size+l] += a[i*size+k]*b[j*size+l];
                }
            }
        }
    }
}

template<size_t N>
void matmul_scalar(const real *__restrict__ a, const real *__restrict__ b, real *__restrict__ c) {
    constexpr size_t size = N;
    for (size_t i=0; i<size; ++i) {
        for (size_t j=0; j<size; ++j) {
            for (size_t k=0; k<size; ++k) {
                c[i*size*size+j*size+k] += a[i*size+j]*b[j*size+k];
            }
        }
    }
}



template<size_t N>
void outer_4_and_4(const real *__restrict__ a, const real *__restrict__ b, real *__restrict__ c) {
    constexpr size_t size = N;
    for (size_t i=0; i<size; ++i) {
        for (size_t j=0; j<size; ++j) {
            for (size_t k=0; k<size; ++k) {
                for (size_t l=0; l<size; ++l) {
                    for (size_t I=0; I<size; ++I) {
                        for (size_t J=0; J<size; ++J) {
                            for (size_t K=0; K<size; ++K) {
                                for (size_t L=0; L<size; ++L) {
                                    c[i*size*size*size*size*size*size*size+
                                            j*size*size*size*size*size*size+
                                            k*size*size*size*size*size+
                                            l*size*size*size*size+
                                            I*size*size*size+
                                            J*size*size+
                                            K*size
                                            +L] += a[i*size*size*size+
                                                    j*size+size+
                                                    k*size+
                                                    l]
                                                    *
                                                    b[I*size*size*size+
                                                    J*size+size+
                                                    K*size+
                                                    L];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


void _voigt(const real * __restrict__ a_data, real * __restrict__ VoigtA) {
    VoigtA[0] = a_data[0];
    VoigtA[1] = a_data[4];
    VoigtA[2] = a_data[8];
    VoigtA[3] = 0.5*(a_data[1]+a_data[3]);
    VoigtA[4] = 0.5*(a_data[2]+a_data[6]);
    VoigtA[5] = 0.5*(a_data[5]+a_data[7]);
    VoigtA[6] = VoigtA[1];
    VoigtA[7] = a_data[40];
    VoigtA[8] = a_data[44];
    VoigtA[9] = 0.5*(a_data[37]+a_data[39]);
    VoigtA[10] = 0.5*(a_data[38]+a_data[42]);
    VoigtA[11] = 0.5*(a_data[41]+a_data[43]);
    VoigtA[12] = VoigtA[2];
    VoigtA[13] = VoigtA[8];
    VoigtA[14] = a_data[80];
    VoigtA[15] = 0.5*(a_data[73]+a_data[75]);
    VoigtA[16] = 0.5*(a_data[74]+a_data[78]);
    VoigtA[17] = 0.5*(a_data[77]+a_data[79]);
    VoigtA[18] = VoigtA[3];
    VoigtA[19] = VoigtA[9];
    VoigtA[20] = VoigtA[15];
    VoigtA[21] = 0.5*(a_data[10]+a_data[12]);
    VoigtA[22] = 0.5*(a_data[11]+a_data[15]);
    VoigtA[23] = 0.5*(a_data[14]+a_data[16]);
    VoigtA[24] = VoigtA[4];
    VoigtA[25] = VoigtA[10];
    VoigtA[26] = VoigtA[16];
    VoigtA[27] = VoigtA[22];
    VoigtA[28] = 0.5*(a_data[20]+a_data[24]);
    VoigtA[29] = 0.5*(a_data[23]+a_data[25]);
    VoigtA[30] = VoigtA[5];
    VoigtA[31] = VoigtA[11];
    VoigtA[32] = VoigtA[17];
    VoigtA[33] = VoigtA[23];
    VoigtA[34] = VoigtA[29];
    VoigtA[35] = 0.5*(a_data[50]+a_data[52]);
}



// ----------------------------------------RUN-----------------------------------------------------------
//static const size_t NITER = 1000000LL;
static const size_t NITER = 1000LL;
// clobber
//template <typename T> void unused(T &&x) { asm("" ::"m"(x)); }

// iterate the same benchmark one million times
template<size_t N>
void iterate_over(const real* a, const real* b, real* out) {
    Tensor<real,N,N,N,N> x; x.random();
    Tensor<real,N,N,N,N> y; y.random();
    unused(b);    unused(a);     unused(out);

//    auto z = einsum<Index<I,J,K,L>,Index<L,M,O,P>>(x,y);
//    print(type_name<decltype(z)>());
    for (volatile size_t i=0; i<NITER; i++){
//        auto z = einsum<Index<I,J,K,L>,Index<L,M,O,P>>(x,y);
//        cyclic_0<real,N,N,N,N>(a,b,out);
        _outer<real,N,N,N,N>(a,b,out);
//        Tensor<double,200,200> x;
//        x.iota(0.);
//        _crossproduct<real,N,N>(a,b,out);
//        _matmul<real,N,N,N>(a,b,out);
        unused(out);
//        auto z = outer(x,y);
//        unused(z);
    }
}

template<size_t N>
void iterate_over_scalar(const real* a, const real* b, real* out) {    
    real a_data[81];
    for (volatile size_t i=0; i<NITER; i++){
//        tensor_cross_scalar<N>(a,b,out);
        matmul_scalar<N>(a,b,out);
//        outer_4_and_4<N>(a,b,out);
        AijBkl<N>(a,b,a_data);   _voigt(a_data,out);
//        AikBjl<N>(a,b,a_data);   _voigt(a_data,out);
        unused(out);
    }
}





// iterate the same benchmark one million times
template<size_t N>
void iterate_over() {
//    Tensor<real,N,N,N,N> x; x.random();
//    Tensor<real,N,N,N,N> y; y.random();
    Tensor<real,N,N,N,8> x; x.random();
    Tensor<real,N,N,N,8> y; y.random();

    for (volatile size_t i=0; i<NITER; i++){
        auto z = outer(x,y);
        x = x+z(1,0,0,0,0,0,0,1);
        unused(z);
    }
}

template<size_t N>
void iterate_over_scalar() {
    Tensor<real,N,N,N,N> x; x.random();
    Tensor<real,N,N,N,N> y; y.random();

    real a_data[N*N*N*N*N*N*N*N];
    for (volatile size_t i=0; i<NITER; i++){
        outer_4_and_4<N>(x.data(),y.data(),a_data);
        x = x+y; a_data[1] = a_data[2];
        unused(a_data);
    }
}



// iterate the same benchmark one million times
template<size_t M,size_t N,size_t P,size_t Q>
void iterate_over(const Tensor<real,M,N,P,Q>& x, const Tensor<real,M,N,P,Q>& y) {
    for (volatile size_t i=0; i<NITER; i++){
        auto z = outer(x,y);
//        z = z+x(1,0,0,1);
        unused(z);
    }
}



// ---------------------------------------- END RUN-----------------------------------------------------------


#endif // BENCHMARK_BACKEND_H

