#ifndef LIBXSMM_BACKEND_H
#define LIBXSMM_BACKEND_H

#include <Fastor/tensor/Tensor.h>

#ifdef FASTOR_USE_LIBXSMM

#include <libxsmm.h>

namespace Fastor {
namespace blas {

// single
template<size_t M, size_t K, size_t N>
FASTOR_INLINE
void matmulNN_libxsmm(
    const float * FASTOR_RESTRICT a_data,
    const float * FASTOR_RESTRICT b_data,
    float * FASTOR_RESTRICT out_data) {

    constexpr int MM= M;
    constexpr int KK= K;
    constexpr int NN= N;
    constexpr float alpha = 1.0;
    constexpr float beta = 0.0;

    constexpr char transa = 'N';
    constexpr char transb = 'N';

    libxsmm_sgemm(
        &transa     /*transa*/,
        &transb     /*transb*/,
        &NN         /*required*/,
        &MM         /*required*/,
        &KK         /*required*/,
        &alpha      /*alpha*/,
        b_data      /*required*/,
        &NN         /*lda*/,
        a_data      /*required*/,
        &KK         /*ldb*/,
        &beta       /*beta*/,
        out_data    /*required*/,
        &NN         /*ldc*/
        );
}

// double
template<size_t M, size_t K, size_t N>
FASTOR_INLINE
void matmulNN_libxsmm(
    const double * FASTOR_RESTRICT a_data,
    const double * FASTOR_RESTRICT b_data,
    double * FASTOR_RESTRICT out_data) {

    constexpr int MM= M;
    constexpr int KK= K;
    constexpr int NN= N;
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    constexpr char transa = 'N';
    constexpr char transb = 'N';

    libxsmm_dgemm(
        &transa     /*transa*/,
        &transb     /*transb*/,
        &NN         /*required*/,
        &MM         /*required*/,
        &KK         /*required*/,
        &alpha      /*alpha*/,
        b_data      /*required*/,
        &NN         /*lda*/,
        a_data      /*required*/,
        &KK         /*ldb*/,
        &beta       /*beta*/,
        out_data    /*required*/,
        &NN         /*ldc*/
        );
}


template<size_t M, size_t K, size_t N>
FASTOR_INLINE
void matmulTN_libxsmm(
    const double * FASTOR_RESTRICT a_data,
    const double * FASTOR_RESTRICT b_data,
    double * FASTOR_RESTRICT out_data) {

    constexpr int MM= M;
    constexpr int KK= K;
    constexpr int NN= N;
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    constexpr char transa = 'N';
    constexpr char transb = 'Y';

    libxsmm_dgemm(
        &transa     /*transa*/,
        &transb     /*transb*/,
        &NN         /*required*/,
        &MM         /*required*/,
        &KK         /*required*/,
        &alpha      /*alpha*/,
        b_data      /*required*/,
        &NN         /*lda*/,
        a_data      /*required*/,
        &MM         /*ldb*/,
        &beta       /*beta*/,
        out_data    /*required*/,
        &NN         /*ldc*/
        );
}


template<size_t M, size_t K, size_t N>
FASTOR_INLINE
void matmulNT_libxsmm(
    const double * FASTOR_RESTRICT a_data,
    const double * FASTOR_RESTRICT b_data,
    double * FASTOR_RESTRICT out_data) {

    constexpr int MM= M;
    constexpr int KK= K;
    constexpr int NN= N;
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    constexpr char transa = 'Y';
    constexpr char transb = 'N';

    libxsmm_dgemm(
        &transa     /*transa*/,
        &transb     /*transb*/,
        &NN         /*required*/,
        &MM         /*required*/,
        &KK         /*required*/,
        &alpha      /*alpha*/,
        b_data      /*required*/,
        &KK         /*lda*/,
        a_data      /*required*/,
        &KK         /*ldb*/,
        &beta       /*beta*/,
        out_data    /*required*/,
        &NN         /*ldc*/
        );
}


template<size_t M, size_t K, size_t N>
FASTOR_INLINE
void matmulTT_libxsmm(
    const double * FASTOR_RESTRICT a_data,
    const double * FASTOR_RESTRICT b_data,
    double * FASTOR_RESTRICT out_data) {

    constexpr int MM= M;
    constexpr int KK= K;
    constexpr int NN= N;
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    constexpr char transa = 'Y';
    constexpr char transb = 'Y';

    libxsmm_dgemm(
        &transa     /*transa*/,
        &transb     /*transb*/,
        &NN         /*required*/,
        &MM         /*required*/,
        &KK         /*required*/,
        &alpha      /*alpha*/,
        b_data      /*required*/,
        &KK         /*lda*/,
        a_data      /*required*/,
        &MM         /*ldb*/,
        &beta       /*beta*/,
        out_data    /*required*/,
        &NN         /*ldc*/
        );
}


template<typename T, size_t M, size_t K, size_t N,
    typename std::enable_if<std::is_same<T,double>::value,bool>::type=0>
FASTOR_INLINE
void matmul_libxsmm(
    const T * FASTOR_RESTRICT a_data,
    const T * FASTOR_RESTRICT b_data,
    T * FASTOR_RESTRICT out_data) {
    matmulNN_libxsmm<M,K,N>(a_data,b_data,out_data);
}

template<typename T, size_t M, size_t K, size_t N,
    typename std::enable_if<std::is_same<T,float>::value,bool>::type=0>
FASTOR_INLINE
void matmul_libxsmm(
    const T * FASTOR_RESTRICT a_data,
    const T * FASTOR_RESTRICT b_data,
    T * FASTOR_RESTRICT out_data) {
    matmulNN_libxsmm<M,K,N>(a_data,b_data,out_data);
}

} // end of namespace blas
} // end of namespace Fastor

#endif // FASTOR_USE_LIBXSMM

#endif // LIBXSMM_BACKEND_H
