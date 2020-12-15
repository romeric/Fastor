#ifndef MKL_BACKEND_H
#define MKL_BACKEND_H

#include <Fastor/tensor/Tensor.h>

#ifdef FASTOR_USE_MKL

// Explicitly activate the jit
#ifndef MKL_DIRECT_CALL_SEQ_JIT
#define MKL_DIRECT_CALL_SEQ_JIT
#endif

#include <mkl.h>

namespace Fastor {
namespace blas {

// single
template<typename T, size_t M, size_t K, size_t N, enable_if_t_<is_same_v_<T,float>,bool> = false>
void matmul_mkl(
    const float * FASTOR_RESTRICT a_data,
    const float * FASTOR_RESTRICT b_data,
    float * FASTOR_RESTRICT out_data) {

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, a_data, K, b_data, N, 0.0, out_data, N);

}

// double
template<typename T, size_t M, size_t K, size_t N, enable_if_t_<is_same_v_<T,double>,bool> = false>
void matmul_mkl(
    const double * FASTOR_RESTRICT a_data,
    const double * FASTOR_RESTRICT b_data,
    double * FASTOR_RESTRICT out_data) {

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, a_data, K, b_data, N, 0.0, out_data, N);

}


#if 0
// dedicated jit api, but the jit kernel has to be created before hand
template<typename T, size_t M, size_t K, size_t N>
Tensor<T,M,N> matmul_mkl_jit_api(const Tensor<T,M,K> &a, const Tensor<T,K,N> &b) {

    Tensor<T,M,N> out;
    void* jitter;
    mkl_jit_status_t status = mkl_jit_create_dgemm(&jitter, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS, M, N, K, 1.0, K, N, 0.0, N);

    dgemm_jit_kernel_t _dgemm_kernel = mkl_jit_get_dgemm_ptr(jitter);

    _dgemm_kernel(jitter, a.data(), b.data(), out.data());

    mkl_jit_destroy(jitter);

    return out;
}
#endif

} // end of namespace blas
} // end of namespace Fastor

#endif // FASTOR_USE_MKL

#endif // MKL_BACKEND_H
