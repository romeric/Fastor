#include <Fastor/Fastor.h>
using namespace Fastor;

// mkl_direct_call_seq_jit_impl
//-----------------------------------------------------------------------------------------------------------
// #include <mkl.h>
#include <mkl_cblas.h>
template<typename T, size_t M, size_t K, size_t N, enable_if_t_<is_same_v_<T,double>,bool> = false>
Tensor<T,M,N> matmul_mkl_direct_call_seq_jit_impl(const Tensor<T,M,K> &a, const Tensor<T,K,N> &b) {
    Tensor<T,M,N> out;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, a.data(), K, b.data(), N, 0.0, out.data(), N);
    return out;
}
template<typename T, size_t M, size_t K, size_t N, enable_if_t_<is_same_v_<T,float>,bool> = false>
Tensor<T,M,N> matmul_mkl_direct_call_seq_jit_impl(const Tensor<T,M,K> &a, const Tensor<T,K,N> &b) {
    Tensor<T,M,N> out;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, a.data(), K, b.data(), N, 0.0, out.data(), N);
    return out;
}

template<typename T, size_t M, size_t K, size_t N>
inline void matmul_mkl_direct_call_seq_jit() {
    Tensor<T,M,K> a(3);
    Tensor<T,K,N> b(4);
    Tensor<T,M,N> c = matmul_mkl_direct_call_seq_jit_impl(a,b);
    unused(c);
}
//-----------------------------------------------------------------------------------------------------------


// mkl_jit_api
//-----------------------------------------------------------------------------------------------------------
template<typename T, size_t M, size_t K, size_t N, enable_if_t_<is_same_v_<T,float>,bool> = false>
void matmul_mkl_jit_api_impl(void *jitter) {
    Tensor<T,M,K> a(3);
    Tensor<T,K,N> b(4);
    Tensor<T,M,N> c;
    sgemm_jit_kernel_t _gemm_kernel = mkl_jit_get_sgemm_ptr(jitter);
    _gemm_kernel(jitter, a.data(), b.data(), c.data());
    unused(c);
}
template<typename T, size_t M, size_t K, size_t N, enable_if_t_<is_same_v_<T,double>,bool> = false>
void matmul_mkl_jit_api_impl(void *jitter) {
    Tensor<T,M,K> a(3);
    Tensor<T,K,N> b(4);
    Tensor<T,M,N> c;
    dgemm_jit_kernel_t _gemm_kernel = mkl_jit_get_dgemm_ptr(jitter);
    _gemm_kernel(jitter, a.data(), b.data(), c.data());
    unused(c);
}

template<typename T, size_t M, size_t K, size_t N, enable_if_t_<is_same_v_<T,float>,bool> = false>
void bench_matmul_mkl_jit_api() {
    void* jitter;
    mkl_jit_status_t status = mkl_jit_create_sgemm(&jitter, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS, M, N, K, 1.0, K, N, 0.0, N);
    timeit(static_cast<void (*)(void*)>(&matmul_mkl_jit_api_impl<T,M,K,N>),jitter);
    mkl_jit_destroy(jitter);
}
template<typename T, size_t M, size_t K, size_t N, enable_if_t_<is_same_v_<T,double>,bool> = false>
void bench_matmul_mkl_jit_api() {
    void* jitter;
    mkl_jit_status_t status = mkl_jit_create_dgemm(&jitter, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS, M, N, K, 1.0, K, N, 0.0, N);
    timeit(static_cast<void (*)(void*)>(&matmul_mkl_jit_api_impl<T,M,K,N>),jitter);
    mkl_jit_destroy(jitter);
}
//-----------------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------------
#include <blaze/math/StaticMatrix.h>
template<typename T, size_t M, size_t K, size_t N>
inline void matmul_blaze_unpadded() {
    blaze::StaticMatrix<T,M,K,blaze::rowMajor,blaze::unaligned,blaze::unpadded> a(3);
    blaze::StaticMatrix<T,K,N,blaze::rowMajor,blaze::unaligned,blaze::unpadded> b(4);
    blaze::StaticMatrix<T,M,N,blaze::rowMajor,blaze::unaligned,blaze::unpadded> c = a*b;
    unused(c);
}

template<typename T, size_t M, size_t K, size_t N>
inline void matmul_blaze_padded() {
    blaze::StaticMatrix<T,M,K,blaze::rowMajor,blaze::aligned,blaze::padded> a(3);
    blaze::StaticMatrix<T,K,N,blaze::rowMajor,blaze::aligned,blaze::padded> b(4);
    blaze::StaticMatrix<T,M,N,blaze::rowMajor,blaze::aligned,blaze::padded> c = a*b;
    unused(c);
}
//-----------------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------------
template<typename T, size_t M, size_t K, size_t N>
inline void matmul_fastor() {
    Tensor<T,M,K> a(3);
    Tensor<T,K,N> b(4);
    Tensor<T,M,N> c = matmul(a,b);
    unused(c);
}
//-----------------------------------------------------------------------------------------------------------




template<typename T, size_t M, size_t K, size_t N>
void bench() {

    println(FBLU(BOLD("Testing size (M, K, N)")), M, K, N,'\n');

    timeit(static_cast<void (*)()>(&matmul_fastor<T,M,K,N>));
    timeit(static_cast<void (*)()>(&matmul_blaze_unpadded<T,M,K,N>));
    timeit(static_cast<void (*)()>(&matmul_blaze_padded<T,M,K,N>));
    timeit(static_cast<void (*)()>(&matmul_mkl_direct_call_seq_jit<T,M,K,N>));
    // bench excluding jitter time
    bench_matmul_mkl_jit_api<T,M,K,N>();
}



int main() {

    // print(__INTEL_MKL__,INTEL_MKL_VERSION);
#ifdef RUN_SINGLE
    using real_ = float;
#else
    using real_ = double;
#endif

    {
        constexpr size_t incr = 0;
        // constexpr size_t M = 8;
        constexpr size_t K = 31;
        constexpr size_t N = 21;

        bench<real_,1+incr,K,N>();
        bench<real_,2+incr,K,N>();
        bench<real_,3+incr,K,N>();
        bench<real_,4+incr,K,N>();
        bench<real_,5+incr,K,N>();
        bench<real_,6+incr,K,N>();
        bench<real_,7+incr,K,N>();
        bench<real_,8+incr,K,N>();
        bench<real_,9+incr,K,N>();
        bench<real_,10+incr,K,N>();
        bench<real_,11+incr,K,N>();
        bench<real_,12+incr,K,N>();
        bench<real_,13+incr,K,N>();
        bench<real_,14+incr,K,N>();
        bench<real_,15+incr,K,N>();
        bench<real_,16+incr,K,N>();
        bench<real_,17+incr,K,N>();
        bench<real_,18+incr,K,N>();
        bench<real_,19+incr,K,N>();
        bench<real_,20+incr,K,N>();
        bench<real_,21+incr,K,N>();
        bench<real_,22+incr,K,N>();
        bench<real_,23+incr,K,N>();
        bench<real_,24+incr,K,N>();
        bench<real_,25+incr,K,N>();
        bench<real_,26+incr,K,N>();
        bench<real_,27+incr,K,N>();
        bench<real_,28+incr,K,N>();
        bench<real_,29+incr,K,N>();
        bench<real_,30+incr,K,N>();
        bench<real_,31+incr,K,N>();
        bench<real_,32+incr,K,N>();
    }

    return 0;

}