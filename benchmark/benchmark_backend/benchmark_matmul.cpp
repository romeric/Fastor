#include <Fastor.h>

using namespace Fastor;

#define NITER 1000000UL


template<typename T, size_t M, size_t N, size_t K>
inline void matmul_scalar(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ out) {
    for (size_t i=0; i<M; ++i)
        for (size_t j=0; j<N; ++j)
            for (size_t k=0; k<K; ++k)
                out[i*N+k] += a[i*K+j]*b[j*N+k];
}

template<typename T, size_t M, size_t N, size_t K>
void iterate_over_scalar(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        matmul_scalar<T,M,N,K>(a,b,out);
        unused(out);
    }
}

template<typename T, size_t M, size_t N, size_t K>
void iterate_over_fastor(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        _matmul<T,M,N,K>(a,b,out);
        unused(out);
    }    
}


template<typename T, size_t M, size_t N, size_t K>
void run() {

    T *a  = static_cast<T*>(_mm_malloc(sizeof(T) * M*K, 32));
    T *b  = static_cast<T*>(_mm_malloc(sizeof(T) * K*N, 32));
    T *out = static_cast<T*>(_mm_malloc(sizeof(T) * M*N, 32));

    std::iota(a,a+M*K,0);
    std::iota(b,b+N*K,0);

    timeit(static_cast<void (*)(const T*, const T*, T*)>(&iterate_over_scalar<T,M,N,K>),a,b,out);
    timeit(static_cast<void (*)(const T*, const T*, T*)>(&iterate_over_fastor<T,M,N,K>),a,b,out);

    _mm_free(a);
    _mm_free(b);
    _mm_free(out);
}


int main() {

    print("Single precision benchmark");
    run<float,2,2,2>();
    run<float,3,3,3>();
    run<float,4,4,4>();
    run<float,8,8,8>();
    print("Double precision benchmark");
    run<double,2,2,2>();
    run<double,3,3,3>();
    run<double,4,4,4>();
    run<double,16,16,16>();

    return 0;
}