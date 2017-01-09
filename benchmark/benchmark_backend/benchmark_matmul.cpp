#include <Fastor.h>
using real = double;
using namespace Fastor;

#define NITER 1000000UL


template<typename T, size_t M, size_t K, size_t N>
inline void matmul_scalar(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ out) {
    std::fill(out,out+M*N,0.);
    for (size_t i=0; i<M; ++i) {
        for (size_t j=0; j<K; ++j) {
            for (size_t k=0; k<N; ++k) {
                out[i*N+k] += a[i*K+j]*b[j*N+k];
            }
        }
    }
}

template<typename T, size_t M, size_t K, size_t N>
void iterate_over_scalar(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        matmul_scalar<T,M,K,N>(a,b,out);
        unused(out);
    }
}

template<typename T, size_t M, size_t K, size_t N>
void iterate_over_fastor(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        _matmul<T,M,K,N>(a,b,out);
        unused(out);
    }    
}


template<typename T, size_t M, size_t K, size_t N>
void run() {

    T *a  = static_cast<T*>(_mm_malloc(sizeof(T) * M*K, 32));
    T *b  = static_cast<T*>(_mm_malloc(sizeof(T) * K*N, 32));
    T *out = static_cast<T*>(_mm_malloc(sizeof(T) * M*N, 32));

    std::iota(a,a+M*K,0.);
    std::iota(b,b+N*K,0.);

    double time_scalar, time_fastor;
    uint64_t cycles_scalar, cycles_fastor;

    std::tie(time_scalar, cycles_scalar) = rtimeit(static_cast<void (*)(const T*, const T*, T*)>(&iterate_over_scalar<T,M,K,N>),a,b,out);
    std::tie(time_fastor, cycles_fastor) = rtimeit(static_cast<void (*)(const T*, const T*, T*)>(&iterate_over_fastor<T,M,K,N>),a,b,out);

    int64_t saved_cycles = int64_t((double)cycles_scalar/(double)(NITER) - (double)cycles_fastor/(double)(NITER));
    auto &&w = std::fixed;
    println(FBLU(BOLD("Matrices size (M, K, N)")), M, K, N);
    println(FGRN(BOLD("Speed-up over scalar code [elapsed time]")), time_scalar/time_fastor, 
        FGRN(BOLD("[saved CPU cycles]")), saved_cycles);
    print();

    _mm_free(a);
    _mm_free(b);
    _mm_free(out);
}


int main() {

    print(FBLU(BOLD("Running tensor matmul benchmarks [Benchmarks SIMD vectorisation]")));
    print("Single precision benchmark");
    run<float,2,2,2>();
    run<float,3,3,3>();
    run<float,4,4,4>();
    run<float,8,8,8>();
    print("Double precision benchmark");
    run<double,2,2,2>();
    run<double,3,3,3>();
    run<double,4,4,4>();
    run<double,8,8,8>();

    print(FBLU(BOLD("Running customised matmul kernels for 2D numerical quadrature [Benchmarks SIMD vectorisation]")));
    run<real,2,3,2>();
    run<real,2,6,2>();
    run<real,2,10,2>();
    run<real,2,15,2>();
    run<real,2,21,2>();
    run<real,2,28,2>();
    run<real,2,36,2>();
    run<real,2,45,2>();
    run<real,2,55,2>();
    run<real,2,66,2>();
    run<real,2,78,2>();
    run<real,2,91,2>();
    run<real,2,105,2>();
    run<real,2,120,2>();
    run<real,2,136,2>();
    run<real,2,153,2>();
    run<real,2,171,2>();
    run<real,2,190,2>();
    run<real,2,210,2>();
    run<real,2,231,2>();
    run<real,2,253,2>();
    run<real,2,276,2>();
    run<real,2,300,2>();
    run<real,2,325,2>();
    run<real,2,351,2>();
    run<real,2,378,2>();
    run<real,2,406,2>();
    run<real,2,435,2>();
    run<real,2,465,2>();
    run<real,2,496,2>();

    print(FBLU(BOLD("Running customised matmul kernels for 3D numerical quadrature [Benchmarks SIMD vectorisation]")));
    run<real,3,4,3>();
    run<real,3,10,3>();
    run<real,3,20,3>();
    run<real,3,35,3>();
    run<real,3,56,3>();
    run<real,3,84,3>();
    run<real,3,120,3>();
    run<real,3,165,3>();
    run<real,3,220,3>();
    run<real,3,286,3>();
    run<real,3,364,3>();
    run<real,3,455,3>();
    run<real,3,560,3>();
    run<real,3,680,3>();
    run<real,3,816,3>();
    run<real,3,969,3>();
    run<real,3,1140,3>();
    run<real,3,1330,3>();
    run<real,3,1540,3>();
    run<real,3,1771,3>();
    run<real,3,2024,3>();
    run<real,3,2300,3>();
    run<real,3,2600,3>();
    run<real,3,2925,3>();
    run<real,3,3276,3>();
    run<real,3,3654,3>();
    run<real,3,4060,3>();
    run<real,3,4495,3>();
    run<real,3,4960,3>();
    run<real,3,5456,3>();


    return 0;
}