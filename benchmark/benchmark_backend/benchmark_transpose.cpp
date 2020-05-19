#include <Fastor/Fastor.h>

using namespace Fastor;

#define NITER 1000000UL


template<typename T, size_t M, size_t N>
inline void transpose_scalar(const T *FASTOR_RESTRICT in, T *FASTOR_RESTRICT out) {
    for (size_t i=0; i<M; ++i)
        for (size_t j=0; j<N; ++j)
            out[j*M+i] = in[i*N+j];
}

template<typename T, size_t M, size_t N>
void iterate_over_scalar(const T *FASTOR_RESTRICT in, T *FASTOR_RESTRICT out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        transpose_scalar<T,M,N>(in,out);
        unused(out);
    }
}

template<typename T, size_t M, size_t N>
void iterate_over_fastor(const T *FASTOR_RESTRICT in, T *FASTOR_RESTRICT out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        _transpose<T,M,N>(in,out);
        unused(out);
    }
}


template<typename T, size_t M, size_t N>
void run() {

    T *in  = static_cast<T*>(_mm_malloc(sizeof(T) * M*N, 32));
    T *out = static_cast<T*>(_mm_malloc(sizeof(T) * M*N, 32));

    std::iota(in,in+M*N,0);

    double time_scalar, time_fastor;
    uint64_t cycles_scalar, cycles_fastor;

    std::tie(time_scalar, cycles_scalar) = rtimeit(static_cast<void (*)(const T*, T*)>(&iterate_over_scalar<T,M,N>),in,out);
    std::tie(time_fastor, cycles_fastor) = rtimeit(static_cast<void (*)(const T*, T*)>(&iterate_over_fastor<T,M,N>),in,out);

    int64_t saved_cycles = int64_t((double)cycles_scalar/(double)(NITER) - (double)cycles_fastor/(double)(NITER));
    auto &&w = std::fixed;
    println(FGRN(BOLD("Speed-up over scalar code [elapsed time]")), time_scalar/time_fastor,
        FGRN(BOLD("[saved CPU cycles]")), saved_cycles);
    print();

    _mm_free(in);
    _mm_free(out);
}


int main() {

    print(FBLU(BOLD("Running tensor transpose operator benchmarks [Benchmarks SIMD vectorisation]")));
    print("Single precision benchmark");
    run<float,2,2>();
    run<float,3,3>();
    run<float,4,4>();
    run<float,8,8>();
    print("Double precision benchmark");
    run<double,2,2>();
    run<double,3,3>();
    run<double,4,4>();

    return 0;
}
