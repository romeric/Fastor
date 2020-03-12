#include <Fastor/Fastor.h>

using namespace Fastor;

#define NITER 1000000UL

template<typename T, size_t M, size_t N>
inline T doublecontract_scalar(const T *FASTOR_RESTRICT in, T *FASTOR_RESTRICT out) {
    T sum = 0;
    for (size_t i=0; i<M; ++i)
        for (size_t j=0; j<N; ++j)
            sum += out[i*N+j]*in[i*N+j];
    return sum;
}

template<typename T, size_t M, size_t N>
void iterate_over_scalar(const T *FASTOR_RESTRICT in, T *FASTOR_RESTRICT out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        T dc = doublecontract_scalar<T,M,N>(in,out);
        unused(dc);
    }
}

template<typename T, size_t M, size_t N>
void iterate_over_fastor(const T *FASTOR_RESTRICT in, T *FASTOR_RESTRICT out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        T dc = _doublecontract<T,M,N>(in,out);
        unused(dc);
    }
}


template<typename T, size_t M, size_t N>
void run() {

    T *in  = static_cast<T*>(_mm_malloc(sizeof(T) * M*N, 32));
    T *out = static_cast<T*>(_mm_malloc(sizeof(T) * M*N, 32));
    std::iota(in,in+M*N,1);
    std::iota(out,out+M*N,2);

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

    print(FBLU(BOLD("Running tensor double index contraction benchmarks [Benchmarks SIMD vectorisation]")));
    print("Single precision benchmark");
    run<float,2,2>();
    run<float,3,3>();
    run<float,4,4>();
    run<float,8,8>();
    run<float,16,16>();
    run<float,32,32>();
    run<float,64,64>();
    print("Double precision benchmark");
    run<double,2,2>();
    run<double,3,3>();
    run<double,4,4>();
    run<double,8,8>();
    run<double,16,16>();
    run<double,32,32>();
    run<double,64,64>();

    return 0;
}