#include <Fastor/Fastor.h>

using namespace Fastor;

#define NITER 1000000UL


template<typename T, size_t N>
inline T trace_scalar(const T *FASTOR_RESTRICT in) {
    T sum = 0;
    for (size_t i=0; i<N; ++i)
            sum += in[i*N+i];
    return sum;
}

template<typename T, size_t N>
void iterate_over_scalar(const T *FASTOR_RESTRICT in) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        T trace = trace_scalar<T,N>(in);
        unused(trace);
    }
}

template<typename T, size_t N>
void iterate_over_fastor(const T *FASTOR_RESTRICT in) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        T trace = _trace<T,N,N>(in);
        unused(trace);
    }
}


template<typename T, size_t M, size_t N>
void run() {

    T *in  = static_cast<T*>(_mm_malloc(sizeof(T) * M*N, 32));
    std::iota(in,in+M*N,0);

    double time_scalar, time_fastor;
    uint64_t cycles_scalar, cycles_fastor;

    std::tie(time_scalar, cycles_scalar) = rtimeit(static_cast<void (*)(const T*)>(&iterate_over_scalar<T,N>),in);
    std::tie(time_fastor, cycles_fastor) = rtimeit(static_cast<void (*)(const T*)>(&iterate_over_fastor<T,N>),in);

    int64_t saved_cycles = int64_t((double)cycles_scalar/(double)(NITER) - (double)cycles_fastor/(double)(NITER));
    auto &&w = std::fixed;
    println(FGRN(BOLD("Speed-up over scalar code [elapsed time]")), time_scalar/time_fastor,
        FGRN(BOLD("[saved CPU cycles]")), saved_cycles);
    print();

    _mm_free(in);
}


int main() {

    print(FBLU(BOLD("Running tensor trace operator benchmarks [Benchmarks SIMD vectorisation]")));
    print("Single precision benchmark");
    run<float,2,2>();
    run<float,3,3>();
    print("Double precision benchmark");
    run<double,2,2>();
    run<double,3,3>();

    return 0;
}