#include <Fastor/Fastor.h>

using namespace Fastor;

#define NITER 1000000UL


template<typename T, size_t N>
inline void cyclic_scalar(const T *FASTOR_RESTRICT a, const T *FASTOR_RESTRICT b, T *FASTOR_RESTRICT out) {
    constexpr size_t size = N;
    for (size_t i=0; i<N; ++i)
        for (size_t j=0; j<N; ++j)
            for (size_t k=0; k<N; ++k)
                for (size_t l=0; l<N; ++l)
                            out[i*size*size*size+j*size*size+k*size+l] += a[i*size+k]*b[j*size+l];
}


template<typename T, size_t N>
void iterate_over_scalar(const T *FASTOR_RESTRICT a, const T *FASTOR_RESTRICT b, T *FASTOR_RESTRICT out) {
    size_t iter = 0;
    T tmp[N*N*N*N];
    for (; iter<NITER; ++iter) {
        cyclic_scalar<T,N>(a,b,tmp);
        _voigt<T,N,N,N,N>(tmp,out);
        unused(out);
    }
}

template<typename T, size_t N>
void iterate_over_fastor(const T *FASTOR_RESTRICT a, const T *FASTOR_RESTRICT b, T *FASTOR_RESTRICT out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        _cyclic<T,N,N,N,N>(a,b,out);
        unused(out);
    }
}

template<typename T, size_t M, size_t N>
void run() {

    T *a  = static_cast<T*>(_mm_malloc(sizeof(T) * M*N, 32));
    T *b  = static_cast<T*>(_mm_malloc(sizeof(T) * M*N, 32));
    T *out;
    if (M*N==4) out = static_cast<T*>(_mm_malloc(sizeof(T) *9, 32));
    else if (M*N==9) out = static_cast<T*>(_mm_malloc(sizeof(T) *36, 32));

    std::iota(a,a+M*N,0);
    std::iota(b,b+M*N,0);

    double time_scalar, time_fastor;
    uint64_t cycles_scalar, cycles_fastor;

    std::tie(time_scalar, cycles_scalar) = rtimeit(static_cast<void (*)(const T*, const T*, T*)>(&iterate_over_scalar<T,N>),a,b,out);
    std::tie(time_fastor, cycles_fastor) = rtimeit(static_cast<void (*)(const T*, const T*, T*)>(&iterate_over_fastor<T,N>),a,b,out);

    int64_t saved_cycles = int64_t((double)cycles_scalar/(double)(NITER) - (double)cycles_fastor/(double)(NITER));
    auto &&w = std::fixed;
    println(FGRN(BOLD("Speed-up over scalar code [elapsed time]")), time_scalar/time_fastor,
        FGRN(BOLD("[saved CPU cycles]")), saved_cycles);
    print();

    _mm_free(a);
    _mm_free(b);
    _mm_free(out);
}


int main() {

    print(FBLU(BOLD("Running cyclic dyadic product benchmarks [Benchmarks SIMD vectorisation and zero elimination]")));
    // Not implemented yet
    // print("Single precision benchmark");
    // run<float,2,2>();
    // run<float,3,3>();
    print("Double precision benchmark");
    run<double,2,2>();
    run<double,3,3>();

    return 0;
}
