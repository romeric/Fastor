#include <Fastor/Fastor.h>

using namespace Fastor;

#define NITER 1000000UL


template<typename T, size_t N>
inline void crossproduct_scalar(const T *FASTOR_RESTRICT a, const T *FASTOR_RESTRICT b, T *FASTOR_RESTRICT out) {
    constexpr T levi_civita[27] = { 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,
                                0., -1.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.};
    constexpr size_t size = N;
    for (size_t i=0; i<N; ++i)
        for (size_t j=0; j<N; ++j)
            for (size_t k=0; k<N; ++k)
                for (size_t I=0; I<N; ++I)
                    for (size_t J=0; J<N; ++J)
                        for (size_t K=0; K<N; ++K)
                            out[i*size+I] += levi_civita[i*size*size+j*size+k]*levi_civita[I*size*size+J*size+K]*a[j*size+J]*b[k*size+K];
}

template<typename T, size_t N>
void iterate_over_scalar(const T *FASTOR_RESTRICT a, const T *FASTOR_RESTRICT b, T *FASTOR_RESTRICT out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        crossproduct_scalar<T,N>(a,b,out);
        unused(a); unused(b); unused(out);

        // further hack for gcc, seemingly  doesn't hurt performance of _crossproduct
        out[1] += out[2];
    }
}

template<typename T, size_t N>
void iterate_over_fastor(const T *FASTOR_RESTRICT a, const T *FASTOR_RESTRICT b, T *FASTOR_RESTRICT out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        _crossproduct<T,N,N,N>(a,b,out);
        unused(a); unused(b); unused(out);

        // further hack for gcc, seemingly  doesn't hurt performance of _crossproduct
        out[1] += out[2];
    }
}


template<typename T, size_t M, size_t N>
void run() {

    T *a  = static_cast<T*>(_mm_malloc(sizeof(T) * M*N, 32));
    T *b  = static_cast<T*>(_mm_malloc(sizeof(T) * M*N, 32));
    T *out = static_cast<T*>(_mm_malloc(sizeof(T) * (M+1)*(N+1), 32)); // take care of 2D to 3D case

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

    print(FBLU(BOLD("Running tensor cross product benchmarks [Benchmarks SIMD vectorisation and zero elimination]")));
    print(FBLU(BOLD("Single precision:")));
    run<float,2,2>();
    run<float,3,3>();
    print(FBLU(BOLD("Double precision:")));
    run<double,2,2>();
    run<double,3,3>();

    return 0;
}