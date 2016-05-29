#include <Fastor.h>

using namespace Fastor;

#define NITER 1000000UL

template <typename T> void unused(T &&x) { asm("" ::"m"(x)); }

template<typename T, size_t M, size_t N>
inline void transpose_scalar(const T *__restrict__ in, T *__restrict__ out) {
    for (size_t i=0; i<M; ++i)
        for (size_t j=0; j<N; ++j)
            out[i*N+j] = in[j*M+i];
}

template<typename T, size_t M, size_t N>
void iterate_over_scalar(const T *__restrict__ in, T *__restrict__ out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        transpose_scalar<T,M,N>(in,out);
        unused(out);
    }
}

template<typename T, size_t M, size_t N>
void iterate_over_fastor(const T *__restrict__ in, T *__restrict__ out) {
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

    timeit(static_cast<void (*)(const T*, T*)>(&iterate_over_scalar<T,M,N>),in,out);
    timeit(static_cast<void (*)(const T*, T*)>(&iterate_over_fastor<T,M,N>),in,out);

    _mm_free(in);
    _mm_free(out);
}


int main() {

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