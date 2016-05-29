#include <Fastor.h>

using namespace Fastor;

#define NITER 1000000UL

template <typename T> void unused(T &&x) { asm("" ::"m"(x)); }

template<typename T, size_t N>
inline T trace_scalar(const T *__restrict__ in) {
    T sum = 0;
    for (size_t i=0; i<N; ++i)
            sum += in[i*N+i];
    return sum;
}

template<typename T, size_t N>
void iterate_over_scalar(const T *__restrict__ in) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        T trace = trace_scalar<T,N>(in);
        unused(trace);
    }
}

template<typename T, size_t N>
void iterate_over_fastor(const T *__restrict__ in) {
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

    timeit(static_cast<void (*)(const T*)>(&iterate_over_scalar<T,N>),in);
    timeit(static_cast<void (*)(const T*)>(&iterate_over_fastor<T,N>),in);

    _mm_free(in);
}


int main() {

    print("Single precision benchmark");
    run<float,2,2>();
    run<float,3,3>();
    print("Double precision benchmark");
    run<double,2,2>();
    run<double,3,3>();

    return 0;
}