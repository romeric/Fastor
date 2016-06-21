#include <Fastor.h>

using namespace Fastor;

#define NITER 1000000UL


template<typename T, size_t N>
inline void cyclic_scalar(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ out) {
    constexpr size_t size = N;
    for (size_t i=0; i<N; ++i)
        for (size_t j=0; j<N; ++j)
            for (size_t k=0; k<N; ++k)
                for (size_t l=0; l<N; ++l)
                            out[i*size*size*size+j*size*size+k*size+l] += a[i*size+k]*b[j*size+l];
}


template<typename T, size_t N>
void iterate_over_scalar(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ out) {
    size_t iter = 0;
    T tmp[N*N*N*N];
    for (; iter<NITER; ++iter) {
        cyclic_scalar<T,N>(a,b,tmp);
        _voigt<T,N,N,N,N>(tmp,out);
        unused(out);
    }
}

template<typename T, size_t N>
void iterate_over_fastor(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ out) {
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

    timeit(static_cast<void (*)(const T*, const T*, T*)>(&iterate_over_scalar<T,N>),a,b,out);
    timeit(static_cast<void (*)(const T*, const T*, T*)>(&iterate_over_fastor<T,N>),a,b,out);

    _mm_free(a);
    _mm_free(b);
    _mm_free(out);
}


int main() {

    // Not implemented yet
    // print("Single precision benchmark");
    // run<float,2,2>();
    // run<float,3,3>();
    print("Double precision benchmark");
    run<double,2,2>();
    run<double,3,3>();

    return 0;
}