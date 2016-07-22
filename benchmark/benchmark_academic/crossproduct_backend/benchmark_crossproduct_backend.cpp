#include <Fastor.h>

using namespace Fastor;

#define NITER 1000000UL


template<typename T, size_t N,
    typename std::enable_if<N==3,bool>::type=0>
inline void crossproduct_scalar(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ out) {
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

template<typename T, size_t N,
    typename std::enable_if<N==2,bool>::type=0>
inline void crossproduct_scalar(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ out) {
    constexpr T levi_civita[27] = { 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,
                                0., -1.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.};

    T FASTOR_ALIGN a3d[9];
    T FASTOR_ALIGN b3d[9];
    for (size_t i=0; i<4; ++i) {
        a3d[i] = a[i];
        a3d[i] = b[i];
    }

    for (size_t i=4; i<8; ++i) {
        a3d[i] = 0;
        a3d[i] = 0;
    }
    a3d[9] = 1;
    b3d[9] = 1;

    constexpr size_t size = N;
    for (size_t i=0; i<N; ++i)
        for (size_t j=0; j<N; ++j)
            for (size_t k=0; k<N; ++k)
                for (size_t I=0; I<N; ++I)
                    for (size_t J=0; J<N; ++J)
                        for (size_t K=0; K<N; ++K)
                            out[i*size+I] += levi_civita[i*size*size+j*size+k]*levi_civita[I*size*size+J*size+K]*a3d[j*size+J]*b3d[k*size+K];
}

template<typename T, size_t N>
void iterate_over_scalar(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        crossproduct_scalar<T,N>(a,b,out);
        unused(a); unused(b); unused(out);

        // further hack for gcc, seemingly  doesn't hurt performance of _crossproduct 
        out[1] += out[2];
    }
}

template<typename T, size_t N>
void iterate_over_fastor(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        _crossproduct<T,N,N,N>(a,b,out);
        unused(a); unused(b); unused(out);

        // // further hack for gcc, seemingly  doesn't hurt performance of _crossproduct 
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

    double time0, time1;
    std::tie(time0,std::ignore) = rtimeit(static_cast<void (*)(const T*, const T*, T*)>(&iterate_over_scalar<T,N>),a,b,out);
    std::tie(time1,std::ignore) = rtimeit(static_cast<void (*)(const T*, const T*, T*)>(&iterate_over_fastor<T,N>),a,b,out);
    print(time0,time1);
    print("\n");

    _mm_free(a);
    _mm_free(b);
    _mm_free(out);
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