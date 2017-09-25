#include <Fastor.h>

using namespace Fastor;

#define NITER 10000UL


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
                            for (size_t p=0; p<N; ++p)
                                for (size_t P=0; P<N; ++P)
                                    for (size_t q=0; q<N; ++q)
                                        for (size_t Q=0; Q<N; ++Q)
                                            out[p*size*size*size*size*size+P*size*size*size*size+i*size*size*size+I*size*size+q*size+Q] += \
                                                levi_civita[i*size*size+j*size+k]*levi_civita[I*size*size+J*size+K]*\
                                                a[p*size*size*size+P*size*size+j*size+J]*b[k*size*size*size+K*size*size+q*size+Q];
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
        _crossproduct<T,N,N,N,N,N,N,N,N>(a,b,out);
        unused(a); unused(b); unused(out);

        // // further hack for gcc, seemingly  doesn't hurt performance of _crossproduct
        out[1] += out[2];
    }
}


template<typename T, size_t M, size_t N, size_t P, size_t Q>
void run() {

    T *a  = static_cast<T*>(_mm_malloc(sizeof(T) * M*N*P*Q, 32));
    T *b  = static_cast<T*>(_mm_malloc(sizeof(T) * M*N*P*Q, 32));
    T *out = static_cast<T*>(_mm_malloc(sizeof(T) * M*N*P*Q * 9, 32));

    std::iota(a,a+M*N*P*Q,0);
    std::iota(b,b+M*N*P*Q,0);

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
    run<float,3,3,3,3>();
    print("Double precision benchmark");
    run<double,3,3,3,3>();

    return 0;
}