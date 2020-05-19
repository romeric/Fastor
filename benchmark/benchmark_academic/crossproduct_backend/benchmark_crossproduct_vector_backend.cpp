#include <Fastor/Fastor.h>

using namespace Fastor;

#define NITER 1000000UL


template<typename T, size_t N>
inline void crossproduct_scalar_vt(const T *FASTOR_RESTRICT a, const T *FASTOR_RESTRICT b, T *FASTOR_RESTRICT out) {
    constexpr T levi_civita[27] = { 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,
                                0., -1.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.};
    constexpr size_t size = N;
    for (size_t i=0; i<N; ++i)
        for (size_t j=0; j<N; ++j)
            for (size_t k=0; k<N; ++k)
                for (size_t l=0; l<N; ++l)
                        out[i*size+j] += levi_civita[i*size*size+k*size+l]*a[k]*b[l*size+j];
}

template<typename T, size_t N>
inline void crossproduct_scalar_tv(const T *FASTOR_RESTRICT a, const T *FASTOR_RESTRICT b, T *FASTOR_RESTRICT out) {
    constexpr T levi_civita[27] = { 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,
                                0., -1.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.};
    constexpr size_t size = N;
    for (size_t i=0; i<N; ++i)
        for (size_t j=0; j<N; ++j)
            for (size_t k=0; k<N; ++k)
                for (size_t l=0; l<N; ++l)
                        out[i*size+j] += levi_civita[j*size*size+k*size+l]*a[i*size+k]*b[l];
}


template<typename T, size_t N>
void iterate_over_scalar_vt(const T *FASTOR_RESTRICT a, const T *FASTOR_RESTRICT b, T *FASTOR_RESTRICT out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        crossproduct_scalar_vt<T,N>(a,b,out);
        unused(a); unused(b); unused(out);

        // further hack for gcc, seemingly  doesn't hurt performance of _crossproduct
        out[1] += out[2];
    }
}

template<typename T, size_t N>
void iterate_over_scalar_tv(const T *FASTOR_RESTRICT a, const T *FASTOR_RESTRICT b, T *FASTOR_RESTRICT out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        crossproduct_scalar_tv<T,N>(a,b,out);
        unused(a); unused(b); unused(out);

        // further hack for gcc, seemingly  doesn't hurt performance of _crossproduct
        out[1] += out[2];
    }
}

template<typename T, size_t N>
void iterate_over_fastor_vt(const T *FASTOR_RESTRICT a, const T *FASTOR_RESTRICT b, T *FASTOR_RESTRICT out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        _crossproduct<T,N,1,N>(a,b,out);
        unused(a); unused(b); unused(out);

        // further hack for gcc, seemingly  doesn't hurt performance of _crossproduct
        out[1] += out[2];
    }
}

template<typename T, size_t N>
void iterate_over_fastor_tv(const T *FASTOR_RESTRICT a, const T *FASTOR_RESTRICT b, T *FASTOR_RESTRICT out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        _crossproduct<T,N,N,1>(a,b,out);
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

    double time0, time1;
    std::tie(time0,std::ignore) = rtimeit(static_cast<void (*)(const T*, const T*, T*)>(&iterate_over_scalar_vt<T,N>),a,b,out);
    std::tie(time1,std::ignore) = rtimeit(static_cast<void (*)(const T*, const T*, T*)>(&iterate_over_fastor_vt<T,N>),a,b,out);
    print(time0,time1);
    print("\n");
    std::tie(time0,std::ignore) = rtimeit(static_cast<void (*)(const T*, const T*, T*)>(&iterate_over_scalar_tv<T,N>),a,b,out);
    std::tie(time1,std::ignore) = rtimeit(static_cast<void (*)(const T*, const T*, T*)>(&iterate_over_fastor_tv<T,N>),a,b,out);
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
