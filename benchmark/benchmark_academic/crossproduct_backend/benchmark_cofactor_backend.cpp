#include <Fastor/Fastor.h>

using namespace Fastor;

#define NITER 1000000UL

template<typename T, size_t N>
void iterate_over_classical(const T *FASTOR_RESTRICT a, T *FASTOR_RESTRICT out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        // static version
        T det = _det<T,N,N>(a);
        FASTOR_ALIGN T adj[N*N];
        _adjoint<T,N>(a,adj);
        FASTOR_ALIGN T trans[N*N];
        _transpose<T,N,N>(adj,trans);
        for (int i=0; i<N*N; ++i)
            out[i] = trans[i]/det;

        unused(a); unused(out);

        // further hack for gcc, seemingly  doesn't hurt performance of _crossproduct
        out[1] += out[2];
    }
}

template<typename T, size_t N,
    typename std::enable_if<N==3,bool>::type = 0>
void iterate_over_fastor_cross(const T *FASTOR_RESTRICT a, T *FASTOR_RESTRICT out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        _crossproduct<T,N,N,N>(a,a,out);
        unused(a); unused(out);

        // further hack for gcc, seemingly  doesn't hurt performance of _crossproduct
        out[1] += out[2];
    }
}

#if 0
template<typename T, size_t N,
    typename std::enable_if<N==2,bool>::type = 0>
void iterate_over_fastor_cross(const T *FASTOR_RESTRICT a, T *FASTOR_RESTRICT out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        _crossproduct<T,PlaneStrain>(a,a,out);
        unused(a); unused(out);

        // further hack for gcc, seemingly  doesn't hurt performance of _crossproduct
        out[1] += out[2];
    }
}
#endif

template<typename T, size_t N>
void iterate_over_fastor_cof(const T *FASTOR_RESTRICT a, T *FASTOR_RESTRICT out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        _cofactor<T,N>(a,out);
        unused(a); unused(out);

        // further hack for gcc, seemingly  doesn't hurt performance of _crossproduct
        out[1] += out[2];
    }
}


template<typename T, size_t M, size_t N>
void run() {


    T *a  = static_cast<T*>(_mm_malloc(sizeof(T) * 9, 32));
    T *out = static_cast<T*>(_mm_malloc(sizeof(T) * 9, 32));

    if (M==3 && N==3) {
        std::iota(a,a+M*N,0);
    }

    if (M==2 && N==2) {
        std::iota(a,a+6,0);
        a[8]=1;
    }


    double time0, time1, time2;
    std::tie(time0,std::ignore) = rtimeit(static_cast<void (*)(const T*, T*)>(&iterate_over_classical<T,N>),a,out);
    std::tie(time1,std::ignore) = rtimeit(static_cast<void (*)(const T*, T*)>(&iterate_over_fastor_cross<T,N>),a,out);
    std::tie(time2,std::ignore) = rtimeit(static_cast<void (*)(const T*, T*)>(&iterate_over_fastor_cof<T,N>),a,out);
    print(time0,time1,time2);
    print("\n");

    _mm_free(a);
    _mm_free(out);
}


int main() {

    print("Single precision benchmark");
#if 0
    run<float,2,2>();
#endif
    run<float,3,3>();
    print("Double precision benchmark");
#if 0
    run<double,2,2>();
#endif
    run<double,3,3>();

    return 0;
}
