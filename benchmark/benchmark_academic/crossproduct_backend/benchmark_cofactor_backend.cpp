#include <Fastor.h>

using namespace Fastor;

#define NITER 1000000UL

template<typename T, size_t N>
void iterate_over_classical(const T *__restrict__ a, T *__restrict__ out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        T det = _det<T,N,N>(a);
        T *adj = static_cast<T*>(_mm_malloc(sizeof(T) * N*N, 32));
        _adjoint<T,N,N>(a,adj);
        T *trans = static_cast<T*>(_mm_malloc(sizeof(T) * N*N, 32));
        _transpose<T,N,N>(adj,trans);
        for (int i=0; i<N*N; ++i)
            out[i] = trans[i]/det;
        
        _mm_free(adj);
        _mm_free(trans);

        unused(a); unused(out);

        // further hack for gcc, seemingly  doesn't hurt performance of _crossproduct 
        out[1] += out[2];
    }
}

template<typename T, size_t N>
void iterate_over_fastor(const T *__restrict__ a, T *__restrict__ out) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        _cofactor<T,N,N>(a,out);
        unused(a); unused(out);

        // further hack for gcc, seemingly  doesn't hurt performance of _crossproduct 
        out[1] += out[2]; 
    }    
}


template<typename T, size_t M, size_t N>
void run() {

    T *a  = static_cast<T*>(_mm_malloc(sizeof(T) * M*N, 32));
    T *out = static_cast<T*>(_mm_malloc(sizeof(T) * (M)*(N), 32)); 
    std::iota(a,a+M*N,0);


    double time0, time1;
    std::tie(time0,std::ignore) = rtimeit(static_cast<void (*)(const T*, T*)>(&iterate_over_classical<T,N>),a,out);
    std::tie(time1,std::ignore) = rtimeit(static_cast<void (*)(const T*, T*)>(&iterate_over_fastor<T,N>),a,out);
    print(time0,time1);
    print("\n");

    _mm_free(a);
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