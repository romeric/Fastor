#include <Fastor/Fastor.h>

using namespace Fastor;

#define NITER 1000000UL


////////////////////////////////////////////////////////////////////////////////////////
// vector-tensor
template<typename T, size_t M, size_t N, 
    typename std::enable_if<M==3 && N==3,bool>::type=0>
inline Tensor<T,M,N> crossproduct_scalar(const Tensor<T,M> &a, const Tensor<T,M,N> &b) {
    constexpr T levi_civita[27] = { 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,
                                0., -1.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.};
    Tensor<T,M,N> out;
    constexpr size_t size = N;
    for (size_t i=0; i<N; ++i)
        for (size_t j=0; j<N; ++j)
            for (size_t k=0; k<N; ++k)
                for (size_t l=0; l<N; ++l)
                    out(i,j) += levi_civita[i*size*size+k*size+l]*a(k)*b(l,j);
    return out;
}

template<typename T, size_t M, size_t N, 
    typename std::enable_if<M==2 && N==2,bool>::type=0>
inline Tensor<T,M+1,N+1> crossproduct_scalar(const Tensor<T,M> &a, const Tensor<T,M,N> &b) {
    constexpr T levi_civita[27] = { 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,
                                0., -1.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.};
    Tensor<T,M+1,N+1> out;
    constexpr size_t size = N+1;
    for (size_t i=0; i<N; ++i)
        for (size_t j=0; j<N; ++j)
            for (size_t k=0; k<N; ++k)
                for (size_t l=0; l<N; ++l)
                    out(i,j) += levi_civita[i*size*size+k*size+l]*a(k)*b(l,j);

    return out;
}
// tensor-vector
template<typename T, size_t M, size_t N, 
    typename std::enable_if<M==3 && N==3,bool>::type=0>
inline Tensor<T,M,N> crossproduct_scalar(const Tensor<T,M,N> &a, const Tensor<T,M> &b) {
    constexpr T levi_civita[27] = { 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,
                                0., -1.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.};
    Tensor<T,M,N> out;
    constexpr size_t size = N;
    for (size_t i=0; i<N; ++i)
        for (size_t j=0; j<N; ++j)
            for (size_t k=0; k<N; ++k)
                for (size_t l=0; l<N; ++l)
                    out(i,j) += levi_civita[j*size*size+k*size+l]*a(i,k)*b(l);
    return out;
}

template<typename T, size_t M, size_t N, 
    typename std::enable_if<M==2 && N==2,bool>::type=0>
inline Tensor<T,M+1,N+1> crossproduct_scalar(const Tensor<T,M,N> &a, const Tensor<T,M> &b) {
    constexpr T levi_civita[27] = { 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,
                                0., -1.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.};
    Tensor<T,M+1,N+1> out;
    constexpr size_t size = N+1;
    for (size_t i=0; i<N; ++i)
        for (size_t j=0; j<N; ++j)
            for (size_t k=0; k<N; ++k)
                for (size_t l=0; l<N; ++l)
                    out(i,j) += levi_civita[j*size*size+k*size+l]*a(i,k)*b(l);

    return out;
}
////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////////////
template<typename T, size_t M, size_t N,
    typename std::enable_if<M==3 && N==3,bool>::type=0>
void iterate_over_scalar(const Tensor<T,M> &a, const Tensor<T,M,N> &b) {
    size_t iter = 0;
    Tensor<T,M,N> out;
    for (; iter<NITER; ++iter) {
        out = crossproduct_scalar(a,b);
        unused(a); unused(b); unused(out);
    }
}

template<typename T, size_t M, size_t N,
    typename std::enable_if<M==2 && N==2,bool>::type=0>
void iterate_over_scalar(const Tensor<T,M> &a, const Tensor<T,M,N> &b) {
    size_t iter = 0;
    Tensor<T,M+1,N+1> out;
    for (; iter<NITER; ++iter) {
        out = crossproduct_scalar(a,b);
        unused(a); unused(b); unused(out);
    }
}

template<typename T, size_t M, size_t N,
    typename std::enable_if<M==3 && N==3,bool>::type=0>
void iterate_over_scalar(const Tensor<T,M,N> &a, const Tensor<T,M> &b) {
    size_t iter = 0;
    Tensor<T,M,N> out;
    for (; iter<NITER; ++iter) {
        out = crossproduct_scalar(a,b);
        unused(a); unused(b); unused(out);
    }
}

template<typename T, size_t M, size_t N,
    typename std::enable_if<M==2 && N==2,bool>::type=0>
void iterate_over_scalar(const Tensor<T,M,N> &a, const Tensor<T,M> &b) {
    size_t iter = 0;
    Tensor<T,M+1,N+1> out;
    for (; iter<NITER; ++iter) {
        out = crossproduct_scalar(a,b);
        unused(a); unused(b); unused(out);
    }
}

//
template<typename T, size_t M, size_t N,
    typename std::enable_if<M==3 && N==3,bool>::type=0>
void iterate_over_fastor(const Tensor<T,M> &a, const Tensor<T,M,N> &b) {
    size_t iter = 0;
    Tensor<T,M,N>  out;
    for (; iter<NITER; ++iter) {
        out = cross(a,b);
        unused(a); unused(b); unused(out);
    }    
}

template<typename T, size_t M, size_t N,
    typename std::enable_if<M==2 && N==2,bool>::type=0>
void iterate_over_fastor(const Tensor<T,M> &a, const Tensor<T,M,N> &b) {
    size_t iter = 0;
    Tensor<T,M+1,N+1>  out;
    for (; iter<NITER; ++iter) {
        out = cross(a,b);
        unused(a); unused(b); unused(out);
    }    
}

template<typename T, size_t M, size_t N,
    typename std::enable_if<M==3 && N==3,bool>::type=0>
void iterate_over_fastor(const Tensor<T,M,N> &a, const Tensor<T,M> &b) {
    size_t iter = 0;
    Tensor<T,M,N>  out;
    for (; iter<NITER; ++iter) {
        out = cross(a,b);
        unused(a); unused(b); unused(out);
    }    
}

template<typename T, size_t M, size_t N,
    typename std::enable_if<M==2 && N==2,bool>::type=0>
void iterate_over_fastor(const Tensor<T,M,N> &a, const Tensor<T,M> &b) {
    size_t iter = 0;
    Tensor<T,M+1,N+1>  out;
    for (; iter<NITER; ++iter) {
        out = cross(a,b);
        unused(a); unused(b); unused(out);
    }    
}
////////////////////////////////////////////////////////////////////////////////////////


template<typename T, size_t M, size_t N>
void run() {

    Tensor<T,M,N> A;
    Tensor<T,M> b;
    A.random(); b.random();

    double time0, time1;
    std::tie(time0,std::ignore) = rtimeit(static_cast<void (*)(const Tensor<T,M,N>&, const Tensor<T,M>&)>(&iterate_over_scalar),A,b);
    std::tie(time1,std::ignore) = rtimeit(static_cast<void (*)(const Tensor<T,M,N>&, const Tensor<T,M>&)>(&iterate_over_fastor),A,b);
    print(time0,time1);
    print("\n");
    std::tie(time0,std::ignore) = rtimeit(static_cast<void (*)(const Tensor<T,M>&, const Tensor<T,M,N>&)>(&iterate_over_scalar),b,A);
    std::tie(time1,std::ignore) = rtimeit(static_cast<void (*)(const Tensor<T,M>&, const Tensor<T,M,N>&)>(&iterate_over_fastor),b,A);
    print(time0,time1);
    print("\n");
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
