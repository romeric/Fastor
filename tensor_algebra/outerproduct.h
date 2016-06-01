#ifndef OUTERPRODUCT_H
#define OUTERPRODUCT_H

#include "tensor/Tensor.h"
#include "indicial.h"

namespace Fastor {

// outer products
//---------------------------------------------------------------------------------------------
// SSE optimisable overload
template<template<typename,size_t...Rest0> class Tensor0,
         template<typename,size_t...Rest1> class Tensor1,
         typename T, size_t ... Rest0, size_t ... Rest1,
         typename std::enable_if<std::is_same<T,double>::value &&
                                 get_value<sizeof...(Rest1),Rest1...>::value % 2 == 0 &&
                                 get_value<sizeof...(Rest1),Rest1...>::value % 4 != 0,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest0...,Rest1...> outer(const Tensor0<T,Rest0...> &a, const Tensor1<T,Rest1...> &b) {
    Tensor<T,Rest0...,Rest1...> out;
    out.zeros();
    T *a_data = a.data();
    T *b_data = b.data();
    T *out_data = out.data();

    constexpr int a_dim = sizeof...(Rest0);
    constexpr int b_dim = sizeof...(Rest1);
    constexpr int out_dim = a_dim+b_dim;
    constexpr std::array<int,a_dim> maxes_a = {Rest0...};
    constexpr std::array<int,b_dim> maxes_b = {Rest1...};
    constexpr std::array<int,out_dim> maxes_out = {Rest0...,Rest1...};

    std::array<int,a_dim> products_a; products_a[0]=0;
    for (int j=a_dim-1; j>0; --j) {
        int num = maxes_a[a_dim-1];
        for (int k=0; k<j-1; ++k) {
            num *= maxes_a[a_dim-1-k-1];
        }
        products_a[j] = num;
    }
    std::array<int,b_dim> products_b; products_b[0]=0;
    for (int j=b_dim-1; j>0; --j) {
        int num = maxes_b[b_dim-1];
        for (int k=0; k<j-1; ++k) {
            num *= maxes_b[b_dim-1-k-1];
        }
        products_b[j] = num;
    }
    std::array<int,out_dim> products_out; products_out[0]=0;
    for (int j=out_dim-1; j>0; --j) {
        int num = maxes_out[out_dim-1];
        for (int k=0; k<j-1; ++k) {
            num *= maxes_out[out_dim-1-k-1];
        }
        products_out[j] = num;
    }

    std::reverse(products_a.begin(),products_a.end());
    std::reverse(products_b.begin(),products_b.end());
    std::reverse(products_out.begin(),products_out.end());

    int as[out_dim];
    std::fill(as,as+out_dim,0);
    int it;

    using V = SIMDVector<T,128>;
    V _vec_a;

    constexpr int stride = get_value<sizeof...(Rest1),Rest1...>::value;
    constexpr int total = prod<Rest0...,Rest1...>::value;
    for (int i = 0; i < total; i+=stride) {
        int remaining = total;
        for (int n = 0; n < out_dim; ++n) {
            remaining /= maxes_out[n];
            as[n] = ( i / remaining ) % maxes_out[n];
        }

        int index_a = as[a_dim-1];
        for(it = 0; it< a_dim; it++) {
            index_a += products_a[it]*as[it];
        }
        int index_b = as[out_dim-1];
        for(it = a_dim; it< out_dim; it++) {
            index_b += products_b[it-a_dim]*as[it];
        }
        int index_out = as[out_dim-1];
        for(it = 0; it< out_dim; it++) {
            index_out += products_out[it]*as[it];
        }

//        out_data[index_out] = a_data[index_a]*b_data[index_b];
        _vec_a.set(*(a_data+index_a));
        V _vec_out = _vec_a*V(b_data+index_b);
        _vec_out.store(out_data+index_out);
    }

    return out;
}


template<template<typename,size_t...Rest0> class Tensor0,
         template<typename,size_t...Rest1> class Tensor1,
         typename T, size_t ... Rest0, size_t ... Rest1,
         typename std::enable_if<std::is_same<T,double>::value &&
                                 get_value<sizeof...(Rest1),Rest1...>::value % 2 == 0 &&
                                 get_value<sizeof...(Rest1),Rest1...>::value % 4 == 0,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest0...,Rest1...> outer(const Tensor0<T,Rest0...> &a, const Tensor1<T,Rest1...> &b) {

    Tensor<T,Rest0...,Rest1...> out;
    out.zeros();
    T *a_data = a.data();
    T *b_data = b.data();
    T *out_data = out.data();

    constexpr int a_dim = sizeof...(Rest0);
    constexpr int b_dim = sizeof...(Rest1);
    constexpr int out_dim = a_dim+b_dim;
    constexpr std::array<int,a_dim> maxes_a = {Rest0...};
    constexpr std::array<int,b_dim> maxes_b = {Rest1...};
    constexpr std::array<int,out_dim> maxes_out = {Rest0...,Rest1...};

    std::array<int,a_dim> products_a; products_a[0]=0;
    for (int j=a_dim-1; j>0; --j) {
        int num = maxes_a[a_dim-1];
        for (int k=0; k<j-1; ++k) {
            num *= maxes_a[a_dim-1-k-1];
        }
        products_a[j] = num;
    }
    std::array<int,b_dim> products_b; products_b[0]=0;
    for (int j=b_dim-1; j>0; --j) {
        int num = maxes_b[b_dim-1];
        for (int k=0; k<j-1; ++k) {
            num *= maxes_b[b_dim-1-k-1];
        }
        products_b[j] = num;
    }
    std::array<int,out_dim> products_out; products_out[0]=0;
    for (int j=out_dim-1; j>0; --j) {
        int num = maxes_out[out_dim-1];
        for (int k=0; k<j-1; ++k) {
            num *= maxes_out[out_dim-1-k-1];
        }
        products_out[j] = num;
    }

    std::reverse(products_a.begin(),products_a.end());
    std::reverse(products_b.begin(),products_b.end());
    std::reverse(products_out.begin(),products_out.end());

    int as[out_dim];
    std::fill(as,as+out_dim,0);
    int it;

    using V = SIMDVector<T,256>;
    V _vec_a;

    constexpr int stride = get_value<sizeof...(Rest1),Rest1...>::value;
    constexpr int total = prod<Rest0...,Rest1...>::value;
    for (int i = 0; i < total; i+=stride) {
        int remaining = total;
        for (int n = 0; n < out_dim; ++n) {
            remaining /= maxes_out[n];
            as[n] = ( i / remaining ) % maxes_out[n];
        }

        int index_a = as[a_dim-1];
        for(it = 0; it< a_dim; it++) {
            index_a += products_a[it]*as[it];
        }
        int index_b = as[out_dim-1];
        for(it = a_dim; it< out_dim; it++) {
            index_b += products_b[it-a_dim]*as[it];
        }
        int index_out = as[out_dim-1];
        for(it = 0; it< out_dim; it++) {
            index_out += products_out[it]*as[it];
        }

        _vec_a.set(*(a_data+index_a));
        V _vec_out = _vec_a*V(b_data+index_b);
        _vec_out.store(out_data+index_out);
    }

    return out;
}


template<template<typename,size_t...Rest0> class Tensor0,
         template<typename,size_t...Rest1> class Tensor1,
         typename T, size_t ... Rest0, size_t ... Rest1,
         typename std::enable_if<std::is_same<T,float>::value &&
                                 get_value<sizeof...(Rest1),Rest1...>::value % 2 == 0 &&
                                 get_value<sizeof...(Rest1),Rest1...>::value % 4 == 0 &&
                                 get_value<sizeof...(Rest1),Rest1...>::value % 8 != 0,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest0...,Rest1...> outer(const Tensor0<T,Rest0...> &a, const Tensor1<T,Rest1...> &b) {

    Tensor<T,Rest0...,Rest1...> out;
    out.zeros();
    T *a_data = a.data();
    T *b_data = b.data();
    T *out_data = out.data();

    constexpr int a_dim = sizeof...(Rest0);
    constexpr int b_dim = sizeof...(Rest1);
    constexpr int out_dim = a_dim+b_dim;
    constexpr std::array<int,a_dim> maxes_a = {Rest0...};
    constexpr std::array<int,b_dim> maxes_b = {Rest1...};
    constexpr std::array<int,out_dim> maxes_out = {Rest0...,Rest1...};

    std::array<int,a_dim> products_a; products_a[0]=0;
    for (int j=a_dim-1; j>0; --j) {
        int num = maxes_a[a_dim-1];
        for (int k=0; k<j-1; ++k) {
            num *= maxes_a[a_dim-1-k-1];
        }
        products_a[j] = num;
    }
    std::array<int,b_dim> products_b; products_b[0]=0;
    for (int j=b_dim-1; j>0; --j) {
        int num = maxes_b[b_dim-1];
        for (int k=0; k<j-1; ++k) {
            num *= maxes_b[b_dim-1-k-1];
        }
        products_b[j] = num;
    }
    std::array<int,out_dim> products_out; products_out[0]=0;
    for (int j=out_dim-1; j>0; --j) {
        int num = maxes_out[out_dim-1];
        for (int k=0; k<j-1; ++k) {
            num *= maxes_out[out_dim-1-k-1];
        }
        products_out[j] = num;
    }

    std::reverse(products_a.begin(),products_a.end());
    std::reverse(products_b.begin(),products_b.end());
    std::reverse(products_out.begin(),products_out.end());

    int as[out_dim];
    std::fill(as,as+out_dim,0);
    int it;

    using V = SIMDVector<T,128>;
    V _vec_a;

//    constexpr int stride = 1;
    constexpr int stride = get_value<sizeof...(Rest1),Rest1...>::value;
    constexpr int total = prod<Rest0...,Rest1...>::value;
    for (int i = 0; i < total; i+=stride) {
        int remaining = total;
        for (int n = 0; n < out_dim; ++n) {
            remaining /= maxes_out[n];
            as[n] = ( i / remaining ) % maxes_out[n];
//            as[n] = 0;
        }

        int index_a = as[a_dim-1];
        for(it = 0; it< a_dim; it++) {
            index_a += products_a[it]*as[it];
        }
        int index_b = as[out_dim-1];
        for(it = a_dim; it< out_dim; it++) {
            index_b += products_b[it-a_dim]*as[it];
        }
        int index_out = as[out_dim-1];
        for(it = 0; it< out_dim; it++) {
            index_out += products_out[it]*as[it];
        }

//        out_data[index_out] = a_data[index_a]*b_data[index_b];

        _vec_a.set(*(a_data+index_a));
        V _vec_out = _vec_a*V(b_data+index_b);
        _vec_out.store(out_data+index_out);

//        _mm_store_ps(out_data+index_out,_mm_set1_ps(0.f));
//        _mm_store_ps(out_data+index_out,_mm_mul_ps(_mm_set1_ps(*(a_data+index_a)),_mm_load_ps(b_data+index_b)));
    }

//    int jt;
//    while(true)
//    {
//        int index_a = as[a_dim-1];
//        for(it = 0; it< a_dim; it++) {
//            index_a += products_a[it]*as[it];
//        }
//        int index_b = as[out_dim-1];
//        for(it = a_dim; it< out_dim; it++) {
//            index_b += products_b[it-a_dim]*as[it];
//        }
//        int index_out = as[out_dim-1];
//        for(it = 0; it< out_dim; it++) {
//            index_out += products_out[it]*as[it];
//        }
////        out_data[index_out] = a_data[index_a]*b_data[index_b];


//        for(jt = out_dim-1 ; jt>=0 ; jt--)
//        {
//            if(++as[jt]<maxes_out[jt])
//                break;
//            else
//                as[jt]=0;
//        }
//        if(jt<0)
//            break;
//    }

    return out;
}


template<template<typename,size_t...Rest0> class Tensor0,
         template<typename,size_t...Rest1> class Tensor1,
         typename T, size_t ... Rest0, size_t ... Rest1,
         typename std::enable_if<std::is_same<T,float>::value &&
                                 get_value<sizeof...(Rest1),Rest1...>::value % 2 == 0 &&
                                 get_value<sizeof...(Rest1),Rest1...>::value % 4 == 0 &&
                                 get_value<sizeof...(Rest1),Rest1...>::value % 8 == 0,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest0...,Rest1...> outer(const Tensor0<T,Rest0...> &a, const Tensor1<T,Rest1...> &b) {

    Tensor<T,Rest0...,Rest1...> out;
    out.zeros();
    T *a_data = a.data();
    T *b_data = b.data();
    T *out_data = out.data();

    constexpr int a_dim = sizeof...(Rest0);
    constexpr int b_dim = sizeof...(Rest1);
    constexpr int out_dim = a_dim+b_dim;
    constexpr std::array<int,a_dim> maxes_a = {Rest0...};
    constexpr std::array<int,b_dim> maxes_b = {Rest1...};
    constexpr std::array<int,out_dim> maxes_out = {Rest0...,Rest1...};

    std::array<int,a_dim> products_a; products_a[0]=0;
    for (int j=a_dim-1; j>0; --j) {
        int num = maxes_a[a_dim-1];
        for (int k=0; k<j-1; ++k) {
            num *= maxes_a[a_dim-1-k-1];
        }
        products_a[j] = num;
    }
    std::array<int,b_dim> products_b; products_b[0]=0;
    for (int j=b_dim-1; j>0; --j) {
        int num = maxes_b[b_dim-1];
        for (int k=0; k<j-1; ++k) {
            num *= maxes_b[b_dim-1-k-1];
        }
        products_b[j] = num;
    }
    std::array<int,out_dim> products_out; products_out[0]=0;
    for (int j=out_dim-1; j>0; --j) {
        int num = maxes_out[out_dim-1];
        for (int k=0; k<j-1; ++k) {
            num *= maxes_out[out_dim-1-k-1];
        }
        products_out[j] = num;
    }

    std::reverse(products_a.begin(),products_a.end());
    std::reverse(products_b.begin(),products_b.end());
    std::reverse(products_out.begin(),products_out.end());

    int as[out_dim];
    std::fill(as,as+out_dim,0);
    int it;

    using V = SIMDVector<T,256>;
    V _vec_a;

    constexpr int stride = get_value<sizeof...(Rest1),Rest1...>::value;
    constexpr int total = prod<Rest0...,Rest1...>::value;
    for (int i = 0; i < total; i+=stride) {
        int remaining = total;
        for (int n = 0; n < out_dim; ++n) {
            remaining /= maxes_out[n];
            as[n] = ( i / remaining ) % maxes_out[n];
        }

        int index_a = as[a_dim-1];
        for(it = 0; it< a_dim; it++) {
            index_a += products_a[it]*as[it];
        }
        int index_b = as[out_dim-1];
        for(it = a_dim; it< out_dim; it++) {
            index_b += products_b[it-a_dim]*as[it];
        }
        int index_out = as[out_dim-1];
        for(it = 0; it< out_dim; it++) {
            index_out += products_out[it]*as[it];
        }

        _vec_a.set(*(a_data+index_a));
        V _vec_out = _vec_a*V(b_data+index_b);
        _vec_out.store(out_data+index_out);
    }

    return out;
}


template<template<typename,size_t...Rest0> class Tensor0,
         template<typename,size_t...Rest1> class Tensor1,
         typename T, size_t ... Rest0, size_t ... Rest1,
         typename std::enable_if<(std::is_same<T,float>::value && get_value<sizeof...(Rest1),Rest1...>::value % 2 == 0) &&
                                 get_value<sizeof...(Rest1),Rest1...>::value % 4 != 0 &&
                                 get_value<sizeof...(Rest1),Rest1...>::value % 8 != 0,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest0...,Rest1...> outer(const Tensor0<T,Rest0...> &a, const Tensor1<T,Rest1...> &b) {
    Tensor<T,Rest0...,Rest1...> out;
    out.zeros();
    T *a_data = a.data();
    T *b_data = b.data();
    T *out_data = out.data();

    constexpr int a_dim = sizeof...(Rest0);
    constexpr int b_dim = sizeof...(Rest1);
    constexpr int out_dim = a_dim+b_dim;
    constexpr std::array<int,a_dim> maxes_a = {Rest0...};
    constexpr std::array<int,b_dim> maxes_b = {Rest1...};
    constexpr std::array<int,out_dim> maxes_out = {Rest0...,Rest1...};

    std::array<int,a_dim> products_a; products_a[0]=0;
    for (int j=a_dim-1; j>0; --j) {
        int num = maxes_a[a_dim-1];
        for (int k=0; k<j-1; ++k) {
            num *= maxes_a[a_dim-1-k-1];
        }
        products_a[j] = num;
    }
    std::array<int,b_dim> products_b; products_b[0]=0;
    for (int j=b_dim-1; j>0; --j) {
        int num = maxes_b[b_dim-1];
        for (int k=0; k<j-1; ++k) {
            num *= maxes_b[b_dim-1-k-1];
        }
        products_b[j] = num;
    }
    std::array<int,out_dim> products_out; products_out[0]=0;
    for (int j=out_dim-1; j>0; --j) {
        int num = maxes_out[out_dim-1];
        for (int k=0; k<j-1; ++k) {
            num *= maxes_out[out_dim-1-k-1];
        }
        products_out[j] = num;
    }

    std::reverse(products_a.begin(),products_a.end());
    std::reverse(products_b.begin(),products_b.end());
    std::reverse(products_out.begin(),products_out.end());

    int as[out_dim];
    std::fill(as,as+out_dim,0);
    int it;

    constexpr int total = prod<Rest0...,Rest1...>::value;
    for (int i = 0; i < total; ++i) {
        int remaining = total;
        for (int n = 0; n < out_dim; ++n) {
            remaining /= maxes_out[n];
            as[n] = ( i / remaining ) % maxes_out[n];
        }

        int index_a = as[a_dim-1];
        for(it = 0; it< a_dim; it++) {
            index_a += products_a[it]*as[it];
        }
        int index_b = as[out_dim-1];
        for(it = a_dim; it< out_dim; it++) {
            index_b += products_b[it-a_dim]*as[it];
        }
        int index_out = as[out_dim-1];
        for(it = 0; it< out_dim; it++) {
            index_out += products_out[it]*as[it];
        }

        out_data[index_out] = a_data[index_a]*b_data[index_b];
    }

    return out;
}
























//    constexpr int total = prod<Rest0...,Rest1...>::value;
//    for (int i = 0; i < total; ++i) {
//        int remaining = total;
//        for (int n = 0; n < out_dim; ++n) {
//            remaining /= maxes_out[n];
//            as[n] = ( i / remaining ) % maxes_out[n];
//        }

//        int index_a = as[a_dim-1];
//        for(it = 0; it< a_dim; it++) {
//            index_a += products_a[it]*as[it];
//        }
//        int index_b = as[out_dim-1];
//        for(it = a_dim; it< out_dim; it++) {
//            index_b += products_b[it-a_dim]*as[it];
//        }
//        int index_out = as[out_dim-1];
//        for(it = 0; it< out_dim; it++) {
//            index_out += products_out[it]*as[it];
//        }

//        out_data[index_out] = a_data[index_a]*b_data[index_b];

//        for (int n = 0; n < out_dim; ++n) {
//            remaining /= maxes_out[n];
//            std::cout << as[n] << " ";
//        }
//        print("");
//    }

}

#endif // OUTERPRODUCT_H

