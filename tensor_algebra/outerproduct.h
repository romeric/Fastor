#ifndef OUTERPRODUCT_H
#define OUTERPRODUCT_H

#include "tensor/Tensor.h"
#include "indicial.h"

namespace Fastor {

// outer products
//---------------------------------------------------------------------------------------------

#ifndef FASTOR_DONT_VECTORISE

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

    constexpr std::array<int,out_dim> maxes_out = {Rest0...,Rest1...};
    constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
    constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
    constexpr std::array<size_t,out_dim> products_out = nprods<Index<Rest0...,Rest1...>,typename std_ext::make_index_sequence<out_dim>::type>::values;

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

        _vec_a.set(*(a_data+index_a));
        V _vec_out = _vec_a*V(b_data+index_b);
        _vec_out.store(out_data+index_out);

#ifdef FASTOR_GCC
        // for benchmark gcc
        unused(_vec_out);
        unused(as);
#endif
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
    constexpr std::array<int,out_dim> maxes_out = {Rest0...,Rest1...};

    constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
    constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
    constexpr std::array<size_t,out_dim> products_out = nprods<Index<Rest0...,Rest1...>,typename std_ext::make_index_sequence<out_dim>::type>::values;


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
    constexpr std::array<int,out_dim> maxes_out = {Rest0...,Rest1...};

    constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
    constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
    constexpr std::array<size_t,out_dim> products_out = nprods<Index<Rest0...,Rest1...>,typename std_ext::make_index_sequence<out_dim>::type>::values;


    int as[out_dim];
    std::fill(as,as+out_dim,0);
    int it;

    constexpr int total = prod<Rest0...,Rest1...>::value;
    using V = SIMDVector<T,128>;
    V _vec_a;

    constexpr int stride = get_value<sizeof...(Rest1),Rest1...>::value;
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
//        unused(index_a); unused(index_b); unused(index_out);

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
    constexpr std::array<int,out_dim> maxes_out = {Rest0...,Rest1...};

    constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
    constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
    constexpr std::array<size_t,out_dim> products_out = nprods<Index<Rest0...,Rest1...>,typename std_ext::make_index_sequence<out_dim>::type>::values;


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

//        asm("#BEGIN");
        _vec_a.set(*(a_data+index_a));
        V _vec_out = _vec_a*V(b_data+index_b);
        _vec_out.store(out_data+index_out,true);

//        _mm256_store_ps(out_data+index_out,_mm256_mul_ps(_mm256_set1_ps(*(a_data+index_a)),_mm256_load_ps(b_data+index_b)));
//        asm("#END");
    }

    return out;
}


// Generic
template<template<typename,size_t...Rest0> class Tensor0,
         template<typename,size_t...Rest1> class Tensor1,
         typename T, size_t ... Rest0, size_t ... Rest1,
         typename std::enable_if<(std::is_same<T,double>::value || std::is_same<T,float>::value) &&
             get_value<sizeof...(Rest1),Rest1...>::value % 2 != 0 &&
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
    constexpr std::array<int,out_dim> maxes_out = {Rest0...,Rest1...};

    constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
    constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
    constexpr std::array<size_t,out_dim> products_out = nprods<Index<Rest0...,Rest1...>,typename std_ext::make_index_sequence<out_dim>::type>::values;

    int as[out_dim];
    std::fill(as,as+out_dim,0);
    int it, jt;

    while(true)
    {
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

        for(jt = out_dim-1 ; jt>=0 ; jt--)
        {
            if(++as[jt]<maxes_out[jt])
                break;
            else
                as[jt]=0;
        }
        if(jt<0)
            break;
    }

    return out;
}


template<template<typename,size_t...Rest0> class Tensor0,
         template<typename,size_t...Rest1> class Tensor1,
         typename T, size_t ... Rest0, size_t ... Rest1,
         typename std::enable_if<!std::is_same<T,double>::value && !std::is_same<T,float>::value,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest0...,Rest1...> outer(const Tensor0<T,Rest0...> &a, const Tensor1<T,Rest1...> &b) {

    Tensor<T,Rest0...,Rest1...> out;
    out.zeros();
    T *a_data = a.data();
    T *b_data = b.data();
    T *out_data = out.data();

    constexpr int a_dim = sizeof...(Rest0);
    constexpr int b_dim = sizeof...(Rest1);
    constexpr int out_dim = a_dim+b_dim;
    constexpr std::array<int,out_dim> maxes_out = {Rest0...,Rest1...};

    constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
    constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
    constexpr std::array<size_t,out_dim> products_out = nprods<Index<Rest0...,Rest1...>,typename std_ext::make_index_sequence<out_dim>::type>::values;

    int as[out_dim];
    std::fill(as,as+out_dim,0);
    int it, jt;

    while(true)
    {
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

        for(jt = out_dim-1 ; jt>=0 ; jt--)
        {
            if(++as[jt]<maxes_out[jt])
                break;
            else
                as[jt]=0;
        }
        if(jt<0)
            break;
    }

    return out;
}


#else

template<template<typename,size_t...Rest0> class Tensor0,
         template<typename,size_t...Rest1> class Tensor1,
         typename T, size_t ... Rest0, size_t ... Rest1>
FASTOR_INLINE Tensor<T,Rest0...,Rest1...> outer(const Tensor0<T,Rest0...> &a, const Tensor1<T,Rest1...> &b) {

    Tensor<T,Rest0...,Rest1...> out;
    out.zeros();
    T *a_data = a.data();
    T *b_data = b.data();
    T *out_data = out.data();

    constexpr int a_dim = sizeof...(Rest0);
    constexpr int b_dim = sizeof...(Rest1);
    constexpr int out_dim = a_dim+b_dim;
    constexpr std::array<int,out_dim> maxes_out = {Rest0...,Rest1...};

    constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
    constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
    constexpr std::array<size_t,out_dim> products_out = nprods<Index<Rest0...,Rest1...>,typename std_ext::make_index_sequence<out_dim>::type>::values;


    int as[out_dim];
    std::fill(as,as+out_dim,0);
    int it;

    using V = SIMDVector<T,sizeof(T)*8>;
    V _vec_a;

    constexpr int stride = 1;
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
        _vec_out.store(out_data+index_out,true);


        unused(_vec_out);
        unused(as);

    }

    return out;
}

#endif


template<typename T, size_t ... Rest0, size_t ... Rest1>
FASTOR_INLINE Tensor<T,Rest0...,Rest1...> dyadic(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) {
    return outer(a,b);
}

}

#endif // OUTERPRODUCT_H

