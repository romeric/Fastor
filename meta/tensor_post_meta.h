#ifndef TENSOR_POST_META_H
#define TENSOR_POST_META_H

#include <tensor/Tensor.h>

namespace Fastor {


template<class T>
struct scalar_type_finder {
    using type = T;
};

template<template <class,class,size_t> class Expr, typename TLhs, typename TRhs, size_t DIMS>
struct scalar_type_finder<Expr<TLhs,TRhs,DIMS>> {
    using type = typename std::conditional<std::is_arithmetic<TLhs>::value, 
        typename scalar_type_finder<TRhs>::type, typename scalar_type_finder<TLhs>::type>::type;
};

template<template <class,size_t> class Expr, typename Nested, size_t DIMS>
struct scalar_type_finder<Expr<Nested,DIMS>> {
    using type = typename scalar_type_finder<Nested>::type;
};

template<typename T, size_t ... Rest>
struct scalar_type_finder<Tensor<T,Rest...>> {
    using type = T;
};




template<class T>
struct is_tensor {
    static constexpr bool value = false;
};

template<class T, size_t ...Rest>
struct is_tensor<Tensor<T,Rest...>> {
    static constexpr bool value = true;
};

template<class T>
struct is_abstracttensor {
    static constexpr bool value = false;
};

template<class T, size_t DIMS>
struct is_abstracttensor<AbstractTensor<T,DIMS>> {
    static constexpr bool value = true;
};

}

#endif // TENSOR_POST_META_H
