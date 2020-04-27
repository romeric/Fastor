#ifndef TENSOR_POST_META_H
#define TENSOR_POST_META_H

#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor_algebra/indicial.h"

namespace Fastor {


//--------------------------------------------------------------------------------------------------------------------//
template<class T>
struct scalar_type_finder {
    using type = T;
};

template<typename T, size_t ... Rest>
struct scalar_type_finder<Tensor<T,Rest...>> {
    using type = T;
};
// This specific specialisation is needed to avoid ambiguity for vectors
template<typename T, size_t N>
struct scalar_type_finder<Tensor<T,N>> {
    using type = T;
};
template<template <class,size_t> class UnaryExpr, typename Expr, size_t DIMS>
struct scalar_type_finder<UnaryExpr<Expr,DIMS>> {
    using type = typename scalar_type_finder<Expr>::type;
};
template<template <class,class,size_t> class Expr, typename TLhs, typename TRhs, size_t DIMS>
struct scalar_type_finder<Expr<TLhs,TRhs,DIMS>> {
    using type = typename std::conditional<std::is_arithmetic<TLhs>::value,
        typename scalar_type_finder<TRhs>::type, typename scalar_type_finder<TLhs>::type>::type;
};
template<template<typename,typename,typename,size_t> class TensorFixedViewExpr,
    typename Expr, typename Seq0, typename Seq1, size_t DIMS>
struct scalar_type_finder<TensorFixedViewExpr<Expr,Seq0,Seq1,DIMS>> {
    using type = typename scalar_type_finder<Expr>::type;
};
template<template<typename,size_t...> class TensorType, typename T, size_t ...Rest, typename ... Fseqs>
struct scalar_type_finder<TensorConstFixedViewExprnD<TensorType<T,Rest...>,Fseqs...>> {
    using type = T;
};
template<template<typename,size_t...> class TensorType, typename T, size_t ...Rest, typename ... Fseqs>
struct scalar_type_finder<TensorFixedViewExprnD<TensorType<T,Rest...>,Fseqs...>> {
    using type = T;
};

template<typename T, size_t ... Rest>
struct scalar_type_finder<TensorMap<T,Rest...>> {
    using type = T;
};
//--------------------------------------------------------------------------------------------------------------------//







//--------------------------------------------------------------------------------------------------------------------//
template<class X>
struct tensor_type_finder {
    using type = Tensor<X>;
};

template<typename T, size_t ... Rest>
struct tensor_type_finder<Tensor<T,Rest...>> {
    using type = Tensor<T,Rest...>;
};
// This specific specialisation is needed to avoid ambiguity for vectors
template<typename T, size_t N>
struct tensor_type_finder<Tensor<T,N>> {
    using type = Tensor<T,N>;
};
template<template<typename,size_t> class UnaryExpr, typename Expr, size_t DIM>
struct tensor_type_finder<UnaryExpr<Expr,DIM>> {
    using type = typename tensor_type_finder<Expr>::type;
};
template<template<class,class,size_t> class BinaryExpr, typename TLhs, typename TRhs, size_t DIMS>
struct tensor_type_finder<BinaryExpr<TLhs,TRhs,DIMS>> {
    // using type = typename tensor_type_finder<TLhs>::type;
    using type = typename std::conditional<std::is_arithmetic<TLhs>::value,
        typename tensor_type_finder<TRhs>::type, typename tensor_type_finder<TLhs>::type>::type;
};
template<template<typename,typename,typename,size_t> class TensorFixedViewExpr,
    typename Expr, typename Seq0, typename Seq1, size_t DIMS>
struct tensor_type_finder<TensorFixedViewExpr<Expr,Seq0,Seq1,DIMS>> {
    using type = typename tensor_type_finder<Expr>::type;
};
template<template<typename,size_t...> class TensorType, typename T, size_t ...Rest, typename ... Fseqs>
struct tensor_type_finder<TensorConstFixedViewExprnD<TensorType<T,Rest...>,Fseqs...>> {
    using type = TensorType<T,Rest...>;
};
template<template<typename,size_t...> class TensorType, typename T, size_t ...Rest, typename ... Fseqs>
struct tensor_type_finder<TensorFixedViewExprnD<TensorType<T,Rest...>,Fseqs...>> {
    using type = TensorType<T,Rest...>;
};

template<typename T, size_t ... Rest>
struct tensor_type_finder<TensorMap<T,Rest...>> {
    using type = Tensor<T,Rest...>;
};
//--------------------------------------------------------------------------------------------------------------------//





//--------------------------------------------------------------------------------------------------------------------//
template<class T>
struct is_tensor {
    static constexpr bool value = false;
};
template<class T, size_t ...Rest>
struct is_tensor<Tensor<T,Rest...>> {
    static constexpr bool value = true;
};
template<typename T>
constexpr bool is_tensor_v = is_tensor<T>::value;
//--------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------------------------------------//
template<class T>
struct is_abstracttensor {
    static constexpr bool value = false;
};
template<class T, size_t DIMS>
struct is_abstracttensor<AbstractTensor<T,DIMS>> {
    static constexpr bool value = true;
};
template<typename T>
constexpr bool is_abstracttensor_v = is_abstracttensor<T>::value;
//--------------------------------------------------------------------------------------------------------------------//





//--------------------------------------------------------------------------------------------------------------------//
// Do not generalise this, as it leads to all kinds of problems
// with binary operator expression involving std::arithmetics
template <class X, class Y, class ... Z>
struct concat_tensor;

template<template<typename,size_t...> class Derived0,
    template<typename,size_t...> class Derived1,
    typename T, size_t ... Rest0, size_t ... Rest1>
struct concat_tensor<Derived0<T,Rest0...>,Derived1<T,Rest1...>> {
    using type = Tensor<T,Rest0...,Rest1...>;
};

template<template<typename,size_t...> class Derived0,
    template<typename,size_t...> class Derived1,
    template<typename,size_t...> class Derived2,
    typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2>
struct concat_tensor<Derived0<T,Rest0...>,Derived1<T,Rest1...>,Derived2<T,Rest2...>> {
    using type = Tensor<T,Rest0...,Rest1...,Rest2...>;
};

template<template<typename,size_t...> class Derived0,
    template<typename,size_t...> class Derived1,
    template<typename,size_t...> class Derived2,
    template<typename,size_t...> class Derived3,
    typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3>
struct concat_tensor<Derived0<T,Rest0...>,Derived1<T,Rest1...>,Derived2<T,Rest2...>,Derived3<T,Rest3...>> {
    using type = Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...>;
};

template <class X, class Y, class ... Z>
using concatenated_tensor_t = typename concat_tensor<X,Y,Z...>::type;
//--------------------------------------------------------------------------------------------------------------------//



//--------------------------------------------------------------------------------------------------------------------//
// Return dimensions of tensor as a std::array and Index<Rest...>
template<class X>
struct get_tensor_dimensions;

template<typename T, size_t ... Rest>
struct get_tensor_dimensions<Tensor<T,Rest...>> {
    static constexpr std::array<size_t,sizeof...(Rest)> dims = {Rest...};
    static constexpr std::array<int,sizeof...(Rest)> dims_int = {Rest...};
    using tensor_to_index = Index<Rest...>;
};

template<typename T, size_t ... Rest>
constexpr std::array<size_t,sizeof...(Rest)> get_tensor_dimensions<Tensor<T,Rest...>>::dims;
template<typename T, size_t ... Rest>
constexpr std::array<int,sizeof...(Rest)> get_tensor_dimensions<Tensor<T,Rest...>>::dims_int;
//--------------------------------------------------------------------------------------------------------------------//





//--------------------------------------------------------------------------------------------------------------------//
// Extract a matrix from a high order tensor
// this is used in places like determinant/inverse of high order tensors
// where the last square matrix (last two dimensions) is needed. If Seq is
// is the same size as dimensions of tensor, this could also be used as
// generic tensor dimension extractor
template<class Tens, class Seq>
struct LastMatrixExtracter;

template<typename T, size_t ... Rest, size_t ... ss>
struct LastMatrixExtracter<Tensor<T,Rest...>,std_ext::index_sequence<ss...>>
{
    static constexpr std::array<size_t,sizeof...(Rest)> dims = {Rest...};
    static constexpr std::array<size_t,sizeof...(ss)> values = {dims[ss]...};
    static constexpr size_t remaining_product = prod<dims[ss]...>::value;
    // static constexpr size_t last_value = dims[sizeof...(Rest)-1];
    using type = Tensor<T,dims[ss]...>;
};

template<typename T, size_t ... Rest, size_t ... ss>
constexpr std::array<size_t,sizeof...(ss)>
LastMatrixExtracter<Tensor<T,Rest...>,std_ext::index_sequence<ss...>>::values;
//--------------------------------------------------------------------------------------------------------------------//

}

#endif // TENSOR_POST_META_H
