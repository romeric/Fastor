#ifndef TENSOR_POST_META_H
#define TENSOR_POST_META_H

#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor_algebra/indicial.h"

namespace Fastor {


/* Classify/specialise Tensor<primitive> as primitive if needed.
  This specialisation hurts the performance of some specialised
  kernels like matmul/norm/LU/inv that unroll aggressively unless
  Tensor<T> is specialised to wrap T only
*/
//--------------------------------------------------------------------------------------------------------------------//
// template<typename T>
// struct is_primitive<Tensor<T>> {
//     static constexpr bool value = is_primitive_v_<T> ? true : false;
// };
//--------------------------------------------------------------------------------------------------------------------//


/* Find the underlying scalar type of an expression */
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
template<typename T, size_t ... Rest>
struct scalar_type_finder<TensorMap<T,Rest...>> {
    using type = T;
};
// This specific specialisation is needed to avoid ambiguity for vectors
template<typename T, size_t N>
struct scalar_type_finder<TensorMap<T,N>> {
    using type = T;
};

template<template <class,size_t> class UnaryExpr, typename Expr, size_t DIMS>
struct scalar_type_finder<UnaryExpr<Expr,DIMS>> {
    using type = typename scalar_type_finder<Expr>::type;
};
template<template <class,class,size_t> class Expr, typename TLhs, typename TRhs, size_t DIMS>
struct scalar_type_finder<Expr<TLhs,TRhs,DIMS>> {
    using type = typename std::conditional<is_primitive_v_<TLhs>,
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
//--------------------------------------------------------------------------------------------------------------------//


/* Find the underlying tensor type of an expression */
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
template<typename T, size_t ... Rest>
struct tensor_type_finder<TensorMap<T,Rest...>> {
    using type = Tensor<T,Rest...>;
};
// This specific specialisation is needed to avoid ambiguity for vectors
template<typename T, size_t N>
struct tensor_type_finder<TensorMap<T,N>> {
    using type = Tensor<T,N>;
};

template<template<typename,size_t> class UnaryExpr, typename Expr, size_t DIM>
struct tensor_type_finder<UnaryExpr<Expr,DIM>> {
    using type = typename tensor_type_finder<Expr>::type;
};
template<template<class,class,size_t> class BinaryExpr, typename TLhs, typename TRhs, size_t DIMS>
struct tensor_type_finder<BinaryExpr<TLhs,TRhs,DIMS>> {
    using type = typename std::conditional<is_primitive_v_<TLhs>,
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
//--------------------------------------------------------------------------------------------------------------------//


/* Is an expression a tensor */
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


/* Is an expression a abstract tensor */
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


/* Convert a tensor to a bool tensor */
//--------------------------------------------------------------------------------------------------------------------//
template <class Tens>
struct to_bool_tensor;

template <typename T, size_t ... Rest>
struct to_bool_tensor<Tensor<T,Rest...>> {
    using type = Tensor<bool,Rest...>;
};

template <class Tens>
using to_bool_tensor_t = typename to_bool_tensor<Tens>::type;
//--------------------------------------------------------------------------------------------------------------------//


/* Concatenate two tensor and make a new tensor type */
//--------------------------------------------------------------------------------------------------------------------//
// Do not generalise this, as it leads to all kinds of problems
// with binary operator expression involving std::arithmetic
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


/* Extract the tensor dimension(s) */
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

// Conditional get tensor dimension
// Return tensor dimension if Idx is within range else return Dim
template<size_t Idx, size_t Dim, class X>
struct if_get_tensor_dimension;

template<size_t Idx, size_t Dim, typename T, size_t ... Rest>
struct if_get_tensor_dimension<Idx,Dim,Tensor<T,Rest...>> {
   static constexpr size_t value = (Idx < sizeof...(Rest)) ? get_value<Idx+1,Rest...>::value : 1;
};

template<size_t Idx, size_t Dim, class X>
static constexpr size_t if_get_tensor_dimension_v = if_get_tensor_dimension<Idx,Dim,X>::value;

// Gives one if Idx is outside range
template<size_t Idx, class X>
static constexpr size_t get_tensor_dimension_v = if_get_tensor_dimension<Idx,1,X>::value;
//--------------------------------------------------------------------------------------------------------------------//


/* Find if a tensor is uniform */
//--------------------------------------------------------------------------------------------------------------------//
template<class T>
struct is_tensor_uniform;

template<typename T, size_t ... Rest>
struct is_tensor_uniform<Tensor<T,Rest...>> {
    static constexpr bool value = no_of_unique<Rest...>::value == 1 ? true : false;
};

// helper function
template<class T>
static constexpr bool is_tensor_uniform_v = is_tensor_uniform<T>::value;
//--------------------------------------------------------------------------------------------------------------------//


/* Extract a matrix from a high order tensor */
//--------------------------------------------------------------------------------------------------------------------//
// This is used in functions like determinant/inverse of high order tensors
// where the last square matrix (last two dimensions) is needed. If Seq is
// is the same size as dimensions of tensor, this could also be used as
// generic tensor dimension extractor
template<class Tens, class Seq>
struct last_matrix_extracter;

template<typename T, size_t ... Rest, size_t ... ss>
struct last_matrix_extracter<Tensor<T,Rest...>,std_ext::index_sequence<ss...>>
{
    static constexpr std::array<size_t,sizeof...(Rest)> dims = {Rest...};
    static constexpr std::array<size_t,sizeof...(ss)> values = {dims[ss]...};
    static constexpr size_t remaining_product = pack_prod<dims[ss]...>::value;
    using type = Tensor<T,dims[ss]...>;
};

template<typename T, size_t ... Rest, size_t ... ss>
constexpr std::array<size_t,sizeof...(ss)>
last_matrix_extracter<Tensor<T,Rest...>,std_ext::index_sequence<ss...>>::values;
//--------------------------------------------------------------------------------------------------------------------//


//-----------------------------------------------------------------------------------------------------------//
template <typename T, typename Idx> struct index_to_tensor;
template <typename T, size_t ...Idx> struct index_to_tensor <T, Index<Idx...>> { using type = Tensor<T,Idx...>; };

// helper
template <typename T, typename Idx>
using index_to_tensor_t = typename index_to_tensor<T,Idx>::type;

template <typename T, typename Idx> struct index_to_tensor_map;
template <typename T, size_t ...Idx> struct index_to_tensor_map <T, Index<Idx...>> { using type = TensorMap<T,Idx...>; };

// helper
template <typename T, typename Idx>
using index_to_tensor_map_t = typename index_to_tensor_map<T,Idx>::type;

//-----------------------------------------------------------------------------------------------------------//


}

#endif // TENSOR_POST_META_H
