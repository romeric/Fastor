#ifndef TENSOR_PRE_META_H
#define TENSOR_PRE_META_H

#include "commons/commons.h"

namespace Fastor {

template<typename T, size_t ... Rest>
class Tensor;

//----------------------------------------------------------------------------------------------------------//
template<typename Expr, size_t DIMS>
struct TensorViewExpr;

template<typename Derived>
struct is_tensor_view {
    static constexpr bool value = false;
};
template<typename T, size_t DIMS, size_t M, size_t N, size_t ...Rest>
struct is_tensor_view<TensorViewExpr<Tensor<T,M,N,Rest...>,DIMS>> {
    static constexpr bool value = true;
};

template<typename Derived>
struct has_tensor_view {
    static constexpr bool value = is_tensor_view<Derived>::value ? true : false;
};
template<typename T, size_t DIMS, size_t M, size_t N, size_t ...Rest>
struct has_tensor_view<TensorViewExpr<Tensor<T,M,N,Rest...>,DIMS>> {
    static constexpr bool value = true;
};
template<template<typename,size_t> class UnaryExpr, typename Expr, size_t DIM>
struct has_tensor_view<UnaryExpr<Expr,DIM>> {
    static constexpr bool value = is_tensor_view<Expr>::value ? true : has_tensor_view<Expr>::value;
};
template<template<class,class,size_t> class BinaryExpr, typename TLhs, typename TRhs, size_t DIMS>
struct has_tensor_view<BinaryExpr<TLhs,TRhs,DIMS>> {
    static constexpr bool value = (std::is_arithmetic<TLhs>::value ? is_tensor_view<TRhs>::value : is_tensor_view<TLhs>::value) ? true :
        (std::is_arithmetic<TLhs>::value ? has_tensor_view<TRhs>::value : has_tensor_view<TLhs>::value);
};
//----------------------------------------------------------------------------------------------------------//


//----------------------------------------------------------------------------------------------------------//
template<typename Expr, typename Seq0, typename Seq1, size_t DIMS>
struct TensorFixedViewExpr2D;

template<typename Derived>
struct is_tensor_fixed_view_2d {
    static constexpr bool value = false;
};
template<typename T, size_t M, size_t N, typename Seq0, typename Seq1>
struct is_tensor_fixed_view_2d<TensorFixedViewExpr2D<Tensor<T,M,N>,Seq0,Seq1,2>> {
    static constexpr bool value = true;
};

template<typename Derived>
struct has_tensor_fixed_view_2d {
    static constexpr bool value = is_tensor_fixed_view_2d<Derived>::value ? true : false;
};
template<typename T, size_t M, size_t N, typename Seq0, typename Seq1>
struct has_tensor_fixed_view_2d<TensorFixedViewExpr2D<Tensor<T,M,N>,Seq0,Seq1,2>> {
    static constexpr bool value = true;
};
template<template<typename,size_t> class UnaryExpr, typename Expr, size_t DIM>
struct has_tensor_fixed_view_2d<UnaryExpr<Expr,DIM>> {
    static constexpr bool value = is_tensor_fixed_view_2d<Expr>::value ? true : has_tensor_view<Expr>::value;
};
template<template<class,class,size_t> class BinaryExpr, typename TLhs, typename TRhs, size_t DIMS>
struct has_tensor_fixed_view_2d<BinaryExpr<TLhs,TRhs,DIMS>> {
    static constexpr bool value = (std::is_arithmetic<TLhs>::value ? is_tensor_fixed_view_2d<TRhs>::value : is_tensor_fixed_view_2d<TLhs>::value) ? true :
        (std::is_arithmetic<TLhs>::value ? has_tensor_view<TRhs>::value : has_tensor_view<TLhs>::value);
};
//----------------------------------------------------------------------------------------------------------//


template<class T>
struct scalar_type_finder;

}

#endif // TENSOR_PRE_META_H