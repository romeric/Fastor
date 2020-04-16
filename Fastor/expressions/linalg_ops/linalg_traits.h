#ifndef LINALG_TRAITS_H
#define LINALG_TRAITS_H


#include "Fastor/expressions/linalg_ops/binary_matmul_op.h"
#include <type_traits>


namespace Fastor {


// Is a binary matmul expression
//----------------------------------------------------------------------------------------------------------//
template<typename Derived>
struct is_binary_matmul_op {
    static constexpr bool value = false;
};
template<typename T, size_t ... Rest0, size_t ... Rest1, size_t DIM>
struct is_binary_matmul_op<BinaryMatMulOp<Tensor<T,Rest0...>,Tensor<T,Rest1...>,DIM>> {
    static constexpr bool value = true;
};

template<typename Derived>
struct has_binary_matmul_op {
    static constexpr bool value = is_binary_matmul_op<Derived>::value ? true : false;
};
template<typename T, size_t ... Rest0, size_t ... Rest1, size_t DIM>
struct has_binary_matmul_op<BinaryMatMulOp<Tensor<T,Rest0...>,Tensor<T,Rest1...>,DIM>> {
    static constexpr bool value = true;
};
template<template<typename,size_t> class UnaryExpr, typename Expr, size_t DIM>
struct has_binary_matmul_op<UnaryExpr<Expr,DIM>> {
    static constexpr bool value = has_binary_matmul_op<Expr>::value;
};
template<template<class,class,size_t> class BinaryExpr, typename TLhs, typename TRhs, size_t DIMS>
struct has_binary_matmul_op<BinaryExpr<TLhs,TRhs,DIMS>> {
    static constexpr bool value = has_binary_matmul_op<TRhs>::value || has_binary_matmul_op<TLhs>::value;
};

// helper
template<typename Derived>
static constexpr bool is_binary_matmul_op_v = is_binary_matmul_op<Derived>::value;
template<typename Derived>
static constexpr bool has_binary_matmul_op_v = has_binary_matmul_op<Derived>::value;
//----------------------------------------------------------------------------------------------------------//


// Is a linear algebra expression
//----------------------------------------------------------------------------------------------------------//
template<typename Derived>
struct has_linalg_op {
    static constexpr bool value = has_binary_matmul_op<Derived>::value;
};

// helper
template<typename Derived>
static constexpr bool has_linalg_op_v = has_linalg_op<Derived>::value;
//----------------------------------------------------------------------------------------------------------//


// Requires immediate evaluation
//----------------------------------------------------------------------------------------------------------//
template<typename Derived>
struct requires_evaluation {
    static constexpr bool value = has_linalg_op<Derived>::value;
};

// helper
template<typename Derived>
static constexpr bool requires_evaluation_v = requires_evaluation<Derived>::value;
//----------------------------------------------------------------------------------------------------------//



} // end of namespace Fastor


#endif // LINALG_TRAITS_H