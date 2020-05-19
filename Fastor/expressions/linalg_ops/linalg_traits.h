#ifndef LINALG_TRAITS_H
#define LINALG_TRAITS_H


#include "Fastor/expressions/linalg_ops/binary_matmul_op.h"
#include "Fastor/expressions/linalg_ops/unary_trans_op.h"
#include "Fastor/expressions/linalg_ops/unary_ctrans_op.h"
#include "Fastor/expressions/linalg_ops/unary_adj_op.h"
#include "Fastor/expressions/linalg_ops/unary_cof_op.h"
#include "Fastor/expressions/linalg_ops/unary_inv_op.h"
#include "Fastor/expressions/linalg_ops/binary_solve_op.h"
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
template<typename Derived0, typename Derived1, size_t DIM>
struct is_binary_matmul_op<BinaryMatMulOp<Derived0,Derived1,DIM>> {
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
template<typename Derived0, typename Derived1, size_t DIM>
struct has_binary_matmul_op<BinaryMatMulOp<Derived0,Derived1,DIM>> {
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


// Is unary trans expression
//----------------------------------------------------------------------------------------------------------//
template<typename Derived>
struct is_unary_trans_op {
    static constexpr bool value = false;
};
template<typename Derived, size_t DIM>
struct is_unary_trans_op<UnaryTransOp<Derived,DIM>> {
    static constexpr bool value = true;
};

template<typename Derived>
struct has_unary_trans_op {
    static constexpr bool value = is_unary_trans_op<Derived>::value ? true : false;
};
template<typename Derived, size_t DIM>
struct has_unary_trans_op<UnaryTransOp<Derived,DIM>> {
    static constexpr bool value = true;
};
template<template<typename,size_t> class UnaryExpr, typename Expr, size_t DIM>
struct has_unary_trans_op<UnaryExpr<Expr,DIM>> {
    static constexpr bool value = has_unary_trans_op<Expr>::value;
};
template<template<class,class,size_t> class BinaryExpr, typename TLhs, typename TRhs, size_t DIMS>
struct has_unary_trans_op<BinaryExpr<TLhs,TRhs,DIMS>> {
    static constexpr bool value = has_unary_trans_op<TRhs>::value || has_unary_trans_op<TLhs>::value;
};

// helper
template<typename Derived>
static constexpr bool is_unary_trans_op_v = is_unary_trans_op<Derived>::value;
template<typename Derived>
static constexpr bool has_unary_trans_op_v = has_unary_trans_op<Derived>::value;
//----------------------------------------------------------------------------------------------------------//


// Is unary ctrans expression
//----------------------------------------------------------------------------------------------------------//
template<typename Derived>
struct is_unary_ctrans_op {
    static constexpr bool value = false;
};
template<typename Derived, size_t DIM>
struct is_unary_ctrans_op<UnaryCTransOp<Derived,DIM>> {
    static constexpr bool value = true;
};

template<typename Derived>
struct has_unary_ctrans_op {
    static constexpr bool value = is_unary_ctrans_op<Derived>::value ? true : false;
};
template<typename Derived, size_t DIM>
struct has_unary_ctrans_op<UnaryCTransOp<Derived,DIM>> {
    static constexpr bool value = true;
};
template<template<typename,size_t> class UnaryExpr, typename Expr, size_t DIM>
struct has_unary_ctrans_op<UnaryExpr<Expr,DIM>> {
    static constexpr bool value = has_unary_ctrans_op<Expr>::value;
};
template<template<class,class,size_t> class BinaryExpr, typename TLhs, typename TRhs, size_t DIMS>
struct has_unary_ctrans_op<BinaryExpr<TLhs,TRhs,DIMS>> {
    static constexpr bool value = has_unary_ctrans_op<TRhs>::value || has_unary_ctrans_op<TLhs>::value;
};

// helper
template<typename Derived>
static constexpr bool is_unary_ctrans_op_v = is_unary_ctrans_op<Derived>::value;
template<typename Derived>
static constexpr bool has_unary_ctrans_op_v = has_unary_ctrans_op<Derived>::value;
//----------------------------------------------------------------------------------------------------------//


// Is unary adj expression
//----------------------------------------------------------------------------------------------------------//
template<typename Derived>
struct is_unary_adj_op {
    static constexpr bool value = false;
};
template<typename Derived, size_t DIM>
struct is_unary_adj_op<UnaryAdjOp<Derived,DIM>> {
    static constexpr bool value = true;
};

template<typename Derived>
struct has_unary_adj_op {
    static constexpr bool value = is_unary_adj_op<Derived>::value ? true : false;
};
template<typename Derived, size_t DIM>
struct has_unary_adj_op<UnaryAdjOp<Derived,DIM>> {
    static constexpr bool value = true;
};
template<template<typename,size_t> class UnaryExpr, typename Expr, size_t DIM>
struct has_unary_adj_op<UnaryExpr<Expr,DIM>> {
    static constexpr bool value = has_unary_adj_op<Expr>::value;
};
template<template<class,class,size_t> class BinaryExpr, typename TLhs, typename TRhs, size_t DIMS>
struct has_unary_adj_op<BinaryExpr<TLhs,TRhs,DIMS>> {
    static constexpr bool value = has_unary_adj_op<TRhs>::value || has_unary_adj_op<TLhs>::value;
};

// helper
template<typename Derived>
static constexpr bool is_unary_adj_op_v = is_unary_adj_op<Derived>::value;
template<typename Derived>
static constexpr bool has_unary_adj_op_v = has_unary_adj_op<Derived>::value;
//----------------------------------------------------------------------------------------------------------//

// Is unary adj expression
//----------------------------------------------------------------------------------------------------------//
template<typename Derived>
struct is_unary_cof_op {
    static constexpr bool value = false;
};
template<typename Derived, size_t DIM>
struct is_unary_cof_op<UnaryCofOp<Derived,DIM>> {
    static constexpr bool value = true;
};

template<typename Derived>
struct has_unary_cof_op {
    static constexpr bool value = is_unary_cof_op<Derived>::value ? true : false;
};
template<typename Derived, size_t DIM>
struct has_unary_cof_op<UnaryCofOp<Derived,DIM>> {
    static constexpr bool value = true;
};
template<template<typename,size_t> class UnaryExpr, typename Expr, size_t DIM>
struct has_unary_cof_op<UnaryExpr<Expr,DIM>> {
    static constexpr bool value = has_unary_cof_op<Expr>::value;
};
template<template<class,class,size_t> class BinaryExpr, typename TLhs, typename TRhs, size_t DIMS>
struct has_unary_cof_op<BinaryExpr<TLhs,TRhs,DIMS>> {
    static constexpr bool value = has_unary_cof_op<TRhs>::value || has_unary_cof_op<TLhs>::value;
};

// helper
template<typename Derived>
static constexpr bool is_unary_cof_op_v = is_unary_cof_op<Derived>::value;
template<typename Derived>
static constexpr bool has_unary_cof_op_v = has_unary_cof_op<Derived>::value;
//----------------------------------------------------------------------------------------------------------//


// Is unary inv expression
//----------------------------------------------------------------------------------------------------------//
template<typename Derived>
struct is_unary_inv_op {
    static constexpr bool value = false;
};
template<typename Derived, size_t DIM>
struct is_unary_inv_op<UnaryInvOp<Derived,DIM>> {
    static constexpr bool value = true;
};

template<typename Derived>
struct has_unary_inv_op {
    static constexpr bool value = is_unary_inv_op<Derived>::value ? true : false;
};
template<typename Derived, size_t DIM>
struct has_unary_inv_op<UnaryInvOp<Derived,DIM>> {
    static constexpr bool value = true;
};
template<template<typename,size_t> class UnaryExpr, typename Expr, size_t DIM>
struct has_unary_inv_op<UnaryExpr<Expr,DIM>> {
    static constexpr bool value = has_unary_inv_op<Expr>::value;
};
template<template<class,class,size_t> class BinaryExpr, typename TLhs, typename TRhs, size_t DIMS>
struct has_unary_inv_op<BinaryExpr<TLhs,TRhs,DIMS>> {
    static constexpr bool value = has_unary_inv_op<TRhs>::value || has_unary_inv_op<TLhs>::value;
};

// helper
template<typename Derived>
static constexpr bool is_unary_inv_op_v = is_unary_inv_op<Derived>::value;
template<typename Derived>
static constexpr bool has_unary_inv_op_v = has_unary_inv_op<Derived>::value;
//----------------------------------------------------------------------------------------------------------//


// Is a linear algebra expression
//----------------------------------------------------------------------------------------------------------//
template<typename Derived>
struct has_linalg_op {
    static constexpr bool value = has_binary_matmul_op<Derived>::value || has_unary_trans_op<Derived>::value  ||
                                  has_unary_ctrans_op<Derived>::value  || has_unary_adj_op<Derived>::value    ||
                                  has_unary_cof_op<Derived>::value     || has_unary_inv_op<Derived>::value;
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
