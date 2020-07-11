#ifndef EXPRESSION_TRAITS_H
#define EXPRESSION_TRAITS_H

#include "Fastor/config/config.h"
#include "Fastor/meta/meta.h"
#include <type_traits>

namespace Fastor {

//------------------------------------------------------------------------------------------------//
template<typename Derived>
struct is_expression {
    static constexpr bool value = false;
};
template<template<typename,size_t> class UnaryExpr, typename Expr, size_t DIM>
struct is_expression<UnaryExpr<Expr,DIM>> {
    static constexpr bool value = true;
};
template<template<class,class,size_t> class BinaryExpr, typename TLhs, typename TRhs, size_t DIMS>
struct is_expression<BinaryExpr<TLhs,TRhs,DIMS>> {
    static constexpr bool value = true;
};

template<typename Derived>
static constexpr bool is_expression_v = is_expression<Derived>::value;
//------------------------------------------------------------------------------------------------//


// Expression binder - by value or ref
//------------------------------------------------------------------------------------------------//
template<class T>
struct expression_binder_type {
#ifndef FASTOR_COPY_EXPR
    using type = conditional_t_<is_expression_v<T> || is_arithmetic_v_<T>, const T, const T&>;
#else
    using type = const T;
#endif
};

template<class T>
using expression_t = typename expression_binder_type<T>::type;
//------------------------------------------------------------------------------------------------//



template<typename T, size_t ... Rest>
class Tensor;

template<typename T, size_t ... Rest>
class TensorMap;

template<typename T, size_t ... Rest>
class SingleValueTensor;

// traits
//----------------------------------------------------------------------------------------------------------//
template<typename Derived>
struct is_unary_bool_op {
    static constexpr bool value = false;
};

template<typename Expr, size_t DIM>
struct is_unary_bool_op<UnaryNotOp<Expr,DIM>> {
    static constexpr bool value = true;
};
template<typename Expr, size_t DIM>
struct is_unary_bool_op<UnaryIsinfOp<Expr,DIM>> {
    static constexpr bool value = true;
};
template<typename Expr, size_t DIM>
struct is_unary_bool_op<UnaryIsnanOp<Expr,DIM>> {
    static constexpr bool value = true;
};
template<typename Expr, size_t DIM>
struct is_unary_bool_op<UnaryIsfiniteOp<Expr,DIM>> {
    static constexpr bool value = true;
};

template<size_t ... Rest>
struct is_unary_bool_op<Tensor<bool,Rest...>> {
    static constexpr bool value = true;
};
template<size_t ... Rest>
struct is_unary_bool_op<TensorMap<bool,Rest...>> {
    static constexpr bool value = true;
};
template<size_t ... Rest>
struct is_unary_bool_op<SingleValueTensor<bool,Rest...>> {
    static constexpr bool value = true;
};

template<typename Derived>
static constexpr bool is_unary_bool_op_v = is_unary_bool_op<Derived>::value;
//----------------------------------------------------------------------------------------------------------//


//----------------------------------------------------------------------------------------------------------//
template<typename Derived>
struct is_binary_cmp_op {
    static constexpr bool value = false;
};
#define FASTOR_MAKE_IS_BINARY_CMP_OP(NAME) \
template<typename TLhs, typename TRhs, size_t DIMS>\
struct is_binary_cmp_op<BinaryCmpOp##NAME <TLhs,TRhs,DIMS>> {\
    static constexpr bool value = true;\
};\

FASTOR_MAKE_IS_BINARY_CMP_OP(EQ)
FASTOR_MAKE_IS_BINARY_CMP_OP(NEQ)
FASTOR_MAKE_IS_BINARY_CMP_OP(LT)
FASTOR_MAKE_IS_BINARY_CMP_OP(GT)
FASTOR_MAKE_IS_BINARY_CMP_OP(LE)
FASTOR_MAKE_IS_BINARY_CMP_OP(GE)
FASTOR_MAKE_IS_BINARY_CMP_OP(AND)
FASTOR_MAKE_IS_BINARY_CMP_OP(OR)

template<size_t ... Rest>
struct is_binary_cmp_op<Tensor<bool,Rest...>> {
    static constexpr bool value = true;
};
template<size_t ... Rest>
struct is_binary_cmp_op<TensorMap<bool,Rest...>> {
    static constexpr bool value = true;
};
template<size_t ... Rest>
struct is_binary_cmp_op<SingleValueTensor<bool,Rest...>> {
    static constexpr bool value = true;
};

template<typename Derived>
static constexpr bool is_binary_cmp_op_v = is_binary_cmp_op<Derived>::value;
//----------------------------------------------------------------------------------------------------------//


/* Is boolean expresssion */
//----------------------------------------------------------------------------------------------------------//
template<typename Derived>
struct is_boolean_expression {
    static constexpr bool value = is_unary_bool_op<Derived>::value || is_binary_cmp_op<Derived>::value;
};

template<typename Derived>
static constexpr bool is_boolean_expression_v = is_boolean_expression<Derived>::value;
//----------------------------------------------------------------------------------------------------------//


//----------------------------------------------------------------------------------------------------------//
template<typename Derived>
struct is_tensor_view {
    static constexpr bool value = false;
};
template<typename T, size_t DIMS, size_t M, size_t N, size_t ...Rest>
struct is_tensor_view<TensorViewExpr<Tensor<T,M,N,Rest...>,DIMS>> {
    static constexpr bool value = true;
};
template<typename T, size_t DIMS, size_t M, size_t N, size_t ...Rest>
struct is_tensor_view<TensorConstViewExpr<Tensor<T,M,N,Rest...>,DIMS>> {
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
template<typename T, size_t DIMS, size_t M, size_t N, size_t ...Rest>
struct has_tensor_view<TensorConstViewExpr<Tensor<T,M,N,Rest...>,DIMS>> {
    static constexpr bool value = true;
};
template<template<typename,size_t> class UnaryExpr, typename Expr, size_t DIM>
struct has_tensor_view<UnaryExpr<Expr,DIM>> {
    static constexpr bool value = has_tensor_view<Expr>::value;
};
template<template<class,class,size_t> class BinaryExpr, typename TLhs, typename TRhs, size_t DIMS>
struct has_tensor_view<BinaryExpr<TLhs,TRhs,DIMS>> {
    static constexpr bool value = has_tensor_view<TRhs>::value || has_tensor_view<TLhs>::value;
};

template<typename Derived>
static constexpr bool has_tensor_view_v = has_tensor_view<Derived>::value;
//----------------------------------------------------------------------------------------------------------//


//----------------------------------------------------------------------------------------------------------//
template<typename Derived>
struct is_tensor_fixed_view_2d {
    static constexpr bool value = false;
};
template<typename T, size_t M, size_t N, typename Seq0, typename Seq1>
struct is_tensor_fixed_view_2d<TensorFixedViewExpr2D<Tensor<T,M,N>,Seq0,Seq1,2>> {
    static constexpr bool value = true;
};
template<typename T, size_t M, size_t N, typename Seq0, typename Seq1>
struct is_tensor_fixed_view_2d<TensorConstFixedViewExpr2D<Tensor<T,M,N>,Seq0,Seq1,2>> {
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
template<typename T, size_t M, size_t N, typename Seq0, typename Seq1>
struct has_tensor_fixed_view_2d<TensorConstFixedViewExpr2D<Tensor<T,M,N>,Seq0,Seq1,2>> {
    static constexpr bool value = true;
};
template<template<typename,size_t> class UnaryExpr, typename Expr, size_t DIM>
struct has_tensor_fixed_view_2d<UnaryExpr<Expr,DIM>> {
    static constexpr bool value = has_tensor_fixed_view_2d<Expr>::value;
};
template<template<class,class,size_t> class BinaryExpr, typename TLhs, typename TRhs, size_t DIMS>
struct has_tensor_fixed_view_2d<BinaryExpr<TLhs,TRhs,DIMS>> {
    static constexpr bool value = has_tensor_fixed_view_2d<TRhs>::value || has_tensor_fixed_view_2d<TLhs>::value;
};

template<typename Derived>
static constexpr bool has_tensor_fixed_view_2d_v = has_tensor_fixed_view_2d<Derived>::value;
//----------------------------------------------------------------------------------------------------------//


//----------------------------------------------------------------------------------------------------------//
template<typename Derived>
struct is_tensor_fixed_view_nd {
    static constexpr bool value = false;
};
template<typename T, size_t ...Rest, typename ...Fseq>
struct is_tensor_fixed_view_nd<TensorFixedViewExprnD<Tensor<T,Rest...>,Fseq...>> {
    static constexpr bool value = true;
};
template<typename T, size_t ...Rest, typename ...Fseq>
struct is_tensor_fixed_view_nd<TensorConstFixedViewExprnD<Tensor<T,Rest...>,Fseq...>> {
    static constexpr bool value = true;
};

template<typename Derived>
struct has_tensor_fixed_view_nd {
    static constexpr bool value = is_tensor_fixed_view_nd<Derived>::value ? true : false;
};
template<typename T, size_t ...Rest, typename ...Fseq>
struct has_tensor_fixed_view_nd<TensorFixedViewExprnD<Tensor<T,Rest...>,Fseq...>> {
    static constexpr bool value = true;
};
template<typename T, size_t ...Rest, typename ...Fseq>
struct has_tensor_fixed_view_nd<TensorConstFixedViewExprnD<Tensor<T,Rest...>,Fseq...>> {
    static constexpr bool value = true;
};
template<template<typename,size_t> class UnaryExpr, typename Expr, size_t DIM>
struct has_tensor_fixed_view_nd<UnaryExpr<Expr,DIM>> {
    static constexpr bool value = has_tensor_fixed_view_nd<Expr>::value;
};
template<template<class,class,size_t> class BinaryExpr, typename TLhs, typename TRhs, size_t DIMS>
struct has_tensor_fixed_view_nd<BinaryExpr<TLhs,TRhs,DIMS>> {
    static constexpr bool value = has_tensor_fixed_view_nd<TRhs>::value || has_tensor_fixed_view_nd<TLhs>::value;
};

template<typename Derived>
static constexpr bool has_tensor_fixed_view_nd_v = has_tensor_fixed_view_nd<Derived>::value;
//----------------------------------------------------------------------------------------------------------//



//------------------------------------------------------------------------------------------------//
template<typename T, typename T2 = void>
struct get_binary_arithmetic_result_type;
template<typename T>
struct get_binary_arithmetic_result_type<T, enable_if_t_<is_primitive_v_<T> > > {
    using type = T;
};
template<typename T>
struct get_binary_arithmetic_result_type<T, enable_if_t_<!is_primitive_v_<T> > > {
    using type = typename T::result_type;
};

template<class Derived>
struct binary_arithmetic_result_type;
template<template<typename,typename,size_t> class BinaryExpr, typename TLhs, typename TRhs, size_t DIM0>
struct binary_arithmetic_result_type< BinaryExpr<TLhs,TRhs,DIM0> > {
    using type = conditional_t_<!is_primitive_v_<TLhs>,
        typename get_binary_arithmetic_result_type<TLhs>::type,
        typename get_binary_arithmetic_result_type<TRhs>::type >;
};
template<class Derived>
using binary_arithmetic_result_t = typename binary_arithmetic_result_type<Derived>::type;
//------------------------------------------------------------------------------------------------//



//------------------------------------------------------------------------------------------------//
template<typename T, typename T2 = void>
struct get_binary_op_simd_vector_type;
template<typename T>
struct get_binary_op_simd_vector_type<T, enable_if_t_<is_primitive_v_<T> > > {
    using type = T;
};
template<typename T>
struct get_binary_op_simd_vector_type<T, enable_if_t_<!is_primitive_v_<T> > > {
    using type = typename T::simd_vector_type;
};

template<class Derived>
struct binary_op_simd_vector_type;
template<template<typename,typename,size_t> class BinaryExpr, typename TLhs, typename TRhs, size_t DIM0>
struct binary_op_simd_vector_type< BinaryExpr<TLhs,TRhs,DIM0> > {
    using type = conditional_t_<!is_primitive_v_<TLhs>,
        typename get_binary_op_simd_vector_type<TLhs>::type,
        typename get_binary_op_simd_vector_type<TRhs>::type >;
};
template<class Derived>
using binary_op_simd_vector_t = typename binary_op_simd_vector_type<Derived>::type;
//------------------------------------------------------------------------------------------------//




template<class T>
struct scalar_type_finder;

} // end of namespace Fastor


#endif // EXPRESSION_TRAITS_H
