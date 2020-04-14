#ifndef EXPRESSION_TRAITS_H
#define EXPRESSION_TRAITS_H

#include "Fastor/commons/commons.h"
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
struct ExprBinderType {
#ifndef FASTOR_COPY_EXPR
    using type = typename std::conditional<std::is_arithmetic<T>::value, const T, const T&>::type;
#else
    using type = T;
#endif
};

template<class T>
using expression_t = typename ExprBinderType<T>::type;
//------------------------------------------------------------------------------------------------//



//------------------------------------------------------------------------------------------------//
template<typename T, typename T2 = void>
struct get_binary_arithmetic_result_type;
template<typename T>
struct get_binary_arithmetic_result_type<T, enable_if_t_<is_arithmetic_v_<T> > > {
    using type = T;
};
template<typename T>
struct get_binary_arithmetic_result_type<T, enable_if_t_<!is_arithmetic_v_<T> > > {
    using type = typename T::result_type;
};

template<class Derived>
struct binary_arithmetic_result_type;
template<template<typename,typename,size_t> class BinaryExpr, typename TLhs, typename TRhs, size_t DIM0>
struct binary_arithmetic_result_type< BinaryExpr<TLhs,TRhs,DIM0> > {
    using type = conditional_t_<!is_arithmetic_v_<TLhs>,
        typename get_binary_arithmetic_result_type<TLhs>::type,
        typename get_binary_arithmetic_result_type<TRhs>::type >;
};
template<class Derived>
using binary_arithmetic_result_t = typename binary_arithmetic_result_type<Derived>::type;
//------------------------------------------------------------------------------------------------//






} // end of namespace Fastor


#endif // EXPRESSION_TRAITS_H