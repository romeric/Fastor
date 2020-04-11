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



} // end of namespace Fastor


#endif // EXPRESSION_TRAITS_H