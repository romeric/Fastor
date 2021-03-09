#ifndef OUTERPRODUCT_H
#define OUTERPRODUCT_H

#include "Fastor/backend/dyadic.h"
#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/TensorTraits.h"


namespace Fastor {

// These set of functions implement the outer/dyadic products of two or multiple tensor expressions
//---------------------------------------------------------------------------------------------------
template<typename T, size_t ...Rest0, size_t ...Rest1>
FASTOR_INLINE Tensor<T,Rest0...,Rest1...>
outer(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) {
     Tensor<T,Rest0...,Rest1...> out;
     _dyadic<T,pack_prod<Rest0...>::value,pack_prod<Rest1...>::value>(a.data(),b.data(),out.data());
     return out;
}

template<typename T, size_t ...Rest0>
FASTOR_INLINE Tensor<T,Rest0...>
outer(const Tensor<T,Rest0...> &a, const Tensor<T,1> &b) {
     Tensor<T,Rest0...,1> out = a*b.toscalar();
     return out;
}
template<typename T, size_t ...Rest1>
FASTOR_INLINE Tensor<T,Rest1...>
outer(const Tensor<T,1> &a, const Tensor<T,Rest1...> &b) {
     Tensor<T,1,Rest1...> out = a.toscalar()*b;
     return out;
}
template<typename T>
FASTOR_INLINE Tensor<T>
outer(const Tensor<T> &a, const Tensor<T> &b) {
     Tensor<T> out;
     _dyadic<T,1,1>(a.data(),b.data(),out.data());
     return out;
}
//---------------------------------------------------------------------------------------------------


// Expressions
//---------------------------------------------------------------------------------------------------
template<typename Derived0, size_t DIM0, typename Derived1, size_t DIM1,
    enable_if_t_<!is_tensor_v<Derived0> && !is_tensor_v<Derived1>,bool> = false >
FASTOR_INLINE
concatenated_tensor_t<typename Derived0::result_type,typename Derived1::result_type>
outer(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b) {
    using lhs_type = typename Derived0::result_type;
    using rhs_type = typename Derived1::result_type;
    return outer(lhs_type(a),rhs_type(b));
}
template<typename Derived0, size_t DIM0, typename Derived1, size_t DIM1,
    enable_if_t_<!is_tensor_v<Derived0> && is_tensor_v<Derived1>,bool> = false >
FASTOR_INLINE
concatenated_tensor_t<typename Derived0::result_type,typename Derived1::result_type>
outer(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b) {
    using lhs_type = typename Derived0::result_type;
    return outer(lhs_type(a),b.self());
}
template<typename Derived0, size_t DIM0, typename Derived1, size_t DIM1,
    enable_if_t_<is_tensor_v<Derived0> && !is_tensor_v<Derived1>,bool> = false >
FASTOR_INLINE
concatenated_tensor_t<typename Derived0::result_type,typename Derived1::result_type>
outer(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b) {
    using rhs_type = typename Derived1::result_type;
    return outer(a.self(),rhs_type(b));
}
//---------------------------------------------------------------------------------------------------


//---------------------------------------------------------------------------------------------------
template<typename T, size_t ... Rest0, size_t ... Rest1>
FASTOR_INLINE Tensor<T,Rest0...,Rest1...>
dyadic(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) {
    return outer(a,b);
}

template<typename Derived0, size_t DIM0, typename Derived1, size_t DIM1>
FASTOR_INLINE
concatenated_tensor_t<typename Derived0::result_type,typename Derived1::result_type>
dyadic(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b) {
    return outer(a,b);
}
//---------------------------------------------------------------------------------------------------


// multiple chained expressions - network outerproduct
//---------------------------------------------------------------------------------------------------
#if FASTOR_CXX_VERSION >= 2014
// template<typename AbstractTensorType0>
// FASTOR_INLINE
// auto
// outer(const AbstractTensorType0& a)
// {
//     return a;
// }

template<typename AbstractTensorType0, typename AbstractTensorType1, typename ... AbstractTensorTypes,
    enable_if_t_<is_greater_equal_v_<sizeof...(AbstractTensorTypes),1>,bool> = false >
FASTOR_INLINE
auto
outer(const AbstractTensorType0& a, const AbstractTensorType1& b, const AbstractTensorTypes& ... rest)
{
    const auto res = outer(a,b);
    return outer(res, rest...);
}

template<typename AbstractTensorType0, typename AbstractTensorType1, typename ... AbstractTensorTypes,
    enable_if_t_<is_greater_equal_v_<sizeof...(AbstractTensorTypes),1>,bool> = false >
FASTOR_INLINE
auto
dyadic(const AbstractTensorType0& a, const AbstractTensorType1& b, const AbstractTensorTypes& ... rest)
{
    return outer(a, b, rest...);
}
#endif
//---------------------------------------------------------------------------------------------------

} // end of namespace Fastor

#endif // OUTERPRODUCT_H

