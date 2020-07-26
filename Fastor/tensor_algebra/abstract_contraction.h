#ifndef ABSTRACT_CONTRACTION_H
#define ABSTRACT_CONTRACTION_H

#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/tensor_algebra/indicial.h"
#include "Fastor/meta/opmin_meta.h"
#include "Fastor/expressions/expression_traits.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"
#include "Fastor/tensor_algebra/contraction.h"
#include "Fastor/tensor_algebra/network_contraction.h"
#include "Fastor/tensor_algebra/network_contraction_no_opmin.h"


namespace Fastor {

#if FASTOR_CXX_VERSION >= 2014
// The following set of functions implement by-pair as well as
// network contraction/einsum for infinite number of expressions
// the expressions are always evaluated so no aliasing occurs
// and the contractions are forwarded to their tensor counterparts
// which perform operation minimiation and are optimised for performance

// Single expression - contraction
//-------------------------------------------------------------------------------------------------
template<class Index_I, typename Derived0, size_t DIM0,
    enable_if_t_<!is_tensor_v<Derived0>,bool> = false>
FASTOR_INLINE
decltype(auto)
contraction(const AbstractTensor<Derived0,DIM0> &a)
{
    typename Derived0::result_type res_a(a);
    return extractor_contract_1<Index_I>::contract_impl(res_a);
}
//-------------------------------------------------------------------------------------------------

// Single expression - einsum
//-------------------------------------------------------------------------------------------------
template<class Index_I, typename Derived0, size_t DIM0,
    enable_if_t_<!is_tensor_v<Derived0>,bool> = false>
FASTOR_INLINE
decltype(auto)
einsum(const AbstractTensor<Derived0,DIM0> &a)
{
    typename Derived0::result_type res_a(a);
    return extractor_contract_1<Index_I>::contract_impl(res_a);
}
//-------------------------------------------------------------------------------------------------


// By pair expressions - contraction
//-------------------------------------------------------------------------------------------------
template<class Index_I, class Index_J, typename Derived0, typename Derived1, size_t DIM0, size_t DIM1,
    enable_if_t_<!is_tensor_v<Derived0> && !is_tensor_v<Derived1>,bool> = false>
FASTOR_INLINE
decltype(auto)
contraction(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b)
{
    typename Derived0::result_type res_a(a);
    typename Derived1::result_type res_b(b);
    return extractor_contract_2<Index_I,Index_J>::contract_impl(res_a,res_b);
}
template<class Index_I, class Index_J, typename Derived0, typename Derived1, size_t DIM0, size_t DIM1,
    enable_if_t_<is_tensor_v<Derived0> && !is_tensor_v<Derived1>,bool> = false>
FASTOR_INLINE
decltype(auto)
contraction(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b)
{
    typename Derived1::result_type res_b(b);
    return extractor_contract_2<Index_I,Index_J>::contract_impl(a,res_b);
}
template<class Index_I, class Index_J, typename Derived0, typename Derived1, size_t DIM0, size_t DIM1,
    enable_if_t_<!is_tensor_v<Derived0> && is_tensor_v<Derived1>,bool> = false>
FASTOR_INLINE
decltype(auto)
contraction(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b)
{
    typename Derived0::result_type res_a(a);
    return extractor_contract_2<Index_I,Index_J>::contract_impl(res_a,b);
}
//-------------------------------------------------------------------------------------------------


// By pair expressions - einsum
//-------------------------------------------------------------------------------------------------
template<class Index_I, class Index_J, typename Derived0, typename Derived1, size_t DIM0, size_t DIM1,
    enable_if_t_<!is_tensor_v<Derived0> && !is_tensor_v<Derived1>,bool> = false>
FASTOR_INLINE
decltype(auto)
einsum(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b)
{
    typename Derived0::result_type res_a(a);
    typename Derived1::result_type res_b(b);
    return einsum<Index_I,Index_J>(res_a,res_b);
}
template<class Index_I, class Index_J, typename Derived0, typename Derived1, size_t DIM0, size_t DIM1,
    enable_if_t_<is_tensor_v<Derived0> && !is_tensor_v<Derived1>,bool> = false>
FASTOR_INLINE
decltype(auto)
einsum(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b)
{
    typename Derived1::result_type res_b(b);
    return einsum<Index_I,Index_J>(a,res_b);
}
template<class Index_I, class Index_J, typename Derived0, typename Derived1, size_t DIM0, size_t DIM1,
    enable_if_t_<!is_tensor_v<Derived0> && is_tensor_v<Derived1>,bool> = false>
FASTOR_INLINE
decltype(auto)
einsum(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b)
{
    typename Derived0::result_type res_a(a);
    return einsum<Index_I,Index_J>(res_a,b);
}
//-------------------------------------------------------------------------------------------------



// network contraction for expressions
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
namespace internal {

// helper functions to evaluate expression in to intermediate tensors
// We evaluate all intermediate tensors and pack them in to a single tuple
FASTOR_INLINE
std::tuple<>
contraction_chain_evaluate()
{
    return std::tuple<>{};
}

// Note that if the expression is a tensor the evaluation is free
// as evaluate returns the tensor itself
template<typename Derived0, size_t DIM0>
FASTOR_INLINE
decltype(auto)
contraction_chain_evaluate(const AbstractTensor<Derived0,DIM0>& a)
{
    return evaluate(a.self());
}

template<typename AbstractTensorType0, typename ... AbstractTensorTypes>
FASTOR_INLINE
decltype(auto)
contraction_chain_evaluate(const AbstractTensorType0& a, const AbstractTensorTypes& ... rest)
{
    return std::tuple_cat(std::make_tuple(evaluate(a)),contraction_chain_evaluate(rest...));
}

// helper functor to unpack the tuple and forward the pack of tensor
// to network contraction for operation minimisation
template<class Index_I, class Index_J, class ... Index_Ks>
struct unpack_contraction_tuple {

    template<typename Tuple, size_t ... I>
    static auto apply(Tuple t, std_ext::index_sequence<I ...>)
    {
         return contraction<Index_I,Index_J,Index_Ks...>(std::get<I>(t) ...);
    }
    template<typename Tuple>
    static auto apply(Tuple t)
    {
        constexpr auto size = std::tuple_size<Tuple>::value;
        return apply(t, std_ext::make_index_sequence<size>{});
    }
};

// helper functor to unpack the tuple and forward the pack of tensor
// to network einsum for operation minimisation
template<class Index_I, class Index_J, class ... Index_Ks>
struct unpack_einsum_tuple {

    template<typename Tuple, size_t ... I>
    static auto apply(Tuple t, std_ext::index_sequence<I ...>)
    {
         return einsum<Index_I,Index_J,Index_Ks...>(std::get<I>(t) ...);
    }
    template<typename Tuple>
    static auto apply(Tuple t)
    {
        constexpr auto size = std::tuple_size<Tuple>::value;
        return apply(t, std_ext::make_index_sequence<size>{});
    }
};

template<class Index_I, class Index_J, class ... Index_Ks>
struct unpack_einsum_helper_tuple {

    template<typename Tuple, size_t ... I>
    static constexpr auto apply(Tuple t, std_ext::index_sequence<I ...>)
    {
         return einsum_helper<Index_I,Index_J,Index_Ks...,decltype(std::get<I>(t)) ...>{};
    }
    template<typename Tuple>
    static constexpr auto apply(Tuple t)
    {
        constexpr auto size = std::tuple_size<Tuple>::value;
        return apply(t, std_ext::make_index_sequence<size>{});
    }
};
} // internal
//-------------------------------------------------------------------------------------------------

// network contraction
template<class Index_I, class Index_J, class ... Index_Ks,
        typename AbstractTensorType0, typename AbstractTensorType1, typename ... AbstractTensorTypes>
FASTOR_INLINE
auto
contraction(const AbstractTensorType0& a, const AbstractTensorType1& b, const AbstractTensorTypes& ... rest)
{
    return internal::unpack_contraction_tuple<Index_I,Index_J,Index_Ks...>::apply(internal::contraction_chain_evaluate(a,b,rest...));
}

// network einsum
template<class Index_I, class Index_J, class ... Index_Ks,
        typename AbstractTensorType0, typename AbstractTensorType1, typename ... AbstractTensorTypes>
FASTOR_INLINE
auto
einsum(const AbstractTensorType0& a, const AbstractTensorType1& b, const AbstractTensorTypes& ... rest)
{
    // network einsum is not defined yet
    return internal::unpack_einsum_tuple<Index_I,Index_J,Index_Ks...>::apply(internal::contraction_chain_evaluate(a,b,rest...));
    // but it dispatches to network contraction anyway and contraction uses by-pair einsum in turn
    // return internal::unpack_contraction_tuple<Index_I,Index_J,Index_Ks...>::apply(internal::contraction_chain_evaluate(a,b,rest...));
}

#endif // CXX 2014
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

} // end of namespace Fastor

#endif // ABSTRACT_CONTRACTION_H
