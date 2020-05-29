#ifndef EXPLICIT_EINSUM_H
#define EXPLICIT_EINSUM_H

#include "Fastor/tensor_algebra/indicial.h"
#include "Fastor/meta/opmin_meta.h"
#include "Fastor/tensor_algebra/permutation.h"
#include "Fastor/tensor_algebra/permute.h"
#include "Fastor/tensor_algebra/einsum.h"
#include "Fastor/tensor_algebra/einsum_network.h"
#include "Fastor/tensor_algebra/abstract_contraction.h"

namespace Fastor {

#if FASTOR_CXX_VERSION >= 2014

// Single tensor
//-----------------------------------------------------------------------------------------------------------------------//
template<class Index_I, class Index_O,
         typename T, size_t ... Rest0>
FASTOR_INLINE
decltype(auto) einsum(const Tensor<T,Rest0...> &a) {
    using _einsum_helper = einsum_helper<Index_I,Tensor<T,Rest0...>>;
    using resulting_index_einsum  = typename _einsum_helper::resulting_index;
    using resulting_tensor_einsum = typename _einsum_helper::resulting_tensor;
    auto res = einsum<Index_I>(a);

    constexpr bool requires_permutation = requires_permute_v<typename Index_O::parent_type, resulting_tensor_einsum>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<typename Index_O::parent_type>(res);
}
//-----------------------------------------------------------------------------------------------------------------------//


// Two tensor (by-pair)
//-----------------------------------------------------------------------------------------------------------------------//
template<class Index_I, class Index_J, class Index_O,
         typename T, size_t ... Rest0, size_t ... Rest1>
FASTOR_INLINE
decltype(auto) einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) {
    using _einsum_helper = einsum_helper<Index_I,Index_J,Tensor<T,Rest0...>,Tensor<T,Rest1...>>;
    using resulting_index_einsum  = typename _einsum_helper::resulting_index;
    using resulting_tensor_einsum = typename _einsum_helper::resulting_tensor;
    auto res = einsum<Index_I,Index_J>(a,b);

    constexpr bool requires_permutation = requires_permute_v<typename Index_O::parent_type, resulting_tensor_einsum>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<typename Index_O::parent_type>(res);
}
//-----------------------------------------------------------------------------------------------------------------------//


// 3 tensor network
//-----------------------------------------------------------------------------------------------------------------------//
template<class Index_I, class Index_J, class Index_K, class Index_O,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2>
FASTOR_INLINE
decltype(auto) einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c) {
    using _einsum_helper = einsum_helper<Index_I,Index_J,Index_K,Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>>;
    using resulting_index_einsum  = typename _einsum_helper::resulting_index;
    using resulting_tensor_einsum = typename _einsum_helper::resulting_tensor;
    auto res = einsum<Index_I,Index_J,Index_K>(a,b,c);

    constexpr bool requires_permutation = requires_permute_v<typename Index_O::parent_type, resulting_tensor_einsum>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<typename Index_O::parent_type>(res);
}
//-----------------------------------------------------------------------------------------------------------------------//


// 4 tensor network
//-----------------------------------------------------------------------------------------------------------------------//
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_O,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3>
FASTOR_INLINE
decltype(auto) einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d) {
    using _einsum_helper = einsum_helper<Index_I,Index_J,Index_K,Index_L,
        Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>>;
    using resulting_index_einsum  = typename _einsum_helper::resulting_index;
    using resulting_tensor_einsum = typename _einsum_helper::resulting_tensor;
    auto res = einsum<Index_I,Index_J,Index_K,Index_L>(a,b,c,d);

    constexpr bool requires_permutation = requires_permute_v<typename Index_O::parent_type, resulting_tensor_einsum>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<typename Index_O::parent_type>(res);
}
//-----------------------------------------------------------------------------------------------------------------------//


// 5 tensor network
//-----------------------------------------------------------------------------------------------------------------------//
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, class Index_O,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4>
FASTOR_INLINE
decltype(auto) einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c,
    const Tensor<T,Rest3...> &d, const Tensor<T,Rest4...> &e) {
    using _einsum_helper = einsum_helper<Index_I,Index_J,Index_K,Index_L,Index_M,
        Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>,Tensor<T,Rest4...>>;
    using resulting_index_einsum  = typename _einsum_helper::resulting_index;
    using resulting_tensor_einsum = typename _einsum_helper::resulting_tensor;
    auto res = einsum<Index_I,Index_J,Index_K,Index_L,Index_M>(a,b,c,d,e);

    constexpr bool requires_permutation = requires_permute_v<typename Index_O::parent_type, resulting_tensor_einsum>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<typename Index_O::parent_type>(res);
}
//-----------------------------------------------------------------------------------------------------------------------//


// 6 tensor network
//-----------------------------------------------------------------------------------------------------------------------//
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, class Index_N, class Index_O,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4, size_t ... Rest5>
FASTOR_INLINE
decltype(auto) einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c,
    const Tensor<T,Rest3...> &d, const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f) {
    using _einsum_helper = einsum_helper<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,
        Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>,Tensor<T,Rest4...>,Tensor<T,Rest5...>>;
    using resulting_index_einsum  = typename _einsum_helper::resulting_index;
    using resulting_tensor_einsum = typename _einsum_helper::resulting_tensor;
    auto res = einsum<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>(a,b,c,d,e,f);

    constexpr bool requires_permutation = requires_permute_v<typename Index_O::parent_type, resulting_tensor_einsum>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<typename Index_O::parent_type>(res);
}
//-----------------------------------------------------------------------------------------------------------------------//



// network einsum for expressions
//-----------------------------------------------------------------------------------------------------------------------//
template<class Index_I, class ... Index_Ks, class Index_O,
        typename AbstractTensorType0, typename AbstractTensorType1, typename ... AbstractTensorTypes>
FASTOR_INLINE
decltype(auto)
einsum(const AbstractTensorType0& a, const AbstractTensorTypes& ... rest)
{
    auto res = internal::unpack_einsum_tuple<Index_I,Index_Ks...>::apply(internal::contraction_chain_evaluate(a,rest...));

    constexpr bool requires_permutation = requires_permute_v<typename Index_O::parent_type, decltype(res)>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<typename Index_O::parent_type>(res);
}
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------


#endif // FASTOR_CXX_VERSION >= 2014

} // end of namespace


#endif // EXPLICIT_EINSUM_H
