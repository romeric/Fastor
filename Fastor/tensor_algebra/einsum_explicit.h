#ifndef EXPLICIT_EINSUM_H
#define EXPLICIT_EINSUM_H

#include "Fastor/tensor_algebra/indicial.h"
#include "Fastor/meta/opmin_meta.h"
#include "Fastor/tensor_algebra/permutation.h"
#include "Fastor/tensor_algebra/permute.h"
#include "Fastor/tensor_algebra/einsum.h"
#include "Fastor/tensor_algebra/network_einsum.h"
#include "Fastor/tensor_algebra/abstract_contraction.h"

namespace Fastor {

#if FASTOR_CXX_VERSION >= 2017

// Single tensor
//-----------------------------------------------------------------------------------------------------------------------//
template<class Index_I, class Index_O,
         typename T, size_t ... Rest0>
FASTOR_INLINE
typename permute_helper<internal::permute_mapped_index_t<
    typename einsum_helper<Index_I,Tensor<T,Rest0...>>::resulting_index, typename Index_O::parent_type>,
    typename einsum_helper<Index_I,Tensor<T,Rest0...>>::resulting_tensor>::resulting_tensor
einsum(const Tensor<T,Rest0...> &a) {
    using _einsum_helper = einsum_helper<Index_I,Tensor<T,Rest0...>>;
    using resulting_index_einsum  = typename _einsum_helper::resulting_index;
    using resulting_tensor_einsum = typename _einsum_helper::resulting_tensor;
    auto res = einsum<Index_I>(a);

    using mapped_index = internal::permute_mapped_index_t<resulting_index_einsum,typename Index_O::parent_type>;
    constexpr bool requires_permutation = requires_permute_v<mapped_index, resulting_tensor_einsum>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<mapped_index>(res);
}
//-----------------------------------------------------------------------------------------------------------------------//


// Two tensor (by-pair)
//-----------------------------------------------------------------------------------------------------------------------//
template<class Index_I, class Index_J, class Index_O,
         typename T, size_t ... Rest0, size_t ... Rest1>
FASTOR_INLINE
typename permute_helper<internal::permute_mapped_index_t<
    typename einsum_helper<Index_I,Index_J,Tensor<T,Rest0...>,Tensor<T,Rest1...>>::resulting_index, typename Index_O::parent_type>,
    typename einsum_helper<Index_I,Index_J,Tensor<T,Rest0...>,Tensor<T,Rest1...>>::resulting_tensor>::resulting_tensor
einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) {
    using _einsum_helper = einsum_helper<Index_I,Index_J,Tensor<T,Rest0...>,Tensor<T,Rest1...>>;
    using resulting_index_einsum  = typename _einsum_helper::resulting_index;
    using resulting_tensor_einsum = typename _einsum_helper::resulting_tensor;
    auto res = einsum<Index_I,Index_J>(a,b);

    using mapped_index = internal::permute_mapped_index_t<resulting_index_einsum,typename Index_O::parent_type>;
    constexpr bool requires_permutation = requires_permute_v<mapped_index, resulting_tensor_einsum>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<mapped_index>(res);
}
//-----------------------------------------------------------------------------------------------------------------------//


// 3 tensor network
//-----------------------------------------------------------------------------------------------------------------------//
template<class Index_I, class Index_J, class Index_K, class Index_O,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2>
FASTOR_INLINE
typename permute_helper<
    internal::permute_mapped_index_t<
        typename einsum_helper<Index_I,Index_J,Index_K,Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>>::resulting_index,
        typename Index_O::parent_type
    >,
    typename einsum_helper<Index_I,Index_J,Index_K,Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>>::resulting_tensor>::resulting_tensor
einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c) {
    using _einsum_helper = einsum_helper<Index_I,Index_J,Index_K,Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>>;
    using resulting_index_einsum  = typename _einsum_helper::resulting_index;
    using resulting_tensor_einsum = typename _einsum_helper::resulting_tensor;
    auto res = einsum<Index_I,Index_J,Index_K>(a,b,c);

    using mapped_index = internal::permute_mapped_index_t<resulting_index_einsum,typename Index_O::parent_type>;
    constexpr bool requires_permutation = requires_permute_v<mapped_index, resulting_tensor_einsum>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<mapped_index>(res);
}
//-----------------------------------------------------------------------------------------------------------------------//


// 4 tensor network
//-----------------------------------------------------------------------------------------------------------------------//
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_O,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3>
FASTOR_INLINE
typename permute_helper<
    internal::permute_mapped_index_t<
        typename einsum_helper<Index_I,Index_J,Index_K,Index_L,
        Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>>::resulting_index,
        typename Index_O::parent_type
    >,
    typename einsum_helper<Index_I,Index_J,Index_K,Index_L,
        Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>>::resulting_tensor>::resulting_tensor
einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d) {
    using _einsum_helper = einsum_helper<Index_I,Index_J,Index_K,Index_L,
        Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>>;
    using resulting_index_einsum  = typename _einsum_helper::resulting_index;
    using resulting_tensor_einsum = typename _einsum_helper::resulting_tensor;
    auto res = einsum<Index_I,Index_J,Index_K,Index_L>(a,b,c,d);

    using mapped_index = internal::permute_mapped_index_t<resulting_index_einsum,typename Index_O::parent_type>;
    constexpr bool requires_permutation = requires_permute_v<mapped_index, resulting_tensor_einsum>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<mapped_index>(res);
}
//-----------------------------------------------------------------------------------------------------------------------//


// 5 tensor network
//-----------------------------------------------------------------------------------------------------------------------//
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, class Index_O,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4>
FASTOR_INLINE
typename permute_helper<
    internal::permute_mapped_index_t<
        typename einsum_helper<Index_I,Index_J,Index_K,Index_L,Index_M,
        Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>,Tensor<T,Rest4...>>::resulting_index,
        typename Index_O::parent_type
    >,
    typename einsum_helper<Index_I,Index_J,Index_K,Index_L,Index_M,
        Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>,Tensor<T,Rest4...>>::resulting_tensor
    >::resulting_tensor
einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c,
    const Tensor<T,Rest3...> &d, const Tensor<T,Rest4...> &e) {
    using _einsum_helper = einsum_helper<Index_I,Index_J,Index_K,Index_L,Index_M,
        Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>,Tensor<T,Rest4...>>;
    using resulting_index_einsum  = typename _einsum_helper::resulting_index;
    using resulting_tensor_einsum = typename _einsum_helper::resulting_tensor;
    auto res = einsum<Index_I,Index_J,Index_K,Index_L,Index_M>(a,b,c,d,e);

    using mapped_index = internal::permute_mapped_index_t<resulting_index_einsum,typename Index_O::parent_type>;
    constexpr bool requires_permutation = requires_permute_v<mapped_index, resulting_tensor_einsum>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<mapped_index>(res);
}
//-----------------------------------------------------------------------------------------------------------------------//


// 6 tensor network
//-----------------------------------------------------------------------------------------------------------------------//
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, class Index_N, class Index_O,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4, size_t ... Rest5>
FASTOR_INLINE
typename permute_helper<
    internal::permute_mapped_index_t<
        typename einsum_helper<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,
        Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>,Tensor<T,Rest4...>,Tensor<T,Rest5...>>::resulting_index,
        typename Index_O::parent_type
    >,
    typename einsum_helper<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,
        Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>,Tensor<T,Rest4...>,Tensor<T,Rest5...>>::resulting_tensor
    >::resulting_tensor
einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c,
    const Tensor<T,Rest3...> &d, const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f) {
    using _einsum_helper = einsum_helper<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,
        Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>,Tensor<T,Rest4...>,Tensor<T,Rest5...>>;
    using resulting_index_einsum  = typename _einsum_helper::resulting_index;
    using resulting_tensor_einsum = typename _einsum_helper::resulting_tensor;
    auto res = einsum<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>(a,b,c,d,e,f);

    using mapped_index = internal::permute_mapped_index_t<resulting_index_einsum,typename Index_O::parent_type>;
    constexpr bool requires_permutation = requires_permute_v<mapped_index, resulting_tensor_einsum>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<mapped_index>(res);
}
//-----------------------------------------------------------------------------------------------------------------------//



// network einsum for expressions
//-----------------------------------------------------------------------------------------------------------------------//
// single expression explicit einsum
template<class Index_I, class Index_O,
        typename AbstractTensorType0, enable_if_t_<!is_tensor_v<AbstractTensorType0>,bool> = false>
FASTOR_INLINE
decltype(auto)
einsum(const AbstractTensorType0& a)
{
    decltype(auto) tmp = evaluate(a);
    auto res = einsum<Index_I>(tmp);
    using resulting_index_einsum = typename einsum_helper<Index_I,decltype(tmp)>::resulting_index;

    using mapped_index = internal::permute_mapped_index_t<resulting_index_einsum,typename Index_O::parent_type>;
    constexpr bool requires_permutation = requires_permute_v<mapped_index, decltype(res)>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<mapped_index>(res);
}

// pair expression explicit einsum
template<class Index_I, class Index_J, class Index_O,
        typename Derived0, typename Derived1, size_t DIM0, size_t DIM1>
FASTOR_INLINE
decltype(auto)
einsum(const AbstractTensor<Derived0,DIM0>& a, const AbstractTensor<Derived1,DIM1>& b)
{
    auto res = einsum<Index_I,Index_J>(a.self(),b.self());
    using ttype0 = typename Derived0::resulting_type;
    using ttype1 = typename Derived1::resulting_type;
    using resulting_index_einsum = typename einsum_helper<Index_I,Index_J,ttype0,ttype1>::resulting_index;

    using mapped_index = internal::permute_mapped_index_t<resulting_index_einsum,typename Index_O::parent_type>;
    constexpr bool requires_permutation = requires_permute_v<mapped_index, decltype(res)>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<mapped_index>(res);
}

// 3 tensor network expression explicit einsum
template<class Index_I, class Index_J, class Index_K, class Index_O,
        typename Derived0, typename Derived1, typename Derived2, size_t DIM0, size_t DIM1, size_t DIM2>
FASTOR_INLINE
decltype(auto)
einsum(const AbstractTensor<Derived0,DIM0>& a, const AbstractTensor<Derived1,DIM1>& b, const AbstractTensor<Derived2,DIM2>& c)
{
    auto res = einsum<Index_I,Index_J,Index_K>(a.self(),b.self(),c.self());
    using ttype0 = typename Derived0::resulting_type;
    using ttype1 = typename Derived1::resulting_type;
    using ttype2 = typename Derived2::resulting_type;
    using resulting_index_einsum = typename einsum_helper<Index_I,Index_J,Index_K,ttype0,ttype1,ttype2>::resulting_index;

    using mapped_index = internal::permute_mapped_index_t<resulting_index_einsum,typename Index_O::parent_type>;
    constexpr bool requires_permutation = requires_permute_v<mapped_index, decltype(res)>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<mapped_index>(res);
}

// 4 tensor network expression explicit einsum
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_O,
        typename Derived0, typename Derived1, typename Derived2, typename Derived3,
        size_t DIM0, size_t DIM1, size_t DIM2, size_t DIM3>
FASTOR_INLINE
decltype(auto)
einsum(
        const AbstractTensor<Derived0,DIM0>& a, const AbstractTensor<Derived1,DIM1>& b,
        const AbstractTensor<Derived2,DIM2>& c, const AbstractTensor<Derived3,DIM3>& d)
{
    auto res = einsum<Index_I,Index_J,Index_K,Index_L>(a.self(),b.self(),c.self(),d.self());
    using ttype0 = typename Derived0::resulting_type;
    using ttype1 = typename Derived1::resulting_type;
    using ttype2 = typename Derived2::resulting_type;
    using ttype3 = typename Derived3::resulting_type;
    using resulting_index_einsum = typename einsum_helper<Index_I,Index_J,Index_K,Index_L,ttype0,ttype1,ttype2,ttype3>::resulting_index;

    using mapped_index = internal::permute_mapped_index_t<resulting_index_einsum,typename Index_O::parent_type>;
    constexpr bool requires_permutation = requires_permute_v<mapped_index, decltype(res)>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<mapped_index>(res);
}

// 5 tensor network expression explicit einsum
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, class Index_O,
        typename Derived0, typename Derived1, typename Derived2, typename Derived3, typename Derived4,
        size_t DIM0, size_t DIM1, size_t DIM2, size_t DIM3, size_t DIM4>
FASTOR_INLINE
decltype(auto)
einsum(
        const AbstractTensor<Derived0,DIM0>& a, const AbstractTensor<Derived1,DIM1>& b,
        const AbstractTensor<Derived2,DIM2>& c, const AbstractTensor<Derived3,DIM3>& d,
        const AbstractTensor<Derived4,DIM4>& e)
{
    auto res = einsum<Index_I,Index_J,Index_K,Index_L,Index_M>(a.self(),b.self(),c.self(),d.self(),e.self());
    using ttype0 = typename Derived0::resulting_type;
    using ttype1 = typename Derived1::resulting_type;
    using ttype2 = typename Derived2::resulting_type;
    using ttype3 = typename Derived3::resulting_type;
    using ttype4 = typename Derived4::resulting_type;
    using resulting_index_einsum = typename einsum_helper<Index_I,Index_J,Index_K,Index_L,Index_M,
        ttype0,ttype1,ttype2,ttype3,ttype4>::resulting_index;

    using mapped_index = internal::permute_mapped_index_t<resulting_index_einsum,typename Index_O::parent_type>;
    constexpr bool requires_permutation = requires_permute_v<mapped_index, decltype(res)>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<mapped_index>(res);
}

// 6 tensor network expression explicit einsum
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, class Index_N, class Index_O,
        typename Derived0, typename Derived1, typename Derived2, typename Derived3, typename Derived4, typename Derived5,
        size_t DIM0, size_t DIM1, size_t DIM2, size_t DIM3, size_t DIM4, size_t DIM5>
FASTOR_INLINE
decltype(auto)
einsum(
        const AbstractTensor<Derived0,DIM0>& a, const AbstractTensor<Derived1,DIM1>& b,
        const AbstractTensor<Derived2,DIM2>& c, const AbstractTensor<Derived3,DIM3>& d,
        const AbstractTensor<Derived4,DIM4>& e, const AbstractTensor<Derived5,DIM5>& f)
{
    auto res = einsum<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>(a.self(),b.self(),c.self(),d.self(),e.self(),f.self());
    using ttype0 = typename Derived0::resulting_type;
    using ttype1 = typename Derived1::resulting_type;
    using ttype2 = typename Derived2::resulting_type;
    using ttype3 = typename Derived3::resulting_type;
    using ttype4 = typename Derived4::resulting_type;
    using ttype5 = typename Derived5::resulting_type;
    using resulting_index_einsum = typename einsum_helper<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,
        ttype0,ttype1,ttype2,ttype3,ttype4,ttype5>::resulting_index;

    using mapped_index = internal::permute_mapped_index_t<resulting_index_einsum,typename Index_O::parent_type>;
    constexpr bool requires_permutation = requires_permute_v<mapped_index, decltype(res)>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<mapped_index>(res);
}

#if !defined(FASTOR_MSVC)
template<class Index_I, class ... Index_Ks, class Index_O,
        typename AbstractTensorType0, typename ... AbstractTensorTypes,
        enable_if_t_<sizeof...(AbstractTensorTypes) >= 6, bool > = false>
FASTOR_INLINE
decltype(auto)
einsum(const AbstractTensorType0& a, const AbstractTensorTypes& ... rest)
{
    auto res = internal::unpack_einsum_tuple<Index_I,Index_Ks...>::apply(internal::contraction_chain_evaluate(a,rest...));
    auto res_idx = internal::unpack_einsum_helper_tuple<Index_I,Index_Ks...>::apply(internal::contraction_chain_evaluate(a,rest...));

    using mapped_index = internal::permute_mapped_index_t<decltype(res_idx),typename Index_O::parent_type>;
    constexpr bool requires_permutation = requires_permute_v<mapped_index, decltype(res)>;
    FASTOR_IF_CONSTEXPR(!requires_permutation) return res;
    return permute<mapped_index>(res);
}
#endif
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------


#endif // CXX 2017

} // end of namespace


#endif // EXPLICIT_EINSUM_H
