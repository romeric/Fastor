#ifndef PERMUTE_H
#define PERMUTE_H

#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/meta/einsum_meta.h"
#include "Fastor/tensor_algebra/indicial.h"
#include "Fastor/tensor_algebra/permutation.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"

namespace Fastor {

namespace internal {

template<typename T, class List, class Tensor, class Seq>
struct new_permute_impl;

template<typename T, size_t ... ls, size_t ... fs, size_t... ss>
struct new_permute_impl<T,Index<ls...>, Tensor<T, fs...>, std_ext::index_sequence<ss...>>{
    constexpr static size_t lst[sizeof...(ls)] = { ls... };
    constexpr static size_t fvals[sizeof...(ls)] = {fs...};
    using type = Tensor<T,fvals[count_less(lst, lst[ss])]...>;
    constexpr static size_t aranger[sizeof...(ss)] = { ss... };
    using index_type = Index<aranger[count_less(lst, lst[ss])]...>;
};

template<class T>
struct new_extractor_perm {};

template<size_t ... Idx>
struct new_extractor_perm<Index<Idx...> > {

    template<typename T, size_t ... Rest>
    static
    FASTOR_INLINE
        typename new_permute_impl<T,Index<Idx...>, Tensor<T,Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Idx)>::type>::type
    permutation_impl(const Tensor<T,Rest...> &a) {

        using OutTensor = typename new_permute_impl<T,Index<Idx...>, Tensor<T,Rest...>,
            typename std_ext::make_index_sequence<sizeof...(Idx)>::type>::type;
        using maxes_out_type = typename put_dims_in_Index<OutTensor>::type;
        constexpr auto& maxes_idx = new_permute_impl<T,Index<Idx...>, Tensor<T,Rest...>,
            typename std_ext::make_index_sequence<sizeof...(Idx)>::type>::index_type::_IndexHolder;

        constexpr int a_dim = sizeof...(Rest);
        constexpr int out_dim = a_dim;
        constexpr std::array<int,a_dim> maxes_a = {Rest...};

        constexpr auto& products_a = nprods<Index<Rest...>,
            typename std_ext::make_index_sequence<a_dim>::type>::values;
        constexpr auto& products_out = nprods<maxes_out_type,
            typename std_ext::make_index_sequence<a_dim>::type>::values;
        // println(maxes_a);
        // print(out_dim);
        // constexpr std::array<size_t,3> maxes_idx = {2,0,1};
        // println<size_t,3>(maxes_idx);
        // println(products_a);
        // println(products_out);
        // print(type_name<maxes_out_type>());
        // print(type_name<OutTensor>());

        OutTensor out;
        // out.zeros();

        T *a_data = a.data();
        T *out_data = out.data();

        int as[out_dim] = {};
        int it,jt;

        while(true)
        {
            int index_a = as[a_dim-1];
            for(it = 0; it< a_dim; it++) {
                index_a += products_a[it]*as[it];
            }
            int index_out = as[maxes_idx[out_dim-1]];
            for(it = 0; it< out_dim-1; it++) {
                index_out += products_out[it]*as[maxes_idx[it]];
            }
            // print(index_out);

            out_data[index_out] = a_data[index_a];

            for(jt = out_dim-1 ; jt>=0 ; jt--)
            {
                if(++as[jt]<maxes_a[jt])
                    break;
                else
                    as[jt]=0;
            }
            if(jt<0)
                break;
        }

        return out;
    }
};

} // internal



template<class Index_I, typename T, size_t ... Rest>
FASTOR_INLINE
typename internal::new_permute_impl<T,Index_I, Tensor<T,Rest...>,
    typename std_ext::make_index_sequence<sizeof...(Rest)>::type>::type
permute(const Tensor<T, Rest...> &a) {
    return internal::new_extractor_perm<Index_I>::permutation_impl(a);
}

template<class Index_I, typename Derived, size_t DIMS,
    enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE
typename internal::new_permute_impl<typename scalar_type_finder<Derived>::type,Index_I,
    typename Derived::result_type,
    typename std_ext::make_index_sequence<DIMS>::type>::type
permute(const AbstractTensor<Derived, DIMS> &a) {
    return internal::new_extractor_perm<Index_I>::permutation_impl(a);
}

template<class Index_I, typename Derived, size_t DIMS,
    enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE
typename internal::new_permute_impl<typename scalar_type_finder<Derived>::type,Index_I,
    typename Derived::result_type,
    typename std_ext::make_index_sequence<DIMS>::type>::type
permute(const AbstractTensor<Derived, DIMS> &a) {
    using result_type = typename Derived::result_type;
    const result_type tmp(a);
    return internal::new_extractor_perm<Index_I>::permutation_impl(tmp);
}


}
#endif // PERMUTE_H
