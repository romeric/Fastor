#ifndef PERMUTE_H
#define PERMUTE_H

#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/meta/einsum_meta.h"
#include "Fastor/tensor_algebra/indicial.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"

namespace Fastor {

namespace internal {

template<class T>
struct new_extractor_perm {};

template<size_t ... Idx>
struct new_extractor_perm<Index<Idx...> > {

    template<typename T, size_t ... Rest>
    static
    FASTOR_INLINE
        typename new_permute_impl<Index<Idx...>, Tensor<T,Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Idx)>::type>::resulting_tensor
    permutation_impl(const Tensor<T,Rest...> &a) {

        using _permute_impl = new_permute_impl<Index<Idx...>, Tensor<T,Rest...>,
            typename std_ext::make_index_sequence<sizeof...(Idx)>::type>;
        using resulting_index  = typename _permute_impl::resulting_index;
        using resulting_tensor = typename _permute_impl::resulting_tensor;
        constexpr bool requires_permutation = _permute_impl::requires_permutation;

        FASTOR_IF_CONSTEXPR(!requires_permutation) return a;

        constexpr auto& maxes_idx = resulting_index::values;
        using maxes_out_type = typename put_dims_in_Index<resulting_tensor>::type;

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
        // print(requires_permutation);

        resulting_tensor out;
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
typename internal::new_permute_impl<Index_I, Tensor<T,Rest...>,
    typename std_ext::make_index_sequence<sizeof...(Rest)>::type>::resulting_tensor
permute(const Tensor<T, Rest...> &a) {
    return internal::new_extractor_perm<Index_I>::permutation_impl(a);
}

template<class Index_I, typename Derived, size_t DIMS,
    enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE
typename internal::new_permute_impl<Index_I,
    typename Derived::result_type,
    typename std_ext::make_index_sequence<DIMS>::type>::resulting_tensor
permute(const AbstractTensor<Derived, DIMS> &a) {
    return internal::new_extractor_perm<Index_I>::permutation_impl(a);
}

template<class Index_I, typename Derived, size_t DIMS,
    enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE
typename internal::new_permute_impl<Index_I,
    typename Derived::result_type,
    typename std_ext::make_index_sequence<DIMS>::type>::resulting_tensor
permute(const AbstractTensor<Derived, DIMS> &a) {
    using result_type = typename Derived::result_type;
    const result_type tmp(a);
    return internal::new_extractor_perm<Index_I>::permutation_impl(tmp);
}


}
#endif // PERMUTE_H
