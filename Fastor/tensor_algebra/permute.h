#ifndef PERMUTE_H
#define PERMUTE_H

#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/meta/einsum_meta.h"
#include "Fastor/tensor_algebra/indicial.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"

namespace Fastor {

namespace internal {

template<class Idx, class Tens, size_t ... Args>
struct NewRecursiveCartesianPerm;

template<typename T, size_t ...Idx, size_t ...Rest, size_t First, size_t ... Lasts>
struct NewRecursiveCartesianPerm<Index<Idx...>, Tensor<T,Rest...>, First, Lasts...> {

    static constexpr size_t out_dim = sizeof...(Rest);
    static
    FASTOR_INLINE
    void Do(const T *a_data, T *out_data, std::array<size_t,out_dim> &as, std::array<size_t,out_dim> &idx) {
        for (size_t i=0; i<First; ++i) {
            idx[sizeof...(Lasts)] = i;
            NewRecursiveCartesianPerm<Index<Idx...>, Tensor<T,Rest...>,Lasts...>::Do(a_data, out_data, as, idx);
        }
    }
};

template<typename T, size_t Last, size_t ...Idx, size_t ...Rest>
struct NewRecursiveCartesianPerm<Index<Idx...>, Tensor<T,Rest...>,Last>
{
    using _permute_impl = new_permute_impl<Index<Idx...>, Tensor<T,Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Idx)>::type>;
    using resulting_tensor = typename _permute_impl::resulting_tensor;
    using resulting_index  = typename _permute_impl::resulting_index;
    using maxes_out_type = typename put_dims_in_Index<resulting_tensor>::type;

    static constexpr size_t a_dim   = sizeof...(Rest);
    static constexpr size_t out_dim = a_dim;
#if FASTOR_CXX_VERSION >= 2017
    using reverse_map = internal::permute_mapped_index_t<Index<Idx...>,make_index_t<a_dim>>;
    static constexpr std::array<size_t,sizeof...(Rest)> maxes_idx = reverse_map::values;
#else
    static constexpr std::array<size_t,sizeof...(Rest)> maxes_idx = resulting_index::values;
#endif

    static constexpr std::array<size_t,sizeof...(Rest)> products_a   = nprods<Index<Rest...>,
        typename std_ext::make_index_sequence<a_dim>::type>::values;
    static constexpr std::array<size_t,sizeof...(Rest)> products_out = nprods<maxes_out_type,
        typename std_ext::make_index_sequence<a_dim>::type>::values;

    static
    FASTOR_INLINE
    void Do(const T *a_data, T *out_data, std::array<size_t,out_dim> &as, std::array<size_t,out_dim> &idx)
    {
        constexpr size_t stride = 1;
        for (size_t i = 0; i < Last; i+=stride) {
            idx[0] = i;
            std::reverse_copy(idx.begin(),idx.end(),as.begin());

#if FASTOR_CXX_VERSION >= 2017
            size_t index_a = as[maxes_idx[a_dim-1]];
            for(size_t it = 0; it< a_dim; it++) {
                index_a += products_a[it]*as[maxes_idx[it]];
            }
            size_t index_out = as[out_dim-1];
            for(size_t it = 0; it< out_dim-1; it++) {
                index_out += products_out[it]*as[it];
            }
#else
            size_t index_a = as[a_dim-1];
            for(size_t it = 0; it< a_dim; it++) {
                index_a += products_a[it]*as[it];
            }
            size_t index_out = as[maxes_idx[out_dim-1]];
            for(size_t it = 0; it< out_dim-1; it++) {
                index_out += products_out[it]*as[maxes_idx[it]];
            }
#endif

            out_data[index_out] = a_data[index_a];
        }
    }
};

template<typename T, size_t Last, size_t ...Idx, size_t ...Rest>
constexpr std::array<size_t,sizeof...(Rest)> NewRecursiveCartesianPerm<Index<Idx...>,
  Tensor<T,Rest...>,Last>::maxes_idx;

template<typename T, size_t Last, size_t ...Idx, size_t ...Rest>
constexpr std::array<size_t,sizeof...(Rest)> NewRecursiveCartesianPerm<Index<Idx...>,
  Tensor<T,Rest...>,Last>::products_a;

template<typename T, size_t Last, size_t ...Idx, size_t ...Rest>
constexpr std::array<size_t,sizeof...(Rest)> NewRecursiveCartesianPerm<Index<Idx...>,
  Tensor<T,Rest...>,Last>::products_out;



template<class Idx, class Tens, class Args>
struct NewRecursiveCartesianPermDispatcher;

template<typename T, size_t ...Idx, size_t ...Rest, size_t ... Args>
struct NewRecursiveCartesianPermDispatcher<Index<Idx...>, Tensor<T,Rest...>, Index<Args...> >
{
    static constexpr size_t out_dim =  sizeof...(Rest);

    static FASTOR_INLINE void Do(const T *a_data, T *out_data,
      std::array<size_t,out_dim> &as, std::array<size_t,out_dim> &idx) {
      return NewRecursiveCartesianPerm<Index<Idx...>,Tensor<T,Rest...>, Args...>::Do(a_data, out_data, as, idx);
    }
};




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
        using resulting_tensor = typename _permute_impl::resulting_tensor;
        constexpr bool requires_permutation = _permute_impl::requires_permutation;
        FASTOR_IF_CONSTEXPR(!requires_permutation) return a;

#if CONTRACT_OPT==-1

        using maxes_out_type = typename put_dims_in_Index<resulting_tensor>::type;

        constexpr size_t a_dim = sizeof...(Rest);
        constexpr size_t out_dim = a_dim;

        constexpr auto& products_a = nprods<Index<Rest...>,
            typename std_ext::make_index_sequence<a_dim>::type>::values;
        constexpr auto& products_out = nprods<maxes_out_type,
            typename std_ext::make_index_sequence<a_dim>::type>::values;

        resulting_tensor out;

        T *a_data = a.data();
        T *out_data = out.data();

        size_t as[out_dim] = {};
        int jt;

#if FASTOR_CXX_VERSION >= 2017
        constexpr std::array<size_t,a_dim> maxes_out = maxes_out_type::values;
        // Map to go from out to in
        // Get the reverse map - this is to get contiguous memory writes
        using reverse_map = internal::permute_mapped_index_t<Index<Idx...>,make_index_t<a_dim>>;
        constexpr auto& maxes_idx = reverse_map::values;
        // print(type_name<mapped_index>());

        while(true)
        {
            size_t index_a = as[maxes_idx[a_dim-1]];
            for(size_t it = 0; it< a_dim; it++) {
                index_a += products_a[it]*as[maxes_idx[it]];
            }
            size_t index_out = as[out_dim-1];
            for(size_t it = 0; it< out_dim-1; it++) {
                index_out += products_out[it]*as[it];
            }
            // print(index_out);
            // print(index_a);

            out_data[index_out] = a_data[index_a];

            for(jt = out_dim-1 ; jt>=0 ; jt--)
            {
                if(++as[jt]<maxes_out[jt])
                    break;
                else
                    as[jt]=0;
            }
            if(jt<0)
                break;
        }
#else
        using resulting_index  = typename _permute_impl::resulting_index;
        constexpr std::array<size_t,a_dim> maxes_a   = {Rest...};
        // Map to go from in to out
        constexpr auto& maxes_idx = resulting_index::values;

        while(true)
        {
            size_t index_a = as[a_dim-1];
            for(size_t it = 0; it< a_dim; it++) {
                index_a += products_a[it]*as[it];
            }
            size_t index_out = as[maxes_idx[out_dim-1]];
            for(size_t it = 0; it< out_dim-1; it++) {
                index_out += products_out[it]*as[maxes_idx[it]];
            }

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
#endif
#else
        resulting_tensor out;

        const T *a_data = a.data();
        T *out_data     = out.data();

        constexpr size_t a_dim   =  sizeof...(Rest);
        constexpr size_t out_dim =  sizeof...(Rest);

        std::array<size_t,out_dim> as  = {};
        std::array<size_t,out_dim> idx = {};

#if FASTOR_CXX_VERSION >= 2017
        using reverse_map = internal::permute_mapped_index_t<Index<Idx...>,make_index_t<a_dim>>;
        using nloops = loop_setter<
                reverse_map,
                resulting_tensor,
                typename std_ext::make_index_sequence<out_dim>::type>;
        using dims_type = typename nloops::dims_type;

        NewRecursiveCartesianPermDispatcher<Index<Idx...>,Tensor<T,Rest...>,dims_type>::Do(a_data,out_data,as,idx);
#else
        using nloops = loop_setter<
                Index<Idx...>,
                Tensor<T,Rest...>,
                typename std_ext::make_index_sequence<out_dim>::type>;
        using dims_type = typename nloops::dims_type;

        NewRecursiveCartesianPermDispatcher<Index<Idx...>,Tensor<T,Rest...>,dims_type>::Do(a_data,out_data,as,idx);
#endif

#endif
        return out;

    }


    // Abstract permutation
    template<typename Derived, size_t DIMS,
        enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
    static
    FASTOR_INLINE
    typename new_permute_impl<
        Index<Idx...>, typename Derived::result_type,
        typename std_ext::make_index_sequence<sizeof...(Idx)>::type>::resulting_tensor
    permutation_impl(const AbstractTensor<Derived,DIMS> &a) {

        using T           = typename Derived::scalar_type;
        using tensor_type = typename Derived::result_type;

        using _permute_impl = new_permute_impl<Index<Idx...>, tensor_type,
            typename std_ext::make_index_sequence<sizeof...(Idx)>::type>;
        using resulting_tensor = typename _permute_impl::resulting_tensor;
        constexpr bool requires_permutation = _permute_impl::requires_permutation;
        FASTOR_IF_CONSTEXPR(!requires_permutation) return a;

        using maxes_out_type = typename put_dims_in_Index<resulting_tensor>::type;

        constexpr size_t a_dim   = DIMS;
        constexpr size_t out_dim = a_dim;

        constexpr auto& products_a = nprods<typename get_tensor_dimensions<tensor_type>::tensor_to_index,
            typename std_ext::make_index_sequence<a_dim>::type>::values;
        constexpr auto& products_out = nprods<maxes_out_type,
            typename std_ext::make_index_sequence<a_dim>::type>::values;

        resulting_tensor out;
        T *out_data = out.data();
        const Derived & a_src = a.self();

        size_t as[out_dim] = {};
        int jt;

#if FASTOR_CXX_VERSION >= 2017
        constexpr std::array<size_t,a_dim> maxes_out = maxes_out_type::values;
        // Map to go from in to out
        // constexpr auto& maxes_idx = resulting_index::values;
        // Map to go from out to in
        // Get the reverse map - this is to get contiguous memory writes
        using reverse_map = internal::permute_mapped_index_t<Index<Idx...>,make_index_t<a_dim>>;
        constexpr auto& maxes_idx = reverse_map::values;

        while(true)
        {
            size_t index_a = as[maxes_idx[a_dim-1]];
            for(size_t it = 0; it< a_dim; it++) {
                index_a += products_a[it]*as[maxes_idx[it]];
            }
            size_t index_out = as[out_dim-1];
            for(size_t it = 0; it< out_dim-1; it++) {
                index_out += products_out[it]*as[it];
            }

            out_data[index_out] = a_src.template eval_s<T>(index_a);

            for(jt = out_dim-1 ; jt>=0 ; jt--)
            {
                if(++as[jt]<maxes_out[jt])
                    break;
                else
                    as[jt]=0;
            }
            if(jt<0)
                break;
        }
#else
        constexpr std::array<size_t,a_dim> maxes_a = get_tensor_dimensions<tensor_type>::dims;
        // Map to go from in to out
        using resulting_index  = typename _permute_impl::resulting_index;
        constexpr auto& maxes_idx = resulting_index::values;

        while(true)
        {
            size_t index_a = as[a_dim-1];
            for(size_t it = 0; it< a_dim; it++) {
                index_a += products_a[it]*as[it];
            }
            size_t index_out = as[maxes_idx[out_dim-1]];
            for(size_t it = 0; it< out_dim-1; it++) {
                index_out += products_out[it]*as[maxes_idx[it]];
            }

            out_data[index_out] = a_src.template eval_s<T>(index_a);

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
#endif
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


} // end of namespace Fastor

#endif // PERMUTE_H
