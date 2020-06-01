#ifndef PERMUTATION_H
#define PERMUTATION_H

#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/tensor_algebra/indicial.h"
#include "Fastor/meta/einsum_meta.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"

namespace Fastor {

namespace internal {

template<class Idx, class Tens, size_t ... Args>
struct RecursiveCartesianPerm;

template<typename T, size_t ...Idx, size_t ...Rest, size_t First, size_t ... Lasts>
struct RecursiveCartesianPerm<Index<Idx...>, Tensor<T,Rest...>, First, Lasts...> {

    static constexpr size_t out_dim = sizeof...(Rest);
    static
    FASTOR_INLINE
    void Do(const T *a_data, T *out_data, std::array<size_t,out_dim> &as, std::array<size_t,out_dim> &idx) {
        for (size_t i=0; i<First; ++i) {
            idx[sizeof...(Lasts)] = i;
            RecursiveCartesianPerm<Index<Idx...>, Tensor<T,Rest...>,Lasts...>::Do(a_data, out_data, as, idx);
        }
    }
};

template<typename T, size_t Last, size_t ...Idx, size_t ...Rest>
struct RecursiveCartesianPerm<Index<Idx...>, Tensor<T,Rest...>,Last>
{
    using _permute_impl = permute_impl<Index<Idx...>, Tensor<T,Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Idx)>::type>;
    using resulting_tensor = typename _permute_impl::resulting_tensor;
    using resulting_index  = typename _permute_impl::resulting_index;
    using maxes_out_type   = typename _permute_impl::maxes_out_type;
    static constexpr std::array<size_t,sizeof...(Rest)> maxes_idx = resulting_index::values;
    static constexpr std::array<size_t,sizeof...(Rest)> maxes_out = maxes_out_type::values;

    static constexpr size_t a_dim = sizeof...(Rest);
    static constexpr size_t out_dim = a_dim;
    static constexpr std::array<size_t,a_dim> maxes_a = {Rest...};

    static constexpr std::array<size_t,sizeof...(Rest)> products_a = nprods<Index<Rest...>,
        typename std_ext::make_index_sequence<a_dim>::type>::values;
    static constexpr std::array<size_t,sizeof...(Rest)> products_out = nprods<maxes_out_type,
        typename std_ext::make_index_sequence<a_dim>::type>::values;

    static
    FASTOR_INLINE
    void Do(const T *a_data, T *out_data, std::array<size_t,out_dim> &as, std::array<size_t,out_dim> &idx)
    {
        constexpr size_t stride = 1;
        for (size_t i=0; i<Last; i+=stride) {
            idx[0] = i;
            std::reverse_copy(idx.begin(),idx.end(),as.begin());

            size_t index_a = as[a_dim-1];
            for(size_t it = 0; it< a_dim; it++) {
                index_a += products_a[it]*as[it];
            }
            size_t index_out = as[maxes_idx[out_dim-1]];
            for(size_t it = 0; it< out_dim-1; it++) {
                index_out += products_out[it]*as[maxes_idx[it]];
            }

            out_data[index_out] = a_data[index_a];
        }
    }
};

template<typename T, size_t Last, size_t ...Idx, size_t ...Rest>
constexpr std::array<size_t,sizeof...(Rest)> RecursiveCartesianPerm<Index<Idx...>,
  Tensor<T,Rest...>,Last>::maxes_idx;

template<typename T, size_t Last, size_t ...Idx, size_t ...Rest>
constexpr std::array<size_t,sizeof...(Rest)> RecursiveCartesianPerm<Index<Idx...>,
  Tensor<T,Rest...>,Last>::maxes_out;

template<typename T, size_t Last, size_t ...Idx, size_t ...Rest>
constexpr std::array<size_t,sizeof...(Rest)> RecursiveCartesianPerm<Index<Idx...>,
  Tensor<T,Rest...>,Last>::products_a;

template<typename T, size_t Last, size_t ...Idx, size_t ...Rest>
constexpr std::array<size_t,sizeof...(Rest)> RecursiveCartesianPerm<Index<Idx...>,
  Tensor<T,Rest...>,Last>::products_out;



// template<typename T, size_t ...Idx, size_t ...Rest>
// struct RecursiveCartesianPerm<Index<Idx...>, Tensor<T,Rest...>>
// {
//     using OutTensor = typename permute_impl<Index<Idx...>, Tensor<T,Rest...>,
//         typename std_ext::make_index_sequence<sizeof...(Idx)>::type>::type;
//     using maxes_out_type = typename permute_impl<Index<Idx...>, Tensor<T,Rest...>,
//         typename std_ext::make_index_sequence<sizeof...(Idx)>::type>::maxes_out_type;
//     using index_type = typename permute_impl<Index<Idx...>, Tensor<T,Rest...>,
//         typename std_ext::make_index_sequence<sizeof...(Idx)>::type>::index_type;
//     static constexpr auto maxes_idx = permute_impl<Index<Idx...>, Tensor<T,Rest...>,
//         typename std_ext::make_index_sequence<sizeof...(Idx)>::type>::index_type::values;
//     static constexpr auto maxes_out = permute_impl<Index<Idx...>, Tensor<T,Rest...>,
//         typename std_ext::make_index_sequence<sizeof...(Idx)>::type>::maxes_out_type::values;

//     static constexpr int a_dim = sizeof...(Rest);
//     static constexpr int out_dim = a_dim;
//     static constexpr std::array<int,a_dim> maxes_a = {Rest...};

//     static constexpr std::array<size_t,sizeof...(Rest)> products_a = nprods<Index<Rest...>,
//         typename std_ext::make_index_sequence<a_dim>::type>::values;
//     static constexpr std::array<size_t,sizeof...(Rest)> products_out = nprods<maxes_out_type,
//         typename std_ext::make_index_sequence<a_dim>::type>::values;

//     static void Do(const T *a_data, T *out_data, std::array<int,out_dim> &as, std::array<int,out_dim> &idx)
//     {
//         std::reverse_copy(idx.begin(),idx.end(),as.begin());

//         int index_a = as[a_dim-1];
//         for(int it = 0; it< a_dim; it++) {
//             index_a += products_a[it]*as[it];
//         }
//         int index_out = as[maxes_idx[out_dim-1]];
//         for(int it = 0; it< out_dim-1; it++) {
//             index_out += products_out[it]*as[maxes_idx[it]];
//         }

//         out_data[index_out] = a_data[index_a];
//     }
// };

// template<typename T, size_t ...Idx, size_t ...Rest>
// constexpr std::array<size_t,sizeof...(Rest)> RecursiveCartesianPerm<Index<Idx...>,
//   Tensor<T,Rest...>>::products_a;

// template<typename T, size_t ...Idx, size_t ...Rest>
// constexpr std::array<size_t,sizeof...(Rest)> RecursiveCartesianPerm<Index<Idx...>,
//   Tensor<T,Rest...>>::products_out;



template<class Idx, class Tens, class Args>
struct RecursiveCartesianPermDispatcher;

template<typename T, size_t ...Idx, size_t ...Rest, size_t ... Args>
struct RecursiveCartesianPermDispatcher<Index<Idx...>, Tensor<T,Rest...>, Index<Args...> >
{
    static constexpr size_t out_dim =  sizeof...(Rest);

    static FASTOR_INLINE void Do(const T *a_data, T *out_data,
      std::array<size_t,out_dim> &as, std::array<size_t,out_dim> &idx) {
      return RecursiveCartesianPerm<Index<Idx...>,Tensor<T,Rest...>, Args...>::Do(a_data, out_data, as, idx);
    }
};






template<class T>
struct extractor_perm {};

template<size_t ... Idx>
struct extractor_perm<Index<Idx...> > {

    template<typename T, size_t ... Rest>
    static
    FASTOR_INLINE
        typename permute_impl<Index<Idx...>, Tensor<T,Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Idx)>::type>::resulting_tensor
    permutation_impl(const Tensor<T,Rest...> &a) {

        using _permute_impl = permute_impl<Index<Idx...>, Tensor<T,Rest...>,
            typename std_ext::make_index_sequence<sizeof...(Idx)>::type>;
        using resulting_tensor = typename _permute_impl::resulting_tensor;
        constexpr bool requires_permutation = _permute_impl::requires_permutation;
        FASTOR_IF_CONSTEXPR(!requires_permutation) return a;

#if CONTRACT_OPT==-1

        using resulting_index  = typename _permute_impl::resulting_index;
        using maxes_out_type = typename _permute_impl::maxes_out_type;
        constexpr auto& maxes_idx = resulting_index::values;
        constexpr auto& maxes_out = maxes_out_type::values;

        constexpr int a_dim = sizeof...(Rest);
        constexpr int out_dim = a_dim;
        constexpr std::array<int,a_dim> maxes_a = {Rest...};

        constexpr auto& products_a = nprods<Index<Rest...>,
            typename std_ext::make_index_sequence<a_dim>::type>::values;
        constexpr auto& products_out = nprods<maxes_out_type,
            typename std_ext::make_index_sequence<a_dim>::type>::values;

        resulting_tensor out;
        out.zeros();

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

#else

        resulting_tensor out;

        const T *a_data = a.data();
        T *out_data     = out.data();

        constexpr size_t out_dim =  sizeof...(Rest);

        std::array<size_t,out_dim> as  = {};
        std::array<size_t,out_dim> idx = {};

        using nloops = loop_setter<
                Index<Idx...>,
                Tensor<T,Rest...>,
                typename std_ext::make_index_sequence<out_dim>::type>;
        using dims_type = typename nloops::dims_type;

        RecursiveCartesianPermDispatcher<Index<Idx...>,Tensor<T,Rest...>,dims_type>::Do(a_data,out_data,as,idx);

#endif
        return out;
    }



    // Abstract permutation
    template<typename Derived, size_t DIMS,
        enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
    static
    FASTOR_INLINE
    typename permute_impl<
        Index<Idx...>, typename Derived::result_type,
        typename std_ext::make_index_sequence<sizeof...(Idx)>::type>::resulting_tensor
    permutation_impl(const AbstractTensor<Derived,DIMS> &a) {

        using T           = typename Derived::scalar_type;
        using tensor_type = typename Derived::result_type;

        using _permute_impl = permute_impl<Index<Idx...>, tensor_type,
            typename std_ext::make_index_sequence<sizeof...(Idx)>::type>;
        using resulting_tensor = typename _permute_impl::resulting_tensor;
        using resulting_index  = typename _permute_impl::resulting_index;
        using maxes_out_type   = typename _permute_impl::maxes_out_type;
        constexpr auto& maxes_idx = resulting_index::values;

        constexpr int a_dim = DIMS;
        constexpr int out_dim = a_dim;
        constexpr std::array<int,a_dim> maxes_a = get_tensor_dimensions<tensor_type>::dims_int;

        constexpr auto& products_a = nprods<typename get_tensor_dimensions<tensor_type>::tensor_to_index,
            typename std_ext::make_index_sequence<a_dim>::type>::values;
        constexpr auto& products_out = nprods<maxes_out_type,
            typename std_ext::make_index_sequence<a_dim>::type>::values;

        resulting_tensor out;
        out.zeros();
        T *out_data = out.data();
        const Derived & a_src = a.self();

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

        return out;
    }
};

} // internal



template<class Index_I, typename T, size_t ... Rest>
FASTOR_INLINE
typename internal::permute_impl<Index_I, Tensor<T,Rest...>,
    typename std_ext::make_index_sequence<sizeof...(Rest)>::type>::resulting_tensor
permutation(const Tensor<T, Rest...> &a) {
    return internal::extractor_perm<Index_I>::permutation_impl(a);
}

template<class Index_I, typename Derived, size_t DIMS,
    enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE
typename internal::permute_impl<Index_I,
    typename Derived::result_type,
    typename std_ext::make_index_sequence<DIMS>::type>::resulting_tensor
permutation(const AbstractTensor<Derived, DIMS> &a) {
    return internal::extractor_perm<Index_I>::permutation_impl(a.self());
}

template<class Index_I, typename Derived, size_t DIMS,
    enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE
typename internal::permute_impl<Index_I,
    typename Derived::result_type,
    typename std_ext::make_index_sequence<DIMS>::type>::resulting_tensor
permutation(const AbstractTensor<Derived, DIMS> &a) {
    using result_type = typename Derived::result_type;
    const result_type tmp(a);
    return internal::extractor_perm<Index_I>::permutation_impl(tmp);
}

} // end of namespace Fastor

#endif // PERMUTATION_H
