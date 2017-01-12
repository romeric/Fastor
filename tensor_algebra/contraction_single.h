#ifndef CONTRACTION_SINGLE_H
#define CONTRACTION_SINGLE_H


#include "tensor/Tensor.h"
#include "indicial.h"

namespace Fastor {


template<class T>
struct extractor_contract_1 {};

template<size_t ... Idx0>
struct extractor_contract_1<Index<Idx0...>> {



template<typename T, size_t ... Rest0>
  static
  typename contraction_impl<Index<Idx0...>, Tensor<T,Rest0...>,
           typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::type
  contract_impl(const Tensor<T,Rest0...> &a) {

      static_assert(!is_single_reduction<Index<Idx0...>>::value,"REDUCTION TO SCALAR REQUESTED. USE REDUCTION FUNCTION INSTEAD");

    using OutTensor = typename contraction_impl<Index<Idx0...>, Tensor<T,Rest0...>,
      typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::type;
    using OutIndices = typename contraction_impl<Index<Idx0...>, Tensor<T,Rest0...>,
        typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::indices;

    OutTensor out;
    out.zeros();
    const T *a_data = a.data();
    T *out_data = out.data();

    constexpr int a_dim = sizeof...(Rest0);
    constexpr int out_dim =  no_of_unique<Idx0...>::value;

    constexpr auto& idx_a = IndexTensors<
            Index<Idx0...>,
            Tensor<T,Rest0...>,
            Index<Idx0...>,Tensor<T,Rest0...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::indices;

    constexpr auto& idx_out = IndexTensors<
            Index<Idx0...>,
            Tensor<T,Rest0...>,
            OutIndices,OutTensor,
            typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::indices;

    using nloops = loop_setter<
              Index<Idx0...>,
              Tensor<T,Rest0...>,
              typename std_ext::make_index_sequence<out_dim>::type>;
    constexpr auto& maxes_out = nloops::dims;
    constexpr int total = nloops::value;

    constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;

    using Index_with_dims = typename put_dims_in_Index<OutTensor>::type;
    constexpr std::array<size_t,OutTensor::Dimension> products_out = \
            nprods<Index_with_dims,typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::values;


    int as[out_dim];
    std::fill(as,as+out_dim,0);
    constexpr int stride = 1;

    int it;
    for (int i = 0; i < total; i+=stride) {
        int remaining = total;
        for (int n = 0; n < out_dim; ++n) {
            remaining /= maxes_out[n];
            as[n] = ( i / remaining ) % maxes_out[n];
        }

        int index_a = as[idx_a[a_dim-1]];
        for(it = 0; it< a_dim; it++) {
            index_a += products_a[it]*as[idx_a[it]];
        }
        int index_out = as[idx_out[OutTensor::Dimension-1]];
        for(it = 0; it< static_cast<int>(OutTensor::Dimension); it++) {
            index_out += products_out[it]*as[idx_out[it]];
        }
//        V _vec_out = V(a_data+index_a) +  V(out_data+index_out);
//        _vec_out.store(out_data+index_out);
        out_data[index_out] += a_data[index_a];
    }

    return out;
}
};


template<class Index_I,
typename T, size_t ... Rest0>
auto contraction(const Tensor<T,Rest0...> &a)
-> decltype(extractor_contract_1<Index_I>::contract_impl(a)) {
return extractor_contract_1<Index_I>::contract_impl(a);
}



}

#endif // CONTRACTION_SINGLE_H
