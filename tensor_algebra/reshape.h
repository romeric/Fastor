#ifndef RESHAPE_H
#define RESHAPE_H

#include "tensor/Tensor.h"
#include "indicial.h"
#include "permutation.h"

namespace Fastor {

template<class T>
struct extractor_reshape {};

template<size_t ... Idx>
struct extractor_reshape<Index<Idx...> > {
  template<typename T, size_t ... Rest> static
    typename permute_impl<T,Index<Idx...>, Tensor<T,Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Idx)>::type>::type
    reshape_impl(const Tensor<T,Rest...> &a) {

        typename permute_impl<T,Index<Idx...>, Tensor<T,Rest...>,
            typename std_ext::make_index_sequence<sizeof...(Idx)>::type>::type out;
        out.zeros();

        //! Reshape involves deep copy, as in place permutation is not possible since Tensor is static
        std::copy(a.data(),a.data()+a.Size,out.data());

        return out;
    }
};



template<class Index_I, typename T, size_t ... Rest>
typename permute_impl<T,Index_I, Tensor<T,Rest...>,
    typename std_ext::make_index_sequence<sizeof...(Rest)>::type>::type reshape(const Tensor<T, Rest...> &a) {
    return extractor_reshape<Index_I>::reshape_impl(a);
}


// flatten
template<typename T, size_t ... Rest>
FASTOR_INLINE Tensor<T,prod<Rest...>::value> flatten(const Tensor<T,Rest...> &a) {
    Tensor<T,prod<Rest...>::value> out;
    std::copy(a.data(),a.data()+a.Size,out.data());
    return out;
}

}
#endif // RESHAPE_H

