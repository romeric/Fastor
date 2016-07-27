#ifndef REDUCTION_H
#define REDUCTION_H

#include "tensor/Tensor.h"
#include "indicial.h"

namespace Fastor {


// reduction
//----------------------------------------------------------------------------------------------
template<typename T, size_t ... Rest>
T reduction(const Tensor<T,Rest...> &a) {
    //! Reduces a multi-dimensional tensor to a scalar
    //!
    //! If a is scalar/Tensor<T> returns the value itself
    //! If a is a vector Tensor<T,N> returns the sum of values
    //! If a is a second order tensor Tensor<T,N,N> returns the trace
    //! If a is a third order tensor Tensor<T,N,N,N> returns a_iii
    //! ...
    //!
    //! The size of the tensor in all dimensions should be equal

    static_assert(no_of_unique<Rest...>::value<=1, "REDUCTION IS ONLY POSSIBLE ON UNIFORM TENSORS");
    constexpr int ndim = sizeof...(Rest);

    T *a_data = a.data();
    if (ndim==0) {
        return a_data[0];
    }
    else if (ndim==1) {
        return a.sum();
    }
    else {
//        using detail::nprods;
        constexpr std::array<size_t,ndim> products = nprods<Index<Rest...>,
                typename std_ext::make_index_sequence<ndim>::type>::values;

        T reductor = static_cast<T>(0);
        for (int i=0; i<a.dimension(0); ++i) {
            int index_a = i;
            for(int it = 0; it< ndim; it++) {
                index_a += products[it]*i;
            }
            reductor += a_data[index_a];
        }
        return reductor;
    }
}

}

#endif // REDUCTION_H

