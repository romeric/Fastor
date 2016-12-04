#ifndef REDUCTION_H
#define REDUCTION_H

#include "tensor/Tensor.h"
#include "indicial.h"
#include "contraction.h"

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
    //! The size of the tensor in all dimensions should be equal (uniform)

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


template<typename T, size_t ... Rest>
T reduction(const Tensor<T,Rest...> &a, const Tensor<T,Rest...> &b) {
    //! Reduction of a tensor pair to a scalar, for instance A_ijklm * B_ijklm
    //! If a and b are scalars/vectors, returns dot product
    //! If a and b are matrices, returns double contraction
    //! For third order tensors returns a_ijk*b_ijk
    //! ...

    const T *a_data = a.data();
    const T *b_data = b.data();

    constexpr int ndim = sizeof...(Rest);
    if (ndim>0) {
        return _doublecontract<T,prod<Rest...>::value,1>(a_data,b_data);
    }
    else {
        return a_data[0]*b_data[0];
    }
}


}

#endif // REDUCTION_H

