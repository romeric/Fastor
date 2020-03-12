#ifndef RESHAPE_H
#define RESHAPE_H

#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor_algebra/indicial.h"
#include "Fastor/tensor_algebra/permutation.h"

namespace Fastor {


template<size_t ... shapes,typename T, size_t ... Rest>
FASTOR_INLINE Tensor<T,shapes...> reshape(const Tensor<T,Rest...> &a) {
    // Call it as reshape<shapes...>(a)
    static_assert(prod<shapes...>::value==prod<Rest...>::value, "SIZE OF TENSOR SHOULD REMAIN THE SAME DURING RESHAPE");
    Tensor<T,shapes...> out;
    //! Reshape involves deep copy, as in place permutation is not possible since Tensor is static
    std::copy(a.data(),a.data()+a.Size,out.data());
    return out;
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

