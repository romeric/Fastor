#ifndef SUMMATION_H
#define SUMMATION_H

#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor_algebra/indicial.h"

namespace Fastor {

// summation
template<typename T, size_t ... Rest>
FASTOR_INLINE T summation(const Tensor<T,Rest...> &a) {
    return a.sum();
}

}

#endif // SUMMATION_H

