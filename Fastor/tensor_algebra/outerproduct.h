#ifndef OUTERPRODUCT_H
#define OUTERPRODUCT_H

#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor_algebra/indicial.h"

namespace Fastor {

// outer products
//---------------------------------------------------------------------------------------------

template<typename T, size_t ...Rest0, size_t ...Rest1>
FASTOR_INLINE Tensor<T,Rest0...,Rest1...>
outer(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) {
     Tensor<T,Rest0...,Rest1...> out;
     _dyadic<T,prod<Rest0...>::value,prod<Rest1...>::value>(a.data(),b.data(),out.data());
     return out;
}

template<typename T, size_t ...Rest0>
FASTOR_INLINE Tensor<T,Rest0...>
outer(const Tensor<T,Rest0...> &a, const Tensor<T,1> &b) {
     Tensor<T,Rest0...> out = a*b.toscalar();
     return out;
}
template<typename T, size_t ...Rest1>
FASTOR_INLINE Tensor<T,Rest1...>
outer(const Tensor<T,1> &a, const Tensor<T,Rest1...> &b) {
     Tensor<T,Rest1...> out = a.toscalar()*b;
     return out;
}
template<typename T>
FASTOR_INLINE Tensor<T>
outer(const Tensor<T> &a, const Tensor<T> &b) {
     Tensor<T> out;
     _dyadic<T,1,1>(a.data(),b.data(),out.data());
     return out;
}


template<typename T, size_t ... Rest0, size_t ... Rest1>
FASTOR_INLINE Tensor<T,Rest0...,Rest1...> dyadic(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) {
    return outer(a,b);
}

} // end of namespace Fastor

#endif // OUTERPRODUCT_H

