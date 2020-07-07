#ifndef TENSOR_FUNCTIONS_H
#define TENSOR_FUNCTIONS_H

#include "Fastor/meta/meta.h"
#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/TensorMap.h"
#include "Fastor/tensor/TensorTraits.h"

namespace Fastor {

/* Turns a row-major tensor to column-major */
template<template<typename,size_t...> class TensorType, typename T, size_t ... Rest>
FASTOR_INLINE Tensor<T,Rest...> tocolumnmajor(const TensorType<T,Rest...> &a) {
    constexpr int Dimension = sizeof...(Rest);
    if (Dimension < 2) {
        return a;
    }
    else {
        Tensor<T,Rest...> out;
        T *arr_out = out.data();
        const T *a_data = a.data();

        if (Dimension == 2) {
            constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
            constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
            for (FASTOR_INDEX i=0; i<M; ++i) {
                for (FASTOR_INDEX j=0; j<N; ++j) {
                    arr_out[i*N+j] = a_data[j*M+i];
                }
            }
        }
        else {
            constexpr int Size = pack_prod<Rest...>::value;
            std::array<size_t,Dimension> products_ = nprods_views<Index<Rest...>,
                typename std_ext::make_index_sequence<Dimension>::type>::values;
            FASTOR_INDEX DimensionHolder[Dimension] = {Rest...};
            std::reverse(DimensionHolder,DimensionHolder+Dimension);
            std::reverse(products_.begin(),products_.end());
            std::array<int,Dimension> as = {};

            int jt;
            FASTOR_INDEX counter=0;
            while(counter < Size)
            {
                FASTOR_INDEX index = 0;
                for (int ii=0; ii<Dimension; ++ii) {
                    index += products_[ii]*as[ii];
                }

                arr_out[index] = a_data[counter];

                counter++;
                for(jt = Dimension-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<DimensionHolder[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        return out;
    }
}

/* Turns a column-major tensor to row-major */
template<template<typename,size_t...> class TensorType, typename T, size_t ... Rest>
FASTOR_INLINE Tensor<T,Rest...> torowmajor(const TensorType<T,Rest...> &a) {
    constexpr int Dimension = sizeof...(Rest);
    if (Dimension < 2) {
        return a;
    }
    else {
        Tensor<T,Rest...> out;
        T *arr_out = out.data();
        const T *a_data = a.data();

        if (Dimension == 2) {
            constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
            constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
            for (FASTOR_INDEX i=0; i<M; ++i) {
                for (FASTOR_INDEX j=0; j<N; ++j) {
                    arr_out[j*M+i] = a_data[i*N+j];
                }
            }
        }
        else {
            constexpr int Size = pack_prod<Rest...>::value;
            std::array<size_t,Dimension> products_ = nprods_views<Index<Rest...>,
                typename std_ext::make_index_sequence<Dimension>::type>::values;
            FASTOR_INDEX DimensionHolder[Dimension] = {Rest...};
            std::reverse(DimensionHolder,DimensionHolder+Dimension);
            std::reverse(products_.begin(),products_.end());
            std::array<int,Dimension> as = {};

            int jt;
            FASTOR_INDEX counter=0;
            while(counter < Size)
            {
                FASTOR_INDEX index = 0;
                for (int ii=0; ii<Dimension; ++ii) {
                    index += products_[ii]*as[ii];
                }

                arr_out[counter] = a_data[index];

                counter++;
                for(jt = Dimension-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<DimensionHolder[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        return out;
    }
}

/* Squeeze - removes dimenions of 1 from a tensor and returns a map
    Note that you cannot use tensor expressions on squeeze as squeeze is
    a view in to a concrete tensor type holding storage
*/
template<template <typename,size_t...> class TensorType, typename T, size_t ... Rest>
FASTOR_INLINE
index_to_tensor_map_t<T,filter_t<1,Rest...>>
squeeze(const TensorType<T,Rest...> &a) {
    return index_to_tensor_map_t<T,filter_t<1,Rest...>>(a.data());
}










#if FASTOR_NIL
// Constant tensors
static FASTOR_INLINE
Tensor<float,3,3,3> levi_civita_ps() {
    Tensor<float,3,3,3> LeCi_ps;
    LeCi_ps(0,1,2) = 1.f;
    LeCi_ps(1,2,0) = 1.f;
    LeCi_ps(2,0,1) = 1.f;
    LeCi_ps(1,0,2) = -1.f;
    LeCi_ps(2,1,0) = -1.f;
    LeCi_ps(0,2,1) = -1.f;

    return LeCi_ps;
}

static FASTOR_INLINE
Tensor<double,3,3,3> levi_civita_pd() {
    Tensor<double,3,3,3> LeCi_pd;
    LeCi_pd(0,1,2) = 1.;
    LeCi_pd(1,2,0) = 1.;
    LeCi_pd(2,0,1) = 1.;
    LeCi_pd(1,0,2) = -1.;
    LeCi_pd(2,1,0) = -1.;
    LeCi_pd(0,2,1) = -1.;

    return LeCi_pd;
}

template<typename T, size_t ... Rest>
static FASTOR_INLINE
Tensor<T,Rest...> kronecker_delta() {
    Tensor<T,Rest...> out; out.eye();
    return out;
}
#endif

}

#endif // TENSOR_FUNCTIONS_H

