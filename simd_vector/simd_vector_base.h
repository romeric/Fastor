#ifndef SIMD_VECTOR_BASE_H
#define SIMD_VECTOR_BASE_H

#include "commons/commons.h"
#include "extended_intrinsics/extintrin.h"
#include "math/internal_math.h"


namespace Fastor {

template<typename,int Abi=256>
struct get_vector_size;

template<>
struct get_vector_size<double,256> {
    static const FASTOR_INDEX size = 4;
};
template<>
struct get_vector_size<float,256> {
    static const FASTOR_INDEX size = 8;
};
template<>
struct get_vector_size<int,256> {
    // Note that 256bit integer arithmatics were introduced under AVX2
    static const FASTOR_INDEX size = 8;
};
template<>
struct get_vector_size<double,128> {
    static const FASTOR_INDEX size = 2;
};
template<>
struct get_vector_size<float,128> {
    static const FASTOR_INDEX size = 4;
};
template<>
struct get_vector_size<int,128> {
    // Note that 256bit integer arithmatics were introduced under AVX2
    static const FASTOR_INDEX size = 4;
};



template <typename T, int Abi=256>
struct SIMDVector;

}

#endif // SIMD_VECTOR_H

