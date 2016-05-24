#ifndef SIMD_VECTOR_BASE_H
#define SIMD_VECTOR_BASE_H

#include "commons/commons.h"
#include "extended_intrinsics/extintrin.h"
#include "math/internal_math.h"


namespace Fastor {

template<typename>
struct get_vector_size;
//#ifdef HAS_AVX
template<>
struct get_vector_size<double> {
    static const FASTOR_INDEX size = 4;
};
template<>
struct get_vector_size<float> {
    static const FASTOR_INDEX size = 8;
};
template<>
struct get_vector_size<int> {
    // This is kept four because most 256bit integer arithmatics
    // are not available under AVX - 256bit integer arithmatics
    // were introduced only under AVX2
    static const FASTOR_INDEX size = 4;
};
//#else
//template<>
//struct get_vector_size<double> {
//    static const FASTOR_INDEX size = 2;
//};
//template<>
//struct get_vector_size<float> {
//    static const FASTOR_INDEX size = 4;
//};
//#endif


template <typename T>
struct SIMDVector;


}

#endif // SIMD_VECTOR_H

