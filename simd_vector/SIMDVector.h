#ifndef SIMDVECTOR_H
#define SIMDVECTOR_H

#include "simd_vector_base.h"
#include "simd_vector_float.h"
#include "simd_vector_double.h"
#include "simd_vector_int.h"
#include "simd_vector_int64.h"


// Generic overloads
namespace Fastor {

template<typename T>
SIMDVector<T> exp(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_exp(a.value);
    return out;
}

template<typename T>
SIMDVector<T> log(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_log(a.value);
    return out;
}

template<typename T, typename U>
SIMDVector<T> pow(const SIMDVector<T> &a, const SIMDVector<U> &b) {
    SIMDVector<T> out;
    out.value = internal_pow(a.value, b.value);
    return out;
}

template<typename T, typename U>
SIMDVector<T> pow(const SIMDVector<T> &a, U bb) {
    SIMDVector<T> out;
    SIMDVector<T> b = static_cast<T>(bb);
    out.value = internal_pow(a.value, b.value);
    return out;
}

template<typename T>
SIMDVector<T> sin(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_sin(a.value);
    return out;
}

template<typename T>
SIMDVector<T> cos(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_cos(a.value);
    return out;
}

template<typename T>
SIMDVector<T> tan(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_tan(a.value);
    return out;
}

template<typename T>
SIMDVector<T> asin(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_asin(a.value);
    return out;
}

template<typename T>
SIMDVector<T> acos(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_acos(a.value);
    return out;
}

template<typename T>
SIMDVector<T> atan(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_atan(a.value);
    return out;
}

template<typename T>
SIMDVector<T> sinh(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_sinh(a.value);
    return out;
}

template<typename T>
SIMDVector<T> cosh(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_cosh(a.value);
    return out;
}

template<typename T>
SIMDVector<T> tanh(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_tanh(a.value);
    return out;
}





// Broadcasting vectorisation on general strides
//----------------------------------------------------------------------------------------------------------------
// 4 word scalar
template<typename T, int ABI,
         typename std::enable_if<sizeof(T)==4 && ABI==32,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int ) {
    vec.set(data[idx]);
}
// 4 word SSE
template<typename T, int ABI,
         typename std::enable_if<sizeof(T)==4 && ABI==128,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+3*general_stride],data[idx+2*general_stride],data[idx+general_stride],data[idx]);
}
// 4 word AVX
template<typename T, int ABI,
         typename std::enable_if<sizeof(T)==4 && ABI==256,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+7*general_stride],data[idx+6*general_stride],
            data[idx+5*general_stride],data[idx+4*general_stride],
            data[idx+3*general_stride],data[idx+2*general_stride],
            data[idx+general_stride],data[idx]);
}
// 4 word AVX 512
template<typename T, int ABI,
         typename std::enable_if<sizeof(T)==4 && ABI==512,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+15*general_stride],data[idx+14*general_stride],
            data[idx+13*general_stride],data[idx+12*general_stride],
            data[idx+11*general_stride],data[idx+10*general_stride],
            data[idx+9*general_stride],data[idx+8*general_stride],
            data[idx+7*general_stride],data[idx+6*general_stride],
            data[idx+5*general_stride],data[idx+4*general_stride],
            data[idx+3*general_stride],data[idx+2*general_stride],
            data[idx+general_stride],data[idx]);
}

// 8 word scalar
template<typename T, int ABI,
         typename std::enable_if<sizeof(T)==8 && ABI==64,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int ) {
    vec.set(data[idx]);
}
// 8 word SSE
template<typename T, int ABI,
         typename std::enable_if<sizeof(T)==8 && ABI==128,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+general_stride],data[idx]);
}
// 8 word AVX
template<typename T, int ABI,
         typename std::enable_if<sizeof(T)==8 && ABI==256,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+3*general_stride],data[idx+2*general_stride],
            data[idx+general_stride],data[idx]);
}
// 8 word AVX 512
template<typename T, int ABI,
         typename std::enable_if<sizeof(T)==8 && ABI==512,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+7*general_stride],data[idx+6*general_stride],
            data[idx+5*general_stride],data[idx+4*general_stride],
            data[idx+3*general_stride],data[idx+2*general_stride],
            data[idx+general_stride],data[idx]);
}
//----------------------------------------------------------------------------------------------------------------



}

#endif // SIMDVECTOR_H

