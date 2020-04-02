#ifndef SIMDVECTOR_H
#define SIMDVECTOR_H

#include "simd_vector_base.h"
#include "simd_vector_float.h"
#include "simd_vector_double.h"
#include "simd_vector_int.h"
#include "simd_vector_int64.h"


// Generic overloads
namespace Fastor {

template<typename T, typename std::enable_if<!std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> exp(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_exp(a.value);
    return out;
}
template<typename T, typename std::enable_if<std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> exp(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::exp(a.value[i]);}
    return out;
}

template<typename T, typename std::enable_if<!std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> log(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_log(a.value);
    return out;
}
template<typename T, typename std::enable_if<std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> log(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::log(a.value[i]);}
    return out;
}

template<typename T, typename U>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> pow(const SIMDVector<T,DEFAULT_ABI> &a, const SIMDVector<U,DEFAULT_ABI> &b) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_pow(a.value, b.value);
    return out;
}

template<typename T, typename U, typename std::enable_if<!std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> pow(const SIMDVector<T,DEFAULT_ABI> &a, U bb) {
    SIMDVector<T,DEFAULT_ABI> out;
    SIMDVector<T,DEFAULT_ABI> b = static_cast<T>(bb);
    out.value = internal_pow(a.value, b.value);
    return out;
}
template<typename T, typename U, typename std::enable_if<std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> pow(const SIMDVector<T,DEFAULT_ABI> &a, U bb) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::pow(a.value[i], bb);}
    return out;
}

template<typename T, typename std::enable_if<!std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> sin(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_sin(a.value);
    return out;
}
template<typename T, typename std::enable_if<std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> sin(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::sin(a.value[i]);}
    return out;
}

template<typename T, typename std::enable_if<!std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> cos(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_cos(a.value);
    return out;
}
template<typename T, typename std::enable_if<std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> cos(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::cos(a.value[i]);}
    return out;
}

template<typename T, typename std::enable_if<!std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> tan(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_tan(a.value);
    return out;
}
template<typename T, typename std::enable_if<std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> tan(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::tan(a.value[i]);}
    return out;
}

template<typename T, typename std::enable_if<!std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> asin(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_asin(a.value);
    return out;
}
template<typename T, typename std::enable_if<std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> asin(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::asin(a.value[i]);}
    return out;
}

template<typename T, typename std::enable_if<!std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> acos(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_acos(a.value);
    return out;
}
template<typename T, typename std::enable_if<std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> acos(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::acos(a.value[i]);}
    return out;
}

template<typename T, typename std::enable_if<!std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> atan(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_atan(a.value);
    return out;
}
template<typename T, typename std::enable_if<std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> atan(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::atan(a.value[i]);}
    return out;
}

template<typename T, typename std::enable_if<!std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> sinh(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_sinh(a.value);
    return out;
}
template<typename T, typename std::enable_if<std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> sinh(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::sinh(a.value[i]);}
    return out;
}

template<typename T, typename std::enable_if<!std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> cosh(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_cosh(a.value);
    return out;
}
template<typename T, typename std::enable_if<std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> cosh(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::cosh(a.value[i]);}
    return out;
}

template<typename T, typename std::enable_if<!std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> tanh(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_tanh(a.value);
    return out;
}
template<typename T, typename std::enable_if<std::is_array<typename SIMDVector<T,DEFAULT_ABI>::value_type>::value,bool>::type=0>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> tanh(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::tanh(a.value[i]);}
    return out;
}





// Broadcasting vectorisation on general strides [gather operations]
//----------------------------------------------------------------------------------------------------------------
// 4 word scalar
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==4 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==32,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int ) {
    vec.set(data[idx]);
}
// 4 word in an 8 - for compatibility
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==4 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==64,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+general_stride],data[idx]);
}
// 4 word SSE
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==4 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==128,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+3*general_stride],data[idx+2*general_stride],data[idx+general_stride],data[idx]);
}
// 4 word AVX
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==4 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==256,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+7*general_stride],data[idx+6*general_stride],
            data[idx+5*general_stride],data[idx+4*general_stride],
            data[idx+3*general_stride],data[idx+2*general_stride],
            data[idx+general_stride],data[idx]);
}
// 4 word AVX 512
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==4 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==512,bool>::type=0>
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
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==8 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==64,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int ) {
    vec.set(data[idx]);
}
// 8 word SSE
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==8 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==128,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+general_stride],data[idx]);
}
// 8 word AVX
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==8 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==256,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+3*general_stride],data[idx+2*general_stride],
            data[idx+general_stride],data[idx]);
}
// 8 word AVX 512
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==8 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==512,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+7*general_stride],data[idx+6*general_stride],
            data[idx+5*general_stride],data[idx+4*general_stride],
            data[idx+3*general_stride],data[idx+2*general_stride],
            data[idx+general_stride],data[idx]);
}

// // 16 word scalar
// template<typename T, int ABI,
//          typename std::enable_if<sizeof(T)==16 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==64,bool>::type=0>
// FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
//     vec.set(data[idx]);
// }
// 16 word scalar/SSE
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==16 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==128,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx]);
}
// 16 word AVX
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==16 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==256,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+general_stride],data[idx]);
}
// 16 word AVX 512
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==16 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==512,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+3*general_stride],data[idx+2*general_stride],
            data[idx+general_stride],data[idx]);
}
//----------------------------------------------------------------------------------------------------------------


// [Gather operations], when strides are not constant (i.e totally random)
//----------------------------------------------------------------------------------------------------------------
// 4 word scalar
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==4 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==32,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, const std::array<int,1> &a) {
    vec.set(data[a[0]]);
}
// 4 word in an 8 - for compatibility
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==4 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==64,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, const std::array<int,2> &a) {
    vec.set(data[a[1]],data[a[0]]);
}
// 4 word SSE
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==4 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==128,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, const std::array<int,4> a) {
    vec.set(data[a[3]],data[a[2]],data[a[1]],data[a[0]]);
}
// 4 word AVX
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==4 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==256,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, const std::array<int,8> a) {
    vec.set(data[a[7]],data[a[6]],data[a[5]],data[a[4]],
            data[a[3]],data[a[2]],data[a[1]],data[a[0]]);
}
// 4 word AVX 512
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==4 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==512,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, const std::array<int,16> a) {
    vec.set(data[a[15]],data[a[14]],data[a[13]],data[a[12]],
            data[a[11]],data[a[10]],data[a[9]],data[a[8]],
            data[a[7]],data[a[6]],data[a[5]],data[a[4]],
            data[a[3]],data[a[2]],data[a[1]],data[a[0]]);
}

// 8 word scalar
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==8 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==64,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, const std::array<int,1> &a) {
    vec.set(data[a[0]]);
}
// 8 word SSE
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==8 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==128,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, const std::array<int,2> a) {
    vec.set(data[a[1]],data[a[0]]);
}
// 8 word AVX
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==8 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==256,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, const std::array<int,4> a) {
    vec.set(data[a[3]],data[a[2]],data[a[1]],data[a[0]]);
}
// 8 word AVX 512
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==8 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==512,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, const std::array<int,8> a) {
    vec.set(data[a[7]],data[a[6]],data[a[5]],data[a[4]],
            data[a[3]],data[a[2]],data[a[1]],data[a[0]]);
}

// 16 word scalar/SSE
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==16 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==128,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, const std::array<int,1> &a) {
    vec.set(data[a[0]]);
}
// 16 word AVX
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==16 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==256,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, const std::array<int,2> a) {
    vec.set(data[a[1]],data[a[0]]);
}
// 16 word AVX 512
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==16 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==512,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, const std::array<int,4> a) {
    vec.set(data[a[3]],data[a[2]],data[a[1]],data[a[0]]);
}
//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------




// Scatter operations
//----------------------------------------------------------------------------------------------------------------
// 4 word scalar
template<typename T, typename ABI, typename Int,
         typename std::enable_if<sizeof(T)==4 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==32,bool>::type=0>
FASTOR_INLINE void data_setter(T *FASTOR_RESTRICT data, const SIMDVector<T,ABI> &vec, Int idx, int ) {
    data[idx] = vec.value;
}
// 4 word in an 8 - for compatibility
template<typename T, typename ABI, typename Int,
         typename std::enable_if<sizeof(T)==4 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==64,bool>::type=0>
FASTOR_INLINE void data_setter(T *FASTOR_RESTRICT data, const SIMDVector<T,ABI> &vec, Int idx, int general_stride=1) {
    data[idx] = vec[0];
    data[idx+general_stride] = vec[1];
}
// 4 word SSE
template<typename T, typename ABI, typename Int,
         typename std::enable_if<sizeof(T)==4 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==128,bool>::type=0>
FASTOR_INLINE void data_setter(T *FASTOR_RESTRICT data, const SIMDVector<T,ABI> &vec, Int idx, int general_stride=1) {
    data[idx] = vec[0];
    data[idx+general_stride] = vec[1];
    data[idx+2*general_stride] = vec[2];
    data[idx+3*general_stride] = vec[3];
}
// 4 word AVX
template<typename T, typename ABI, typename Int,
         typename std::enable_if<sizeof(T)==4 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==256,bool>::type=0>
FASTOR_INLINE void data_setter(T *FASTOR_RESTRICT data, const SIMDVector<T,ABI> &vec, Int idx, int general_stride=1) {
    data[idx] = vec[0];
    data[idx+general_stride] = vec[1];
    data[idx+2*general_stride] = vec[2];
    data[idx+3*general_stride] = vec[3];
    data[idx+4*general_stride] = vec[4];
    data[idx+5*general_stride] = vec[5];
    data[idx+6*general_stride] = vec[6];
    data[idx+7*general_stride] = vec[7];
}
// 4 word AVX 512
template<typename T, typename ABI, typename Int,
         typename std::enable_if<sizeof(T)==4 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==512,bool>::type=0>
FASTOR_INLINE void data_setter(T *FASTOR_RESTRICT data, const SIMDVector<T,ABI> &vec, Int idx, int general_stride=1) {
    data[idx] = vec[0];
    data[idx+general_stride] = vec[1];
    data[idx+2*general_stride] = vec[2];
    data[idx+3*general_stride] = vec[3];
    data[idx+4*general_stride] = vec[4];
    data[idx+5*general_stride] = vec[5];
    data[idx+6*general_stride] = vec[6];
    data[idx+7*general_stride] = vec[7];
    data[idx+8*general_stride] = vec[8];
    data[idx+9*general_stride] = vec[9];
    data[idx+10*general_stride] = vec[10];
    data[idx+11*general_stride] = vec[11];
    data[idx+12*general_stride] = vec[12];
    data[idx+13*general_stride] = vec[13];
    data[idx+14*general_stride] = vec[14];
    data[idx+15*general_stride] = vec[15];
}

// 8 word scalar
template<typename T, typename ABI, typename Int,
         typename std::enable_if<sizeof(T)==8 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==64,bool>::type=0>
FASTOR_INLINE void data_setter(T *FASTOR_RESTRICT data, const SIMDVector<T,ABI> &vec, Int idx, int ) {
    data[idx] = vec.value;
}
// 8 word SSE
template<typename T, typename ABI, typename Int,
         typename std::enable_if<sizeof(T)==8 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==128,bool>::type=0>
FASTOR_INLINE void data_setter(T *FASTOR_RESTRICT data, const SIMDVector<T,ABI> &vec, Int idx, int general_stride) {
    data[idx] = vec[0];
    data[idx+general_stride] = vec[1];
}
// 8 word AVX
template<typename T, typename ABI, typename Int,
         typename std::enable_if<sizeof(T)==8 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==256,bool>::type=0>
FASTOR_INLINE void data_setter(T *FASTOR_RESTRICT data, const SIMDVector<T,ABI> &vec, Int idx, int general_stride) {
    data[idx] = vec[0];
    data[idx+general_stride] = vec[1];
    data[idx+2*general_stride] = vec[2];
    data[idx+3*general_stride] = vec[3];
}
// 8 word AVX 512
template<typename T, typename ABI, typename Int,
         typename std::enable_if<sizeof(T)==8 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==512,bool>::type=0>
FASTOR_INLINE void data_setter(T *FASTOR_RESTRICT data, const SIMDVector<T,ABI> &vec, Int idx, int general_stride) {
    data[idx] = vec[0];
    data[idx+general_stride] = vec[1];
    data[idx+2*general_stride] = vec[2];
    data[idx+3*general_stride] = vec[3];
    data[idx+4*general_stride] = vec[4];
    data[idx+5*general_stride] = vec[5];
    data[idx+6*general_stride] = vec[6];
    data[idx+7*general_stride] = vec[7];
}

// 16 word scalar/SSE
template<typename T, typename ABI, typename Int,
         typename std::enable_if<sizeof(T)==16 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==128,bool>::type=0>
FASTOR_INLINE void data_setter(T *FASTOR_RESTRICT data, const SIMDVector<T,ABI> &vec, Int idx, int ) {
    data[idx] = vec.value;
}
// 16 word AVX
template<typename T, typename ABI, typename Int,
         typename std::enable_if<sizeof(T)==16 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==256,bool>::type=0>
FASTOR_INLINE void data_setter(T *FASTOR_RESTRICT data, const SIMDVector<T,ABI> &vec, Int idx, int general_stride) {
    data[idx] = vec[0];
    data[idx+general_stride] = vec[1];
}
// 16 word AVX 512
template<typename T, typename ABI, typename Int,
         typename std::enable_if<sizeof(T)==16 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==512,bool>::type=0>
FASTOR_INLINE void data_setter(T *FASTOR_RESTRICT data, const SIMDVector<T,ABI> &vec, Int idx, int general_stride) {
    data[idx] = vec[0];
    data[idx+general_stride] = vec[1];
    data[idx+2*general_stride] = vec[2];
    data[idx+3*general_stride] = vec[3];
}
//----------------------------------------------------------------------------------------------------------------









// Mask load and store operations implemented as free functions till SIMDVector finds a proper
// companion masked vector. Note that these require at least AVX even the SSE variants
//----------------------------------------------------------------------------------------------------------------
template<typename V>
FASTOR_INLINE V
maskload(const typename V::scalar_value_type * FASTOR_RESTRICT a, const int (&maska)[V::Size]) {
    // masked array is reversed like other intrinsics
    typename V::scalar_value_type val_out[V::Size] = {}; // zero out the rest
    for (FASTOR_INDEX i=0; i<V::Size; ++i) {
        if (maska[i] == -1) {
            val_out[V::Size - i - 1] = a[V::Size - i - 1];
        }
    }
    return val_out;
}
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx>
maskload<SIMDVector<double,simd_abi::avx>>(const double * FASTOR_RESTRICT a, const int (&maska)[4]) {
    __m256i mask = _mm256_set_epi64x(maska[0],maska[1],maska[2],maska[3]);
    return _mm256_maskload_pd(a,(__m256i) mask);
}
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx>
maskload<SIMDVector<float,simd_abi::avx>>(const float * FASTOR_RESTRICT a, const int (&maska)[8]) {
    __m256i mask = _mm256_set_epi32(maska[0],maska[1],maska[2],maska[3],maska[4],maska[5],maska[6],maska[7]);
    return _mm256_maskload_ps(a,(__m256i) mask);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse>
maskload<SIMDVector<double,simd_abi::sse>>(const double * FASTOR_RESTRICT a, const int (&maska)[2]) {
    __m128i mask = _mm_set_epi64x(maska[0],maska[1]);
    return _mm_maskload_pd(a,(__m128i) mask);
}
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse>
maskload<SIMDVector<float,simd_abi::sse>>(const float * FASTOR_RESTRICT a, const int (&maska)[4]) {
    __m128i mask = _mm_set_epi32(maska[0],maska[1],maska[2],maska[3]);
    return _mm_maskload_ps(a,(__m128i) mask);
}
#endif // FASTOR_AVX_IMPL


// Ideally this style should be followed to be compatible with actual intrinsics signatures
#if 0
template<typename V>
FASTOR_INLINE V
maskload(V &vec, const typename V::scalar_value_type * FASTOR_RESTRICT a, const int (&maska)[V::Size]);
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx>
maskload(SIMDVector<double,simd_abi::avx> &vec, const double * FASTOR_RESTRICT a, const int (&maska)[4]) {
    __m256i mask = _mm256_set_epi64x(maska[0],maska[1],maska[2],maska[3]);
    vec.value(_mm256_maskload_pd(a,(__m256i) mask));
}
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx>
maskload(SIMDVector<float,simd_abi::avx> &vec, const float * FASTOR_RESTRICT a, const int (&maska)[8]) {
    __m256i mask = _mm256_set_epi32(maska[0],maska[1],maska[2],maska[3],maska[4],maska[5],maska[6],maska[7]);
    vec.value(_mm256_maskload_ps(a,(__m256i) mask));
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse>
maskload(SIMDVector<double,simd_abi::sse> &vec, const double * FASTOR_RESTRICT a, const int (&maska)[2]) {
    __m128i mask = _mm_set_epi64x(maska[0],maska[1]);
    vec.value(_mm_maskload_pd(a,(__m128i) mask));
}
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse>
maskload(SIMDVector<float,simd_abi::sse> &vec, const float * FASTOR_RESTRICT a, const int (&maska)[4]) {
    __m128i mask = _mm_set_epi32(maska[0],maska[1],maska[2],maska[3]);
    vec.value(_mm_maskload_ps(a,(__m128i) mask));
}
#endif // FASTOR_AVX_IMPL
#endif


template<typename V>
FASTOR_INLINE
void maskstore(typename V::scalar_value_type * FASTOR_RESTRICT a, const int (&maska)[V::Size], V& v) {
    for (FASTOR_INDEX i=0; i<V::Size; ++i) {
        if (maska[i] == -1) {
            a[V::Size - i - 1] = v[V::Size - i - 1];
        }
    }
}
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE
void maskstore(double * FASTOR_RESTRICT a, const int (&maska)[4], SIMDVector<double,simd_abi::avx> &v) {
    __m256i mask = _mm256_set_epi64x(maska[0],maska[1],maska[2],maska[3]);
    _mm256_maskstore_pd(a,(__m256i) mask, v.value);
}
template<>
FASTOR_INLINE
void maskstore(float * FASTOR_RESTRICT a, const int (&maska)[8], SIMDVector<float,simd_abi::avx> &v) {
    __m256i mask = _mm256_set_epi32(maska[0],maska[1],maska[2],maska[3],maska[4],maska[5],maska[6],maska[7]);
    _mm256_maskstore_ps(a,(__m256i) mask, v.value);
}
template<>
FASTOR_INLINE
void maskstore(double * FASTOR_RESTRICT a, const int (&maska)[2], SIMDVector<double,simd_abi::sse> &v) {
    __m128i mask = _mm_set_epi64x(maska[0],maska[1]);
    _mm_maskstore_pd(a,(__m128i) mask, v.value);
}
template<>
FASTOR_INLINE
void maskstore(float * FASTOR_RESTRICT a, const int (&maska)[4], SIMDVector<float,simd_abi::sse> &v) {
    __m128i mask = _mm_set_epi32(maska[0],maska[1],maska[2],maska[3]);
    _mm_maskstore_ps(a,(__m128i) mask, v.value);
}
#endif // FASTOR_AVX_IMPL
//----------------------------------------------------------------------------------------------------------------







// FMAs
//----------------------------------------------------------------------------------------------------------------
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> fmadd(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b, const SIMDVector<T,ABI> &c) {
    return a*b+c;
}
// Note that fmsub alternatively adds and subtracts simd vectors
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> fmsub(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b, const SIMDVector<T,ABI> &c) {
    return a*b-c;
}

#ifdef FASTOR_FMA_IMPL

template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> fmadd<float,simd_abi::sse>(
    const SIMDVector<float,simd_abi::sse> &a, const SIMDVector<float,simd_abi::sse> &b, const SIMDVector<float,simd_abi::sse> &c) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_fmadd_ps(a.value,b.value,c.value);
    return out;
}
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> fmadd<float,simd_abi::avx>(
    const SIMDVector<float,simd_abi::avx> &a, const SIMDVector<float,simd_abi::avx> &b, const SIMDVector<float,simd_abi::avx> &c) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_fmadd_ps(a.value,b.value,c.value);
    return out;
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> fmadd<double,simd_abi::sse>(
    const SIMDVector<double,simd_abi::sse> &a, const SIMDVector<double,simd_abi::sse> &b, const SIMDVector<double,simd_abi::sse> &c) {
    SIMDVector<double,simd_abi::sse> out;
    out.value = _mm_fmadd_pd(a.value,b.value,c.value);
    return out;
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> fmadd<double,simd_abi::avx>(
    const SIMDVector<double,simd_abi::avx> &a, const SIMDVector<double,simd_abi::avx> &b, const SIMDVector<double,simd_abi::avx> &c) {
    SIMDVector<double,simd_abi::avx> out;
    out.value = _mm256_fmadd_pd(a.value,b.value,c.value);
    return out;
}

template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> fmsub<float,simd_abi::sse>(
    const SIMDVector<float,simd_abi::sse> &a, const SIMDVector<float,simd_abi::sse> &b, const SIMDVector<float,simd_abi::sse> &c) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_fmsub_ps(a.value,b.value,c.value);
    return out;
}
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> fmsub<float,simd_abi::avx>(
    const SIMDVector<float,simd_abi::avx> &a, const SIMDVector<float,simd_abi::avx> &b, const SIMDVector<float,simd_abi::avx> &c) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_fmsub_ps(a.value,b.value,c.value);
    return out;
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> fmsub<double,simd_abi::sse>(
    const SIMDVector<double,simd_abi::sse> &a, const SIMDVector<double,simd_abi::sse> &b, const SIMDVector<double,simd_abi::sse> &c) {
    SIMDVector<double,simd_abi::sse> out;
    out.value = _mm_fmsub_pd(a.value,b.value,c.value);
    return out;
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> fmsub<double,simd_abi::avx>(
    const SIMDVector<double,simd_abi::avx> &a, const SIMDVector<double,simd_abi::avx> &b, const SIMDVector<double,simd_abi::avx> &c) {
    SIMDVector<double,simd_abi::avx> out;
    out.value = _mm256_fmsub_pd(a.value,b.value,c.value);
    return out;
}

#endif
//----------------------------------------------------------------------------------------------------------------

}

#endif // SIMDVECTOR_H

