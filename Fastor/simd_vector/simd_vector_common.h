#ifndef SIMD_VECTOR_COMMON_H
#define SIMD_VECTOR_COMMON_H

#include "Fastor/meta/meta.h"
#include "Fastor/simd_vector/simd_vector_base.h"
#include "Fastor/simd_vector/simd_vector_scalar.h"
#include "Fastor/simd_vector/simd_vector_float.h"
#include "Fastor/simd_vector/simd_vector_double.h"
#include "Fastor/simd_vector/simd_vector_int32.h"
#include "Fastor/simd_vector/simd_vector_int64.h"
#include "Fastor/simd_vector/simd_vector_complex_scalar.h"
#include "Fastor/simd_vector/simd_vector_complex_float.h"
#include "Fastor/simd_vector/simd_vector_complex_double.h"


namespace Fastor {

/* Common functions for all SIMDVector types
*/

// This is for generic use in tensor expressions that need a uniform simd type
// between all of them
//----------------------------------------------------------------------------------------------------------//
template<typename TT>
struct choose_best_simd_vector {
    using T = remove_cv_ref_t<TT>;
    using type = typename std::conditional< std::is_same<T,float>::value                    ||
                                            std::is_same<T,double>::value                   ||
                                            std::is_same<T,std::complex<float>>::value      ||
                                            std::is_same<T,std::complex<double>>::value     ||
                                            std::is_same<T,int32_t>::value                  ||
                                            std::is_same<T,int64_t>::value,
                                            SIMDVector<T,simd_abi::native>,
                                            SIMDVector<T,simd_abi::scalar>
                >::type;
};

// helper function
template<typename T>
using choose_best_simd_vector_t = typename choose_best_simd_vector<T>::type;
//----------------------------------------------------------------------------------------------------------//


//----------------------------------------------------------------------------------------------------------//
// Get architecture native size/width of the SIMDVector - for instance for [T=double and AVX512] returns 8
template<typename T>
static constexpr size_t native_simd_size_v = internal::get_simd_vector_size<SIMDVector<T,simd_abi::native>>::value;
// Get Fastor supported size/width of the SIMDVector
template<typename T>
static constexpr size_t simd_size_v = internal::get_simd_vector_size<choose_best_simd_vector_t<T>>::value;
//----------------------------------------------------------------------------------------------------------//


// Broadcasting vectorisation on general strides [gather operations]
//----------------------------------------------------------------------------------------------------------------
// 1 word scalar
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==1 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==8,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int ) {
    vec.set(data[idx]);
}
// 1 word in a 2 - for compatibility
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==1 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==16,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+general_stride],data[idx]);
}
// 1 word in a 4 - for compatibility
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==1 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==32,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+3*general_stride],data[idx+2*general_stride],
            data[idx+general_stride],data[idx]);
}
// 1 word in a 8 - for compatibility
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==1 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==64,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+7*general_stride],data[idx+6*general_stride],
            data[idx+5*general_stride],data[idx+4*general_stride],
            data[idx+3*general_stride],data[idx+2*general_stride],
            data[idx+general_stride],data[idx]);
}
// 1 word - sse
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==1 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==128,bool>::type=0>
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
// 1 word avx
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==1 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==256,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+31*general_stride],data[idx+30*general_stride],
            data[idx+29*general_stride],data[idx+28*general_stride],
            data[idx+27*general_stride],data[idx+26*general_stride],
            data[idx+25*general_stride],data[idx+24*general_stride],
            data[idx+23*general_stride],data[idx+22*general_stride],
            data[idx+21*general_stride],data[idx+20*general_stride],
            data[idx+19*general_stride],data[idx+18*general_stride],
            data[idx+17*general_stride],data[idx+16*general_stride],
            data[idx+15*general_stride],data[idx+14*general_stride],
            data[idx+13*general_stride],data[idx+12*general_stride],
            data[idx+11*general_stride],data[idx+10*general_stride],
            data[idx+9 *general_stride],data[idx+8 *general_stride],
            data[idx+7 *general_stride],data[idx+6 *general_stride],
            data[idx+5 *general_stride],data[idx+4 *general_stride],
            data[idx+3 *general_stride],data[idx+2 *general_stride],
            data[idx+general_stride]   ,data[idx]);
}
// 1 word avx512
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==1 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==512,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+63*general_stride],data[idx+62*general_stride],
            data[idx+61*general_stride],data[idx+60*general_stride],
            data[idx+59*general_stride],data[idx+58*general_stride],
            data[idx+57*general_stride],data[idx+56*general_stride],
            data[idx+55*general_stride],data[idx+54*general_stride],
            data[idx+53*general_stride],data[idx+52*general_stride],
            data[idx+51*general_stride],data[idx+50*general_stride],
            data[idx+49*general_stride],data[idx+48*general_stride],
            data[idx+47*general_stride],data[idx+46*general_stride],
            data[idx+45*general_stride],data[idx+44*general_stride],
            data[idx+43*general_stride],data[idx+42*general_stride],
            data[idx+41*general_stride],data[idx+40*general_stride],
            data[idx+39*general_stride],data[idx+38*general_stride],
            data[idx+37*general_stride],data[idx+36*general_stride],
            data[idx+35*general_stride],data[idx+34*general_stride],
            data[idx+33*general_stride],data[idx+32*general_stride],
            data[idx+31*general_stride],data[idx+30*general_stride],
            data[idx+29*general_stride],data[idx+28*general_stride],
            data[idx+27*general_stride],data[idx+26*general_stride],
            data[idx+25*general_stride],data[idx+24*general_stride],
            data[idx+23*general_stride],data[idx+22*general_stride],
            data[idx+21*general_stride],data[idx+20*general_stride],
            data[idx+19*general_stride],data[idx+18*general_stride],
            data[idx+17*general_stride],data[idx+16*general_stride],
            data[idx+15*general_stride],data[idx+14*general_stride],
            data[idx+13*general_stride],data[idx+12*general_stride],
            data[idx+11*general_stride],data[idx+10*general_stride],
            data[idx+9 *general_stride],data[idx+8 *general_stride],
            data[idx+7 *general_stride],data[idx+6 *general_stride],
            data[idx+5 *general_stride],data[idx+4 *general_stride],
            data[idx+3 *general_stride],data[idx+2 *general_stride],
            data[idx+general_stride]   ,data[idx]);
}

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
    vec.set(data[idx+general_stride],data[idx]);
}
// 16 word AVX
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==16 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==256,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+3*general_stride],data[idx+2*general_stride],
            data[idx+general_stride],data[idx]);
}
// 16 word AVX 512
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==16 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==512,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, int idx, int general_stride) {
    vec.set(data[idx+7*general_stride],data[idx+6*general_stride],
            data[idx+5*general_stride],data[idx+4*general_stride],
            data[idx+3*general_stride],data[idx+2*general_stride],
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
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, const std::array<int,2> &a) {
    vec.set(data[a[1]],data[a[0]]);
}
// 16 word AVX
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==16 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==256,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, const std::array<int,4> a) {
    vec.set(data[a[3]],data[a[2]],data[a[1]],data[a[0]]);
}
// 16 word AVX 512
template<typename T, typename ABI,
         typename std::enable_if<sizeof(T)==16 && internal::get_simd_vector_size<SIMDVector<T,ABI>>::bitsize==512,bool>::type=0>
FASTOR_INLINE void vector_setter(SIMDVector<T,ABI> &vec, const T *data, const std::array<int,8> a) {
    vec.set(data[a[7]],data[a[6]],data[a[5]],data[a[4]],
            data[a[3]],data[a[2]],data[a[1]],data[a[0]]);
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
#ifdef FASTOR_AVX2_IMPL
template<>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
maskload<SIMDVector<std::complex<double>,simd_abi::avx>>(const std::complex<double> * FASTOR_RESTRICT a, const int (&maska)[4]) {
    // Split the mask in to a higher and lower part - we need two masks for this
    __m256i mask0 = _mm256_set_epi64x(maska[2],maska[2],maska[3],maska[3]);
    __m256d lo    = _mm256_maskload_pd(reinterpret_cast<const double*>(a  ), (__m256i) mask0);
    __m256i mask1 = _mm256_set_epi64x(maska[0],maska[0],maska[1],maska[1]);
    __m256d hi    = _mm256_maskload_pd(reinterpret_cast<const double*>(a+2), (__m256i) mask1);
    __m256d value_r, value_i;
    arrange_from_load(value_r, value_i, lo, hi);
    return SIMDVector<std::complex<double>,simd_abi::avx>(value_r,value_i);
}
template<>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
maskload<SIMDVector<std::complex<float>,simd_abi::avx>>(const std::complex<float> * FASTOR_RESTRICT a, const int (&maska)[8]) {
    // Split the mask in to a higher and lower part - we need two masks for this
    __m256i mask0 = _mm256_set_epi32(maska[4],maska[4],maska[5],maska[5],maska[6],maska[6],maska[7],maska[7]);
    __m256 lo     = _mm256_maskload_ps(reinterpret_cast<const float*>(a  ), (__m256i) mask0);
    __m256i mask1 = _mm256_set_epi32(maska[0],maska[0],maska[1],maska[1],maska[2],maska[2],maska[3],maska[3]);
    __m256 hi     = _mm256_maskload_ps(reinterpret_cast<const float*>(a+4), (__m256i) mask1);
    __m256 value_r, value_i;
    arrange_from_load(value_r, value_i, lo, hi);
    return SIMDVector<std::complex<float>,simd_abi::avx>(value_r,value_i);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx>
maskload<SIMDVector<double,simd_abi::avx>>(const double * FASTOR_RESTRICT a, const int (&maska)[4]) {
    __m256i mask = _mm256_set_epi64x(maska[0],maska[1],maska[2],maska[3]);
    return _mm256_maskload_pd(a,(__m256i) mask);
}
template<>
FASTOR_INLINE SIMDVector<Int64,simd_abi::avx>
maskload<SIMDVector<Int64,simd_abi::avx>>(const Int64 * FASTOR_RESTRICT a, const int (&maska)[4]) {
    __m256i mask = _mm256_set_epi64x(maska[0],maska[1],maska[2],maska[3]);
    return _mm256_maskload_epi64(reinterpret_cast<const long long int*>(a),(__m256i) mask);
}
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx>
maskload<SIMDVector<float,simd_abi::avx>>(const float * FASTOR_RESTRICT a, const int (&maska)[8]) {
    __m256i mask = _mm256_set_epi32(maska[0],maska[1],maska[2],maska[3],maska[4],maska[5],maska[6],maska[7]);
    return _mm256_maskload_ps(a,(__m256i) mask);
}
template<>
FASTOR_INLINE SIMDVector<int,simd_abi::avx>
maskload<SIMDVector<int,simd_abi::avx>>(const int * FASTOR_RESTRICT a, const int (&maska)[8]) {
    __m256i mask = _mm256_set_epi32(maska[0],maska[1],maska[2],maska[3],maska[4],maska[5],maska[6],maska[7]);
    return _mm256_maskload_epi32(a,(__m256i) mask);
}
template<>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
maskload<SIMDVector<std::complex<double>,simd_abi::sse>>(const std::complex<double> * FASTOR_RESTRICT a, const int (&maska)[2]) {
    // Split the mask in to a higher and lower part - we need two masks for this
    __m128i mask0 = _mm_set_epi64x(maska[1],maska[1]);
    __m128d lo    = _mm_maskload_pd(reinterpret_cast<const double*>(a  ), (__m128i) mask0);
    __m128i mask1 = _mm_set_epi64x(maska[0],maska[0]);
    __m128d hi    = _mm_maskload_pd(reinterpret_cast<const double*>(a+1), (__m128i) mask1);
    __m128d value_r, value_i;
    arrange_from_load(value_r, value_i, lo, hi);
    return SIMDVector<std::complex<double>,simd_abi::sse>(value_r,value_i);
}
template<>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
maskload<SIMDVector<std::complex<float>,simd_abi::sse>>(const std::complex<float> * FASTOR_RESTRICT a, const int (&maska)[4]) {
    // Split the mask in to a higher and lower part - we need two masks for this
    __m128i mask0 = _mm_set_epi32(maska[2],maska[2],maska[3],maska[3]);
    __m128  lo    = _mm_maskload_ps(reinterpret_cast<const float*>(a  ), (__m128i) mask0);
    __m128i mask1 = _mm_set_epi32(maska[0],maska[0],maska[1],maska[1]);
    __m128  hi    = _mm_maskload_ps(reinterpret_cast<const float*>(a+2), (__m128i) mask1);
    __m128  value_r, value_i;
    arrange_from_load(value_r, value_i, lo, hi);
    return SIMDVector<std::complex<float>,simd_abi::sse>(value_r,value_i);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse>
maskload<SIMDVector<double,simd_abi::sse>>(const double * FASTOR_RESTRICT a, const int (&maska)[2]) {
    __m128i mask = _mm_set_epi64x(maska[0],maska[1]);
    return _mm_maskload_pd(a,(__m128i) mask);
}
template<>
FASTOR_INLINE SIMDVector<Int64,simd_abi::sse>
maskload<SIMDVector<Int64,simd_abi::sse>>(const Int64 * FASTOR_RESTRICT a, const int (&maska)[2]) {
    __m128i mask = _mm_set_epi64x(maska[0],maska[1]);
    return _mm_maskload_epi64(reinterpret_cast<const long long int*>(a),(__m128i) mask);
}
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse>
maskload<SIMDVector<float,simd_abi::sse>>(const float * FASTOR_RESTRICT a, const int (&maska)[4]) {
    __m128i mask = _mm_set_epi32(maska[0],maska[1],maska[2],maska[3]);
    return _mm_maskload_ps(a,(__m128i) mask);
}
template<>
FASTOR_INLINE SIMDVector<int,simd_abi::sse>
maskload<SIMDVector<int,simd_abi::sse>>(const int * FASTOR_RESTRICT a, const int (&maska)[4]) {
    __m128i mask = _mm_set_epi32(maska[0],maska[1],maska[2],maska[3]);
    return _mm_maskload_epi32(a,(__m128i) mask);
}
#endif // FASTOR_AVX2_IMPL


template<typename V>
FASTOR_INLINE
void maskstore(typename V::scalar_value_type * FASTOR_RESTRICT a, const int (&maska)[V::Size], V& v) {
    for (FASTOR_INDEX i=0; i<V::Size; ++i) {
        if (maska[i] == -1) {
            a[V::Size - i - 1] = v[V::Size - i - 1];
        }
    }
}
#ifdef FASTOR_AVX2_IMPL
template<>
FASTOR_INLINE
void maskstore(std::complex<double> * FASTOR_RESTRICT a, const int (&maska)[4], SIMDVector<std::complex<double>,simd_abi::avx> &v) {
    // Split the mask in to a higher and lower part - we need two masks for this
    __m256i mask0 = _mm256_set_epi64x(maska[2],maska[2],maska[3],maska[3]);
    __m256i mask1 = _mm256_set_epi64x(maska[0],maska[0],maska[1],maska[1]);
    __m256d lo, hi;
    arrange_for_store(lo, hi, v.value_r, v.value_i);
    _mm256_maskstore_pd(reinterpret_cast<double*>(a  ), (__m256i) mask0, lo);
    _mm256_maskstore_pd(reinterpret_cast<double*>(a+2), (__m256i) mask1, hi);
}
template<>
FASTOR_INLINE
void maskstore(std::complex<float> * FASTOR_RESTRICT a, const int (&maska)[8], SIMDVector<std::complex<float>,simd_abi::avx> &v) {
    // Split the mask in to a higher and lower part - we need two masks for this
    __m256i mask0 = _mm256_set_epi32(maska[4],maska[4],maska[5],maska[5],maska[6],maska[6],maska[7],maska[7]);
    __m256i mask1 = _mm256_set_epi32(maska[0],maska[0],maska[1],maska[1],maska[2],maska[2],maska[3],maska[3]);
    __m256 lo, hi;
    arrange_for_store(lo, hi, v.value_r, v.value_i);
    _mm256_maskstore_ps(reinterpret_cast<float*>(a  ), (__m256i) mask0, lo);
    _mm256_maskstore_ps(reinterpret_cast<float*>(a+4), (__m256i) mask1, hi);
}
template<>
FASTOR_INLINE
void maskstore(double * FASTOR_RESTRICT a, const int (&maska)[4], SIMDVector<double,simd_abi::avx> &v) {
    __m256i mask = _mm256_set_epi64x(maska[0],maska[1],maska[2],maska[3]);
    _mm256_maskstore_pd(a,(__m256i) mask, v.value);
}
template<>
FASTOR_INLINE
void maskstore(Int64 * FASTOR_RESTRICT a, const int (&maska)[4], SIMDVector<Int64,simd_abi::avx> &v) {
    __m256i mask = _mm256_set_epi64x(maska[0],maska[1],maska[2],maska[3]);
    _mm256_maskstore_epi64(reinterpret_cast<long long int*>(a),(__m256i) mask, v.value);
}
template<>
FASTOR_INLINE
void maskstore(float * FASTOR_RESTRICT a, const int (&maska)[8], SIMDVector<float,simd_abi::avx> &v) {
    __m256i mask = _mm256_set_epi32(maska[0],maska[1],maska[2],maska[3],maska[4],maska[5],maska[6],maska[7]);
    _mm256_maskstore_ps(a,(__m256i) mask, v.value);
}
template<>
FASTOR_INLINE
void maskstore(int * FASTOR_RESTRICT a, const int (&maska)[8], SIMDVector<int,simd_abi::avx> &v) {
    __m256i mask = _mm256_set_epi32(maska[0],maska[1],maska[2],maska[3],maska[4],maska[5],maska[6],maska[7]);
    _mm256_maskstore_epi32(a,(__m256i) mask, v.value);
}
template<>
FASTOR_INLINE
void maskstore(std::complex<double> * FASTOR_RESTRICT a, const int (&maska)[2], SIMDVector<std::complex<double>,simd_abi::sse> &v) {
    // Split the mask in to a higher and lower part - we need two masks for this
    __m128i mask0 = _mm_set_epi64x(maska[1],maska[1]);
    __m128i mask1 = _mm_set_epi64x(maska[0],maska[0]);
    __m128d lo, hi;
    arrange_for_store(lo, hi, v.value_r, v.value_i);
    _mm_maskstore_pd(reinterpret_cast<double*>(a  ), (__m128i) mask0, lo);
    _mm_maskstore_pd(reinterpret_cast<double*>(a+1), (__m128i) mask1, hi);
}
template<>
FASTOR_INLINE
void maskstore(std::complex<float> * FASTOR_RESTRICT a, const int (&maska)[4], SIMDVector<std::complex<float>,simd_abi::sse> &v) {
    // Split the mask in to a higher and lower part - we need two masks for this
    __m128i mask0 = _mm_set_epi32(maska[2],maska[2],maska[3],maska[3]);
    __m128i mask1 = _mm_set_epi32(maska[0],maska[0],maska[1],maska[1]);
    __m128 lo, hi;
    arrange_for_store(lo, hi, v.value_r, v.value_i);
    _mm_maskstore_ps(reinterpret_cast<float*>(a  ), (__m128i) mask0, lo);
    _mm_maskstore_ps(reinterpret_cast<float*>(a+2), (__m128i) mask1, hi);
}
template<>
FASTOR_INLINE
void maskstore(double * FASTOR_RESTRICT a, const int (&maska)[2], SIMDVector<double,simd_abi::sse> &v) {
    __m128i mask = _mm_set_epi64x(maska[0],maska[1]);
    _mm_maskstore_pd(a,(__m128i) mask, v.value);
}
template<>
FASTOR_INLINE
void maskstore(Int64 * FASTOR_RESTRICT a, const int (&maska)[2], SIMDVector<Int64,simd_abi::sse> &v) {
    __m128i mask = _mm_set_epi64x(maska[0],maska[1]);
    _mm_maskstore_epi64(reinterpret_cast<long long int*>(a),(__m128i) mask, v.value);
}
template<>
FASTOR_INLINE
void maskstore(float * FASTOR_RESTRICT a, const int (&maska)[4], SIMDVector<float,simd_abi::sse> &v) {
    __m128i mask = _mm_set_epi32(maska[0],maska[1],maska[2],maska[3]);
    _mm_maskstore_ps(a,(__m128i) mask, v.value);
}
template<>
FASTOR_INLINE
void maskstore(int * FASTOR_RESTRICT a, const int (&maska)[4], SIMDVector<int,simd_abi::sse> &v) {
    __m128i mask = _mm_set_epi32(maska[0],maska[1],maska[2],maska[3]);
    _mm_maskstore_epi32(a,(__m128i) mask, v.value);
}
#endif // FASTOR_AVX2_IMPL
//----------------------------------------------------------------------------------------------------------------







// FMAs
//----------------------------------------------------------------------------------------------------------------
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> fmadd(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b, const SIMDVector<T,ABI> &c) {
    return a*b+c;
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> fmsub(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b, const SIMDVector<T,ABI> &c) {
    return a*b-c;
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> fnmadd(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b, const SIMDVector<T,ABI> &c) {
    return c-a*b;
}

#ifdef FASTOR_FMA_IMPL
// fmadd
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
#ifdef FASTOR_AVX512F_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> fmadd<float,simd_abi::avx512>(
    const SIMDVector<float,simd_abi::avx512> &a, const SIMDVector<float,simd_abi::avx512> &b, const SIMDVector<float,simd_abi::avx512> &c) {
    SIMDVector<float,simd_abi::avx512> out;
    out.value = _mm512_fmadd_ps(a.value,b.value,c.value);
    return out;
}
#endif
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
#ifdef FASTOR_AVX512F_IMPL
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> fmadd<double,simd_abi::avx512>(
    const SIMDVector<double,simd_abi::avx512> &a, const SIMDVector<double,simd_abi::avx512> &b, const SIMDVector<double,simd_abi::avx512> &c) {
    SIMDVector<double,simd_abi::avx512> out;
    out.value = _mm512_fmadd_pd(a.value,b.value,c.value);
    return out;
}
#endif
template<>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse> fmadd<std::complex<float>,simd_abi::sse>(
    const SIMDVector<std::complex<float>,simd_abi::sse> &a,
    const SIMDVector<std::complex<float>,simd_abi::sse> &b,
    const SIMDVector<std::complex<float>,simd_abi::sse> &c) {
    SIMDVector<std::complex<float>,simd_abi::sse> out;
    // ar*br - ai*bi + cr
    out.value_r = _mm_fnmadd_ps(a.value_i,b.value_i,_mm_fmadd_ps(a.value_r,b.value_r,c.value_r));
    // ar*bi + ai*br + ci
    out.value_i = _mm_fmadd_ps (a.value_i,b.value_r,_mm_fmadd_ps(a.value_r,b.value_i,c.value_i));
    return out;
}
template<>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx> fmadd<std::complex<float>,simd_abi::avx>(
    const SIMDVector<std::complex<float>,simd_abi::avx> &a,
    const SIMDVector<std::complex<float>,simd_abi::avx> &b,
    const SIMDVector<std::complex<float>,simd_abi::avx> &c) {
    SIMDVector<std::complex<float>,simd_abi::avx> out;
    // ar*br - ai*bi + cr
    out.value_r = _mm256_fnmadd_ps(a.value_i,b.value_i,_mm256_fmadd_ps(a.value_r,b.value_r,c.value_r));
    // ar*bi + ai*br + ci
    out.value_i = _mm256_fmadd_ps (a.value_i,b.value_r,_mm256_fmadd_ps(a.value_r,b.value_i,c.value_i));
    return out;
}
#ifdef FASTOR_AVX512F_IMPL
template<>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512> fmadd<std::complex<float>,simd_abi::avx512>(
    const SIMDVector<std::complex<float>,simd_abi::avx512> &a,
    const SIMDVector<std::complex<float>,simd_abi::avx512> &b,
    const SIMDVector<std::complex<float>,simd_abi::avx512> &c) {
    SIMDVector<std::complex<float>,simd_abi::avx512> out;
    // ar*br - ai*bi + cr
    out.value_r = _mm512_fnmadd_ps(a.value_i,b.value_i,_mm512_fmadd_ps(a.value_r,b.value_r,c.value_r));
    // ar*bi + ai*br + ci
    out.value_i = _mm512_fmadd_ps (a.value_i,b.value_r,_mm512_fmadd_ps(a.value_r,b.value_i,c.value_i));
    return out;
}
#endif
template<>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse> fmadd<std::complex<double>,simd_abi::sse>(
    const SIMDVector<std::complex<double>,simd_abi::sse> &a,
    const SIMDVector<std::complex<double>,simd_abi::sse> &b,
    const SIMDVector<std::complex<double>,simd_abi::sse> &c) {
    SIMDVector<std::complex<double>,simd_abi::sse> out;
    // ar*br - ai*bi + cr
    out.value_r = _mm_fnmadd_pd(a.value_i,b.value_i,_mm_fmadd_pd(a.value_r,b.value_r,c.value_r));
    // ar*bi + ai*br + ci
    out.value_i = _mm_fmadd_pd (a.value_i,b.value_r,_mm_fmadd_pd(a.value_r,b.value_i,c.value_i));
    return out;
}
template<>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx> fmadd<std::complex<double>,simd_abi::avx>(
    const SIMDVector<std::complex<double>,simd_abi::avx> &a,
    const SIMDVector<std::complex<double>,simd_abi::avx> &b,
    const SIMDVector<std::complex<double>,simd_abi::avx> &c) {
    SIMDVector<std::complex<double>,simd_abi::avx> out;
    // ar*br - ai*bi + cr
    out.value_r = _mm256_fnmadd_pd(a.value_i,b.value_i,_mm256_fmadd_pd(a.value_r,b.value_r,c.value_r));
    // ar*bi + ai*br + ci
    out.value_i = _mm256_fmadd_pd (a.value_i,b.value_r,_mm256_fmadd_pd(a.value_r,b.value_i,c.value_i));
    return out;
}
#ifdef FASTOR_AVX512F_IMPL
template<>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512> fmadd<std::complex<double>,simd_abi::avx512>(
    const SIMDVector<std::complex<double>,simd_abi::avx512> &a,
    const SIMDVector<std::complex<double>,simd_abi::avx512> &b,
    const SIMDVector<std::complex<double>,simd_abi::avx512> &c) {
    SIMDVector<std::complex<double>,simd_abi::avx512> out;
    // ar*br - ai*bi + cr
    out.value_r = _mm512_fnmadd_pd(a.value_i,b.value_i,_mm512_fmadd_pd(a.value_r,b.value_r,c.value_r));
    // ar*bi + ai*br + ci
    out.value_i = _mm512_fmadd_pd (a.value_i,b.value_r,_mm512_fmadd_pd(a.value_r,b.value_i,c.value_i));
    return out;
}
#endif

// fmsub
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
#ifdef FASTOR_AVX512F_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> fmsub<float,simd_abi::avx512>(
    const SIMDVector<float,simd_abi::avx512> &a, const SIMDVector<float,simd_abi::avx512> &b, const SIMDVector<float,simd_abi::avx512> &c) {
    SIMDVector<float,simd_abi::avx512> out;
    out.value = _mm512_fmsub_ps(a.value,b.value,c.value);
    return out;
}
#endif
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
#ifdef FASTOR_AVX512F_IMPL
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> fmsub<double,simd_abi::avx512>(
    const SIMDVector<double,simd_abi::avx512> &a, const SIMDVector<double,simd_abi::avx512> &b, const SIMDVector<double,simd_abi::avx512> &c) {
    SIMDVector<double,simd_abi::avx512> out;
    out.value = _mm512_fmsub_pd(a.value,b.value,c.value);
    return out;
}
#endif
template<>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse> fmsub<std::complex<float>,simd_abi::sse>(
    const SIMDVector<std::complex<float>,simd_abi::sse> &a,
    const SIMDVector<std::complex<float>,simd_abi::sse> &b,
    const SIMDVector<std::complex<float>,simd_abi::sse> &c) {
    SIMDVector<std::complex<float>,simd_abi::sse> out;
    // ar*br - ai*bi - cr
    out.value_r = _mm_fnmadd_ps(a.value_i,b.value_i,_mm_fmsub_ps(a.value_r,b.value_r,c.value_r));
    // ar*bi + ai*br - ci
    out.value_i = _mm_fmadd_ps (a.value_i,b.value_r,_mm_fmsub_ps(a.value_r,b.value_i,c.value_i));
    return out;
}
template<>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx> fmsub<std::complex<float>,simd_abi::avx>(
    const SIMDVector<std::complex<float>,simd_abi::avx> &a,
    const SIMDVector<std::complex<float>,simd_abi::avx> &b,
    const SIMDVector<std::complex<float>,simd_abi::avx> &c) {
    SIMDVector<std::complex<float>,simd_abi::avx> out;
    // ar*br - ai*bi + cr
    out.value_r = _mm256_fnmadd_ps(a.value_i,b.value_i,_mm256_fmsub_ps(a.value_r,b.value_r,c.value_r));
    // ar*bi + ai*br + ci
    out.value_i = _mm256_fmadd_ps (a.value_i,b.value_r,_mm256_fmsub_ps(a.value_r,b.value_i,c.value_i));
    return out;
}
#ifdef FASTOR_AVX512F_IMPL
template<>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512> fmsub<std::complex<float>,simd_abi::avx512>(
    const SIMDVector<std::complex<float>,simd_abi::avx512> &a,
    const SIMDVector<std::complex<float>,simd_abi::avx512> &b,
    const SIMDVector<std::complex<float>,simd_abi::avx512> &c) {
    SIMDVector<std::complex<float>,simd_abi::avx512> out;
    // ar*br - ai*bi + cr
    out.value_r = _mm512_fnmadd_ps(a.value_i,b.value_i,_mm512_fmsub_ps(a.value_r,b.value_r,c.value_r));
    // ar*bi + ai*br + ci
    out.value_i = _mm512_fmadd_ps (a.value_i,b.value_r,_mm512_fmsub_ps(a.value_r,b.value_i,c.value_i));
    return out;
}
#endif
template<>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse> fmsub<std::complex<double>,simd_abi::sse>(
    const SIMDVector<std::complex<double>,simd_abi::sse> &a,
    const SIMDVector<std::complex<double>,simd_abi::sse> &b,
    const SIMDVector<std::complex<double>,simd_abi::sse> &c) {
    SIMDVector<std::complex<double>,simd_abi::sse> out;
    // ar*br - ai*bi - cr
    out.value_r = _mm_fnmadd_pd(a.value_i,b.value_i,_mm_fmsub_pd(a.value_r,b.value_r,c.value_r));
    // ar*bi + ai*br - ci
    out.value_i = _mm_fmadd_pd (a.value_i,b.value_r,_mm_fmsub_pd(a.value_r,b.value_i,c.value_i));
    return out;
}
template<>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx> fmsub<std::complex<double>,simd_abi::avx>(
    const SIMDVector<std::complex<double>,simd_abi::avx> &a,
    const SIMDVector<std::complex<double>,simd_abi::avx> &b,
    const SIMDVector<std::complex<double>,simd_abi::avx> &c) {
    SIMDVector<std::complex<double>,simd_abi::avx> out;
    // ar*br - ai*bi + cr
    out.value_r = _mm256_fnmadd_pd(a.value_i,b.value_i,_mm256_fmsub_pd(a.value_r,b.value_r,c.value_r));
    // ar*bi + ai*br + ci
    out.value_i = _mm256_fmadd_pd (a.value_i,b.value_r,_mm256_fmsub_pd(a.value_r,b.value_i,c.value_i));
    return out;
}
#ifdef FASTOR_AVX512F_IMPL
template<>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512> fmsub<std::complex<double>,simd_abi::avx512>(
    const SIMDVector<std::complex<double>,simd_abi::avx512> &a,
    const SIMDVector<std::complex<double>,simd_abi::avx512> &b,
    const SIMDVector<std::complex<double>,simd_abi::avx512> &c) {
    SIMDVector<std::complex<double>,simd_abi::avx512> out;
    // ar*br - ai*bi + cr
    out.value_r = _mm512_fnmadd_pd(a.value_i,b.value_i,_mm512_fmsub_pd(a.value_r,b.value_r,c.value_r));
    // ar*bi + ai*br + ci
    out.value_i = _mm512_fmadd_pd (a.value_i,b.value_r,_mm512_fmsub_pd(a.value_r,b.value_i,c.value_i));
    return out;
}
#endif

// fnmadd
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> fnmadd<float,simd_abi::sse>(
    const SIMDVector<float,simd_abi::sse> &a, const SIMDVector<float,simd_abi::sse> &b, const SIMDVector<float,simd_abi::sse> &c) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_fnmadd_ps(a.value,b.value,c.value);
    return out;
}
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> fnmadd<float,simd_abi::avx>(
    const SIMDVector<float,simd_abi::avx> &a, const SIMDVector<float,simd_abi::avx> &b, const SIMDVector<float,simd_abi::avx> &c) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_fnmadd_ps(a.value,b.value,c.value);
    return out;
}
#ifdef FASTOR_AVX512F_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> fnmadd<float,simd_abi::avx512>(
    const SIMDVector<float,simd_abi::avx512> &a, const SIMDVector<float,simd_abi::avx512> &b, const SIMDVector<float,simd_abi::avx512> &c) {
    SIMDVector<float,simd_abi::avx512> out;
    out.value = _mm512_fnmadd_ps(a.value,b.value,c.value);
    return out;
}
#endif
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> fnmadd<double,simd_abi::sse>(
    const SIMDVector<double,simd_abi::sse> &a, const SIMDVector<double,simd_abi::sse> &b, const SIMDVector<double,simd_abi::sse> &c) {
    SIMDVector<double,simd_abi::sse> out;
    out.value = _mm_fnmadd_pd(a.value,b.value,c.value);
    return out;
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> fnmadd<double,simd_abi::avx>(
    const SIMDVector<double,simd_abi::avx> &a, const SIMDVector<double,simd_abi::avx> &b, const SIMDVector<double,simd_abi::avx> &c) {
    SIMDVector<double,simd_abi::avx> out;
    out.value = _mm256_fnmadd_pd(a.value,b.value,c.value);
    return out;
}
#ifdef FASTOR_AVX512F_IMPL
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> fnmadd<double,simd_abi::avx512>(
    const SIMDVector<double,simd_abi::avx512> &a, const SIMDVector<double,simd_abi::avx512> &b, const SIMDVector<double,simd_abi::avx512> &c) {
    SIMDVector<double,simd_abi::avx512> out;
    out.value = _mm512_fnmadd_pd(a.value,b.value,c.value);
    return out;
}
#endif
template<>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse> fnmadd<std::complex<float>,simd_abi::sse>(
    const SIMDVector<std::complex<float>,simd_abi::sse> &a,
    const SIMDVector<std::complex<float>,simd_abi::sse> &b,
    const SIMDVector<std::complex<float>,simd_abi::sse> &c) {
    SIMDVector<std::complex<float>,simd_abi::sse> out;
    // -ar*br + ai*bi + cr
    out.value_r = _mm_fmadd_ps (a.value_i,b.value_i,_mm_fnmadd_ps(a.value_r,b.value_r,c.value_r));
    // -ar*bi - ai*br + ci
    out.value_i = _mm_fnmadd_ps(a.value_i,b.value_r,_mm_fnmadd_ps(a.value_r,b.value_i,c.value_i));
    return out;
}
template<>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx> fnmadd<std::complex<float>,simd_abi::avx>(
    const SIMDVector<std::complex<float>,simd_abi::avx> &a,
    const SIMDVector<std::complex<float>,simd_abi::avx> &b,
    const SIMDVector<std::complex<float>,simd_abi::avx> &c) {
    SIMDVector<std::complex<float>,simd_abi::avx> out;
    // -ar*br + ai*bi + cr
    out.value_r = _mm256_fmadd_ps (a.value_i,b.value_i,_mm256_fnmadd_ps(a.value_r,b.value_r,c.value_r));
    // -ar*bi - ai*br + ci
    out.value_i = _mm256_fnmadd_ps(a.value_i,b.value_r,_mm256_fnmadd_ps(a.value_r,b.value_i,c.value_i));
    return out;
}
#ifdef FASTOR_AVX512F_IMPL
template<>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512> fnmadd<std::complex<float>,simd_abi::avx512>(
    const SIMDVector<std::complex<float>,simd_abi::avx512> &a,
    const SIMDVector<std::complex<float>,simd_abi::avx512> &b,
    const SIMDVector<std::complex<float>,simd_abi::avx512> &c) {
    SIMDVector<std::complex<float>,simd_abi::avx512> out;
    // -ar*br + ai*bi + cr
    out.value_r = _mm512_fmadd_ps (a.value_i,b.value_i,_mm512_fnmadd_ps(a.value_r,b.value_r,c.value_r));
    // -ar*bi - ai*br + ci
    out.value_i = _mm512_fnmadd_ps(a.value_i,b.value_r,_mm512_fnmadd_ps(a.value_r,b.value_i,c.value_i));
    return out;
}
#endif
template<>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse> fnmadd<std::complex<double>,simd_abi::sse>(
    const SIMDVector<std::complex<double>,simd_abi::sse> &a,
    const SIMDVector<std::complex<double>,simd_abi::sse> &b,
    const SIMDVector<std::complex<double>,simd_abi::sse> &c) {
    SIMDVector<std::complex<double>,simd_abi::sse> out;
    // -ar*br + ai*bi + cr
    out.value_r = _mm_fmadd_pd (a.value_i,b.value_i,_mm_fnmadd_pd(a.value_r,b.value_r,c.value_r));
    // -ar*bi - ai*br + ci
    out.value_i = _mm_fnmadd_pd(a.value_i,b.value_r,_mm_fnmadd_pd(a.value_r,b.value_i,c.value_i));
    return out;
}
template<>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx> fnmadd<std::complex<double>,simd_abi::avx>(
    const SIMDVector<std::complex<double>,simd_abi::avx> &a,
    const SIMDVector<std::complex<double>,simd_abi::avx> &b,
    const SIMDVector<std::complex<double>,simd_abi::avx> &c) {
    SIMDVector<std::complex<double>,simd_abi::avx> out;
    // -ar*br + ai*bi + cr
    out.value_r = _mm256_fmadd_pd (a.value_i,b.value_i,_mm256_fnmadd_pd(a.value_r,b.value_r,c.value_r));
    // -ar*bi - ai*br + ci
    out.value_i = _mm256_fnmadd_pd(a.value_i,b.value_r,_mm256_fnmadd_pd(a.value_r,b.value_i,c.value_i));
    return out;
}
#ifdef FASTOR_AVX512F_IMPL
template<>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512> fnmadd<std::complex<double>,simd_abi::avx512>(
    const SIMDVector<std::complex<double>,simd_abi::avx512> &a,
    const SIMDVector<std::complex<double>,simd_abi::avx512> &b,
    const SIMDVector<std::complex<double>,simd_abi::avx512> &c) {
    SIMDVector<std::complex<double>,simd_abi::avx512> out;
    // -ar*br + ai*bi + cr
    out.value_r = _mm512_fmadd_pd (a.value_i,b.value_i,_mm512_fnmadd_pd(a.value_r,b.value_r,c.value_r));
    // -ar*bi - ai*br + ci
    out.value_i = _mm512_fnmadd_pd(a.value_i,b.value_r,_mm512_fnmadd_pd(a.value_r,b.value_i,c.value_i));
    return out;
}
#endif

#endif
//----------------------------------------------------------------------------------------------------------------




// Binary comparison ops
//----------------------------------------------------------------------------------------------------------------//
#define FASTOR_MAKE_BINARY_CMP_SIMDVECTORS_OPS_(OP) \
template<typename T, typename ABI> \
FASTOR_INLINE SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> operator OP(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b) { \
    constexpr FASTOR_INDEX Size = SIMDVector<T,ABI>::Size;\
    FASTOR_ARCH_ALIGN T val_a[Size];\
    a.store(val_a);\
    FASTOR_ARCH_ALIGN T val_b[Size];\
    b.store(val_b);\
    SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> out;\
    FASTOR_ARCH_ALIGN bool val_out[Size];\
    out.store(val_out);\
    for (FASTOR_INDEX i=0; i<Size; ++i) {\
        val_out[i] = val_a[i] OP val_b[i];\
    }\
    out.load(val_out);\
    return out;\
}\

FASTOR_MAKE_BINARY_CMP_SIMDVECTORS_OPS_(==)
FASTOR_MAKE_BINARY_CMP_SIMDVECTORS_OPS_(!=)
FASTOR_MAKE_BINARY_CMP_SIMDVECTORS_OPS_(>)
FASTOR_MAKE_BINARY_CMP_SIMDVECTORS_OPS_(<)
FASTOR_MAKE_BINARY_CMP_SIMDVECTORS_OPS_(>=)
FASTOR_MAKE_BINARY_CMP_SIMDVECTORS_OPS_(<=)
FASTOR_MAKE_BINARY_CMP_SIMDVECTORS_OPS_(&&)
FASTOR_MAKE_BINARY_CMP_SIMDVECTORS_OPS_(||)


#define FASTOR_MAKE_BINARY_CMP_SIMDVECTOR_SCALAR_OPS_(OP) \
template<typename T, typename U, typename ABI> \
FASTOR_INLINE SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> operator OP(const SIMDVector<T,ABI> &a, U b) { \
    constexpr FASTOR_INDEX Size = SIMDVector<T,ABI>::Size;\
    FASTOR_ARCH_ALIGN T val_a[Size];\
    a.store(val_a);\
    SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> out;\
    FASTOR_ARCH_ALIGN bool val_out[Size];\
    out.store(val_out);\
    for (FASTOR_INDEX i=0; i<Size; ++i) {\
        val_out[i] = val_a[i] OP T(b);\
    }\
    out.load(val_out);\
    return out;\
}\

FASTOR_MAKE_BINARY_CMP_SIMDVECTOR_SCALAR_OPS_(==)
FASTOR_MAKE_BINARY_CMP_SIMDVECTOR_SCALAR_OPS_(!=)
FASTOR_MAKE_BINARY_CMP_SIMDVECTOR_SCALAR_OPS_(>)
FASTOR_MAKE_BINARY_CMP_SIMDVECTOR_SCALAR_OPS_(<)
FASTOR_MAKE_BINARY_CMP_SIMDVECTOR_SCALAR_OPS_(>=)
FASTOR_MAKE_BINARY_CMP_SIMDVECTOR_SCALAR_OPS_(<=)
FASTOR_MAKE_BINARY_CMP_SIMDVECTOR_SCALAR_OPS_(&&)
FASTOR_MAKE_BINARY_CMP_SIMDVECTOR_SCALAR_OPS_(||)


#define FASTOR_MAKE_BINARY_CMP_SCALAR_SIMDVECTOR_OPS_(OP) \
template<typename T, typename U, typename ABI> \
FASTOR_INLINE SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> operator OP(U a, const SIMDVector<T,ABI> &b) { \
    constexpr FASTOR_INDEX Size = SIMDVector<T,ABI>::Size;\
    FASTOR_ARCH_ALIGN T val_b[Size];\
    b.store(val_b);\
    SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> out;\
    FASTOR_ARCH_ALIGN bool val_out[Size];\
    out.store(val_out);\
    for (FASTOR_INDEX i=0; i<Size; ++i) {\
        val_out[i] = T(a) OP val_b[i];\
    }\
    out.load(val_out);\
    return out;\
}\

FASTOR_MAKE_BINARY_CMP_SCALAR_SIMDVECTOR_OPS_(==)
FASTOR_MAKE_BINARY_CMP_SCALAR_SIMDVECTOR_OPS_(!=)
FASTOR_MAKE_BINARY_CMP_SCALAR_SIMDVECTOR_OPS_(>)
FASTOR_MAKE_BINARY_CMP_SCALAR_SIMDVECTOR_OPS_(<)
FASTOR_MAKE_BINARY_CMP_SCALAR_SIMDVECTOR_OPS_(>=)
FASTOR_MAKE_BINARY_CMP_SCALAR_SIMDVECTOR_OPS_(<=)
FASTOR_MAKE_BINARY_CMP_SCALAR_SIMDVECTOR_OPS_(&&)
FASTOR_MAKE_BINARY_CMP_SCALAR_SIMDVECTOR_OPS_(||)
//----------------------------------------------------------------------------------------------------------------//


} // end of namespace Fastor

#endif // SIMD_VECTOR_COMMON_H
