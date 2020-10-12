#ifndef SLEEF_BACKEND_U10_H
#define SLEEF_BACKEND_U10_H

#if defined(FASTOR_USE_SLEEF_U10) || defined(FASTOR_USE_SLEEF)

#include <sleef.h>

namespace Fastor {

// exp
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> exp(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_expf4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> exp(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_expd2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> exp(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_expf8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> exp(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_expd4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> exp(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_expf16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> exp(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_expd8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// exp2
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> exp2(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_exp2f4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> exp2(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_exp2d2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> exp2(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_exp2f8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> exp2(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_exp2d4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> exp2(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_exp2f16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> exp2(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_exp2d8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// expm1
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> expm1(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_expm1f4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> expm1(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_expm1d2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> expm1(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_expm1f8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> expm1(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_expm1d4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> expm1(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_expm1f16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> expm1(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_expm1d8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// log
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> log(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_logf4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> log(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_logd2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> log(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_logf8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> log(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_logd4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> log(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_logf16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> log(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_logd8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// log10
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> log10(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_log10f4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> log10(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_log10d2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> log10(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_log10f8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> log10(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_log10d4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> log10(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_log10f16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> log10(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_log10d8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// log2
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> log2(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_log2f4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> log2(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_log2d2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> log2(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_log2f8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> log2(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_log2d4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> log2(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_log2f16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> log2(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_log2d8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// log1p
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> log1p(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_log1pf4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> log1p(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_log1pd2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> log1p(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_log1pf8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> log1p(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_log1pd4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> log1p(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_log1pf16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> log1p(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_log1pd8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// pow - other variants are automatically taken care off through the generic overloads
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> pow(const SIMDVector<float,simd_abi::sse> &a, const SIMDVector<float,simd_abi::sse> &b) {
    return Sleef_powf4_u10(a.value, b.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> pow(const SIMDVector<double,simd_abi::sse> &a, const SIMDVector<double,simd_abi::sse> &b) {
    return Sleef_powd2_u10(a.value, b.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> pow(const SIMDVector<float,simd_abi::avx> &a, const SIMDVector<float,simd_abi::avx> &b) {
    return Sleef_powf8_u10(a.value, b.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> pow(const SIMDVector<double,simd_abi::avx> &a, const SIMDVector<double,simd_abi::avx> &b) {
    return Sleef_powd4_u10(a.value, b.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> pow(const SIMDVector<float,simd_abi::avx512> &a, const SIMDVector<float,simd_abi::avx512> &b) {
    return Sleef_powf16_u10(a.value, b.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> pow(const SIMDVector<double,simd_abi::avx512> &a, const SIMDVector<double,simd_abi::avx512> &b) {
    return Sleef_powd8_u10(a.value, b.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// cbrt
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> cbrt(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_cbrtf4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> cbrt(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_cbrtd2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> cbrt(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_cbrtf8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> cbrt(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_cbrtd4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> cbrt(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_cbrtf16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> cbrt(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_cbrtd8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// sin
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> sin(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_sinf4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> sin(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_sind2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> sin(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_sinf8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> sin(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_sind4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> sin(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_sinf16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> sin(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_sind8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// cos
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> cos(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_cosf4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> cos(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_cosd2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> cos(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_cosf8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> cos(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_cosd4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> cos(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_cosf16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> cos(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_cosd8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// tan
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> tan(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_tanf4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> tan(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_tand2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> tan(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_tanf8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> tan(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_tand4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> tan(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_tanf16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> tan(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_tand8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// asin
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> asin(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_asinf4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> asin(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_asind2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> asin(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_asinf8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> asin(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_asind4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> asin(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_asinf16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> asin(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_asind8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// acos
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> acos(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_acosf4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> acos(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_acosd2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> acos(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_acosf8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> acos(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_acosd4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> acos(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_acosf16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> acos(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_acosd8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// atan
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> atan(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_atanf4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> atan(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_atand2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> atan(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_atanf8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> atan(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_atand4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> atan(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_atanf16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> atan(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_atand8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// atan2 - other variants are automatically taken care off through the generic overloads
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> atan2(const SIMDVector<float,simd_abi::sse> &a, const SIMDVector<float,simd_abi::sse> &b) {
    return Sleef_atan2f4_u10(a.value, b.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> atan2(const SIMDVector<double,simd_abi::sse> &a, const SIMDVector<double,simd_abi::sse> &b) {
    return Sleef_atan2d2_u10(a.value, b.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> atan2(const SIMDVector<float,simd_abi::avx> &a, const SIMDVector<float,simd_abi::avx> &b) {
    return Sleef_atan2f8_u10(a.value, b.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> atan2(const SIMDVector<double,simd_abi::avx> &a, const SIMDVector<double,simd_abi::avx> &b) {
    return Sleef_atan2d4_u10(a.value, b.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> atan2(const SIMDVector<float,simd_abi::avx512> &a, const SIMDVector<float,simd_abi::avx512> &b) {
    return Sleef_atan2f16_u10(a.value, b.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> atan2(const SIMDVector<double,simd_abi::avx512> &a, const SIMDVector<double,simd_abi::avx512> &b) {
    return Sleef_atan2d8_u10(a.value, b.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// sinh
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> sinh(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_sinhf4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> sinh(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_sinhd2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> sinh(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_sinhf8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> sinh(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_sinhd4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> sinh(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_sinhf16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> sinh(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_sinhd8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// cosh
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> cosh(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_coshf4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> cosh(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_coshd2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> cosh(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_coshf8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> cosh(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_coshd4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> cosh(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_coshf16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> cosh(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_coshd8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// tanh
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> tanh(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_tanhf4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> tanh(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_tanhd2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> tanh(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_tanhf8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> tanh(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_tanhd4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> tanh(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_tanhf16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> tanh(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_tanhd8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// asinh
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> asinh(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_asinhf4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> asinh(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_asinhd2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> asinh(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_asinhf8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> asinh(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_asinhd4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> asinh(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_asinhf16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> asinh(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_asinhd8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// acosh
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> acosh(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_acoshf4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> acosh(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_acoshd2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> acosh(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_acoshf8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> acosh(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_acoshd4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> acosh(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_acoshf16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> acosh(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_acoshd8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// atanh
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> atanh(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_atanhf4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> atanh(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_atanhd2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> atanh(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_atanhf8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> atanh(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_atanhd4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> atanh(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_atanhf16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> atanh(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_atanhd8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// erf
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> erf(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_erff4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> erf(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_erfd2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> erf(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_erff8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> erf(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_erfd4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> erf(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_erff16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> erf(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_erfd8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// tgamma
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> tgamma(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_tgammaf4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> tgamma(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_tgammad2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> tgamma(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_tgammaf8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> tgamma(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_tgammad4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> tgamma(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_tgammaf16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> tgamma(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_tgammad8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// lgamma
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> lgamma(const SIMDVector<float,simd_abi::sse> &a) {
    return Sleef_lgammaf4_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> lgamma(const SIMDVector<double,simd_abi::sse> &a) {
    return Sleef_lgammad2_u10(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> lgamma(const SIMDVector<float,simd_abi::avx> &a) {
    return Sleef_lgammaf8_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> lgamma(const SIMDVector<double,simd_abi::avx> &a) {
    return Sleef_lgammad4_u10(a.value);
}
#endif
#ifdef FASTOR_AVX512_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> lgamma(const SIMDVector<float,simd_abi::avx512> &a) {
    return Sleef_lgammaf16_u10(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> lgamma(const SIMDVector<double,simd_abi::avx512> &a) {
    return Sleef_lgammad8_u10(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// hypot - other variants are automatically taken care off through the generic overloads
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> hypot(const SIMDVector<float,simd_abi::sse> &a, const SIMDVector<float,simd_abi::sse> &b) {
    return Sleef_hypotf4_u05(a.value, b.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> hypot(const SIMDVector<double,simd_abi::sse> &a, const SIMDVector<double,simd_abi::sse> &b) {
    return Sleef_hypotd2_u05(a.value, b.value);
}
#endif
#ifdef FASTOR_AVX2_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> hypot(const SIMDVector<float,simd_abi::avx> &a, const SIMDVector<float,simd_abi::avx> &b) {
    return Sleef_hypotf8_u05avx2(a.value, b.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> hypot(const SIMDVector<double,simd_abi::avx> &a, const SIMDVector<double,simd_abi::avx> &b) {
    return Sleef_hypotd4_u05avx2(a.value, b.value);
}
#elif defined(FASTOR_AVX_IMPL)
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> hypot(const SIMDVector<float,simd_abi::avx> &a, const SIMDVector<float,simd_abi::avx> &b) {
    return Sleef_hypotf8_u05avx(a.value, b.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> hypot(const SIMDVector<double,simd_abi::avx> &a, const SIMDVector<double,simd_abi::avx> &b) {
    return Sleef_hypotd4_u05avx(a.value, b.value);
}
#endif
#ifdef FASTOR_AVX512F_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> hypot(const SIMDVector<float,simd_abi::avx512> &a, const SIMDVector<float,simd_abi::avx512> &b) {
    return Sleef_hypotf16_u05avx512f(a.value, b.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> hypot(const SIMDVector<double,simd_abi::avx512> &a, const SIMDVector<double,simd_abi::avx512> &b) {
    return Sleef_hypotd8_u05avx512f(a.value, b.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//

} // end of namespace Fastor

#endif // FASTOR_USE_SLEEF_U10

#endif // SLEEF_BACKEND_U10_H
