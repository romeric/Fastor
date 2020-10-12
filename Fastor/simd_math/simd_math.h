#ifndef SIMD_MATH_H
#define SIMD_MATH_H

#include "Fastor/meta/meta.h"
#include "Fastor/simd_vector/extintrin.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include <cmath>

// SHUT GCC6 -Wignored-attributes WARNINGS
#ifdef __GNUC__
#if __GNUC__==6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#endif

namespace Fastor {

// minimum
//----------------------------------------------------------------------------------------------------------//
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> min(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::min(((T*)&a)[i],((T*)&b)[i]); }
    return out;
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> min(const SIMDVector<T,ABI> &a, T b) {
    return min(a,SIMDVector<T,ABI>(b));
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> min(T a, const SIMDVector<T,ABI> &b) {
    return min(SIMDVector<T,ABI>(a),b);
}
#ifdef FASTOR_SSE2_IMPL
#ifdef FASTOR_SSE4_1_IMPL
template<>
FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> min(const SIMDVector<int32_t,simd_abi::sse> &a, const SIMDVector<int32_t,simd_abi::sse> &b) {
    return _mm_min_epi32(a.value,b.value);
}
#endif
#if defined(FASTOR_AVX512F_IMPL) && defined(FASTOR_AVX512VL_IMPL)
template<>
FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> min(const SIMDVector<int64_t,simd_abi::sse> &a, const SIMDVector<int64_t,simd_abi::sse> &b) {
    return _mm_min_epi64(a.value,b.value);
}
#endif
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> min(const SIMDVector<float,simd_abi::sse> &a, const SIMDVector<float,simd_abi::sse> &b) {
    return _mm_min_ps(a.value,b.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> min(const SIMDVector<double,simd_abi::sse> &a, const SIMDVector<double,simd_abi::sse> &b) {
    return _mm_min_pd(a.value,b.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
#ifdef FASTOR_AVX2_IMPL
template<>
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> min(const SIMDVector<int32_t,simd_abi::avx> &a, const SIMDVector<int32_t,simd_abi::avx> &b) {
    return _mm256_min_epi32(a.value,b.value);
}
#endif
#if defined(FASTOR_AVX512F_IMPL) && defined(FASTOR_AVX512VL_IMPL)
template<>
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx> min(const SIMDVector<int64_t,simd_abi::avx> &a, const SIMDVector<int64_t,simd_abi::avx> &b) {
    return _mm256_min_epi64(a.value,b.value);
}
#endif
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> min(const SIMDVector<float,simd_abi::avx> &a, const SIMDVector<float,simd_abi::avx> &b) {
    return _mm256_min_ps(a.value,b.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> min(const SIMDVector<double,simd_abi::avx> &a, const SIMDVector<double,simd_abi::avx> &b) {
    return _mm256_min_pd(a.value,b.value);
}
#endif
#ifdef FASTOR_AVX512F_IMPL
template<>
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> min(const SIMDVector<int32_t,simd_abi::avx512> &a, const SIMDVector<int32_t,simd_abi::avx512> &b) {
    return _mm512_min_epi32(a.value,b.value);
}
template<>
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> min(const SIMDVector<int64_t,simd_abi::avx512> &a, const SIMDVector<int64_t,simd_abi::avx512> &b) {
    return _mm512_min_epi64(a.value,b.value);
}
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> min(const SIMDVector<float,simd_abi::avx512> &a, const SIMDVector<float,simd_abi::avx512> &b) {
    return _mm512_min_ps(a.value,b.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> min(const SIMDVector<double,simd_abi::avx512> &a, const SIMDVector<double,simd_abi::avx512> &b) {
    return _mm512_min_pd(a.value,b.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// maximum
//----------------------------------------------------------------------------------------------------------//
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> max(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::max(((T*)&a)[i],((T*)&b)[i]); }
    return out;
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> max(const SIMDVector<T,ABI> &a, T b) {
    return max(a,SIMDVector<T,ABI>(b));
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> max(T a, const SIMDVector<T,ABI> &b) {
    return max(SIMDVector<T,ABI>(a),b);
}
#ifdef FASTOR_SSE2_IMPL
#ifdef FASTOR_SSE4_1_IMPL
template<>
FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> max(const SIMDVector<int32_t,simd_abi::sse> &a, const SIMDVector<int32_t,simd_abi::sse> &b) {
    return _mm_max_epi32(a.value,b.value);
}
#endif
#if defined(FASTOR_AVX512F_IMPL) && defined(FASTOR_AVX512VL_IMPL)
template<>
FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> max(const SIMDVector<int64_t,simd_abi::sse> &a, const SIMDVector<int64_t,simd_abi::sse> &b) {
    return _mm_max_epi64(a.value,b.value);
}
#endif
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> max(const SIMDVector<float,simd_abi::sse> &a, const SIMDVector<float,simd_abi::sse> &b) {
    return _mm_max_ps(a.value,b.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> max(const SIMDVector<double,simd_abi::sse> &a, const SIMDVector<double,simd_abi::sse> &b) {
    return _mm_max_pd(a.value,b.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
#ifdef FASTOR_AVX2_IMPL
template<>
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> max(const SIMDVector<int32_t,simd_abi::avx> &a, const SIMDVector<int32_t,simd_abi::avx> &b) {
    return _mm256_max_epi32(a.value,b.value);
}
#endif
#if defined(FASTOR_AVX512F_IMPL) && defined(FASTOR_AVX512VL_IMPL)
template<>
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx> max(const SIMDVector<int64_t,simd_abi::avx> &a, const SIMDVector<int64_t,simd_abi::avx> &b) {
    return _mm256_max_epi64(a.value,b.value);
}
#endif
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> max(const SIMDVector<float,simd_abi::avx> &a, const SIMDVector<float,simd_abi::avx> &b) {
    return _mm256_max_ps(a.value,b.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> max(const SIMDVector<double,simd_abi::avx> &a, const SIMDVector<double,simd_abi::avx> &b) {
    return _mm256_max_pd(a.value,b.value);
}
#endif
#ifdef FASTOR_AVX512F_IMPL
template<>
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> max(const SIMDVector<int32_t,simd_abi::avx512> &a, const SIMDVector<int32_t,simd_abi::avx512> &b) {
    return _mm512_max_epi32(a.value,b.value);
}
template<>
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> max(const SIMDVector<int64_t,simd_abi::avx512> &a, const SIMDVector<int64_t,simd_abi::avx512> &b) {
    return _mm512_max_epi64(a.value,b.value);
}
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx512> max(const SIMDVector<float,simd_abi::avx512> &a, const SIMDVector<float,simd_abi::avx512> &b) {
    return _mm512_max_ps(a.value,b.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> max(const SIMDVector<double,simd_abi::avx512> &a, const SIMDVector<double,simd_abi::avx512> &b) {
    return _mm512_max_pd(a.value,b.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//


// ceil
//----------------------------------------------------------------------------------------------------------//
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> ceil(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::ceil(((T*)&a)[i]);}
    return out;
}
#ifdef FASTOR_SSE4_1_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> ceil(const SIMDVector<float,simd_abi::sse> &a) {
    return _mm_ceil_ps(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> ceil(const SIMDVector<double,simd_abi::sse> &a) {
    return _mm_ceil_pd(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> ceil(const SIMDVector<float,simd_abi::avx> &a) {
    return _mm256_ceil_ps(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> ceil(const SIMDVector<double,simd_abi::avx> &a) {
    return _mm256_ceil_pd(a.value);
}
#endif
// Part of SVML
// #ifdef FASTOR_AVX512F_IMPL
// template<>
// FASTOR_INLINE SIMDVector<float,simd_abi::avx512> ceil(const SIMDVector<float,simd_abi::avx512> &a) {
//     return _mm512_ceil_ps(a.value);
// }
// template<>
// FASTOR_INLINE SIMDVector<double,simd_abi::avx512> ceil(const SIMDVector<double,simd_abi::avx512> &a) {
//     return _mm512_ceil_pd(a.value);
// }
// #endif
//----------------------------------------------------------------------------------------------------------//


// round
//----------------------------------------------------------------------------------------------------------//
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> round(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::round(((T*)&a)[i]);}
    return out;
}
#ifdef FASTOR_SSE4_1_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> round(const SIMDVector<float,simd_abi::sse> &a) {
    return _mm_round_ps(a.value, ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> round(const SIMDVector<double,simd_abi::sse> &a) {
    return _mm_round_pd(a.value, ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> round(const SIMDVector<float,simd_abi::avx> &a) {
    return _mm256_round_ps(a.value, ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> round(const SIMDVector<double,simd_abi::avx> &a) {
    return _mm256_round_pd(a.value, ( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
}
#endif
//----------------------------------------------------------------------------------------------------------//


// floor
//----------------------------------------------------------------------------------------------------------//
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> floor(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::floor(((T*)&a)[i]);}
    return out;
}
#ifdef FASTOR_SSE4_1_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::sse> floor(const SIMDVector<float,simd_abi::sse> &a) {
    return _mm_floor_ps(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::sse> floor(const SIMDVector<double,simd_abi::sse> &a) {
    return _mm_floor_pd(a.value);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE SIMDVector<float,simd_abi::avx> floor(const SIMDVector<float,simd_abi::avx> &a) {
    return _mm256_floor_ps(a.value);
}
template<>
FASTOR_INLINE SIMDVector<double,simd_abi::avx> floor(const SIMDVector<double,simd_abi::avx> &a) {
    return _mm256_floor_pd(a.value);
}
#endif
//----------------------------------------------------------------------------------------------------------//



// remaining math functions from STL
//----------------------------------------------------------------------------------------------------------------------//
//----------------------------------------------------------------------------------------------------------------------//

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> exp(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::exp(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> exp2(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::exp2(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> expm1(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::expm1(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> log(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::log(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> log10(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::log10(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> log2(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::log2(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> log1p(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::log1p(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> pow(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i = 0; i < SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::pow(((T*)&a)[i], ((T*)&b)[i]);}
    return out;
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> pow(const SIMDVector<T,ABI> &a, T b) {
    return pow(a, SIMDVector<T,ABI>(b));
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> pow(T a, const SIMDVector<T,ABI> &b) {
    return pow(SIMDVector<T,ABI>(a),b);
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> cbrt(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::cbrt(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> sin(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::sin(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> cos(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::cos(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> tan(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::tan(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> asin(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::asin(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> acos(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::acos(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> atan(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::atan(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> atan2(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i = 0; i < SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::atan2(((T*)&a)[i], ((T*)&b)[i]);}
    return out;
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> atan2(const SIMDVector<T,ABI> &a, T b) {
    return atan2(a, SIMDVector<T,ABI>(b));
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> atan2(T a, const SIMDVector<T,ABI> &b) {
    return atan2(SIMDVector<T,ABI>(a),b);
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> sinh(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::sinh(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> cosh(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::cosh(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> tanh(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::tanh(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> asinh(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::asinh(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> acosh(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::acosh(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> atanh(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::atanh(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> erf(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::erf(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> tgamma(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::tgamma(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> lgamma(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::lgamma(((T*)&a)[i]);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> hypot(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i = 0; i < SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::hypot(((T*)&a)[i], ((T*)&b)[i]);}
    return out;
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> hypot(const SIMDVector<T,ABI> &a, T b) {
    return hypot(a, SIMDVector<T,ABI>(b));
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> hypot(T a, const SIMDVector<T,ABI> &b) {
    return hypot(SIMDVector<T,ABI>(a),b);
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> trunc(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::trunc(((T*)&a)[i]);}
    return out;
}
//----------------------------------------------------------------------------------------------------------------------//
//----------------------------------------------------------------------------------------------------------------------//



// Boolean arithmetic
// ! or not
//----------------------------------------------------------------------------------------------------------//
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> operator!(const SIMDVector<T,ABI> &a) {
    SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((bool*)&out)[i] = !(((T*)&a)[i]); }
    return out;
}
//----------------------------------------------------------------------------------------------------------//


// isinf/nan/finite
//----------------------------------------------------------------------------------------------------------//
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> isinf(const SIMDVector<T,ABI> &a) {
    SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((bool*)&out)[i] = std::isinf(((T*)&a)[i]); }
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> isnan(const SIMDVector<T,ABI> &a) {
    SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((bool*)&out)[i] = std::isnan(((T*)&a)[i]); }
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> isfinite(const SIMDVector<T,ABI> &a) {
    SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((bool*)&out)[i] = std::isfinite(((T*)&a)[i]); }
    return out;
}
//----------------------------------------------------------------------------------------------------------//


} // end of namespace Fastor


// Include all backends
#include "Fastor/simd_math/sleef_backend.h"

#endif // SIMD_MATH_H
