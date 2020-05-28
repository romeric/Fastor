#ifndef SIMD_MATH_H
#define SIMD_MATH_H

#include "Fastor/meta/meta.h"
#include "Fastor/simd_vector/extintrin.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include <cmath>

namespace Fastor {

// minimum
//----------------------------------------------------------------------------------------------------------//
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> min(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::min(((T*)&a)[i],((T*)&b)[i]); }
    return out;
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



// vdt math functions
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_USE_VDT
#include <vdt/vdtMath.h>

#ifdef FASTOR_SSE2_IMPL
inline __m128 internal_exp(__m128 a) {
   __m128 out;
   for (FASTOR_INDEX i=0; i<4UL; i++) {
      ((float*)&out)[i] = vdt::fast_expf(((float*)&a)[i]);
   }
   return out;
}
inline __m128 internal_log(__m128 a) {
   __m128 out;
   for (FASTOR_INDEX i=0; i<4UL; i++) {
      ((float*)&out)[i] = vdt::fast_logf(((float*)&a)[i]);
   }
   return out;
}
inline __m128 internal_sin(__m128 a) {
   __m128 out;
   for (FASTOR_INDEX i=0; i<4UL; i++) {
       ((float*)&out)[i] = vdt::fast_sinf(((float*)&a)[i]);
   }
   return out;
}
inline __m128 internal_cos(__m128 a) {
   __m128 out;
   for (FASTOR_INDEX i=0; i<4UL; i++) {
       ((float*)&out)[i] = vdt::fast_cosf(((float*)&a)[i]);
   }
   return out;
}
inline __m128 internal_tan(__m128 a) {
   __m128 out;
   for (FASTOR_INDEX i=0; i<4UL; i++) {
       ((float*)&out)[i] = vdt::fast_tanf(((float*)&a)[i]);
   }
   return out;
}
inline __m128 internal_asin(__m128 a) {
   __m128 out;
   for (FASTOR_INDEX i=0; i<4UL; i++) {
       ((float*)&out)[i] = vdt::fast_asinf(((float*)&a)[i]);
   }
   return out;
}
inline __m128 internal_acos(__m128 a) {
   __m128 out;
   for (FASTOR_INDEX i=0; i<4UL; i++) {
       ((float*)&out)[i] = vdt::fast_acosf(((float*)&a)[i]);
   }
   return out;
}
inline __m128 internal_atan(__m128 a) {
   __m128 out;
   for (FASTOR_INDEX i=0; i<4UL; i++) {
       ((float*)&out)[i] = vdt::fast_atanf(((float*)&a)[i]);
   }
   return out;
}

inline __m128d internal_exp(__m128d a) {
   __m128d out;
   for (FASTOR_INDEX i=0; i<2UL; i++) {
      ((double*)&out)[i] = vdt::fast_exp(((double*)&a)[i]);
   }
   return out;
}
inline __m128d internal_log(__m128d a) {
   __m128d out;
   for (FASTOR_INDEX i=0; i<2UL; i++) {
      ((double*)&out)[i] = vdt::fast_log(((double*)&a)[i]);
   }
   return out;
}
inline __m128d internal_sin(__m128d a) {
   __m128d out;
   for (FASTOR_INDEX i=0; i<2UL; i++) {
       ((double*)&out)[i] = vdt::fast_sin(((double*)&a)[i]);
   }
   return out;
}
inline __m128d internal_cos(__m128d a) {
   __m128d out;
   for (FASTOR_INDEX i=0; i<2UL; i++) {
       ((double*)&out)[i] = vdt::fast_cos(((double*)&a)[i]);
   }
   return out;
}
inline __m128d internal_tan(__m128d a) {
   __m128d out;
   for (FASTOR_INDEX i=0; i<2UL; i++) {
       ((double*)&out)[i] = vdt::fast_tan(((double*)&a)[i]);
   }
   return out;
}
inline __m128d internal_asin(__m128d a) {
   __m128d out;
   for (FASTOR_INDEX i=0; i<2UL; i++) {
       ((double*)&out)[i] = vdt::fast_asin(((double*)&a)[i]);
   }
   return out;
}
inline __m128d internal_acos(__m128d a) {
   __m128d out;
   for (FASTOR_INDEX i=0; i<2UL; i++) {
       ((double*)&out)[i] = vdt::fast_acos(((double*)&a)[i]);
   }
   return out;
}
inline __m128d internal_atan(__m128d a) {
   __m128d out;
   for (FASTOR_INDEX i=0; i<2UL; i++) {
       ((double*)&out)[i] = vdt::fast_atan(((double*)&a)[i]);
   }
   return out;
}
#endif

#ifdef FASTOR_AVX_IMPL
inline __m256 internal_exp(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<8UL; i++) {
       ((float*)&out)[i] = vdt::fast_expf(((float*)&a)[i]);
   }
   return out;
}
inline __m256 internal_log(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<8UL; i++) {
       ((float*)&out)[i] = vdt::fast_logf(((float*)&a)[i]);
   }
   return out;
}
// This can give inaccurate results
//inline __m256 internal_pow(__m256 a, __m256 b) {
//   __m256 out;
//   for (FASTOR_INDEX i=0; i<8UL; i++) {
//       out[i] = vdt::fast_expf(a[i]*vdt::fast_logf(b[i]));
//   }
//   return out;
//}
inline __m256 internal_sin(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<8UL; i++) {
       ((float*)&out)[i] = vdt::fast_sinf(((float*)&a)[i]);
   }
   return out;
}
inline __m256 internal_cos(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<8UL; i++) {
       ((float*)&out)[i] = vdt::fast_cosf(((float*)&a)[i]);
   }
   return out;
}
inline __m256 internal_tan(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<8UL; i++) {
       ((float*)&out)[i] = vdt::fast_tanf(((float*)&a)[i]);
   }
   return out;
}
inline __m256 internal_asin(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<8UL; i++) {
       ((float*)&out)[i] = vdt::fast_asinf(((float*)&a)[i]);
   }
   return out;
}
inline __m256 internal_acos(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<8UL; i++) {
       ((float*)&out)[i] = vdt::fast_acosf(((float*)&a)[i]);
   }
   return out;
}
inline __m256 internal_atan(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<8UL; i++) {
       ((float*)&out)[i] = vdt::fast_atanf(((float*)&a)[i]);
   }
   return out;
}


inline __m256d internal_exp(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<4UL; i++) {
       ((double*)&out)[i] = vdt::fast_exp(((double*)&a)[i]);
   }
   return out;
}
__m256d internal_log(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<4UL; i++) {
       ((double*)&out)[i] = vdt::fast_log(((double*)&a)[i]);
   }
   return out;
}
// This can give inaccurate results
//inline __m256d internal_pow(__m256d a, __m256d b) {
//   __m256d out;
//   for (FASTOR_INDEX i=0; i<4UL; i++) {
//       out[i] = vdt::fast_exp(a[i]*vdt::fast_log(b[i]));
//   }
//   return out;
//}
inline __m256d internal_sin(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<4UL; i++) {
       ((double*)&out)[i] = vdt::fast_sin(((double*)&a)[i]);
   }
   return out;
}
inline __m256d internal_cos(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<4UL; i++) {
       ((double*)&out)[i] = vdt::fast_cos(((double*)&a)[i]);
   }
   return out;
}
inline __m256d internal_tan(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<4UL; i++) {
       ((double*)&out)[i] = vdt::fast_tan(((double*)&a)[i]);
   }
   return out;
}
inline __m256d internal_asin(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<4UL; i++) {
       ((double*)&out)[i] = vdt::fast_asin(((double*)&a)[i]);
   }
   return out;
}
inline __m256d internal_acos(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<4UL; i++) {
       ((double*)&out)[i] = vdt::fast_acos(((double*)&a)[i]);
   }
   return out;
}
inline __m256d internal_atan(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<4UL; i++) {
       ((double*)&out)[i] = vdt::fast_atan(((double*)&a)[i]);
   }
   return out;
}
#endif
//----------------------------------------------------------------------------------------------------------//

#else

//----------------------------------------------------------------------------------------------------------//
// SHUT GCC6 -Wignored-attributes WARNINGS
#ifdef __GNUC__
#if __GNUC__==6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#endif

template<typename T>
inline T internal_exp(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<simd_size_v<remove_all_t<decltype(a[0])>>; ++i) {
       out[i] = std::exp(a[i]);
   }
   return out;
}

template<typename T>
inline T internal_log(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<simd_size_v<remove_all_t<decltype(a[0])>>; ++i) {
       out[i] = std::log(a[i]);
   }
   return out;
}

template<typename T>
inline T internal_sin(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<simd_size_v<remove_all_t<decltype(a[0])>>; ++i) {
       out[i] = std::sin(a[i]);
   }
   return out;
}

template<typename T>
inline T internal_cos(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<simd_size_v<remove_all_t<decltype(a[0])>>; ++i) {
       out[i] = std::cos(a[i]);
   }
   return out;
}

template<typename T>
inline T internal_tan(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<simd_size_v<remove_all_t<decltype(a[0])>>; ++i) {
       out[i] = std::tan(a[i]);
   }
   return out;
}

template<typename T>
inline T internal_asin(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<simd_size_v<remove_all_t<decltype(a[0])>>; ++i) {
       out[i] = std::asin(a[i]);
   }
   return out;
}

template<typename T>
inline T internal_acos(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<simd_size_v<remove_all_t<decltype(a[0])>>; ++i) {
       out[i] = std::acos(a[i]);
   }
   return out;
}

template<typename T>
inline T internal_atan(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<simd_size_v<remove_all_t<decltype(a[0])>>; ++i) {
       out[i] = std::atan(a[i]);
   }
   return out;
}

#ifdef FASTOR_SSE2_IMPL
template<>
inline __m128 internal_exp(__m128 a) {
   __m128 out;

   for (FASTOR_INDEX i=0UL; i<4UL; ++i) {
       ((float*)&out)[i] = std::exp(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m128 internal_log(__m128 a) {
   __m128 out;

   for (FASTOR_INDEX i=0UL; i<4UL; ++i) {
       ((float*)&out)[i] = std::log(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m128 internal_sin(__m128 a) {
   __m128 out;

   for (FASTOR_INDEX i=0UL; i<4UL; ++i) {
       ((float*)&out)[i] = std::sin(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m128 internal_cos(__m128 a) {
   __m128 out;

   for (FASTOR_INDEX i=0UL; i<4UL; ++i) {
       ((float*)&out)[i] = std::cos(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m128 internal_tan(__m128 a) {
   __m128 out;

   for (FASTOR_INDEX i=0UL; i<4UL; ++i) {
       ((float*)&out)[i] = std::tan(((float*)&a)[i]);
   }
   return out;
}
inline __m128 internal_asin(__m128 a) {
   __m128 out;

   for (FASTOR_INDEX i=0UL; i<4UL; ++i) {
       ((float*)&out)[i] = std::asin(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m128 internal_acos(__m128 a) {
   __m128 out;

   for (FASTOR_INDEX i=0UL; i<4UL; ++i) {
       ((float*)&out)[i] = std::acos(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m128 internal_atan(__m128 a) {
   __m128 out;

   for (FASTOR_INDEX i=0UL; i<4UL; ++i) {
       ((float*)&out)[i] = std::atan(((float*)&a)[i]);
   }
   return out;
}


template<>
inline __m128d internal_exp(__m128d a) {
   __m128d out;

   for (FASTOR_INDEX i=0UL; i<2UL; ++i) {
       ((double*)&out)[i] = std::exp(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m128d internal_log(__m128d a) {
   __m128d out;

   for (FASTOR_INDEX i=0UL; i<2UL; ++i) {
       ((double*)&out)[i] = std::log(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m128d internal_sin(__m128d a) {
   __m128d out;

   for (FASTOR_INDEX i=0UL; i<2UL; ++i) {
       ((double*)&out)[i] = std::sin(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m128d internal_cos(__m128d a) {
   __m128d out;

   for (FASTOR_INDEX i=0UL; i<2UL; ++i) {
       ((double*)&out)[i] = std::cos(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m128d internal_tan(__m128d a) {
   __m128d out;

   for (FASTOR_INDEX i=0UL; i<2UL; ++i) {
       ((double*)&out)[i] = std::tan(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m128d internal_asin(__m128d a) {
   __m128d out;

   for (FASTOR_INDEX i=0UL; i<2UL; ++i) {
       ((double*)&out)[i] = std::asin(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m128d internal_acos(__m128d a) {
   __m128d out;

   for (FASTOR_INDEX i=0UL; i<2UL; ++i) {
       ((double*)&out)[i] = std::acos(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m128d internal_atan(__m128d a) {
   __m128d out;

   for (FASTOR_INDEX i=0UL; i<2UL; ++i) {
       ((double*)&out)[i] = std::atan(((double*)&a)[i]);
   }
   return out;
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
inline __m256 internal_exp(__m256 a) {
   __m256 out;

   for (FASTOR_INDEX i=0UL; i<8UL; ++i) {
       ((float*)&out)[i] = std::exp(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m256 internal_log(__m256 a) {
   __m256 out;

   for (FASTOR_INDEX i=0UL; i<8UL; ++i) {
       ((float*)&out)[i] = std::log(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m256 internal_sin(__m256 a) {
   __m256 out;

   for (FASTOR_INDEX i=0UL; i<8UL; ++i) {
       ((float*)&out)[i] = std::sin(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m256 internal_cos(__m256 a) {
   __m256 out;

   for (FASTOR_INDEX i=0UL; i<8UL; ++i) {
       ((float*)&out)[i] = std::cos(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m256 internal_tan(__m256 a) {
   __m256 out;

   for (FASTOR_INDEX i=0UL; i<8UL; ++i) {
       ((float*)&out)[i] = std::tan(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m256 internal_asin(__m256 a) {
   __m256 out;

   for (FASTOR_INDEX i=0UL; i<8UL; ++i) {
       ((float*)&out)[i] = std::asin(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m256 internal_acos(__m256 a) {
   __m256 out;

   for (FASTOR_INDEX i=0UL; i<8UL; ++i) {
       ((float*)&out)[i] = std::acos(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m256 internal_atan(__m256 a) {
   __m256 out;

   for (FASTOR_INDEX i=0UL; i<8UL; ++i) {
       ((float*)&out)[i] = std::atan(((float*)&a)[i]);
   }
   return out;
}


template<>
inline __m256d internal_exp(__m256d a) {
   __m256d out;

   for (FASTOR_INDEX i=0UL; i<4UL; ++i) {
       ((double*)&out)[i] = std::exp(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m256d internal_log(__m256d a) {
   __m256d out;

   for (FASTOR_INDEX i=0UL; i<4UL; ++i) {
       ((double*)&out)[i] = std::log(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m256d internal_sin(__m256d a) {
   __m256d out;

   for (FASTOR_INDEX i=0UL; i<4UL; ++i) {
       ((double*)&out)[i] = std::sin(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m256d internal_cos(__m256d a) {
   __m256d out;

   for (FASTOR_INDEX i=0UL; i<4UL; ++i) {
       ((double*)&out)[i] = std::cos(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m256d internal_tan(__m256d a) {
   __m256d out;

   for (FASTOR_INDEX i=0UL; i<4UL; ++i) {
       ((double*)&out)[i] = std::tan(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m256d internal_asin(__m256d a) {
   __m256d out;

   for (FASTOR_INDEX i=0UL; i<4UL; ++i) {
       ((double*)&out)[i] = std::asin(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m256d internal_acos(__m256d a) {
   __m256d out;

   for (FASTOR_INDEX i=0UL; i<4UL; ++i) {
       ((double*)&out)[i] = std::acos(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m256d internal_atan(__m256d a) {
   __m256d out;

   for (FASTOR_INDEX i=0UL; i<4UL; ++i) {
       ((double*)&out)[i] = std::atan(((double*)&a)[i]);
   }
   return out;
}
#endif
#endif


// not available in vdt
template<typename T, typename U>
inline T internal_pow(T a, U b) {
   T out;
   for (FASTOR_INDEX i=0; i<simd_size_v<remove_all_t<decltype(a[0])>>; ++i) {
       out[i] = std::pow(a[i],b[i]);
   }
   return out;
}

template<typename T>
inline T internal_sinh(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<simd_size_v<remove_all_t<decltype(a[0])>>; ++i) {
       out[i] = std::sinh(a[i]);
   }
   return out;
}

template<typename T>
inline T internal_cosh(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<simd_size_v<remove_all_t<decltype(a[0])>>; ++i) {
       out[i] = std::cosh(a[i]);
   }
   return out;
}

template<typename T>
inline T internal_tanh(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<simd_size_v<remove_all_t<decltype(a[0])>>; ++i) {
       out[i] = std::tanh(a[i]);
   }
   return out;
}
//----------------------------------------------------------------------------------------------------------//




// specialisation for doubles - necessary for SIMDVector<T,32>
//----------------------------------------------------------------------------------------------------------//
template<>
inline float internal_exp(float a) {
  return std::exp(a);
}
template<>
inline float internal_log(float a) {
  return std::log(a);
}
template<>
inline float internal_sin(float a) {
  return std::sin(a);
}
template<>
inline float internal_cos(float a) {
  return std::cos(a);
}
template<>
inline float internal_tan(float a) {
  return std::tan(a);
}
template<>
inline float internal_asin(float a) {
  return std::asin(a);
}
template<>
inline float internal_acos(float a) {
  return std::acos(a);
}
template<>
inline float internal_atan(float a) {
  return std::atan(a);
}
template<>
inline float internal_sinh(float a) {
  return std::sinh(a);
}
template<>
inline float internal_cosh(float a) {
  return std::cosh(a);
}
template<>
inline float internal_tanh(float a) {
  return std::tanh(a);
}
template<>
inline float internal_pow(float a, float b) {
  return std::pow(a,b);
}
template<>
inline float internal_pow(float a, double b) {
  return std::pow(a,b);
}
template<>
inline float internal_pow(float a, int b) {
  return std::pow(a,b);
}



// specialisation for doubles - necessary for SIMDVector<T,64>
template<>
inline double internal_exp(double a) {
  return std::exp(a);
}
template<>
inline double internal_log(double a) {
  return std::log(a);
}
template<>
inline double internal_sin(double a) {
  return std::sin(a);
}
template<>
inline double internal_cos(double a) {
  return std::cos(a);
}
template<>
inline double internal_tan(double a) {
  return std::tan(a);
}
template<>
inline double internal_asin(double a) {
  return std::asin(a);
}
template<>
inline double internal_acos(double a) {
  return std::acos(a);
}
template<>
inline double internal_atan(double a) {
  return std::atan(a);
}
template<>
inline double internal_sinh(double a) {
  return std::sinh(a);
}
template<>
inline double internal_cosh(double a) {
  return std::cosh(a);
}
template<>
inline double internal_tanh(double a) {
  return std::tanh(a);
}
template<>
inline double internal_pow(double a, double b) {
  return std::pow(a,b);
}
template<>
inline double internal_pow(double a, float b) {
  return std::pow(a,b);
}
template<>
inline double internal_pow(double a, int b) {
  return std::pow(a,b);
}
//----------------------------------------------------------------------------------------------------------------------//
//----------------------------------------------------------------------------------------------------------------------//






// SIMDVector overloads with internal math types
//----------------------------------------------------------------------------------------------------------------------//
//----------------------------------------------------------------------------------------------------------------------//
template<typename T, enable_if_t_<!is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> exp(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_exp(a.value);
    return out;
}
template<typename T, enable_if_t_<is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> exp(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::exp(a.value[i]);}
    return out;
}

template<typename T, enable_if_t_<!is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> log(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_log(a.value);
    return out;
}
template<typename T, enable_if_t_<is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> log(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::log(a.value[i]);}
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

template<typename T, typename U>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> pow(const SIMDVector<T,DEFAULT_ABI> &a, const SIMDVector<U,DEFAULT_ABI> &b) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_pow(a.value, b.value);
    return out;
}

template<typename T, typename U, enable_if_t_<!is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> pow(const SIMDVector<T,DEFAULT_ABI> &a, U bb) {
    SIMDVector<T,DEFAULT_ABI> out;
    SIMDVector<T,DEFAULT_ABI> b = static_cast<T>(bb);
    out.value = internal_pow(a.value, b.value);
    return out;
}
template<typename T, typename U, enable_if_t_<is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> pow(const SIMDVector<T,DEFAULT_ABI> &a, U bb) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::pow(a.value[i], bb);}
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> cbrt(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::cbrt(((T*)&a)[i]);}
    return out;
}

template<typename T, enable_if_t_<!is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> sin(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_sin(a.value);
    return out;
}
template<typename T, enable_if_t_<is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> sin(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::sin(a.value[i]);}
    return out;
}

template<typename T, enable_if_t_<!is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> cos(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_cos(a.value);
    return out;
}
template<typename T, enable_if_t_<is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> cos(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::cos(a.value[i]);}
    return out;
}

template<typename T, enable_if_t_<!is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> tan(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_tan(a.value);
    return out;
}
template<typename T, enable_if_t_<is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> tan(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::tan(a.value[i]);}
    return out;
}

template<typename T, enable_if_t_<!is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> asin(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_asin(a.value);
    return out;
}
template<typename T, enable_if_t_<is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> asin(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::asin(a.value[i]);}
    return out;
}

template<typename T, enable_if_t_<!is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> acos(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_acos(a.value);
    return out;
}
template<typename T, enable_if_t_<is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> acos(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::acos(a.value[i]);}
    return out;
}

template<typename T, enable_if_t_<!is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> atan(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_atan(a.value);
    return out;
}
template<typename T, enable_if_t_<is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> atan(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::atan(a.value[i]);}
    return out;
}

template<typename T, enable_if_t_<!is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> sinh(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_sinh(a.value);
    return out;
}
template<typename T, enable_if_t_<is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> sinh(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::sinh(a.value[i]);}
    return out;
}

template<typename T, enable_if_t_<!is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> cosh(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_cosh(a.value);
    return out;
}
template<typename T, enable_if_t_<is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> cosh(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::cosh(a.value[i]);}
    return out;
}

template<typename T, enable_if_t_<!is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> tanh(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    out.value = internal_tanh(a.value);
    return out;
}
template<typename T, enable_if_t_<is_array_v_<typename SIMDVector<T,DEFAULT_ABI>::value_type>,bool> = false>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> tanh(const SIMDVector<T,DEFAULT_ABI> &a) {
    SIMDVector<T,DEFAULT_ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,DEFAULT_ABI>::Size; i++) { out.value[i] = std::tanh(a.value[i]);}
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
FASTOR_INLINE SIMDVector<T,ABI> trunc(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; i++) { ((T*)&out)[i] = std::trunc(((T*)&a)[i]);}
    return out;
}
//----------------------------------------------------------------------------------------------------------------------//
//----------------------------------------------------------------------------------------------------------------------//


} // end of namespace Fastor

#endif // SIMD_MATH_H

