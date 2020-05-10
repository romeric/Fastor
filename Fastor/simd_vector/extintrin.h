#ifndef EXT_INTRIN_H
#define EXT_INTRIN_H

#include "Fastor/commons/commons.h"
#include "Fastor/meta/meta.h"
#include <cmath>


namespace Fastor {

// Macros for immediate construction
//----------------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
#define ZEROPS  (_mm_set1_ps(0.f))
#define ZEROPD  (_mm_set1_pd(0.0))
#define MZEROPS (_mm_set1_ps(-0.f))
#define MZEROPD (_mm_set1_pd(-0.0))
#define ONEPS   (_mm_set1_ps(1.f))
#define ONEPD   (_mm_set1_pd(1.0))
#define HALFPS  (_mm_set1_ps(0.5f))
#define HALFPD  (_mm_set1_pd(0.5))
#define TWOPS   (_mm_set1_ps(2.0f))
#define TOWPD   (_mm_set1_pd(2.0))
#endif
#ifdef FASTOR_AVX_IMPL
#define VZEROPS  (_mm256_set1_ps(0.f))
#define VZEROPD  (_mm256_set1_pd(0.0))
#define MVZEROPS (_mm256_set1_ps(-0.f))
#define MVZEROPD (_mm256_set1_pd(-0.0))
#define VONEPS   (_mm256_set1_ps(1.f))
#define VONEPD   (_mm256_set1_pd(1.0))
#define VHALFPS  (_mm256_set1_ps(0.5f))
#define VHALFPD  (_mm256_set1_pd(0.5))
#define VTWOPS   (_mm256_set1_ps(2.0f))
#define VTOWPD   (_mm256_set1_pd(2.0))
#endif
//----------------------------------------------------------------------------------------------------------------//


// Mask load the 3 lower parts
//----------------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
FASTOR_INLINE __m128 _mm_loadl3_ps(const float *arr) {
#ifdef FASTOR_HAS_AVX512_MASKS
    return _mm_mask_load_ps(ZEROPS, (__mmask8)0x07, arr);
#elif defined(FASTOR_AVX_IMPL)
    __m128i mask = _mm_set_epi32(0,-1,-1,-1);
    return _mm_maskload_ps(arr,(__m128i) mask);
#else
    __m128i xy = _mm_loadl_epi64((const __m128i*)arr);
    __m128 z   = _mm_load_ss(&arr[2]);
    return _mm_movelh_ps(_mm_castsi128_ps(xy), z);
#endif
}

FASTOR_INLINE __m128 _mm_loadul3_ps(const float *arr) {
#ifdef FASTOR_HAS_AVX512_MASKS
    return _mm_mask_loadu_ps(ZEROPS, (__mmask8)0x07, arr);
#elif defined(FASTOR_AVX_IMPL)
    // AVX maskloads apparently have no alignment requirement
    __m128i mask = _mm_set_epi32(0,-1,-1,-1);
    return _mm_maskload_ps(arr,(__m128i) mask);
#else
    __m128 x = _mm_load_ss( arr  );
    __m128 y = _mm_load_ss(&arr[1]);
    __m128 z = _mm_load_ss(&arr[2]);
    __m128 xy = _mm_movelh_ps(x, y);
    return _mm_shuffle_ps(xy, z, _MM_SHUFFLE(2, 0, 2, 0));
#endif
}
#endif

#ifdef FASTOR_AVX_IMPL
FASTOR_INLINE __m256d _mm256_loadl3_pd(const double *arr) {
#ifdef FASTOR_HAS_AVX512_MASKS
    return _mm256_mask_load_pd(VZEROPD, (__mmask8)0x07, arr);
#else
    __m256i mask = _mm256_set_epi64x(0,-1,-1,-1);
    return _mm256_maskload_pd(arr,(__m256i) mask);
#endif
}

FASTOR_INLINE __m256d _mm256_loadul3_pd(const double *arr) {
#ifdef FASTOR_HAS_AVX512_MASKS
    return _mm256_mask_loadu_pd(VZEROPD, (__mmask8)0x07, arr);
#else
    // AVX maskloads apparently have no alignment requirement
    __m256i mask = _mm256_set_epi64x(0,-1,-1,-1);
    return _mm256_maskload_pd(arr,(__m256i) mask);
    // __m128d xy   = _mm_loadu_pd(arr);
    // __m128d z    = _mm_load_sd(&arr[2]);
    // __m256d vec  = _mm256_castpd128_pd256(xy);
    // return _mm256_insertf128_pd(vec, z,0x1);
#endif
}
#endif

// Mask store the 3 lower parts
#ifdef FASTOR_SSE2_IMPL
FASTOR_INLINE void _mm_storel3_ps(float *arr, __m128 value) {
#ifdef FASTOR_HAS_AVX512_MASKS
    _mm_mask_store_ps(arr, (__mmask8)0x07, value);
#elif defined(FASTOR_AVX_IMPL)
    __m128i mask = _mm_set_epi32(0,-1,-1,-1);
    _mm_maskstore_ps(arr, (__m128i)mask, value);
#else
    _mm_storel_pi((__m64*)arr, value);
    _mm_store_ss(&arr[2],_mm_shuffle_ps(value,value,0x2));
#endif
}

FASTOR_INLINE void _mm_storeul3_ps(float *arr, __m128 value) {
#ifdef FASTOR_HAS_AVX512_MASKS
    _mm_mask_storeu_ps(arr, (__mmask8)0x07, value);
#elif defined(FASTOR_AVX_IMPL)
    __m128i mask = _mm_set_epi32(0,-1,-1,-1);
    _mm_maskstore_ps(arr, (__m128i)mask, value);
#else
    _mm_storel_pi((__m64*)arr, value);
    _mm_store_ss(&arr[2],_mm_shuffle_ps(value,value,0x2));
#endif
}
#endif

#ifdef FASTOR_AVX_IMPL
FASTOR_INLINE void _mm256_storel3_pd(double *arr, __m256d value) {
#ifdef FASTOR_HAS_AVX512_MASKS
    _mm256_mask_store_pd(arr, (__mmask8)0x07, value);
#else
    __m256i mask = _mm256_set_epi64x(0,-1,-1,-1);
    _mm256_maskstore_pd(arr, (__m256i)mask, value);
#endif
}

FASTOR_INLINE void _mm256_storeul3_pd(double *arr, __m256d value) {
#ifdef FASTOR_HAS_AVX512_MASKS
    _mm256_mask_storeu_pd(arr, (__mmask8)0x07, value);
#else
    // AVX maskloads apparently have no alignment requirement
    __m256i mask = _mm256_set_epi64x(0,-1,-1,-1);
    _mm256_maskstore_pd(arr, (__m256i)mask, value);
    // _mm_storeu_pd(arr  , _mm256_castpd256_pd128(value)    );
    // _mm_store_sd (arr+2, _mm256_extractf128_pd (value,0x1));
#endif
}
#endif
//----------------------------------------------------------------------------------------------------------------//



//! Horizontal summation/multiplication of registers
//----------------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
FASTOR_INLINE int _mm_sum_epi32(__m128i a) {
    // W/O HADD: IVY 5 - HW 5 - SKY 6
    // __m128i c = _mm_hadd_epi32(a,a); // SSSE3 one extra op
    __m128i c = _mm_add_epi32(a,_mm_shuffle_epi32(a,_MM_SHUFFLE(2,3,0,1)));
    __m128i d = _mm_add_epi32(c,_mm_shuffle_epi32(c,_MM_SHUFFLE(0,1,2,3)));
    return _mm_cvtsi128_si32(d);
}

static int _mm_prod_epi32(__m128i a) {
    // IVY 13 - HW 13 - SKY 12
    __m128i c = _mm_mul_epu32(a,_mm_shuffle_epi32(a,_MM_SHUFFLE(2,3,0,1)));
    __m128i d = _mm_mul_epu32(c,_mm_shuffle_epi32(c,_MM_SHUFFLE(2,2,2,2)));
    return _mm_cvtsi128_si32(d);
}
#endif

#ifdef FASTOR_USE_HADD
#ifdef FASTOR_SSSE3_IMPL
FASTOR_INLINE float _mm_sum_ps(__m128 a) {
    // 10 OPS
    float sum32;
    __m128 sum = _mm_hadd_ps(a, a);
    _mm_store_ss(&sum32,_mm_hadd_ps(sum, sum));
    return sum32;
}
FASTOR_INLINE double _mm_sum_pd(__m128d a) {
    // 5 OPS
    double sum64;
    _mm_store_sd(&sum64,_mm_hadd_pd(a, a));
    return sum64;
}
#endif
#else
#ifdef FASTOR_SSE2_IMPL
FASTOR_INLINE float _mm_sum_ps(__m128 a) {
    // 8 OPS
#ifdef FASTOR_SSE3_IMPL
    __m128 shuf = _mm_movehdup_ps(a);
#else
    __m128 shuf = _mm_shuffle_ps(a,a, _MM_SHUFFLE(3,3,1,1));
#endif
    __m128 sums = _mm_add_ps(a, shuf);
    shuf        = _mm_movehl_ps(shuf, sums);
    sums        = _mm_add_ss(sums, shuf);
    return        _mm_cvtss_f32(sums);
}
FASTOR_INLINE double _mm_sum_pd(__m128d a) {
    // 4 OPS
    __m128 shuftmp= _mm_movehl_ps(ZEROPS, _mm_castpd_ps(a));
    __m128d shuf  = _mm_castps_pd(shuftmp);
    return  _mm_cvtsd_f64(_mm_add_sd(a, shuf));
}

FASTOR_INLINE float _mm_prod_ps(__m128 a) {
    // 12 OPS
#ifdef FASTOR_SSE3_IMPL
    __m128 shuf = _mm_movehdup_ps(a);
#else
    __m128 shuf = _mm_shuffle_ps(a,a, _MM_SHUFFLE(3,3,1,1));
#endif
    __m128 prods = _mm_mul_ps(a, shuf);
    shuf         = _mm_movehl_ps(shuf, prods);
    prods        = _mm_mul_ss(prods, shuf);
    return       _mm_cvtss_f32(prods);
}
FASTOR_INLINE double _mm_prod_pd(__m128d a) {
    // 6 OPS
    __m128 shuftmp= _mm_movehl_ps(ZEROPS, _mm_castpd_ps(a));
    __m128d shuf  = _mm_castps_pd(shuftmp);
    return  _mm_cvtsd_f64(_mm_mul_sd(a, shuf));
}
#endif
#endif

#ifdef FASTOR_AVX_IMPL

FASTOR_INLINE float _mm256_sum_ps(__m256 a) {
#ifdef FASTOR_USE_HADD
    // IVY 14 OPS - HW 16 OPS
    __m256 sum    = _mm256_hadd_ps(a, a);
    sum           = _mm256_hadd_ps(sum, sum);
    __m128 result = _mm_add_ps(_mm256_castps256_ps128(sum),_mm256_extractf128_ps(sum, 0x1));
    return _mm_cvtss_f32(result);
#else
   // IVY 14 OPS
   return _mm_sum_ps(_mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,0x1)));
#endif
}
FASTOR_INLINE double _mm256_sum_pd(__m256d a) {
#ifdef FASTOR_USE_HADD
    // IVY 9 OPS - HW - 11 OPS
    __m256d sum = _mm256_hadd_pd(a, a);
#else
    // IVY 8 OPS - HW 10 OPS - SKY 11 OPS (BUT 2 PARALLEL ADDS SO POTENTIALLY 7OPS)
    __m256d sum = _mm256_add_pd(a, _mm256_shuffle_pd(a,a,0x5));
#endif
    __m128d result = _mm_add_sd(_mm256_castpd256_pd128(sum),_mm256_extractf128_pd(sum, 0x1));
    return _mm_cvtsd_f64(result);
}


FASTOR_INLINE float _mm256_prod_ps(__m256 a) {
    // ~ IVY 30 OPS - HW 32 OPS
    return _mm_prod_ps(_mm256_castps256_ps128(a))*_mm_prod_ps(_mm256_extractf128_ps(a, 0x1));
}
FASTOR_INLINE double _mm256_prod_pd(__m256d a) {
    // IVY 12 OPS - HW - 14 OPS
    __m256d sum = _mm256_mul_pd(a, _mm256_shuffle_pd(a,a,0x5));
    __m128d sum_high = _mm256_extractf128_pd(sum, 0x1);
    __m128d result = _mm_mul_sd(sum_high, _mm256_castpd256_pd128(sum));
    return _mm_cvtsd_f64(result);
}

#endif
//----------------------------------------------------------------------------------------------------------------//



//! Reversing a register
//----------------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
FASTOR_INLINE __m128 _mm_reverse_ps(__m128 a) {
    // 1OP
    return _mm_shuffle_ps(a,a,_MM_SHUFFLE(0,1,2,3));
}
FASTOR_INLINE __m128d _mm_reverse_pd(__m128d a) {
    // 1OP
    return _mm_shuffle_pd(a,a,0x1);
}
FASTOR_INLINE __m128i _mm_reverse_epi32(__m128i v) {
    // 1 OP
    return _mm_shuffle_epi32(v, 0x1b);
}
FASTOR_INLINE __m128i _mm_reverse_epi64(__m128i v) {
    // 1 OP
    return _mm_castpd_si128(_mm_reverse_pd(_mm_castsi128_pd(v)));
}
#endif
#ifdef FASTOR_AVX_IMPL
FASTOR_INLINE __m256 _mm256_reverse_ps(__m256 a) {
    // IVY 2OPS / HW 4OPS
    __m256 r1 = _mm256_permute2f128_ps(a,a,0x1);
    return _mm256_shuffle_ps(r1,r1,27);
}
FASTOR_INLINE __m256d _mm256_reverse_pd(__m256d a) {
    // IVY 2OPS / HW 4OPS
    __m256d r1 = _mm256_permute2f128_pd(a,a,0x1);
    return _mm256_shuffle_pd(r1,r1,5);
}
FASTOR_INLINE __m256i _mm256_reverse_epi32(__m256i v) {
    // IVY 2OPS / HW 4OPS
    return _mm256_castps_si256(_mm256_reverse_ps(_mm256_castsi256_ps(v)));
    /*
    // 8 OPS
    __m128i lo = _mm_shuffle_epi32(_mm256_castsi256_si128(_a));
    __m128i hi = _mm_shuffle_epi32(_mm256_extractf128_si256(_a,1));
    __m256i out = _mm256_castsi128_si256(lo);
    out = _mm256_insertf128_si256(out,hi,1);
    return out;
    */
}
FASTOR_INLINE __m256i _mm256_reverse_epi64(__m256i v) {
    // IVY 2OPS / HW 4OPS
    return _mm256_castpd_si256(_mm256_reverse_pd(_mm256_castsi256_pd(v)));
}
#endif
//----------------------------------------------------------------------------------------------------------------//



//! Bit shifting - for extracting values and so on
//----------------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
FASTOR_INLINE __m128 _mm_shift1_ps(__m128 a) {
    // 1OP
    return _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(a), 4));
}
FASTOR_INLINE __m128 _mm_shift2_ps(__m128 a) {
    // 1OP
    return _mm_shuffle_ps(ZEROPS, a, 0x40);
}
FASTOR_INLINE __m128 _mm_shift3_ps(__m128 a) {
    // 2OPS
    __m128 shift2 = _mm_shuffle_ps(ZEROPS, a, 0x40);
    return _mm_shuffle_ps(ZEROPS,shift2,_MM_SHUFFLE(2,0,2,0));
}
#endif
#ifdef FASTOR_AVX_IMPL
FASTOR_INLINE __m256 _mm256_shift1_ps(__m256 a) {
    // IVY 3OPS / HW 5OPS
    __m256 r1 = _mm256_permute_ps(a,_MM_SHUFFLE(2,1,0,3));
    __m256 r2 = _mm256_permute2f128_ps(r1,r1,41);
    return _mm256_blend_ps(r1,r2,0x11);
}
FASTOR_INLINE __m256 _mm256_shift2_ps(__m256 a) {
    // IVY 3OPS / HW 5OPS
    __m256 r1 = _mm256_permute_ps(a,_MM_SHUFFLE(1,0,3,2));
    __m256 r2 = _mm256_permute2f128_ps(r1,r1,41);
    return _mm256_blend_ps(r1,r2,0x33);
}
FASTOR_INLINE __m256 _mm256_shift3_ps(__m256 a) {
    // IVY 1OPS / HW 3OPS
    return _mm256_permute2f128_ps(a,a,41);
}
FASTOR_INLINE __m256 _mm256_shift4_ps(__m256 a) {
    // IVY 1OPS / HW 3OPS
    return _mm256_permute2f128_ps(a,a,42);
}
FASTOR_INLINE __m256 _mm256_shift5_ps(__m256 a) {
    // IVY 2 OPS - HW 4 OPS
    __m128 r1 = _mm_shift1_ps(_mm256_castps256_ps128(a));
    return _mm256_insertf128_ps(VZEROPS,r1,0x1);
}
FASTOR_INLINE __m256 _mm256_shift6_ps(__m256 a) {
    // IVY 2 OPS - HW 4 OPS
    __m128 r1 = _mm_shift2_ps(_mm256_castps256_ps128(a));
    return _mm256_insertf128_ps(VZEROPS,r1,0x1);
}
FASTOR_INLINE __m256 _mm256_shift7_ps(__m256 a) {
    // IVY 2 OPS - HW 4 OPS
    __m128 r1 = _mm_shift3_ps(_mm256_castps256_ps128(a));
    return _mm256_insertf128_ps(VZEROPS,r1,0x1);
}
#endif
#ifdef FASTOR_SSE2_IMPL
FASTOR_INLINE __m128d _mm_shift1_pd(__m128d a) {
    // 1OP
    return _mm_shuffle_pd(ZEROPD,a,0x1);
}
#endif
#ifdef FASTOR_AVX_IMPL
FASTOR_INLINE __m256d _mm256_shift1_pd(__m256d a) {
    // IVY - 4 OPS / HW - 8 OPS
    __m128d r1 = _mm256_castpd256_pd128(a);
    __m128d r2 = _mm256_extractf128_pd(a,0x1);
    __m128d r3 = _mm_shuffle_pd(r1,r2,0x1);
    __m256d r4 = _mm256_castpd128_pd256(_mm_shift1_pd(r1));
    return _mm256_insertf128_pd(r4,r3,0x1);
}
FASTOR_INLINE __m256d _mm256_shift2_pd(__m256d a) {
    // IVY - 1OP / HW - 3OPS
    return _mm256_permute2f128_pd(a,a,8);
}
FASTOR_INLINE __m256d _mm256_shift3_pd(__m256d a) {
    // IVY - 2OPS / HW - 4OPS
    __m256d r1 = _mm256_castpd128_pd256(_mm_shift1_pd(_mm256_castpd256_pd128(a)));
    return _mm256_permute2f128_pd(r1,r1,0x1);
}
#endif
//----------------------------------------------------------------------------------------------------------------//


// Negation
//----------------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
// Change sign of a register - all one cycle
FASTOR_INLINE __m128 _mm_neg_ps(__m128 a) {
    return _mm_xor_ps(a, MZEROPS);
}
FASTOR_INLINE __m128d _mm_neg_pd(__m128d a) {
    return _mm_xor_pd(a, MZEROPD);
}
#endif
#ifdef FASTOR_AVX_IMPL
FASTOR_INLINE __m256 _mm256_neg_ps(__m256 a) {
    return _mm256_xor_ps(a, MVZEROPS);
}
FASTOR_INLINE __m256d _mm256_neg_pd(__m256d a) {
    return _mm256_xor_pd(a, MVZEROPD);
}
#endif
//----------------------------------------------------------------------------------------------------------------//


// Absolute values
//----------------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
// Absolute value of a register - all one cycle
FASTOR_INLINE __m128 _mm_abs_ps(__m128 x) {
    static const __m128 sign_mask = _mm_set1_ps(-0.f); // -0.f = 1 << 31
    return _mm_andnot_ps(sign_mask, x);
}
FASTOR_INLINE __m128d _mm_abs_pd(__m128d x) {
    static const __m128d sign_mask = _mm_set1_pd(-0.); // -0. = 1 << 63
    return _mm_andnot_pd(sign_mask, x); // !sign_mask & x
}
#endif
#ifdef FASTOR_AVX_IMPL
FASTOR_INLINE __m256 _mm256_abs_ps(__m256 x) {
    static const __m256 sign_mask = _mm256_set1_ps(-0.f); // -0.f = 1 << 31
    return _mm256_andnot_ps(sign_mask, x);
}
FASTOR_INLINE __m256d _mm256_abs_pd(__m256d x) {
    static const __m256d sign_mask = _mm256_set1_pd(-0.); // -0. = 1 << 63
    return _mm256_andnot_pd(sign_mask, x); // !sign_mask & x
}
#endif
//----------------------------------------------------------------------------------------------------------------//


// Horizontal max
//----------------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
// maximum value in a register - horizontal max
FASTOR_INLINE float _mm_hmax_ps(__m128 a) {
    // 8OPS
    __m128 max0 = _mm_max_ps(a,_mm_reverse_ps(a));
    __m128 tmp = _mm_shuffle_ps(max0,max0,_MM_SHUFFLE(0,0,0,1));
    return _mm_cvtss_f32(_mm_max_ps(max0,tmp));
}
FASTOR_INLINE double _mm_hmax_pd(__m128d a) {
    // 4OPS
    return _mm_cvtsd_f64(_mm_max_pd(a,_mm_reverse_pd(a)));
}
#endif
#ifdef FASTOR_AVX_IMPL
FASTOR_INLINE float _mm256_hmax_ps(__m256 a) {
    // IVY 18OPS / HW 24 OPS
    __m128 lo = _mm256_castps256_ps128(a);
    __m128 max0 = _mm_max_ps(lo,_mm_reverse_ps(lo));
    __m128 tmp0 = _mm_shuffle_ps(max0,max0,_MM_SHUFFLE(0,0,0,1));
    __m128 max_lo = _mm_max_ps(max0,tmp0);

    __m128 hi = _mm256_extractf128_ps(a,0x1);
    __m128 max1 = _mm_max_ps(hi,_mm_reverse_ps(hi));
    __m128 tmp1 = _mm_shuffle_ps(max1,max1,_MM_SHUFFLE(0,0,0,1));
    __m128 max_hi = _mm_max_ps(max1,tmp1);

    return _mm_cvtss_f32(_mm_max_ps(max_lo,max_hi));
}
FASTOR_INLINE double _mm256_hmax_pd(__m256d a) {
    // IVY 9OPS / HW 11 OPS
    __m256d max0 = _mm256_max_pd(a,_mm256_reverse_pd(a));
    __m256d tmp = _mm256_shuffle_pd(max0,max0,_MM_SHUFFLE(0,0,0,1));
    return _mm_cvtsd_f64(_mm256_castpd256_pd128(_mm256_max_pd(max0,tmp)));
}
#endif
//----------------------------------------------------------------------------------------------------------------//


// Horizontal min
//----------------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
FASTOR_INLINE float _mm_hmin_ps(__m128 a) {
    // 8OPS
    __m128 max0 = _mm_min_ps(a,_mm_reverse_ps(a));
    __m128 tmp = _mm_shuffle_ps(max0,max0,_MM_SHUFFLE(0,0,0,1));
    return _mm_cvtss_f32(_mm_min_ps(max0,tmp));
}
FASTOR_INLINE double _mm_hmin_pd(__m128d a) {
    // 4OPS
    return _mm_cvtsd_f64(_mm_min_pd(a,_mm_reverse_pd(a)));
}
#endif
#ifdef FASTOR_AVX_IMPL
FASTOR_INLINE float _mm256_hmin_ps(__m256 a) {
    // IVY 18OPS / HW 24 OPS
    __m128 lo = _mm256_castps256_ps128(a);
    __m128 max0 = _mm_min_ps(lo,_mm_reverse_ps(lo));
    __m128 tmp0 = _mm_shuffle_ps(max0,max0,_MM_SHUFFLE(0,0,0,1));
    __m128 max_lo = _mm_min_ps(max0,tmp0);

    __m128 hi = _mm256_extractf128_ps(a,0x1);
    __m128 max1 = _mm_min_ps(hi,_mm_reverse_ps(hi));
    __m128 tmp1 = _mm_shuffle_ps(max1,max1,_MM_SHUFFLE(0,0,0,1));
    __m128 max_hi = _mm_min_ps(max1,tmp1);

    return _mm_cvtss_f32(_mm_min_ps(max_lo,max_hi));
}
FASTOR_INLINE double _mm256_hmin_pd(__m256d a) {
    // IVY 9OPS / HW 11 OPS
    __m256d max0 = _mm256_min_pd(a,_mm256_reverse_pd(a));
    __m256d tmp = _mm256_shuffle_pd(max0,max0,_MM_SHUFFLE(0,0,0,1));
    return _mm_cvtsd_f64(_mm256_castpd256_pd128(_mm256_min_pd(max0,tmp)));
}
#endif
//----------------------------------------------------------------------------------------------------------------//


// Indexing a register
//----------------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
FASTOR_INLINE float _mm_get0_ps(__m128 a) {
    // NO OP
    return _mm_cvtss_f32(a);
}
FASTOR_INLINE float _mm_get1_ps(__m128 a) {
    // 1 OP
    return _mm_cvtss_f32(_mm_shuffle_ps(a,a,_MM_SHUFFLE(0,0,0,1)));
}
FASTOR_INLINE float _mm_get2_ps(__m128 a) {
    // 1 OP
    return _mm_cvtss_f32(_mm_shuffle_ps(a,a,_MM_SHUFFLE(0,0,0,2)));
}
FASTOR_INLINE float _mm_get3_ps(__m128 a) {
    // 1 OP
    return _mm_cvtss_f32(_mm_shuffle_ps(a,a,_MM_SHUFFLE(0,0,0,3)));
}
FASTOR_INLINE double _mm_get0_pd(__m128d a) {
    // NO OP
    return _mm_cvtsd_f64(a);
}
FASTOR_INLINE double _mm_get1_pd(__m128d a) {
    // 1 OP
    return _mm_cvtsd_f64(_mm_shuffle_pd(a,a,_MM_SHUFFLE2(0,1)));
}
#endif
#ifdef FASTOR_AVX_IMPL
FASTOR_INLINE float _mm256_get0_ps(__m256 a) {
    // NO OP
    return _mm_cvtss_f32(_mm256_castps256_ps128(a));
}
FASTOR_INLINE float _mm256_get1_ps(__m256 a) {
    // 1 OP
    __m128 lower = _mm256_castps256_ps128(a);
    return _mm_cvtss_f32(_mm_shuffle_ps(lower,lower,_MM_SHUFFLE(0,0,0,1)));
}
FASTOR_INLINE float _mm256_get2_ps(__m256 a) {
    // 1 OP
    __m128 lower = _mm256_castps256_ps128(a);
    return _mm_cvtss_f32(_mm_shuffle_ps(lower,lower,_MM_SHUFFLE(0,0,0,2)));
}
FASTOR_INLINE float _mm256_get3_ps(__m256 a) {
    // NO OP
    __m128 lower = _mm256_castps256_ps128(a);
    return _mm_cvtss_f32(_mm_shuffle_ps(lower,lower,_MM_SHUFFLE(0,0,0,3)));
}
FASTOR_INLINE float _mm256_get4_ps(__m256 a) {
    // IVY 1OP / HW 3OPS
    return _mm_cvtss_f32(_mm256_extractf128_ps(a,0x1));
}
FASTOR_INLINE float _mm256_get5_ps(__m256 a) {
    // IVY 2OPS/ HW 4OPS
    __m128 higher = _mm256_extractf128_ps(a,0x1);
    return _mm_cvtss_f32(_mm_shuffle_ps(higher,higher,_MM_SHUFFLE(0,0,0,1)));
}
FASTOR_INLINE float _mm256_get6_ps(__m256 a) {
    // IVY 2OPS/ HW 4OPS
    __m128 higher = _mm256_extractf128_ps(a,0x1);
    return _mm_cvtss_f32(_mm_shuffle_ps(higher,higher,_MM_SHUFFLE(0,0,0,2)));
}
FASTOR_INLINE float _mm256_get7_ps(__m256 a) {
    // IVY 2OPS/ HW 4OPS
    __m128 higher = _mm256_extractf128_ps(a,0x1);
    return _mm_cvtss_f32(_mm_shuffle_ps(higher,higher,_MM_SHUFFLE(0,0,0,3)));
}
FASTOR_INLINE double _mm256_get0_pd(__m256d a) {
    // NO OP
    return _mm_cvtsd_f64(_mm256_castpd256_pd128(a));
}
FASTOR_INLINE double _mm256_get1_pd(__m256d a) {
    // 1 OP
    __m128d lower = _mm256_castpd256_pd128(a);
    return _mm_cvtsd_f64(_mm_shuffle_pd(lower,lower,_MM_SHUFFLE2(0,1)));
}
FASTOR_INLINE double _mm256_get2_pd(__m256d a) {
    // IVY 1OP / HW 3OPS
    return _mm_cvtsd_f64(_mm256_extractf128_pd(a,0x1));
}
FASTOR_INLINE double _mm256_get3_pd(__m256d a) {
    // IVY 2OPS / HW 4OPS
    __m128d higher = _mm256_extractf128_pd(a,0x1);
    return _mm_cvtsd_f64(_mm_shuffle_pd(higher,higher,_MM_SHUFFLE2(0,1)));
}
#endif
//----------------------------------------------------------------------------------------------------------------//



// Integral arithmetics that are not available pre AVX2
//----------------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
FASTOR_INLINE __m128i _mm_mul_epi32x(__m128i a, __m128i b)
{
#ifdef FASTOR_SSE4_1_IMPL
    return _mm_mullo_epi32(a, b);
#else // SSE2
    __m128i a13 = _mm_shuffle_epi32(a, 0xF5);              // (-,a3,-,a1)
    __m128i b13 = _mm_shuffle_epi32(b, 0xF5);              // (-,b3,-,b1)
    __m128i prod02 = _mm_mul_epu32(a, b);                  // (-,a2*b2,-,a0*b0)
    __m128i prod13 = _mm_mul_epu32(a13, b13);              // (-,a3*b3,-,a1*b1)
    __m128i prod01 = _mm_unpacklo_epi32(prod02, prod13);   // (-,-,a1*b1,a0*b0)
    __m128i prod23 = _mm_unpackhi_epi32(prod02, prod13);   // (-,-,a3*b3,a2*b2)
    return           _mm_unpacklo_epi64(prod01, prod23);   // (ab3,ab2,ab1,ab0)
#endif


}
#endif

#ifdef FASTOR_SSE2_IMPL
FASTOR_INLINE __m128i _mm_mul_epi64(__m128i _a, __m128i _b) {
    __m128i out;
   for (FASTOR_INDEX i=0; i<2; i++) {
       ((int64_t*)&out)[i] = (((int64_t*)&_a)[i])*(((int64_t*)&_b)[i]);
   }
    return out;
}
#endif

#ifdef FASTOR_AVX_IMPL
// #ifndef FASTOR_AVX2_IMPL
FASTOR_INLINE __m256i _mm256_add_epi32x(__m256i _a, __m256i _b) {
    __m128i low_a = _mm256_castsi256_si128(_a);
    __m128i high_a = _mm256_extractf128_si256(_a,1);
    __m128i low_b = _mm256_castsi256_si128(_b);
    __m128i high_b = _mm256_extractf128_si256(_b,1);
    __m128i low = _mm_add_epi32(low_a,low_b);
    __m128i high = _mm_add_epi32(high_a,high_b);
    __m256i out = _mm256_castsi128_si256(low);
    out = _mm256_insertf128_si256(out,high,1);
    return out;
}

FASTOR_INLINE __m256i _mm256_sub_epi32x(__m256i _a, __m256i _b) {
    __m128i low_a = _mm256_castsi256_si128(_a);
    __m128i high_a = _mm256_extractf128_si256(_a,1);
    __m128i low_b = _mm256_castsi256_si128(_b);
    __m128i high_b = _mm256_extractf128_si256(_b,1);
    __m128i low = _mm_sub_epi32(low_a,low_b);
    __m128i high = _mm_sub_epi32(high_a,high_b);
    __m256i out = _mm256_castsi128_si256(low);
    out = _mm256_insertf128_si256(out,high,1);
    return out;
}

FASTOR_INLINE __m256i _mm256_mul_epi32x(__m256i _a, __m256i _b) {
    __m128i low_a = _mm256_castsi256_si128(_a);
    __m128i high_a = _mm256_extractf128_si256(_a,0x1);
    __m128i low_b = _mm256_castsi256_si128(_b);
    __m128i high_b = _mm256_extractf128_si256(_b,0x1);
    __m128i low = _mm_mul_epi32x(low_a,low_b);
    __m128i high = _mm_mul_epi32x(high_a,high_b);
    __m256i out = _mm256_castsi128_si256(low);
    out = _mm256_insertf128_si256(out,high,0x1);
    return out;
}

// 64bit
FASTOR_INLINE __m256i _mm256_add_epi64x(__m256i _a, __m256i _b) {
    __m128i low_a = _mm256_castsi256_si128(_a);
    __m128i high_a = _mm256_extractf128_si256(_a,1);
    __m128i low_b = _mm256_castsi256_si128(_b);
    __m128i high_b = _mm256_extractf128_si256(_b,1);
    __m128i low = _mm_add_epi64(low_a,low_b);
    __m128i high = _mm_add_epi64(high_a,high_b);
    __m256i out = _mm256_castsi128_si256(low);
    out = _mm256_insertf128_si256(out,high,1);
    return out;
}

FASTOR_INLINE __m256i _mm256_sub_epi64x(__m256i _a, __m256i _b) {
    __m128i low_a = _mm256_castsi256_si128(_a);
    __m128i high_a = _mm256_extractf128_si256(_a,1);
    __m128i low_b = _mm256_castsi256_si128(_b);
    __m128i high_b = _mm256_extractf128_si256(_b,1);
    __m128i low = _mm_sub_epi64(low_a,low_b);
    __m128i high = _mm_sub_epi64(high_a,high_b);
    __m256i out = _mm256_castsi128_si256(low);
    out = _mm256_insertf128_si256(out,high,1);
    return out;
}
// #else
// Note that these instruction work on alternating bytes
// FASTOR_INLINE __m256i _mm256_add_epi32x(__m256i _a, __m256i _b) {
//     return _mm256_add_epi32(_a,_b);
// }
// FASTOR_INLINE __m256i _mm256_sub_epi32x(__m256i _a, __m256i _b) {
//     return _mm256_sub_epi32(_a,_b);
// }
// FASTOR_INLINE __m256i _mm256_mul_epi32x(__m256i _a, __m256i _b) {
//     return _mm256_mul_epi32(_a,_b);
// }
// FASTOR_INLINE __m256i _mm256_add_epi64x(__m256i _a, __m256i _b) {
//     return _mm256_add_epi64(_a,_b);
// }
// FASTOR_INLINE __m256i _mm256_sub_epi64x(__m256i _a, __m256i _b) {
//     return _mm256_sub_epi64(_a,_b);
// }
// #endif

FASTOR_INLINE __m256i _mm256_div_epi32x(__m256i _a, __m256i _b) {
    // YIELDS INCORRECT
    int *a_data = (int*) &_a;
    int *b_data = (int*) &_b;
    int FASTOR_ALIGN out_data[8];
    for (int i=0; i<8; ++i)
        out_data[i] = a_data[i]/b_data[i];
    __m256i out = _mm256_setzero_si256();
    _mm256_store_si256((__m256i*)out_data,out);
    return out;
}

FASTOR_INLINE __m256i _mm256_mul_epi64x(__m256i _a, __m256i _b) {
    __m128i low_a = _mm256_castsi256_si128(_a);
    __m128i high_a = _mm256_extractf128_si256(_a,0x1);
    __m128i low_b = _mm256_castsi256_si128(_b);
    __m128i high_b = _mm256_extractf128_si256(_b,0x1);
    __m128i low = _mm_mul_epi64(low_a,low_b);
    __m128i high = _mm_mul_epi64(high_a,high_b);
    __m256i out = _mm256_castsi128_si256(low);
    out = _mm256_insertf128_si256(out,high,0x1);
    return out;
}
#endif
//----------------------------------------------------------------------------------------------------------------//



//! Some further auxilary functions C++ only
//----------------------------------------------------------------------------------------------------------------//
//----------------------------------------------------------------------------------------------------------------//
#if defined(__cplusplus)
#ifdef FASTOR_SSE2_IMPL
static FASTOR_INLINE __m128d _add_pd(__m128d a) {
    // IVY 4 OPS
    __m128 shuftmp= _mm_movehl_ps(ZEROPS, _mm_castpd_ps(a));
    __m128d shuf  = _mm_castps_pd(shuftmp);
    return  _mm_add_sd(a, shuf);
}
#endif

#ifdef FASTOR_USE_HADD // hadd is beneficial here and the flag is used in the opposite way
#ifdef FASTOR_AVX_IMPL
static FASTOR_INLINE __m128d _add_pd(__m256d a) {
    // IVY 12 OPS / HW 14 OPS
    __m128d sum_low = _add_pd(_mm256_castpd256_pd128(a));
    __m128d sum_high = _add_pd(_mm256_extractf128_pd(sum_low, 0x1));
    return _mm_add_pd(sum_high, sum_low);
}
#endif
#else
#ifdef FASTOR_AVX_IMPL
static FASTOR_INLINE __m128d _add_pd(__m256d a) {
    // IVY 9 OPS / HW 11 OPS
    __m256d sum_low = _mm256_hadd_pd(a, a);
    __m128d sum_high = _mm256_extractf128_pd(sum_low, 0x1);
    return _mm_add_pd(sum_high, _mm256_castpd256_pd128(sum_low));
}
#endif
#endif
#ifdef FASTOR_SSE3_IMPL
FASTOR_INLINE __m128 _add_ps(__m128 a) {
    // 8 OPS
    __m128 shuf = _mm_movehdup_ps(a);        // line up elements 3,1 with 2,0
    __m128 sums = _mm_add_ps(a, shuf);
    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums        = _mm_add_ss(sums, shuf);
    return      sums;
}
#endif
#ifdef FASTOR_AVX_IMPL
FASTOR_INLINE __m128 _add_ps(__m256 a) {
    // IVY 20 OPS / HW 22 OPS
    __m128 sum_low = _add_ps(_mm256_castps256_ps128(a));
    __m128 sum_high = _add_ps(_mm256_extractf128_ps(a,0x1));
    return _mm_add_ss(sum_low,sum_high);
}
#endif

// horizontal add_sub
#ifdef FASTOR_SSE2_IMPL
FASTOR_INLINE __m128 _addsub_ps(__m128 a) {
    // 8 OPS
    // If a = [a0 a1 a2 a3] this function returns (a1+a3)-(a0+a2)
    // Note that only the first element of __m128 corresponds to this
    __m128 shuf = _mm_shuffle_ps(a,a,_MM_SHUFFLE(1,0,3,2));
    __m128 sums = _mm_add_ps(a, shuf);
    shuf        = _mm_shuffle_ps(sums,sums,_MM_SHUFFLE(2,3,0,1));
    return _mm_sub_ps(shuf, sums);
}

FASTOR_INLINE __m128 _mulsub_ps(__m128 a) {
    // 10 OPS
    // If a = [a0 a1 a2 a3] this function returns (a1*a3)-(a0+a2)
    // Note that only the first element of __m128 corresponds to this
    __m128 shuf = _mm_shuffle_ps(a,a,_MM_SHUFFLE(1,0,3,2));
    __m128 muls = _mm_mul_ps(a, shuf);
    shuf        = _mm_shuffle_ps(muls,muls,_MM_SHUFFLE(2,3,0,1));
    return _mm_sub_ps(shuf, muls);
}

FASTOR_INLINE __m128d _hsub_pd(__m128d a) {
    // horizontal sub, returns a[0] - a[1]
    // 4 OPS
    return _mm_sub_sd(a,_mm_shuffle_pd(a,a,0x1));
}
#endif

#ifdef FASTOR_AVX_IMPL
// Similar to SSE4 _mm_dp_pd for dot product
FASTOR_INLINE __m128d _mm256_dp_pd(__m256d __X, __m256d __Y) {
    return _add_pd(_mm256_mul_pd(__X, __Y));
}
#endif
//----------------------------------------------------------------------------------------------------------------//


//----------------------------------------------------------------------------------------------------------------//
// Additional math functions for scalars -> the name sqrts is to remove ambiguity with libm sqrt
template<typename T, typename std::enable_if<std::is_arithmetic<T>::value,bool>::type=0>
FASTOR_INLINE T sqrts(T a) {return std::sqrt(a);}
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE float sqrts(float a) {return _mm_cvtss_f32(_mm_sqrt_ps(_mm_set1_ps(a)));}
template<>
FASTOR_INLINE double sqrts(double a) {return _mm_cvtsd_f64(_mm_sqrt_pd(_mm_set1_pd(a)));}
#endif

//----------------------------------------------------------------------------------------------------------------//
#endif
//----------------------------------------------------------------------------------------------------------------//
//----------------------------------------------------------------------------------------------------------------//



//----------------------------------------------------------------------------------------------------------------//
// helper functions for going from array to mask and vice-versa
// used when AVX512 masking is available
template <int N, enable_if_t_<N==2 || N==4 || N==8,bool> = false>
inline uint8_t array_to_mask(const int (&b)[N])
{
    uint8_t c = 0;
    for (int i=0; i < N; ++i) {
        if (b[i] == -1) {
            c |= 1 << (N - i - 1);
        }
    }
    return c;
}
template <int N, enable_if_t_<N==16,bool> = false>
inline uint16_t array_to_mask(const int (&b)[N])
{
    uint16_t c = 0;
    for (int i=0; i < N; ++i) {
        if (b[i] == -1) {
            c |= 1 << (N - i - 1);
        }
    }
    return c;
}
template <int N, enable_if_t_<N==32,bool> = false>
inline uint32_t array_to_mask(const int (&b)[N])
{
    uint32_t c = 0;
    for (int i=0; i < N; ++i) {
        if (b[i] == -1) {
            c |= 1 << (N - i - 1);
        }
    }
    return c;
}
template <int N, enable_if_t_<N==64,bool> = false>
inline uint64_t array_to_mask(const int (&b)[N])
{
    uint64_t c = 0;
    for (int i=0; i < N; ++i) {
        if (b[i] == -1) {
            c |= 1 << (N - i - 1);
        }
    }
    return c;
}


template <int N, enable_if_t_<N==2 || N==4 || N==8,bool> = false>
inline void mask_to_array(uint8_t c, int (&b)[N])
{
    for (int i=0; i < N; ++i)
        b[i] = (c & (1 << (N - i -1))) != 0;

    // set bits need to be -1 not 1
    for (int i=0; i < N; ++i)
        b[i] *= -1;
}
template <int N, enable_if_t_<N==16,bool> = false>
inline void mask_to_array(uint16_t c, int (&b)[N])
{
    for (int i=0; i < N; ++i)
        b[i] = (c & (1 << (N - i -1))) != 0;

    // set bits need to be -1 not 1
    for (int i=0; i < N; ++i)
        b[i] *= -1;
}
template <int N, enable_if_t_<N==32,bool> = false>
inline void mask_to_array(uint32_t c, int (&b)[N])
{
    for (int i=0; i < N; ++i)
        b[i] = (c & (1 << (N - i -1))) != 0;

    // set bits need to be -1 not 1
    for (int i=0; i < N; ++i)
        b[i] *= -1;
}
template <int N, enable_if_t_<N==64,bool> = false>
inline void mask_to_array(uint64_t c, int (&b)[N])
{
    for (int i=0; i < N; ++i)
        b[i] = (c & (1 << (N - i -1))) != 0;

    // set bits need to be -1 not 1
    for (int i=0; i < N; ++i)
        b[i] *= -1;
}
//----------------------------------------------------------------------------------------------------------------//

} // end of namespace Fastor


#endif // EXT_INTRIN_H
