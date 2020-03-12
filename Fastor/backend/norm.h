#ifndef NORM_H
#define NORM_H

#include "Fastor/commons/commons.h"
#include "Fastor/extended_intrinsics/extintrin.h"
#include "Fastor/simd_vector/SIMDVector.h"

namespace Fastor {

template<typename T, size_t N>
FASTOR_INLINE double _norm_nonfloating(const T* FASTOR_RESTRICT a) {

    using V = SIMDVector<T,DEFAULT_ABI>;
    constexpr int size = N;
    constexpr int stride = V::Size;
    int i = 0;

    // Unroll upto register size
    V vec_a, vec_out;
    for (; i< ROUND_DOWN(size,stride); i+=stride) {
        vec_a.load(a+i);
        vec_out += vec_a*vec_a;
    }
    // Take care of the remainder
    T scalar = static_cast<T>(0);
    for (; i< size; ++i) {
        scalar += a[i]*a[i];
    }
    return sqrts(static_cast<double>(vec_out.sum() + scalar));
}


template<typename T, size_t N>
FASTOR_INLINE T _norm(const T* FASTOR_RESTRICT a) {

    using V = SIMDVector<T,DEFAULT_ABI>;
    constexpr int size = N;
    constexpr int stride = V::Size;
    int i = 0;

    // Unroll upto register size
    V vec_a, vec_out;
    for (; i< ROUND_DOWN(size,stride); i+=stride) {
        vec_a.load(a+i);
#ifdef __FMA__
        vec_out = fmadd(vec_a,vec_a,vec_out);
#else
        vec_out += vec_a*vec_a;
#endif
    }
    // Take care of the remainder
    T scalar = static_cast<T>(0);
    for (; i< size; ++i) {
        scalar += a[i]*a[i];
    }
    return sqrts(vec_out.sum() + scalar);
}

#ifdef __SSE4_2__
template<>
FASTOR_INLINE float _norm<float,4>(const float * FASTOR_RESTRICT a) {
    // IVY 33 OPS / HW 31 OPS
    __m128 a_reg = _mm_load_ps(a);
    return _mm_cvtss_f32(_mm_sqrt_ps(_add_ps(_mm_mul_ps(a_reg,a_reg))));
}
#endif
#ifdef __AVX__
template<>
FASTOR_INLINE float _norm<float,9>(const float * FASTOR_RESTRICT a) {
    // IVY & HW 61 OPS
    __m256 a_reg = _mm256_load_ps(a);
    __m128 a_end = _mm_load_ss(a+8);
    __m128 a0 = _add_ps(_mm256_mul_ps(a_reg,a_reg));
    __m128 a1 = _add_ps(_mm_mul_ps(a_end,a_end));
    return _mm_cvtss_f32(_mm_sqrt_ps(_mm_add_ss(a0,a1)));
}


template<>
FASTOR_INLINE double _norm<double,4>(const double * FASTOR_RESTRICT a) {
    // IVY 34 OPS / HW 36 OPS
    __m256d a_reg = _mm256_load_pd(a);
    return _mm_cvtsd_f64(_mm_sqrt_pd(_add_pd(_mm256_mul_pd(a_reg,a_reg))));
}

template<>
FASTOR_INLINE double _norm<double,9>(const double * FASTOR_RESTRICT a) {
    // IVY 63 OPS / HW 67 OPS
    __m256d a_low = _mm256_load_pd(a);
    __m256d a_high = _mm256_load_pd(a+4);
    __m128d a_end = _mm_load_sd(a+8);
    __m128d a0 = _add_pd(_mm256_mul_pd(a_low,a_low));
    __m128d a1 = _add_pd(_mm256_mul_pd(a_high,a_high));
    __m128d a2 = _add_pd(_mm_mul_pd(a_end,a_end));
    return _mm_cvtsd_f64(_mm_sqrt_pd(_mm_add_sd(a2,(_mm_add_sd(a0,a1)))));
}
#endif


}

#endif // NORM_H

