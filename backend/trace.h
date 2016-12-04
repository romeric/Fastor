#ifndef TRACE_H
#define TRACE_H

#include "commons/commons.h"
#include "extended_intrinsics/extintrin.h"

namespace Fastor {


template<typename T, size_t M, size_t N, typename std::enable_if<M==N,bool>::type=0>
FASTOR_INLINE T _trace(const T * __restrict__ a) {
    T sum = static_cast<T>(0);
    for (FASTOR_INDEX i=0; i<M; ++i)
        sum +=a[i*N+i];
}

template<>
FASTOR_INLINE double _trace<double,2,2>(const double * __restrict__ a) {
    // AVX VERSION
    // IVY 5 OPS / HW 7 OPS
//    __m256d a_reg = _mm256_load_pd(a);
//    __m128d a_high = _mm256_extractf128_pd(a_reg,0x1);
//    return _mm_cvtsd_f64(_mm_add_sd(_mm256_castpd256_pd128(a_reg),_mm_shuffle_pd(a_high,a_high,0x1)));

    // SSE VERSION
    // 3 OPS
    __m128d a0 = _mm_load_sd(a);
    __m128d a1 = _mm_load_sd(a+3);
    return _mm_cvtsd_f64(_mm_add_pd(a0,a1));
}

template<>
FASTOR_INLINE double _trace<double,3,3>(const double * __restrict__ a) {
    // No benefit in AVX
    return _mm_cvtsd_f64(_mm_add_sd(_mm_load_sd(a),_mm_add_sd(_mm_load_sd(a+4),_mm_load_sd(a+8))));
}

template<>
FASTOR_INLINE float _trace<float,2,2>(const float * __restrict__ a) {
    __m128 a_reg = _mm_load_ps(a);
    return _mm_cvtss_f32(_mm_add_ss(a_reg,_mm_reverse_ps(a_reg)));
}

template<>
FASTOR_INLINE float _trace<float,3,3>(const float * __restrict__ a) {
    __m256 a_reg = _mm256_load_ps(a);
    __m128 sum_two = _mm_add_ps(_mm256_castps256_ps128(a_reg),_mm256_extractf128_ps(a_reg,0x1));
    return _mm_cvtss_f32(_mm_add_ss(sum_two,_mm_load_ss(a+8)));
}

}

#endif // TRACE_H

