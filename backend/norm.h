#ifndef NORM_H
#define NORM_H

#include "commons/commons.h"
#include "extended_intrinsics/extintrin.h"
#include "simd_vector/SIMDVector.h"

namespace Fastor {

template<typename T, size_t N>
FASTOR_INLINE T _norm(const T* __restrict__ a) {

    constexpr int size = N;
    constexpr int unroll_upto = SIMDVector<T>::unroll_size(size);
    constexpr int stride = SIMDVector<T>::Size;
    int i = 0;

    // Unroll upto register size
    SIMDVector<T> vec_a=static_cast<T>(0), vec_out=static_cast<T>(0);
    for (; i< unroll_upto; i+=stride) {
        vec_a.load(a+i);
        vec_out += vec_a*vec_a;
    }
    // Take care of the remainder
    T scalar = static_cast<T>(0);
    for (FASTOR_INDEX j=i; j< size; j++) {
        scalar += a[j]*a[j];
    }
    return std::sqrt(vec_out.sum() + scalar);
}

template<>
FASTOR_INLINE float _norm<float,4>(const float * __restrict__ a) {
    // IVY 33 OPS / HW 31 OPS
    __m128 a_reg = _mm_load_ps(a);
    return _mm_cvtss_f32(_mm_sqrt_ps(_add_ps(_mm_mul_ps(a_reg,a_reg))));
}

template<>
FASTOR_INLINE float _norm<float,9>(const float * __restrict__ a) {
    // IVY & HW 45 OPS
    __m256 a_reg = _mm256_load_ps(a);
    return _mm_cvtss_f32(_mm_sqrt_ps(_add_ps(_mm256_mul_ps(a_reg,a_reg))));
}


template<>
FASTOR_INLINE double _norm<double,4>(const double * __restrict__ a) {
    // IVY 34 OPS / HW 36 OPS
    __m256d a_reg = _mm256_load_pd(a);
    return _mm_cvtsd_f64(_mm_sqrt_pd(_add_pd(_mm256_mul_pd(a_reg,a_reg))));
}

template<>
FASTOR_INLINE double _norm<double,9>(const double * __restrict__ a) {
    // IVY 103 OPS / HW 107 OPS
    __m256d a_low = _mm256_load_pd(a);
    __m256d a_high = _mm256_load_pd(a+4);
    __m128d a_end = _mm_load_sd(a+8);
    __m128d a0 = _mm_sqrt_pd(_add_pd(_mm256_mul_pd(a_low,a_low)));
    __m128d a1 = _mm_sqrt_pd(_add_pd(_mm256_mul_pd(a_high,a_high)));
    __m128d a2 = _mm_sqrt_pd(_add_pd(_mm_mul_pd(a_end,a_end)));
    return _mm_cvtsd_f64(_mm_add_sd(a2,(_mm_add_sd(a0,a1))));
}


}

#endif // NORM_H

