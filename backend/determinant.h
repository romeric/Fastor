#ifndef DETERMINANT_H
#define DETERMINANT_H

#include "commons/commons.h"
#include "extended_intrinsics/extintrin.h"

namespace Fastor {

template<typename T, size_t M, size_t N>
FASTOR_INLINE T _det(const T* __restrict__ a);

#ifdef __AVX__
template<>
FASTOR_INLINE float _det<float,2,2>(const float* __restrict__ a) {
    // 10 OPS
    __m128 a1 = _mm_load_ps(a);
    __m128 a2 = _mm_shuffle_ps(a1,a1,_MM_SHUFFLE(0,1,2,3));
    __m128 a3 = _mm_mul_ps(a1,a2);
    return _mm_cvtss_f32(_mm_sub_ss(a3,_mm_shuffle_ps(a3,a3,_MM_SHUFFLE(0,0,0,1))));
}

template<>
FASTOR_INLINE float _det<float,3,3>(const float* __restrict__ a) {
    // ?? OPS
    __m128 r0 = {a[2],a[1],a[0],0.};
    __m128 r1 = {a[3],a[5],a[4],0.};
    __m128 r2 = {a[7],a[6],a[8],0.};

    __m128 r3 = {a[6],a[7],a[8],0.};
    __m128 r4 = {a[4],a[5],a[3],0.};
    __m128 r5 = {a[2],a[0],a[1],0.};

    __m128 out0 = _mm_mul_ps(r2,_mm_mul_ps(r0,r1));
    __m128 out1 = _mm_mul_ps(r3,_mm_mul_ps(r4,r5));

    return _mm_cvtss_f32(_mm_sub_ss(_add_ps(out0),_add_ps(out1)));
}

template<>
FASTOR_INLINE double _det<double,2,2>(const double* __restrict__ a) {
    // 10 OPS
    __m128d a1 = _mm_load_pd(a);
    __m128d a2 = _mm_load_pd(a+2);
    __m128d a3 = _mm_mul_pd(a1,_mm_shuffle_pd(a2,a2,1));
    return _mm_cvtsd_f64(_mm_sub_pd(a3,_mm_shuffle_pd(a3,a3,0x1)));
}

template<>
FASTOR_INLINE double _det<double,3,3>(const double* __restrict__ a) {
    // ?? OPS
    __m256d r0 = {a[2],a[1],a[0],0.};
    __m256d r1 = {a[3],a[5],a[4],0.};
    __m256d r2 = {a[7],a[6],a[8],0.};

    __m256d r3 = {a[6],a[7],a[8],0.};
    __m256d r4 = {a[4],a[5],a[3],0.};
    __m256d r5 = {a[2],a[0],a[1],0.};

    __m256d out0 = _mm256_mul_pd(r2,_mm256_mul_pd(r0,r1));
    __m256d out1 = _mm256_mul_pd(r3,_mm256_mul_pd(r4,r5));

    return _mm_cvtsd_f64(_mm_sub_sd(_add_pd(out0),_add_pd(out1)));
}

#endif

}

#endif // DETERMINANT_H

