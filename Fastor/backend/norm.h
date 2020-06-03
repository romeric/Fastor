#ifndef NORM_H
#define NORM_H

#include "Fastor/config/config.h"
#include "Fastor/meta/meta.h"
#include "Fastor/simd_vector/extintrin.h"
#include "Fastor/simd_vector/SIMDVector.h"

namespace Fastor {

template<typename T, size_t N,
#ifdef FASTOR_AVX512_IMPL
    enable_if_t_<is_less_v_<8*internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size, N >, bool> = false>
#else
    enable_if_t_<is_less_v_<4*internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size, N >, bool> = false>
#endif
FASTOR_INLINE T _norm(const T* FASTOR_RESTRICT a) {

    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;
    T _scal = 0;
#ifdef FASTOR_AVX512_IMPL
    V omm0, omm1, omm2, omm3, omm4, omm5, omm6, omm7;
#else
    V omm0, omm1, omm2, omm3;
#endif
    FASTOR_INDEX i = 0;

    // With AVX utilises all the 16 registers but hurts the performance
    // due to spill if eval has created temporary registers so only
    // activated for AVX512
#ifdef FASTOR_AVX512_IMPL
    for (; i < ROUND_DOWN(N,8*V::Size); i+=8*V::Size) {
        const V smm0(&a[i]           , false);
        const V smm1(&a[i+V::Size]   , false);
        const V smm2(&a[i+2*V::Size] , false);
        const V smm3(&a[i+3*V::Size] , false);
        const V smm4(&a[i+4*V::Size] , false);
        const V smm5(&a[i+5*V::Size] , false);
        const V smm6(&a[i+6*V::Size] , false);
        const V smm7(&a[i+7*V::Size] , false);
        omm0 = fmadd(smm0,smm0,omm0);
        omm1 = fmadd(smm1,smm1,omm1);
        omm2 = fmadd(smm2,smm2,omm2);
        omm3 = fmadd(smm3,smm3,omm3);
        omm4 = fmadd(smm4,smm4,omm4);
        omm5 = fmadd(smm5,smm5,omm5);
        omm6 = fmadd(smm6,smm6,omm6);
        omm7 = fmadd(smm7,smm7,omm7);
    }
#endif
    for (; i < ROUND_DOWN(N,4*V::Size); i+=4*V::Size) {
        const V smm0(&a[i]           , false);
        const V smm1(&a[i+V::Size]   , false);
        const V smm2(&a[i+2*V::Size] , false);
        const V smm3(&a[i+3*V::Size] , false);
        omm0 = fmadd(smm0,smm0,omm0);
        omm1 = fmadd(smm1,smm1,omm1);
        omm2 = fmadd(smm2,smm2,omm2);
        omm3 = fmadd(smm3,smm3,omm3);
    }
    for (; i < ROUND_DOWN(N,2*V::Size); i+=2*V::Size) {
        const V smm0(&a[i]           , false);
        const V smm1(&a[i+V::Size]   , false);
        omm0 = fmadd(smm0,smm0,omm0);
        omm1 = fmadd(smm1,smm1,omm1);
    }
    for (; i < ROUND_DOWN(N,V::Size); i+=V::Size) {
        const V smm0(&a[i]           , false);
        omm0 = fmadd(smm0,smm0,omm0);
    }
    for (; i < N; ++i) {
        const auto smm0(a[i]);
        _scal += smm0*smm0;
    }
#ifdef FASTOR_AVX512_IMPL
    return sqrts( (omm0 + omm1 + omm2 + omm3 + omm4 + omm5 + omm6 + omm7).sum() + _scal);
#else
    return sqrts( (omm0 + omm1 + omm2 + omm3).sum() + _scal);
#endif
}

template<typename T, size_t N,
#ifdef FASTOR_AVX512_IMPL
    enable_if_t_<is_greater_equal_v_<8*internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size, N >, bool> = false>
#else
    enable_if_t_<is_greater_equal_v_<4*internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size, N >, bool> = false>
#endif
FASTOR_INLINE T _norm(const T* FASTOR_RESTRICT a) {

    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;
    T _scal = 0;
    V omm0;
    FASTOR_INDEX i = 0;
    for (; i < ROUND_DOWN(N,V::Size); i+=V::Size) {
        const V smm0(&a[i]           , false);
        omm0 = fmadd(smm0,smm0,omm0);
    }
    for (; i < N; ++i) {
        const auto smm0(a[i]);
        _scal += smm0*smm0;
    }
    return sqrts( omm0.sum() + _scal);
}

#ifdef FASTOR_SSE4_2_IMPL
template<>
FASTOR_INLINE float _norm<float,4>(const float * FASTOR_RESTRICT a) {
    // IVY 33 OPS / HW 31 OPS
    __m128 a_reg = _mm_load_ps(a);
    return _mm_cvtss_f32(_mm_sqrt_ps(_add_ps(_mm_mul_ps(a_reg,a_reg))));
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE float _norm<float,9>(const float * FASTOR_RESTRICT a) {
    // IVY & HW 61 OPS
    __m256 a_reg = _mm256_loadu_ps(a);
    __m128 a_end = _mm_load_ss(a+8);
    __m128 a0 = _add_ps(_mm256_mul_ps(a_reg,a_reg));
    __m128 a1 = _add_ps(_mm_mul_ps(a_end,a_end));
    return _mm_cvtss_f32(_mm_sqrt_ps(_mm_add_ss(a0,a1)));
}


template<>
FASTOR_INLINE double _norm<double,4>(const double * FASTOR_RESTRICT a) {
    // IVY 34 OPS / HW 36 OPS
    __m256d a_reg = _mm256_loadu_pd(a);
    return _mm_cvtsd_f64(_mm_sqrt_pd(_add_pd(_mm256_mul_pd(a_reg,a_reg))));
}

template<>
FASTOR_INLINE double _norm<double,9>(const double * FASTOR_RESTRICT a) {
    // IVY 63 OPS / HW 67 OPS
    __m256d a_low = _mm256_loadu_pd(a);
    __m256d a_high = _mm256_loadu_pd(a+4);
    __m128d a_end = _mm_load_sd(a+8);
    __m128d a0 = _add_pd(_mm256_mul_pd(a_low,a_low));
    __m128d a1 = _add_pd(_mm256_mul_pd(a_high,a_high));
    __m128d a2 = _add_pd(_mm_mul_pd(a_end,a_end));
    return _mm_cvtsd_f64(_mm_sqrt_pd(_mm_add_sd(a2,(_mm_add_sd(a0,a1)))));
}
#endif


}

#endif // NORM_H

