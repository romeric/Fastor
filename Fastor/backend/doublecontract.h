#ifndef DOUBLECONTRACT_H
#define DOUBLECONTRACT_H

#include "Fastor/config/config.h"
#include "Fastor/meta/meta.h"
#include "Fastor/simd_vector/SIMDVector.h"

namespace Fastor {


template<typename T, size_t M, size_t N,
    enable_if_t_<is_greater_equal_v_<4*choose_best_simd_t<SIMDVector<T,DEFAULT_ABI>,M*N>::Size, M*N >, bool> = false>
FASTOR_INLINE T _doublecontract(const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b) {

    constexpr size_t Size = M*N;
    using V = choose_best_simd_t<SIMDVector<T,DEFAULT_ABI>,Size>;

    V omm0;
    size_t i = 0;
    for (; i< ROUND_DOWN(Size,V::Size); i+=V::Size) {
        const V amm0(&a[i],false);
        const V bmm0(&b[i],false);

        omm0 = fmadd(amm0,bmm0,omm0);
    }
    T scalar = static_cast<T>(0);
    for (; i < Size; ++i) {
        scalar += a[i]*b[i];
    }
    return omm0.sum() + scalar;
}

template<typename T, size_t M, size_t N,
    enable_if_t_<is_less_v_<4*choose_best_simd_t<SIMDVector<T,DEFAULT_ABI>,M*N>::Size, M*N >, bool> = false>
FASTOR_INLINE T _doublecontract(const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b) {

    constexpr size_t Size = M*N;
    using V = choose_best_simd_t<SIMDVector<T,DEFAULT_ABI>,Size>;

    V omm0, omm1, omm2, omm3;
    size_t i = 0;
    for (; i< ROUND_DOWN(Size,4*V::Size); i+=4*V::Size) {
        const V amm0(&a[i],false);
        const V amm1(&a[i+V::Size],false);
        const V amm2(&a[i+2*V::Size],false);
        const V amm3(&a[i+3*V::Size],false);

        const V bmm0(&b[i],false);
        const V bmm1(&b[i+V::Size],false);
        const V bmm2(&b[i+2*V::Size],false);
        const V bmm3(&b[i+3*V::Size],false);

        omm0 = fmadd(amm0,bmm0,omm0);
        omm1 = fmadd(amm1,bmm1,omm1);
        omm2 = fmadd(amm2,bmm2,omm2);
        omm3 = fmadd(amm3,bmm3,omm3);
    }
    for (; i< ROUND_DOWN(Size,2*V::Size); i+=2*V::Size) {
        const V amm0(&a[i],false);
        const V amm1(&a[i+V::Size],false);

        const V bmm0(&b[i],false);
        const V bmm1(&b[i+V::Size],false);

        omm0 = fmadd(amm0,bmm0,omm0);
        omm1 = fmadd(amm1,bmm1,omm1);
    }
    for (; i< ROUND_DOWN(Size,V::Size); i+=V::Size) {
        const V amm0(&a[i],false);
        const V bmm0(&b[i],false);

        omm0 = fmadd(amm0,bmm0,omm0);
    }
    T scalar = static_cast<T>(0);
    for (; i < Size; ++i) {
        scalar += a[i]*b[i];
    }
    return (omm0 + omm1 + omm2 + omm3).sum() + scalar;
}


#ifdef FASTOR_AVX_IMPL

template<>
FASTOR_INLINE float _doublecontract<float,2,2>(const float* FASTOR_RESTRICT a, const float* FASTOR_RESTRICT b) {
    return _mm_sum_ps(_mm_mul_ps(_mm_loadu_ps(a),_mm_loadu_ps(b)));
}

template<>
FASTOR_INLINE float _doublecontract<float,3,3>(const float* FASTOR_RESTRICT a, const float* FASTOR_RESTRICT b) {
    float r1 = _mm256_sum_ps(_mm256_mul_ps(_mm256_loadu_ps(a),_mm256_loadu_ps(b)));
    float r2 = _mm_sum_ps(_mm_mul_ss(_mm_load_ss(a+8),_mm_load_ss(b+8)));
    return r1+r2;
}

template<>
FASTOR_INLINE double _doublecontract<double,2,2>(const double* FASTOR_RESTRICT a, const double* FASTOR_RESTRICT b) {
    return _mm256_sum_pd(_mm256_mul_pd(_mm256_loadu_pd(a),_mm256_loadu_pd(b)));
}

template<>
FASTOR_INLINE double _doublecontract<double,3,3>(const double* FASTOR_RESTRICT a, const double* FASTOR_RESTRICT b) {
    __m256d r1 = _mm256_mul_pd(_mm256_loadu_pd(a),_mm256_loadu_pd(b));
    __m256d r2 = _mm256_mul_pd(_mm256_loadu_pd(a+4),_mm256_loadu_pd(b+4));
    __m128d r3 = _mm_mul_sd(_mm_load_sd(a+8),_mm_load_sd(b+8));
    __m128d r4 = _add_pd(_mm256_add_pd(r1,r2));
    __m128d summ = _mm_add_pd(_add_pd(r3),r4);
    return _mm_cvtsd_f64(summ);
}


#endif

// doublecontract and transpose
template<typename T, size_t M, size_t N>
FASTOR_INLINE T _doublecontract_transpose(const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b) {
    T dc = static_cast<T>(0);
    for (FASTOR_INDEX i=0; i<M; ++i)
        for (FASTOR_INDEX j=0; j<N; ++j)
            dc += a[i*N+j]*b[j*M+i];
    return dc;
}


}

#endif // DOUBLECONTRACT_H

