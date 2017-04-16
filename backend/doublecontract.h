#ifndef DOUBLECONTRACT_H
#define DOUBLECONTRACT_H

#include "commons/commons.h"
#include "simd_vector/SIMDVector.h"

namespace Fastor {

template<typename T, size_t M, size_t N>
FASTOR_INLINE T _doublecontract(const T* __restrict__ a, const T* __restrict__ b) {

    using V = SIMDVector<T,DEFAULT_ABI>;
    constexpr int size = M*N;
    constexpr int stride = V::Size;
    int i = 0;

    V vec_a, vec_b, vec_out;
    for (; i< ROUND_DOWN(size,stride); i+=stride) {
        vec_a.load(a+i);
        vec_b.load(b+i);
#ifndef __FMA__
        vec_out += vec_a*vec_b;
#else
        vec_out = fmadd(vec_a,vec_b,vec_out);
#endif
    }
    T scalar = static_cast<T>(0);
    for (; i < size; ++i) {
        scalar += a[i]*b[i];
    }
    return vec_out.sum() + scalar;
}

#ifdef __AVX__

template<>
FASTOR_INLINE float _doublecontract<float,2,2>(const float* __restrict__ a, const float* __restrict__ b) {
    return _mm_sum_ps(_mm_mul_ps(_mm_load_ps(a),_mm_load_ps(b)));
}

template<>
FASTOR_INLINE float _doublecontract<float,3,3>(const float* __restrict__ a, const float* __restrict__ b) {
    float r1 = _mm256_sum_ps(_mm256_mul_ps(_mm256_load_ps(a),_mm256_load_ps(b)));
    float r2 = _mm_sum_ps(_mm_mul_ss(_mm_load_ss(a+8),_mm_load_ss(b+8)));
    return r1+r2;
}

template<>
FASTOR_INLINE double _doublecontract<double,2,2>(const double* __restrict__ a, const double* __restrict__ b) {
    return _mm256_sum_pd(_mm256_mul_pd(_mm256_load_pd(a),_mm256_load_pd(b)));
}

template<>
FASTOR_INLINE double _doublecontract<double,3,3>(const double* __restrict__ a, const double* __restrict__ b) {
    __m256d r1 = _mm256_mul_pd(_mm256_load_pd(a),_mm256_load_pd(b));
    __m256d r2 = _mm256_mul_pd(_mm256_load_pd(a+4),_mm256_load_pd(b+4));
    __m128d r3 = _mm_mul_sd(_mm_load_sd(a+8),_mm_load_sd(b+8));
    __m128d r4 = _add_pd(_mm256_add_pd(r1,r2));
    __m128d summ = _mm_add_pd(_add_pd(r3),r4);
    return _mm_cvtsd_f64(summ);
}


#endif

// doublecontract and transpose
template<typename T, size_t M, size_t N>
FASTOR_INLINE T _doublecontract_transpose(const T* __restrict__ a, const T* __restrict__ b) {
    T dc = static_cast<T>(0);
    for (FASTOR_INDEX i=0; i<M; ++i)
        for (FASTOR_INDEX j=0; j<N; ++j)
            dc += a[i*N+j]*b[j*M+i];
    return dc;
}


}

#endif // DOUBLECONTRACT_H

