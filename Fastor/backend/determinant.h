#ifndef DETERMINANT_H
#define DETERMINANT_H

#include "Fastor/config/config.h"
#include "Fastor/simd_vector/extintrin.h"
#include "Fastor/meta/tensor_meta.h"

namespace Fastor {


#ifndef FASTOR_AVX_IMPL
template<typename T, size_t M, size_t N, typename std::enable_if<M==2 && N==2, bool>::type=0>
#else
template<typename T, size_t M, size_t N, typename std::enable_if<!std::is_same<T,double>::value &&
    !std::is_same<T,float>::value && M==2 && N==2, bool>::type=0>
#endif
FASTOR_INLINE T _det(const T* FASTOR_RESTRICT a) {
    return a[0] * a[3] - a[1] * a[2];
}


#ifndef FASTOR_AVX_IMPL
template<typename T, size_t M, size_t N, typename std::enable_if<M==3 && N==3, bool>::type=0>
#else
template<typename T, size_t M, size_t N, typename std::enable_if<!std::is_same<T,double>::value &&
    !std::is_same<T,float>::value && M==3 && N==3, bool>::type=0>
#endif
FASTOR_INLINE T _det(const T* FASTOR_RESTRICT a) {
    return a[0]*a[4]*a[8] + a[1]*a[5]*a[6] + a[2]*a[3]*a[7] - a[2]*a[4]*a[6] - a[1]*a[3]*a[8] - a[0]*a[5]*a[7];
}

template<typename T, size_t M, size_t N, typename std::enable_if<M==4 && N==4, bool>::type=0>
FASTOR_INLINE T _det(const T* FASTOR_RESTRICT m) {
    return   m[12] * m[9]  * m[6]  * m[3]   -  m[8] * m[13] * m[6]  * m[3]   -
             m[12] * m[5]  * m[10] * m[3]   +  m[4] * m[13] * m[10] * m[3]   +
             m[8]  * m[5]  * m[14] * m[3]   -  m[4] * m[9]  * m[14] * m[3]   -
             m[12] * m[9]  * m[2]  * m[7]   +  m[8] * m[13] * m[2]  * m[7]   +
             m[12] * m[1]  * m[10] * m[7]   -  m[0] * m[13] * m[10] * m[7]   -
             m[8]  * m[1]  * m[14] * m[7]   +  m[0] * m[9]  * m[14] * m[7]   +
             m[12] * m[5]  * m[2]  * m[11]  -  m[4] * m[13] * m[2]  * m[11]  -
             m[12] * m[1]  * m[6]  * m[11]  +  m[0] * m[13] * m[6]  * m[11]  +
             m[4]  * m[1]  * m[14] * m[11]  -  m[0] * m[5]  * m[14] * m[11]  -
             m[8]  * m[5]  * m[2]  * m[15]  +  m[4] * m[9]  * m[2]  * m[15]  +
             m[8]  * m[1]  * m[6]  * m[15]  -  m[0] * m[9]  * m[6]  * m[15]  -
             m[4]  * m[1]  * m[10] * m[15]  +  m[0] * m[5]  * m[10] * m[15];
}

template<typename T, size_t M, size_t N, typename std::enable_if<is_greater<M,4>::value || is_greater<N,4>::value, bool>::type=0>
FASTOR_INLINE T _det(const T* FASTOR_RESTRICT a) {
    static_assert(M==N, "2D TENSOR MUST BE SQUARE");
    assert(false && "2D TENSOR MUST BE SQUARE");
}


#ifdef FASTOR_AVX_IMPL
template<typename T, size_t M, size_t N, typename std::enable_if<!std::is_same<T,double>::value &&
    std::is_same<T,float>::value && M==2 && N==2, bool>::type=0>
FASTOR_INLINE T _det(const T* FASTOR_RESTRICT a) {
    // 10 OPS
    __m128 a1 = _mm_load_ps(a);
    __m128 a2 = _mm_shuffle_ps(a1,a1,_MM_SHUFFLE(0,1,2,3));
    __m128 a3 = _mm_mul_ps(a1,a2);
    return _mm_cvtss_f32(_mm_sub_ss(a3,_mm_shuffle_ps(a3,a3,_MM_SHUFFLE(0,0,0,1))));
}

template<typename T, size_t M, size_t N, typename std::enable_if<!std::is_same<T,double>::value &&
    std::is_same<T,float>::value && M==3 && N==3, bool>::type=0>
FASTOR_INLINE T _det(const T* FASTOR_RESTRICT a) {
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

template<typename T, size_t M, size_t N, typename std::enable_if<std::is_same<T,double>::value &&
    !std::is_same<T,float>::value && M==2 && N==2, bool>::type=0>
FASTOR_INLINE T _det(const T* FASTOR_RESTRICT a) {
    // 10 OPS
    __m128d a1 = _mm_load_pd(a);
    __m128d a2 = _mm_load_pd(a+2);
    __m128d a3 = _mm_mul_pd(a1,_mm_shuffle_pd(a2,a2,1));
    return _mm_cvtsd_f64(_mm_sub_pd(a3,_mm_shuffle_pd(a3,a3,0x1)));
}

template<typename T, size_t M, size_t N, typename std::enable_if<std::is_same<T,double>::value &&
    !std::is_same<T,float>::value && M==3 && N==3, bool>::type=0>
FASTOR_INLINE T _det(const T* FASTOR_RESTRICT a) {
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

