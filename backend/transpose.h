#ifndef TRANSPOSE_H
#define TRANSPOSE_H


#include "commons/commons.h"
#include "extended_intrinsics/extintrin.h"

namespace Fastor {

template<typename T, size_t M, size_t N>
FASTOR_INLINE void _transpose(const T * __restrict__ a, T * __restrict__ out) {
    for (size_t i=0; i< M; ++i)
        for (size_t j=0; j<N; ++j)
            out[j*M+i] = a[i*N+j];
}

#ifdef __SSE4_2__
template<>
FASTOR_INLINE void _transpose<float,2,2>(const float * __restrict__ a, float * __restrict__ out) {
    __m128 a_reg = _mm_load_ps(a);
    _mm_store_ps(out,_mm_shuffle_ps(a_reg,a_reg,_MM_SHUFFLE(3,1,2,0)));
}

template<>
FASTOR_INLINE void _transpose<float,3,3>(const float * __restrict__ a, float * __restrict__ out) {
    __m128 a_low = _mm_load_ps(a);
    __m128 a_high = _mm_load_ps(a+4);
    __m128 a_end = _mm_load_ss(a+8);

    __m128 col0 = _mm_shuffle_ps(a_low,a_high,_MM_SHUFFLE(0,2,3,0));
    __m128 col1 = _mm_shuffle_ps(a_high,a_low,_MM_SHUFFLE(2,2,0,3));

    _mm_store_ps(out,col0);
    _mm_store_ss(out+3,_mm_shuffle_ps(a_low,a_low,_MM_SHUFFLE(1,1,1,1)));
    _mm_store_ps(out+4,col1);
    _mm_store_ss(out+7,_mm_shuffle_ps(a_high,a_high,_MM_SHUFFLE(1,1,1,1)));
    _mm_store_ss(out+8,a_end);
}

template<>
FASTOR_INLINE void _transpose<float,4,4>(const float * __restrict__ a, float * __restrict__ out) {
    __m128 row1 = _mm_load_ps(a);
    __m128 row2 = _mm_load_ps(a+4);
    __m128 row3 = _mm_load_ps(a+8);
    __m128 row4 = _mm_load_ps(a+12);
     _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
     _mm_store_ps(out, row1);
     _mm_store_ps(out+4, row2);
     _mm_store_ps(out+8, row3);
     _mm_store_ps(out+12, row4);
}
#endif

#ifdef __AVX__
template<>
FASTOR_INLINE void _transpose<float,8,8>(const float * __restrict__ a, float * __restrict__ out) {
    __m256 row1 = _mm256_load_ps(a);
    __m256 row2 = _mm256_load_ps(a+8);
    __m256 row3 = _mm256_load_ps(a+16);
    __m256 row4 = _mm256_load_ps(a+24);
    __m256 row5 = _mm256_load_ps(a+32);
    __m256 row6 = _mm256_load_ps(a+40);
    __m256 row7 = _mm256_load_ps(a+48);
    __m256 row8 = _mm256_load_ps(a+56);
     _MM_TRANSPOSE8_PS(row1, row2, row3, row4, row5, row6, row7, row8);
     _mm256_store_ps(out, row1);
     _mm256_store_ps(out+8, row2);
     _mm256_store_ps(out+16, row3);
     _mm256_store_ps(out+24, row4);
     _mm256_store_ps(out+32, row5);
     _mm256_store_ps(out+40, row6);
     _mm256_store_ps(out+48, row7);
     _mm256_store_ps(out+56, row8);
}


template<>
FASTOR_INLINE void _transpose<double,2,2>(const double* __restrict__ a, double* __restrict__ out) {
    // IVY 4 OPS / HW 8 OPS
    __m256d a1 =  _mm256_load_pd(a);
    __m128d a2 =  _mm256_castpd256_pd128(a1);
    __m128d a3 =  _mm256_extractf128_pd(a1,0x1);
    __m128d a4 = _mm_shuffle_pd(a2,a3,0x0);
    a3 = _mm_shuffle_pd(a2,a3,0x3);
    a1 = _mm256_castpd128_pd256(a4);
    a1 = _mm256_insertf128_pd(a1,a3,0x1);
    _mm256_store_pd(out,a1);
}
#endif

#ifdef __SSE4_2__
template<>
FASTOR_INLINE void _transpose<double,3,3>(const double* __restrict__ a, double* __restrict__ out) {
    /*-------------------------------------------------------*/
    // SSE VERSION - Requires 32byte alignment
    // all loads are 16 byte aligned if a is 32byte aligned
    __m128d a11 = _mm_load_pd(a);
    __m128d a12 = _mm_load_pd(a+2);
    __m128d a21 = _mm_load_pd(a+4);
    __m128d a22 = _mm_load_pd(a+6);

    // all stores are aligned
    _mm_store_pd(out,_mm_shuffle_pd(a11,a12,0x2));
    _mm_storer_pd(out+2,_mm_shuffle_pd(a11,a22,0x1));
    _mm_store_pd(out+4,_mm_shuffle_pd(a21,a22,0x2));
    _mm_store_pd(out+6,_mm_shuffle_pd(a12,a21,0x2));
    _mm_store_sd(out+8,_mm_load_sd(a+8));
    /*-------------------------------------------------------*/

//    /*-------------------------------------------------------*/
//    // AVX VERSION - NOTE THAT AVX cannot shuffle across 128bit boundaries
//    // so the AVX version requires more instruction although number of load/store
//    // is reduced to 6
//    __m256d row1 = _mm256_load_pd(a);
//    __m256d row2 = _mm256_load_pd(a+4);

//    __m128d a11 = _mm256_extractf128_pd(row1,0x0);
//    __m128d a12 = _mm256_extractf128_pd(row1,0x1);
//    __m128d a21 = _mm256_extractf128_pd(row2,0x0);
//    __m128d a22 = _mm256_extractf128_pd(row2,0x1);

//    row1 = _mm256_insertf128_pd(row1,_mm_shuffle_pd(a11,a12,0x2),0x0);
//    row1 = _mm256_insertf128_pd(row1,_mm_shuffle_pd(a22,a11,0x2),0x1);
//    row2 = _mm256_insertf128_pd(row2,_mm_shuffle_pd(a21,a22,0x2),0x0);
//    row2 = _mm256_insertf128_pd(row2,_mm_shuffle_pd(a12,a21,0x2),0x1);

//    _mm256_store_pd(out,row1);
//    _mm256_store_pd(out+4,row2);
//    _mm_store_sd(out+8,_mm_load_sd(a+8));
//    /*-------------------------------------------------------*/
}
#endif

#ifdef __AVX__
template<>
FASTOR_INLINE void _transpose<double,4,4>(const double * __restrict__ a, double * __restrict__ out) {
    __m256d row1 = _mm256_load_pd(a);
    __m256d row2 = _mm256_load_pd(a+4);
    __m256d row3 = _mm256_load_pd(a+8);
    __m256d row4 = _mm256_load_pd(a+12);
     _MM_TRANSPOSE4_PD(row1, row2, row3, row4);
     _mm256_store_pd(out, row1);
     _mm256_store_pd(out+4, row2);
     _mm256_store_pd(out+8, row3);
     _mm256_store_pd(out+12, row4);
}
#endif

}

#endif // TRANSPOSE_H

