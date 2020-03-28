#ifndef TRANSPOSE_H
#define TRANSPOSE_H


#include "Fastor/commons/commons.h"
#include "Fastor/extended_intrinsics/extintrin.h"

namespace Fastor {

template<typename T, size_t M, size_t N>
FASTOR_INLINE void _transpose(const T * FASTOR_RESTRICT a, T * FASTOR_RESTRICT out) {
    for (size_t j=0; j<N; ++j)
        for (size_t i=0; i< M; ++i)
            out[j*M+i] = a[i*N+j];
}

#ifdef FASTOR_SSE4_2_IMPL
template<>
FASTOR_INLINE void _transpose<float,2,2>(const float * FASTOR_RESTRICT a, float * FASTOR_RESTRICT out) {
    __m128 a_reg = _mm_load_ps(a);
    _mm_store_ps(out,_mm_shuffle_ps(a_reg,a_reg,_MM_SHUFFLE(3,1,2,0)));
}

template<>
FASTOR_INLINE void _transpose<float,3,3>(const float * FASTOR_RESTRICT a, float * FASTOR_RESTRICT out) {
    __m128 row0 = _mm_load_ps(a);
    __m128 row1 = _mm_loadu_ps(a+3);
    __m128 row2 = _mm_loadu_ps(a+6);

    __m128 T0 = _mm_unpacklo_ps(row0,row1);
    __m128 T1 = _mm_unpackhi_ps(row0,row1);

    row0 = _mm_movelh_ps ( T0,row2 );
    row1 = _mm_shuffle_ps( T0,row2, _MM_SHUFFLE(3,1,3,2) );
    row2 = _mm_shuffle_ps( T1,row2, _MM_SHUFFLE(3,2,1,0) );

    _mm_store_ps(out,row0);
    _mm_storeu_ps(out+3,row1);
    _mm_storeu_ps(out+6,row2);
}

template<>
FASTOR_INLINE void _transpose<float,4,4>(const float * FASTOR_RESTRICT a, float * FASTOR_RESTRICT out) {
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

#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE void _transpose<float,8,8>(const float * FASTOR_RESTRICT a, float * FASTOR_RESTRICT out) {
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
FASTOR_INLINE void _transpose<double,2,2>(const double* FASTOR_RESTRICT a, double* FASTOR_RESTRICT out) {
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

#ifdef FASTOR_SSE4_2_IMPL
template<>
FASTOR_INLINE void _transpose<double,3,3>(const double* FASTOR_RESTRICT a, double* FASTOR_RESTRICT out) {
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

#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE void _transpose<double,4,4>(const double * FASTOR_RESTRICT a, double * FASTOR_RESTRICT out) {
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

