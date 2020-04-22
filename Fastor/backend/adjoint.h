#ifndef ADJOINT_H
#define ADJOINT_H


#include "Fastor/simd_vector/extintrin.h"

//! The implementation for adjoint and cofactor are the same, just the store
//! operations are swapped

namespace Fastor {


template<typename T, size_t M, size_t N>
FASTOR_HINT_INLINE void _adjoint(const T * FASTOR_RESTRICT a, T * FASTOR_RESTRICT out) {
    if (M==2 && N==2) {
        out[0] = a[3];
        out[1] = -a[1];
        out[2] = -a[2];
        out[3] = a[0];
    }
    else if (M==3 && N==3) {
        out[0] =  (a[4]*a[8] - a[7]*a[5]);
        out[1] = -(a[1]*a[8] - a[7]*a[2]);
        out[2] =  (a[1]*a[5] - a[4]*a[2]);
        out[3] = -(a[3]*a[8] - a[6]*a[5]);
        out[4] =  (a[0]*a[8] - a[6]*a[2]);
        out[5] = -(a[0]*a[5] - a[3]*a[2]);
        out[6] =  (a[3]*a[7] - a[6]*a[4]);
        out[7] = -(a[0]*a[7] - a[6]*a[1]);
        out[8] =  (a[0]*a[4] - a[3]*a[1]);
    }
    else {
        assert(false && "METHOD NOT YET IMPLEMENTED");
    }
}

#ifdef FASTOR_SSE4_2_IMPL
template<>
FASTOR_HINT_INLINE void _adjoint<float,2,2>(const float * FASTOR_RESTRICT a, float * FASTOR_RESTRICT out) {
    // 4 OPS
    __m128 a_reg = _mm_load_ps(a);
    __m128 a0 = _mm_shuffle_ps(a_reg,a_reg,_MM_SHUFFLE(0,3,2,1));
    __m128 a1 = _mm_shuffle_ps(a0,_mm_neg_ps(a0),_MM_SHUFFLE(0,1,2,3));
    _mm_store_ps(out,_mm_shuffle_ps(a1,a1,_MM_SHUFFLE(0,2,3,1)));
}

template<>
FASTOR_HINT_INLINE void _adjoint<float,3,3>(const float * FASTOR_RESTRICT a, float * FASTOR_RESTRICT out) {
    // 110 OPS
    __m128 a_low = _mm_load_ps(a);
    __m128 a_high = _mm_load_ps(a+4);
    __m128 a_end = _mm_load_ss(a+8);

    // c_00
    __m128 a1 = _mm_shuffle_ps(a_end,a_high,_MM_SHUFFLE(3,3,0,0));
    a1 = _mm_shuffle_ps(a1,a1,_MM_SHUFFLE(2,1,2,1));
    __m128 c_00 = _mm_shuffle_ps(a_high,a1,_MM_SHUFFLE(2,3,0,1));
    c_00 = _mulsub_ps(c_00);
    // c_01
    __m128 a2 = _mm_shuffle_ps(a_low,a_high,_MM_SHUFFLE(1,1,3,3));
    a2 = _mm_shuffle_ps(a2,a2,_MM_SHUFFLE(2,0,2,0));
    __m128 a6 = _mm_shuffle_ps(a_high,a_end,_MM_SHUFFLE(0,0,2,2));
    a6 = _mm_shuffle_ps(a6,a6,_MM_SHUFFLE(2,0,2,0));
    __m128 c_01 = _mm_shuffle_ps(a2,a6,_MM_SHUFFLE(2,3,1,0));
    c_01 = _mulsub_ps(c_01);
    // c_02
    __m128 a3 = _mm_shuffle_ps(a_low,a_high,_MM_SHUFFLE(0,0,3,3));
    a3 = _mm_shuffle_ps(a3,a3,_MM_SHUFFLE(0,2,0,2));
    __m128 c_02 = _mm_shuffle_ps(a3,a_high,_MM_SHUFFLE(3,2,1,0));
    c_02 = _mulsub_ps(c_02);
    // c_10
    __m128 a4 = _mm_shuffle_ps(a_low,a_low,_MM_SHUFFLE(2,1,2,1));
    __m128 c_10 = _mm_shuffle_ps(a4,a1,_MM_SHUFFLE(3,2,1,0));
    c_10 = _mulsub_ps(c_10);
    // c_11
    __m128 a5 = _mm_shuffle_ps(a_low,a_low,_MM_SHUFFLE(0,2,0,2));
    __m128 c_11 = _mm_shuffle_ps(a5,a6,_MM_SHUFFLE(3,2,1,0));
    c_11 = _mulsub_ps(c_11);
    // c_12
    __m128 a7 = _mm_shuffle_ps(a_low,a_low,_MM_SHUFFLE(0,1,0,1));
    __m128 c_12 = _mm_shuffle_ps(a7,a_high,_MM_SHUFFLE(2,3,0,1));
    c_12 = _mulsub_ps(c_12);
    // c_20
    __m128 c_20 = _mm_shuffle_ps(a_low,a_high,_MM_SHUFFLE(1,0,1,2));
    c_20 = _mulsub_ps(c_20);
    // c_21
    __m128 c_21 = _mm_shuffle_ps(a2,a5,_MM_SHUFFLE(2,3,0,1));
    c_21 = _mulsub_ps(c_21);
    // c_22
    __m128 c_22 = _mm_shuffle_ps(a_low,a3,_MM_SHUFFLE(2,3,0,1));
    c_22 = _mulsub_ps(c_22);

    _mm_store_ss(out,c_00);
    _mm_store_ss(out+1,c_10);
    _mm_store_ss(out+2,c_20);
    _mm_store_ss(out+3,c_01);
    _mm_store_ss(out+4,c_11);
    _mm_store_ss(out+5,c_21);
    _mm_store_ss(out+6,c_02);
    _mm_store_ss(out+7,c_12);
    _mm_store_ss(out+8,c_22);
}

template<>
FASTOR_HINT_INLINE void _adjoint<double,2,2>(const double * FASTOR_RESTRICT a, double * FASTOR_RESTRICT out) {
    // 2 OPS
    _mm_store_sd(out+3,_mm_load_sd(a));
    _mm_store_sd(out,_mm_load_sd(a+3));
    _mm_store_sd(out+1,_mm_neg_pd(_mm_load_sd(a+1)));
    _mm_store_sd(out+2,_mm_neg_pd(_mm_load_sd(a+2)));
}

template<>
FASTOR_HINT_INLINE void _adjoint<double,3,3>(const double * FASTOR_RESTRICT a, double * FASTOR_RESTRICT out) {
    // 96 OPS
    __m128d a00 = _mm_load_sd(a);
    __m128d a01 = _mm_load_sd(a+1);
    __m128d a02 = _mm_load_sd(a+2);
    __m128d a10 = _mm_load_sd(a+3);
    __m128d a11 = _mm_load_sd(a+4);
    __m128d a12 = _mm_load_sd(a+5);
    __m128d a20 = _mm_load_sd(a+6);
    __m128d a21 = _mm_load_sd(a+7);
    __m128d a22 = _mm_load_sd(a+8);

    // c_00
    __m128d a0 = _mm_shuffle_pd(a11,a12,0x0);
    __m128d a1 = _mm_shuffle_pd(a22,a21,0x0);
    __m128d c_00 = _hsub_pd(_mm_mul_pd(a0,a1));
    // c_01
    __m128d a2 = _mm_shuffle_pd(a12,a10,0x0);
    __m128d a3 = _mm_shuffle_pd(a20,a22,0x0);
    __m128d c_01 = _hsub_pd(_mm_mul_pd(a2,a3));
    // c_02
    __m128d a4 = _mm_shuffle_pd(a10,a11,0x0);
    __m128d a5 = _mm_shuffle_pd(a21,a20,0x0);
    __m128d c_02 = _hsub_pd(_mm_mul_pd(a4,a5));
    // c_10
    __m128d a6 = _mm_shuffle_pd(a01,a02,0x0);
    __m128d c_10 = _mm_neg_pd(_hsub_pd(_mm_mul_pd(a1,a6)));
    // c_11
    __m128d a7 = _mm_shuffle_pd(a02,a00,0x0);
    __m128d c_11 = _mm_neg_pd(_hsub_pd(_mm_mul_pd(a3,a7)));
    // c_12
    __m128d a8 = _mm_shuffle_pd(a00,a01,0x0);
    __m128d c_12 = _mm_neg_pd(_hsub_pd(_mm_mul_pd(a8,a5)));
    // c_20
    __m128d c_20 = _hsub_pd(_mm_mul_pd(a6,_mm_shuffle_pd(a0,a0,0x1)));
    // c_21
    __m128d c_21 = _hsub_pd(_mm_mul_pd(a7,_mm_shuffle_pd(a2,a2,0x1)));
    // c_21
    __m128d c_22 = _hsub_pd(_mm_mul_pd(a8,_mm_shuffle_pd(a4,a4,0x1)));

    _mm_store_sd(out,c_00);
    _mm_store_sd(out+1,c_10);
    _mm_store_sd(out+2,c_20);
    _mm_store_sd(out+3,c_01);
    _mm_store_sd(out+4,c_11);
    _mm_store_sd(out+5,c_21);
    _mm_store_sd(out+6,c_02);
    _mm_store_sd(out+7,c_12);
    _mm_store_sd(out+8,c_22);
}

#endif

}

#endif // ADJOINT_H

