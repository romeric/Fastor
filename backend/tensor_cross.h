#ifndef TENSOR_CROSS_H
#define TENSOR_CROSS_H

#include "commons/commons.h"
#include "extended_intrinsics/extintrin.h"

template<typename T, size_t M, size_t N>
void _crossproduct(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ c);

template<>
FASTOR_INLINE void _crossproduct<double, 2,2>(const double *__restrict__ a, const double *__restrict__ b, double *__restrict__ c) {
    // 25 OPS without HADD / 27 OPS with HADD
    // Load a data - c has to have 9 elements as
    __m128d a_00 = _mm_load_sd(a);
    __m128d a_01 = _mm_load_sd(a+1);
    __m128d a_10 = _mm_load_sd(a+3);
    __m128d a_11 = _mm_load_sd(a+4);
    // Load b data
    __m128d b_00 = _mm_load_sd(b);
    __m128d b_01 = _mm_load_sd(b+1);
    __m128d b_10 = _mm_load_sd(b+3);
    __m128d b_11 = _mm_load_sd(b+4);

    // compute element by element
#ifdef USE_HADD
    // c_22
    __m128d tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_00,a_11,0x0),_mm_shuffle_pd(b_11,b_00,0x0));
    tmp0 = _mm_hadd_pd(tmp0,tmp0);
    __m128d tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_01,a_10,0x0),_mm_shuffle_pd(b_10,b_01,0x0));
    tmp1 = _mm_hadd_pd(tmp1,tmp1);
    __m128d c_22 = _mm_sub_pd(tmp0,tmp1);
#else
    // c_22
    __m128d tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_00,a_11,0x0),_mm_shuffle_pd(b_11,b_00,0x0));
    __m128d tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_01,a_10,0x0),_mm_shuffle_pd(b_10,b_01,0x0));
    __m128d c_22 = _mm_sub_pd(_add_pd(tmp0),_add_pd(tmp1));
#endif

    _mm_store_sd(c+8,c_22);
    // Zero out the rest
    _mm256_store_pd(c,VZEROPD);
    _mm256_store_pd(c+4,VZEROPD);
}

template<>
FASTOR_INLINE void _crossproduct<double,3,3>(const double *__restrict__ a, const double *__restrict__ b, double *__restrict__ c) {
    // 225 OPS without HADD / 243 OPS with HADD
    // Load a data
    __m128d a_00 = _mm_load_sd(a);
    __m128d a_01 = _mm_load_sd(a+1);
    __m128d a_02 = _mm_load_sd(a+2);
    __m128d a_10 = _mm_load_sd(a+3);
    __m128d a_11 = _mm_load_sd(a+4);
    __m128d a_12 = _mm_load_sd(a+5);
    __m128d a_20 = _mm_load_sd(a+6);
    __m128d a_21 = _mm_load_sd(a+7);
    __m128d a_22 = _mm_load_sd(a+8);
    // Load b data
    __m128d b_00 = _mm_load_sd(b);
    __m128d b_01 = _mm_load_sd(b+1);
    __m128d b_02 = _mm_load_sd(b+2);
    __m128d b_10 = _mm_load_sd(b+3);
    __m128d b_11 = _mm_load_sd(b+4);
    __m128d b_12 = _mm_load_sd(b+5);
    __m128d b_20 = _mm_load_sd(b+6);
    __m128d b_21 = _mm_load_sd(b+7);
    __m128d b_22 = _mm_load_sd(b+8);

#ifdef USE_HADD
    // Using hadd
    // compute element by element
    // c_00
    __m128d tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_11,a_22,0x0),_mm_shuffle_pd(b_22,b_11,0x0));
    tmp0 = _mm_hadd_pd(tmp0,tmp0);
    __m128d tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_12,a_21,0x0),_mm_shuffle_pd(b_21,b_12,0x0));
    tmp1 = _mm_hadd_pd(tmp1,tmp1);
    __m128d c_00 = _mm_sub_pd(tmp0,tmp1);
    // c_01
    tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_12,a_20,0x0),_mm_shuffle_pd(b_20,b_12,0x0));
    tmp0 = _mm_hadd_pd(tmp0,tmp0);
    tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_10,a_22,0x0),_mm_shuffle_pd(b_22,b_10,0x0));
    tmp1 = _mm_hadd_pd(tmp1,tmp1);
    __m128d c_01 = _mm_sub_pd(tmp0,tmp1);
    // c_02
    tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_10,a_21,0x0),_mm_shuffle_pd(b_21,b_10,0x0));
    tmp0 = _mm_hadd_pd(tmp0,tmp0);
    tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_11,a_20,0x0),_mm_shuffle_pd(b_20,b_11,0x0));
    tmp1 = _mm_hadd_pd(tmp1,tmp1);
    __m128d c_02 = _mm_sub_pd(tmp0,tmp1);
    // c_10
    tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_02,a_21,0x0),_mm_shuffle_pd(b_21,b_02,0x0));
    tmp0 = _mm_hadd_pd(tmp0,tmp0);
    tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_01,a_22,0x0),_mm_shuffle_pd(b_22,b_01,0x0));
    tmp1 = _mm_hadd_pd(tmp1,tmp1);
    __m128d c_10 = _mm_sub_pd(tmp0,tmp1);
    // c_11
    tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_00,a_22,0x0),_mm_shuffle_pd(b_22,b_00,0x0));
    tmp0 = _mm_hadd_pd(tmp0,tmp0);
    tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_02,a_20,0x0),_mm_shuffle_pd(b_20,b_02,0x0));
    tmp1 = _mm_hadd_pd(tmp1,tmp1);
    __m128d c_11 = _mm_sub_pd(tmp0,tmp1);
    // c_12
    tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_01,a_20,0x0),_mm_shuffle_pd(b_20,b_01,0x0));
    tmp0 = _mm_hadd_pd(tmp0,tmp0);
    tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_00,a_21,0x0),_mm_shuffle_pd(b_21,b_00,0x0));
    tmp1 = _mm_hadd_pd(tmp1,tmp1);
    __m128d c_12 = _mm_sub_pd(tmp0,tmp1);
    // c_20
    tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_01,a_12,0x0),_mm_shuffle_pd(b_12,b_01,0x0));
    tmp0 = _mm_hadd_pd(tmp0,tmp0);
    tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_02,a_11,0x0),_mm_shuffle_pd(b_11,b_02,0x0));
    tmp1 = _mm_hadd_pd(tmp1,tmp1);
    __m128d c_20 = _mm_sub_pd(tmp0,tmp1);
    // c_21
    tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_02,a_10,0x0),_mm_shuffle_pd(b_10,b_02,0x0));
    tmp0 = _mm_hadd_pd(tmp0,tmp0);
    tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_00,a_12,0x0),_mm_shuffle_pd(b_12,b_00,0x0));
    tmp1 = _mm_hadd_pd(tmp1,tmp1);
    __m128d c_21 = _mm_sub_pd(tmp0,tmp1);
    // c_22
    tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_00,a_11,0x0),_mm_shuffle_pd(b_11,b_00,0x0));
    tmp0 = _mm_hadd_pd(tmp0,tmp0);
    tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_01,a_10,0x0),_mm_shuffle_pd(b_10,b_01,0x0));
    tmp1 = _mm_hadd_pd(tmp1,tmp1);
    __m128d c_22 = _mm_sub_pd(tmp0,tmp1);
#else
    // compute element by element
    // c_00
    __m128d tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_11,a_22,0x0),_mm_shuffle_pd(b_22,b_11,0x0));
    __m128d tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_12,a_21,0x0),_mm_shuffle_pd(b_21,b_12,0x0));
    __m128d c_00 = _mm_sub_pd(_add_pd(tmp0),_add_pd(tmp1));
    // c_01
    tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_12,a_20,0x0),_mm_shuffle_pd(b_20,b_12,0x0));
    tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_10,a_22,0x0),_mm_shuffle_pd(b_22,b_10,0x0));
    __m128d c_01 = _mm_sub_pd(_add_pd(tmp0),_add_pd(tmp1));
    // c_02
    tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_10,a_21,0x0),_mm_shuffle_pd(b_21,b_10,0x0));
    tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_11,a_20,0x0),_mm_shuffle_pd(b_20,b_11,0x0));
    __m128d c_02 = _mm_sub_pd(_add_pd(tmp0),_add_pd(tmp1));;
    // c_10
    tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_02,a_21,0x0),_mm_shuffle_pd(b_21,b_02,0x0));
    tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_01,a_22,0x0),_mm_shuffle_pd(b_22,b_01,0x0));
    __m128d c_10 = _mm_sub_pd(_add_pd(tmp0),_add_pd(tmp1));
    // c_11
    tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_00,a_22,0x0),_mm_shuffle_pd(b_22,b_00,0x0));
    tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_02,a_20,0x0),_mm_shuffle_pd(b_20,b_02,0x0));
    __m128d c_11 = _mm_sub_pd(_add_pd(tmp0),_add_pd(tmp1));
    // c_12
    tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_01,a_20,0x0),_mm_shuffle_pd(b_20,b_01,0x0));
    tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_00,a_21,0x0),_mm_shuffle_pd(b_21,b_00,0x0));
    __m128d c_12 = _mm_sub_pd(_add_pd(tmp0),_add_pd(tmp1));
    // c_20
    tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_01,a_12,0x0),_mm_shuffle_pd(b_12,b_01,0x0));
    tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_02,a_11,0x0),_mm_shuffle_pd(b_11,b_02,0x0));
    __m128d c_20 = _mm_sub_pd(_add_pd(tmp0),_add_pd(tmp1));
    // c_21
    tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_02,a_10,0x0),_mm_shuffle_pd(b_10,b_02,0x0));
    tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_00,a_12,0x0),_mm_shuffle_pd(b_12,b_00,0x0));
    __m128d c_21 = _mm_sub_pd(_add_pd(tmp0),_add_pd(tmp1));
    // c_22
    tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_00,a_11,0x0),_mm_shuffle_pd(b_11,b_00,0x0));
    tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_01,a_10,0x0),_mm_shuffle_pd(b_10,b_01,0x0));
    __m128d c_22 = _mm_sub_pd(_add_pd(tmp0),_add_pd(tmp1));
#endif
    // store
    _mm_store_sd(c,c_00);
    _mm_store_sd(c+1,c_01);
    _mm_store_sd(c+2,c_02);
    _mm_store_sd(c+3,c_10);
    _mm_store_sd(c+4,c_11);
    _mm_store_sd(c+5,c_12);
    _mm_store_sd(c+6,c_20);
    _mm_store_sd(c+7,c_21);
    _mm_store_sd(c+8,c_22);
}


template<>
void _crossproduct<float,2,2>(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c) {
    // 15 OPS + 4 extra shuffle to make aligned loading = 19 OPS
    __m128 a0 = _mm_load_ps(a);
    __m128 tmp0 = _mm_load_ps(a+4);
    __m128 a1 = _mm_shuffle_ps(tmp0,a0,_MM_SHUFFLE(0,3,1,0));
    a1 = _mm_shuffle_ps(a1,a1,_MM_SHUFFLE(3,1,0,2));
    // Load b data
    __m128 b0 = _mm_load_ps(b);
    __m128 tmp1 = _mm_load_ps(b+4);
    __m128 b1 = _mm_shuffle_ps(tmp1,b0,_MM_SHUFFLE(0,3,1,0));
    b1 = _mm_shuffle_ps(b1,b1,_MM_SHUFFLE(3,1,0,2));

    // c_22
    __m128 c_22 = _addsub_ps(_mm_mul_ps(_mm_shuffle_ps(a0,a1,_MM_SHUFFLE(1,0,0,1)),
                                        _mm_shuffle_ps(b1,b0,_MM_SHUFFLE(0,1,1,0))));

    _mm_store_ss(c+8,c_22);
    // Zero the rest
    _mm256_store_ps(c,VZEROPS);
}

template<>
void _crossproduct<float,3,3>(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c) {
    // 135 OPS + 6 extra shuffle to make aligned loading = 141 OPS
    // Don't unalign load - Bad performance on IVY

    __m128 a0 = _mm_load_ps(a);
    __m128 tmp0 = _mm_load_ps(a+4);
    __m128 a_end = _mm_load_ss(a+8);
    __m128 a1 = _mm_shuffle_ps(tmp0,a0,_MM_SHUFFLE(0,3,1,0));
    a1 = _mm_shuffle_ps(a1,a1,_MM_SHUFFLE(3,1,0,2));
    __m128 a2 = _mm_shuffle_ps(tmp0,a_end,_MM_SHUFFLE(1,0,3,2));
    // Load b data
    __m128 b0 = _mm_load_ps(b);
    __m128 tmp1 = _mm_load_ps(b+4);
    __m128 b_end = _mm_load_ss(b+8);
    __m128 b1 = _mm_shuffle_ps(tmp1,b0,_MM_SHUFFLE(0,3,1,0));
    b1 = _mm_shuffle_ps(b1,b1,_MM_SHUFFLE(3,1,0,2));
    __m128 b2 = _mm_shuffle_ps(tmp1,b_end,_MM_SHUFFLE(1,0,3,2));

    // c_00
    __m128 c_00 = _addsub_ps(_mm_mul_ps(_mm_shuffle_ps(a1,a2,_MM_SHUFFLE(2,1,1,2)),
                                        _mm_shuffle_ps(b2,b1,_MM_SHUFFLE(1,2,2,1))));
    // c_01
    __m128 c_01 = _addsub_ps(_mm_mul_ps(_mm_shuffle_ps(a1,a2,_MM_SHUFFLE(0,2,2,0)),
                                        _mm_shuffle_ps(b2,b1,_MM_SHUFFLE(2,0,0,2))));
    // c_01
    __m128 c_02 = _addsub_ps(_mm_mul_ps(_mm_shuffle_ps(a1,a2,_MM_SHUFFLE(1,0,0,1)),
                                        _mm_shuffle_ps(b2,b1,_MM_SHUFFLE(0,1,1,0))));
    // c_10
    __m128 c_10 = _addsub_ps(_mm_mul_ps(_mm_shuffle_ps(a0,a2,_MM_SHUFFLE(1,2,2,1)),
                                        _mm_shuffle_ps(b2,b0,_MM_SHUFFLE(2,1,1,2))));
    // c_11
    __m128 c_11 = _addsub_ps(_mm_mul_ps(_mm_shuffle_ps(a0,a2,_MM_SHUFFLE(2,0,0,2)),
                                        _mm_shuffle_ps(b2,b0,_MM_SHUFFLE(0,2,2,0))));
    // c_12
    __m128 c_12 = _addsub_ps(_mm_mul_ps(_mm_shuffle_ps(a0,a2,_MM_SHUFFLE(0,1,1,0)),
                                        _mm_shuffle_ps(b2,b0,_MM_SHUFFLE(1,0,0,1))));
    // c_20
    __m128 c_20 = _addsub_ps(_mm_mul_ps(_mm_shuffle_ps(a0,a1,_MM_SHUFFLE(2,1,1,2)),
                                        _mm_shuffle_ps(b1,b0,_MM_SHUFFLE(1,2,2,1))));
    // c_21
    __m128 c_21 = _addsub_ps(_mm_mul_ps(_mm_shuffle_ps(a0,a1,_MM_SHUFFLE(0,2,2,0)),
                                        _mm_shuffle_ps(b1,b0,_MM_SHUFFLE(2,0,0,2))));
    // c_22
    __m128 c_22 = _addsub_ps(_mm_mul_ps(_mm_shuffle_ps(a0,a1,_MM_SHUFFLE(1,0,0,1)),
                                        _mm_shuffle_ps(b1,b0,_MM_SHUFFLE(0,1,1,0))));

    _mm_store_ss(c,c_00);
    _mm_store_ss(c+1,c_01);
    _mm_store_ss(c+2,c_02);
    _mm_store_ss(c+3,c_10);
    _mm_store_ss(c+4,c_11);
    _mm_store_ss(c+5,c_12);
    _mm_store_ss(c+6,c_20);
    _mm_store_ss(c+7,c_21);
    _mm_store_ss(c+8,c_22);
}


#endif // TENSOR_CROSS_H

