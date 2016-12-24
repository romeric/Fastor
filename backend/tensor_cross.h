#ifndef TENSOR_CROSS_H
#define TENSOR_CROSS_H

#include "commons/commons.h"
#include "extended_intrinsics/extintrin.h"

template<typename T, size_t M, size_t K, size_t N>
void _crossproduct(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ c);

#ifdef __SSE4_2__
template<>
FASTOR_INLINE void _crossproduct<double,2,2,2>(const double *__restrict__ a, const double *__restrict__ b, double *__restrict__ c) {
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
FASTOR_INLINE void _crossproduct<double,3,3,3>(const double *__restrict__ a, const double *__restrict__ b, double *__restrict__ c) {
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
void _crossproduct<float,2,2,2>(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c) {
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
void _crossproduct<float,3,3,3>(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c) {
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

//-----------------------------------------------------------------------------------------------
// for plane strain problems
template<typename T, int ProblemType>
void _crossproduct(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ c);

template<>
FASTOR_INLINE void _crossproduct<double,PlaneStrain>(const double *__restrict__ a, const double *__restrict__ b, double *__restrict__ c) {
    // For plane strain problems a and b need to be 3D with last element a[8]=b[8]=1
    // This is a cross product implementation not a cofactor so the result needs to multiplied by 0.5 explicitly
    // if the cofactor is desired
    // Note that in ultimate case you might not need the last row/column of the output matrix hence computing c22
    // and storing it is un-necessary, hence the (c) array should really be 2x2=4 in length
    // Load a data
    __m128d a_00 = _mm_load_sd(a);
    __m128d a_01 = _mm_load_sd(a+1);
    __m128d a_10 = _mm_load_sd(a+3);
    __m128d a_11 = _mm_load_sd(a+4);
    __m128d a_22 = _mm_load_sd(a+8);
    // Load b data
    __m128d b_00 = _mm_load_sd(b);
    __m128d b_01 = _mm_load_sd(b+1);
    __m128d b_10 = _mm_load_sd(b+3);
    __m128d b_11 = _mm_load_sd(b+4);
    __m128d b_22 = _mm_load_sd(b+8);

    // c_00
    __m128d tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_11,a_22,0x0),_mm_shuffle_pd(b_22,b_11,0x0));
    __m128d c_00 = _add_pd(tmp0);
    // c_01
    __m128d tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_10,a_22,0x0),_mm_shuffle_pd(b_22,b_10,0x0));
    __m128d c_01 = _mm_neg_pd(_add_pd(tmp1));
    // c_10
    tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_01,a_22,0x0),_mm_shuffle_pd(b_22,b_01,0x0));
    __m128d c_10 = _mm_neg_pd(_add_pd(tmp1));
    // c_11
    tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_00,a_22,0x0),_mm_shuffle_pd(b_22,b_00,0x0));
    __m128d c_11 = _add_pd(tmp0);
//    // c_22
//    tmp0 = _mm_mul_pd(_mm_shuffle_pd(a_00,a_11,0x0),_mm_shuffle_pd(b_11,b_00,0x0));
//    tmp1 = _mm_mul_pd(_mm_shuffle_pd(a_01,a_10,0x0),_mm_shuffle_pd(b_10,b_01,0x0));
//    __m128d c_22 = _mm_sub_pd(_add_pd(tmp0),_add_pd(tmp1));

    // zero first
    _mm256_store_pd(c,VZEROPD);
//    _mm256_store_pd(c+4,VZEROPD);
    // store
    _mm_store_sd(c,c_00);
    _mm_store_sd(c+1,c_01);
    _mm_store_sd(c+3,c_10);
    _mm_store_sd(c+4,c_11);

    _mm_store_sd(c,c_00);
    _mm_store_sd(c+1,c_01);
    _mm_store_sd(c+2,c_10);
    _mm_store_sd(c+3,c_11);
//    _mm_store_sd(c+8,c_22);
}


template<>
void _crossproduct<float,PlaneStrain>(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c) {
    // For plane strain problems a and b need to be 3D with last element a[8]=b[8]=1
    // This is a cross product implementation not a cofactor so the result needs to multiplied by 0.5 explicitly
    // if the cofactor is desired
    // Note that in ultimate case you might not need the last row/column of the output matrix hence computing c22
    // and storing it is un-necessary, hence the (c) array should really be 2x2=4 in length

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
    // c_10
    __m128 c_10 = _addsub_ps(_mm_mul_ps(_mm_shuffle_ps(a0,a2,_MM_SHUFFLE(1,2,2,1)),
                                        _mm_shuffle_ps(b2,b0,_MM_SHUFFLE(2,1,1,2))));
    // c_11
    __m128 c_11 = _addsub_ps(_mm_mul_ps(_mm_shuffle_ps(a0,a2,_MM_SHUFFLE(2,0,0,2)),
                                        _mm_shuffle_ps(b2,b0,_MM_SHUFFLE(0,2,2,0))));
    // c_22
    __m128 c_22 = _addsub_ps(_mm_mul_ps(_mm_shuffle_ps(a0,a1,_MM_SHUFFLE(1,0,0,1)),
                                        _mm_shuffle_ps(b1,b0,_MM_SHUFFLE(0,1,1,0))));

    // zero first
    _mm256_store_ps(c,VZEROPS);
    // store
    _mm_store_ss(c,c_00);
    _mm_store_ss(c+1,c_01);
    _mm_store_ss(c+3,c_10);
    _mm_store_ss(c+4,c_11);
    _mm_store_ss(c+8,c_22);
}

#endif
//------------------------------------------------------------------------------------


// vectors and 2nd order tensors
template<>
inline void _crossproduct<float,3,1,3>(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c) {
    // vector-tensor cross product - regitster based
    float A1 = a[0];
    float A2 = a[1];
    float A3 = a[2];

    float B1_1 = b[0];
    float B1_2 = b[1];
    float B1_3 = b[2];
    float B2_1 = b[3];
    float B2_2 = b[4];
    float B2_3 = b[5];
    float B3_1 = b[6];
    float B3_2 = b[7];
    float B3_3 = b[8];

    c[0] = A2*B3_1 - A3*B2_1;
    c[1] = A2*B3_2 - A3*B2_2;
    c[2] = A2*B3_3 - A3*B2_3;
    c[3] = A3*B1_1 - A1*B3_1;
    c[4] = A3*B1_2 - A1*B3_2;
    c[5] = A3*B1_3 - A1*B3_3;
    c[6] = A1*B2_1 - A2*B1_1;
    c[7] = A1*B2_2 - A2*B1_2;
    c[8] = A1*B2_3 - A2*B1_3;

//    [ A2*B3_1 - A3*B2_1, A2*B3_2 - A3*B2_2, A2*B3_3 - A3*B2_3]
//    [ A3*B1_1 - A1*B3_1, A3*B1_2 - A1*B3_2, A3*B1_3 - A1*B3_3]
//    [ A1*B2_1 - A2*B1_1, A1*B2_2 - A2*B1_2, A1*B2_3 - A2*B1_3]
}

template<>
inline void _crossproduct<float,2,1,2>(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c) {
    // vector-tensor cross product - regitster based
    float A1 = a[0];
    float A2 = a[1];

    float B1_1 = b[0];
    float B1_2 = b[1];
    float B2_1 = b[3];
    float B2_2 = b[4];

    c[0] = 0;
    c[1] = 0;
    c[2] = 0;
    c[3] = 0;
    c[4] = 0;
    c[5] = 0;
    c[6] = A1*B2_1 - A2*B1_1;
    c[7] = A1*B2_2 - A2*B1_2;
    c[8] = 0;
}

template<>
inline void _crossproduct<double,3,1,3>(const double *__restrict__ a, const double *__restrict__ b, double *__restrict__ c) {
    // vector-tensor cross product - regitster based
    double A1 = a[0];
    double A2 = a[1];
    double A3 = a[2];

    double B1_1 = b[0];
    double B1_2 = b[1];
    double B1_3 = b[2];
    double B2_1 = b[3];
    double B2_2 = b[4];
    double B2_3 = b[5];
    double B3_1 = b[6];
    double B3_2 = b[7];
    double B3_3 = b[8];

    c[0] = A2*B3_1 - A3*B2_1;
    c[1] = A2*B3_2 - A3*B2_2;
    c[2] = A2*B3_3 - A3*B2_3;
    c[3] = A3*B1_1 - A1*B3_1;
    c[4] = A3*B1_2 - A1*B3_2;
    c[5] = A3*B1_3 - A1*B3_3;
    c[6] = A1*B2_1 - A2*B1_1;
    c[7] = A1*B2_2 - A2*B1_2;
    c[8] = A1*B2_3 - A2*B1_3;

//    [ A2*B3_1 - A3*B2_1, A2*B3_2 - A3*B2_2, A2*B3_3 - A3*B2_3]
//    [ A3*B1_1 - A1*B3_1, A3*B1_2 - A1*B3_2, A3*B1_3 - A1*B3_3]
//    [ A1*B2_1 - A2*B1_1, A1*B2_2 - A2*B1_2, A1*B2_3 - A2*B1_3]
}

template<>
inline void _crossproduct<double,2,1,2>(const double *__restrict__ a, const double *__restrict__ b, double *__restrict__ c) {
    // vector-tensor cross product - regitster based
    double A1 = a[0];
    double A2 = a[1];

    double B1_1 = b[0];
    double B1_2 = b[1];
    double B2_1 = b[3];
    double B2_2 = b[4];

    c[0] = 0;
    c[1] = 0;
    c[2] = 0;
    c[3] = 0;
    c[4] = 0;
    c[5] = 0;
    c[6] = A1*B2_1 - A2*B1_1;
    c[7] = A1*B2_2 - A2*B1_2;
    c[8] = 0;
}

//
template<>
void _crossproduct<float,3,3,1>(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c) {
    // tensor-vector cross product - regitster based
    float B1 = b[0];
    float B2 = b[1];
    float B3 = b[2];

    float A1_1 = a[0];
    float A1_2 = a[1];
    float A1_3 = a[2];
    float A2_1 = a[3];
    float A2_2 = a[4];
    float A2_3 = a[5];
    float A3_1 = a[6];
    float A3_2 = a[7];
    float A3_3 = a[8];

    c[0] = A1_2*B3 - A1_3*B2;
    c[1] = A1_3*B1 - A1_1*B3;
    c[2] = A1_1*B2 - A1_2*B1;
    c[3] = A2_2*B3 - A2_3*B2;
    c[4] = A2_3*B1 - A2_1*B3;
    c[5] = A2_1*B2 - A2_2*B1;
    c[6] = A3_2*B3 - A3_3*B2;
    c[7] = A3_3*B1 - A3_1*B3;
    c[8] = A3_1*B2 - A3_2*B1;

//    [ A1_2*B3 - A1_3*B2, A1_3*B1 - A1_1*B3, A1_1*B2 - A1_2*B1]
//    [ A2_2*B3 - A2_3*B2, A2_3*B1 - A2_1*B3, A2_1*B2 - A2_2*B1]
//    [ A3_2*B3 - A3_3*B2, A3_3*B1 - A3_1*B3, A3_1*B2 - A3_2*B1]
}


template<>
void _crossproduct<float,2,2,1>(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c) {
    // tensor-vector cross product - regitster based
    float B1 = b[0];
    float B2 = b[1];

    float A1_1 = a[0];
    float A1_2 = a[1];
    float A2_1 = a[3];
    float A2_2 = a[4];

    c[0] = 0;
    c[1] = 0;
    c[2] = A1_1*B2 - A1_2*B1;
    c[3] = 0;
    c[4] = 0;
    c[5] = A2_1*B2 - A2_2*B1;
    c[6] = 0;
    c[7] = 0;
    c[8] = 0;
}

template<>
void _crossproduct<double,3,3,1>(const double *__restrict__ a, const double *__restrict__ b, double *__restrict__ c) {
    // tensor-vector cross product - regitster based
    double B1 = b[0];
    double B2 = b[1];
    double B3 = b[2];

    double A1_1 = a[0];
    double A1_2 = a[1];
    double A1_3 = a[2];
    double A2_1 = a[3];
    double A2_2 = a[4];
    double A2_3 = a[5];
    double A3_1 = a[6];
    double A3_2 = a[7];
    double A3_3 = a[8];

    c[0] = A1_2*B3 - A1_3*B2;
    c[1] = A1_3*B1 - A1_1*B3;
    c[2] = A1_1*B2 - A1_2*B1;
    c[3] = A2_2*B3 - A2_3*B2;
    c[4] = A2_3*B1 - A2_1*B3;
    c[5] = A2_1*B2 - A2_2*B1;
    c[6] = A3_2*B3 - A3_3*B2;
    c[7] = A3_3*B1 - A3_1*B3;
    c[8] = A3_1*B2 - A3_2*B1;

//    [ A1_2*B3 - A1_3*B2, A1_3*B1 - A1_1*B3, A1_1*B2 - A1_2*B1]
//    [ A2_2*B3 - A2_3*B2, A2_3*B1 - A2_1*B3, A2_1*B2 - A2_2*B1]
//    [ A3_2*B3 - A3_3*B2, A3_3*B1 - A3_1*B3, A3_1*B2 - A3_2*B1]
}

template<>
void _crossproduct<double,2,2,1>(const double *__restrict__ a, const double *__restrict__ b, double *__restrict__ c) {
    // tensor-vector cross product - regitster based
    double B1 = b[0];
    double B2 = b[1];

    double A1_1 = a[0];
    double A1_2 = a[1];
    double A2_1 = a[3];
    double A2_2 = a[4];

    c[0] = 0;
    c[1] = 0;
    c[2] = A1_1*B2 - A1_2*B1;
    c[3] = 0;
    c[4] = 0;
    c[5] = A2_1*B2 - A2_2*B1;
    c[6] = 0;
    c[7] = 0;
    c[8] = 0;
}







////////
//(AxB)_{PiIQ} = E_{ijk}E_{IJK}A_{PjJ}B_{kKQ}
// tensor cross product of 3rd order tensors
template<typename T, size_t I, size_t J, size_t K, size_t L, size_t M, size_t N>
void _crossproduct(const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C) {

    T A000 = A[0];
    T A001 = A[1];
    T A002 = A[2];
    T A010 = A[3];
    T A011 = A[4];
    T A012 = A[5];
    T A020 = A[6];
    T A021 = A[7];
    T A022 = A[8];
    T A100 = A[9];
    T A101 = A[10];
    T A102 = A[11];
    T A110 = A[12];
    T A111 = A[13];
    T A112 = A[14];
    T A120 = A[15];
    T A121 = A[16];
    T A122 = A[17];
    T A200 = A[18];
    T A201 = A[19];
    T A202 = A[20];
    T A210 = A[21];
    T A211 = A[22];
    T A212 = A[23];
    T A220 = A[24];
    T A221 = A[25];
    T A222 = A[26];


    T B000 = B[0];
    T B001 = B[1];
    T B002 = B[2];
    T B010 = B[3];
    T B011 = B[4];
    T B012 = B[5];
    T B020 = B[6];
    T B021 = B[7];
    T B022 = B[8];
    T B100 = B[9];
    T B101 = B[10];
    T B102 = B[11];
    T B110 = B[12];
    T B111 = B[13];
    T B112 = B[14];
    T B120 = B[15];
    T B121 = B[16];
    T B122 = B[17];
    T B200 = B[18];
    T B201 = B[19];
    T B202 = B[20];
    T B210 = B[21];
    T B211 = B[22];
    T B212 = B[23];
    T B220 = B[24];
    T B221 = B[25];
    T B222 = B[26];


    C[0] = A101*B022 - A102*B012 + A202*B011 - A201*B021;
    C[1] = A111*B022 - A112*B012 + A212*B011 - A211*B021;
    C[2] = A121*B022 - A122*B012 + A222*B011 - A221*B021;
    C[3] = A102*B002 - A100*B022 - A202*B001 + A200*B021;
    C[4] = A112*B002 - A110*B022 - A212*B001 + A210*B021;
    C[5] = A122*B002 - A120*B022 - A222*B001 + A220*B021;
    C[6] = A100*B012 - A101*B002 + A201*B001 - A200*B011;
    C[7] = A110*B012 - A111*B002 + A211*B001 - A210*B011;
    C[8] = A120*B012 - A121*B002 + A221*B001 - A220*B011;
    C[9] = A002*B012 - A001*B022 - A202*B010 + A201*B020;
    C[10] = A012*B012 - A011*B022 - A212*B010 + A211*B020;
    C[11] = A022*B012 - A021*B022 - A222*B010 + A221*B020;
    C[12] = A000*B022 - A002*B002 + A202*B000 - A200*B020;
    C[13] = A010*B022 - A012*B002 + A212*B000 - A210*B020;
    C[14] = A020*B022 - A022*B002 + A222*B000 - A220*B020;
    C[15] = A001*B002 - A000*B012 - A201*B000 + A200*B010;
    C[16] = A011*B002 - A010*B012 - A211*B000 + A210*B010;
    C[17] = A021*B002 - A020*B012 - A221*B000 + A220*B010;
    C[18] = A001*B021 - A002*B011 + A102*B010 - A101*B020;
    C[19] = A011*B021 - A012*B011 + A112*B010 - A111*B020;
    C[20] = A021*B021 - A022*B011 + A122*B010 - A121*B020;
    C[21] = A002*B001 - A000*B021 - A102*B000 + A100*B020;
    C[22] = A012*B001 - A010*B021 - A112*B000 + A110*B020;
    C[23] = A022*B001 - A020*B021 - A122*B000 + A120*B020;
    C[24] = A000*B011 - A001*B001 + A101*B000 - A100*B010;
    C[25] = A010*B011 - A011*B001 + A111*B000 - A110*B010;
    C[26] = A020*B011 - A021*B001 + A121*B000 - A120*B010;
    C[27] = A101*B122 - A102*B112 + A202*B111 - A201*B121;
    C[28] = A111*B122 - A112*B112 + A212*B111 - A211*B121;
    C[29] = A121*B122 - A122*B112 + A222*B111 - A221*B121;
    C[30] = A102*B102 - A100*B122 - A202*B101 + A200*B121;
    C[31] = A112*B102 - A110*B122 - A212*B101 + A210*B121;
    C[32] = A122*B102 - A120*B122 - A222*B101 + A220*B121;
    C[33] = A100*B112 - A101*B102 + A201*B101 - A200*B111;
    C[34] = A110*B112 - A111*B102 + A211*B101 - A210*B111;
    C[35] = A120*B112 - A121*B102 + A221*B101 - A220*B111;
    C[36] = A002*B112 - A001*B122 - A202*B110 + A201*B120;
    C[37] = A012*B112 - A011*B122 - A212*B110 + A211*B120;
    C[38] = A022*B112 - A021*B122 - A222*B110 + A221*B120;
    C[39] = A000*B122 - A002*B102 + A202*B100 - A200*B120;
    C[40] = A010*B122 - A012*B102 + A212*B100 - A210*B120;
    C[41] = A020*B122 - A022*B102 + A222*B100 - A220*B120;
    C[42] = A001*B102 - A000*B112 - A201*B100 + A200*B110;
    C[43] = A011*B102 - A010*B112 - A211*B100 + A210*B110;
    C[44] = A021*B102 - A020*B112 - A221*B100 + A220*B110;
    C[45] = A001*B121 - A002*B111 + A102*B110 - A101*B120;
    C[46] = A011*B121 - A012*B111 + A112*B110 - A111*B120;
    C[47] = A021*B121 - A022*B111 + A122*B110 - A121*B120;
    C[48] = A002*B101 - A000*B121 - A102*B100 + A100*B120;
    C[49] = A012*B101 - A010*B121 - A112*B100 + A110*B120;
    C[50] = A022*B101 - A020*B121 - A122*B100 + A120*B120;
    C[51] = A000*B111 - A001*B101 + A101*B100 - A100*B110;
    C[52] = A010*B111 - A011*B101 + A111*B100 - A110*B110;
    C[53] = A020*B111 - A021*B101 + A121*B100 - A120*B110;
    C[54] = A101*B222 - A102*B212 + A202*B211 - A201*B221;
    C[55] = A111*B222 - A112*B212 + A212*B211 - A211*B221;
    C[56] = A121*B222 - A122*B212 + A222*B211 - A221*B221;
    C[57] = A102*B202 - A100*B222 - A202*B201 + A200*B221;
    C[58] = A112*B202 - A110*B222 - A212*B201 + A210*B221;
    C[59] = A122*B202 - A120*B222 - A222*B201 + A220*B221;
    C[60] = A100*B212 - A101*B202 + A201*B201 - A200*B211;
    C[61] = A110*B212 - A111*B202 + A211*B201 - A210*B211;
    C[62] = A120*B212 - A121*B202 + A221*B201 - A220*B211;
    C[63] = A002*B212 - A001*B222 - A202*B210 + A201*B220;
    C[64] = A012*B212 - A011*B222 - A212*B210 + A211*B220;
    C[65] = A022*B212 - A021*B222 - A222*B210 + A221*B220;
    C[66] = A000*B222 - A002*B202 + A202*B200 - A200*B220;
    C[67] = A010*B222 - A012*B202 + A212*B200 - A210*B220;
    C[68] = A020*B222 - A022*B202 + A222*B200 - A220*B220;
    C[69] = A001*B202 - A000*B212 - A201*B200 + A200*B210;
    C[70] = A011*B202 - A010*B212 - A211*B200 + A210*B210;
    C[71] = A021*B202 - A020*B212 - A221*B200 + A220*B210;
    C[72] = A001*B221 - A002*B211 + A102*B210 - A101*B220;
    C[73] = A011*B221 - A012*B211 + A112*B210 - A111*B220;
    C[74] = A021*B221 - A022*B211 + A122*B210 - A121*B220;
    C[75] = A002*B201 - A000*B221 - A102*B200 + A100*B220;
    C[76] = A012*B201 - A010*B221 - A112*B200 + A110*B220;
    C[77] = A022*B201 - A020*B221 - A122*B200 + A120*B220;
    C[78] = A000*B211 - A001*B201 + A101*B200 - A100*B210;
    C[79] = A010*B211 - A011*B201 + A111*B200 - A110*B210;
    C[80] = A020*B211 - A021*B201 + A121*B200 - A120*B210;

}



////////
//(AxB)_{pPiIqQ} = E_{ijk}E_{IJK}A_{pPjJ}B_{kKqQ}
// tensor cross product of 4th order tensors
template<typename T, size_t I, size_t J, size_t K, size_t L, size_t M, size_t N, size_t O, size_t P>
void _crossproduct(const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C) {

    T A0000 = A[0];
    T A0001 = A[1];
    T A0002 = A[2];
    T A0010 = A[3];
    T A0011 = A[4];
    T A0012 = A[5];
    T A0020 = A[6];
    T A0021 = A[7];
    T A0022 = A[8];
    T A0100 = A[9];
    T A0101 = A[10];
    T A0102 = A[11];
    T A0110 = A[12];
    T A0111 = A[13];
    T A0112 = A[14];
    T A0120 = A[15];
    T A0121 = A[16];
    T A0122 = A[17];
    T A0200 = A[18];
    T A0201 = A[19];
    T A0202 = A[20];
    T A0210 = A[21];
    T A0211 = A[22];
    T A0212 = A[23];
    T A0220 = A[24];
    T A0221 = A[25];
    T A0222 = A[26];
    T A1000 = A[27];
    T A1001 = A[28];
    T A1002 = A[29];
    T A1010 = A[30];
    T A1011 = A[31];
    T A1012 = A[32];
    T A1020 = A[33];
    T A1021 = A[34];
    T A1022 = A[35];
    T A1100 = A[36];
    T A1101 = A[37];
    T A1102 = A[38];
    T A1110 = A[39];
    T A1111 = A[40];
    T A1112 = A[41];
    T A1120 = A[42];
    T A1121 = A[43];
    T A1122 = A[44];
    T A1200 = A[45];
    T A1201 = A[46];
    T A1202 = A[47];
    T A1210 = A[48];
    T A1211 = A[49];
    T A1212 = A[50];
    T A1220 = A[51];
    T A1221 = A[52];
    T A1222 = A[53];
    T A2000 = A[54];
    T A2001 = A[55];
    T A2002 = A[56];
    T A2010 = A[57];
    T A2011 = A[58];
    T A2012 = A[59];
    T A2020 = A[60];
    T A2021 = A[61];
    T A2022 = A[62];
    T A2100 = A[63];
    T A2101 = A[64];
    T A2102 = A[65];
    T A2110 = A[66];
    T A2111 = A[67];
    T A2112 = A[68];
    T A2120 = A[69];
    T A2121 = A[70];
    T A2122 = A[71];
    T A2200 = A[72];
    T A2201 = A[73];
    T A2202 = A[74];
    T A2210 = A[75];
    T A2211 = A[76];
    T A2212 = A[77];
    T A2220 = A[78];
    T A2221 = A[79];
    T A2222 = A[80];


    T B0000 = B[0];
    T B0001 = B[1];
    T B0002 = B[2];
    T B0010 = B[3];
    T B0011 = B[4];
    T B0012 = B[5];
    T B0020 = B[6];
    T B0021 = B[7];
    T B0022 = B[8];
    T B0100 = B[9];
    T B0101 = B[10];
    T B0102 = B[11];
    T B0110 = B[12];
    T B0111 = B[13];
    T B0112 = B[14];
    T B0120 = B[15];
    T B0121 = B[16];
    T B0122 = B[17];
    T B0200 = B[18];
    T B0201 = B[19];
    T B0202 = B[20];
    T B0210 = B[21];
    T B0211 = B[22];
    T B0212 = B[23];
    T B0220 = B[24];
    T B0221 = B[25];
    T B0222 = B[26];
    T B1000 = B[27];
    T B1001 = B[28];
    T B1002 = B[29];
    T B1010 = B[30];
    T B1011 = B[31];
    T B1012 = B[32];
    T B1020 = B[33];
    T B1021 = B[34];
    T B1022 = B[35];
    T B1100 = B[36];
    T B1101 = B[37];
    T B1102 = B[38];
    T B1110 = B[39];
    T B1111 = B[40];
    T B1112 = B[41];
    T B1120 = B[42];
    T B1121 = B[43];
    T B1122 = B[44];
    T B1200 = B[45];
    T B1201 = B[46];
    T B1202 = B[47];
    T B1210 = B[48];
    T B1211 = B[49];
    T B1212 = B[50];
    T B1220 = B[51];
    T B1221 = B[52];
    T B1222 = B[53];
    T B2000 = B[54];
    T B2001 = B[55];
    T B2002 = B[56];
    T B2010 = B[57];
    T B2011 = B[58];
    T B2012 = B[59];
    T B2020 = B[60];
    T B2021 = B[61];
    T B2022 = B[62];
    T B2100 = B[63];
    T B2101 = B[64];
    T B2102 = B[65];
    T B2110 = B[66];
    T B2111 = B[67];
    T B2112 = B[68];
    T B2120 = B[69];
    T B2121 = B[70];
    T B2122 = B[71];
    T B2200 = B[72];
    T B2201 = B[73];
    T B2202 = B[74];
    T B2210 = B[75];
    T B2211 = B[76];
    T B2212 = B[77];
    T B2220 = B[78];
    T B2221 = B[79];
    T B2222 = B[80];


    C[0] = A1100*B0022 - A1200*B0012 - A2100*B0021 + A2200*B0011;
    C[1] = A1110*B0022 - A1210*B0012 - A2110*B0021 + A2210*B0011;
    C[2] = A1120*B0022 - A1220*B0012 - A2120*B0021 + A2220*B0011;
    C[3] = A1101*B0022 - A1201*B0012 - A2101*B0021 + A2201*B0011;
    C[4] = A1111*B0022 - A1211*B0012 - A2111*B0021 + A2211*B0011;
    C[5] = A1121*B0022 - A1221*B0012 - A2121*B0021 + A2221*B0011;
    C[6] = A1102*B0022 - A1202*B0012 - A2102*B0021 + A2202*B0011;
    C[7] = A1112*B0022 - A1212*B0012 - A2112*B0021 + A2212*B0011;
    C[8] = A1122*B0022 - A1222*B0012 - A2122*B0021 + A2222*B0011;
    C[9] = A1200*B0002 - A1000*B0022 + A2000*B0021 - A2200*B0001;
    C[10] = A1210*B0002 - A1010*B0022 + A2010*B0021 - A2210*B0001;
    C[11] = A1220*B0002 - A1020*B0022 + A2020*B0021 - A2220*B0001;
    C[12] = A1201*B0002 - A1001*B0022 + A2001*B0021 - A2201*B0001;
    C[13] = A1211*B0002 - A1011*B0022 + A2011*B0021 - A2211*B0001;
    C[14] = A1221*B0002 - A1021*B0022 + A2021*B0021 - A2221*B0001;
    C[15] = A1202*B0002 - A1002*B0022 + A2002*B0021 - A2202*B0001;
    C[16] = A1212*B0002 - A1012*B0022 + A2012*B0021 - A2212*B0001;
    C[17] = A1222*B0002 - A1022*B0022 + A2022*B0021 - A2222*B0001;
    C[18] = A1000*B0012 - A1100*B0002 - A2000*B0011 + A2100*B0001;
    C[19] = A1010*B0012 - A1110*B0002 - A2010*B0011 + A2110*B0001;
    C[20] = A1020*B0012 - A1120*B0002 - A2020*B0011 + A2120*B0001;
    C[21] = A1001*B0012 - A1101*B0002 - A2001*B0011 + A2101*B0001;
    C[22] = A1011*B0012 - A1111*B0002 - A2011*B0011 + A2111*B0001;
    C[23] = A1021*B0012 - A1121*B0002 - A2021*B0011 + A2121*B0001;
    C[24] = A1002*B0012 - A1102*B0002 - A2002*B0011 + A2102*B0001;
    C[25] = A1012*B0012 - A1112*B0002 - A2012*B0011 + A2112*B0001;
    C[26] = A1022*B0012 - A1122*B0002 - A2022*B0011 + A2122*B0001;
    C[27] = A0200*B0012 - A0100*B0022 + A2100*B0020 - A2200*B0010;
    C[28] = A0210*B0012 - A0110*B0022 + A2110*B0020 - A2210*B0010;
    C[29] = A0220*B0012 - A0120*B0022 + A2120*B0020 - A2220*B0010;
    C[30] = A0201*B0012 - A0101*B0022 + A2101*B0020 - A2201*B0010;
    C[31] = A0211*B0012 - A0111*B0022 + A2111*B0020 - A2211*B0010;
    C[32] = A0221*B0012 - A0121*B0022 + A2121*B0020 - A2221*B0010;
    C[33] = A0202*B0012 - A0102*B0022 + A2102*B0020 - A2202*B0010;
    C[34] = A0212*B0012 - A0112*B0022 + A2112*B0020 - A2212*B0010;
    C[35] = A0222*B0012 - A0122*B0022 + A2122*B0020 - A2222*B0010;
    C[36] = A0000*B0022 - A0200*B0002 - A2000*B0020 + A2200*B0000;
    C[37] = A0010*B0022 - A0210*B0002 - A2010*B0020 + A2210*B0000;
    C[38] = A0020*B0022 - A0220*B0002 - A2020*B0020 + A2220*B0000;
    C[39] = A0001*B0022 - A0201*B0002 - A2001*B0020 + A2201*B0000;
    C[40] = A0011*B0022 - A0211*B0002 - A2011*B0020 + A2211*B0000;
    C[41] = A0021*B0022 - A0221*B0002 - A2021*B0020 + A2221*B0000;
    C[42] = A0002*B0022 - A0202*B0002 - A2002*B0020 + A2202*B0000;
    C[43] = A0012*B0022 - A0212*B0002 - A2012*B0020 + A2212*B0000;
    C[44] = A0022*B0022 - A0222*B0002 - A2022*B0020 + A2222*B0000;
    C[45] = A0100*B0002 - A0000*B0012 + A2000*B0010 - A2100*B0000;
    C[46] = A0110*B0002 - A0010*B0012 + A2010*B0010 - A2110*B0000;
    C[47] = A0120*B0002 - A0020*B0012 + A2020*B0010 - A2120*B0000;
    C[48] = A0101*B0002 - A0001*B0012 + A2001*B0010 - A2101*B0000;
    C[49] = A0111*B0002 - A0011*B0012 + A2011*B0010 - A2111*B0000;
    C[50] = A0121*B0002 - A0021*B0012 + A2021*B0010 - A2121*B0000;
    C[51] = A0102*B0002 - A0002*B0012 + A2002*B0010 - A2102*B0000;
    C[52] = A0112*B0002 - A0012*B0012 + A2012*B0010 - A2112*B0000;
    C[53] = A0122*B0002 - A0022*B0012 + A2022*B0010 - A2122*B0000;
    C[54] = A0100*B0021 - A0200*B0011 - A1100*B0020 + A1200*B0010;
    C[55] = A0110*B0021 - A0210*B0011 - A1110*B0020 + A1210*B0010;
    C[56] = A0120*B0021 - A0220*B0011 - A1120*B0020 + A1220*B0010;
    C[57] = A0101*B0021 - A0201*B0011 - A1101*B0020 + A1201*B0010;
    C[58] = A0111*B0021 - A0211*B0011 - A1111*B0020 + A1211*B0010;
    C[59] = A0121*B0021 - A0221*B0011 - A1121*B0020 + A1221*B0010;
    C[60] = A0102*B0021 - A0202*B0011 - A1102*B0020 + A1202*B0010;
    C[61] = A0112*B0021 - A0212*B0011 - A1112*B0020 + A1212*B0010;
    C[62] = A0122*B0021 - A0222*B0011 - A1122*B0020 + A1222*B0010;
    C[63] = A0200*B0001 - A0000*B0021 + A1000*B0020 - A1200*B0000;
    C[64] = A0210*B0001 - A0010*B0021 + A1010*B0020 - A1210*B0000;
    C[65] = A0220*B0001 - A0020*B0021 + A1020*B0020 - A1220*B0000;
    C[66] = A0201*B0001 - A0001*B0021 + A1001*B0020 - A1201*B0000;
    C[67] = A0211*B0001 - A0011*B0021 + A1011*B0020 - A1211*B0000;
    C[68] = A0221*B0001 - A0021*B0021 + A1021*B0020 - A1221*B0000;
    C[69] = A0202*B0001 - A0002*B0021 + A1002*B0020 - A1202*B0000;
    C[70] = A0212*B0001 - A0012*B0021 + A1012*B0020 - A1212*B0000;
    C[71] = A0222*B0001 - A0022*B0021 + A1022*B0020 - A1222*B0000;
    C[72] = A0000*B0011 - A0100*B0001 - A1000*B0010 + A1100*B0000;
    C[73] = A0010*B0011 - A0110*B0001 - A1010*B0010 + A1110*B0000;
    C[74] = A0020*B0011 - A0120*B0001 - A1020*B0010 + A1120*B0000;
    C[75] = A0001*B0011 - A0101*B0001 - A1001*B0010 + A1101*B0000;
    C[76] = A0011*B0011 - A0111*B0001 - A1011*B0010 + A1111*B0000;
    C[77] = A0021*B0011 - A0121*B0001 - A1021*B0010 + A1121*B0000;
    C[78] = A0002*B0011 - A0102*B0001 - A1002*B0010 + A1102*B0000;
    C[79] = A0012*B0011 - A0112*B0001 - A1012*B0010 + A1112*B0000;
    C[80] = A0022*B0011 - A0122*B0001 - A1022*B0010 + A1122*B0000;
    C[81] = A1100*B0122 - A1200*B0112 - A2100*B0121 + A2200*B0111;
    C[82] = A1110*B0122 - A1210*B0112 - A2110*B0121 + A2210*B0111;
    C[83] = A1120*B0122 - A1220*B0112 - A2120*B0121 + A2220*B0111;
    C[84] = A1101*B0122 - A1201*B0112 - A2101*B0121 + A2201*B0111;
    C[85] = A1111*B0122 - A1211*B0112 - A2111*B0121 + A2211*B0111;
    C[86] = A1121*B0122 - A1221*B0112 - A2121*B0121 + A2221*B0111;
    C[87] = A1102*B0122 - A1202*B0112 - A2102*B0121 + A2202*B0111;
    C[88] = A1112*B0122 - A1212*B0112 - A2112*B0121 + A2212*B0111;
    C[89] = A1122*B0122 - A1222*B0112 - A2122*B0121 + A2222*B0111;
    C[90] = A1200*B0102 - A1000*B0122 + A2000*B0121 - A2200*B0101;
    C[91] = A1210*B0102 - A1010*B0122 + A2010*B0121 - A2210*B0101;
    C[92] = A1220*B0102 - A1020*B0122 + A2020*B0121 - A2220*B0101;
    C[93] = A1201*B0102 - A1001*B0122 + A2001*B0121 - A2201*B0101;
    C[94] = A1211*B0102 - A1011*B0122 + A2011*B0121 - A2211*B0101;
    C[95] = A1221*B0102 - A1021*B0122 + A2021*B0121 - A2221*B0101;
    C[96] = A1202*B0102 - A1002*B0122 + A2002*B0121 - A2202*B0101;
    C[97] = A1212*B0102 - A1012*B0122 + A2012*B0121 - A2212*B0101;
    C[98] = A1222*B0102 - A1022*B0122 + A2022*B0121 - A2222*B0101;
    C[99] = A1000*B0112 - A1100*B0102 - A2000*B0111 + A2100*B0101;
    C[100] = A1010*B0112 - A1110*B0102 - A2010*B0111 + A2110*B0101;
    C[101] = A1020*B0112 - A1120*B0102 - A2020*B0111 + A2120*B0101;
    C[102] = A1001*B0112 - A1101*B0102 - A2001*B0111 + A2101*B0101;
    C[103] = A1011*B0112 - A1111*B0102 - A2011*B0111 + A2111*B0101;
    C[104] = A1021*B0112 - A1121*B0102 - A2021*B0111 + A2121*B0101;
    C[105] = A1002*B0112 - A1102*B0102 - A2002*B0111 + A2102*B0101;
    C[106] = A1012*B0112 - A1112*B0102 - A2012*B0111 + A2112*B0101;
    C[107] = A1022*B0112 - A1122*B0102 - A2022*B0111 + A2122*B0101;
    C[108] = A0200*B0112 - A0100*B0122 + A2100*B0120 - A2200*B0110;
    C[109] = A0210*B0112 - A0110*B0122 + A2110*B0120 - A2210*B0110;
    C[110] = A0220*B0112 - A0120*B0122 + A2120*B0120 - A2220*B0110;
    C[111] = A0201*B0112 - A0101*B0122 + A2101*B0120 - A2201*B0110;
    C[112] = A0211*B0112 - A0111*B0122 + A2111*B0120 - A2211*B0110;
    C[113] = A0221*B0112 - A0121*B0122 + A2121*B0120 - A2221*B0110;
    C[114] = A0202*B0112 - A0102*B0122 + A2102*B0120 - A2202*B0110;
    C[115] = A0212*B0112 - A0112*B0122 + A2112*B0120 - A2212*B0110;
    C[116] = A0222*B0112 - A0122*B0122 + A2122*B0120 - A2222*B0110;
    C[117] = A0000*B0122 - A0200*B0102 - A2000*B0120 + A2200*B0100;
    C[118] = A0010*B0122 - A0210*B0102 - A2010*B0120 + A2210*B0100;
    C[119] = A0020*B0122 - A0220*B0102 - A2020*B0120 + A2220*B0100;
    C[120] = A0001*B0122 - A0201*B0102 - A2001*B0120 + A2201*B0100;
    C[121] = A0011*B0122 - A0211*B0102 - A2011*B0120 + A2211*B0100;
    C[122] = A0021*B0122 - A0221*B0102 - A2021*B0120 + A2221*B0100;
    C[123] = A0002*B0122 - A0202*B0102 - A2002*B0120 + A2202*B0100;
    C[124] = A0012*B0122 - A0212*B0102 - A2012*B0120 + A2212*B0100;
    C[125] = A0022*B0122 - A0222*B0102 - A2022*B0120 + A2222*B0100;
    C[126] = A0100*B0102 - A0000*B0112 + A2000*B0110 - A2100*B0100;
    C[127] = A0110*B0102 - A0010*B0112 + A2010*B0110 - A2110*B0100;
    C[128] = A0120*B0102 - A0020*B0112 + A2020*B0110 - A2120*B0100;
    C[129] = A0101*B0102 - A0001*B0112 + A2001*B0110 - A2101*B0100;
    C[130] = A0111*B0102 - A0011*B0112 + A2011*B0110 - A2111*B0100;
    C[131] = A0121*B0102 - A0021*B0112 + A2021*B0110 - A2121*B0100;
    C[132] = A0102*B0102 - A0002*B0112 + A2002*B0110 - A2102*B0100;
    C[133] = A0112*B0102 - A0012*B0112 + A2012*B0110 - A2112*B0100;
    C[134] = A0122*B0102 - A0022*B0112 + A2022*B0110 - A2122*B0100;
    C[135] = A0100*B0121 - A0200*B0111 - A1100*B0120 + A1200*B0110;
    C[136] = A0110*B0121 - A0210*B0111 - A1110*B0120 + A1210*B0110;
    C[137] = A0120*B0121 - A0220*B0111 - A1120*B0120 + A1220*B0110;
    C[138] = A0101*B0121 - A0201*B0111 - A1101*B0120 + A1201*B0110;
    C[139] = A0111*B0121 - A0211*B0111 - A1111*B0120 + A1211*B0110;
    C[140] = A0121*B0121 - A0221*B0111 - A1121*B0120 + A1221*B0110;
    C[141] = A0102*B0121 - A0202*B0111 - A1102*B0120 + A1202*B0110;
    C[142] = A0112*B0121 - A0212*B0111 - A1112*B0120 + A1212*B0110;
    C[143] = A0122*B0121 - A0222*B0111 - A1122*B0120 + A1222*B0110;
    C[144] = A0200*B0101 - A0000*B0121 + A1000*B0120 - A1200*B0100;
    C[145] = A0210*B0101 - A0010*B0121 + A1010*B0120 - A1210*B0100;
    C[146] = A0220*B0101 - A0020*B0121 + A1020*B0120 - A1220*B0100;
    C[147] = A0201*B0101 - A0001*B0121 + A1001*B0120 - A1201*B0100;
    C[148] = A0211*B0101 - A0011*B0121 + A1011*B0120 - A1211*B0100;
    C[149] = A0221*B0101 - A0021*B0121 + A1021*B0120 - A1221*B0100;
    C[150] = A0202*B0101 - A0002*B0121 + A1002*B0120 - A1202*B0100;
    C[151] = A0212*B0101 - A0012*B0121 + A1012*B0120 - A1212*B0100;
    C[152] = A0222*B0101 - A0022*B0121 + A1022*B0120 - A1222*B0100;
    C[153] = A0000*B0111 - A0100*B0101 - A1000*B0110 + A1100*B0100;
    C[154] = A0010*B0111 - A0110*B0101 - A1010*B0110 + A1110*B0100;
    C[155] = A0020*B0111 - A0120*B0101 - A1020*B0110 + A1120*B0100;
    C[156] = A0001*B0111 - A0101*B0101 - A1001*B0110 + A1101*B0100;
    C[157] = A0011*B0111 - A0111*B0101 - A1011*B0110 + A1111*B0100;
    C[158] = A0021*B0111 - A0121*B0101 - A1021*B0110 + A1121*B0100;
    C[159] = A0002*B0111 - A0102*B0101 - A1002*B0110 + A1102*B0100;
    C[160] = A0012*B0111 - A0112*B0101 - A1012*B0110 + A1112*B0100;
    C[161] = A0022*B0111 - A0122*B0101 - A1022*B0110 + A1122*B0100;
    C[162] = A1100*B0222 - A1200*B0212 - A2100*B0221 + A2200*B0211;
    C[163] = A1110*B0222 - A1210*B0212 - A2110*B0221 + A2210*B0211;
    C[164] = A1120*B0222 - A1220*B0212 - A2120*B0221 + A2220*B0211;
    C[165] = A1101*B0222 - A1201*B0212 - A2101*B0221 + A2201*B0211;
    C[166] = A1111*B0222 - A1211*B0212 - A2111*B0221 + A2211*B0211;
    C[167] = A1121*B0222 - A1221*B0212 - A2121*B0221 + A2221*B0211;
    C[168] = A1102*B0222 - A1202*B0212 - A2102*B0221 + A2202*B0211;
    C[169] = A1112*B0222 - A1212*B0212 - A2112*B0221 + A2212*B0211;
    C[170] = A1122*B0222 - A1222*B0212 - A2122*B0221 + A2222*B0211;
    C[171] = A1200*B0202 - A1000*B0222 + A2000*B0221 - A2200*B0201;
    C[172] = A1210*B0202 - A1010*B0222 + A2010*B0221 - A2210*B0201;
    C[173] = A1220*B0202 - A1020*B0222 + A2020*B0221 - A2220*B0201;
    C[174] = A1201*B0202 - A1001*B0222 + A2001*B0221 - A2201*B0201;
    C[175] = A1211*B0202 - A1011*B0222 + A2011*B0221 - A2211*B0201;
    C[176] = A1221*B0202 - A1021*B0222 + A2021*B0221 - A2221*B0201;
    C[177] = A1202*B0202 - A1002*B0222 + A2002*B0221 - A2202*B0201;
    C[178] = A1212*B0202 - A1012*B0222 + A2012*B0221 - A2212*B0201;
    C[179] = A1222*B0202 - A1022*B0222 + A2022*B0221 - A2222*B0201;
    C[180] = A1000*B0212 - A1100*B0202 - A2000*B0211 + A2100*B0201;
    C[181] = A1010*B0212 - A1110*B0202 - A2010*B0211 + A2110*B0201;
    C[182] = A1020*B0212 - A1120*B0202 - A2020*B0211 + A2120*B0201;
    C[183] = A1001*B0212 - A1101*B0202 - A2001*B0211 + A2101*B0201;
    C[184] = A1011*B0212 - A1111*B0202 - A2011*B0211 + A2111*B0201;
    C[185] = A1021*B0212 - A1121*B0202 - A2021*B0211 + A2121*B0201;
    C[186] = A1002*B0212 - A1102*B0202 - A2002*B0211 + A2102*B0201;
    C[187] = A1012*B0212 - A1112*B0202 - A2012*B0211 + A2112*B0201;
    C[188] = A1022*B0212 - A1122*B0202 - A2022*B0211 + A2122*B0201;
    C[189] = A0200*B0212 - A0100*B0222 + A2100*B0220 - A2200*B0210;
    C[190] = A0210*B0212 - A0110*B0222 + A2110*B0220 - A2210*B0210;
    C[191] = A0220*B0212 - A0120*B0222 + A2120*B0220 - A2220*B0210;
    C[192] = A0201*B0212 - A0101*B0222 + A2101*B0220 - A2201*B0210;
    C[193] = A0211*B0212 - A0111*B0222 + A2111*B0220 - A2211*B0210;
    C[194] = A0221*B0212 - A0121*B0222 + A2121*B0220 - A2221*B0210;
    C[195] = A0202*B0212 - A0102*B0222 + A2102*B0220 - A2202*B0210;
    C[196] = A0212*B0212 - A0112*B0222 + A2112*B0220 - A2212*B0210;
    C[197] = A0222*B0212 - A0122*B0222 + A2122*B0220 - A2222*B0210;
    C[198] = A0000*B0222 - A0200*B0202 - A2000*B0220 + A2200*B0200;
    C[199] = A0010*B0222 - A0210*B0202 - A2010*B0220 + A2210*B0200;
    C[200] = A0020*B0222 - A0220*B0202 - A2020*B0220 + A2220*B0200;
    C[201] = A0001*B0222 - A0201*B0202 - A2001*B0220 + A2201*B0200;
    C[202] = A0011*B0222 - A0211*B0202 - A2011*B0220 + A2211*B0200;
    C[203] = A0021*B0222 - A0221*B0202 - A2021*B0220 + A2221*B0200;
    C[204] = A0002*B0222 - A0202*B0202 - A2002*B0220 + A2202*B0200;
    C[205] = A0012*B0222 - A0212*B0202 - A2012*B0220 + A2212*B0200;
    C[206] = A0022*B0222 - A0222*B0202 - A2022*B0220 + A2222*B0200;
    C[207] = A0100*B0202 - A0000*B0212 + A2000*B0210 - A2100*B0200;
    C[208] = A0110*B0202 - A0010*B0212 + A2010*B0210 - A2110*B0200;
    C[209] = A0120*B0202 - A0020*B0212 + A2020*B0210 - A2120*B0200;
    C[210] = A0101*B0202 - A0001*B0212 + A2001*B0210 - A2101*B0200;
    C[211] = A0111*B0202 - A0011*B0212 + A2011*B0210 - A2111*B0200;
    C[212] = A0121*B0202 - A0021*B0212 + A2021*B0210 - A2121*B0200;
    C[213] = A0102*B0202 - A0002*B0212 + A2002*B0210 - A2102*B0200;
    C[214] = A0112*B0202 - A0012*B0212 + A2012*B0210 - A2112*B0200;
    C[215] = A0122*B0202 - A0022*B0212 + A2022*B0210 - A2122*B0200;
    C[216] = A0100*B0221 - A0200*B0211 - A1100*B0220 + A1200*B0210;
    C[217] = A0110*B0221 - A0210*B0211 - A1110*B0220 + A1210*B0210;
    C[218] = A0120*B0221 - A0220*B0211 - A1120*B0220 + A1220*B0210;
    C[219] = A0101*B0221 - A0201*B0211 - A1101*B0220 + A1201*B0210;
    C[220] = A0111*B0221 - A0211*B0211 - A1111*B0220 + A1211*B0210;
    C[221] = A0121*B0221 - A0221*B0211 - A1121*B0220 + A1221*B0210;
    C[222] = A0102*B0221 - A0202*B0211 - A1102*B0220 + A1202*B0210;
    C[223] = A0112*B0221 - A0212*B0211 - A1112*B0220 + A1212*B0210;
    C[224] = A0122*B0221 - A0222*B0211 - A1122*B0220 + A1222*B0210;
    C[225] = A0200*B0201 - A0000*B0221 + A1000*B0220 - A1200*B0200;
    C[226] = A0210*B0201 - A0010*B0221 + A1010*B0220 - A1210*B0200;
    C[227] = A0220*B0201 - A0020*B0221 + A1020*B0220 - A1220*B0200;
    C[228] = A0201*B0201 - A0001*B0221 + A1001*B0220 - A1201*B0200;
    C[229] = A0211*B0201 - A0011*B0221 + A1011*B0220 - A1211*B0200;
    C[230] = A0221*B0201 - A0021*B0221 + A1021*B0220 - A1221*B0200;
    C[231] = A0202*B0201 - A0002*B0221 + A1002*B0220 - A1202*B0200;
    C[232] = A0212*B0201 - A0012*B0221 + A1012*B0220 - A1212*B0200;
    C[233] = A0222*B0201 - A0022*B0221 + A1022*B0220 - A1222*B0200;
    C[234] = A0000*B0211 - A0100*B0201 - A1000*B0210 + A1100*B0200;
    C[235] = A0010*B0211 - A0110*B0201 - A1010*B0210 + A1110*B0200;
    C[236] = A0020*B0211 - A0120*B0201 - A1020*B0210 + A1120*B0200;
    C[237] = A0001*B0211 - A0101*B0201 - A1001*B0210 + A1101*B0200;
    C[238] = A0011*B0211 - A0111*B0201 - A1011*B0210 + A1111*B0200;
    C[239] = A0021*B0211 - A0121*B0201 - A1021*B0210 + A1121*B0200;
    C[240] = A0002*B0211 - A0102*B0201 - A1002*B0210 + A1102*B0200;
    C[241] = A0012*B0211 - A0112*B0201 - A1012*B0210 + A1112*B0200;
    C[242] = A0022*B0211 - A0122*B0201 - A1022*B0210 + A1122*B0200;
    C[243] = A1100*B1022 - A1200*B1012 - A2100*B1021 + A2200*B1011;
    C[244] = A1110*B1022 - A1210*B1012 - A2110*B1021 + A2210*B1011;
    C[245] = A1120*B1022 - A1220*B1012 - A2120*B1021 + A2220*B1011;
    C[246] = A1101*B1022 - A1201*B1012 - A2101*B1021 + A2201*B1011;
    C[247] = A1111*B1022 - A1211*B1012 - A2111*B1021 + A2211*B1011;
    C[248] = A1121*B1022 - A1221*B1012 - A2121*B1021 + A2221*B1011;
    C[249] = A1102*B1022 - A1202*B1012 - A2102*B1021 + A2202*B1011;
    C[250] = A1112*B1022 - A1212*B1012 - A2112*B1021 + A2212*B1011;
    C[251] = A1122*B1022 - A1222*B1012 - A2122*B1021 + A2222*B1011;
    C[252] = A1200*B1002 - A1000*B1022 + A2000*B1021 - A2200*B1001;
    C[253] = A1210*B1002 - A1010*B1022 + A2010*B1021 - A2210*B1001;
    C[254] = A1220*B1002 - A1020*B1022 + A2020*B1021 - A2220*B1001;
    C[255] = A1201*B1002 - A1001*B1022 + A2001*B1021 - A2201*B1001;
    C[256] = A1211*B1002 - A1011*B1022 + A2011*B1021 - A2211*B1001;
    C[257] = A1221*B1002 - A1021*B1022 + A2021*B1021 - A2221*B1001;
    C[258] = A1202*B1002 - A1002*B1022 + A2002*B1021 - A2202*B1001;
    C[259] = A1212*B1002 - A1012*B1022 + A2012*B1021 - A2212*B1001;
    C[260] = A1222*B1002 - A1022*B1022 + A2022*B1021 - A2222*B1001;
    C[261] = A1000*B1012 - A1100*B1002 - A2000*B1011 + A2100*B1001;
    C[262] = A1010*B1012 - A1110*B1002 - A2010*B1011 + A2110*B1001;
    C[263] = A1020*B1012 - A1120*B1002 - A2020*B1011 + A2120*B1001;
    C[264] = A1001*B1012 - A1101*B1002 - A2001*B1011 + A2101*B1001;
    C[265] = A1011*B1012 - A1111*B1002 - A2011*B1011 + A2111*B1001;
    C[266] = A1021*B1012 - A1121*B1002 - A2021*B1011 + A2121*B1001;
    C[267] = A1002*B1012 - A1102*B1002 - A2002*B1011 + A2102*B1001;
    C[268] = A1012*B1012 - A1112*B1002 - A2012*B1011 + A2112*B1001;
    C[269] = A1022*B1012 - A1122*B1002 - A2022*B1011 + A2122*B1001;
    C[270] = A0200*B1012 - A0100*B1022 + A2100*B1020 - A2200*B1010;
    C[271] = A0210*B1012 - A0110*B1022 + A2110*B1020 - A2210*B1010;
    C[272] = A0220*B1012 - A0120*B1022 + A2120*B1020 - A2220*B1010;
    C[273] = A0201*B1012 - A0101*B1022 + A2101*B1020 - A2201*B1010;
    C[274] = A0211*B1012 - A0111*B1022 + A2111*B1020 - A2211*B1010;
    C[275] = A0221*B1012 - A0121*B1022 + A2121*B1020 - A2221*B1010;
    C[276] = A0202*B1012 - A0102*B1022 + A2102*B1020 - A2202*B1010;
    C[277] = A0212*B1012 - A0112*B1022 + A2112*B1020 - A2212*B1010;
    C[278] = A0222*B1012 - A0122*B1022 + A2122*B1020 - A2222*B1010;
    C[279] = A0000*B1022 - A0200*B1002 - A2000*B1020 + A2200*B1000;
    C[280] = A0010*B1022 - A0210*B1002 - A2010*B1020 + A2210*B1000;
    C[281] = A0020*B1022 - A0220*B1002 - A2020*B1020 + A2220*B1000;
    C[282] = A0001*B1022 - A0201*B1002 - A2001*B1020 + A2201*B1000;
    C[283] = A0011*B1022 - A0211*B1002 - A2011*B1020 + A2211*B1000;
    C[284] = A0021*B1022 - A0221*B1002 - A2021*B1020 + A2221*B1000;
    C[285] = A0002*B1022 - A0202*B1002 - A2002*B1020 + A2202*B1000;
    C[286] = A0012*B1022 - A0212*B1002 - A2012*B1020 + A2212*B1000;
    C[287] = A0022*B1022 - A0222*B1002 - A2022*B1020 + A2222*B1000;
    C[288] = A0100*B1002 - A0000*B1012 + A2000*B1010 - A2100*B1000;
    C[289] = A0110*B1002 - A0010*B1012 + A2010*B1010 - A2110*B1000;
    C[290] = A0120*B1002 - A0020*B1012 + A2020*B1010 - A2120*B1000;
    C[291] = A0101*B1002 - A0001*B1012 + A2001*B1010 - A2101*B1000;
    C[292] = A0111*B1002 - A0011*B1012 + A2011*B1010 - A2111*B1000;
    C[293] = A0121*B1002 - A0021*B1012 + A2021*B1010 - A2121*B1000;
    C[294] = A0102*B1002 - A0002*B1012 + A2002*B1010 - A2102*B1000;
    C[295] = A0112*B1002 - A0012*B1012 + A2012*B1010 - A2112*B1000;
    C[296] = A0122*B1002 - A0022*B1012 + A2022*B1010 - A2122*B1000;
    C[297] = A0100*B1021 - A0200*B1011 - A1100*B1020 + A1200*B1010;
    C[298] = A0110*B1021 - A0210*B1011 - A1110*B1020 + A1210*B1010;
    C[299] = A0120*B1021 - A0220*B1011 - A1120*B1020 + A1220*B1010;
    C[300] = A0101*B1021 - A0201*B1011 - A1101*B1020 + A1201*B1010;
    C[301] = A0111*B1021 - A0211*B1011 - A1111*B1020 + A1211*B1010;
    C[302] = A0121*B1021 - A0221*B1011 - A1121*B1020 + A1221*B1010;
    C[303] = A0102*B1021 - A0202*B1011 - A1102*B1020 + A1202*B1010;
    C[304] = A0112*B1021 - A0212*B1011 - A1112*B1020 + A1212*B1010;
    C[305] = A0122*B1021 - A0222*B1011 - A1122*B1020 + A1222*B1010;
    C[306] = A0200*B1001 - A0000*B1021 + A1000*B1020 - A1200*B1000;
    C[307] = A0210*B1001 - A0010*B1021 + A1010*B1020 - A1210*B1000;
    C[308] = A0220*B1001 - A0020*B1021 + A1020*B1020 - A1220*B1000;
    C[309] = A0201*B1001 - A0001*B1021 + A1001*B1020 - A1201*B1000;
    C[310] = A0211*B1001 - A0011*B1021 + A1011*B1020 - A1211*B1000;
    C[311] = A0221*B1001 - A0021*B1021 + A1021*B1020 - A1221*B1000;
    C[312] = A0202*B1001 - A0002*B1021 + A1002*B1020 - A1202*B1000;
    C[313] = A0212*B1001 - A0012*B1021 + A1012*B1020 - A1212*B1000;
    C[314] = A0222*B1001 - A0022*B1021 + A1022*B1020 - A1222*B1000;
    C[315] = A0000*B1011 - A0100*B1001 - A1000*B1010 + A1100*B1000;
    C[316] = A0010*B1011 - A0110*B1001 - A1010*B1010 + A1110*B1000;
    C[317] = A0020*B1011 - A0120*B1001 - A1020*B1010 + A1120*B1000;
    C[318] = A0001*B1011 - A0101*B1001 - A1001*B1010 + A1101*B1000;
    C[319] = A0011*B1011 - A0111*B1001 - A1011*B1010 + A1111*B1000;
    C[320] = A0021*B1011 - A0121*B1001 - A1021*B1010 + A1121*B1000;
    C[321] = A0002*B1011 - A0102*B1001 - A1002*B1010 + A1102*B1000;
    C[322] = A0012*B1011 - A0112*B1001 - A1012*B1010 + A1112*B1000;
    C[323] = A0022*B1011 - A0122*B1001 - A1022*B1010 + A1122*B1000;
    C[324] = A1100*B1122 - A1200*B1112 - A2100*B1121 + A2200*B1111;
    C[325] = A1110*B1122 - A1210*B1112 - A2110*B1121 + A2210*B1111;
    C[326] = A1120*B1122 - A1220*B1112 - A2120*B1121 + A2220*B1111;
    C[327] = A1101*B1122 - A1201*B1112 - A2101*B1121 + A2201*B1111;
    C[328] = A1111*B1122 - A1211*B1112 - A2111*B1121 + A2211*B1111;
    C[329] = A1121*B1122 - A1221*B1112 - A2121*B1121 + A2221*B1111;
    C[330] = A1102*B1122 - A1202*B1112 - A2102*B1121 + A2202*B1111;
    C[331] = A1112*B1122 - A1212*B1112 - A2112*B1121 + A2212*B1111;
    C[332] = A1122*B1122 - A1222*B1112 - A2122*B1121 + A2222*B1111;
    C[333] = A1200*B1102 - A1000*B1122 + A2000*B1121 - A2200*B1101;
    C[334] = A1210*B1102 - A1010*B1122 + A2010*B1121 - A2210*B1101;
    C[335] = A1220*B1102 - A1020*B1122 + A2020*B1121 - A2220*B1101;
    C[336] = A1201*B1102 - A1001*B1122 + A2001*B1121 - A2201*B1101;
    C[337] = A1211*B1102 - A1011*B1122 + A2011*B1121 - A2211*B1101;
    C[338] = A1221*B1102 - A1021*B1122 + A2021*B1121 - A2221*B1101;
    C[339] = A1202*B1102 - A1002*B1122 + A2002*B1121 - A2202*B1101;
    C[340] = A1212*B1102 - A1012*B1122 + A2012*B1121 - A2212*B1101;
    C[341] = A1222*B1102 - A1022*B1122 + A2022*B1121 - A2222*B1101;
    C[342] = A1000*B1112 - A1100*B1102 - A2000*B1111 + A2100*B1101;
    C[343] = A1010*B1112 - A1110*B1102 - A2010*B1111 + A2110*B1101;
    C[344] = A1020*B1112 - A1120*B1102 - A2020*B1111 + A2120*B1101;
    C[345] = A1001*B1112 - A1101*B1102 - A2001*B1111 + A2101*B1101;
    C[346] = A1011*B1112 - A1111*B1102 - A2011*B1111 + A2111*B1101;
    C[347] = A1021*B1112 - A1121*B1102 - A2021*B1111 + A2121*B1101;
    C[348] = A1002*B1112 - A1102*B1102 - A2002*B1111 + A2102*B1101;
    C[349] = A1012*B1112 - A1112*B1102 - A2012*B1111 + A2112*B1101;
    C[350] = A1022*B1112 - A1122*B1102 - A2022*B1111 + A2122*B1101;
    C[351] = A0200*B1112 - A0100*B1122 + A2100*B1120 - A2200*B1110;
    C[352] = A0210*B1112 - A0110*B1122 + A2110*B1120 - A2210*B1110;
    C[353] = A0220*B1112 - A0120*B1122 + A2120*B1120 - A2220*B1110;
    C[354] = A0201*B1112 - A0101*B1122 + A2101*B1120 - A2201*B1110;
    C[355] = A0211*B1112 - A0111*B1122 + A2111*B1120 - A2211*B1110;
    C[356] = A0221*B1112 - A0121*B1122 + A2121*B1120 - A2221*B1110;
    C[357] = A0202*B1112 - A0102*B1122 + A2102*B1120 - A2202*B1110;
    C[358] = A0212*B1112 - A0112*B1122 + A2112*B1120 - A2212*B1110;
    C[359] = A0222*B1112 - A0122*B1122 + A2122*B1120 - A2222*B1110;
    C[360] = A0000*B1122 - A0200*B1102 - A2000*B1120 + A2200*B1100;
    C[361] = A0010*B1122 - A0210*B1102 - A2010*B1120 + A2210*B1100;
    C[362] = A0020*B1122 - A0220*B1102 - A2020*B1120 + A2220*B1100;
    C[363] = A0001*B1122 - A0201*B1102 - A2001*B1120 + A2201*B1100;
    C[364] = A0011*B1122 - A0211*B1102 - A2011*B1120 + A2211*B1100;
    C[365] = A0021*B1122 - A0221*B1102 - A2021*B1120 + A2221*B1100;
    C[366] = A0002*B1122 - A0202*B1102 - A2002*B1120 + A2202*B1100;
    C[367] = A0012*B1122 - A0212*B1102 - A2012*B1120 + A2212*B1100;
    C[368] = A0022*B1122 - A0222*B1102 - A2022*B1120 + A2222*B1100;
    C[369] = A0100*B1102 - A0000*B1112 + A2000*B1110 - A2100*B1100;
    C[370] = A0110*B1102 - A0010*B1112 + A2010*B1110 - A2110*B1100;
    C[371] = A0120*B1102 - A0020*B1112 + A2020*B1110 - A2120*B1100;
    C[372] = A0101*B1102 - A0001*B1112 + A2001*B1110 - A2101*B1100;
    C[373] = A0111*B1102 - A0011*B1112 + A2011*B1110 - A2111*B1100;
    C[374] = A0121*B1102 - A0021*B1112 + A2021*B1110 - A2121*B1100;
    C[375] = A0102*B1102 - A0002*B1112 + A2002*B1110 - A2102*B1100;
    C[376] = A0112*B1102 - A0012*B1112 + A2012*B1110 - A2112*B1100;
    C[377] = A0122*B1102 - A0022*B1112 + A2022*B1110 - A2122*B1100;
    C[378] = A0100*B1121 - A0200*B1111 - A1100*B1120 + A1200*B1110;
    C[379] = A0110*B1121 - A0210*B1111 - A1110*B1120 + A1210*B1110;
    C[380] = A0120*B1121 - A0220*B1111 - A1120*B1120 + A1220*B1110;
    C[381] = A0101*B1121 - A0201*B1111 - A1101*B1120 + A1201*B1110;
    C[382] = A0111*B1121 - A0211*B1111 - A1111*B1120 + A1211*B1110;
    C[383] = A0121*B1121 - A0221*B1111 - A1121*B1120 + A1221*B1110;
    C[384] = A0102*B1121 - A0202*B1111 - A1102*B1120 + A1202*B1110;
    C[385] = A0112*B1121 - A0212*B1111 - A1112*B1120 + A1212*B1110;
    C[386] = A0122*B1121 - A0222*B1111 - A1122*B1120 + A1222*B1110;
    C[387] = A0200*B1101 - A0000*B1121 + A1000*B1120 - A1200*B1100;
    C[388] = A0210*B1101 - A0010*B1121 + A1010*B1120 - A1210*B1100;
    C[389] = A0220*B1101 - A0020*B1121 + A1020*B1120 - A1220*B1100;
    C[390] = A0201*B1101 - A0001*B1121 + A1001*B1120 - A1201*B1100;
    C[391] = A0211*B1101 - A0011*B1121 + A1011*B1120 - A1211*B1100;
    C[392] = A0221*B1101 - A0021*B1121 + A1021*B1120 - A1221*B1100;
    C[393] = A0202*B1101 - A0002*B1121 + A1002*B1120 - A1202*B1100;
    C[394] = A0212*B1101 - A0012*B1121 + A1012*B1120 - A1212*B1100;
    C[395] = A0222*B1101 - A0022*B1121 + A1022*B1120 - A1222*B1100;
    C[396] = A0000*B1111 - A0100*B1101 - A1000*B1110 + A1100*B1100;
    C[397] = A0010*B1111 - A0110*B1101 - A1010*B1110 + A1110*B1100;
    C[398] = A0020*B1111 - A0120*B1101 - A1020*B1110 + A1120*B1100;
    C[399] = A0001*B1111 - A0101*B1101 - A1001*B1110 + A1101*B1100;
    C[400] = A0011*B1111 - A0111*B1101 - A1011*B1110 + A1111*B1100;
    C[401] = A0021*B1111 - A0121*B1101 - A1021*B1110 + A1121*B1100;
    C[402] = A0002*B1111 - A0102*B1101 - A1002*B1110 + A1102*B1100;
    C[403] = A0012*B1111 - A0112*B1101 - A1012*B1110 + A1112*B1100;
    C[404] = A0022*B1111 - A0122*B1101 - A1022*B1110 + A1122*B1100;
    C[405] = A1100*B1222 - A1200*B1212 - A2100*B1221 + A2200*B1211;
    C[406] = A1110*B1222 - A1210*B1212 - A2110*B1221 + A2210*B1211;
    C[407] = A1120*B1222 - A1220*B1212 - A2120*B1221 + A2220*B1211;
    C[408] = A1101*B1222 - A1201*B1212 - A2101*B1221 + A2201*B1211;
    C[409] = A1111*B1222 - A1211*B1212 - A2111*B1221 + A2211*B1211;
    C[410] = A1121*B1222 - A1221*B1212 - A2121*B1221 + A2221*B1211;
    C[411] = A1102*B1222 - A1202*B1212 - A2102*B1221 + A2202*B1211;
    C[412] = A1112*B1222 - A1212*B1212 - A2112*B1221 + A2212*B1211;
    C[413] = A1122*B1222 - A1222*B1212 - A2122*B1221 + A2222*B1211;
    C[414] = A1200*B1202 - A1000*B1222 + A2000*B1221 - A2200*B1201;
    C[415] = A1210*B1202 - A1010*B1222 + A2010*B1221 - A2210*B1201;
    C[416] = A1220*B1202 - A1020*B1222 + A2020*B1221 - A2220*B1201;
    C[417] = A1201*B1202 - A1001*B1222 + A2001*B1221 - A2201*B1201;
    C[418] = A1211*B1202 - A1011*B1222 + A2011*B1221 - A2211*B1201;
    C[419] = A1221*B1202 - A1021*B1222 + A2021*B1221 - A2221*B1201;
    C[420] = A1202*B1202 - A1002*B1222 + A2002*B1221 - A2202*B1201;
    C[421] = A1212*B1202 - A1012*B1222 + A2012*B1221 - A2212*B1201;
    C[422] = A1222*B1202 - A1022*B1222 + A2022*B1221 - A2222*B1201;
    C[423] = A1000*B1212 - A1100*B1202 - A2000*B1211 + A2100*B1201;
    C[424] = A1010*B1212 - A1110*B1202 - A2010*B1211 + A2110*B1201;
    C[425] = A1020*B1212 - A1120*B1202 - A2020*B1211 + A2120*B1201;
    C[426] = A1001*B1212 - A1101*B1202 - A2001*B1211 + A2101*B1201;
    C[427] = A1011*B1212 - A1111*B1202 - A2011*B1211 + A2111*B1201;
    C[428] = A1021*B1212 - A1121*B1202 - A2021*B1211 + A2121*B1201;
    C[429] = A1002*B1212 - A1102*B1202 - A2002*B1211 + A2102*B1201;
    C[430] = A1012*B1212 - A1112*B1202 - A2012*B1211 + A2112*B1201;
    C[431] = A1022*B1212 - A1122*B1202 - A2022*B1211 + A2122*B1201;
    C[432] = A0200*B1212 - A0100*B1222 + A2100*B1220 - A2200*B1210;
    C[433] = A0210*B1212 - A0110*B1222 + A2110*B1220 - A2210*B1210;
    C[434] = A0220*B1212 - A0120*B1222 + A2120*B1220 - A2220*B1210;
    C[435] = A0201*B1212 - A0101*B1222 + A2101*B1220 - A2201*B1210;
    C[436] = A0211*B1212 - A0111*B1222 + A2111*B1220 - A2211*B1210;
    C[437] = A0221*B1212 - A0121*B1222 + A2121*B1220 - A2221*B1210;
    C[438] = A0202*B1212 - A0102*B1222 + A2102*B1220 - A2202*B1210;
    C[439] = A0212*B1212 - A0112*B1222 + A2112*B1220 - A2212*B1210;
    C[440] = A0222*B1212 - A0122*B1222 + A2122*B1220 - A2222*B1210;
    C[441] = A0000*B1222 - A0200*B1202 - A2000*B1220 + A2200*B1200;
    C[442] = A0010*B1222 - A0210*B1202 - A2010*B1220 + A2210*B1200;
    C[443] = A0020*B1222 - A0220*B1202 - A2020*B1220 + A2220*B1200;
    C[444] = A0001*B1222 - A0201*B1202 - A2001*B1220 + A2201*B1200;
    C[445] = A0011*B1222 - A0211*B1202 - A2011*B1220 + A2211*B1200;
    C[446] = A0021*B1222 - A0221*B1202 - A2021*B1220 + A2221*B1200;
    C[447] = A0002*B1222 - A0202*B1202 - A2002*B1220 + A2202*B1200;
    C[448] = A0012*B1222 - A0212*B1202 - A2012*B1220 + A2212*B1200;
    C[449] = A0022*B1222 - A0222*B1202 - A2022*B1220 + A2222*B1200;
    C[450] = A0100*B1202 - A0000*B1212 + A2000*B1210 - A2100*B1200;
    C[451] = A0110*B1202 - A0010*B1212 + A2010*B1210 - A2110*B1200;
    C[452] = A0120*B1202 - A0020*B1212 + A2020*B1210 - A2120*B1200;
    C[453] = A0101*B1202 - A0001*B1212 + A2001*B1210 - A2101*B1200;
    C[454] = A0111*B1202 - A0011*B1212 + A2011*B1210 - A2111*B1200;
    C[455] = A0121*B1202 - A0021*B1212 + A2021*B1210 - A2121*B1200;
    C[456] = A0102*B1202 - A0002*B1212 + A2002*B1210 - A2102*B1200;
    C[457] = A0112*B1202 - A0012*B1212 + A2012*B1210 - A2112*B1200;
    C[458] = A0122*B1202 - A0022*B1212 + A2022*B1210 - A2122*B1200;
    C[459] = A0100*B1221 - A0200*B1211 - A1100*B1220 + A1200*B1210;
    C[460] = A0110*B1221 - A0210*B1211 - A1110*B1220 + A1210*B1210;
    C[461] = A0120*B1221 - A0220*B1211 - A1120*B1220 + A1220*B1210;
    C[462] = A0101*B1221 - A0201*B1211 - A1101*B1220 + A1201*B1210;
    C[463] = A0111*B1221 - A0211*B1211 - A1111*B1220 + A1211*B1210;
    C[464] = A0121*B1221 - A0221*B1211 - A1121*B1220 + A1221*B1210;
    C[465] = A0102*B1221 - A0202*B1211 - A1102*B1220 + A1202*B1210;
    C[466] = A0112*B1221 - A0212*B1211 - A1112*B1220 + A1212*B1210;
    C[467] = A0122*B1221 - A0222*B1211 - A1122*B1220 + A1222*B1210;
    C[468] = A0200*B1201 - A0000*B1221 + A1000*B1220 - A1200*B1200;
    C[469] = A0210*B1201 - A0010*B1221 + A1010*B1220 - A1210*B1200;
    C[470] = A0220*B1201 - A0020*B1221 + A1020*B1220 - A1220*B1200;
    C[471] = A0201*B1201 - A0001*B1221 + A1001*B1220 - A1201*B1200;
    C[472] = A0211*B1201 - A0011*B1221 + A1011*B1220 - A1211*B1200;
    C[473] = A0221*B1201 - A0021*B1221 + A1021*B1220 - A1221*B1200;
    C[474] = A0202*B1201 - A0002*B1221 + A1002*B1220 - A1202*B1200;
    C[475] = A0212*B1201 - A0012*B1221 + A1012*B1220 - A1212*B1200;
    C[476] = A0222*B1201 - A0022*B1221 + A1022*B1220 - A1222*B1200;
    C[477] = A0000*B1211 - A0100*B1201 - A1000*B1210 + A1100*B1200;
    C[478] = A0010*B1211 - A0110*B1201 - A1010*B1210 + A1110*B1200;
    C[479] = A0020*B1211 - A0120*B1201 - A1020*B1210 + A1120*B1200;
    C[480] = A0001*B1211 - A0101*B1201 - A1001*B1210 + A1101*B1200;
    C[481] = A0011*B1211 - A0111*B1201 - A1011*B1210 + A1111*B1200;
    C[482] = A0021*B1211 - A0121*B1201 - A1021*B1210 + A1121*B1200;
    C[483] = A0002*B1211 - A0102*B1201 - A1002*B1210 + A1102*B1200;
    C[484] = A0012*B1211 - A0112*B1201 - A1012*B1210 + A1112*B1200;
    C[485] = A0022*B1211 - A0122*B1201 - A1022*B1210 + A1122*B1200;
    C[486] = A1100*B2022 - A1200*B2012 - A2100*B2021 + A2200*B2011;
    C[487] = A1110*B2022 - A1210*B2012 - A2110*B2021 + A2210*B2011;
    C[488] = A1120*B2022 - A1220*B2012 - A2120*B2021 + A2220*B2011;
    C[489] = A1101*B2022 - A1201*B2012 - A2101*B2021 + A2201*B2011;
    C[490] = A1111*B2022 - A1211*B2012 - A2111*B2021 + A2211*B2011;
    C[491] = A1121*B2022 - A1221*B2012 - A2121*B2021 + A2221*B2011;
    C[492] = A1102*B2022 - A1202*B2012 - A2102*B2021 + A2202*B2011;
    C[493] = A1112*B2022 - A1212*B2012 - A2112*B2021 + A2212*B2011;
    C[494] = A1122*B2022 - A1222*B2012 - A2122*B2021 + A2222*B2011;
    C[495] = A1200*B2002 - A1000*B2022 + A2000*B2021 - A2200*B2001;
    C[496] = A1210*B2002 - A1010*B2022 + A2010*B2021 - A2210*B2001;
    C[497] = A1220*B2002 - A1020*B2022 + A2020*B2021 - A2220*B2001;
    C[498] = A1201*B2002 - A1001*B2022 + A2001*B2021 - A2201*B2001;
    C[499] = A1211*B2002 - A1011*B2022 + A2011*B2021 - A2211*B2001;
    C[500] = A1221*B2002 - A1021*B2022 + A2021*B2021 - A2221*B2001;
    C[501] = A1202*B2002 - A1002*B2022 + A2002*B2021 - A2202*B2001;
    C[502] = A1212*B2002 - A1012*B2022 + A2012*B2021 - A2212*B2001;
    C[503] = A1222*B2002 - A1022*B2022 + A2022*B2021 - A2222*B2001;
    C[504] = A1000*B2012 - A1100*B2002 - A2000*B2011 + A2100*B2001;
    C[505] = A1010*B2012 - A1110*B2002 - A2010*B2011 + A2110*B2001;
    C[506] = A1020*B2012 - A1120*B2002 - A2020*B2011 + A2120*B2001;
    C[507] = A1001*B2012 - A1101*B2002 - A2001*B2011 + A2101*B2001;
    C[508] = A1011*B2012 - A1111*B2002 - A2011*B2011 + A2111*B2001;
    C[509] = A1021*B2012 - A1121*B2002 - A2021*B2011 + A2121*B2001;
    C[510] = A1002*B2012 - A1102*B2002 - A2002*B2011 + A2102*B2001;
    C[511] = A1012*B2012 - A1112*B2002 - A2012*B2011 + A2112*B2001;
    C[512] = A1022*B2012 - A1122*B2002 - A2022*B2011 + A2122*B2001;
    C[513] = A0200*B2012 - A0100*B2022 + A2100*B2020 - A2200*B2010;
    C[514] = A0210*B2012 - A0110*B2022 + A2110*B2020 - A2210*B2010;
    C[515] = A0220*B2012 - A0120*B2022 + A2120*B2020 - A2220*B2010;
    C[516] = A0201*B2012 - A0101*B2022 + A2101*B2020 - A2201*B2010;
    C[517] = A0211*B2012 - A0111*B2022 + A2111*B2020 - A2211*B2010;
    C[518] = A0221*B2012 - A0121*B2022 + A2121*B2020 - A2221*B2010;
    C[519] = A0202*B2012 - A0102*B2022 + A2102*B2020 - A2202*B2010;
    C[520] = A0212*B2012 - A0112*B2022 + A2112*B2020 - A2212*B2010;
    C[521] = A0222*B2012 - A0122*B2022 + A2122*B2020 - A2222*B2010;
    C[522] = A0000*B2022 - A0200*B2002 - A2000*B2020 + A2200*B2000;
    C[523] = A0010*B2022 - A0210*B2002 - A2010*B2020 + A2210*B2000;
    C[524] = A0020*B2022 - A0220*B2002 - A2020*B2020 + A2220*B2000;
    C[525] = A0001*B2022 - A0201*B2002 - A2001*B2020 + A2201*B2000;
    C[526] = A0011*B2022 - A0211*B2002 - A2011*B2020 + A2211*B2000;
    C[527] = A0021*B2022 - A0221*B2002 - A2021*B2020 + A2221*B2000;
    C[528] = A0002*B2022 - A0202*B2002 - A2002*B2020 + A2202*B2000;
    C[529] = A0012*B2022 - A0212*B2002 - A2012*B2020 + A2212*B2000;
    C[530] = A0022*B2022 - A0222*B2002 - A2022*B2020 + A2222*B2000;
    C[531] = A0100*B2002 - A0000*B2012 + A2000*B2010 - A2100*B2000;
    C[532] = A0110*B2002 - A0010*B2012 + A2010*B2010 - A2110*B2000;
    C[533] = A0120*B2002 - A0020*B2012 + A2020*B2010 - A2120*B2000;
    C[534] = A0101*B2002 - A0001*B2012 + A2001*B2010 - A2101*B2000;
    C[535] = A0111*B2002 - A0011*B2012 + A2011*B2010 - A2111*B2000;
    C[536] = A0121*B2002 - A0021*B2012 + A2021*B2010 - A2121*B2000;
    C[537] = A0102*B2002 - A0002*B2012 + A2002*B2010 - A2102*B2000;
    C[538] = A0112*B2002 - A0012*B2012 + A2012*B2010 - A2112*B2000;
    C[539] = A0122*B2002 - A0022*B2012 + A2022*B2010 - A2122*B2000;
    C[540] = A0100*B2021 - A0200*B2011 - A1100*B2020 + A1200*B2010;
    C[541] = A0110*B2021 - A0210*B2011 - A1110*B2020 + A1210*B2010;
    C[542] = A0120*B2021 - A0220*B2011 - A1120*B2020 + A1220*B2010;
    C[543] = A0101*B2021 - A0201*B2011 - A1101*B2020 + A1201*B2010;
    C[544] = A0111*B2021 - A0211*B2011 - A1111*B2020 + A1211*B2010;
    C[545] = A0121*B2021 - A0221*B2011 - A1121*B2020 + A1221*B2010;
    C[546] = A0102*B2021 - A0202*B2011 - A1102*B2020 + A1202*B2010;
    C[547] = A0112*B2021 - A0212*B2011 - A1112*B2020 + A1212*B2010;
    C[548] = A0122*B2021 - A0222*B2011 - A1122*B2020 + A1222*B2010;
    C[549] = A0200*B2001 - A0000*B2021 + A1000*B2020 - A1200*B2000;
    C[550] = A0210*B2001 - A0010*B2021 + A1010*B2020 - A1210*B2000;
    C[551] = A0220*B2001 - A0020*B2021 + A1020*B2020 - A1220*B2000;
    C[552] = A0201*B2001 - A0001*B2021 + A1001*B2020 - A1201*B2000;
    C[553] = A0211*B2001 - A0011*B2021 + A1011*B2020 - A1211*B2000;
    C[554] = A0221*B2001 - A0021*B2021 + A1021*B2020 - A1221*B2000;
    C[555] = A0202*B2001 - A0002*B2021 + A1002*B2020 - A1202*B2000;
    C[556] = A0212*B2001 - A0012*B2021 + A1012*B2020 - A1212*B2000;
    C[557] = A0222*B2001 - A0022*B2021 + A1022*B2020 - A1222*B2000;
    C[558] = A0000*B2011 - A0100*B2001 - A1000*B2010 + A1100*B2000;
    C[559] = A0010*B2011 - A0110*B2001 - A1010*B2010 + A1110*B2000;
    C[560] = A0020*B2011 - A0120*B2001 - A1020*B2010 + A1120*B2000;
    C[561] = A0001*B2011 - A0101*B2001 - A1001*B2010 + A1101*B2000;
    C[562] = A0011*B2011 - A0111*B2001 - A1011*B2010 + A1111*B2000;
    C[563] = A0021*B2011 - A0121*B2001 - A1021*B2010 + A1121*B2000;
    C[564] = A0002*B2011 - A0102*B2001 - A1002*B2010 + A1102*B2000;
    C[565] = A0012*B2011 - A0112*B2001 - A1012*B2010 + A1112*B2000;
    C[566] = A0022*B2011 - A0122*B2001 - A1022*B2010 + A1122*B2000;
    C[567] = A1100*B2122 - A1200*B2112 - A2100*B2121 + A2200*B2111;
    C[568] = A1110*B2122 - A1210*B2112 - A2110*B2121 + A2210*B2111;
    C[569] = A1120*B2122 - A1220*B2112 - A2120*B2121 + A2220*B2111;
    C[570] = A1101*B2122 - A1201*B2112 - A2101*B2121 + A2201*B2111;
    C[571] = A1111*B2122 - A1211*B2112 - A2111*B2121 + A2211*B2111;
    C[572] = A1121*B2122 - A1221*B2112 - A2121*B2121 + A2221*B2111;
    C[573] = A1102*B2122 - A1202*B2112 - A2102*B2121 + A2202*B2111;
    C[574] = A1112*B2122 - A1212*B2112 - A2112*B2121 + A2212*B2111;
    C[575] = A1122*B2122 - A1222*B2112 - A2122*B2121 + A2222*B2111;
    C[576] = A1200*B2102 - A1000*B2122 + A2000*B2121 - A2200*B2101;
    C[577] = A1210*B2102 - A1010*B2122 + A2010*B2121 - A2210*B2101;
    C[578] = A1220*B2102 - A1020*B2122 + A2020*B2121 - A2220*B2101;
    C[579] = A1201*B2102 - A1001*B2122 + A2001*B2121 - A2201*B2101;
    C[580] = A1211*B2102 - A1011*B2122 + A2011*B2121 - A2211*B2101;
    C[581] = A1221*B2102 - A1021*B2122 + A2021*B2121 - A2221*B2101;
    C[582] = A1202*B2102 - A1002*B2122 + A2002*B2121 - A2202*B2101;
    C[583] = A1212*B2102 - A1012*B2122 + A2012*B2121 - A2212*B2101;
    C[584] = A1222*B2102 - A1022*B2122 + A2022*B2121 - A2222*B2101;
    C[585] = A1000*B2112 - A1100*B2102 - A2000*B2111 + A2100*B2101;
    C[586] = A1010*B2112 - A1110*B2102 - A2010*B2111 + A2110*B2101;
    C[587] = A1020*B2112 - A1120*B2102 - A2020*B2111 + A2120*B2101;
    C[588] = A1001*B2112 - A1101*B2102 - A2001*B2111 + A2101*B2101;
    C[589] = A1011*B2112 - A1111*B2102 - A2011*B2111 + A2111*B2101;
    C[590] = A1021*B2112 - A1121*B2102 - A2021*B2111 + A2121*B2101;
    C[591] = A1002*B2112 - A1102*B2102 - A2002*B2111 + A2102*B2101;
    C[592] = A1012*B2112 - A1112*B2102 - A2012*B2111 + A2112*B2101;
    C[593] = A1022*B2112 - A1122*B2102 - A2022*B2111 + A2122*B2101;
    C[594] = A0200*B2112 - A0100*B2122 + A2100*B2120 - A2200*B2110;
    C[595] = A0210*B2112 - A0110*B2122 + A2110*B2120 - A2210*B2110;
    C[596] = A0220*B2112 - A0120*B2122 + A2120*B2120 - A2220*B2110;
    C[597] = A0201*B2112 - A0101*B2122 + A2101*B2120 - A2201*B2110;
    C[598] = A0211*B2112 - A0111*B2122 + A2111*B2120 - A2211*B2110;
    C[599] = A0221*B2112 - A0121*B2122 + A2121*B2120 - A2221*B2110;
    C[600] = A0202*B2112 - A0102*B2122 + A2102*B2120 - A2202*B2110;
    C[601] = A0212*B2112 - A0112*B2122 + A2112*B2120 - A2212*B2110;
    C[602] = A0222*B2112 - A0122*B2122 + A2122*B2120 - A2222*B2110;
    C[603] = A0000*B2122 - A0200*B2102 - A2000*B2120 + A2200*B2100;
    C[604] = A0010*B2122 - A0210*B2102 - A2010*B2120 + A2210*B2100;
    C[605] = A0020*B2122 - A0220*B2102 - A2020*B2120 + A2220*B2100;
    C[606] = A0001*B2122 - A0201*B2102 - A2001*B2120 + A2201*B2100;
    C[607] = A0011*B2122 - A0211*B2102 - A2011*B2120 + A2211*B2100;
    C[608] = A0021*B2122 - A0221*B2102 - A2021*B2120 + A2221*B2100;
    C[609] = A0002*B2122 - A0202*B2102 - A2002*B2120 + A2202*B2100;
    C[610] = A0012*B2122 - A0212*B2102 - A2012*B2120 + A2212*B2100;
    C[611] = A0022*B2122 - A0222*B2102 - A2022*B2120 + A2222*B2100;
    C[612] = A0100*B2102 - A0000*B2112 + A2000*B2110 - A2100*B2100;
    C[613] = A0110*B2102 - A0010*B2112 + A2010*B2110 - A2110*B2100;
    C[614] = A0120*B2102 - A0020*B2112 + A2020*B2110 - A2120*B2100;
    C[615] = A0101*B2102 - A0001*B2112 + A2001*B2110 - A2101*B2100;
    C[616] = A0111*B2102 - A0011*B2112 + A2011*B2110 - A2111*B2100;
    C[617] = A0121*B2102 - A0021*B2112 + A2021*B2110 - A2121*B2100;
    C[618] = A0102*B2102 - A0002*B2112 + A2002*B2110 - A2102*B2100;
    C[619] = A0112*B2102 - A0012*B2112 + A2012*B2110 - A2112*B2100;
    C[620] = A0122*B2102 - A0022*B2112 + A2022*B2110 - A2122*B2100;
    C[621] = A0100*B2121 - A0200*B2111 - A1100*B2120 + A1200*B2110;
    C[622] = A0110*B2121 - A0210*B2111 - A1110*B2120 + A1210*B2110;
    C[623] = A0120*B2121 - A0220*B2111 - A1120*B2120 + A1220*B2110;
    C[624] = A0101*B2121 - A0201*B2111 - A1101*B2120 + A1201*B2110;
    C[625] = A0111*B2121 - A0211*B2111 - A1111*B2120 + A1211*B2110;
    C[626] = A0121*B2121 - A0221*B2111 - A1121*B2120 + A1221*B2110;
    C[627] = A0102*B2121 - A0202*B2111 - A1102*B2120 + A1202*B2110;
    C[628] = A0112*B2121 - A0212*B2111 - A1112*B2120 + A1212*B2110;
    C[629] = A0122*B2121 - A0222*B2111 - A1122*B2120 + A1222*B2110;
    C[630] = A0200*B2101 - A0000*B2121 + A1000*B2120 - A1200*B2100;
    C[631] = A0210*B2101 - A0010*B2121 + A1010*B2120 - A1210*B2100;
    C[632] = A0220*B2101 - A0020*B2121 + A1020*B2120 - A1220*B2100;
    C[633] = A0201*B2101 - A0001*B2121 + A1001*B2120 - A1201*B2100;
    C[634] = A0211*B2101 - A0011*B2121 + A1011*B2120 - A1211*B2100;
    C[635] = A0221*B2101 - A0021*B2121 + A1021*B2120 - A1221*B2100;
    C[636] = A0202*B2101 - A0002*B2121 + A1002*B2120 - A1202*B2100;
    C[637] = A0212*B2101 - A0012*B2121 + A1012*B2120 - A1212*B2100;
    C[638] = A0222*B2101 - A0022*B2121 + A1022*B2120 - A1222*B2100;
    C[639] = A0000*B2111 - A0100*B2101 - A1000*B2110 + A1100*B2100;
    C[640] = A0010*B2111 - A0110*B2101 - A1010*B2110 + A1110*B2100;
    C[641] = A0020*B2111 - A0120*B2101 - A1020*B2110 + A1120*B2100;
    C[642] = A0001*B2111 - A0101*B2101 - A1001*B2110 + A1101*B2100;
    C[643] = A0011*B2111 - A0111*B2101 - A1011*B2110 + A1111*B2100;
    C[644] = A0021*B2111 - A0121*B2101 - A1021*B2110 + A1121*B2100;
    C[645] = A0002*B2111 - A0102*B2101 - A1002*B2110 + A1102*B2100;
    C[646] = A0012*B2111 - A0112*B2101 - A1012*B2110 + A1112*B2100;
    C[647] = A0022*B2111 - A0122*B2101 - A1022*B2110 + A1122*B2100;
    C[648] = A1100*B2222 - A1200*B2212 - A2100*B2221 + A2200*B2211;
    C[649] = A1110*B2222 - A1210*B2212 - A2110*B2221 + A2210*B2211;
    C[650] = A1120*B2222 - A1220*B2212 - A2120*B2221 + A2220*B2211;
    C[651] = A1101*B2222 - A1201*B2212 - A2101*B2221 + A2201*B2211;
    C[652] = A1111*B2222 - A1211*B2212 - A2111*B2221 + A2211*B2211;
    C[653] = A1121*B2222 - A1221*B2212 - A2121*B2221 + A2221*B2211;
    C[654] = A1102*B2222 - A1202*B2212 - A2102*B2221 + A2202*B2211;
    C[655] = A1112*B2222 - A1212*B2212 - A2112*B2221 + A2212*B2211;
    C[656] = A1122*B2222 - A1222*B2212 - A2122*B2221 + A2222*B2211;
    C[657] = A1200*B2202 - A1000*B2222 + A2000*B2221 - A2200*B2201;
    C[658] = A1210*B2202 - A1010*B2222 + A2010*B2221 - A2210*B2201;
    C[659] = A1220*B2202 - A1020*B2222 + A2020*B2221 - A2220*B2201;
    C[660] = A1201*B2202 - A1001*B2222 + A2001*B2221 - A2201*B2201;
    C[661] = A1211*B2202 - A1011*B2222 + A2011*B2221 - A2211*B2201;
    C[662] = A1221*B2202 - A1021*B2222 + A2021*B2221 - A2221*B2201;
    C[663] = A1202*B2202 - A1002*B2222 + A2002*B2221 - A2202*B2201;
    C[664] = A1212*B2202 - A1012*B2222 + A2012*B2221 - A2212*B2201;
    C[665] = A1222*B2202 - A1022*B2222 + A2022*B2221 - A2222*B2201;
    C[666] = A1000*B2212 - A1100*B2202 - A2000*B2211 + A2100*B2201;
    C[667] = A1010*B2212 - A1110*B2202 - A2010*B2211 + A2110*B2201;
    C[668] = A1020*B2212 - A1120*B2202 - A2020*B2211 + A2120*B2201;
    C[669] = A1001*B2212 - A1101*B2202 - A2001*B2211 + A2101*B2201;
    C[670] = A1011*B2212 - A1111*B2202 - A2011*B2211 + A2111*B2201;
    C[671] = A1021*B2212 - A1121*B2202 - A2021*B2211 + A2121*B2201;
    C[672] = A1002*B2212 - A1102*B2202 - A2002*B2211 + A2102*B2201;
    C[673] = A1012*B2212 - A1112*B2202 - A2012*B2211 + A2112*B2201;
    C[674] = A1022*B2212 - A1122*B2202 - A2022*B2211 + A2122*B2201;
    C[675] = A0200*B2212 - A0100*B2222 + A2100*B2220 - A2200*B2210;
    C[676] = A0210*B2212 - A0110*B2222 + A2110*B2220 - A2210*B2210;
    C[677] = A0220*B2212 - A0120*B2222 + A2120*B2220 - A2220*B2210;
    C[678] = A0201*B2212 - A0101*B2222 + A2101*B2220 - A2201*B2210;
    C[679] = A0211*B2212 - A0111*B2222 + A2111*B2220 - A2211*B2210;
    C[680] = A0221*B2212 - A0121*B2222 + A2121*B2220 - A2221*B2210;
    C[681] = A0202*B2212 - A0102*B2222 + A2102*B2220 - A2202*B2210;
    C[682] = A0212*B2212 - A0112*B2222 + A2112*B2220 - A2212*B2210;
    C[683] = A0222*B2212 - A0122*B2222 + A2122*B2220 - A2222*B2210;
    C[684] = A0000*B2222 - A0200*B2202 - A2000*B2220 + A2200*B2200;
    C[685] = A0010*B2222 - A0210*B2202 - A2010*B2220 + A2210*B2200;
    C[686] = A0020*B2222 - A0220*B2202 - A2020*B2220 + A2220*B2200;
    C[687] = A0001*B2222 - A0201*B2202 - A2001*B2220 + A2201*B2200;
    C[688] = A0011*B2222 - A0211*B2202 - A2011*B2220 + A2211*B2200;
    C[689] = A0021*B2222 - A0221*B2202 - A2021*B2220 + A2221*B2200;
    C[690] = A0002*B2222 - A0202*B2202 - A2002*B2220 + A2202*B2200;
    C[691] = A0012*B2222 - A0212*B2202 - A2012*B2220 + A2212*B2200;
    C[692] = A0022*B2222 - A0222*B2202 - A2022*B2220 + A2222*B2200;
    C[693] = A0100*B2202 - A0000*B2212 + A2000*B2210 - A2100*B2200;
    C[694] = A0110*B2202 - A0010*B2212 + A2010*B2210 - A2110*B2200;
    C[695] = A0120*B2202 - A0020*B2212 + A2020*B2210 - A2120*B2200;
    C[696] = A0101*B2202 - A0001*B2212 + A2001*B2210 - A2101*B2200;
    C[697] = A0111*B2202 - A0011*B2212 + A2011*B2210 - A2111*B2200;
    C[698] = A0121*B2202 - A0021*B2212 + A2021*B2210 - A2121*B2200;
    C[699] = A0102*B2202 - A0002*B2212 + A2002*B2210 - A2102*B2200;
    C[700] = A0112*B2202 - A0012*B2212 + A2012*B2210 - A2112*B2200;
    C[701] = A0122*B2202 - A0022*B2212 + A2022*B2210 - A2122*B2200;
    C[702] = A0100*B2221 - A0200*B2211 - A1100*B2220 + A1200*B2210;
    C[703] = A0110*B2221 - A0210*B2211 - A1110*B2220 + A1210*B2210;
    C[704] = A0120*B2221 - A0220*B2211 - A1120*B2220 + A1220*B2210;
    C[705] = A0101*B2221 - A0201*B2211 - A1101*B2220 + A1201*B2210;
    C[706] = A0111*B2221 - A0211*B2211 - A1111*B2220 + A1211*B2210;
    C[707] = A0121*B2221 - A0221*B2211 - A1121*B2220 + A1221*B2210;
    C[708] = A0102*B2221 - A0202*B2211 - A1102*B2220 + A1202*B2210;
    C[709] = A0112*B2221 - A0212*B2211 - A1112*B2220 + A1212*B2210;
    C[710] = A0122*B2221 - A0222*B2211 - A1122*B2220 + A1222*B2210;
    C[711] = A0200*B2201 - A0000*B2221 + A1000*B2220 - A1200*B2200;
    C[712] = A0210*B2201 - A0010*B2221 + A1010*B2220 - A1210*B2200;
    C[713] = A0220*B2201 - A0020*B2221 + A1020*B2220 - A1220*B2200;
    C[714] = A0201*B2201 - A0001*B2221 + A1001*B2220 - A1201*B2200;
    C[715] = A0211*B2201 - A0011*B2221 + A1011*B2220 - A1211*B2200;
    C[716] = A0221*B2201 - A0021*B2221 + A1021*B2220 - A1221*B2200;
    C[717] = A0202*B2201 - A0002*B2221 + A1002*B2220 - A1202*B2200;
    C[718] = A0212*B2201 - A0012*B2221 + A1012*B2220 - A1212*B2200;
    C[719] = A0222*B2201 - A0022*B2221 + A1022*B2220 - A1222*B2200;
    C[720] = A0000*B2211 - A0100*B2201 - A1000*B2210 + A1100*B2200;
    C[721] = A0010*B2211 - A0110*B2201 - A1010*B2210 + A1110*B2200;
    C[722] = A0020*B2211 - A0120*B2201 - A1020*B2210 + A1120*B2200;
    C[723] = A0001*B2211 - A0101*B2201 - A1001*B2210 + A1101*B2200;
    C[724] = A0011*B2211 - A0111*B2201 - A1011*B2210 + A1111*B2200;
    C[725] = A0021*B2211 - A0121*B2201 - A1021*B2210 + A1121*B2200;
    C[726] = A0002*B2211 - A0102*B2201 - A1002*B2210 + A1102*B2200;
    C[727] = A0012*B2211 - A0112*B2201 - A1012*B2210 + A1112*B2200;
    C[728] = A0022*B2211 - A0122*B2201 - A1022*B2210 + A1122*B2200;
}




#endif // TENSOR_CROSS_H

