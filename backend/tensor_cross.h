#ifndef TENSOR_CROSS_H
#define TENSOR_CROSS_H

#include "commons/commons.h"
#include "extended_intrinsics/extintrin.h"

namespace Fastor {

template<typename T, size_t M, size_t K, size_t N>
inline void _crossproduct(const T *FASTOR_RESTRICT a, const T *FASTOR_RESTRICT b, T *FASTOR_RESTRICT c) {
    assert(false && "CROSS PRODUCT IS ONLY A 3D OPERATOR");
}

#ifdef __SSE4_2__

template<>
FASTOR_INLINE void _crossproduct<double,2,2,2>(const double *FASTOR_RESTRICT a, const double *FASTOR_RESTRICT b, double *FASTOR_RESTRICT c) {
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
#ifdef __AVX__
    _mm256_store_pd(c,VZEROPD);
    _mm256_store_pd(c+4,VZEROPD);
#else
    _mm_store_pd(c,ZEROPD);
    _mm_store_pd(c+2,ZEROPD);
    _mm_store_pd(c+4,ZEROPD);
    _mm_store_pd(c+6,ZEROPD);
#endif
}

template<>
FASTOR_INLINE void _crossproduct<double,3,3,3>(const double *FASTOR_RESTRICT a, const double *FASTOR_RESTRICT b, double *FASTOR_RESTRICT c) {
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
FASTOR_INLINE void _crossproduct<float,2,2,2>(const float *FASTOR_RESTRICT a, const float *FASTOR_RESTRICT b, float *FASTOR_RESTRICT c) {
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
#ifdef __AVX__
    _mm256_store_ps(c,VZEROPS);
#else
    _mm_store_ps(c,ZEROPS);
    _mm_store_ps(c+2,ZEROPS);
#endif
}

template<>
FASTOR_INLINE void _crossproduct<float,3,3,3>(const float *FASTOR_RESTRICT a, const float *FASTOR_RESTRICT b, float *FASTOR_RESTRICT c) {
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
void _crossproduct(const T *FASTOR_RESTRICT a, const T *FASTOR_RESTRICT b, T *FASTOR_RESTRICT c);

template<>
FASTOR_INLINE void _crossproduct<double,PlaneStrain>(const double *FASTOR_RESTRICT a, const double *FASTOR_RESTRICT b, double *FASTOR_RESTRICT c) {
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
#ifdef __AVX__
    _mm256_store_pd(c,VZEROPD);
#else
    _mm_store_pd(c,ZEROPD);
    _mm_store_pd(c+2,ZEROPD);
#endif
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
FASTOR_INLINE void _crossproduct<float,PlaneStrain>(const float *FASTOR_RESTRICT a, const float *FASTOR_RESTRICT b, float *FASTOR_RESTRICT c) {
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
#ifdef __AVX__
    _mm256_store_ps(c,VZEROPS);
#else
    _mm_store_ps(c,ZEROPS);
    _mm_store_ps(c+2,ZEROPS);
#endif
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
FASTOR_HINT_INLINE void _crossproduct<float,3,1,3>(const float *FASTOR_RESTRICT a, const float *FASTOR_RESTRICT b, float *FASTOR_RESTRICT c) {
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
FASTOR_HINT_INLINE void _crossproduct<float,2,1,2>(const float *FASTOR_RESTRICT a, const float *FASTOR_RESTRICT b, float *FASTOR_RESTRICT c) {
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
FASTOR_HINT_INLINE void _crossproduct<double,3,1,3>(const double *FASTOR_RESTRICT a, const double *FASTOR_RESTRICT b, double *FASTOR_RESTRICT c) {
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
FASTOR_HINT_INLINE void _crossproduct<double,2,1,2>(const double *FASTOR_RESTRICT a, const double *FASTOR_RESTRICT b, double *FASTOR_RESTRICT c) {
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
FASTOR_HINT_INLINE void _crossproduct<float,3,3,1>(const float *FASTOR_RESTRICT a, const float *FASTOR_RESTRICT b, float *FASTOR_RESTRICT c) {
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
FASTOR_HINT_INLINE void _crossproduct<float,2,2,1>(const float *FASTOR_RESTRICT a, const float *FASTOR_RESTRICT b, float *FASTOR_RESTRICT c) {
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
FASTOR_HINT_INLINE void _crossproduct<double,3,3,1>(const double *FASTOR_RESTRICT a, const double *FASTOR_RESTRICT b, double *FASTOR_RESTRICT c) {
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
FASTOR_HINT_INLINE void _crossproduct<double,2,2,1>(const double *FASTOR_RESTRICT a, const double *FASTOR_RESTRICT b, double *FASTOR_RESTRICT c) {
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
FASTOR_HINT_INLINE void _crossproduct(const T *FASTOR_RESTRICT A, const T *FASTOR_RESTRICT B, T *FASTOR_RESTRICT C) {

    T _A000 = A[0];
    T _A001 = A[1];
    T _A002 = A[2];
    T _A010 = A[3];
    T _A011 = A[4];
    T _A012 = A[5];
    T _A020 = A[6];
    T _A021 = A[7];
    T _A022 = A[8];
    T _A100 = A[9];
    T _A101 = A[10];
    T _A102 = A[11];
    T _A110 = A[12];
    T _A111 = A[13];
    T _A112 = A[14];
    T _A120 = A[15];
    T _A121 = A[16];
    T _A122 = A[17];
    T _A200 = A[18];
    T _A201 = A[19];
    T _A202 = A[20];
    T _A210 = A[21];
    T _A211 = A[22];
    T _A212 = A[23];
    T _A220 = A[24];
    T _A221 = A[25];
    T _A222 = A[26];


    T _B000 = B[0];
    T _B001 = B[1];
    T _B002 = B[2];
    T _B010 = B[3];
    T _B011 = B[4];
    T _B012 = B[5];
    T _B020 = B[6];
    T _B021 = B[7];
    T _B022 = B[8];
    T _B100 = B[9];
    T _B101 = B[10];
    T _B102 = B[11];
    T _B110 = B[12];
    T _B111 = B[13];
    T _B112 = B[14];
    T _B120 = B[15];
    T _B121 = B[16];
    T _B122 = B[17];
    T _B200 = B[18];
    T _B201 = B[19];
    T _B202 = B[20];
    T _B210 = B[21];
    T _B211 = B[22];
    T _B212 = B[23];
    T _B220 = B[24];
    T _B221 = B[25];
    T _B222 = B[26];


    C[0] = _A101*_B022 - _A102*_B012 + _A202*_B011 - _A201*_B021;
    C[1] = _A111*_B022 - _A112*_B012 + _A212*_B011 - _A211*_B021;
    C[2] = _A121*_B022 - _A122*_B012 + _A222*_B011 - _A221*_B021;
    C[3] = _A102*_B002 - _A100*_B022 - _A202*_B001 + _A200*_B021;
    C[4] = _A112*_B002 - _A110*_B022 - _A212*_B001 + _A210*_B021;
    C[5] = _A122*_B002 - _A120*_B022 - _A222*_B001 + _A220*_B021;
    C[6] = _A100*_B012 - _A101*_B002 + _A201*_B001 - _A200*_B011;
    C[7] = _A110*_B012 - _A111*_B002 + _A211*_B001 - _A210*_B011;
    C[8] = _A120*_B012 - _A121*_B002 + _A221*_B001 - _A220*_B011;
    C[9] = _A002*_B012 - _A001*_B022 - _A202*_B010 + _A201*_B020;
    C[10] = _A012*_B012 - _A011*_B022 - _A212*_B010 + _A211*_B020;
    C[11] = _A022*_B012 - _A021*_B022 - _A222*_B010 + _A221*_B020;
    C[12] = _A000*_B022 - _A002*_B002 + _A202*_B000 - _A200*_B020;
    C[13] = _A010*_B022 - _A012*_B002 + _A212*_B000 - _A210*_B020;
    C[14] = _A020*_B022 - _A022*_B002 + _A222*_B000 - _A220*_B020;
    C[15] = _A001*_B002 - _A000*_B012 - _A201*_B000 + _A200*_B010;
    C[16] = _A011*_B002 - _A010*_B012 - _A211*_B000 + _A210*_B010;
    C[17] = _A021*_B002 - _A020*_B012 - _A221*_B000 + _A220*_B010;
    C[18] = _A001*_B021 - _A002*_B011 + _A102*_B010 - _A101*_B020;
    C[19] = _A011*_B021 - _A012*_B011 + _A112*_B010 - _A111*_B020;
    C[20] = _A021*_B021 - _A022*_B011 + _A122*_B010 - _A121*_B020;
    C[21] = _A002*_B001 - _A000*_B021 - _A102*_B000 + _A100*_B020;
    C[22] = _A012*_B001 - _A010*_B021 - _A112*_B000 + _A110*_B020;
    C[23] = _A022*_B001 - _A020*_B021 - _A122*_B000 + _A120*_B020;
    C[24] = _A000*_B011 - _A001*_B001 + _A101*_B000 - _A100*_B010;
    C[25] = _A010*_B011 - _A011*_B001 + _A111*_B000 - _A110*_B010;
    C[26] = _A020*_B011 - _A021*_B001 + _A121*_B000 - _A120*_B010;
    C[27] = _A101*_B122 - _A102*_B112 + _A202*_B111 - _A201*_B121;
    C[28] = _A111*_B122 - _A112*_B112 + _A212*_B111 - _A211*_B121;
    C[29] = _A121*_B122 - _A122*_B112 + _A222*_B111 - _A221*_B121;
    C[30] = _A102*_B102 - _A100*_B122 - _A202*_B101 + _A200*_B121;
    C[31] = _A112*_B102 - _A110*_B122 - _A212*_B101 + _A210*_B121;
    C[32] = _A122*_B102 - _A120*_B122 - _A222*_B101 + _A220*_B121;
    C[33] = _A100*_B112 - _A101*_B102 + _A201*_B101 - _A200*_B111;
    C[34] = _A110*_B112 - _A111*_B102 + _A211*_B101 - _A210*_B111;
    C[35] = _A120*_B112 - _A121*_B102 + _A221*_B101 - _A220*_B111;
    C[36] = _A002*_B112 - _A001*_B122 - _A202*_B110 + _A201*_B120;
    C[37] = _A012*_B112 - _A011*_B122 - _A212*_B110 + _A211*_B120;
    C[38] = _A022*_B112 - _A021*_B122 - _A222*_B110 + _A221*_B120;
    C[39] = _A000*_B122 - _A002*_B102 + _A202*_B100 - _A200*_B120;
    C[40] = _A010*_B122 - _A012*_B102 + _A212*_B100 - _A210*_B120;
    C[41] = _A020*_B122 - _A022*_B102 + _A222*_B100 - _A220*_B120;
    C[42] = _A001*_B102 - _A000*_B112 - _A201*_B100 + _A200*_B110;
    C[43] = _A011*_B102 - _A010*_B112 - _A211*_B100 + _A210*_B110;
    C[44] = _A021*_B102 - _A020*_B112 - _A221*_B100 + _A220*_B110;
    C[45] = _A001*_B121 - _A002*_B111 + _A102*_B110 - _A101*_B120;
    C[46] = _A011*_B121 - _A012*_B111 + _A112*_B110 - _A111*_B120;
    C[47] = _A021*_B121 - _A022*_B111 + _A122*_B110 - _A121*_B120;
    C[48] = _A002*_B101 - _A000*_B121 - _A102*_B100 + _A100*_B120;
    C[49] = _A012*_B101 - _A010*_B121 - _A112*_B100 + _A110*_B120;
    C[50] = _A022*_B101 - _A020*_B121 - _A122*_B100 + _A120*_B120;
    C[51] = _A000*_B111 - _A001*_B101 + _A101*_B100 - _A100*_B110;
    C[52] = _A010*_B111 - _A011*_B101 + _A111*_B100 - _A110*_B110;
    C[53] = _A020*_B111 - _A021*_B101 + _A121*_B100 - _A120*_B110;
    C[54] = _A101*_B222 - _A102*_B212 + _A202*_B211 - _A201*_B221;
    C[55] = _A111*_B222 - _A112*_B212 + _A212*_B211 - _A211*_B221;
    C[56] = _A121*_B222 - _A122*_B212 + _A222*_B211 - _A221*_B221;
    C[57] = _A102*_B202 - _A100*_B222 - _A202*_B201 + _A200*_B221;
    C[58] = _A112*_B202 - _A110*_B222 - _A212*_B201 + _A210*_B221;
    C[59] = _A122*_B202 - _A120*_B222 - _A222*_B201 + _A220*_B221;
    C[60] = _A100*_B212 - _A101*_B202 + _A201*_B201 - _A200*_B211;
    C[61] = _A110*_B212 - _A111*_B202 + _A211*_B201 - _A210*_B211;
    C[62] = _A120*_B212 - _A121*_B202 + _A221*_B201 - _A220*_B211;
    C[63] = _A002*_B212 - _A001*_B222 - _A202*_B210 + _A201*_B220;
    C[64] = _A012*_B212 - _A011*_B222 - _A212*_B210 + _A211*_B220;
    C[65] = _A022*_B212 - _A021*_B222 - _A222*_B210 + _A221*_B220;
    C[66] = _A000*_B222 - _A002*_B202 + _A202*_B200 - _A200*_B220;
    C[67] = _A010*_B222 - _A012*_B202 + _A212*_B200 - _A210*_B220;
    C[68] = _A020*_B222 - _A022*_B202 + _A222*_B200 - _A220*_B220;
    C[69] = _A001*_B202 - _A000*_B212 - _A201*_B200 + _A200*_B210;
    C[70] = _A011*_B202 - _A010*_B212 - _A211*_B200 + _A210*_B210;
    C[71] = _A021*_B202 - _A020*_B212 - _A221*_B200 + _A220*_B210;
    C[72] = _A001*_B221 - _A002*_B211 + _A102*_B210 - _A101*_B220;
    C[73] = _A011*_B221 - _A012*_B211 + _A112*_B210 - _A111*_B220;
    C[74] = _A021*_B221 - _A022*_B211 + _A122*_B210 - _A121*_B220;
    C[75] = _A002*_B201 - _A000*_B221 - _A102*_B200 + _A100*_B220;
    C[76] = _A012*_B201 - _A010*_B221 - _A112*_B200 + _A110*_B220;
    C[77] = _A022*_B201 - _A020*_B221 - _A122*_B200 + _A120*_B220;
    C[78] = _A000*_B211 - _A001*_B201 + _A101*_B200 - _A100*_B210;
    C[79] = _A010*_B211 - _A011*_B201 + _A111*_B200 - _A110*_B210;
    C[80] = _A020*_B211 - _A021*_B201 + _A121*_B200 - _A120*_B210;

}



////////
//(AxB)_{pPiIqQ} = E_{ijk}E_{IJK}A_{pPjJ}B_{kKqQ}
// tensor cross product of 4th order tensors
template<typename T, size_t I, size_t J, size_t K, size_t L, size_t M, size_t N, size_t O, size_t P>
FASTOR_HINT_INLINE void _crossproduct(const T *FASTOR_RESTRICT A, const T *FASTOR_RESTRICT B, T *FASTOR_RESTRICT C) {

    T _A0000 = A[0];
    T _A0001 = A[1];
    T _A0002 = A[2];
    T _A0010 = A[3];
    T _A0011 = A[4];
    T _A0012 = A[5];
    T _A0020 = A[6];
    T _A0021 = A[7];
    T _A0022 = A[8];
    T _A0100 = A[9];
    T _A0101 = A[10];
    T _A0102 = A[11];
    T _A0110 = A[12];
    T _A0111 = A[13];
    T _A0112 = A[14];
    T _A0120 = A[15];
    T _A0121 = A[16];
    T _A0122 = A[17];
    T _A0200 = A[18];
    T _A0201 = A[19];
    T _A0202 = A[20];
    T _A0210 = A[21];
    T _A0211 = A[22];
    T _A0212 = A[23];
    T _A0220 = A[24];
    T _A0221 = A[25];
    T _A0222 = A[26];
    T _A1000 = A[27];
    T _A1001 = A[28];
    T _A1002 = A[29];
    T _A1010 = A[30];
    T _A1011 = A[31];
    T _A1012 = A[32];
    T _A1020 = A[33];
    T _A1021 = A[34];
    T _A1022 = A[35];
    T _A1100 = A[36];
    T _A1101 = A[37];
    T _A1102 = A[38];
    T _A1110 = A[39];
    T _A1111 = A[40];
    T _A1112 = A[41];
    T _A1120 = A[42];
    T _A1121 = A[43];
    T _A1122 = A[44];
    T _A1200 = A[45];
    T _A1201 = A[46];
    T _A1202 = A[47];
    T _A1210 = A[48];
    T _A1211 = A[49];
    T _A1212 = A[50];
    T _A1220 = A[51];
    T _A1221 = A[52];
    T _A1222 = A[53];
    T _A2000 = A[54];
    T _A2001 = A[55];
    T _A2002 = A[56];
    T _A2010 = A[57];
    T _A2011 = A[58];
    T _A2012 = A[59];
    T _A2020 = A[60];
    T _A2021 = A[61];
    T _A2022 = A[62];
    T _A2100 = A[63];
    T _A2101 = A[64];
    T _A2102 = A[65];
    T _A2110 = A[66];
    T _A2111 = A[67];
    T _A2112 = A[68];
    T _A2120 = A[69];
    T _A2121 = A[70];
    T _A2122 = A[71];
    T _A2200 = A[72];
    T _A2201 = A[73];
    T _A2202 = A[74];
    T _A2210 = A[75];
    T _A2211 = A[76];
    T _A2212 = A[77];
    T _A2220 = A[78];
    T _A2221 = A[79];
    T _A2222 = A[80];


    T _B0000 = B[0];
    T _B0001 = B[1];
    T _B0002 = B[2];
    T _B0010 = B[3];
    T _B0011 = B[4];
    T _B0012 = B[5];
    T _B0020 = B[6];
    T _B0021 = B[7];
    T _B0022 = B[8];
    T _B0100 = B[9];
    T _B0101 = B[10];
    T _B0102 = B[11];
    T _B0110 = B[12];
    T _B0111 = B[13];
    T _B0112 = B[14];
    T _B0120 = B[15];
    T _B0121 = B[16];
    T _B0122 = B[17];
    T _B0200 = B[18];
    T _B0201 = B[19];
    T _B0202 = B[20];
    T _B0210 = B[21];
    T _B0211 = B[22];
    T _B0212 = B[23];
    T _B0220 = B[24];
    T _B0221 = B[25];
    T _B0222 = B[26];
    T _B1000 = B[27];
    T _B1001 = B[28];
    T _B1002 = B[29];
    T _B1010 = B[30];
    T _B1011 = B[31];
    T _B1012 = B[32];
    T _B1020 = B[33];
    T _B1021 = B[34];
    T _B1022 = B[35];
    T _B1100 = B[36];
    T _B1101 = B[37];
    T _B1102 = B[38];
    T _B1110 = B[39];
    T _B1111 = B[40];
    T _B1112 = B[41];
    T _B1120 = B[42];
    T _B1121 = B[43];
    T _B1122 = B[44];
    T _B1200 = B[45];
    T _B1201 = B[46];
    T _B1202 = B[47];
    T _B1210 = B[48];
    T _B1211 = B[49];
    T _B1212 = B[50];
    T _B1220 = B[51];
    T _B1221 = B[52];
    T _B1222 = B[53];
    T _B2000 = B[54];
    T _B2001 = B[55];
    T _B2002 = B[56];
    T _B2010 = B[57];
    T _B2011 = B[58];
    T _B2012 = B[59];
    T _B2020 = B[60];
    T _B2021 = B[61];
    T _B2022 = B[62];
    T _B2100 = B[63];
    T _B2101 = B[64];
    T _B2102 = B[65];
    T _B2110 = B[66];
    T _B2111 = B[67];
    T _B2112 = B[68];
    T _B2120 = B[69];
    T _B2121 = B[70];
    T _B2122 = B[71];
    T _B2200 = B[72];
    T _B2201 = B[73];
    T _B2202 = B[74];
    T _B2210 = B[75];
    T _B2211 = B[76];
    T _B2212 = B[77];
    T _B2220 = B[78];
    T _B2221 = B[79];
    T _B2222 = B[80];


    C[0] = _A1100*_B0022 - _A1200*_B0012 - _A2100*_B0021 + _A2200*_B0011;
    C[1] = _A1110*_B0022 - _A1210*_B0012 - _A2110*_B0021 + _A2210*_B0011;
    C[2] = _A1120*_B0022 - _A1220*_B0012 - _A2120*_B0021 + _A2220*_B0011;
    C[3] = _A1101*_B0022 - _A1201*_B0012 - _A2101*_B0021 + _A2201*_B0011;
    C[4] = _A1111*_B0022 - _A1211*_B0012 - _A2111*_B0021 + _A2211*_B0011;
    C[5] = _A1121*_B0022 - _A1221*_B0012 - _A2121*_B0021 + _A2221*_B0011;
    C[6] = _A1102*_B0022 - _A1202*_B0012 - _A2102*_B0021 + _A2202*_B0011;
    C[7] = _A1112*_B0022 - _A1212*_B0012 - _A2112*_B0021 + _A2212*_B0011;
    C[8] = _A1122*_B0022 - _A1222*_B0012 - _A2122*_B0021 + _A2222*_B0011;
    C[9] = _A1200*_B0002 - _A1000*_B0022 + _A2000*_B0021 - _A2200*_B0001;
    C[10] = _A1210*_B0002 - _A1010*_B0022 + _A2010*_B0021 - _A2210*_B0001;
    C[11] = _A1220*_B0002 - _A1020*_B0022 + _A2020*_B0021 - _A2220*_B0001;
    C[12] = _A1201*_B0002 - _A1001*_B0022 + _A2001*_B0021 - _A2201*_B0001;
    C[13] = _A1211*_B0002 - _A1011*_B0022 + _A2011*_B0021 - _A2211*_B0001;
    C[14] = _A1221*_B0002 - _A1021*_B0022 + _A2021*_B0021 - _A2221*_B0001;
    C[15] = _A1202*_B0002 - _A1002*_B0022 + _A2002*_B0021 - _A2202*_B0001;
    C[16] = _A1212*_B0002 - _A1012*_B0022 + _A2012*_B0021 - _A2212*_B0001;
    C[17] = _A1222*_B0002 - _A1022*_B0022 + _A2022*_B0021 - _A2222*_B0001;
    C[18] = _A1000*_B0012 - _A1100*_B0002 - _A2000*_B0011 + _A2100*_B0001;
    C[19] = _A1010*_B0012 - _A1110*_B0002 - _A2010*_B0011 + _A2110*_B0001;
    C[20] = _A1020*_B0012 - _A1120*_B0002 - _A2020*_B0011 + _A2120*_B0001;
    C[21] = _A1001*_B0012 - _A1101*_B0002 - _A2001*_B0011 + _A2101*_B0001;
    C[22] = _A1011*_B0012 - _A1111*_B0002 - _A2011*_B0011 + _A2111*_B0001;
    C[23] = _A1021*_B0012 - _A1121*_B0002 - _A2021*_B0011 + _A2121*_B0001;
    C[24] = _A1002*_B0012 - _A1102*_B0002 - _A2002*_B0011 + _A2102*_B0001;
    C[25] = _A1012*_B0012 - _A1112*_B0002 - _A2012*_B0011 + _A2112*_B0001;
    C[26] = _A1022*_B0012 - _A1122*_B0002 - _A2022*_B0011 + _A2122*_B0001;
    C[27] = _A0200*_B0012 - _A0100*_B0022 + _A2100*_B0020 - _A2200*_B0010;
    C[28] = _A0210*_B0012 - _A0110*_B0022 + _A2110*_B0020 - _A2210*_B0010;
    C[29] = _A0220*_B0012 - _A0120*_B0022 + _A2120*_B0020 - _A2220*_B0010;
    C[30] = _A0201*_B0012 - _A0101*_B0022 + _A2101*_B0020 - _A2201*_B0010;
    C[31] = _A0211*_B0012 - _A0111*_B0022 + _A2111*_B0020 - _A2211*_B0010;
    C[32] = _A0221*_B0012 - _A0121*_B0022 + _A2121*_B0020 - _A2221*_B0010;
    C[33] = _A0202*_B0012 - _A0102*_B0022 + _A2102*_B0020 - _A2202*_B0010;
    C[34] = _A0212*_B0012 - _A0112*_B0022 + _A2112*_B0020 - _A2212*_B0010;
    C[35] = _A0222*_B0012 - _A0122*_B0022 + _A2122*_B0020 - _A2222*_B0010;
    C[36] = _A0000*_B0022 - _A0200*_B0002 - _A2000*_B0020 + _A2200*_B0000;
    C[37] = _A0010*_B0022 - _A0210*_B0002 - _A2010*_B0020 + _A2210*_B0000;
    C[38] = _A0020*_B0022 - _A0220*_B0002 - _A2020*_B0020 + _A2220*_B0000;
    C[39] = _A0001*_B0022 - _A0201*_B0002 - _A2001*_B0020 + _A2201*_B0000;
    C[40] = _A0011*_B0022 - _A0211*_B0002 - _A2011*_B0020 + _A2211*_B0000;
    C[41] = _A0021*_B0022 - _A0221*_B0002 - _A2021*_B0020 + _A2221*_B0000;
    C[42] = _A0002*_B0022 - _A0202*_B0002 - _A2002*_B0020 + _A2202*_B0000;
    C[43] = _A0012*_B0022 - _A0212*_B0002 - _A2012*_B0020 + _A2212*_B0000;
    C[44] = _A0022*_B0022 - _A0222*_B0002 - _A2022*_B0020 + _A2222*_B0000;
    C[45] = _A0100*_B0002 - _A0000*_B0012 + _A2000*_B0010 - _A2100*_B0000;
    C[46] = _A0110*_B0002 - _A0010*_B0012 + _A2010*_B0010 - _A2110*_B0000;
    C[47] = _A0120*_B0002 - _A0020*_B0012 + _A2020*_B0010 - _A2120*_B0000;
    C[48] = _A0101*_B0002 - _A0001*_B0012 + _A2001*_B0010 - _A2101*_B0000;
    C[49] = _A0111*_B0002 - _A0011*_B0012 + _A2011*_B0010 - _A2111*_B0000;
    C[50] = _A0121*_B0002 - _A0021*_B0012 + _A2021*_B0010 - _A2121*_B0000;
    C[51] = _A0102*_B0002 - _A0002*_B0012 + _A2002*_B0010 - _A2102*_B0000;
    C[52] = _A0112*_B0002 - _A0012*_B0012 + _A2012*_B0010 - _A2112*_B0000;
    C[53] = _A0122*_B0002 - _A0022*_B0012 + _A2022*_B0010 - _A2122*_B0000;
    C[54] = _A0100*_B0021 - _A0200*_B0011 - _A1100*_B0020 + _A1200*_B0010;
    C[55] = _A0110*_B0021 - _A0210*_B0011 - _A1110*_B0020 + _A1210*_B0010;
    C[56] = _A0120*_B0021 - _A0220*_B0011 - _A1120*_B0020 + _A1220*_B0010;
    C[57] = _A0101*_B0021 - _A0201*_B0011 - _A1101*_B0020 + _A1201*_B0010;
    C[58] = _A0111*_B0021 - _A0211*_B0011 - _A1111*_B0020 + _A1211*_B0010;
    C[59] = _A0121*_B0021 - _A0221*_B0011 - _A1121*_B0020 + _A1221*_B0010;
    C[60] = _A0102*_B0021 - _A0202*_B0011 - _A1102*_B0020 + _A1202*_B0010;
    C[61] = _A0112*_B0021 - _A0212*_B0011 - _A1112*_B0020 + _A1212*_B0010;
    C[62] = _A0122*_B0021 - _A0222*_B0011 - _A1122*_B0020 + _A1222*_B0010;
    C[63] = _A0200*_B0001 - _A0000*_B0021 + _A1000*_B0020 - _A1200*_B0000;
    C[64] = _A0210*_B0001 - _A0010*_B0021 + _A1010*_B0020 - _A1210*_B0000;
    C[65] = _A0220*_B0001 - _A0020*_B0021 + _A1020*_B0020 - _A1220*_B0000;
    C[66] = _A0201*_B0001 - _A0001*_B0021 + _A1001*_B0020 - _A1201*_B0000;
    C[67] = _A0211*_B0001 - _A0011*_B0021 + _A1011*_B0020 - _A1211*_B0000;
    C[68] = _A0221*_B0001 - _A0021*_B0021 + _A1021*_B0020 - _A1221*_B0000;
    C[69] = _A0202*_B0001 - _A0002*_B0021 + _A1002*_B0020 - _A1202*_B0000;
    C[70] = _A0212*_B0001 - _A0012*_B0021 + _A1012*_B0020 - _A1212*_B0000;
    C[71] = _A0222*_B0001 - _A0022*_B0021 + _A1022*_B0020 - _A1222*_B0000;
    C[72] = _A0000*_B0011 - _A0100*_B0001 - _A1000*_B0010 + _A1100*_B0000;
    C[73] = _A0010*_B0011 - _A0110*_B0001 - _A1010*_B0010 + _A1110*_B0000;
    C[74] = _A0020*_B0011 - _A0120*_B0001 - _A1020*_B0010 + _A1120*_B0000;
    C[75] = _A0001*_B0011 - _A0101*_B0001 - _A1001*_B0010 + _A1101*_B0000;
    C[76] = _A0011*_B0011 - _A0111*_B0001 - _A1011*_B0010 + _A1111*_B0000;
    C[77] = _A0021*_B0011 - _A0121*_B0001 - _A1021*_B0010 + _A1121*_B0000;
    C[78] = _A0002*_B0011 - _A0102*_B0001 - _A1002*_B0010 + _A1102*_B0000;
    C[79] = _A0012*_B0011 - _A0112*_B0001 - _A1012*_B0010 + _A1112*_B0000;
    C[80] = _A0022*_B0011 - _A0122*_B0001 - _A1022*_B0010 + _A1122*_B0000;
    C[81] = _A1100*_B0122 - _A1200*_B0112 - _A2100*_B0121 + _A2200*_B0111;
    C[82] = _A1110*_B0122 - _A1210*_B0112 - _A2110*_B0121 + _A2210*_B0111;
    C[83] = _A1120*_B0122 - _A1220*_B0112 - _A2120*_B0121 + _A2220*_B0111;
    C[84] = _A1101*_B0122 - _A1201*_B0112 - _A2101*_B0121 + _A2201*_B0111;
    C[85] = _A1111*_B0122 - _A1211*_B0112 - _A2111*_B0121 + _A2211*_B0111;
    C[86] = _A1121*_B0122 - _A1221*_B0112 - _A2121*_B0121 + _A2221*_B0111;
    C[87] = _A1102*_B0122 - _A1202*_B0112 - _A2102*_B0121 + _A2202*_B0111;
    C[88] = _A1112*_B0122 - _A1212*_B0112 - _A2112*_B0121 + _A2212*_B0111;
    C[89] = _A1122*_B0122 - _A1222*_B0112 - _A2122*_B0121 + _A2222*_B0111;
    C[90] = _A1200*_B0102 - _A1000*_B0122 + _A2000*_B0121 - _A2200*_B0101;
    C[91] = _A1210*_B0102 - _A1010*_B0122 + _A2010*_B0121 - _A2210*_B0101;
    C[92] = _A1220*_B0102 - _A1020*_B0122 + _A2020*_B0121 - _A2220*_B0101;
    C[93] = _A1201*_B0102 - _A1001*_B0122 + _A2001*_B0121 - _A2201*_B0101;
    C[94] = _A1211*_B0102 - _A1011*_B0122 + _A2011*_B0121 - _A2211*_B0101;
    C[95] = _A1221*_B0102 - _A1021*_B0122 + _A2021*_B0121 - _A2221*_B0101;
    C[96] = _A1202*_B0102 - _A1002*_B0122 + _A2002*_B0121 - _A2202*_B0101;
    C[97] = _A1212*_B0102 - _A1012*_B0122 + _A2012*_B0121 - _A2212*_B0101;
    C[98] = _A1222*_B0102 - _A1022*_B0122 + _A2022*_B0121 - _A2222*_B0101;
    C[99] = _A1000*_B0112 - _A1100*_B0102 - _A2000*_B0111 + _A2100*_B0101;
    C[100] = _A1010*_B0112 - _A1110*_B0102 - _A2010*_B0111 + _A2110*_B0101;
    C[101] = _A1020*_B0112 - _A1120*_B0102 - _A2020*_B0111 + _A2120*_B0101;
    C[102] = _A1001*_B0112 - _A1101*_B0102 - _A2001*_B0111 + _A2101*_B0101;
    C[103] = _A1011*_B0112 - _A1111*_B0102 - _A2011*_B0111 + _A2111*_B0101;
    C[104] = _A1021*_B0112 - _A1121*_B0102 - _A2021*_B0111 + _A2121*_B0101;
    C[105] = _A1002*_B0112 - _A1102*_B0102 - _A2002*_B0111 + _A2102*_B0101;
    C[106] = _A1012*_B0112 - _A1112*_B0102 - _A2012*_B0111 + _A2112*_B0101;
    C[107] = _A1022*_B0112 - _A1122*_B0102 - _A2022*_B0111 + _A2122*_B0101;
    C[108] = _A0200*_B0112 - _A0100*_B0122 + _A2100*_B0120 - _A2200*_B0110;
    C[109] = _A0210*_B0112 - _A0110*_B0122 + _A2110*_B0120 - _A2210*_B0110;
    C[110] = _A0220*_B0112 - _A0120*_B0122 + _A2120*_B0120 - _A2220*_B0110;
    C[111] = _A0201*_B0112 - _A0101*_B0122 + _A2101*_B0120 - _A2201*_B0110;
    C[112] = _A0211*_B0112 - _A0111*_B0122 + _A2111*_B0120 - _A2211*_B0110;
    C[113] = _A0221*_B0112 - _A0121*_B0122 + _A2121*_B0120 - _A2221*_B0110;
    C[114] = _A0202*_B0112 - _A0102*_B0122 + _A2102*_B0120 - _A2202*_B0110;
    C[115] = _A0212*_B0112 - _A0112*_B0122 + _A2112*_B0120 - _A2212*_B0110;
    C[116] = _A0222*_B0112 - _A0122*_B0122 + _A2122*_B0120 - _A2222*_B0110;
    C[117] = _A0000*_B0122 - _A0200*_B0102 - _A2000*_B0120 + _A2200*_B0100;
    C[118] = _A0010*_B0122 - _A0210*_B0102 - _A2010*_B0120 + _A2210*_B0100;
    C[119] = _A0020*_B0122 - _A0220*_B0102 - _A2020*_B0120 + _A2220*_B0100;
    C[120] = _A0001*_B0122 - _A0201*_B0102 - _A2001*_B0120 + _A2201*_B0100;
    C[121] = _A0011*_B0122 - _A0211*_B0102 - _A2011*_B0120 + _A2211*_B0100;
    C[122] = _A0021*_B0122 - _A0221*_B0102 - _A2021*_B0120 + _A2221*_B0100;
    C[123] = _A0002*_B0122 - _A0202*_B0102 - _A2002*_B0120 + _A2202*_B0100;
    C[124] = _A0012*_B0122 - _A0212*_B0102 - _A2012*_B0120 + _A2212*_B0100;
    C[125] = _A0022*_B0122 - _A0222*_B0102 - _A2022*_B0120 + _A2222*_B0100;
    C[126] = _A0100*_B0102 - _A0000*_B0112 + _A2000*_B0110 - _A2100*_B0100;
    C[127] = _A0110*_B0102 - _A0010*_B0112 + _A2010*_B0110 - _A2110*_B0100;
    C[128] = _A0120*_B0102 - _A0020*_B0112 + _A2020*_B0110 - _A2120*_B0100;
    C[129] = _A0101*_B0102 - _A0001*_B0112 + _A2001*_B0110 - _A2101*_B0100;
    C[130] = _A0111*_B0102 - _A0011*_B0112 + _A2011*_B0110 - _A2111*_B0100;
    C[131] = _A0121*_B0102 - _A0021*_B0112 + _A2021*_B0110 - _A2121*_B0100;
    C[132] = _A0102*_B0102 - _A0002*_B0112 + _A2002*_B0110 - _A2102*_B0100;
    C[133] = _A0112*_B0102 - _A0012*_B0112 + _A2012*_B0110 - _A2112*_B0100;
    C[134] = _A0122*_B0102 - _A0022*_B0112 + _A2022*_B0110 - _A2122*_B0100;
    C[135] = _A0100*_B0121 - _A0200*_B0111 - _A1100*_B0120 + _A1200*_B0110;
    C[136] = _A0110*_B0121 - _A0210*_B0111 - _A1110*_B0120 + _A1210*_B0110;
    C[137] = _A0120*_B0121 - _A0220*_B0111 - _A1120*_B0120 + _A1220*_B0110;
    C[138] = _A0101*_B0121 - _A0201*_B0111 - _A1101*_B0120 + _A1201*_B0110;
    C[139] = _A0111*_B0121 - _A0211*_B0111 - _A1111*_B0120 + _A1211*_B0110;
    C[140] = _A0121*_B0121 - _A0221*_B0111 - _A1121*_B0120 + _A1221*_B0110;
    C[141] = _A0102*_B0121 - _A0202*_B0111 - _A1102*_B0120 + _A1202*_B0110;
    C[142] = _A0112*_B0121 - _A0212*_B0111 - _A1112*_B0120 + _A1212*_B0110;
    C[143] = _A0122*_B0121 - _A0222*_B0111 - _A1122*_B0120 + _A1222*_B0110;
    C[144] = _A0200*_B0101 - _A0000*_B0121 + _A1000*_B0120 - _A1200*_B0100;
    C[145] = _A0210*_B0101 - _A0010*_B0121 + _A1010*_B0120 - _A1210*_B0100;
    C[146] = _A0220*_B0101 - _A0020*_B0121 + _A1020*_B0120 - _A1220*_B0100;
    C[147] = _A0201*_B0101 - _A0001*_B0121 + _A1001*_B0120 - _A1201*_B0100;
    C[148] = _A0211*_B0101 - _A0011*_B0121 + _A1011*_B0120 - _A1211*_B0100;
    C[149] = _A0221*_B0101 - _A0021*_B0121 + _A1021*_B0120 - _A1221*_B0100;
    C[150] = _A0202*_B0101 - _A0002*_B0121 + _A1002*_B0120 - _A1202*_B0100;
    C[151] = _A0212*_B0101 - _A0012*_B0121 + _A1012*_B0120 - _A1212*_B0100;
    C[152] = _A0222*_B0101 - _A0022*_B0121 + _A1022*_B0120 - _A1222*_B0100;
    C[153] = _A0000*_B0111 - _A0100*_B0101 - _A1000*_B0110 + _A1100*_B0100;
    C[154] = _A0010*_B0111 - _A0110*_B0101 - _A1010*_B0110 + _A1110*_B0100;
    C[155] = _A0020*_B0111 - _A0120*_B0101 - _A1020*_B0110 + _A1120*_B0100;
    C[156] = _A0001*_B0111 - _A0101*_B0101 - _A1001*_B0110 + _A1101*_B0100;
    C[157] = _A0011*_B0111 - _A0111*_B0101 - _A1011*_B0110 + _A1111*_B0100;
    C[158] = _A0021*_B0111 - _A0121*_B0101 - _A1021*_B0110 + _A1121*_B0100;
    C[159] = _A0002*_B0111 - _A0102*_B0101 - _A1002*_B0110 + _A1102*_B0100;
    C[160] = _A0012*_B0111 - _A0112*_B0101 - _A1012*_B0110 + _A1112*_B0100;
    C[161] = _A0022*_B0111 - _A0122*_B0101 - _A1022*_B0110 + _A1122*_B0100;
    C[162] = _A1100*_B0222 - _A1200*_B0212 - _A2100*_B0221 + _A2200*_B0211;
    C[163] = _A1110*_B0222 - _A1210*_B0212 - _A2110*_B0221 + _A2210*_B0211;
    C[164] = _A1120*_B0222 - _A1220*_B0212 - _A2120*_B0221 + _A2220*_B0211;
    C[165] = _A1101*_B0222 - _A1201*_B0212 - _A2101*_B0221 + _A2201*_B0211;
    C[166] = _A1111*_B0222 - _A1211*_B0212 - _A2111*_B0221 + _A2211*_B0211;
    C[167] = _A1121*_B0222 - _A1221*_B0212 - _A2121*_B0221 + _A2221*_B0211;
    C[168] = _A1102*_B0222 - _A1202*_B0212 - _A2102*_B0221 + _A2202*_B0211;
    C[169] = _A1112*_B0222 - _A1212*_B0212 - _A2112*_B0221 + _A2212*_B0211;
    C[170] = _A1122*_B0222 - _A1222*_B0212 - _A2122*_B0221 + _A2222*_B0211;
    C[171] = _A1200*_B0202 - _A1000*_B0222 + _A2000*_B0221 - _A2200*_B0201;
    C[172] = _A1210*_B0202 - _A1010*_B0222 + _A2010*_B0221 - _A2210*_B0201;
    C[173] = _A1220*_B0202 - _A1020*_B0222 + _A2020*_B0221 - _A2220*_B0201;
    C[174] = _A1201*_B0202 - _A1001*_B0222 + _A2001*_B0221 - _A2201*_B0201;
    C[175] = _A1211*_B0202 - _A1011*_B0222 + _A2011*_B0221 - _A2211*_B0201;
    C[176] = _A1221*_B0202 - _A1021*_B0222 + _A2021*_B0221 - _A2221*_B0201;
    C[177] = _A1202*_B0202 - _A1002*_B0222 + _A2002*_B0221 - _A2202*_B0201;
    C[178] = _A1212*_B0202 - _A1012*_B0222 + _A2012*_B0221 - _A2212*_B0201;
    C[179] = _A1222*_B0202 - _A1022*_B0222 + _A2022*_B0221 - _A2222*_B0201;
    C[180] = _A1000*_B0212 - _A1100*_B0202 - _A2000*_B0211 + _A2100*_B0201;
    C[181] = _A1010*_B0212 - _A1110*_B0202 - _A2010*_B0211 + _A2110*_B0201;
    C[182] = _A1020*_B0212 - _A1120*_B0202 - _A2020*_B0211 + _A2120*_B0201;
    C[183] = _A1001*_B0212 - _A1101*_B0202 - _A2001*_B0211 + _A2101*_B0201;
    C[184] = _A1011*_B0212 - _A1111*_B0202 - _A2011*_B0211 + _A2111*_B0201;
    C[185] = _A1021*_B0212 - _A1121*_B0202 - _A2021*_B0211 + _A2121*_B0201;
    C[186] = _A1002*_B0212 - _A1102*_B0202 - _A2002*_B0211 + _A2102*_B0201;
    C[187] = _A1012*_B0212 - _A1112*_B0202 - _A2012*_B0211 + _A2112*_B0201;
    C[188] = _A1022*_B0212 - _A1122*_B0202 - _A2022*_B0211 + _A2122*_B0201;
    C[189] = _A0200*_B0212 - _A0100*_B0222 + _A2100*_B0220 - _A2200*_B0210;
    C[190] = _A0210*_B0212 - _A0110*_B0222 + _A2110*_B0220 - _A2210*_B0210;
    C[191] = _A0220*_B0212 - _A0120*_B0222 + _A2120*_B0220 - _A2220*_B0210;
    C[192] = _A0201*_B0212 - _A0101*_B0222 + _A2101*_B0220 - _A2201*_B0210;
    C[193] = _A0211*_B0212 - _A0111*_B0222 + _A2111*_B0220 - _A2211*_B0210;
    C[194] = _A0221*_B0212 - _A0121*_B0222 + _A2121*_B0220 - _A2221*_B0210;
    C[195] = _A0202*_B0212 - _A0102*_B0222 + _A2102*_B0220 - _A2202*_B0210;
    C[196] = _A0212*_B0212 - _A0112*_B0222 + _A2112*_B0220 - _A2212*_B0210;
    C[197] = _A0222*_B0212 - _A0122*_B0222 + _A2122*_B0220 - _A2222*_B0210;
    C[198] = _A0000*_B0222 - _A0200*_B0202 - _A2000*_B0220 + _A2200*_B0200;
    C[199] = _A0010*_B0222 - _A0210*_B0202 - _A2010*_B0220 + _A2210*_B0200;
    C[200] = _A0020*_B0222 - _A0220*_B0202 - _A2020*_B0220 + _A2220*_B0200;
    C[201] = _A0001*_B0222 - _A0201*_B0202 - _A2001*_B0220 + _A2201*_B0200;
    C[202] = _A0011*_B0222 - _A0211*_B0202 - _A2011*_B0220 + _A2211*_B0200;
    C[203] = _A0021*_B0222 - _A0221*_B0202 - _A2021*_B0220 + _A2221*_B0200;
    C[204] = _A0002*_B0222 - _A0202*_B0202 - _A2002*_B0220 + _A2202*_B0200;
    C[205] = _A0012*_B0222 - _A0212*_B0202 - _A2012*_B0220 + _A2212*_B0200;
    C[206] = _A0022*_B0222 - _A0222*_B0202 - _A2022*_B0220 + _A2222*_B0200;
    C[207] = _A0100*_B0202 - _A0000*_B0212 + _A2000*_B0210 - _A2100*_B0200;
    C[208] = _A0110*_B0202 - _A0010*_B0212 + _A2010*_B0210 - _A2110*_B0200;
    C[209] = _A0120*_B0202 - _A0020*_B0212 + _A2020*_B0210 - _A2120*_B0200;
    C[210] = _A0101*_B0202 - _A0001*_B0212 + _A2001*_B0210 - _A2101*_B0200;
    C[211] = _A0111*_B0202 - _A0011*_B0212 + _A2011*_B0210 - _A2111*_B0200;
    C[212] = _A0121*_B0202 - _A0021*_B0212 + _A2021*_B0210 - _A2121*_B0200;
    C[213] = _A0102*_B0202 - _A0002*_B0212 + _A2002*_B0210 - _A2102*_B0200;
    C[214] = _A0112*_B0202 - _A0012*_B0212 + _A2012*_B0210 - _A2112*_B0200;
    C[215] = _A0122*_B0202 - _A0022*_B0212 + _A2022*_B0210 - _A2122*_B0200;
    C[216] = _A0100*_B0221 - _A0200*_B0211 - _A1100*_B0220 + _A1200*_B0210;
    C[217] = _A0110*_B0221 - _A0210*_B0211 - _A1110*_B0220 + _A1210*_B0210;
    C[218] = _A0120*_B0221 - _A0220*_B0211 - _A1120*_B0220 + _A1220*_B0210;
    C[219] = _A0101*_B0221 - _A0201*_B0211 - _A1101*_B0220 + _A1201*_B0210;
    C[220] = _A0111*_B0221 - _A0211*_B0211 - _A1111*_B0220 + _A1211*_B0210;
    C[221] = _A0121*_B0221 - _A0221*_B0211 - _A1121*_B0220 + _A1221*_B0210;
    C[222] = _A0102*_B0221 - _A0202*_B0211 - _A1102*_B0220 + _A1202*_B0210;
    C[223] = _A0112*_B0221 - _A0212*_B0211 - _A1112*_B0220 + _A1212*_B0210;
    C[224] = _A0122*_B0221 - _A0222*_B0211 - _A1122*_B0220 + _A1222*_B0210;
    C[225] = _A0200*_B0201 - _A0000*_B0221 + _A1000*_B0220 - _A1200*_B0200;
    C[226] = _A0210*_B0201 - _A0010*_B0221 + _A1010*_B0220 - _A1210*_B0200;
    C[227] = _A0220*_B0201 - _A0020*_B0221 + _A1020*_B0220 - _A1220*_B0200;
    C[228] = _A0201*_B0201 - _A0001*_B0221 + _A1001*_B0220 - _A1201*_B0200;
    C[229] = _A0211*_B0201 - _A0011*_B0221 + _A1011*_B0220 - _A1211*_B0200;
    C[230] = _A0221*_B0201 - _A0021*_B0221 + _A1021*_B0220 - _A1221*_B0200;
    C[231] = _A0202*_B0201 - _A0002*_B0221 + _A1002*_B0220 - _A1202*_B0200;
    C[232] = _A0212*_B0201 - _A0012*_B0221 + _A1012*_B0220 - _A1212*_B0200;
    C[233] = _A0222*_B0201 - _A0022*_B0221 + _A1022*_B0220 - _A1222*_B0200;
    C[234] = _A0000*_B0211 - _A0100*_B0201 - _A1000*_B0210 + _A1100*_B0200;
    C[235] = _A0010*_B0211 - _A0110*_B0201 - _A1010*_B0210 + _A1110*_B0200;
    C[236] = _A0020*_B0211 - _A0120*_B0201 - _A1020*_B0210 + _A1120*_B0200;
    C[237] = _A0001*_B0211 - _A0101*_B0201 - _A1001*_B0210 + _A1101*_B0200;
    C[238] = _A0011*_B0211 - _A0111*_B0201 - _A1011*_B0210 + _A1111*_B0200;
    C[239] = _A0021*_B0211 - _A0121*_B0201 - _A1021*_B0210 + _A1121*_B0200;
    C[240] = _A0002*_B0211 - _A0102*_B0201 - _A1002*_B0210 + _A1102*_B0200;
    C[241] = _A0012*_B0211 - _A0112*_B0201 - _A1012*_B0210 + _A1112*_B0200;
    C[242] = _A0022*_B0211 - _A0122*_B0201 - _A1022*_B0210 + _A1122*_B0200;
    C[243] = _A1100*_B1022 - _A1200*_B1012 - _A2100*_B1021 + _A2200*_B1011;
    C[244] = _A1110*_B1022 - _A1210*_B1012 - _A2110*_B1021 + _A2210*_B1011;
    C[245] = _A1120*_B1022 - _A1220*_B1012 - _A2120*_B1021 + _A2220*_B1011;
    C[246] = _A1101*_B1022 - _A1201*_B1012 - _A2101*_B1021 + _A2201*_B1011;
    C[247] = _A1111*_B1022 - _A1211*_B1012 - _A2111*_B1021 + _A2211*_B1011;
    C[248] = _A1121*_B1022 - _A1221*_B1012 - _A2121*_B1021 + _A2221*_B1011;
    C[249] = _A1102*_B1022 - _A1202*_B1012 - _A2102*_B1021 + _A2202*_B1011;
    C[250] = _A1112*_B1022 - _A1212*_B1012 - _A2112*_B1021 + _A2212*_B1011;
    C[251] = _A1122*_B1022 - _A1222*_B1012 - _A2122*_B1021 + _A2222*_B1011;
    C[252] = _A1200*_B1002 - _A1000*_B1022 + _A2000*_B1021 - _A2200*_B1001;
    C[253] = _A1210*_B1002 - _A1010*_B1022 + _A2010*_B1021 - _A2210*_B1001;
    C[254] = _A1220*_B1002 - _A1020*_B1022 + _A2020*_B1021 - _A2220*_B1001;
    C[255] = _A1201*_B1002 - _A1001*_B1022 + _A2001*_B1021 - _A2201*_B1001;
    C[256] = _A1211*_B1002 - _A1011*_B1022 + _A2011*_B1021 - _A2211*_B1001;
    C[257] = _A1221*_B1002 - _A1021*_B1022 + _A2021*_B1021 - _A2221*_B1001;
    C[258] = _A1202*_B1002 - _A1002*_B1022 + _A2002*_B1021 - _A2202*_B1001;
    C[259] = _A1212*_B1002 - _A1012*_B1022 + _A2012*_B1021 - _A2212*_B1001;
    C[260] = _A1222*_B1002 - _A1022*_B1022 + _A2022*_B1021 - _A2222*_B1001;
    C[261] = _A1000*_B1012 - _A1100*_B1002 - _A2000*_B1011 + _A2100*_B1001;
    C[262] = _A1010*_B1012 - _A1110*_B1002 - _A2010*_B1011 + _A2110*_B1001;
    C[263] = _A1020*_B1012 - _A1120*_B1002 - _A2020*_B1011 + _A2120*_B1001;
    C[264] = _A1001*_B1012 - _A1101*_B1002 - _A2001*_B1011 + _A2101*_B1001;
    C[265] = _A1011*_B1012 - _A1111*_B1002 - _A2011*_B1011 + _A2111*_B1001;
    C[266] = _A1021*_B1012 - _A1121*_B1002 - _A2021*_B1011 + _A2121*_B1001;
    C[267] = _A1002*_B1012 - _A1102*_B1002 - _A2002*_B1011 + _A2102*_B1001;
    C[268] = _A1012*_B1012 - _A1112*_B1002 - _A2012*_B1011 + _A2112*_B1001;
    C[269] = _A1022*_B1012 - _A1122*_B1002 - _A2022*_B1011 + _A2122*_B1001;
    C[270] = _A0200*_B1012 - _A0100*_B1022 + _A2100*_B1020 - _A2200*_B1010;
    C[271] = _A0210*_B1012 - _A0110*_B1022 + _A2110*_B1020 - _A2210*_B1010;
    C[272] = _A0220*_B1012 - _A0120*_B1022 + _A2120*_B1020 - _A2220*_B1010;
    C[273] = _A0201*_B1012 - _A0101*_B1022 + _A2101*_B1020 - _A2201*_B1010;
    C[274] = _A0211*_B1012 - _A0111*_B1022 + _A2111*_B1020 - _A2211*_B1010;
    C[275] = _A0221*_B1012 - _A0121*_B1022 + _A2121*_B1020 - _A2221*_B1010;
    C[276] = _A0202*_B1012 - _A0102*_B1022 + _A2102*_B1020 - _A2202*_B1010;
    C[277] = _A0212*_B1012 - _A0112*_B1022 + _A2112*_B1020 - _A2212*_B1010;
    C[278] = _A0222*_B1012 - _A0122*_B1022 + _A2122*_B1020 - _A2222*_B1010;
    C[279] = _A0000*_B1022 - _A0200*_B1002 - _A2000*_B1020 + _A2200*_B1000;
    C[280] = _A0010*_B1022 - _A0210*_B1002 - _A2010*_B1020 + _A2210*_B1000;
    C[281] = _A0020*_B1022 - _A0220*_B1002 - _A2020*_B1020 + _A2220*_B1000;
    C[282] = _A0001*_B1022 - _A0201*_B1002 - _A2001*_B1020 + _A2201*_B1000;
    C[283] = _A0011*_B1022 - _A0211*_B1002 - _A2011*_B1020 + _A2211*_B1000;
    C[284] = _A0021*_B1022 - _A0221*_B1002 - _A2021*_B1020 + _A2221*_B1000;
    C[285] = _A0002*_B1022 - _A0202*_B1002 - _A2002*_B1020 + _A2202*_B1000;
    C[286] = _A0012*_B1022 - _A0212*_B1002 - _A2012*_B1020 + _A2212*_B1000;
    C[287] = _A0022*_B1022 - _A0222*_B1002 - _A2022*_B1020 + _A2222*_B1000;
    C[288] = _A0100*_B1002 - _A0000*_B1012 + _A2000*_B1010 - _A2100*_B1000;
    C[289] = _A0110*_B1002 - _A0010*_B1012 + _A2010*_B1010 - _A2110*_B1000;
    C[290] = _A0120*_B1002 - _A0020*_B1012 + _A2020*_B1010 - _A2120*_B1000;
    C[291] = _A0101*_B1002 - _A0001*_B1012 + _A2001*_B1010 - _A2101*_B1000;
    C[292] = _A0111*_B1002 - _A0011*_B1012 + _A2011*_B1010 - _A2111*_B1000;
    C[293] = _A0121*_B1002 - _A0021*_B1012 + _A2021*_B1010 - _A2121*_B1000;
    C[294] = _A0102*_B1002 - _A0002*_B1012 + _A2002*_B1010 - _A2102*_B1000;
    C[295] = _A0112*_B1002 - _A0012*_B1012 + _A2012*_B1010 - _A2112*_B1000;
    C[296] = _A0122*_B1002 - _A0022*_B1012 + _A2022*_B1010 - _A2122*_B1000;
    C[297] = _A0100*_B1021 - _A0200*_B1011 - _A1100*_B1020 + _A1200*_B1010;
    C[298] = _A0110*_B1021 - _A0210*_B1011 - _A1110*_B1020 + _A1210*_B1010;
    C[299] = _A0120*_B1021 - _A0220*_B1011 - _A1120*_B1020 + _A1220*_B1010;
    C[300] = _A0101*_B1021 - _A0201*_B1011 - _A1101*_B1020 + _A1201*_B1010;
    C[301] = _A0111*_B1021 - _A0211*_B1011 - _A1111*_B1020 + _A1211*_B1010;
    C[302] = _A0121*_B1021 - _A0221*_B1011 - _A1121*_B1020 + _A1221*_B1010;
    C[303] = _A0102*_B1021 - _A0202*_B1011 - _A1102*_B1020 + _A1202*_B1010;
    C[304] = _A0112*_B1021 - _A0212*_B1011 - _A1112*_B1020 + _A1212*_B1010;
    C[305] = _A0122*_B1021 - _A0222*_B1011 - _A1122*_B1020 + _A1222*_B1010;
    C[306] = _A0200*_B1001 - _A0000*_B1021 + _A1000*_B1020 - _A1200*_B1000;
    C[307] = _A0210*_B1001 - _A0010*_B1021 + _A1010*_B1020 - _A1210*_B1000;
    C[308] = _A0220*_B1001 - _A0020*_B1021 + _A1020*_B1020 - _A1220*_B1000;
    C[309] = _A0201*_B1001 - _A0001*_B1021 + _A1001*_B1020 - _A1201*_B1000;
    C[310] = _A0211*_B1001 - _A0011*_B1021 + _A1011*_B1020 - _A1211*_B1000;
    C[311] = _A0221*_B1001 - _A0021*_B1021 + _A1021*_B1020 - _A1221*_B1000;
    C[312] = _A0202*_B1001 - _A0002*_B1021 + _A1002*_B1020 - _A1202*_B1000;
    C[313] = _A0212*_B1001 - _A0012*_B1021 + _A1012*_B1020 - _A1212*_B1000;
    C[314] = _A0222*_B1001 - _A0022*_B1021 + _A1022*_B1020 - _A1222*_B1000;
    C[315] = _A0000*_B1011 - _A0100*_B1001 - _A1000*_B1010 + _A1100*_B1000;
    C[316] = _A0010*_B1011 - _A0110*_B1001 - _A1010*_B1010 + _A1110*_B1000;
    C[317] = _A0020*_B1011 - _A0120*_B1001 - _A1020*_B1010 + _A1120*_B1000;
    C[318] = _A0001*_B1011 - _A0101*_B1001 - _A1001*_B1010 + _A1101*_B1000;
    C[319] = _A0011*_B1011 - _A0111*_B1001 - _A1011*_B1010 + _A1111*_B1000;
    C[320] = _A0021*_B1011 - _A0121*_B1001 - _A1021*_B1010 + _A1121*_B1000;
    C[321] = _A0002*_B1011 - _A0102*_B1001 - _A1002*_B1010 + _A1102*_B1000;
    C[322] = _A0012*_B1011 - _A0112*_B1001 - _A1012*_B1010 + _A1112*_B1000;
    C[323] = _A0022*_B1011 - _A0122*_B1001 - _A1022*_B1010 + _A1122*_B1000;
    C[324] = _A1100*_B1122 - _A1200*_B1112 - _A2100*_B1121 + _A2200*_B1111;
    C[325] = _A1110*_B1122 - _A1210*_B1112 - _A2110*_B1121 + _A2210*_B1111;
    C[326] = _A1120*_B1122 - _A1220*_B1112 - _A2120*_B1121 + _A2220*_B1111;
    C[327] = _A1101*_B1122 - _A1201*_B1112 - _A2101*_B1121 + _A2201*_B1111;
    C[328] = _A1111*_B1122 - _A1211*_B1112 - _A2111*_B1121 + _A2211*_B1111;
    C[329] = _A1121*_B1122 - _A1221*_B1112 - _A2121*_B1121 + _A2221*_B1111;
    C[330] = _A1102*_B1122 - _A1202*_B1112 - _A2102*_B1121 + _A2202*_B1111;
    C[331] = _A1112*_B1122 - _A1212*_B1112 - _A2112*_B1121 + _A2212*_B1111;
    C[332] = _A1122*_B1122 - _A1222*_B1112 - _A2122*_B1121 + _A2222*_B1111;
    C[333] = _A1200*_B1102 - _A1000*_B1122 + _A2000*_B1121 - _A2200*_B1101;
    C[334] = _A1210*_B1102 - _A1010*_B1122 + _A2010*_B1121 - _A2210*_B1101;
    C[335] = _A1220*_B1102 - _A1020*_B1122 + _A2020*_B1121 - _A2220*_B1101;
    C[336] = _A1201*_B1102 - _A1001*_B1122 + _A2001*_B1121 - _A2201*_B1101;
    C[337] = _A1211*_B1102 - _A1011*_B1122 + _A2011*_B1121 - _A2211*_B1101;
    C[338] = _A1221*_B1102 - _A1021*_B1122 + _A2021*_B1121 - _A2221*_B1101;
    C[339] = _A1202*_B1102 - _A1002*_B1122 + _A2002*_B1121 - _A2202*_B1101;
    C[340] = _A1212*_B1102 - _A1012*_B1122 + _A2012*_B1121 - _A2212*_B1101;
    C[341] = _A1222*_B1102 - _A1022*_B1122 + _A2022*_B1121 - _A2222*_B1101;
    C[342] = _A1000*_B1112 - _A1100*_B1102 - _A2000*_B1111 + _A2100*_B1101;
    C[343] = _A1010*_B1112 - _A1110*_B1102 - _A2010*_B1111 + _A2110*_B1101;
    C[344] = _A1020*_B1112 - _A1120*_B1102 - _A2020*_B1111 + _A2120*_B1101;
    C[345] = _A1001*_B1112 - _A1101*_B1102 - _A2001*_B1111 + _A2101*_B1101;
    C[346] = _A1011*_B1112 - _A1111*_B1102 - _A2011*_B1111 + _A2111*_B1101;
    C[347] = _A1021*_B1112 - _A1121*_B1102 - _A2021*_B1111 + _A2121*_B1101;
    C[348] = _A1002*_B1112 - _A1102*_B1102 - _A2002*_B1111 + _A2102*_B1101;
    C[349] = _A1012*_B1112 - _A1112*_B1102 - _A2012*_B1111 + _A2112*_B1101;
    C[350] = _A1022*_B1112 - _A1122*_B1102 - _A2022*_B1111 + _A2122*_B1101;
    C[351] = _A0200*_B1112 - _A0100*_B1122 + _A2100*_B1120 - _A2200*_B1110;
    C[352] = _A0210*_B1112 - _A0110*_B1122 + _A2110*_B1120 - _A2210*_B1110;
    C[353] = _A0220*_B1112 - _A0120*_B1122 + _A2120*_B1120 - _A2220*_B1110;
    C[354] = _A0201*_B1112 - _A0101*_B1122 + _A2101*_B1120 - _A2201*_B1110;
    C[355] = _A0211*_B1112 - _A0111*_B1122 + _A2111*_B1120 - _A2211*_B1110;
    C[356] = _A0221*_B1112 - _A0121*_B1122 + _A2121*_B1120 - _A2221*_B1110;
    C[357] = _A0202*_B1112 - _A0102*_B1122 + _A2102*_B1120 - _A2202*_B1110;
    C[358] = _A0212*_B1112 - _A0112*_B1122 + _A2112*_B1120 - _A2212*_B1110;
    C[359] = _A0222*_B1112 - _A0122*_B1122 + _A2122*_B1120 - _A2222*_B1110;
    C[360] = _A0000*_B1122 - _A0200*_B1102 - _A2000*_B1120 + _A2200*_B1100;
    C[361] = _A0010*_B1122 - _A0210*_B1102 - _A2010*_B1120 + _A2210*_B1100;
    C[362] = _A0020*_B1122 - _A0220*_B1102 - _A2020*_B1120 + _A2220*_B1100;
    C[363] = _A0001*_B1122 - _A0201*_B1102 - _A2001*_B1120 + _A2201*_B1100;
    C[364] = _A0011*_B1122 - _A0211*_B1102 - _A2011*_B1120 + _A2211*_B1100;
    C[365] = _A0021*_B1122 - _A0221*_B1102 - _A2021*_B1120 + _A2221*_B1100;
    C[366] = _A0002*_B1122 - _A0202*_B1102 - _A2002*_B1120 + _A2202*_B1100;
    C[367] = _A0012*_B1122 - _A0212*_B1102 - _A2012*_B1120 + _A2212*_B1100;
    C[368] = _A0022*_B1122 - _A0222*_B1102 - _A2022*_B1120 + _A2222*_B1100;
    C[369] = _A0100*_B1102 - _A0000*_B1112 + _A2000*_B1110 - _A2100*_B1100;
    C[370] = _A0110*_B1102 - _A0010*_B1112 + _A2010*_B1110 - _A2110*_B1100;
    C[371] = _A0120*_B1102 - _A0020*_B1112 + _A2020*_B1110 - _A2120*_B1100;
    C[372] = _A0101*_B1102 - _A0001*_B1112 + _A2001*_B1110 - _A2101*_B1100;
    C[373] = _A0111*_B1102 - _A0011*_B1112 + _A2011*_B1110 - _A2111*_B1100;
    C[374] = _A0121*_B1102 - _A0021*_B1112 + _A2021*_B1110 - _A2121*_B1100;
    C[375] = _A0102*_B1102 - _A0002*_B1112 + _A2002*_B1110 - _A2102*_B1100;
    C[376] = _A0112*_B1102 - _A0012*_B1112 + _A2012*_B1110 - _A2112*_B1100;
    C[377] = _A0122*_B1102 - _A0022*_B1112 + _A2022*_B1110 - _A2122*_B1100;
    C[378] = _A0100*_B1121 - _A0200*_B1111 - _A1100*_B1120 + _A1200*_B1110;
    C[379] = _A0110*_B1121 - _A0210*_B1111 - _A1110*_B1120 + _A1210*_B1110;
    C[380] = _A0120*_B1121 - _A0220*_B1111 - _A1120*_B1120 + _A1220*_B1110;
    C[381] = _A0101*_B1121 - _A0201*_B1111 - _A1101*_B1120 + _A1201*_B1110;
    C[382] = _A0111*_B1121 - _A0211*_B1111 - _A1111*_B1120 + _A1211*_B1110;
    C[383] = _A0121*_B1121 - _A0221*_B1111 - _A1121*_B1120 + _A1221*_B1110;
    C[384] = _A0102*_B1121 - _A0202*_B1111 - _A1102*_B1120 + _A1202*_B1110;
    C[385] = _A0112*_B1121 - _A0212*_B1111 - _A1112*_B1120 + _A1212*_B1110;
    C[386] = _A0122*_B1121 - _A0222*_B1111 - _A1122*_B1120 + _A1222*_B1110;
    C[387] = _A0200*_B1101 - _A0000*_B1121 + _A1000*_B1120 - _A1200*_B1100;
    C[388] = _A0210*_B1101 - _A0010*_B1121 + _A1010*_B1120 - _A1210*_B1100;
    C[389] = _A0220*_B1101 - _A0020*_B1121 + _A1020*_B1120 - _A1220*_B1100;
    C[390] = _A0201*_B1101 - _A0001*_B1121 + _A1001*_B1120 - _A1201*_B1100;
    C[391] = _A0211*_B1101 - _A0011*_B1121 + _A1011*_B1120 - _A1211*_B1100;
    C[392] = _A0221*_B1101 - _A0021*_B1121 + _A1021*_B1120 - _A1221*_B1100;
    C[393] = _A0202*_B1101 - _A0002*_B1121 + _A1002*_B1120 - _A1202*_B1100;
    C[394] = _A0212*_B1101 - _A0012*_B1121 + _A1012*_B1120 - _A1212*_B1100;
    C[395] = _A0222*_B1101 - _A0022*_B1121 + _A1022*_B1120 - _A1222*_B1100;
    C[396] = _A0000*_B1111 - _A0100*_B1101 - _A1000*_B1110 + _A1100*_B1100;
    C[397] = _A0010*_B1111 - _A0110*_B1101 - _A1010*_B1110 + _A1110*_B1100;
    C[398] = _A0020*_B1111 - _A0120*_B1101 - _A1020*_B1110 + _A1120*_B1100;
    C[399] = _A0001*_B1111 - _A0101*_B1101 - _A1001*_B1110 + _A1101*_B1100;
    C[400] = _A0011*_B1111 - _A0111*_B1101 - _A1011*_B1110 + _A1111*_B1100;
    C[401] = _A0021*_B1111 - _A0121*_B1101 - _A1021*_B1110 + _A1121*_B1100;
    C[402] = _A0002*_B1111 - _A0102*_B1101 - _A1002*_B1110 + _A1102*_B1100;
    C[403] = _A0012*_B1111 - _A0112*_B1101 - _A1012*_B1110 + _A1112*_B1100;
    C[404] = _A0022*_B1111 - _A0122*_B1101 - _A1022*_B1110 + _A1122*_B1100;
    C[405] = _A1100*_B1222 - _A1200*_B1212 - _A2100*_B1221 + _A2200*_B1211;
    C[406] = _A1110*_B1222 - _A1210*_B1212 - _A2110*_B1221 + _A2210*_B1211;
    C[407] = _A1120*_B1222 - _A1220*_B1212 - _A2120*_B1221 + _A2220*_B1211;
    C[408] = _A1101*_B1222 - _A1201*_B1212 - _A2101*_B1221 + _A2201*_B1211;
    C[409] = _A1111*_B1222 - _A1211*_B1212 - _A2111*_B1221 + _A2211*_B1211;
    C[410] = _A1121*_B1222 - _A1221*_B1212 - _A2121*_B1221 + _A2221*_B1211;
    C[411] = _A1102*_B1222 - _A1202*_B1212 - _A2102*_B1221 + _A2202*_B1211;
    C[412] = _A1112*_B1222 - _A1212*_B1212 - _A2112*_B1221 + _A2212*_B1211;
    C[413] = _A1122*_B1222 - _A1222*_B1212 - _A2122*_B1221 + _A2222*_B1211;
    C[414] = _A1200*_B1202 - _A1000*_B1222 + _A2000*_B1221 - _A2200*_B1201;
    C[415] = _A1210*_B1202 - _A1010*_B1222 + _A2010*_B1221 - _A2210*_B1201;
    C[416] = _A1220*_B1202 - _A1020*_B1222 + _A2020*_B1221 - _A2220*_B1201;
    C[417] = _A1201*_B1202 - _A1001*_B1222 + _A2001*_B1221 - _A2201*_B1201;
    C[418] = _A1211*_B1202 - _A1011*_B1222 + _A2011*_B1221 - _A2211*_B1201;
    C[419] = _A1221*_B1202 - _A1021*_B1222 + _A2021*_B1221 - _A2221*_B1201;
    C[420] = _A1202*_B1202 - _A1002*_B1222 + _A2002*_B1221 - _A2202*_B1201;
    C[421] = _A1212*_B1202 - _A1012*_B1222 + _A2012*_B1221 - _A2212*_B1201;
    C[422] = _A1222*_B1202 - _A1022*_B1222 + _A2022*_B1221 - _A2222*_B1201;
    C[423] = _A1000*_B1212 - _A1100*_B1202 - _A2000*_B1211 + _A2100*_B1201;
    C[424] = _A1010*_B1212 - _A1110*_B1202 - _A2010*_B1211 + _A2110*_B1201;
    C[425] = _A1020*_B1212 - _A1120*_B1202 - _A2020*_B1211 + _A2120*_B1201;
    C[426] = _A1001*_B1212 - _A1101*_B1202 - _A2001*_B1211 + _A2101*_B1201;
    C[427] = _A1011*_B1212 - _A1111*_B1202 - _A2011*_B1211 + _A2111*_B1201;
    C[428] = _A1021*_B1212 - _A1121*_B1202 - _A2021*_B1211 + _A2121*_B1201;
    C[429] = _A1002*_B1212 - _A1102*_B1202 - _A2002*_B1211 + _A2102*_B1201;
    C[430] = _A1012*_B1212 - _A1112*_B1202 - _A2012*_B1211 + _A2112*_B1201;
    C[431] = _A1022*_B1212 - _A1122*_B1202 - _A2022*_B1211 + _A2122*_B1201;
    C[432] = _A0200*_B1212 - _A0100*_B1222 + _A2100*_B1220 - _A2200*_B1210;
    C[433] = _A0210*_B1212 - _A0110*_B1222 + _A2110*_B1220 - _A2210*_B1210;
    C[434] = _A0220*_B1212 - _A0120*_B1222 + _A2120*_B1220 - _A2220*_B1210;
    C[435] = _A0201*_B1212 - _A0101*_B1222 + _A2101*_B1220 - _A2201*_B1210;
    C[436] = _A0211*_B1212 - _A0111*_B1222 + _A2111*_B1220 - _A2211*_B1210;
    C[437] = _A0221*_B1212 - _A0121*_B1222 + _A2121*_B1220 - _A2221*_B1210;
    C[438] = _A0202*_B1212 - _A0102*_B1222 + _A2102*_B1220 - _A2202*_B1210;
    C[439] = _A0212*_B1212 - _A0112*_B1222 + _A2112*_B1220 - _A2212*_B1210;
    C[440] = _A0222*_B1212 - _A0122*_B1222 + _A2122*_B1220 - _A2222*_B1210;
    C[441] = _A0000*_B1222 - _A0200*_B1202 - _A2000*_B1220 + _A2200*_B1200;
    C[442] = _A0010*_B1222 - _A0210*_B1202 - _A2010*_B1220 + _A2210*_B1200;
    C[443] = _A0020*_B1222 - _A0220*_B1202 - _A2020*_B1220 + _A2220*_B1200;
    C[444] = _A0001*_B1222 - _A0201*_B1202 - _A2001*_B1220 + _A2201*_B1200;
    C[445] = _A0011*_B1222 - _A0211*_B1202 - _A2011*_B1220 + _A2211*_B1200;
    C[446] = _A0021*_B1222 - _A0221*_B1202 - _A2021*_B1220 + _A2221*_B1200;
    C[447] = _A0002*_B1222 - _A0202*_B1202 - _A2002*_B1220 + _A2202*_B1200;
    C[448] = _A0012*_B1222 - _A0212*_B1202 - _A2012*_B1220 + _A2212*_B1200;
    C[449] = _A0022*_B1222 - _A0222*_B1202 - _A2022*_B1220 + _A2222*_B1200;
    C[450] = _A0100*_B1202 - _A0000*_B1212 + _A2000*_B1210 - _A2100*_B1200;
    C[451] = _A0110*_B1202 - _A0010*_B1212 + _A2010*_B1210 - _A2110*_B1200;
    C[452] = _A0120*_B1202 - _A0020*_B1212 + _A2020*_B1210 - _A2120*_B1200;
    C[453] = _A0101*_B1202 - _A0001*_B1212 + _A2001*_B1210 - _A2101*_B1200;
    C[454] = _A0111*_B1202 - _A0011*_B1212 + _A2011*_B1210 - _A2111*_B1200;
    C[455] = _A0121*_B1202 - _A0021*_B1212 + _A2021*_B1210 - _A2121*_B1200;
    C[456] = _A0102*_B1202 - _A0002*_B1212 + _A2002*_B1210 - _A2102*_B1200;
    C[457] = _A0112*_B1202 - _A0012*_B1212 + _A2012*_B1210 - _A2112*_B1200;
    C[458] = _A0122*_B1202 - _A0022*_B1212 + _A2022*_B1210 - _A2122*_B1200;
    C[459] = _A0100*_B1221 - _A0200*_B1211 - _A1100*_B1220 + _A1200*_B1210;
    C[460] = _A0110*_B1221 - _A0210*_B1211 - _A1110*_B1220 + _A1210*_B1210;
    C[461] = _A0120*_B1221 - _A0220*_B1211 - _A1120*_B1220 + _A1220*_B1210;
    C[462] = _A0101*_B1221 - _A0201*_B1211 - _A1101*_B1220 + _A1201*_B1210;
    C[463] = _A0111*_B1221 - _A0211*_B1211 - _A1111*_B1220 + _A1211*_B1210;
    C[464] = _A0121*_B1221 - _A0221*_B1211 - _A1121*_B1220 + _A1221*_B1210;
    C[465] = _A0102*_B1221 - _A0202*_B1211 - _A1102*_B1220 + _A1202*_B1210;
    C[466] = _A0112*_B1221 - _A0212*_B1211 - _A1112*_B1220 + _A1212*_B1210;
    C[467] = _A0122*_B1221 - _A0222*_B1211 - _A1122*_B1220 + _A1222*_B1210;
    C[468] = _A0200*_B1201 - _A0000*_B1221 + _A1000*_B1220 - _A1200*_B1200;
    C[469] = _A0210*_B1201 - _A0010*_B1221 + _A1010*_B1220 - _A1210*_B1200;
    C[470] = _A0220*_B1201 - _A0020*_B1221 + _A1020*_B1220 - _A1220*_B1200;
    C[471] = _A0201*_B1201 - _A0001*_B1221 + _A1001*_B1220 - _A1201*_B1200;
    C[472] = _A0211*_B1201 - _A0011*_B1221 + _A1011*_B1220 - _A1211*_B1200;
    C[473] = _A0221*_B1201 - _A0021*_B1221 + _A1021*_B1220 - _A1221*_B1200;
    C[474] = _A0202*_B1201 - _A0002*_B1221 + _A1002*_B1220 - _A1202*_B1200;
    C[475] = _A0212*_B1201 - _A0012*_B1221 + _A1012*_B1220 - _A1212*_B1200;
    C[476] = _A0222*_B1201 - _A0022*_B1221 + _A1022*_B1220 - _A1222*_B1200;
    C[477] = _A0000*_B1211 - _A0100*_B1201 - _A1000*_B1210 + _A1100*_B1200;
    C[478] = _A0010*_B1211 - _A0110*_B1201 - _A1010*_B1210 + _A1110*_B1200;
    C[479] = _A0020*_B1211 - _A0120*_B1201 - _A1020*_B1210 + _A1120*_B1200;
    C[480] = _A0001*_B1211 - _A0101*_B1201 - _A1001*_B1210 + _A1101*_B1200;
    C[481] = _A0011*_B1211 - _A0111*_B1201 - _A1011*_B1210 + _A1111*_B1200;
    C[482] = _A0021*_B1211 - _A0121*_B1201 - _A1021*_B1210 + _A1121*_B1200;
    C[483] = _A0002*_B1211 - _A0102*_B1201 - _A1002*_B1210 + _A1102*_B1200;
    C[484] = _A0012*_B1211 - _A0112*_B1201 - _A1012*_B1210 + _A1112*_B1200;
    C[485] = _A0022*_B1211 - _A0122*_B1201 - _A1022*_B1210 + _A1122*_B1200;
    C[486] = _A1100*_B2022 - _A1200*_B2012 - _A2100*_B2021 + _A2200*_B2011;
    C[487] = _A1110*_B2022 - _A1210*_B2012 - _A2110*_B2021 + _A2210*_B2011;
    C[488] = _A1120*_B2022 - _A1220*_B2012 - _A2120*_B2021 + _A2220*_B2011;
    C[489] = _A1101*_B2022 - _A1201*_B2012 - _A2101*_B2021 + _A2201*_B2011;
    C[490] = _A1111*_B2022 - _A1211*_B2012 - _A2111*_B2021 + _A2211*_B2011;
    C[491] = _A1121*_B2022 - _A1221*_B2012 - _A2121*_B2021 + _A2221*_B2011;
    C[492] = _A1102*_B2022 - _A1202*_B2012 - _A2102*_B2021 + _A2202*_B2011;
    C[493] = _A1112*_B2022 - _A1212*_B2012 - _A2112*_B2021 + _A2212*_B2011;
    C[494] = _A1122*_B2022 - _A1222*_B2012 - _A2122*_B2021 + _A2222*_B2011;
    C[495] = _A1200*_B2002 - _A1000*_B2022 + _A2000*_B2021 - _A2200*_B2001;
    C[496] = _A1210*_B2002 - _A1010*_B2022 + _A2010*_B2021 - _A2210*_B2001;
    C[497] = _A1220*_B2002 - _A1020*_B2022 + _A2020*_B2021 - _A2220*_B2001;
    C[498] = _A1201*_B2002 - _A1001*_B2022 + _A2001*_B2021 - _A2201*_B2001;
    C[499] = _A1211*_B2002 - _A1011*_B2022 + _A2011*_B2021 - _A2211*_B2001;
    C[500] = _A1221*_B2002 - _A1021*_B2022 + _A2021*_B2021 - _A2221*_B2001;
    C[501] = _A1202*_B2002 - _A1002*_B2022 + _A2002*_B2021 - _A2202*_B2001;
    C[502] = _A1212*_B2002 - _A1012*_B2022 + _A2012*_B2021 - _A2212*_B2001;
    C[503] = _A1222*_B2002 - _A1022*_B2022 + _A2022*_B2021 - _A2222*_B2001;
    C[504] = _A1000*_B2012 - _A1100*_B2002 - _A2000*_B2011 + _A2100*_B2001;
    C[505] = _A1010*_B2012 - _A1110*_B2002 - _A2010*_B2011 + _A2110*_B2001;
    C[506] = _A1020*_B2012 - _A1120*_B2002 - _A2020*_B2011 + _A2120*_B2001;
    C[507] = _A1001*_B2012 - _A1101*_B2002 - _A2001*_B2011 + _A2101*_B2001;
    C[508] = _A1011*_B2012 - _A1111*_B2002 - _A2011*_B2011 + _A2111*_B2001;
    C[509] = _A1021*_B2012 - _A1121*_B2002 - _A2021*_B2011 + _A2121*_B2001;
    C[510] = _A1002*_B2012 - _A1102*_B2002 - _A2002*_B2011 + _A2102*_B2001;
    C[511] = _A1012*_B2012 - _A1112*_B2002 - _A2012*_B2011 + _A2112*_B2001;
    C[512] = _A1022*_B2012 - _A1122*_B2002 - _A2022*_B2011 + _A2122*_B2001;
    C[513] = _A0200*_B2012 - _A0100*_B2022 + _A2100*_B2020 - _A2200*_B2010;
    C[514] = _A0210*_B2012 - _A0110*_B2022 + _A2110*_B2020 - _A2210*_B2010;
    C[515] = _A0220*_B2012 - _A0120*_B2022 + _A2120*_B2020 - _A2220*_B2010;
    C[516] = _A0201*_B2012 - _A0101*_B2022 + _A2101*_B2020 - _A2201*_B2010;
    C[517] = _A0211*_B2012 - _A0111*_B2022 + _A2111*_B2020 - _A2211*_B2010;
    C[518] = _A0221*_B2012 - _A0121*_B2022 + _A2121*_B2020 - _A2221*_B2010;
    C[519] = _A0202*_B2012 - _A0102*_B2022 + _A2102*_B2020 - _A2202*_B2010;
    C[520] = _A0212*_B2012 - _A0112*_B2022 + _A2112*_B2020 - _A2212*_B2010;
    C[521] = _A0222*_B2012 - _A0122*_B2022 + _A2122*_B2020 - _A2222*_B2010;
    C[522] = _A0000*_B2022 - _A0200*_B2002 - _A2000*_B2020 + _A2200*_B2000;
    C[523] = _A0010*_B2022 - _A0210*_B2002 - _A2010*_B2020 + _A2210*_B2000;
    C[524] = _A0020*_B2022 - _A0220*_B2002 - _A2020*_B2020 + _A2220*_B2000;
    C[525] = _A0001*_B2022 - _A0201*_B2002 - _A2001*_B2020 + _A2201*_B2000;
    C[526] = _A0011*_B2022 - _A0211*_B2002 - _A2011*_B2020 + _A2211*_B2000;
    C[527] = _A0021*_B2022 - _A0221*_B2002 - _A2021*_B2020 + _A2221*_B2000;
    C[528] = _A0002*_B2022 - _A0202*_B2002 - _A2002*_B2020 + _A2202*_B2000;
    C[529] = _A0012*_B2022 - _A0212*_B2002 - _A2012*_B2020 + _A2212*_B2000;
    C[530] = _A0022*_B2022 - _A0222*_B2002 - _A2022*_B2020 + _A2222*_B2000;
    C[531] = _A0100*_B2002 - _A0000*_B2012 + _A2000*_B2010 - _A2100*_B2000;
    C[532] = _A0110*_B2002 - _A0010*_B2012 + _A2010*_B2010 - _A2110*_B2000;
    C[533] = _A0120*_B2002 - _A0020*_B2012 + _A2020*_B2010 - _A2120*_B2000;
    C[534] = _A0101*_B2002 - _A0001*_B2012 + _A2001*_B2010 - _A2101*_B2000;
    C[535] = _A0111*_B2002 - _A0011*_B2012 + _A2011*_B2010 - _A2111*_B2000;
    C[536] = _A0121*_B2002 - _A0021*_B2012 + _A2021*_B2010 - _A2121*_B2000;
    C[537] = _A0102*_B2002 - _A0002*_B2012 + _A2002*_B2010 - _A2102*_B2000;
    C[538] = _A0112*_B2002 - _A0012*_B2012 + _A2012*_B2010 - _A2112*_B2000;
    C[539] = _A0122*_B2002 - _A0022*_B2012 + _A2022*_B2010 - _A2122*_B2000;
    C[540] = _A0100*_B2021 - _A0200*_B2011 - _A1100*_B2020 + _A1200*_B2010;
    C[541] = _A0110*_B2021 - _A0210*_B2011 - _A1110*_B2020 + _A1210*_B2010;
    C[542] = _A0120*_B2021 - _A0220*_B2011 - _A1120*_B2020 + _A1220*_B2010;
    C[543] = _A0101*_B2021 - _A0201*_B2011 - _A1101*_B2020 + _A1201*_B2010;
    C[544] = _A0111*_B2021 - _A0211*_B2011 - _A1111*_B2020 + _A1211*_B2010;
    C[545] = _A0121*_B2021 - _A0221*_B2011 - _A1121*_B2020 + _A1221*_B2010;
    C[546] = _A0102*_B2021 - _A0202*_B2011 - _A1102*_B2020 + _A1202*_B2010;
    C[547] = _A0112*_B2021 - _A0212*_B2011 - _A1112*_B2020 + _A1212*_B2010;
    C[548] = _A0122*_B2021 - _A0222*_B2011 - _A1122*_B2020 + _A1222*_B2010;
    C[549] = _A0200*_B2001 - _A0000*_B2021 + _A1000*_B2020 - _A1200*_B2000;
    C[550] = _A0210*_B2001 - _A0010*_B2021 + _A1010*_B2020 - _A1210*_B2000;
    C[551] = _A0220*_B2001 - _A0020*_B2021 + _A1020*_B2020 - _A1220*_B2000;
    C[552] = _A0201*_B2001 - _A0001*_B2021 + _A1001*_B2020 - _A1201*_B2000;
    C[553] = _A0211*_B2001 - _A0011*_B2021 + _A1011*_B2020 - _A1211*_B2000;
    C[554] = _A0221*_B2001 - _A0021*_B2021 + _A1021*_B2020 - _A1221*_B2000;
    C[555] = _A0202*_B2001 - _A0002*_B2021 + _A1002*_B2020 - _A1202*_B2000;
    C[556] = _A0212*_B2001 - _A0012*_B2021 + _A1012*_B2020 - _A1212*_B2000;
    C[557] = _A0222*_B2001 - _A0022*_B2021 + _A1022*_B2020 - _A1222*_B2000;
    C[558] = _A0000*_B2011 - _A0100*_B2001 - _A1000*_B2010 + _A1100*_B2000;
    C[559] = _A0010*_B2011 - _A0110*_B2001 - _A1010*_B2010 + _A1110*_B2000;
    C[560] = _A0020*_B2011 - _A0120*_B2001 - _A1020*_B2010 + _A1120*_B2000;
    C[561] = _A0001*_B2011 - _A0101*_B2001 - _A1001*_B2010 + _A1101*_B2000;
    C[562] = _A0011*_B2011 - _A0111*_B2001 - _A1011*_B2010 + _A1111*_B2000;
    C[563] = _A0021*_B2011 - _A0121*_B2001 - _A1021*_B2010 + _A1121*_B2000;
    C[564] = _A0002*_B2011 - _A0102*_B2001 - _A1002*_B2010 + _A1102*_B2000;
    C[565] = _A0012*_B2011 - _A0112*_B2001 - _A1012*_B2010 + _A1112*_B2000;
    C[566] = _A0022*_B2011 - _A0122*_B2001 - _A1022*_B2010 + _A1122*_B2000;
    C[567] = _A1100*_B2122 - _A1200*_B2112 - _A2100*_B2121 + _A2200*_B2111;
    C[568] = _A1110*_B2122 - _A1210*_B2112 - _A2110*_B2121 + _A2210*_B2111;
    C[569] = _A1120*_B2122 - _A1220*_B2112 - _A2120*_B2121 + _A2220*_B2111;
    C[570] = _A1101*_B2122 - _A1201*_B2112 - _A2101*_B2121 + _A2201*_B2111;
    C[571] = _A1111*_B2122 - _A1211*_B2112 - _A2111*_B2121 + _A2211*_B2111;
    C[572] = _A1121*_B2122 - _A1221*_B2112 - _A2121*_B2121 + _A2221*_B2111;
    C[573] = _A1102*_B2122 - _A1202*_B2112 - _A2102*_B2121 + _A2202*_B2111;
    C[574] = _A1112*_B2122 - _A1212*_B2112 - _A2112*_B2121 + _A2212*_B2111;
    C[575] = _A1122*_B2122 - _A1222*_B2112 - _A2122*_B2121 + _A2222*_B2111;
    C[576] = _A1200*_B2102 - _A1000*_B2122 + _A2000*_B2121 - _A2200*_B2101;
    C[577] = _A1210*_B2102 - _A1010*_B2122 + _A2010*_B2121 - _A2210*_B2101;
    C[578] = _A1220*_B2102 - _A1020*_B2122 + _A2020*_B2121 - _A2220*_B2101;
    C[579] = _A1201*_B2102 - _A1001*_B2122 + _A2001*_B2121 - _A2201*_B2101;
    C[580] = _A1211*_B2102 - _A1011*_B2122 + _A2011*_B2121 - _A2211*_B2101;
    C[581] = _A1221*_B2102 - _A1021*_B2122 + _A2021*_B2121 - _A2221*_B2101;
    C[582] = _A1202*_B2102 - _A1002*_B2122 + _A2002*_B2121 - _A2202*_B2101;
    C[583] = _A1212*_B2102 - _A1012*_B2122 + _A2012*_B2121 - _A2212*_B2101;
    C[584] = _A1222*_B2102 - _A1022*_B2122 + _A2022*_B2121 - _A2222*_B2101;
    C[585] = _A1000*_B2112 - _A1100*_B2102 - _A2000*_B2111 + _A2100*_B2101;
    C[586] = _A1010*_B2112 - _A1110*_B2102 - _A2010*_B2111 + _A2110*_B2101;
    C[587] = _A1020*_B2112 - _A1120*_B2102 - _A2020*_B2111 + _A2120*_B2101;
    C[588] = _A1001*_B2112 - _A1101*_B2102 - _A2001*_B2111 + _A2101*_B2101;
    C[589] = _A1011*_B2112 - _A1111*_B2102 - _A2011*_B2111 + _A2111*_B2101;
    C[590] = _A1021*_B2112 - _A1121*_B2102 - _A2021*_B2111 + _A2121*_B2101;
    C[591] = _A1002*_B2112 - _A1102*_B2102 - _A2002*_B2111 + _A2102*_B2101;
    C[592] = _A1012*_B2112 - _A1112*_B2102 - _A2012*_B2111 + _A2112*_B2101;
    C[593] = _A1022*_B2112 - _A1122*_B2102 - _A2022*_B2111 + _A2122*_B2101;
    C[594] = _A0200*_B2112 - _A0100*_B2122 + _A2100*_B2120 - _A2200*_B2110;
    C[595] = _A0210*_B2112 - _A0110*_B2122 + _A2110*_B2120 - _A2210*_B2110;
    C[596] = _A0220*_B2112 - _A0120*_B2122 + _A2120*_B2120 - _A2220*_B2110;
    C[597] = _A0201*_B2112 - _A0101*_B2122 + _A2101*_B2120 - _A2201*_B2110;
    C[598] = _A0211*_B2112 - _A0111*_B2122 + _A2111*_B2120 - _A2211*_B2110;
    C[599] = _A0221*_B2112 - _A0121*_B2122 + _A2121*_B2120 - _A2221*_B2110;
    C[600] = _A0202*_B2112 - _A0102*_B2122 + _A2102*_B2120 - _A2202*_B2110;
    C[601] = _A0212*_B2112 - _A0112*_B2122 + _A2112*_B2120 - _A2212*_B2110;
    C[602] = _A0222*_B2112 - _A0122*_B2122 + _A2122*_B2120 - _A2222*_B2110;
    C[603] = _A0000*_B2122 - _A0200*_B2102 - _A2000*_B2120 + _A2200*_B2100;
    C[604] = _A0010*_B2122 - _A0210*_B2102 - _A2010*_B2120 + _A2210*_B2100;
    C[605] = _A0020*_B2122 - _A0220*_B2102 - _A2020*_B2120 + _A2220*_B2100;
    C[606] = _A0001*_B2122 - _A0201*_B2102 - _A2001*_B2120 + _A2201*_B2100;
    C[607] = _A0011*_B2122 - _A0211*_B2102 - _A2011*_B2120 + _A2211*_B2100;
    C[608] = _A0021*_B2122 - _A0221*_B2102 - _A2021*_B2120 + _A2221*_B2100;
    C[609] = _A0002*_B2122 - _A0202*_B2102 - _A2002*_B2120 + _A2202*_B2100;
    C[610] = _A0012*_B2122 - _A0212*_B2102 - _A2012*_B2120 + _A2212*_B2100;
    C[611] = _A0022*_B2122 - _A0222*_B2102 - _A2022*_B2120 + _A2222*_B2100;
    C[612] = _A0100*_B2102 - _A0000*_B2112 + _A2000*_B2110 - _A2100*_B2100;
    C[613] = _A0110*_B2102 - _A0010*_B2112 + _A2010*_B2110 - _A2110*_B2100;
    C[614] = _A0120*_B2102 - _A0020*_B2112 + _A2020*_B2110 - _A2120*_B2100;
    C[615] = _A0101*_B2102 - _A0001*_B2112 + _A2001*_B2110 - _A2101*_B2100;
    C[616] = _A0111*_B2102 - _A0011*_B2112 + _A2011*_B2110 - _A2111*_B2100;
    C[617] = _A0121*_B2102 - _A0021*_B2112 + _A2021*_B2110 - _A2121*_B2100;
    C[618] = _A0102*_B2102 - _A0002*_B2112 + _A2002*_B2110 - _A2102*_B2100;
    C[619] = _A0112*_B2102 - _A0012*_B2112 + _A2012*_B2110 - _A2112*_B2100;
    C[620] = _A0122*_B2102 - _A0022*_B2112 + _A2022*_B2110 - _A2122*_B2100;
    C[621] = _A0100*_B2121 - _A0200*_B2111 - _A1100*_B2120 + _A1200*_B2110;
    C[622] = _A0110*_B2121 - _A0210*_B2111 - _A1110*_B2120 + _A1210*_B2110;
    C[623] = _A0120*_B2121 - _A0220*_B2111 - _A1120*_B2120 + _A1220*_B2110;
    C[624] = _A0101*_B2121 - _A0201*_B2111 - _A1101*_B2120 + _A1201*_B2110;
    C[625] = _A0111*_B2121 - _A0211*_B2111 - _A1111*_B2120 + _A1211*_B2110;
    C[626] = _A0121*_B2121 - _A0221*_B2111 - _A1121*_B2120 + _A1221*_B2110;
    C[627] = _A0102*_B2121 - _A0202*_B2111 - _A1102*_B2120 + _A1202*_B2110;
    C[628] = _A0112*_B2121 - _A0212*_B2111 - _A1112*_B2120 + _A1212*_B2110;
    C[629] = _A0122*_B2121 - _A0222*_B2111 - _A1122*_B2120 + _A1222*_B2110;
    C[630] = _A0200*_B2101 - _A0000*_B2121 + _A1000*_B2120 - _A1200*_B2100;
    C[631] = _A0210*_B2101 - _A0010*_B2121 + _A1010*_B2120 - _A1210*_B2100;
    C[632] = _A0220*_B2101 - _A0020*_B2121 + _A1020*_B2120 - _A1220*_B2100;
    C[633] = _A0201*_B2101 - _A0001*_B2121 + _A1001*_B2120 - _A1201*_B2100;
    C[634] = _A0211*_B2101 - _A0011*_B2121 + _A1011*_B2120 - _A1211*_B2100;
    C[635] = _A0221*_B2101 - _A0021*_B2121 + _A1021*_B2120 - _A1221*_B2100;
    C[636] = _A0202*_B2101 - _A0002*_B2121 + _A1002*_B2120 - _A1202*_B2100;
    C[637] = _A0212*_B2101 - _A0012*_B2121 + _A1012*_B2120 - _A1212*_B2100;
    C[638] = _A0222*_B2101 - _A0022*_B2121 + _A1022*_B2120 - _A1222*_B2100;
    C[639] = _A0000*_B2111 - _A0100*_B2101 - _A1000*_B2110 + _A1100*_B2100;
    C[640] = _A0010*_B2111 - _A0110*_B2101 - _A1010*_B2110 + _A1110*_B2100;
    C[641] = _A0020*_B2111 - _A0120*_B2101 - _A1020*_B2110 + _A1120*_B2100;
    C[642] = _A0001*_B2111 - _A0101*_B2101 - _A1001*_B2110 + _A1101*_B2100;
    C[643] = _A0011*_B2111 - _A0111*_B2101 - _A1011*_B2110 + _A1111*_B2100;
    C[644] = _A0021*_B2111 - _A0121*_B2101 - _A1021*_B2110 + _A1121*_B2100;
    C[645] = _A0002*_B2111 - _A0102*_B2101 - _A1002*_B2110 + _A1102*_B2100;
    C[646] = _A0012*_B2111 - _A0112*_B2101 - _A1012*_B2110 + _A1112*_B2100;
    C[647] = _A0022*_B2111 - _A0122*_B2101 - _A1022*_B2110 + _A1122*_B2100;
    C[648] = _A1100*_B2222 - _A1200*_B2212 - _A2100*_B2221 + _A2200*_B2211;
    C[649] = _A1110*_B2222 - _A1210*_B2212 - _A2110*_B2221 + _A2210*_B2211;
    C[650] = _A1120*_B2222 - _A1220*_B2212 - _A2120*_B2221 + _A2220*_B2211;
    C[651] = _A1101*_B2222 - _A1201*_B2212 - _A2101*_B2221 + _A2201*_B2211;
    C[652] = _A1111*_B2222 - _A1211*_B2212 - _A2111*_B2221 + _A2211*_B2211;
    C[653] = _A1121*_B2222 - _A1221*_B2212 - _A2121*_B2221 + _A2221*_B2211;
    C[654] = _A1102*_B2222 - _A1202*_B2212 - _A2102*_B2221 + _A2202*_B2211;
    C[655] = _A1112*_B2222 - _A1212*_B2212 - _A2112*_B2221 + _A2212*_B2211;
    C[656] = _A1122*_B2222 - _A1222*_B2212 - _A2122*_B2221 + _A2222*_B2211;
    C[657] = _A1200*_B2202 - _A1000*_B2222 + _A2000*_B2221 - _A2200*_B2201;
    C[658] = _A1210*_B2202 - _A1010*_B2222 + _A2010*_B2221 - _A2210*_B2201;
    C[659] = _A1220*_B2202 - _A1020*_B2222 + _A2020*_B2221 - _A2220*_B2201;
    C[660] = _A1201*_B2202 - _A1001*_B2222 + _A2001*_B2221 - _A2201*_B2201;
    C[661] = _A1211*_B2202 - _A1011*_B2222 + _A2011*_B2221 - _A2211*_B2201;
    C[662] = _A1221*_B2202 - _A1021*_B2222 + _A2021*_B2221 - _A2221*_B2201;
    C[663] = _A1202*_B2202 - _A1002*_B2222 + _A2002*_B2221 - _A2202*_B2201;
    C[664] = _A1212*_B2202 - _A1012*_B2222 + _A2012*_B2221 - _A2212*_B2201;
    C[665] = _A1222*_B2202 - _A1022*_B2222 + _A2022*_B2221 - _A2222*_B2201;
    C[666] = _A1000*_B2212 - _A1100*_B2202 - _A2000*_B2211 + _A2100*_B2201;
    C[667] = _A1010*_B2212 - _A1110*_B2202 - _A2010*_B2211 + _A2110*_B2201;
    C[668] = _A1020*_B2212 - _A1120*_B2202 - _A2020*_B2211 + _A2120*_B2201;
    C[669] = _A1001*_B2212 - _A1101*_B2202 - _A2001*_B2211 + _A2101*_B2201;
    C[670] = _A1011*_B2212 - _A1111*_B2202 - _A2011*_B2211 + _A2111*_B2201;
    C[671] = _A1021*_B2212 - _A1121*_B2202 - _A2021*_B2211 + _A2121*_B2201;
    C[672] = _A1002*_B2212 - _A1102*_B2202 - _A2002*_B2211 + _A2102*_B2201;
    C[673] = _A1012*_B2212 - _A1112*_B2202 - _A2012*_B2211 + _A2112*_B2201;
    C[674] = _A1022*_B2212 - _A1122*_B2202 - _A2022*_B2211 + _A2122*_B2201;
    C[675] = _A0200*_B2212 - _A0100*_B2222 + _A2100*_B2220 - _A2200*_B2210;
    C[676] = _A0210*_B2212 - _A0110*_B2222 + _A2110*_B2220 - _A2210*_B2210;
    C[677] = _A0220*_B2212 - _A0120*_B2222 + _A2120*_B2220 - _A2220*_B2210;
    C[678] = _A0201*_B2212 - _A0101*_B2222 + _A2101*_B2220 - _A2201*_B2210;
    C[679] = _A0211*_B2212 - _A0111*_B2222 + _A2111*_B2220 - _A2211*_B2210;
    C[680] = _A0221*_B2212 - _A0121*_B2222 + _A2121*_B2220 - _A2221*_B2210;
    C[681] = _A0202*_B2212 - _A0102*_B2222 + _A2102*_B2220 - _A2202*_B2210;
    C[682] = _A0212*_B2212 - _A0112*_B2222 + _A2112*_B2220 - _A2212*_B2210;
    C[683] = _A0222*_B2212 - _A0122*_B2222 + _A2122*_B2220 - _A2222*_B2210;
    C[684] = _A0000*_B2222 - _A0200*_B2202 - _A2000*_B2220 + _A2200*_B2200;
    C[685] = _A0010*_B2222 - _A0210*_B2202 - _A2010*_B2220 + _A2210*_B2200;
    C[686] = _A0020*_B2222 - _A0220*_B2202 - _A2020*_B2220 + _A2220*_B2200;
    C[687] = _A0001*_B2222 - _A0201*_B2202 - _A2001*_B2220 + _A2201*_B2200;
    C[688] = _A0011*_B2222 - _A0211*_B2202 - _A2011*_B2220 + _A2211*_B2200;
    C[689] = _A0021*_B2222 - _A0221*_B2202 - _A2021*_B2220 + _A2221*_B2200;
    C[690] = _A0002*_B2222 - _A0202*_B2202 - _A2002*_B2220 + _A2202*_B2200;
    C[691] = _A0012*_B2222 - _A0212*_B2202 - _A2012*_B2220 + _A2212*_B2200;
    C[692] = _A0022*_B2222 - _A0222*_B2202 - _A2022*_B2220 + _A2222*_B2200;
    C[693] = _A0100*_B2202 - _A0000*_B2212 + _A2000*_B2210 - _A2100*_B2200;
    C[694] = _A0110*_B2202 - _A0010*_B2212 + _A2010*_B2210 - _A2110*_B2200;
    C[695] = _A0120*_B2202 - _A0020*_B2212 + _A2020*_B2210 - _A2120*_B2200;
    C[696] = _A0101*_B2202 - _A0001*_B2212 + _A2001*_B2210 - _A2101*_B2200;
    C[697] = _A0111*_B2202 - _A0011*_B2212 + _A2011*_B2210 - _A2111*_B2200;
    C[698] = _A0121*_B2202 - _A0021*_B2212 + _A2021*_B2210 - _A2121*_B2200;
    C[699] = _A0102*_B2202 - _A0002*_B2212 + _A2002*_B2210 - _A2102*_B2200;
    C[700] = _A0112*_B2202 - _A0012*_B2212 + _A2012*_B2210 - _A2112*_B2200;
    C[701] = _A0122*_B2202 - _A0022*_B2212 + _A2022*_B2210 - _A2122*_B2200;
    C[702] = _A0100*_B2221 - _A0200*_B2211 - _A1100*_B2220 + _A1200*_B2210;
    C[703] = _A0110*_B2221 - _A0210*_B2211 - _A1110*_B2220 + _A1210*_B2210;
    C[704] = _A0120*_B2221 - _A0220*_B2211 - _A1120*_B2220 + _A1220*_B2210;
    C[705] = _A0101*_B2221 - _A0201*_B2211 - _A1101*_B2220 + _A1201*_B2210;
    C[706] = _A0111*_B2221 - _A0211*_B2211 - _A1111*_B2220 + _A1211*_B2210;
    C[707] = _A0121*_B2221 - _A0221*_B2211 - _A1121*_B2220 + _A1221*_B2210;
    C[708] = _A0102*_B2221 - _A0202*_B2211 - _A1102*_B2220 + _A1202*_B2210;
    C[709] = _A0112*_B2221 - _A0212*_B2211 - _A1112*_B2220 + _A1212*_B2210;
    C[710] = _A0122*_B2221 - _A0222*_B2211 - _A1122*_B2220 + _A1222*_B2210;
    C[711] = _A0200*_B2201 - _A0000*_B2221 + _A1000*_B2220 - _A1200*_B2200;
    C[712] = _A0210*_B2201 - _A0010*_B2221 + _A1010*_B2220 - _A1210*_B2200;
    C[713] = _A0220*_B2201 - _A0020*_B2221 + _A1020*_B2220 - _A1220*_B2200;
    C[714] = _A0201*_B2201 - _A0001*_B2221 + _A1001*_B2220 - _A1201*_B2200;
    C[715] = _A0211*_B2201 - _A0011*_B2221 + _A1011*_B2220 - _A1211*_B2200;
    C[716] = _A0221*_B2201 - _A0021*_B2221 + _A1021*_B2220 - _A1221*_B2200;
    C[717] = _A0202*_B2201 - _A0002*_B2221 + _A1002*_B2220 - _A1202*_B2200;
    C[718] = _A0212*_B2201 - _A0012*_B2221 + _A1012*_B2220 - _A1212*_B2200;
    C[719] = _A0222*_B2201 - _A0022*_B2221 + _A1022*_B2220 - _A1222*_B2200;
    C[720] = _A0000*_B2211 - _A0100*_B2201 - _A1000*_B2210 + _A1100*_B2200;
    C[721] = _A0010*_B2211 - _A0110*_B2201 - _A1010*_B2210 + _A1110*_B2200;
    C[722] = _A0020*_B2211 - _A0120*_B2201 - _A1020*_B2210 + _A1120*_B2200;
    C[723] = _A0001*_B2211 - _A0101*_B2201 - _A1001*_B2210 + _A1101*_B2200;
    C[724] = _A0011*_B2211 - _A0111*_B2201 - _A1011*_B2210 + _A1111*_B2200;
    C[725] = _A0021*_B2211 - _A0121*_B2201 - _A1021*_B2210 + _A1121*_B2200;
    C[726] = _A0002*_B2211 - _A0102*_B2201 - _A1002*_B2210 + _A1102*_B2200;
    C[727] = _A0012*_B2211 - _A0112*_B2201 - _A1012*_B2210 + _A1112*_B2200;
    C[728] = _A0022*_B2211 - _A0122*_B2201 - _A1022*_B2210 + _A1122*_B2200;
}


}


#endif // TENSOR_CROSS_H

