#ifndef INVERSE_H
#define INVERSE_H

#include "Fastor/commons/commons.h"
#include "Fastor/extended_intrinsics/extintrin.h"

namespace Fastor {

template<typename T, size_t N>
FASTOR_INLINE void _inverse(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst);

template<>
FASTOR_INLINE void _inverse<float,2>(const float *FASTOR_RESTRICT src, float *FASTOR_RESTRICT dst)
{
    float det;

    float src0 = src[0];
    float src1 = src[1];
    float src2 = src[2];
    float src3 = src[3];

    /* Compute adjoint: */
    dst[0] = + src3;
    dst[1] = - src1;
    dst[2] = - src2;
    dst[3] = + src0;

    /* Compute determinant: */
    det = src0 * dst[0] + src1 * dst[2];

    /* Multiply adjoint with reciprocal of determinant: */
    det = static_cast<float>(1.0) / det;
    dst[0] *= det;
    dst[1] *= det;
    dst[2] *= det;
    dst[3] *= det;
#ifdef FASTOR_AVX_IMPL
    // This might hurt the performance as the data is already loaded
//    _mm256_store_pd(dst,_mm256_mul_pd(_mm256_load_pd(dst),_mm256_set1_pd(det)));
#endif
}


template<>
FASTOR_INLINE void _inverse<float,3>(const float *FASTOR_RESTRICT src, float *FASTOR_RESTRICT dst)
{
    float det;

    float src0 = src[0];
    float src1 = src[1];
    float src2 = src[2];
    float src3 = src[3];
    float src4 = src[4];
    float src5 = src[5];
    float src6 = src[6];
    float src7 = src[7];
    float src8 = src[8];

    /* Compute adjoint: */
    dst[0] = + src4 * src8 - src5 * src7;
    dst[1] = - src1 * src8 + src2 * src7;
    dst[2] = + src1 * src5 - src2 * src4;
    dst[3] = - src3 * src8 + src5 * src6;
    dst[4] = + src0 * src8 - src2 * src6;
    dst[5] = - src0 * src5 + src2 * src3;
    dst[6] = + src3 * src7 - src4 * src6;
    dst[7] = - src0 * src7 + src1 * src6;
    dst[8] = + src0 * src4 - src1 * src3;

    /* Compute determinant: */
    det = src0 * dst[0] + src1 * dst[3] + src2 * dst[6];

    /* Multiply adjoint with reciprocal of determinant: */
    det = static_cast<float>(1.0) / det;

    dst[0] *= det;
    dst[1] *= det;
    dst[2] *= det;
    dst[3] *= det;
    dst[4] *= det;
    dst[5] *= det;
    dst[6] *= det;
    dst[7] *= det;
    dst[8] *= det;
}

template<>
FASTOR_INLINE void _inverse<double,2>(const double *FASTOR_RESTRICT src, double *FASTOR_RESTRICT dst)
{
    double det;

    double src0 = src[0];
    double src1 = src[1];
    double src2 = src[2];
    double src3 = src[3];

    /* Compute adjoint: */
    dst[0] = + src3;
    dst[1] = - src1;
    dst[2] = - src2;
    dst[3] = + src0;

    /* Compute determinant: */
    det = src0 * dst[0] + src1 * dst[2];

    /* Multiply adjoint with reciprocal of determinant: */
    det = static_cast<double>(1.0) / det;
    dst[0] *= det;
    dst[1] *= det;
    dst[2] *= det;
    dst[3] *= det;
#ifdef FASTOR_AVX_IMPL
    // This might hurt the performance as the data is already loaded
//    _mm256_store_pd(dst,_mm256_mul_pd(_mm256_load_pd(dst),_mm256_set1_pd(det)));
#endif
}


template<>
FASTOR_INLINE void _inverse<double,3>(const double *FASTOR_RESTRICT src, double *FASTOR_RESTRICT dst)
{
    double det;

    double src0 = src[0];
    double src1 = src[1];
    double src2 = src[2];
    double src3 = src[3];
    double src4 = src[4];
    double src5 = src[5];
    double src6 = src[6];
    double src7 = src[7];
    double src8 = src[8];

    /* Compute adjoint: */
    dst[0] = + src4 * src8 - src5 * src7;
    dst[1] = - src1 * src8 + src2 * src7;
    dst[2] = + src1 * src5 - src2 * src4;
    dst[3] = - src3 * src8 + src5 * src6;
    dst[4] = + src0 * src8 - src2 * src6;
    dst[5] = - src0 * src5 + src2 * src3;
    dst[6] = + src3 * src7 - src4 * src6;
    dst[7] = - src0 * src7 + src1 * src6;
    dst[8] = + src0 * src4 - src1 * src3;

    /* Compute determinant: */
    det = src0 * dst[0] + src1 * dst[3] + src2 * dst[6];

    /* Multiply adjoint with reciprocal of determinant: */
    det = static_cast<double>(1.0) / det;

    dst[0] *= det;
    dst[1] *= det;
    dst[2] *= det;
    dst[3] *= det;
    dst[4] *= det;
    dst[5] *= det;
    dst[6] *= det;
    dst[7] *= det;
    dst[8] *= det;

#ifdef FASTOR_AVX_IMPL
    // This might hurt the performance as the data is already loaded
//    __m256d v0 = _mm256_set1_pd(det);
//    _mm256_store_pd(dst,_mm256_mul_pd(_mm256_load_pd(dst),v0));
//    _mm256_store_pd(dst+4,_mm256_mul_pd(_mm256_load_pd(dst+4),v0));
//    _mm_store_sd(dst+8,_mm_mul_sd(_mm_load_sd(dst+8), _mm256_castpd256_pd128(v0)));
#endif

}

#if FASTOR_NIL
#ifdef FASTOR_SSE4_2_IMPL

FASTOR_INLINE __m128 _mm_dot_ps(__m128 v1, __m128 v2)
{
   __m128 mul0 = _mm_mul_ps(v1, v2);
   __m128 swp0 = _mm_shuffle_ps(mul0, mul0, _MM_SHUFFLE(2, 3, 0, 1));
   __m128 add0 = _mm_add_ps(mul0, swp0);
   __m128 swp1 = _mm_shuffle_ps(add0, add0, _MM_SHUFFLE(0, 1, 2, 3));
   __m128 add1 = _mm_add_ps(add0, swp1);
   return add1;
}

FASTOR_INLINE void _inverse_4x4_sse(__m128 const in[4], __m128 out[4])
{
    __m128 Fac0;
    {
        //  valType SubFactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
        //  valType SubFactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
        //  valType SubFactor06 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
        //  valType SubFactor13 = m[1][2] * m[2][3] - m[2][2] * m[1][3];

        __m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(3, 3, 3, 3));
        __m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(2, 2, 2, 2));

        __m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(2, 2, 2, 2));
        __m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
        __m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
        __m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(3, 3, 3, 3));

        __m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
        __m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
        Fac0 = _mm_sub_ps(Mul00, Mul01);

//        bool stop = true;
    }

    __m128 Fac1;
    {
        //  valType SubFactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
        //  valType SubFactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
        //  valType SubFactor07 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
        //  valType SubFactor14 = m[1][1] * m[2][3] - m[2][1] * m[1][3];

        __m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(3, 3, 3, 3));
        __m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(1, 1, 1, 1));

        __m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(1, 1, 1, 1));
        __m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
        __m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
        __m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(3, 3, 3, 3));

        __m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
        __m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
        Fac1 = _mm_sub_ps(Mul00, Mul01);

//         bool stop = true;
    }


    __m128 Fac2;
    {
        //  valType SubFactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
        //  valType SubFactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
        //  valType SubFactor08 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
        //  valType SubFactor15 = m[1][1] * m[2][2] - m[2][1] * m[1][2];

        __m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(2, 2, 2, 2));
        __m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(1, 1, 1, 1));

        __m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(1, 1, 1, 1));
        __m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
        __m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
        __m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(2, 2, 2, 2));

        __m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
        __m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
        Fac2 = _mm_sub_ps(Mul00, Mul01);

//        bool stop = true;
    }

    __m128 Fac3;
    {
        //  valType SubFactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
        //  valType SubFactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
        //  valType SubFactor09 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
        //  valType SubFactor16 = m[1][0] * m[2][3] - m[2][0] * m[1][3];

        __m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(3, 3, 3, 3));
        __m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(0, 0, 0, 0));

        __m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(0, 0, 0, 0));
        __m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
        __m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
        __m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(3, 3, 3, 3));

        __m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
        __m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
        Fac3 = _mm_sub_ps(Mul00, Mul01);

//        bool stop = true;
    }

    __m128 Fac4;
    {
        //  valType SubFactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
        //  valType SubFactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
        //  valType SubFactor10 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
        //  valType SubFactor17 = m[1][0] * m[2][2] - m[2][0] * m[1][2];

        __m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(2, 2, 2, 2));
        __m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(0, 0, 0, 0));

        __m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(0, 0, 0, 0));
        __m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
        __m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
        __m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(2, 2, 2, 2));

        __m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
        __m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
        Fac4 = _mm_sub_ps(Mul00, Mul01);

//        bool stop = true;
    }

    __m128 Fac5;
    {
        //  valType SubFactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
        //  valType SubFactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
        //  valType SubFactor12 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
        //  valType SubFactor18 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

        __m128 Swp0a = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(1, 1, 1, 1));
        __m128 Swp0b = _mm_shuffle_ps(in[3], in[2], _MM_SHUFFLE(0, 0, 0, 0));

        __m128 Swp00 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(0, 0, 0, 0));
        __m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
        __m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
        __m128 Swp03 = _mm_shuffle_ps(in[2], in[1], _MM_SHUFFLE(1, 1, 1, 1));

        __m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
        __m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
        Fac5 = _mm_sub_ps(Mul00, Mul01);

//        bool stop = true;
    }

    __m128 SignA = _mm_set_ps( 1.0f,-1.0f, 1.0f,-1.0f);
    __m128 SignB = _mm_set_ps(-1.0f, 1.0f,-1.0f, 1.0f);

    // m[1][0]
    // m[0][0]
    // m[0][0]
    // m[0][0]
    __m128 Temp0 = _mm_shuffle_ps(in[1], in[0], _MM_SHUFFLE(0, 0, 0, 0));
    __m128 Vec0 = _mm_shuffle_ps(Temp0, Temp0, _MM_SHUFFLE(2, 2, 2, 0));

    // m[1][1]
    // m[0][1]
    // m[0][1]
    // m[0][1]
    __m128 Temp1 = _mm_shuffle_ps(in[1], in[0], _MM_SHUFFLE(1, 1, 1, 1));
    __m128 Vec1 = _mm_shuffle_ps(Temp1, Temp1, _MM_SHUFFLE(2, 2, 2, 0));

    // m[1][2]
    // m[0][2]
    // m[0][2]
    // m[0][2]
    __m128 Temp2 = _mm_shuffle_ps(in[1], in[0], _MM_SHUFFLE(2, 2, 2, 2));
    __m128 Vec2 = _mm_shuffle_ps(Temp2, Temp2, _MM_SHUFFLE(2, 2, 2, 0));

    // m[1][3]
    // m[0][3]
    // m[0][3]
    // m[0][3]
    __m128 Temp3 = _mm_shuffle_ps(in[1], in[0], _MM_SHUFFLE(3, 3, 3, 3));
    __m128 Vec3 = _mm_shuffle_ps(Temp3, Temp3, _MM_SHUFFLE(2, 2, 2, 0));

    // col0
    // + (Vec1[0] * Fac0[0] - Vec2[0] * Fac1[0] + Vec3[0] * Fac2[0]),
    // - (Vec1[1] * Fac0[1] - Vec2[1] * Fac1[1] + Vec3[1] * Fac2[1]),
    // + (Vec1[2] * Fac0[2] - Vec2[2] * Fac1[2] + Vec3[2] * Fac2[2]),
    // - (Vec1[3] * Fac0[3] - Vec2[3] * Fac1[3] + Vec3[3] * Fac2[3]),
    __m128 Mul00 = _mm_mul_ps(Vec1, Fac0);
    __m128 Mul01 = _mm_mul_ps(Vec2, Fac1);
    __m128 Mul02 = _mm_mul_ps(Vec3, Fac2);
    __m128 Sub00 = _mm_sub_ps(Mul00, Mul01);
    __m128 Add00 = _mm_add_ps(Sub00, Mul02);
    __m128 Inv0 = _mm_mul_ps(SignB, Add00);

    // col1
    // - (Vec0[0] * Fac0[0] - Vec2[0] * Fac3[0] + Vec3[0] * Fac4[0]),
    // + (Vec0[0] * Fac0[1] - Vec2[1] * Fac3[1] + Vec3[1] * Fac4[1]),
    // - (Vec0[0] * Fac0[2] - Vec2[2] * Fac3[2] + Vec3[2] * Fac4[2]),
    // + (Vec0[0] * Fac0[3] - Vec2[3] * Fac3[3] + Vec3[3] * Fac4[3]),
    __m128 Mul03 = _mm_mul_ps(Vec0, Fac0);
    __m128 Mul04 = _mm_mul_ps(Vec2, Fac3);
    __m128 Mul05 = _mm_mul_ps(Vec3, Fac4);
    __m128 Sub01 = _mm_sub_ps(Mul03, Mul04);
    __m128 Add01 = _mm_add_ps(Sub01, Mul05);
    __m128 Inv1 = _mm_mul_ps(SignA, Add01);

    // col2
    // + (Vec0[0] * Fac1[0] - Vec1[0] * Fac3[0] + Vec3[0] * Fac5[0]),
    // - (Vec0[0] * Fac1[1] - Vec1[1] * Fac3[1] + Vec3[1] * Fac5[1]),
    // + (Vec0[0] * Fac1[2] - Vec1[2] * Fac3[2] + Vec3[2] * Fac5[2]),
    // - (Vec0[0] * Fac1[3] - Vec1[3] * Fac3[3] + Vec3[3] * Fac5[3]),
    __m128 Mul06 = _mm_mul_ps(Vec0, Fac1);
    __m128 Mul07 = _mm_mul_ps(Vec1, Fac3);
    __m128 Mul08 = _mm_mul_ps(Vec3, Fac5);
    __m128 Sub02 = _mm_sub_ps(Mul06, Mul07);
    __m128 Add02 = _mm_add_ps(Sub02, Mul08);
    __m128 Inv2 = _mm_mul_ps(SignB, Add02);

    // col3
    // - (Vec1[0] * Fac2[0] - Vec1[0] * Fac4[0] + Vec2[0] * Fac5[0]),
    // + (Vec1[0] * Fac2[1] - Vec1[1] * Fac4[1] + Vec2[1] * Fac5[1]),
    // - (Vec1[0] * Fac2[2] - Vec1[2] * Fac4[2] + Vec2[2] * Fac5[2]),
    // + (Vec1[0] * Fac2[3] - Vec1[3] * Fac4[3] + Vec2[3] * Fac5[3]));
    __m128 Mul09 = _mm_mul_ps(Vec0, Fac2);
    __m128 Mul10 = _mm_mul_ps(Vec1, Fac4);
    __m128 Mul11 = _mm_mul_ps(Vec2, Fac5);
    __m128 Sub03 = _mm_sub_ps(Mul09, Mul10);
    __m128 Add03 = _mm_add_ps(Sub03, Mul11);
    __m128 Inv3 = _mm_mul_ps(SignA, Add03);

    __m128 Row0 = _mm_shuffle_ps(Inv0, Inv1, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 Row1 = _mm_shuffle_ps(Inv2, Inv3, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 Row2 = _mm_shuffle_ps(Row0, Row1, _MM_SHUFFLE(2, 0, 2, 0));

    //  valType Determinant = m[0][0] * Inverse[0][0]
    //                      + m[0][1] * Inverse[1][0]
    //                      + m[0][2] * Inverse[2][0]
    //                      + m[0][3] * Inverse[3][0];
#ifdef FASTOR_SSE4_1_IMPL
    __m128 Det0 = _mm_dp_ps(in[0], Row2, 0xff);
#else
    __m128 Det0 = _mm_dot_ps(in[0], Row2);
#endif
    // Avoid compilation error by using _mm_dot_ps instead
    // __m128 Det0 = _mm_dot_ps(in[0], Row2);
    __m128 Rcp0 = _mm_div_ps(ONEPS, Det0);
    //__m128 Rcp0 = _mm_rcp_ps(Det0);

    //  Inverse /= Determinant;
    out[0] = _mm_mul_ps(Inv0, Rcp0);
    out[1] = _mm_mul_ps(Inv1, Rcp0);
    out[2] = _mm_mul_ps(Inv2, Rcp0);
    out[3] = _mm_mul_ps(Inv3, Rcp0);
}

#endif
#endif


template<typename T>
FASTOR_INLINE void _inverse_4x4_scalar(const T *FASTOR_RESTRICT m, T *FASTOR_RESTRICT invOut)
{
    T inv[16], det;
    int i;

    inv[0] = m[5]  * m[10] * m[15] -
             m[5]  * m[11] * m[14] -
             m[9]  * m[6]  * m[15] +
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] -
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] +
              m[4]  * m[11] * m[14] +
              m[8]  * m[6]  * m[15] -
              m[8]  * m[7]  * m[14] -
              m[12] * m[6]  * m[11] +
              m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] -
             m[4]  * m[11] * m[13] -
             m[8]  * m[5] * m[15] +
             m[8]  * m[7] * m[13] +
             m[12] * m[5] * m[11] -
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] +
               m[4]  * m[10] * m[13] +
               m[8]  * m[5] * m[14] -
               m[8]  * m[6] * m[13] -
               m[12] * m[5] * m[10] +
               m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] +
              m[1]  * m[11] * m[14] +
              m[9]  * m[2] * m[15] -
              m[9]  * m[3] * m[14] -
              m[13] * m[2] * m[11] +
              m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] -
             m[0]  * m[11] * m[14] -
             m[8]  * m[2] * m[15] +
             m[8]  * m[3] * m[14] +
             m[12] * m[2] * m[11] -
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] +
              m[0]  * m[11] * m[13] +
              m[8]  * m[1] * m[15] -
              m[8]  * m[3] * m[13] -
              m[12] * m[1] * m[11] +
              m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] -
              m[0]  * m[10] * m[13] -
              m[8]  * m[1] * m[14] +
              m[8]  * m[2] * m[13] +
              m[12] * m[1] * m[10] -
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] -
             m[1]  * m[7] * m[14] -
             m[5]  * m[2] * m[15] +
             m[5]  * m[3] * m[14] +
             m[13] * m[2] * m[7] -
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] +
              m[0]  * m[7] * m[14] +
              m[4]  * m[2] * m[15] -
              m[4]  * m[3] * m[14] -
              m[12] * m[2] * m[7] +
              m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] -
              m[0]  * m[7] * m[13] -
              m[4]  * m[1] * m[15] +
              m[4]  * m[3] * m[13] +
              m[12] * m[1] * m[7] -
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] +
               m[0]  * m[6] * m[13] +
               m[4]  * m[1] * m[14] -
               m[4]  * m[2] * m[13] -
               m[12] * m[1] * m[6] +
               m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] +
              m[1] * m[7] * m[10] +
              m[5] * m[2] * m[11] -
              m[5] * m[3] * m[10] -
              m[9] * m[2] * m[7] +
              m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] -
             m[0] * m[7] * m[10] -
             m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] -
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] +
               m[0] * m[7] * m[9] +
               m[4] * m[1] * m[11] -
               m[4] * m[3] * m[9] -
               m[8] * m[1] * m[7] +
               m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] -
              m[0] * m[6] * m[9] -
              m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] -
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];


    det = T(1.0) / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;
}


template<>
FASTOR_INLINE void _inverse<float,4>(const float *FASTOR_RESTRICT src, float *FASTOR_RESTRICT dst)
{
    _inverse_4x4_scalar<float>(src,dst);
}

template<>
FASTOR_INLINE void _inverse<double,4>(const double *FASTOR_RESTRICT src, double *FASTOR_RESTRICT dst)
{
    _inverse_4x4_scalar<double>(src,dst);
}

} // end of namespace Fastor

#endif // INVERSE_H

