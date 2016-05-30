#ifndef EXT_INTRIN_H
#define EXT_INTRIN_H

#include "commons/commons.h"

//!                HANDY EXTENSION TO INTRINSICS
//!---------------------------------------------------------------//
#ifdef USE_HADD
//! Horizontal summation of registers
FASTOR_INLINE float _mm_sum_ps(__m128 a) {
    // 10 OPS
    float sum32;
    __m128 sum = _mm_hadd_ps(a, a);
    _mm_store_ss(&sum32,_mm_hadd_ps(sum, sum));
    return sum32;
}
FASTOR_INLINE double _mm_sum_pd(__m128d a) {
    // 5 OPS
    double sum64;
    _mm_store_sd(&sum64,_mm_hadd_pd(a, a));
    return sum64;
}
#else
FASTOR_INLINE float _mm_sum_ps(__m128 a) {
    // 8 OPS
    __m128 shuf = _mm_movehdup_ps(a);
    __m128 sums = _mm_add_ps(a, shuf);
    shuf        = _mm_movehl_ps(shuf, sums);
    sums        = _mm_add_ss(sums, shuf);
    return        _mm_cvtss_f32(sums);
}
FASTOR_INLINE double _mm_sum_pd(__m128d a) {
    // 4 OPS
    __m128 shuftmp= _mm_movehl_ps(ZEROPS, _mm_castpd_ps(a));
    __m128d shuf  = _mm_castps_pd(shuftmp);
    return  _mm_cvtsd_f64(_mm_add_sd(a, shuf));
}
#endif
FASTOR_INLINE float _mm256_sum_ps(__m256 a) {
//#ifdef USE_HADD
    // IVY 14 OPS - HW 16 OPS
    __m256 sum = _mm256_hadd_ps(a, a);
    sum = _mm256_hadd_ps(sum, sum);
    __m128 sum_high = _mm256_extractf128_ps(sum, 0x1);
    __m128 result = _mm_add_ps(sum_high, _mm256_castps256_ps128(sum));
    return _mm_cvtss_f32(result);
//#else
//    // IVY 19 OPS - Ends up being more expensive
//    return _mm_sum_ps(_mm256_castps256_ps128(a)) +
//            _mm_sum_ps(_mm256_extractf128_ps(a,0x1));
//#endif
}
FASTOR_INLINE double _mm256_sum_pd(__m256d a) {
#ifdef USE_HADD
    // IVY 9 OPS - HW - 11 OPS
    __m256d sum = _mm256_hadd_pd(a, a);
#else
    // IVY 8 OPS - HW - 10 OPS
    __m256d sum = _mm256_add_pd(a, _mm256_shuffle_pd(a,a,0x5));
#endif
    __m128d sum_high = _mm256_extractf128_pd(sum, 0x1);
    __m128d result = _mm_add_sd(sum_high, _mm256_castpd256_pd128(sum));
    return _mm_cvtsd_f64(result);
}
//!---------------------------------------------------------------//
//!
//!---------------------------------------------------------------//
//! Reversing a register
FASTOR_INLINE __m128 _mm_reverse_ps(__m128 a) {
    // 1OP
    return _mm_shuffle_ps(a,a,_MM_SHUFFLE(0,1,2,3));
}
FASTOR_INLINE __m128d _mm_reverse_pd(__m128d a) {
    // 1OP
    return _mm_shuffle_pd(a,a,0x1);
}
FASTOR_INLINE __m256 _mm256_reverse_ps(__m256 a) {
    // IVY 2OPS / HW 4OPS
    __m256 r1 = _mm256_permute2f128_ps(a,a,0x1);
    return _mm256_shuffle_ps(r1,r1,27);
}
FASTOR_INLINE __m256d _mm256_reverse_pd(__m256d a) {
    // IVY 2OPS / HW 4OPS
    __m256d r1 = _mm256_permute2f128_pd(a,a,0x1);
    return _mm256_shuffle_pd(r1,r1,5);
}
//!---------------------------------------------------------------//
//!
//!---------------------------------------------------------------//
//! Bit shifting
FASTOR_INLINE __m128 _mm_shift1_ps(__m128 a) {
    // 1OP
    return _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(a), 4));
}
FASTOR_INLINE __m128 _mm_shift2_ps(__m128 a) {
    // 1OP
    return _mm_shuffle_ps(ZEROPS, a, 0x40);
}
FASTOR_INLINE __m128 _mm_shift3_ps(__m128 a) {
    // 2OPS
    __m128 shift2 = _mm_shuffle_ps(ZEROPS, a, 0x40);
    return _mm_shuffle_ps(ZEROPS,shift2,_MM_SHUFFLE(2,0,2,0));
}
FASTOR_INLINE __m256 _mm256_shift1_ps(__m256 a) {
    // IVY 3OPS / HW 5OPS
    __m256 r1 = _mm256_permute_ps(a,_MM_SHUFFLE(2,1,0,3));
    __m256 r2 = _mm256_permute2f128_ps(r1,r1,41);
    return _mm256_blend_ps(r1,r2,0x11);
}
FASTOR_INLINE __m256 _mm256_shift2_ps(__m256 a) {
    // IVY 3OPS / HW 5OPS
    __m256 r1 = _mm256_permute_ps(a,_MM_SHUFFLE(1,0,3,2));
    __m256 r2 = _mm256_permute2f128_ps(r1,r1,41);
    return _mm256_blend_ps(r1,r2,0x33);
}
FASTOR_INLINE __m256 _mm256_shift3_ps(__m256 a) {
    // IVY 1OPS / HW 3OPS
    return _mm256_permute2f128_ps(a,a,41);
}
FASTOR_INLINE __m256 _mm256_shift4_ps(__m256 a) {
    // IVY 1OPS / HW 3OPS
    return _mm256_permute2f128_ps(a,a,42);
}
FASTOR_INLINE __m256 _mm256_shift5_ps(__m256 a) {
    // IVY 2 OPS - HW 4 OPS
    __m128 r1 = _mm_shift1_ps(_mm256_castps256_ps128(a));
    return _mm256_insertf128_ps(VZEROPS,r1,0x1);
}
FASTOR_INLINE __m256 _mm256_shift6_ps(__m256 a) {
    // IVY 2 OPS - HW 4 OPS
    __m128 r1 = _mm_shift2_ps(_mm256_castps256_ps128(a));
    return _mm256_insertf128_ps(VZEROPS,r1,0x1);
}
FASTOR_INLINE __m256 _mm256_shift7_ps(__m256 a) {
    // IVY 2 OPS - HW 4 OPS
    __m128 r1 = _mm_shift3_ps(_mm256_castps256_ps128(a));
    return _mm256_insertf128_ps(VZEROPS,r1,0x1);
}

FASTOR_INLINE __m128d _mm_shift1_pd(__m128d a) {
    // 1OP
    return _mm_shuffle_pd(ZEROPD,a,0x1);
}
FASTOR_INLINE __m256d _mm256_shift1_pd(__m256d a) {
    // IVY - 4 OPS / HW - 8 OPS
    __m128d r1 = _mm256_castpd256_pd128(a);
    __m128d r2 = _mm256_extractf128_pd(a,0x1);
    __m128d r3 = _mm_shuffle_pd(r1,r2,0x1);
    __m256d r4 = _mm256_castpd128_pd256(_mm_shift1_pd(r1));
    return _mm256_insertf128_pd(r4,r3,0x1);
}
FASTOR_INLINE __m256d _mm256_shift2_pd(__m256d a) {
    // IVY - 1OP / HW - 3OPS
    return _mm256_permute2f128_pd(a,a,8);
}
FASTOR_INLINE __m256d _mm256_shift3_pd(__m256d a) {
    // IVY - 2OPS / HW - 4OPS
    __m256d r1 = _mm256_castpd128_pd256(_mm_shift1_pd(_mm256_castpd256_pd128(a)));
    return _mm256_permute2f128_pd(r1,r1,0x1);
}
//!---------------------------------------------------------------//
//!
//!---------------------------------------------------------------//
// Change sign of a register - all one cycle
FASTOR_INLINE __m128 _mm_neg_ps(__m128 a) {
    return _mm_xor_ps(a, MZEROPS);
}
FASTOR_INLINE __m128d _mm_neg_pd(__m128d a) {
    return _mm_xor_pd(a, MZEROPD);
}
FASTOR_INLINE __m256 _mm256_neg_ps(__m256 a) {
    return _mm256_xor_ps(a, MVZEROPS);
}
FASTOR_INLINE __m256d _mm256_neg_pd(__m256d a) {
    return _mm256_xor_pd(a, MVZEROPD);
}
//!---------------------------------------------------------------//
//!
//!---------------------------------------------------------------//
// maximum value in a register - horizontal max
FASTOR_INLINE float _mm_hmax_ps(__m128 a) {
    // 8OPS
    __m128 max0 = _mm_max_ps(a,_mm_reverse_ps(a));
    __m128 tmp = _mm_shuffle_ps(max0,max0,_MM_SHUFFLE(0,0,0,1));
    return _mm_cvtss_f32(_mm_max_ps(max0,tmp));
}
FASTOR_INLINE double _mm_hmax_pd(__m128d a) {
    // 4OPS
    return _mm_cvtsd_f64(_mm_max_pd(a,_mm_reverse_pd(a)));
}
FASTOR_INLINE float _mm256_hmax_ps(__m256 a) {
    // IVY 18OPS / HW 24 OPS
    __m128 lo = _mm256_castps256_ps128(a);
    __m128 max0 = _mm_max_ps(lo,_mm_reverse_ps(lo));
    __m128 tmp0 = _mm_shuffle_ps(max0,max0,_MM_SHUFFLE(0,0,0,1));
    __m128 max_lo = _mm_max_ps(max0,tmp0);

    __m128 hi = _mm256_extractf128_ps(a,0x1);
    __m128 max1 = _mm_max_ps(hi,_mm_reverse_ps(hi));
    __m128 tmp1 = _mm_shuffle_ps(max1,max1,_MM_SHUFFLE(0,0,0,1));
    __m128 max_hi = _mm_max_ps(max1,tmp1);

    return _mm_cvtss_f32(_mm_max_ps(max_lo,max_hi));
}
FASTOR_INLINE double _mm256_hmax_pd(__m256d a) {
    // IVY 9OPS / HW 11 OPS
    __m256d max0 = _mm256_max_pd(a,_mm256_reverse_pd(a));
    __m256d tmp = _mm256_shuffle_pd(max0,max0,_MM_SHUFFLE(0,0,0,1));
    return _mm_cvtsd_f64(_mm256_castpd256_pd128(_mm256_max_pd(max0,tmp)));
}
//!---------------------------------------------------------------//
//!
//!---------------------------------------------------------------//
// minimum value in a register - horizontal min
FASTOR_INLINE float _mm_hmin_ps(__m128 a) {
    // 8OPS
    __m128 max0 = _mm_min_ps(a,_mm_reverse_ps(a));
    __m128 tmp = _mm_shuffle_ps(max0,max0,_MM_SHUFFLE(0,0,0,1));
    return _mm_cvtss_f32(_mm_min_ps(max0,tmp));
}
FASTOR_INLINE double _mm_hmin_pd(__m128d a) {
    // 4OPS
    return _mm_cvtsd_f64(_mm_min_pd(a,_mm_reverse_pd(a)));
}
FASTOR_INLINE float _mm256_hmin_ps(__m256 a) {
    // IVY 18OPS / HW 24 OPS
    __m128 lo = _mm256_castps256_ps128(a);
    __m128 max0 = _mm_min_ps(lo,_mm_reverse_ps(lo));
    __m128 tmp0 = _mm_shuffle_ps(max0,max0,_MM_SHUFFLE(0,0,0,1));
    __m128 max_lo = _mm_min_ps(max0,tmp0);

    __m128 hi = _mm256_extractf128_ps(a,0x1);
    __m128 max1 = _mm_min_ps(hi,_mm_reverse_ps(hi));
    __m128 tmp1 = _mm_shuffle_ps(max1,max1,_MM_SHUFFLE(0,0,0,1));
    __m128 max_hi = _mm_min_ps(max1,tmp1);

    return _mm_cvtss_f32(_mm_min_ps(max_lo,max_hi));
}
FASTOR_INLINE double _mm256_hmin_pd(__m256d a) {
    // IVY 9OPS / HW 11 OPS
    __m256d max0 = _mm256_min_pd(a,_mm256_reverse_pd(a));
    __m256d tmp = _mm256_shuffle_pd(max0,max0,_MM_SHUFFLE(0,0,0,1));
    return _mm_cvtsd_f64(_mm256_castpd256_pd128(_mm256_min_pd(max0,tmp)));
}
//!---------------------------------------------------
// indexing a register
FASTOR_INLINE float _mm_get0_ps(__m128 a) {
    // NO OP
    return _mm_cvtss_f32(a);
}
FASTOR_INLINE float _mm_get1_ps(__m128 a) {
    // 1 OP
    return _mm_cvtss_f32(_mm_shuffle_ps(a,a,_MM_SHUFFLE(0,0,0,1)));
}
FASTOR_INLINE float _mm_get2_ps(__m128 a) {
    // 1 OP
    return _mm_cvtss_f32(_mm_shuffle_ps(a,a,_MM_SHUFFLE(0,0,0,2)));
}
FASTOR_INLINE float _mm_get3_ps(__m128 a) {
    // 1 OP
    return _mm_cvtss_f32(_mm_shuffle_ps(a,a,_MM_SHUFFLE(0,0,0,3)));
}
FASTOR_INLINE double _mm_get0_pd(__m128d a) {
    // NO OP
    return _mm_cvtsd_f64(a);
}
FASTOR_INLINE double _mm_get1_pd(__m128d a) {
    // 1 OP
    return _mm_cvtsd_f64(_mm_shuffle_pd(a,a,_MM_SHUFFLE2(0,1)));
}
FASTOR_INLINE float _mm256_get0_ps(__m256 a) {
    // NO OP
    return _mm_cvtss_f32(_mm256_castps256_ps128(a));
}
FASTOR_INLINE float _mm256_get1_ps(__m256 a) {
    // 1 OP
    __m128 lower = _mm256_castps256_ps128(a);
    return _mm_cvtss_f32(_mm_shuffle_ps(lower,lower,_MM_SHUFFLE(0,0,0,1)));
}
FASTOR_INLINE float _mm256_get2_ps(__m256 a) {
    // 1 OP
    __m128 lower = _mm256_castps256_ps128(a);
    return _mm_cvtss_f32(_mm_shuffle_ps(lower,lower,_MM_SHUFFLE(0,0,0,2)));
}
FASTOR_INLINE float _mm256_get3_ps(__m256 a) {
    // NO OP
    __m128 lower = _mm256_castps256_ps128(a);
    return _mm_cvtss_f32(_mm_shuffle_ps(lower,lower,_MM_SHUFFLE(0,0,0,3)));
}
FASTOR_INLINE float _mm256_get4_ps(__m256 a) {
    // IVY 1OP / HW 3OPS
    return _mm_cvtss_f32(_mm256_extractf128_ps(a,0x1));
}
FASTOR_INLINE float _mm256_get5_ps(__m256 a) {
    // IVY 2OPS/ HW 4OPS
    __m128 higher = _mm256_extractf128_ps(a,0x1);
    return _mm_cvtss_f32(_mm_shuffle_ps(higher,higher,_MM_SHUFFLE(0,0,0,1)));
}
FASTOR_INLINE float _mm256_get6_ps(__m256 a) {
    // IVY 2OPS/ HW 4OPS
    __m128 higher = _mm256_extractf128_ps(a,0x1);
    return _mm_cvtss_f32(_mm_shuffle_ps(higher,higher,_MM_SHUFFLE(0,0,0,2)));
}
FASTOR_INLINE float _mm256_get7_ps(__m256 a) {
    // IVY 2OPS/ HW 4OPS
    __m128 higher = _mm256_extractf128_ps(a,0x1);
    return _mm_cvtss_f32(_mm_shuffle_ps(higher,higher,_MM_SHUFFLE(0,0,0,3)));
}
FASTOR_INLINE double _mm256_get0_pd(__m256d a) {
    // NO OP
    return _mm_cvtsd_f64(_mm256_castpd256_pd128(a));
}
FASTOR_INLINE double _mm256_get1_pd(__m256d a) {
    // 1 OP
    __m128d lower = _mm256_castpd256_pd128(a);
    return _mm_cvtsd_f64(_mm_shuffle_pd(lower,lower,_MM_SHUFFLE2(0,1)));
}
FASTOR_INLINE double _mm256_get2_pd(__m256d a) {
    // IVY 1OP / HW 3OPS
    return _mm_cvtsd_f64(_mm256_extractf128_pd(a,0x1));
}
FASTOR_INLINE double _mm256_get3_pd(__m256d a) {
    // IVY 2OPS / HW 4OPS
    __m128d higher = _mm256_extractf128_pd(a,0x1);
    return _mm_cvtsd_f64(_mm_shuffle_pd(higher,higher,_MM_SHUFFLE2(0,1)));
}



//!----------------------------------------------------------------
// Equivalent ot _MM_TRANSPOSE_PS
#define _MM_TRANSPOSE4_PD(row0,row1,row2,row3)                                 \
{                                                                \
    __m256d tmp3, tmp2, tmp1, tmp0;                              \
                                                                 \
    tmp0 = _mm256_shuffle_pd((row0),(row1), 0x0);                    \
    tmp2 = _mm256_shuffle_pd((row0),(row1), 0xF);                \
    tmp1 = _mm256_shuffle_pd((row2),(row3), 0x0);                    \
    tmp3 = _mm256_shuffle_pd((row2),(row3), 0xF);                \
                                                                 \
    (row0) = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);   \
    (row1) = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);   \
    (row2) = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);   \
    (row3) = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);   \
}
// For 8x8 PS
FASTOR_INLINE void _MM_TRANSPOSE8_PS(__m256 &row0, __m256 &row1, __m256 &row2,
                              __m256 &row3, __m256 &row4, __m256 &row5,
                              __m256 &row6, __m256 &row7) {
    __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
    __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
    __t0 = _mm256_unpacklo_ps(row0, row1);
    __t1 = _mm256_unpackhi_ps(row0, row1);
    __t2 = _mm256_unpacklo_ps(row2, row3);
    __t3 = _mm256_unpackhi_ps(row2, row3);
    __t4 = _mm256_unpacklo_ps(row4, row5);
    __t5 = _mm256_unpackhi_ps(row4, row5);
    __t6 = _mm256_unpacklo_ps(row6, row7);
    __t7 = _mm256_unpackhi_ps(row6, row7);
    __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
    __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
    __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
    __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
    __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
    __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
    __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
    __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
    row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
    row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
    row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
    row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
    row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
    row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
    row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
    row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}






//!-----------------------------------------------------------------
//! Some further auxilary functions C++ only
static FASTOR_INLINE __m128d _add_pd(__m128d a) {
    // IVY 4 OPS
    __m128 shuftmp= _mm_movehl_ps(ZEROPS, _mm_castpd_ps(a));  // there is no movhlpd
    __m128d shuf  = _mm_castps_pd(shuftmp);
    return  _mm_add_sd(a, shuf);
}

#ifdef USE_HADD
static FASTOR_INLINE __m128d _add_pd(__m256d a) {
    // IVY 12 OPS / HW 14 OPS
    __m128d sum_low = _add_pd(_mm256_castpd256_pd128(a));
    __m128d sum_high = _add_pd(_mm256_extractf128_pd(sum_low, 0x1));
    return _mm_add_pd(sum_high, sum_low);
}
#else
static FASTOR_INLINE __m128d _add_pd(__m256d a) {
    // IVY 9 OPS / HW 11 OPS
    __m256d sum_low = _mm256_hadd_pd(a, a);
    __m128d sum_high = _mm256_extractf128_pd(sum_low, 0x1);
    return _mm_add_pd(sum_high, _mm256_castpd256_pd128(sum_low));
}
#endif
FASTOR_INLINE __m128 _add_ps(__m128 a) {
    // 8 OPS
    __m128 shuf = _mm_movehdup_ps(a);        // line up elements 3,1 with 2,0
    __m128 sums = _mm_add_ps(a, shuf);
    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums        = _mm_add_ss(sums, shuf);
    return      sums;
}
FASTOR_INLINE __m128 _add_ps(__m256 a) {
    // IVY 20 OPS / HW 22 OPS
    __m128 sum_low = _add_ps(_mm256_castps256_ps128(a));
    __m128 sum_high = _add_ps(_mm256_extractf128_ps(a,0x1));
    return _mm_add_ss(sum_low,sum_high);
}

// horizontal add_sub
FASTOR_INLINE __m128 _addsub_ps(__m128 a) {
    // 8 OPS
    // If a = [a0 a1 a2 a3] this function returns (a1+a3)-(a0+a2)
    // Note that only the first element of __m128 corresponds to this
    __m128 shuf = _mm_shuffle_ps(a,a,_MM_SHUFFLE(1,0,3,2));
    __m128 sums = _mm_add_ps(a, shuf);
    shuf        = _mm_shuffle_ps(sums,sums,_MM_SHUFFLE(2,3,0,1));
    return _mm_sub_ps(shuf, sums);
}


FASTOR_INLINE __m128 _mulsub_ps(__m128 a) {
    // 10 OPS
    // If a = [a0 a1 a2 a3] this function returns (a1*a3)-(a0+a2)
    // Note that only the first element of __m128 corresponds to this
    __m128 shuf = _mm_shuffle_ps(a,a,_MM_SHUFFLE(1,0,3,2));
    __m128 muls = _mm_mul_ps(a, shuf);
    shuf        = _mm_shuffle_ps(muls,muls,_MM_SHUFFLE(2,3,0,1));
    return _mm_sub_ps(shuf, muls);
}

FASTOR_INLINE __m128d _hsub_pd(__m128d a) {
    // horizontal sub, returns a[0] - a[1]
    // 4 OPS  W/O HSUB - 5 OPS WITH HSUB
#ifdef USE_HADD
    return _mm_hsub_pd(a,a);
#else
    return _mm_sub_sd(a,_mm_shuffle_pd(a,a,0x1));
#endif
}

//!-----------------------------------------------------------------


// Similar to SSE4 _mm_dp_pd for dot product
__m128d _mm256_dp_pd(__m256d __X, __m256d __Y) {
    return _add_pd(_mm256_mul_pd(__X, __Y));
}


#endif // EXT_INTRIN_H

