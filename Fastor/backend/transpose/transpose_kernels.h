#ifndef TRANSPOSE_KERNELS_H
#define TRANSPOSE_KERNELS_H

#include "Fastor/config/config.h"
#include "Fastor/simd_vector/extintrin.h"

namespace Fastor {

namespace internal {

#ifdef FASTOR_SSE_IMPL
// 4x4 PS - defined
// _MM_TRANSPOSE4_PS
#endif

#ifdef FASTOR_AVX_IMPL

// 4x4 PD
FASTOR_INLINE void _MM_TRANSPOSE4_PD(__m256d &row0, __m256d &row1, __m256d &row2, __m256d &row3)
{
    __m256d tmp3, tmp2, tmp1, tmp0;
    tmp0 = _mm256_shuffle_pd((row0),(row1), 0x0);
    tmp2 = _mm256_shuffle_pd((row0),(row1), 0xF);
    tmp1 = _mm256_shuffle_pd((row2),(row3), 0x0);
    tmp3 = _mm256_shuffle_pd((row2),(row3), 0xF);
    row0 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);
    row1 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);
    row2 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);
    row3 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);
}

// 8x8 PS
FASTOR_INLINE void _MM_TRANSPOSE8_PS(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3,
                                     __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7)
{
    __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
    __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
    __t0  = _mm256_unpacklo_ps(row0, row1);
    __t1  = _mm256_unpackhi_ps(row0, row1);
    __t2  = _mm256_unpacklo_ps(row2, row3);
    __t3  = _mm256_unpackhi_ps(row2, row3);
    __t4  = _mm256_unpacklo_ps(row4, row5);
    __t5  = _mm256_unpackhi_ps(row4, row5);
    __t6  = _mm256_unpacklo_ps(row6, row7);
    __t7  = _mm256_unpackhi_ps(row6, row7);
    __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
    __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
    __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
    __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
    __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
    __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
    __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
    __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
    row0  = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
    row1  = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
    row2  = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
    row3  = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
    row4  = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
    row5  = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
    row6  = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
    row7  = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}
#endif

#ifdef FASTOR_AVX512F_IMPL
// 8x8 PD
inline void _MM_TRANSPOSE8_PD(__m512d &row0, __m512d &row1, __m512d &row2, __m512d &row3,
                              __m512d &row4, __m512d &row5, __m512d &row6, __m512d &row7)
{
    __m512d __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
    __m512d __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;

    FASTOR_ARCH_ALIGN constexpr int64_t idx1[8] = {0, 8 , 1 , 9 , 4 , 12, 5 , 13};
    FASTOR_ARCH_ALIGN constexpr int64_t idx2[8] = {2, 10, 3 , 11, 6 , 14, 7 , 15};
    FASTOR_ARCH_ALIGN constexpr int64_t idx3[8] = {0, 1 , 8 , 9 , 4 , 5 , 12, 13};
    FASTOR_ARCH_ALIGN constexpr int64_t idx4[8] = {2, 3 , 10, 11, 6 , 7 , 14, 15};
    FASTOR_ARCH_ALIGN constexpr int64_t idx5[8] = {4, 5 , 6 , 7 , 12, 13, 14, 15};

    __m512i vidx1 = _mm512_load_epi64(idx1);
    __m512i vidx2 = _mm512_load_epi64(idx2);
    __m512i vidx3 = _mm512_load_epi64(idx3);
    __m512i vidx4 = _mm512_load_epi64(idx4);
    __m512i vidx5 = _mm512_load_epi64(idx5);

    __t0 = _mm512_permutex2var_pd(row0, vidx1, row1);
    __t1 = _mm512_permutex2var_pd(row0, vidx2, row1);
    __t2 = _mm512_permutex2var_pd(row2, vidx1, row3);
    __t3 = _mm512_permutex2var_pd(row2, vidx2, row3);
    __t4 = _mm512_permutex2var_pd(row4, vidx1, row5);
    __t5 = _mm512_permutex2var_pd(row4, vidx2, row5);
    __t6 = _mm512_permutex2var_pd(row6, vidx1, row7);
    __t7 = _mm512_permutex2var_pd(row6, vidx2, row7);

    __tt0 = _mm512_permutex2var_pd(__t0, vidx3, __t2);
    __tt1 = _mm512_permutex2var_pd(__t0, vidx4, __t2);
    __tt2 = _mm512_permutex2var_pd(__t1, vidx3, __t3);
    __tt3 = _mm512_permutex2var_pd(__t1, vidx4, __t3);
    __tt4 = _mm512_permutex2var_pd(__t4, vidx3, __t6);
    __tt5 = _mm512_permutex2var_pd(__t4, vidx4, __t6);
    __tt6 = _mm512_permutex2var_pd(__t5, vidx3, __t7);
    __tt7 = _mm512_permutex2var_pd(__t5, vidx4, __t7);

    row0 = _mm512_insertf64x4(__tt0,_mm512_castpd512_pd256(__tt4),0x1);
    row1 = _mm512_insertf64x4(__tt1,_mm512_castpd512_pd256(__tt5),0x1);
    row2 = _mm512_insertf64x4(__tt2,_mm512_castpd512_pd256(__tt6),0x1);
    row3 = _mm512_insertf64x4(__tt3,_mm512_castpd512_pd256(__tt7),0x1);
    row4 = _mm512_permutex2var_pd(__tt0, vidx5, __tt4);
    row5 = _mm512_permutex2var_pd(__tt1, vidx5, __tt5);
    row6 = _mm512_permutex2var_pd(__tt2, vidx5, __tt6);
    row7 = _mm512_permutex2var_pd(__tt3, vidx5, __tt7);
}
#endif


#if defined(FASTOR_AVX512F_IMPL) && defined(FASTOR_AVX512DQ_IMPL)
// 16x16
FASTOR_INLINE void _MM_TRANSPOSE16_PS(const float * FASTOR_RESTRICT mat, float * FASTOR_RESTRICT matT)
{
      __m512 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;
      __m512 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;

    int mask;
    FASTOR_ARCH_ALIGN constexpr int64_t idx1[8]  = {2, 3, 0, 1, 6, 7, 4, 5};
    FASTOR_ARCH_ALIGN constexpr int64_t idx2[8]  = {1, 0, 3, 2, 5, 4, 7, 6};
    FASTOR_ARCH_ALIGN constexpr int32_t idx3[16] = {1, 0, 3, 2, 5 ,4 ,7 ,6 ,9 ,8 , 11, 10, 13, 12 ,15, 14};
    __m512i vidx1 = _mm512_load_epi64(idx1);
    __m512i vidx2 = _mm512_load_epi64(idx2);
    __m512i vidx3 = _mm512_load_epi32(idx3);

    t0 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(&mat[ 0*16+0])), _mm256_loadu_ps(&mat[ 8*16+0]), 1);
    t1 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(&mat[ 1*16+0])), _mm256_loadu_ps(&mat[ 9*16+0]), 1);
    t2 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(&mat[ 2*16+0])), _mm256_loadu_ps(&mat[10*16+0]), 1);
    t3 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(&mat[ 3*16+0])), _mm256_loadu_ps(&mat[11*16+0]), 1);
    t4 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(&mat[ 4*16+0])), _mm256_loadu_ps(&mat[12*16+0]), 1);
    t5 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(&mat[ 5*16+0])), _mm256_loadu_ps(&mat[13*16+0]), 1);
    t6 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(&mat[ 6*16+0])), _mm256_loadu_ps(&mat[14*16+0]), 1);
    t7 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(&mat[ 7*16+0])), _mm256_loadu_ps(&mat[15*16+0]), 1);

    t8 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(&mat[ 0*16+8])), _mm256_loadu_ps(&mat[ 8*16+8]), 1);
    t9 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(&mat[ 1*16+8])), _mm256_loadu_ps(&mat[ 9*16+8]), 1);
    ta = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(&mat[ 2*16+8])), _mm256_loadu_ps(&mat[10*16+8]), 1);
    tb = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(&mat[ 3*16+8])), _mm256_loadu_ps(&mat[11*16+8]), 1);
    tc = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(&mat[ 4*16+8])), _mm256_loadu_ps(&mat[12*16+8]), 1);
    td = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(&mat[ 5*16+8])), _mm256_loadu_ps(&mat[13*16+8]), 1);
    te = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(&mat[ 6*16+8])), _mm256_loadu_ps(&mat[14*16+8]), 1);
    tf = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_loadu_ps(&mat[ 7*16+8])), _mm256_loadu_ps(&mat[15*16+8]), 1);

    mask= 0xcc;
    r0 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(t0), (__mmask8)mask, vidx1, _mm512_castps_pd(t4)));
    r1 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(t1), (__mmask8)mask, vidx1, _mm512_castps_pd(t5)));
    r2 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(t2), (__mmask8)mask, vidx1, _mm512_castps_pd(t6)));
    r3 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(t3), (__mmask8)mask, vidx1, _mm512_castps_pd(t7)));
    r8 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(t8), (__mmask8)mask, vidx1, _mm512_castps_pd(tc)));
    r9 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(t9), (__mmask8)mask, vidx1, _mm512_castps_pd(td)));
    ra = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(ta), (__mmask8)mask, vidx1, _mm512_castps_pd(te)));
    rb = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(tb), (__mmask8)mask, vidx1, _mm512_castps_pd(tf)));

    mask= 0x33;
    r4 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(t4), (__mmask8)mask, vidx1, _mm512_castps_pd(t0)));
    r5 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(t5), (__mmask8)mask, vidx1, _mm512_castps_pd(t1)));
    r6 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(t6), (__mmask8)mask, vidx1, _mm512_castps_pd(t2)));
    r7 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(t7), (__mmask8)mask, vidx1, _mm512_castps_pd(t3)));
    rc = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(tc), (__mmask8)mask, vidx1, _mm512_castps_pd(t8)));
    rd = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(td), (__mmask8)mask, vidx1, _mm512_castps_pd(t9)));
    re = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(te), (__mmask8)mask, vidx1, _mm512_castps_pd(ta)));
    rf = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(tf), (__mmask8)mask, vidx1, _mm512_castps_pd(tb)));

    mask = 0xaa;
    t0 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(r0), (__mmask8)mask, vidx2, _mm512_castps_pd(r2)));
    t1 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(r1), (__mmask8)mask, vidx2, _mm512_castps_pd(r3)));
    t4 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(r4), (__mmask8)mask, vidx2, _mm512_castps_pd(r6)));
    t5 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(r5), (__mmask8)mask, vidx2, _mm512_castps_pd(r7)));
    t8 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(r8), (__mmask8)mask, vidx2, _mm512_castps_pd(ra)));
    t9 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(r9), (__mmask8)mask, vidx2, _mm512_castps_pd(rb)));
    tc = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(rc), (__mmask8)mask, vidx2, _mm512_castps_pd(re)));
    td = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(rd), (__mmask8)mask, vidx2, _mm512_castps_pd(rf)));

    mask = 0x55;
    t2 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(r2), (__mmask8)mask, vidx2, _mm512_castps_pd(r0)));
    t3 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(r3), (__mmask8)mask, vidx2, _mm512_castps_pd(r1)));
    t6 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(r6), (__mmask8)mask, vidx2, _mm512_castps_pd(r4)));
    t7 = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(r7), (__mmask8)mask, vidx2, _mm512_castps_pd(r5)));
    ta = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(ra), (__mmask8)mask, vidx2, _mm512_castps_pd(r8)));
    tb = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(rb), (__mmask8)mask, vidx2, _mm512_castps_pd(r9)));
    te = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(re), (__mmask8)mask, vidx2, _mm512_castps_pd(rc)));
    tf = _mm512_castpd_ps(_mm512_mask_permutexvar_pd(_mm512_castps_pd(rf), (__mmask8)mask, vidx2, _mm512_castps_pd(rd)));

    mask = 0xaaaa;
    r0 = _mm512_mask_permutexvar_ps(t0, (__mmask16)mask, vidx3, t1);
    r2 = _mm512_mask_permutexvar_ps(t2, (__mmask16)mask, vidx3, t3);
    r4 = _mm512_mask_permutexvar_ps(t4, (__mmask16)mask, vidx3, t5);
    r6 = _mm512_mask_permutexvar_ps(t6, (__mmask16)mask, vidx3, t7);
    r8 = _mm512_mask_permutexvar_ps(t8, (__mmask16)mask, vidx3, t9);
    ra = _mm512_mask_permutexvar_ps(ta, (__mmask16)mask, vidx3, tb);
    rc = _mm512_mask_permutexvar_ps(tc, (__mmask16)mask, vidx3, td);
    re = _mm512_mask_permutexvar_ps(te, (__mmask16)mask, vidx3, tf);

    mask = 0x5555;
    r1 = _mm512_mask_permutexvar_ps(t1, (__mmask16)mask, vidx3, t0);
    r3 = _mm512_mask_permutexvar_ps(t3, (__mmask16)mask, vidx3, t2);
    r5 = _mm512_mask_permutexvar_ps(t5, (__mmask16)mask, vidx3, t4);
    r7 = _mm512_mask_permutexvar_ps(t7, (__mmask16)mask, vidx3, t6);
    r9 = _mm512_mask_permutexvar_ps(t9, (__mmask16)mask, vidx3, t8);
    rb = _mm512_mask_permutexvar_ps(tb, (__mmask16)mask, vidx3, ta);
    rd = _mm512_mask_permutexvar_ps(td, (__mmask16)mask, vidx3, tc);
    rf = _mm512_mask_permutexvar_ps(tf, (__mmask16)mask, vidx3, te);

    _mm512_storeu_ps(&matT[ 0*16], r0);
    _mm512_storeu_ps(&matT[ 1*16], r1);
    _mm512_storeu_ps(&matT[ 2*16], r2);
    _mm512_storeu_ps(&matT[ 3*16], r3);
    _mm512_storeu_ps(&matT[ 4*16], r4);
    _mm512_storeu_ps(&matT[ 5*16], r5);
    _mm512_storeu_ps(&matT[ 6*16], r6);
    _mm512_storeu_ps(&matT[ 7*16], r7);
    _mm512_storeu_ps(&matT[ 8*16], r8);
    _mm512_storeu_ps(&matT[ 9*16], r9);
    _mm512_storeu_ps(&matT[10*16], ra);
    _mm512_storeu_ps(&matT[11*16], rb);
    _mm512_storeu_ps(&matT[12*16], rc);
    _mm512_storeu_ps(&matT[13*16], rd);
    _mm512_storeu_ps(&matT[14*16], re);
    _mm512_storeu_ps(&matT[15*16], rf);
}

#endif

} // end of namespace internal
} // end of namespace Fastor



#endif // TRANSPOSE_KERNELS_H
