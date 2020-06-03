#ifndef CYCLIC_0_H
#define CYCLIC_0_H

#include "Fastor/config/config.h"
#include "Fastor/simd_vector/extintrin.h"

namespace Fastor {


//! Version 0 of cyclic product of two second order tensors i.e. C_ijkl = A_ik * B_jl

template<typename T, size_t M0, size_t N0, size_t M1, size_t N1>
FASTOR_HINT_INLINE void _cyclic(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {
    for (size_t i=0; i<M0; ++i) {
        for (size_t j=0; j<N0; ++j) {
            for (size_t k=0; k<M1; ++k) {
                for (size_t l=0; l<N1; ++l) {
                    out[i*N1*M1*N0+j*M1*N0+k*N0+l] += a[i*N0+k]*b[j*N1+l];
                }
            }
        }
    }
}

#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_HINT_INLINE void _cyclic<double,2,2,2,2>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {
    __m256d as = _mm256_load_pd(a);
    __m256d bs = _mm256_load_pd(b);

    // First 2x2 block
    __m256d c0 = _mm256_mul_pd(as,bs);
    _mm_store_sd(out,_mm256_castpd256_pd128(c0));
    _mm_store_sd(out+1,_mm_shuffle_pd(_mm256_castpd256_pd128(c0),_mm256_castpd256_pd128(c0),0x1));
    __m128d c0_high = _mm256_extractf128_pd(c0,0x1);
    _mm_store_sd(out+3,_mm_shuffle_pd(_mm256_castpd256_pd128(c0),_mm256_castpd256_pd128(c0),0x1));
    _mm_store_sd(out+4,_mm_shuffle_pd(c0_high,c0_high,0x1));
    // c_02
    __m128d a0 = _mm256_castpd256_pd128(as);
    __m128d b0 = _mm256_castpd256_pd128(bs);
    b0 = _mm_shuffle_pd(b0,b0,0x1);
    __m128d c1 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a0,b0)));
    _mm_store_sd(out+2,c1);
    _mm_store_sd(out+6,c1);
    // c_12
    __m128d a1 = _mm256_extractf128_pd(as,0x1);
    __m128d b1 = _mm256_extractf128_pd(bs,0x1);
    b1 = _mm_shuffle_pd(b1,b1,0x1);
    __m128d c2 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a1,b1)));
    _mm_store_sd(out+5,c2);
    _mm_store_sd(out+7,c2);
    // c_22
    __m128d c3 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a0,b1)));
    _mm_store_sd(out+8,c3);
}

template<>
FASTOR_HINT_INLINE void _cyclic<double,3,3,3,3>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {
    //  34+ OPS
    __m256d a_low = _mm256_load_pd(a);
    __m256d a_high = _mm256_load_pd(a+4);
    __m128d a_end = _mm_load_sd(a+8);

    __m256d b_low = _mm256_load_pd(b);
    __m256d b_high = _mm256_load_pd(b+4);
    __m128d b_end = _mm_load_sd(b+8);

    // The first 3x3 block
    __m256d c0 = _mm256_mul_pd(a_low,b_low);
    __m256d c1 = _mm256_mul_pd(a_high,b_high);
    __m128d c2 = _mm_mul_sd(a_end,b_end);

    _mm256_store_pd(out,c0);
    _mm_store_sd(out+6,_mm_set_sd(_mm256_get1_pd(c0)));
    _mm256_storeu_pd(out+7,c1);
//    _mm_store_sd(out+12,_mm_set_sd(_mm256_get2_pd(c0)));
    _mm_store_sd(out+12,_mm256_extractf128_pd(c0,0x1));
    _mm_store_sd(out+13,_mm_set_sd(_mm256_get1_pd(c1)));
    _mm_store_sd(out+14,c2);

    // rest
    // c_03
    __m128d a0 = _mm256_castpd256_pd128(a_low);
    __m128d b0 = _mm256_castpd256_pd128(b_low);
    b0 = _mm_shuffle_pd(b0,b0,0x1);
    __m128d c3 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a0,b0)));
    _mm_store_sd(out+3,c3);
    _mm_store_sd(out+18,c3);
    // c_04
    __m128d a1 = _mm_set_pd(_mm256_get0_pd(a_low),_mm256_get2_pd(a_low));
    __m128d b1 = _mm_set_pd(_mm256_get2_pd(b_low),_mm256_get0_pd(b_low));
    __m128d c4 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a1,b1)));
    _mm_store_sd(out+4,c4);
    _mm_store_sd(out+24,c4);
    // c_05
    __m128d a2 = _mm_set_pd(_mm256_get1_pd(a_low),_mm256_get2_pd(a_low));
    __m128d b2 = _mm_set_pd(_mm256_get2_pd(b_low),_mm256_get1_pd(b_low));
    __m128d c5 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a2,b2)));
    _mm_store_sd(out+5,c5);
    _mm_store_sd(out+30,c5);
    // c_13
    __m128d a3 = _mm_setr_pd(_mm256_get3_pd(a_low),_mm256_get0_pd(a_high));
    __m128d b3 = _mm_setr_pd(_mm256_get0_pd(b_high),_mm256_get3_pd(b_low));
    __m128d c6 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a3,b3)));
    _mm_store_sd(out+9,c6);
    _mm_store_sd(out+19,c6);
    // c_14
    __m128d a4 = _mm_set_pd(_mm256_get3_pd(a_low),_mm256_get1_pd(a_high));
    __m128d b4 = _mm_set_pd(_mm256_get1_pd(b_high),_mm256_get3_pd(b_low));
    __m128d c7 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a4,b4)));
    _mm_store_sd(out+10,c7);
    _mm_store_sd(out+25,c7);
    // c_15
    __m128d a5 = _mm256_castpd256_pd128(a_high);
    __m128d b5 = _mm256_castpd256_pd128(b_high);
    b5 = _mm_shuffle_pd(b5,b5,0x1);
    __m128d c8 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a5,b5)));
    _mm_store_sd(out+11,c8);
    _mm_store_sd(out+31,c8);
    // c_23
    __m128d a6 = _mm256_extractf128_pd(a_high,0x1);
    __m128d b6 = _mm256_extractf128_pd(b_high,0x1);
    b6 = _mm_shuffle_pd(b6,b6,0x1);
    __m128d c9 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a6,b6)));
    _mm_store_sd(out+15,c9);
    _mm_store_sd(out+20,c9);
    // c_24
    __m128d a7 = _mm_set_pd(_mm256_get2_pd(a_high),_mm_get0_pd(a_end));
    __m128d b7 = _mm_set_pd(_mm_get0_pd(b_end),_mm256_get2_pd(b_high));
    __m128d c10 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a7,b7)));
    _mm_store_sd(out+16,c10);
    _mm_store_sd(out+26,c10);
    // c_25
    __m128d a8 = _mm_set_pd(_mm256_get3_pd(a_high),_mm_get0_pd(a_end));
    __m128d b8 = _mm_set_pd(_mm_get0_pd(b_end),_mm256_get3_pd(b_high));
    __m128d c11 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a8,b8)));
    _mm_store_sd(out+17,c11);
    _mm_store_sd(out+32,c11);
    // c_33
    __m128d c12 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a0,b3)));
    _mm_store_sd(out+21,c12);
    // c_34
    __m128d c13 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a1,b4)));
    _mm_store_sd(out+22,c13);
    // c_43
    c13 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a0,b6)));
    _mm_store_sd(out+27,c13);
    // c_35
    __m128d c14 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a2,_mm_shuffle_pd(b5,b5,0x1))));
    _mm_store_sd(out+23,c14);
    // c_53
    c14 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a3,b6)));
    _mm_store_sd(out+33,c14);
    // c_44
    __m128d c15 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a1,b7)));
    _mm_store_sd(out+28,c15);
    // c_45
    __m128d c16 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a2,b8)));
    _mm_store_sd(out+29,c16);
    c16 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a4,b7)));
    _mm_store_sd(out+34,c16);
    // c_55
    __m128d c17 = _mm_mul_pd(HALFPD,_add_pd(_mm_mul_pd(a5,_mm_shuffle_pd(b8,b8,0x1))));
    _mm_store_sd(out+35,c17);
}

#endif

}

#endif // CYCLIC_0_H

