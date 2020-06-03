#ifndef OUTER_H
#define OUTER_H

#include "Fastor/config/config.h"

namespace Fastor {

template<typename T, size_t M0, size_t N0, size_t M1, size_t N1>
FASTOR_HINT_INLINE void _outer(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {
    for (size_t i=0; i<M0; ++i) {
        for (size_t j=0; j<N0; ++j) {
            for (size_t k=0; k<M1; ++k) {
                for (size_t l=0; l<N1; ++l) {
                    out[i*N1*M1*N0+j*M1*N0+k*N0+l] += a[i*N0+j]*b[k*N1+l];
                }
            }
        }
    }
}

#ifdef FASTOR_SSE4_2_IMPL

// The followings are Voigt overloads

template<>
FASTOR_HINT_INLINE void _outer<float,2,2,2,2>(const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {
    // 31 OPS
    // Fetch a to L1-cache
//    _mm_prefetch(a,_MM_HINT_T0);
    __m128 a0 = _mm_set1_ps(a[0]);
    __m128 a1 = _mm_set1_ps(a[1]);
//    __m128 a2 = _mm_set1_ps(a[2]);
    __m128 a3 = _mm_set1_ps(a[3]);
    __m128 bs = _mm_load_ps(b);

    __m128 r0 = _mm_shuffle_ps(bs,bs,_MM_SHUFFLE(1,2,3,0));
    __m128 r1 = _mm_shuffle_ps(bs,bs,_MM_SHUFFLE(3,1,2,0));
    __m128 r2 = _mm_mul_ps(HALFPS,_mm_add_ps(r1,bs));
    r2 = _mm_shuffle_ps(r2,r2,_MM_SHUFFLE(3,0,2,1));
//    __m128 r3 = _mm_shift1_ps(_mm_shuffle_ps(r0,r2,_MM_SHUFFLE(0,0,1,0)));
    __m128 r3 = _mm_shuffle_ps(r0,r2,_MM_SHUFFLE(0,0,1,0));

    // row0
    __m128 row0 = _mm_mul_ps(a0,r3);
    _mm_store_ps(out, row0);
    // row1
    __m128 row1 = _mm_mul_ps(a3,r3);
    row1 = _mm_shuffle_ps(row1,row1,_MM_SHUFFLE(0,3,2,1));
    _mm_store_ss(out+3,_mm_shuffle_ps(row0,row0,_MM_SHUFFLE(0,0,0,1)));
//    _mm_storeu_ps(out+4,row1);
    _mm_store_ps(out+4,row1);
    // row2
    __m128 row2 = _mm_mul_ps(a1,r3);
    row2 = _mm_shuffle_ps(row2,row2,_MM_SHUFFLE(1,0,3,2));
    _mm_store_ss(out+6,_mm_shuffle_ps(row0,row0,_MM_SHUFFLE(0,0,0,2)));
    _mm_store_ss(out+7,_mm_shuffle_ps(row1,row1,_MM_SHUFFLE(0,0,0,1)));
    _mm_store_ss(out+8, row2);
}

template<>
FASTOR_HINT_INLINE void _outer<float,3,3,3,3>(const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {
    // 81 OPS
    // Fetch a to L1-cache
//    _mm_prefetch(a,_MM_HINT_T0);

    __m128 a0 = _mm_set1_ps(a[0]);
    __m128 a1 = _mm_set1_ps(a[1]);
    __m128 a2 = _mm_set1_ps(a[2]);
//    __m128 a3 = _mm_set1_ps(a[3]);
    __m128 a4 = _mm_set1_ps(a[4]);
    __m128 a5 = _mm_set1_ps(a[5]);
//    __m128 a6 = _mm_set1_ps(a[6]);
//    __m128 a7 = _mm_set1_ps(a[7]);
    __m128 a8 = _mm_set1_ps(a[8]);

    __m128 b_low = _mm_load_ps(b);
    __m128 b_high =_mm_load_ps(b+4);
    __m128 b_end = _mm_load_ss(b+8);

    __m128 b_diag = _mm_shuffle_ps(b_low,b_high,_MM_SHUFFLE(1,0,1,0));
    b_diag = _mm_shuffle_ps(b_diag,b_end,_MM_SHUFFLE(1,0,2,0));
    __m128 ofdiag_str = _mm_shift1_ps(_mm_shuffle_ps(b_low,b_high,_MM_SHUFFLE(3,1,2,1)));
    __m128 ofdiag_rev = _mm_shuffle_ps(b_low,b_high,_MM_SHUFFLE(3,2,2,3));
    ofdiag_rev = _mm_shift1_ps(_mm_shuffle_ps(ofdiag_rev,ofdiag_rev,_MM_SHUFFLE(3,3,2,0)));

    // Compute this only once
    __m128 half_add_diag = _mm_mul_ps(HALFPS,_mm_add_ps(ofdiag_str,ofdiag_rev));

    // row0
    __m128 c0_d = _mm_mul_ps(b_diag,a0);
    __m128 c0_off = _mm_mul_ps(a0,half_add_diag);
    c0_off = _mm_shuffle_ps(c0_off,c0_off,_MM_SHUFFLE(0,3,2,1));
    _mm_store_ps(out,c0_d);
    _mm_storeu_ps(out+3,c0_off);
    // row1
    __m128 c1_d = _mm_mul_ps(b_diag,a4);
    __m128 c1_off = _mm_mul_ps(a4,half_add_diag);
    c1_off = _mm_shuffle_ps(c1_off,c1_off,_MM_SHUFFLE(0,3,2,1));
    c1_d = _mm_shuffle_ps(c1_d,c1_d,_MM_SHUFFLE(0,3,2,1));
    _mm_store_ss(out+6,_mm_shuffle_ps(c0_d,c0_d,_MM_SHUFFLE(0,0,0,1)));
    _mm_storeu_ps(out+7,c1_d);
    _mm_storeu_ps(out+9,c1_off);
    // row2
    __m128 c2_d = _mm_mul_ps(b_diag,a8);
    __m128 c2_off = _mm_mul_ps(a8,half_add_diag);
    c2_off = _mm_shuffle_ps(c2_off,c2_off,_MM_SHUFFLE(0,3,2,1));
    _mm_store_ss(out+12,_mm_shuffle_ps(c0_d,c0_d,_MM_SHUFFLE(0,0,0,2)));
    _mm_store_ss(out+13,_mm_shuffle_ps(c1_d,c1_d,_MM_SHUFFLE(0,0,0,1)));
    _mm_store_ss(out+14,_mm_shuffle_ps(c2_d,c2_d,_MM_SHUFFLE(0,0,0,2)));
    _mm_storeu_ps(out+15,c2_off);
    // row3
    __m128 c3_off = _mm_mul_ps(a1,half_add_diag);
    c3_off = _mm_shuffle_ps(c3_off,c3_off,_MM_SHUFFLE(0,3,2,1));
    _mm_store_ss(out+18,c0_off);
    _mm_store_ss(out+19,c1_off);
    _mm_store_ss(out+20,c2_off);
    _mm_storeu_ps(out+21,c3_off);
    // row4
    __m128 c4_off = _mm_mul_ps(a2,half_add_diag);
    c4_off = _mm_shuffle_ps(c4_off,c4_off,_MM_SHUFFLE(0,3,2,1));
    _mm_store_ss(out+24,_mm_shuffle_ps(c0_off,c0_off,_MM_SHUFFLE(0,0,0,1)));
    _mm_store_ss(out+25,_mm_shuffle_ps(c1_off,c1_off,_MM_SHUFFLE(0,0,0,1)));
    _mm_store_ss(out+26,_mm_shuffle_ps(c2_off,c2_off,_MM_SHUFFLE(0,0,0,1)));
    _mm_storeu_ps(out+27,c4_off);
    // row5
    __m128 c5_off = _mm_mul_ps(a5,half_add_diag);
    c5_off = _mm_shuffle_ps(c5_off,c5_off,_MM_SHUFFLE(0,3,2,1));
    _mm_store_ss(out+30,_mm_shuffle_ps(c0_off,c0_off,_MM_SHUFFLE(0,0,0,2)));
    _mm_store_ss(out+31,_mm_shuffle_ps(c1_off,c1_off,_MM_SHUFFLE(0,0,0,2)));
    _mm_store_ss(out+32,_mm_shuffle_ps(c2_off,c2_off,_MM_SHUFFLE(0,0,0,2)));
    _mm_storeu_ps(out+33,c5_off);

    // row0
//    __m128 c0_d = _mm_mul_ps(b_diag,a0);
//    __m128 c_os = _mm_mul_ps(ofdiag_str,a0);
//    __m128 c_or = _mm_mul_ps(ofdiag_rev,a0);
//    __m128 c0_off = _mm_mul_ps(HALFPS,_mm_add_ps(c_os,c_or));
//    c0_off = _mm_shuffle_ps(c0_off,c0_off,_MM_SHUFFLE(0,3,2,1));
//    _mm_store_ps(out,c0_d);
//    _mm_storeu_ps(out+3,c0_off);

//    // row1
//    __m128 c1_d = _mm_mul_ps(b_diag,a4);
//    c_os = _mm_mul_ps(ofdiag_str,a4);
//    c_or = _mm_mul_ps(ofdiag_rev,a4);
//    __m128 c1_off = _mm_mul_ps(HALFPS,_mm_add_ps(c_os,c_or));
//    c1_off = _mm_shuffle_ps(c1_off,c1_off,_MM_SHUFFLE(0,3,2,1));
//    c1_d = _mm_shuffle_ps(c1_d,c1_d,_MM_SHUFFLE(0,3,2,1));
//    _mm_store_ss(out+6,_mm_shuffle_ps(c0_d,c0_d,_MM_SHUFFLE(0,0,0,1)));
//    _mm_storeu_ps(out+7,c1_d);
//    _mm_storeu_ps(out+9,c1_off);

//    // row2
//    __m128 c2_d = _mm_mul_ps(b_diag,a8);
//    c_os = _mm_mul_ps(ofdiag_str,a8);
//    c_or = _mm_mul_ps(ofdiag_rev,a8);
//    __m128 c2_off = _mm_mul_ps(HALFPS,_mm_add_ps(c_os,c_or));
//    c2_off = _mm_shuffle_ps(c2_off,c2_off,_MM_SHUFFLE(0,3,2,1));
//    _mm_store_ss(out+12,_mm_shuffle_ps(c0_d,c0_d,_MM_SHUFFLE(0,0,0,2)));
//    _mm_store_ss(out+13,_mm_shuffle_ps(c1_d,c1_d,_MM_SHUFFLE(0,0,0,1)));
//    _mm_store_ss(out+14,_mm_shuffle_ps(c2_d,c2_d,_MM_SHUFFLE(0,0,0,2)));
//    _mm_storeu_ps(out+15,c2_off);

//    // row3
//    c_os = _mm_mul_ps(ofdiag_str,a1);
//    c_or = _mm_mul_ps(ofdiag_rev,a1);
//    __m128 c3_off = _mm_mul_ps(HALFPS,_mm_add_ps(c_os,c_or));
//    c3_off = _mm_shuffle_ps(c3_off,c3_off,_MM_SHUFFLE(0,3,2,1));
//    _mm_store_ss(out+18,c0_off);
//    _mm_store_ss(out+19,c1_off);
//    _mm_store_ss(out+20,c2_off);
//    _mm_storeu_ps(out+21,c3_off);

//    // row4
//    c_os = _mm_mul_ps(ofdiag_str,a2);
//    c_or = _mm_mul_ps(ofdiag_rev,a2);
//    __m128 c4_off = _mm_mul_ps(HALFPS,_mm_add_ps(c_os,c_or));
//    c4_off = _mm_shuffle_ps(c4_off,c4_off,_MM_SHUFFLE(0,3,2,1));
//    _mm_store_ss(out+24,_mm_shuffle_ps(c0_off,c0_off,_MM_SHUFFLE(0,0,0,1)));
//    _mm_store_ss(out+25,_mm_shuffle_ps(c1_off,c1_off,_MM_SHUFFLE(0,0,0,1)));
//    _mm_store_ss(out+26,_mm_shuffle_ps(c2_off,c2_off,_MM_SHUFFLE(0,0,0,1)));
//    _mm_storeu_ps(out+27,c4_off);

//    // row5
//    c_os = _mm_mul_ps(ofdiag_str,a5);
//    c_or = _mm_mul_ps(ofdiag_rev,a5);
//    __m128 c5_off = _mm_mul_ps(HALFPS,_mm_add_ps(c_os,c_or));
//    c5_off = _mm_shuffle_ps(c5_off,c5_off,_MM_SHUFFLE(0,3,2,1));
//    _mm_store_ss(out+30,_mm_shuffle_ps(c0_off,c0_off,_MM_SHUFFLE(0,0,0,2)));
//    _mm_store_ss(out+31,_mm_shuffle_ps(c1_off,c1_off,_MM_SHUFFLE(0,0,0,2)));
//    _mm_store_ss(out+32,_mm_shuffle_ps(c2_off,c2_off,_MM_SHUFFLE(0,0,0,2)));
//    _mm_storeu_ps(out+33,c5_off);

}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_HINT_INLINE void _outer<double,2,2,2,2>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {
    // Fetch a to L1-cache
//    _mm_prefetch(a,_MM_HINT_T0);
    __m256d a0 = _mm256_set1_pd(a[0]);
    __m256d a1 = _mm256_set1_pd(a[1]);
    __m256d a3 = _mm256_set1_pd(a[3]);
    __m256d bs = _mm256_load_pd(b);

    __m128d r0 = _mm_setr_pd(_mm256_get2_pd(bs),_mm256_get1_pd(bs));
    __m128d r1 = _mm_setr_pd(_mm256_get1_pd(bs),_mm256_get2_pd(bs));
    __m128d r2 = _mm_mul_pd(HALFPD,_mm_add_pd(r0,r1));
    __m256d r3 = _mm256_setr_pd(_mm256_get0_pd(bs),_mm256_get3_pd(bs),_mm_get0_pd(r2),0.0);

    // row0
    __m256d row0 = _mm256_mul_pd(a0,r3);
    _mm256_store_pd(out, row0);
    // row1
    __m256d row1 = _mm256_mul_pd(a3,r3);
    _mm_store_sd(out+3,_mm_set_sd(_mm256_get1_pd(row0)));
    _mm_store_pd(out+4,_mm_setr_pd(_mm256_get1_pd(row1),_mm256_get2_pd(row1)));
    // row2
    __m256d row2 = _mm256_mul_pd(a1,r3);
    _mm_store_sd(out+6,_mm256_extractf128_pd(row0,0x1));
    _mm_store_sd(out+7,_mm256_extractf128_pd(row1,0x1));
    _mm_store_sd(out+8,_mm256_extractf128_pd(row2,0x1));
}

template<>
FASTOR_HINT_INLINE void _outer<double,3,3,3,3>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {
    //  OPS
//    _mm_prefetch(a,_MM_HINT_T0);

    __m256d a0 = _mm256_set1_pd(a[0]);
    __m256d a1 = _mm256_set1_pd(a[1]);
    __m256d a2 = _mm256_set1_pd(a[2]);
    __m256d a4 = _mm256_set1_pd(a[4]);
    __m256d a5 = _mm256_set1_pd(a[5]);
    __m256d a8 = _mm256_set1_pd(a[8]);

    __m256d b_low = _mm256_load_pd(b);
    __m256d b_high = _mm256_load_pd(b+4);
    __m128d b_end = _mm_load_sd(b+8);

    __m256d b_diag = _mm256_setr_pd(_mm_cvtsd_f64(_mm256_castpd256_pd128(b_low)),
                                  _mm_cvtsd_f64(_mm256_castpd256_pd128(b_high)),
                                  _mm_cvtsd_f64(b_end),0.0);

    __m256d ofdiag_str = _mm256_setr_pd(b[1],b[2],b[5],0.0);
    __m256d ofdiag_rev = _mm256_setr_pd(b[3],b[6],b[7],0.0);
    // Compute this only once
    __m256d half_add_diag = _mm256_mul_pd(VHALFPD,_mm256_add_pd(ofdiag_str,ofdiag_rev));

    // row0
    __m256d c0_d = _mm256_mul_pd(b_diag,a0);
    __m256d c0_off = _mm256_mul_pd(a0,half_add_diag);
    _mm256_store_pd(out,c0_d);
    _mm256_storeu_pd(out+3,c0_off);
    // row1
    __m256d c1_d = _mm256_mul_pd(b_diag,a4);
    __m256d c1_off = _mm256_mul_pd(a4,half_add_diag);
    _mm256_storeu_pd(out+6,c1_d);
    _mm_store_sd(out+6,_mm256_castpd256_pd128(_mm256_shuffle_pd(c0_d,c0_d,_MM_SHUFFLE(0,0,0,1))));
    _mm256_storeu_pd(out+9,c1_off);
    // row2
    __m256d c2_d = _mm256_mul_pd(b_diag,a8);
    __m256d c2_off = _mm256_mul_pd(a8,half_add_diag);
    _mm_store_sd(out+12,_mm_set_sd(_mm256_get2_pd(c0_d)));
    _mm_store_sd(out+13,_mm_set_sd(_mm256_get2_pd(c1_d)));
    _mm_store_sd(out+14,_mm_set_sd(_mm256_get2_pd(c2_d)));
    _mm256_storeu_pd(out+15,c2_off);
    // row3
    __m256d c3_off = _mm256_mul_pd(a1,half_add_diag);
    _mm_store_sd(out+18,_mm256_castpd256_pd128(c0_off));
    _mm_store_sd(out+19,_mm256_castpd256_pd128(c1_off));
    _mm_store_sd(out+20,_mm256_castpd256_pd128(c2_off));
    _mm256_storeu_pd(out+21,c3_off);
    // row4
    __m256d c4_off = _mm256_mul_pd(a2,half_add_diag);
    _mm_store_sd(out+24,_mm_set_sd(_mm256_get1_pd(c0_off)));
    _mm_store_sd(out+25,_mm_set_sd(_mm256_get1_pd(c1_off)));
    _mm_store_sd(out+26,_mm_set_sd(_mm256_get1_pd(c2_off)));
    _mm256_storeu_pd(out+27,c4_off);
    // row4
    __m256d c5_off = _mm256_mul_pd(a5,half_add_diag);
    _mm_store_sd(out+30,_mm_set_sd(_mm256_get2_pd(c0_off)));
    _mm_store_sd(out+31,_mm_set_sd(_mm256_get2_pd(c1_off)));
    _mm_store_sd(out+32,_mm_set_sd(_mm256_get2_pd(c2_off)));
    _mm256_storeu_pd(out+33,c5_off);
}
#endif

}

#endif // OUTER_H

