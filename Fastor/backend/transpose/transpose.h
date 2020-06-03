#ifndef TRANSPOSE_H
#define TRANSPOSE_H


#include "Fastor/config/config.h"
#include "Fastor/backend/transpose/transpose_kernels.h"
#include "Fastor/simd_vector/extintrin.h"
#include "Fastor/simd_vector/SIMDVector.h"

namespace Fastor {

// Forward declare
namespace internal {
template<typename T, size_t M, size_t N>
FASTOR_INLINE void _transpose_dispatch(const T * FASTOR_RESTRICT a, T * FASTOR_RESTRICT out);
} // internal


//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_AVX_IMPL

template<typename T, size_t M, size_t N>
FASTOR_INLINE void _transpose(const T * FASTOR_RESTRICT a, T * FASTOR_RESTRICT out) {

    using V = SIMDVector<T,DEFAULT_ABI>;

    // Block sizes of 8x8 i.e. numSIMDRows=1
    // numSIMDCols=1 and innerBlock=outerBlock=1
    // give a much greater speed up, but causes
    // significant slow-down for issue #42

#ifndef FASTOR_TRANS_OUTER_BLOCK_SIZE
    constexpr size_t numSIMDRows = 1UL;
#else
    constexpr size_t numSIMDRows = FASTOR_TRANS_OUTER_BLOCK_SIZE;
#endif
#ifndef FASTOR_TRANS_INNER_BLOCK_SIZE
    constexpr size_t numSIMDCols = 1UL;
#else
    constexpr size_t numSIMDCols = FASTOR_TRANS_INNER_BLOCK_SIZE;
#endif

    constexpr size_t innerBlock = V::Size * numSIMDCols;
    constexpr size_t outerBlock = V::Size * numSIMDRows;

    FASTOR_ARCH_ALIGN T pack_a[outerBlock*innerBlock];
    FASTOR_ARCH_ALIGN T pack_out[outerBlock*innerBlock];

    constexpr size_t M0 = M / innerBlock * innerBlock;
    constexpr size_t N0 = N / outerBlock * outerBlock;
    V _vec;

    // For row-major matrices we go over N
    // and then M to get contiguous writes
    size_t j=0;
    for (; j<N0; j+=outerBlock) {

        size_t i=0;
        for (; i< M0; i+=innerBlock) {
            // Pack A
            for (size_t ii=0; ii<innerBlock; ++ii) {
                _vec.load(&a[(i+ii)*N+(j)],false);
                _vec.store(&pack_a[ii*outerBlock]);
            }
            // Perform transpose on pack_a and get the result
            // on pack_out
            internal::_transpose_dispatch<T,innerBlock,outerBlock>(pack_a,pack_out);
            // Unpack pack_out to out
            for (size_t jj=0; jj<outerBlock; ++jj) {
                _vec.load(&pack_out[jj*innerBlock]);
                _vec.store(&out[(j+jj)*M+(i)],false);
            }
        }

        // Remainer M - M0 columns (of c)
        for (; i< M; ++i) {
            for (size_t jj=0; jj<outerBlock; ++jj) {
                out[(j+jj)*M+(i)] = a[i*N+j+jj];
            }
        }
    }

    // Remainder N - N0 rows (of c)
    for (; j<N; ++j) {
        for (size_t i=0; i< M; ++i) {
            out[(j)*M+(i)] = a[i*N+j];
        }
    }
}

#else

template<typename T, size_t M, size_t N>
FASTOR_INLINE void _transpose(const T * FASTOR_RESTRICT a, T * FASTOR_RESTRICT out) {
    for (size_t j=0; j<N; ++j)
        for (size_t i=0; i< M; ++i)
            out[j*M+i] = a[i*N+j];
}
#endif
//----------------------------------------------------------------------------------------------------------//



// Specialisations - float
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE void _transpose<float,2,2>(const float * FASTOR_RESTRICT a, float * FASTOR_RESTRICT out) {
    __m128 a_reg = _mm_loadu_ps(a);
    _mm_storeu_ps(out,_mm_shuffle_ps(a_reg,a_reg,_MM_SHUFFLE(3,1,2,0)));
}

template<>
FASTOR_INLINE void _transpose<float,3,3>(const float * FASTOR_RESTRICT a, float * FASTOR_RESTRICT out) {
#ifndef FASTOR_AVX2_IMPL
    // 5 OPS
    __m128 row0 = _mm_loadu_ps(a);
    __m128 row1 = _mm_loadu_ps(a+3);
    __m128 row2 = _mm_loadu_ps(a+6);

    __m128 T0   = _mm_unpacklo_ps(row0,row1);
    __m128 T1   = _mm_unpackhi_ps(row0,row1);

    row0 = _mm_movelh_ps ( T0,row2 );
    row1 = _mm_shuffle_ps( T0,row2, _MM_SHUFFLE(3,1,3,2) );
    row2 = _mm_shuffle_ps( T1,row2, _MM_SHUFFLE(3,2,1,0) );

    _mm_storeu_ps(out,row0);
    _mm_storeu_ps(out+3,row1);
    _mm_storeu_ps(out+6,row2); // out of range for out[9]
#else
    // 3 OPS
    // gcc/clang emit vpermsps tht operate on (%rsp)
    // less pressure on shuffle port perhaps
    __m256 trans07           = _mm256_loadu_ps(a);
    const __m256i trans_mask = _mm256_setr_epi32(
         0,3,6,
         1,4,7,
         2,5);
    // does not shuffle across 256 lanes, only 128 lanes
    // __m256 _res = _mm256_permutevar_ps(trans07, trans_mask);
    // this one shuffles correctly
    __m256 _res = _mm256_permutevar8x32_ps(trans07, trans_mask);
    _mm256_storeu_ps(out,_res);
    // out[8] = a[8];
    _mm_store_ss(out+8,_mm_load_ss(a+8));
#endif
}

template<>
FASTOR_INLINE void _transpose<float,4,4>(const float * FASTOR_RESTRICT a, float * FASTOR_RESTRICT out) {
#ifdef FASTOR_AVX512F_IMPL
   __m512 amm  = _mm512_loadu_ps(a);
   __m512i idx = _mm512_setr_epi32( 0, 4, 8, 12,
                                    1, 5, 9, 13,
                                    2, 6, 10, 14,
                                    3, 7, 11, 15);
    __m512 omm = _mm512_permutexvar_ps(idx, amm);
    _mm512_storeu_ps(out, omm);
#else
    __m128 row1 = _mm_loadu_ps(a);
    __m128 row2 = _mm_loadu_ps(a+4);
    __m128 row3 = _mm_loadu_ps(a+8);
    __m128 row4 = _mm_loadu_ps(a+12);
     _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
     _mm_storeu_ps(out   , row1);
     _mm_storeu_ps(out+4 , row2);
     _mm_storeu_ps(out+8 , row3);
     _mm_storeu_ps(out+12, row4);
#endif
}
#endif

#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE void _transpose<float,8,8>(const float * FASTOR_RESTRICT a, float * FASTOR_RESTRICT out) {
    __m256 row1 = _mm256_loadu_ps(a);
    __m256 row2 = _mm256_loadu_ps(a+8);
    __m256 row3 = _mm256_loadu_ps(a+16);
    __m256 row4 = _mm256_loadu_ps(a+24);
    __m256 row5 = _mm256_loadu_ps(a+32);
    __m256 row6 = _mm256_loadu_ps(a+40);
    __m256 row7 = _mm256_loadu_ps(a+48);
    __m256 row8 = _mm256_loadu_ps(a+56);
    internal::_MM_TRANSPOSE8_PS(row1, row2, row3, row4, row5, row6, row7, row8);
    _mm256_storeu_ps(out, row1);
    _mm256_storeu_ps(out+8, row2);
    _mm256_storeu_ps(out+16, row3);
    _mm256_storeu_ps(out+24, row4);
    _mm256_storeu_ps(out+32, row5);
    _mm256_storeu_ps(out+40, row6);
    _mm256_storeu_ps(out+48, row7);
    _mm256_storeu_ps(out+56, row8);
}
#endif

#if defined(FASTOR_AVX512F_IMPL) && defined(FASTOR_AVX512DQ_IMPL)
template<>
FASTOR_INLINE void _transpose<float,16,16>(const float * FASTOR_RESTRICT a, float * FASTOR_RESTRICT out) {
    internal::_MM_TRANSPOSE16_PS(a,out);
}
#endif
//----------------------------------------------------------------------------------------------------------//



// Specialisations - double
//----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE void _transpose<double,2,2>(const double* FASTOR_RESTRICT a, double* FASTOR_RESTRICT out) {
    /*-------------------------------------------------------*/
    // 2 OPS
    __m128d row0 = _mm_loadu_pd(a);
    __m128d row1 = _mm_loadu_pd(a+2);
    __m128d tmp  = row0;
    row0         = _mm_shuffle_pd(row0,row1,0x0);
    row1         = _mm_shuffle_pd(tmp ,row1,0x3);
    _mm_storeu_pd(out  ,row0);
    _mm_storeu_pd(out+2,row1);
    /*-------------------------------------------------------*/

    /*-------------------------------------------------------*/
//    // AVX VERSION
    // // IVY 4 OPS / HW 8 OPS
    // __m256d a1 =  _mm256_loadu_pd(a);
    // __m128d a2 =  _mm256_castpd256_pd128(a1);
    // __m128d a3 =  _mm256_extractf128_pd(a1,0x1);
    // __m128d a4 =  _mm_shuffle_pd(a2,a3,0x0);
    // a3         = _mm_shuffle_pd(a2,a3,0x3);
    // a1         = _mm256_castpd128_pd256(a4);
    // a1         = _mm256_insertf128_pd(a1,a3,0x1);
    // _mm256_storeu_pd(out,a1);
    /*-------------------------------------------------------*/
}
#endif

#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE void _transpose<double,3,3>(const double* FASTOR_RESTRICT a, double* FASTOR_RESTRICT out) {
    // AVX512 is the fastest & AVX version despite more instructions is faster than SSE
#if defined(FASTOR_AVX512F_IMPL)
   // AVX512
   /*-------------------------------------------------------*/
    __m512d a07 = _mm512_loadu_pd(a);
    const __m512i trans_mask = _mm512_setr_epi64(
        0,3,6,
        1,4,7,
        2,5);
    __m512d trans07 = _mm512_permutexvar_pd(trans_mask, a07);
    _mm512_storeu_pd(out,trans07);
    _mm_store_sd(out+8,_mm_load_sd(a+8));
   /*-------------------------------------------------------*/
#elif defined(FASTOR_AVX_IMPL)
   // AVX
   /*-------------------------------------------------------*/
   __m256d row1 = _mm256_loadu_pd(a);
   __m256d row2 = _mm256_loadu_pd(a+4);

   __m128d a11 = _mm256_castpd256_pd128(row1);
   __m128d a12 = _mm256_extractf128_pd(row1,0x1);
   __m128d a21 = _mm256_castpd256_pd128(row2);
   __m128d a22 = _mm256_extractf128_pd(row2,0x1);

   row1 = _mm256_castpd128_pd256(_mm_shuffle_pd(a11,a12,0x2));
   row1 = _mm256_insertf128_pd(row1,_mm_shuffle_pd(a22,a11,0x2),0x1);
   row2 = _mm256_castpd128_pd256(_mm_shuffle_pd(a21,a22,0x2));
   row2 = _mm256_insertf128_pd(row2,_mm_shuffle_pd(a12,a21,0x2),0x1);

   _mm256_storeu_pd(out,row1);
   _mm256_storeu_pd(out+4,row2);
   _mm_store_sd(out+8,_mm_load_sd(a+8));
   /*-------------------------------------------------------*/
#else
    // SSE
    /*-------------------------------------------------------*/
    __m128d a11 = _mm_loadu_pd(a);
    __m128d a12 = _mm_loadu_pd(a+2);
    __m128d a21 = _mm_loadu_pd(a+4);
    __m128d a22 = _mm_loadu_pd(a+6);

    _mm_storeu_pd(out  ,_mm_shuffle_pd(a11,a12,0x2));
    _mm_storeu_pd(out+2,_mm_shuffle_pd(a22,a11,0x2));
    _mm_storeu_pd(out+4,_mm_shuffle_pd(a21,a22,0x2));
    _mm_storeu_pd(out+6,_mm_shuffle_pd(a12,a21,0x2));
    _mm_store_sd (out+8,_mm_load_sd(a+8));
    /*-------------------------------------------------------*/
#endif
}
#endif

#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE void _transpose<double,4,4>(const double * FASTOR_RESTRICT a, double * FASTOR_RESTRICT out) {
#ifdef FASTOR_AVX512F_IMPL
    __m512d amm0 = _mm512_loadu_pd(a);
    __m512d amm1 = _mm512_loadu_pd(a+8);
    __m512i idx0 = _mm512_setr_epi64(0, 4,  8, 12, 1, 5,  9, 13);
    __m512i idx1 = _mm512_setr_epi64(2, 6, 10, 14, 3, 7, 11, 15);
    __m512d omm0 = _mm512_permutex2var_pd(amm0, idx0, amm1);
    __m512d omm1 = _mm512_permutex2var_pd(amm0, idx1, amm1);
    _mm512_storeu_pd(out  , omm0);
    _mm512_storeu_pd(out+8, omm1);
#else
    __m256d row1 = _mm256_loadu_pd(a);
    __m256d row2 = _mm256_loadu_pd(a+4);
    __m256d row3 = _mm256_loadu_pd(a+8);
    __m256d row4 = _mm256_loadu_pd(a+12);
    internal::_MM_TRANSPOSE4_PD(row1, row2, row3, row4);
    _mm256_storeu_pd(out, row1);
    _mm256_storeu_pd(out+4, row2);
    _mm256_storeu_pd(out+8, row3);
    _mm256_storeu_pd(out+12, row4);
#endif
}
#endif

#if defined(FASTOR_AVX512F_IMPL)
template<>
FASTOR_INLINE void _transpose<double,8,8>(const double * FASTOR_RESTRICT a, double * FASTOR_RESTRICT out) {
    __m512d row0 = _mm512_loadu_pd(a);
    __m512d row1 = _mm512_loadu_pd(a+8);
    __m512d row2 = _mm512_loadu_pd(a+16);
    __m512d row3 = _mm512_loadu_pd(a+24);
    __m512d row4 = _mm512_loadu_pd(a+32);
    __m512d row5 = _mm512_loadu_pd(a+40);
    __m512d row6 = _mm512_loadu_pd(a+48);
    __m512d row7 = _mm512_loadu_pd(a+56);
    internal::_MM_TRANSPOSE8_PD(row0,row1,row2,row3,row4,row5,row6,row7);
    _mm512_storeu_pd(out   , row0);
    _mm512_storeu_pd(out+8 , row1);
    _mm512_storeu_pd(out+16, row2);
    _mm512_storeu_pd(out+24, row3);
    _mm512_storeu_pd(out+32, row4);
    _mm512_storeu_pd(out+40, row5);
    _mm512_storeu_pd(out+48, row6);
    _mm512_storeu_pd(out+56, row7);
}
#elif defined(FASTOR_AVX_IMPL)
template<>
FASTOR_INLINE void _transpose<double,8,8>(const double * FASTOR_RESTRICT a, double * FASTOR_RESTRICT out) {

    {
        __m256d row1 = _mm256_loadu_pd(&a[0]);
        __m256d row2 = _mm256_loadu_pd(&a[8]);
        __m256d row3 = _mm256_loadu_pd(&a[16]);
        __m256d row4 = _mm256_loadu_pd(&a[24]);
        internal::_MM_TRANSPOSE4_PD(row1, row2, row3, row4);
        _mm256_storeu_pd(&out[0],  row1);
        _mm256_storeu_pd(&out[8],  row2);
        _mm256_storeu_pd(&out[16], row3);
        _mm256_storeu_pd(&out[24], row4);
    }

    {
        __m256d row1 = _mm256_loadu_pd(&a[32]);
        __m256d row2 = _mm256_loadu_pd(&a[40]);
        __m256d row3 = _mm256_loadu_pd(&a[48]);
        __m256d row4 = _mm256_loadu_pd(&a[56]);
        internal::_MM_TRANSPOSE4_PD(row1, row2, row3, row4);
        _mm256_storeu_pd(&out[4],  row1);
        _mm256_storeu_pd(&out[12], row2);
        _mm256_storeu_pd(&out[20], row3);
        _mm256_storeu_pd(&out[28], row4);
    }

    {
        __m256d row1 = _mm256_loadu_pd(&a[4]);
        __m256d row2 = _mm256_loadu_pd(&a[12]);
        __m256d row3 = _mm256_loadu_pd(&a[20]);
        __m256d row4 = _mm256_loadu_pd(&a[28]);
        internal::_MM_TRANSPOSE4_PD(row1, row2, row3, row4);
        _mm256_storeu_pd(&out[32], row1);
        _mm256_storeu_pd(&out[40], row2);
        _mm256_storeu_pd(&out[48], row3);
        _mm256_storeu_pd(&out[56], row4);
    }

    {
        __m256d row1 = _mm256_loadu_pd(&a[36]);
        __m256d row2 = _mm256_loadu_pd(&a[44]);
        __m256d row3 = _mm256_loadu_pd(&a[52]);
        __m256d row4 = _mm256_loadu_pd(&a[60]);
        internal::_MM_TRANSPOSE4_PD(row1, row2, row3, row4);
        _mm256_storeu_pd(&out[36], row1);
        _mm256_storeu_pd(&out[44], row2);
        _mm256_storeu_pd(&out[52], row3);
        _mm256_storeu_pd(&out[60], row4);
    }
}
#endif
//----------------------------------------------------------------------------------------------------------//




//----------------------------------------------------------------------------------------------------------//
namespace internal {
// To get around compilers recusive inlining depth issue
template<typename T, size_t M, size_t N>
FASTOR_INLINE void _transpose_dispatch(const T * FASTOR_RESTRICT a, T * FASTOR_RESTRICT out) {
    for (size_t j=0; j<N; ++j)
        for (size_t i=0; i< M; ++i)
            out[j*M+i] = a[i*N+j];
}
template<>
FASTOR_INLINE void _transpose_dispatch<float,2,2>(const float * FASTOR_RESTRICT a, float * FASTOR_RESTRICT out) {
    _transpose<float,2,2>(a,out);
}
template<>
FASTOR_INLINE void _transpose_dispatch<float,3,3>(const float * FASTOR_RESTRICT a, float * FASTOR_RESTRICT out) {
    _transpose<float,3,3>(a,out);
}
template<>
FASTOR_INLINE void _transpose_dispatch<float,4,4>(const float * FASTOR_RESTRICT a, float * FASTOR_RESTRICT out) {
    _transpose<float,4,4>(a,out);
}
template<>
FASTOR_INLINE void _transpose_dispatch<float,8,8>(const float * FASTOR_RESTRICT a, float * FASTOR_RESTRICT out) {
    _transpose<float,8,8>(a,out);
}
template<>
FASTOR_INLINE void _transpose_dispatch<float,16,16>(const float * FASTOR_RESTRICT a, float * FASTOR_RESTRICT out) {
    _transpose<float,16,16>(a,out);
}
template<>
FASTOR_INLINE void _transpose_dispatch<double,2,2>(const double * FASTOR_RESTRICT a, double * FASTOR_RESTRICT out) {
    _transpose<double,2,2>(a,out);
}
template<>
FASTOR_INLINE void _transpose_dispatch<double,3,3>(const double * FASTOR_RESTRICT a, double * FASTOR_RESTRICT out) {
    _transpose<double,3,3>(a,out);
}
template<>
FASTOR_INLINE void _transpose_dispatch<double,4,4>(const double * FASTOR_RESTRICT a, double * FASTOR_RESTRICT out) {
    _transpose<double,4,4>(a,out);
}
template<>
FASTOR_INLINE void _transpose_dispatch<double,8,8>(const double * FASTOR_RESTRICT a, double * FASTOR_RESTRICT out) {
    _transpose<double,8,8>(a,out);
}
} // internal
//----------------------------------------------------------------------------------------------------------//

}

#endif // TRANSPOSE_H

