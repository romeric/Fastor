#ifndef MATMUL_H
#define MATMUL_H

#include "Fastor/backend/matmul/matmul_kernels.h"

#ifdef FASTOR_USE_LIBXSMM
#include "Fastor/backend/matmul/libxsmm_backend.h"
#endif

namespace Fastor {



// Forward declare
//-----------------------------------------------------------------------------------------------------------
namespace internal {
template<typename T, size_t M, size_t N>
FASTOR_INLINE
void _matvecmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out);
} // internal
//-----------------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------
#ifndef FASTOR_USE_LIBXSMM
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<!(M!=K && M==N && (M==2UL || M==3UL || M==4UL || M==8UL)),bool>::type = 0>
#else
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<
            !(M!=K && M==N && (M==2UL || M==3UL || M==4UL|| M==8UL))
            && is_less_equal<M*N*K/internal::meta_cube<FASTOR_BLAS_SWITCH_MATRIX_SIZE>::value,1>::value,
            bool>::type = 0>
#endif
FASTOR_INLINE
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    // Matrix-vector specialisation
    FASTOR_IF_CONSTEXPR (N==1) {
        internal::_matvecmul<T,M,K>(a,b,out);
        return;
    }

    using V = SIMDVector<T,DEFAULT_ABI>;

    // Use specialised kernels
    FASTOR_IF_CONSTEXPR( M<=16UL && (N==V::Size && V::Size!=1)) {
        internal::_matmul_mk_simd_width<T,M,K,N>(a,b,out);
        return;
    }

    FASTOR_IF_CONSTEXPR( M==N && M==K && ((M==12UL || M==24UL) && std::is_same<T,float>::value)) {
        internal::_matmul_mkn_square<T,M,K,N>(a,b,out);
        return;
    }
    FASTOR_IF_CONSTEXPR( M==N && M==K && ((M==33) && std::is_same<T,float>::value)) {
        internal::_matmul_mkn_non_square<T,M,K,N>(a,b,out);
        return;
    }

    // This is the correct logic for the time being as in
    // when AVX is available we want maskloads and when
    // avx512 is available we don't since maskload/maskstores
    // are not available for avx512 only mask_load/stores are
    // available yet
#if defined(FASTOR_AVX_IMPL) && !defined(FASTOR_AVX512_IMPL)
    // If the remainder is 1, just treat the remainders in scalar mode
    FASTOR_IF_CONSTEXPR( M*N*K > 27UL && N % V::Size <= 1UL) {
        internal::_matmul_base<T,M,K,N>(a,b,out);
        return;
    }
    else FASTOR_IF_CONSTEXPR( M*N*K > 27UL && N % V::Size > 1UL) {
        internal::_matmul_base_masked<T,M,K,N>(a,b,out);
        return;
    }
#else
    FASTOR_IF_CONSTEXPR( M*N*K > 27UL ) {
        internal::_matmul_base<T,M,K,N>(a,b,out);
        return;
    }
#endif
    else {
        constexpr int ROUND_ = ROUND_DOWN(N,V::Size);
        for (size_t j=0; j<M; ++j) {
            size_t k=0;
            for (; k<ROUND_; k+=V::Size) {
                V out_row;
                for (size_t i=0; i<K; ++i) {
                    const V brow(&b[i*N+k],false);
                    const V vec_a(a[j*K+i]);
                    out_row = fmadd(vec_a,brow,out_row);
                }
                out_row.store(&out[k+N*j],false);
            }
            for (; k<N; k++) {
                T out_row = 0.;
                for (size_t i=0; i<K; ++i) {
                    out_row += a[j*K+i]*b[i*N+k];
                }
                out[N*j+k] = out_row;
            }
        }
    }
}


#ifdef FASTOR_USE_LIBXSMM
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<
            !(M!=K && M==N && (M==2 || M==3 || M==4))
            && is_greater<M*N*K/internal::meta_cube<FASTOR_BLAS_SWITCH_MATRIX_SIZE>::value,1>::value,
            bool>::type = 0>
FASTOR_INLINE
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {
    blas::matmul_libxsmm<T,M,K,N>(a,b,c);
}
#endif
//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------
#include "Fastor/backend/matmul/matmul_specialisations_kernels.h"
//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------



//! Matrix-vector multiplication
//-----------------------------------------------------------------------------------------------------------
// Don't call this function directly as it's name is unconventional
// It gets called from within matmul anyway so always call matmul
namespace internal {

template<typename T, size_t M, size_t N>
FASTOR_INLINE
void _matvecmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    using V = SIMDVector<T,DEFAULT_ABI>;
    constexpr int ROUND = ROUND_DOWN(N,V::Size);

    // V _vec_a, _vec_b;
    // for (int i=0; i< M; ++i) {
    //     V _vec_out;
    //     int j = 0;
    //     for (; j< ROUND; j+=V::Size) {
    //         _vec_a.load(&a[i*N+j]);
    //         _vec_b.load(&b[j]);
    //         _vec_out += _vec_a*_vec_b;
    //     }
    //     // _vec_out.store(&out[i]);
    //     T out_s = 0;
    //     for (; j< ROUND; j+=V::Size) {
    //         out_s += a[i*N+j]*b[j];
    //     }
    //     out[i]= _vec_out.sum() + out_s;
    // }

    // Unroll the outer loop to get two independent parallel chains
    // of accumulators. This gives you two FMAs for 3 loads (2 from a and one from b)
    int i=0;
    for (; i<ROUND_DOWN(M,2); i+=2) {
        V _vec_out0, _vec_out1;
        int j = 0;
        for (; j< ROUND; j+=V::Size) {
            V _vec_a0(&a[i*N+j]);
            V _vec_a1(&a[(i+1)*N+j]);
            V _vec_b(&b[j]);

            _vec_out0 = fmadd(_vec_a0,_vec_b,_vec_out0);
            _vec_out1 = fmadd(_vec_a1,_vec_b,_vec_out1);
        }
        T out_s0 = 0;
        T out_s1 = 0;
        for (; j< N; j+=1) {
            out_s0 += a[i*N+j]*b[j];
            out_s1 += a[(i+1)*N+j]*b[j];
        }
        out[i]= _vec_out0.sum() + out_s0;
        out[i+1]= _vec_out1.sum() + out_s1;
    }

    for (; i<M; ++i) {
        V _vec_out0;
        int j = 0;
        for (; j< ROUND; j+=V::Size) {
            V _vec_a0(&a[i*N+j]);
            V _vec_b(&b[j]);

            _vec_out0 = fmadd(_vec_a0,_vec_b,_vec_out0);
        }
        T out_s0 = 0;
        for (; j< N; j+=1) {
            out_s0 += a[i*N+j]*b[j];
        }
        out[i]= _vec_out0.sum() + out_s0;
    }
}

}


#ifdef FASTOR_SSE4_2_IMPL
template<>
FASTOR_INLINE
void _matmul<float,2,2,1>(const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {
    // 11 OPS
    __m128 a_reg = _mm_load_ps(a);
    __m128 vec_b = _mm_load_ps(b);

    __m128 b0 = _mm_movelh_ps(vec_b,vec_b);
    __m128 res = _mm_mul_ps(a_reg,b0);
    __m128 res2 = _mm_shuffle_ps(res,res,_MM_SHUFFLE(2,3,0,1));
    res2 = _mm_add_ps(res,res2);
    __m128 res3 = _mm_shuffle_ps(res2,res2,_MM_SHUFFLE(3,1,2,0));
    _mm_storel_pi((__m64*) out,res3);
}

template<>
FASTOR_INLINE void _matmul<float,3,3,1>(const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {
    // IVY/HW 47 OPS

    // 12 ss loads so probably inefficent
    __m128 amm0 = _mm_loadul3_ps(a);
    __m128 amm1 = _mm_loadul3_ps(&a[3]);
    __m128 amm2 = _mm_loadul3_ps(&a[6]);
    __m128 bmm  = _mm_loadul3_ps(b);

    // This is probably more efficient but compiler depdendent
    // __m128 amm0 = _mm_setr_ps(a[0],a[1],a[2],0.f);
    // __m128 amm1 = _mm_setr_ps(a[3],a[4],a[5],0.f);
    // __m128 amm2 = _mm_setr_ps(a[6],a[7],a[8],0.f);
    // __m128 bmm  = _mm_setr_ps(b[0],b[1],b[2],0.f);

    out[0] =_mm_sum_ps(_mm_mul_ps(amm0,bmm));
    out[1] =_mm_sum_ps(_mm_mul_ps(amm1,bmm));
    out[2] =_mm_sum_ps(_mm_mul_ps(amm2,bmm));
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE void _matmul<double,2,2,1>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {
    // IVY 15 OPS - HW 19 OPS
    __m256d a_reg = _mm256_load_pd(a);
    __m128d b_vec = _mm_load_pd(b);

    __m256d b0 = _mm256_castpd128_pd256(b_vec);
    b0 = _mm256_insertf128_pd(b0,b_vec,0x1);
    __m256d res = _mm256_mul_pd(a_reg,b0);
    _mm_store_sd(out,_add_pd(_mm256_castpd256_pd128(res)));
    _mm_store_sd(out+1,_add_pd(_mm256_extractf128_pd(res,0x1)));
}


template<>
FASTOR_INLINE void _matmul<double,3,3,1>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {
    // IVY 58 OPS - HW 84 OPS
    __m128d a0 = _mm_load_pd(a);
    __m128d a1 = _mm_load_sd(a+2);
    __m256d row0 = _mm256_castpd128_pd256(a0);
    row0 = _mm256_shift1_pd(_mm256_insertf128_pd(row0,a1,0x1));

    __m128d a2 = _mm_reverse_pd(_mm_load_sd(a+3));
    __m128d a3 = _mm_load_pd(a+4);
    __m256d row1 = _mm256_castpd128_pd256(a2);
    row1 = _mm256_insertf128_pd(row1,a3,0x1);

    __m128d a4 = _mm_load_pd(a+6);
    __m128d a5 = _mm_load_sd(a+8);
    __m256d row2 = _mm256_castpd128_pd256(a4);
    row2 = _mm256_shift1_pd(_mm256_insertf128_pd(row2,a5,0x1));

    __m256d vec_b = _mm256_shift1_pd(_mm256_load_pd(b));

    __m128d c0 = _add_pd(_mm256_mul_pd(row0,vec_b));
    __m128d c1 = _add_pd(_mm256_mul_pd(row1,vec_b));
    __m128d c2 = _add_pd(_mm256_mul_pd(row2,vec_b));

    _mm_store_sd(out,c0);
    _mm_store_sd(out+1,c1);
    _mm_store_sd(out+2,c2);

    // // Alternatively
    // // SKY 49 OPS + 4*_mm256_set_epi64x operations
    // // ICE 87 OPS + 4*_mm256_set_epi64x operations
    // __m256d amm0 = _mm256_loadul3_pd(a);
    // __m256d amm1 = _mm256_loadul3_pd(&a[3]);
    // __m256d amm2 = _mm256_loadul3_pd(&a[6]);
    // __m256d bmm  = _mm256_loadul3_pd(b);

    // // This is probably more efficient but compiler depdendent
    // // __m256d amm0 = _mm_setr_ps(a[0],a[1],a[2],0.f);
    // // __m256d amm1 = _mm_setr_ps(a[3],a[4],a[5],0.f);
    // // __m256d amm2 = _mm_setr_ps(a[6],a[7],a[8],0.f);
    // // __m256d bmm  = _mm_setr_ps(b[0],b[1],b[2],0.f);

    // out[0] =_mm256_sum_pd(_mm256_mul_pd(amm0,bmm));
    // out[1] =_mm256_sum_pd(_mm256_mul_pd(amm1,bmm));
    // out[2] =_mm256_sum_pd(_mm256_mul_pd(amm2,bmm));
}
#endif



} // end of namespace

#endif // MATMUL_H

