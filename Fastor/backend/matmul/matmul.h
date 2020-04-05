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

    using nativeV = SIMDVector<T,DEFAULT_ABI>;
    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;

    // Use specialised kernels
    FASTOR_IF_CONSTEXPR( M==N && M==K && ((M==12UL || M==24UL || M==33) && std::is_same<T,float>::value)) {
        internal::_matmul_mkn_non_square<T,M,K,N>(a,b,out);
        return;
    }

    FASTOR_IF_CONSTEXPR((N==V::Size || N==2*V::Size) && V::Size!=1UL) {
        internal::_matmul_mk_lessthan2simd<T,M,K,N>(a,b,out);
        return;
    }

#if defined(FASTOR_AVX_IMPL) && !defined(FASTOR_AVX512_IMPL)
    // Works for AVX512. Only maskload/maskstore should be overloaded or ifdefed
    // to use mask_load/mask_store of avx512
    FASTOR_IF_CONSTEXPR((N<2*V::Size && N!=1UL)) {
        internal::_matmul_mk_lessthan2simd<T,M,K,N>(a,b,out);
        return;
    }
#endif

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
        // For all other cases where M,N,K is too small
        // this simple version is sufficient
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


} // end of namespace

#endif // MATMUL_H
