#ifndef MATMUL_H
#define MATMUL_H

#include "Fastor/meta/meta.h"
#include "Fastor/backend/matmul/matmul_kernels.h"

#ifdef FASTOR_USE_LIBXSMM
#include "Fastor/backend/matmul/libxsmm_backend.h"
#endif
#ifdef FASTOR_USE_MKL
#include "Fastor/backend/matmul/mkl_backend.h"
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
#if !defined(FASTOR_USE_LIBXSMM) && !defined(FASTOR_USE_MKL)
template<typename T, size_t M, size_t K, size_t N,
         enable_if_t_<!(M!=K && M==N && (M==2UL || M==3UL || M==4UL || M==8UL) && (is_same_v_<T,float> || is_same_v_<T,double>) ),bool> = 0>
#else
template<typename T, size_t M, size_t K, size_t N,
         enable_if_t_<
            !(M!=K && M==N && (M==2UL || M==3UL || M==4UL || M==8UL) && (is_same_v_<T,float> || is_same_v_<T,double>) )
            && is_less_equal<M*N*K/internal::meta_cube<FASTOR_BLAS_SWITCH_MATRIX_SIZE>::value,1>::value,
            bool> = 0>
#endif
FASTOR_INLINE
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    // Non-primitive types
    FASTOR_IF_CONSTEXPR (!is_primitive_v_<T>) {
        internal::_matmul_base_non_primitive<T,M,K,N>(a,b,out);
        return;
    }

    // Matrix-vector specialisation
    FASTOR_IF_CONSTEXPR (N==1UL) {
        internal::_matvecmul<T,M,K>(a,b,out);
        return;
    }

    using nativeV = SIMDVector<T,DEFAULT_ABI>;
    using V = choose_best_simd_t<nativeV,N>;

    // Use specialised kernels
    FASTOR_IF_CONSTEXPR((N==V::Size || N==2*V::Size || N==3*V::Size || N==4*V::Size || N==5*V::Size) && V::Size!=1UL) {
        internal::_matmul_mk_smalln<T,M,K,N>(a,b,out);
        return;
    }

#if defined(FASTOR_AVX2_IMPL) || defined(FASTOR_HAS_AVX512_MASKS)
    FASTOR_IF_CONSTEXPR((N<5*V::Size && N!=1UL)) {
        internal::_matmul_mk_smalln<T,M,K,N>(a,b,out);
        return;
    }
#endif

#if defined(FASTOR_AVX2_IMPL) || defined(FASTOR_HAS_AVX512_MASKS)
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
    else
    {
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


#if defined(FASTOR_USE_LIBXSMM) && !defined(FASTOR_USE_MKL)
template<typename T, size_t M, size_t K, size_t N,
        enable_if_t_<
            !(M!=K && M==N && (M==2UL || M==3UL || M==4UL || M==8UL) && (is_same_v_<T,float> || is_same_v_<T,double>) )
            && is_greater<M*N*K/internal::meta_cube<FASTOR_BLAS_SWITCH_MATRIX_SIZE>::value,1>::value,
            bool> = 0>
FASTOR_INLINE
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {
    blas::matmul_libxsmm<T,M,K,N>(a,b,c);
}
#endif

#if !defined(FASTOR_USE_LIBXSMM) && defined(FASTOR_USE_MKL)
template<typename T, size_t M, size_t K, size_t N,
        enable_if_t_<
            !(M!=K && M==N && (M==2UL || M==3UL || M==4UL || M==8UL) && (is_same_v_<T,float> || is_same_v_<T,double>) )
            && is_greater<M*N*K/internal::meta_cube<FASTOR_BLAS_SWITCH_MATRIX_SIZE>::value,1>::value,
            bool> = 0>
FASTOR_INLINE
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {
    blas::matmul_mkl<T,M,K,N>(a,b,c);
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
