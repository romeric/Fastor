#ifndef MATMUL_MK_SMALLODDN_H
#define MATMUL_MK_SMALLODDN_H


#include "Fastor/simd_vector/simd_vector_abi.h"
#include "Fastor/simd_vector/SIMDVector.h"

namespace Fastor {

namespace internal {



// This implementation covers all matrix-matrix multiplications with any M and K and
// and N<=5*SIMDVector::Size. Given that it uses choose_best_simd_type it can switch
// between SSE, AVX and AVX512 to cover all ranges of N


//-----------------------------------------------------------------------------------------------------------
#ifdef FASTOR_HAS_AVX512_MASKS
template<typename T, typename V, typename MaskType, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==9, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const MaskType mask,
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#else
template<typename T, typename V, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==9, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const int (&maska)[V::Size],
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#endif


#ifdef FASTOR_HAS_AVX512_MASKS
        V bmm0;
        bmm0.mask_load(&b[0],mask,false);
#else
        const V bmm0(maskload<V>(&b[0],maska));
#endif

        const V amm0(a[(j  )*K]);
        const V amm1(a[(j+1)*K]);
        const V amm2(a[(j+2)*K]);
        const V amm3(a[(j+3)*K]);
        const V amm4(a[(j+4)*K]);
        const V amm5(a[(j+5)*K]);
        const V amm6(a[(j+6)*K]);
        const V amm7(a[(j+7)*K]);
        const V amm8(a[(j+8)*K]);

        // row 0
        V omm0(amm0*bmm0);
        // row 1
        V omm1(amm1*bmm0);
        // row 2
        V omm2(amm2*bmm0);
        // row 3
        V omm3(amm3*bmm0);
        // row 4
        V omm4(amm4*bmm0);
        // row 5
        V omm5(amm5*bmm0);
        // row 6
        V omm6(amm6*bmm0);
        // row 7
        V omm7(amm7*bmm0);
        // row 8
        V omm8(amm8*bmm0);

        for (size_t i=1; i<K; ++i) {
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm0.mask_load(&b[i*N],mask,false);
#else
            const V bmm0(maskload<V>(&b[i*N],maska));
#endif
            const V amm0(a[(j  )*K+i]);
            const V amm1(a[(j+1)*K+i]);
            const V amm2(a[(j+2)*K+i]);
            const V amm3(a[(j+3)*K+i]);
            const V amm4(a[(j+4)*K+i]);
            const V amm5(a[(j+5)*K+i]);
            const V amm6(a[(j+6)*K+i]);
            const V amm7(a[(j+7)*K+i]);

            const V amm8(a[(j+8)*K+i]);

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            // row 1
            omm1  = fmadd(amm1,bmm0,omm1);
            // row 2
            omm2  = fmadd(amm2,bmm0,omm2);
            // row 3
            omm3  = fmadd(amm3,bmm0,omm3);
            // row 4
            omm4  = fmadd(amm4,bmm0,omm4);
            // row 5
            omm5  = fmadd(amm5,bmm0,omm5);
            // row 6
            omm6  = fmadd(amm6,bmm0,omm6);
            // row 7
            omm7  = fmadd(amm7,bmm0,omm7);
            // row 8
            omm8  = fmadd(amm8,bmm0,omm8);
        }

        omm0.store(&out[(j  )*N],false);
        omm1.store(&out[(j+1)*N],false);
        omm2.store(&out[(j+2)*N],false);
        omm3.store(&out[(j+3)*N],false);
        omm4.store(&out[(j+4)*N],false);
        omm5.store(&out[(j+5)*N],false);
        omm6.store(&out[(j+6)*N],false);
        omm7.store(&out[(j+7)*N],false);
#ifdef FASTOR_HAS_AVX512_MASKS
        omm8.mask_store(&out[(j+8)*N],mask,false);
#else
        maskstore(&out[(j+8)*N],maska,omm8);
#endif
        return;
}


#ifdef FASTOR_HAS_AVX512_MASKS
template<typename T, typename V, typename MaskType, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==8, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const MaskType mask,
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#else
template<typename T, typename V, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==8, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const int (&maska)[V::Size],
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#endif

#ifdef FASTOR_HAS_AVX512_MASKS
        V bmm0;
        bmm0.mask_load(&b[0],mask,false);
#else
        const V bmm0(maskload<V>(&b[0],maska));
#endif

        const V amm0(a[(j  )*K]);
        const V amm1(a[(j+1)*K]);
        const V amm2(a[(j+2)*K]);
        const V amm3(a[(j+3)*K]);
        const V amm4(a[(j+4)*K]);
        const V amm5(a[(j+5)*K]);
        const V amm6(a[(j+6)*K]);
        const V amm7(a[(j+7)*K]);

        // row 0
        V omm0(amm0*bmm0);
        // row 1
        V omm1(amm1*bmm0);
        // row 2
        V omm2(amm2*bmm0);
        // row 3
        V omm3(amm3*bmm0);
        // row 4
        V omm4(amm4*bmm0);
        // row 5
        V omm5(amm5*bmm0);
        // row 6
        V omm6(amm6*bmm0);
        // row 7
        V omm7(amm7*bmm0);

        for (size_t i=1; i<K; ++i) {
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm0.mask_load(&b[i*N],mask,false);
#else
            const V bmm0(maskload<V>(&b[i*N],maska));
#endif
            const V amm0(a[(j  )*K+i]);
            const V amm1(a[(j+1)*K+i]);
            const V amm2(a[(j+2)*K+i]);
            const V amm3(a[(j+3)*K+i]);
            const V amm4(a[(j+4)*K+i]);
            const V amm5(a[(j+5)*K+i]);
            const V amm6(a[(j+6)*K+i]);
            const V amm7(a[(j+7)*K+i]);

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            // row 1
            omm1  = fmadd(amm1,bmm0,omm1);
            // row 2
            omm2  = fmadd(amm2,bmm0,omm2);
            // row 3
            omm3  = fmadd(amm3,bmm0,omm3);
            // row 4
            omm4  = fmadd(amm4,bmm0,omm4);
            // row 5
            omm5  = fmadd(amm5,bmm0,omm5);
            // row 6
            omm6  = fmadd(amm6,bmm0,omm6);
            // row 7
            omm7  = fmadd(amm7,bmm0,omm7);
        }

        omm0.store(&out[(j  )*N],false);
        omm1.store(&out[(j+1)*N],false);
        omm2.store(&out[(j+2)*N],false);
        omm3.store(&out[(j+3)*N],false);
        omm4.store(&out[(j+4)*N],false);
        omm5.store(&out[(j+5)*N],false);
        omm6.store(&out[(j+6)*N],false);
#ifdef FASTOR_HAS_AVX512_MASKS
        omm7.mask_store(&out[(j+7)*N],mask,false);
#else
        maskstore(&out[(j+7)*N],maska,omm7);
#endif
        return;
}


#ifdef FASTOR_HAS_AVX512_MASKS
template<typename T, typename V, typename MaskType, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==7, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const MaskType mask,
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#else
template<typename T, typename V, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==7, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const int (&maska)[V::Size],
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#endif

#ifdef FASTOR_HAS_AVX512_MASKS
        V bmm0;
        bmm0.mask_load(&b[0],mask,false);
#else
        const V bmm0(maskload<V>(&b[0],maska));
#endif

        const V amm0(a[(j  )*K]);
        const V amm1(a[(j+1)*K]);
        const V amm2(a[(j+2)*K]);
        const V amm3(a[(j+3)*K]);
        const V amm4(a[(j+4)*K]);
        const V amm5(a[(j+5)*K]);
        const V amm6(a[(j+6)*K]);

        // row 0
        V omm0(amm0*bmm0);
        // row 1
        V omm1(amm1*bmm0);
        // row 2
        V omm2(amm2*bmm0);
        // row 3
        V omm3(amm3*bmm0);
        // row 4
        V omm4(amm4*bmm0);
        // row 5
        V omm5(amm5*bmm0);
        // row 6
        V omm6(amm6*bmm0);

        for (size_t i=1; i<K; ++i) {
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm0.mask_load(&b[i*N],mask,false);
#else
            const V bmm0(maskload<V>(&b[i*N],maska));
#endif
            const V amm0(a[(j  )*K+i]);
            const V amm1(a[(j+1)*K+i]);
            const V amm2(a[(j+2)*K+i]);
            const V amm3(a[(j+3)*K+i]);
            const V amm4(a[(j+4)*K+i]);
            const V amm5(a[(j+5)*K+i]);
            const V amm6(a[(j+6)*K+i]);

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            // row 1
            omm1  = fmadd(amm1,bmm0,omm1);
            // row 2
            omm2  = fmadd(amm2,bmm0,omm2);
            // row 3
            omm3  = fmadd(amm3,bmm0,omm3);
            // row 4
            omm4  = fmadd(amm4,bmm0,omm4);
            // row 5
            omm5  = fmadd(amm5,bmm0,omm5);
            // row 6
            omm6  = fmadd(amm6,bmm0,omm6);
        }

        omm0.store(&out[(j  )*N],false);
        omm1.store(&out[(j+1)*N],false);
        omm2.store(&out[(j+2)*N],false);
        omm3.store(&out[(j+3)*N],false);
        omm4.store(&out[(j+4)*N],false);
        omm5.store(&out[(j+5)*N],false);
#ifdef FASTOR_HAS_AVX512_MASKS
        omm6.mask_store(&out[(j+6)*N],mask,false);
#else
        maskstore(&out[(j+6)*N],maska,omm6);
#endif
        return;
}


#ifdef FASTOR_HAS_AVX512_MASKS
template<typename T, typename V, typename MaskType, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==6, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const MaskType mask,
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#else
template<typename T, typename V, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==6, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const int (&maska)[V::Size],
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#endif

#ifdef FASTOR_HAS_AVX512_MASKS
        V bmm0;
        bmm0.mask_load(&b[0],mask,false);
#else
        const V bmm0(maskload<V>(&b[0],maska));
#endif

        const V amm0(a[(j  )*K]);
        const V amm1(a[(j+1)*K]);
        const V amm2(a[(j+2)*K]);
        const V amm3(a[(j+3)*K]);
        const V amm4(a[(j+4)*K]);
        const V amm5(a[(j+5)*K]);

        // row 0
        V omm0(amm0*bmm0);
        // row 1
        V omm1(amm1*bmm0);
        // row 2
        V omm2(amm2*bmm0);
        // row 3
        V omm3(amm3*bmm0);
        // row 4
        V omm4(amm4*bmm0);
        // row 5
        V omm5(amm5*bmm0);

        for (size_t i=1; i<K; ++i) {
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm0.mask_load(&b[i*N],mask,false);
#else
            const V bmm0(maskload<V>(&b[i*N],maska));
#endif
            const V amm0(a[(j  )*K+i]);
            const V amm1(a[(j+1)*K+i]);
            const V amm2(a[(j+2)*K+i]);
            const V amm3(a[(j+3)*K+i]);
            const V amm4(a[(j+4)*K+i]);
            const V amm5(a[(j+5)*K+i]);

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            // row 1
            omm1  = fmadd(amm1,bmm0,omm1);
            // row 2
            omm2  = fmadd(amm2,bmm0,omm2);
            // row 3
            omm3  = fmadd(amm3,bmm0,omm3);
            // row 4
            omm4  = fmadd(amm4,bmm0,omm4);
            // row 5
            omm5  = fmadd(amm5,bmm0,omm5);
        }

        omm0.store(&out[(j  )*N],false);
        omm1.store(&out[(j+1)*N],false);
        omm2.store(&out[(j+2)*N],false);
        omm3.store(&out[(j+3)*N],false);
        omm4.store(&out[(j+4)*N],false);
#ifdef FASTOR_HAS_AVX512_MASKS
        omm5.mask_store(&out[(j+5)*N],mask,false);
#else
        maskstore(&out[(j+5)*N],maska,omm5);
#endif
        return;

}


#ifdef FASTOR_HAS_AVX512_MASKS
template<typename T, typename V, typename MaskType, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==5, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const MaskType mask,
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#else
template<typename T, typename V, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==5, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const int (&maska)[V::Size],
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#endif

#ifdef FASTOR_HAS_AVX512_MASKS
        V bmm0;
        bmm0.mask_load(&b[0],mask,false);
#else
        const V bmm0(maskload<V>(&b[0],maska));
#endif

        const V amm0(a[(j  )*K]);
        const V amm1(a[(j+1)*K]);
        const V amm2(a[(j+2)*K]);
        const V amm3(a[(j+3)*K]);
        const V amm4(a[(j+4)*K]);

        // row 0
        V omm0(amm0*bmm0);
        // row 1
        V omm1(amm1*bmm0);
        // row 2
        V omm2(amm2*bmm0);
        // row 3
        V omm3(amm3*bmm0);
        // row 4
        V omm4(amm4*bmm0);

        for (size_t i=1; i<K; ++i) {
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm0.mask_load(&b[i*N],mask,false);
#else
            const V bmm0(maskload<V>(&b[i*N],maska));
#endif
            const V amm0(a[(j  )*K+i]);
            const V amm1(a[(j+1)*K+i]);
            const V amm2(a[(j+2)*K+i]);
            const V amm3(a[(j+3)*K+i]);
            const V amm4(a[(j+4)*K+i]);

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            // row 1
            omm1  = fmadd(amm1,bmm0,omm1);
            // row 2
            omm2  = fmadd(amm2,bmm0,omm2);
            // row 3
            omm3  = fmadd(amm3,bmm0,omm3);
            // row 4
            omm4  = fmadd(amm4,bmm0,omm4);
        }

        omm0.store(&out[(j  )*N],false);
        omm1.store(&out[(j+1)*N],false);
        omm2.store(&out[(j+2)*N],false);
        omm3.store(&out[(j+3)*N],false);
#ifdef FASTOR_HAS_AVX512_MASKS
        omm4.mask_store(&out[(j+4)*N],mask,false);
#else
        maskstore(&out[(j+4)*N],maska,omm4);
#endif
        return;

}


#ifdef FASTOR_HAS_AVX512_MASKS
template<typename T, typename V, typename MaskType, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==4, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const MaskType mask,
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#else
template<typename T, typename V, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==4, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const int (&maska)[V::Size],
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#endif

#ifdef FASTOR_HAS_AVX512_MASKS
        V bmm0;
        bmm0.mask_load(&b[0],mask,false);
#else
        const V bmm0(maskload<V>(&b[0],maska));
#endif

        const V amm0(a[(j  )*K]);
        const V amm1(a[(j+1)*K]);
        const V amm2(a[(j+2)*K]);
        const V amm3(a[(j+3)*K]);

        // row 0
        V omm0(amm0*bmm0);
        // row 1
        V omm1(amm1*bmm0);
        // row 2
        V omm2(amm2*bmm0);
        // row 3
        V omm3(amm3*bmm0);

        for (size_t i=1; i<K; ++i) {
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm0.mask_load(&b[i*N],mask,false);
#else
            const V bmm0(maskload<V>(&b[i*N],maska));
#endif
            const V amm0(a[(j  )*K+i]);
            const V amm1(a[(j+1)*K+i]);
            const V amm2(a[(j+2)*K+i]);
            const V amm3(a[(j+3)*K+i]);

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            // row 1
            omm1  = fmadd(amm1,bmm0,omm1);
            // row 2
            omm2  = fmadd(amm2,bmm0,omm2);
            // row 3
            omm3  = fmadd(amm3,bmm0,omm3);
        }

        omm0.store(&out[(j  )*N],false);
        omm1.store(&out[(j+1)*N],false);
        omm2.store(&out[(j+2)*N],false);
#ifdef FASTOR_HAS_AVX512_MASKS
        omm3.mask_store(&out[(j+3)*N],mask,false);
#else
        maskstore(&out[(j+3)*N],maska,omm3);
#endif
        return;

}


#ifdef FASTOR_HAS_AVX512_MASKS
template<typename T, typename V, typename MaskType, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==3, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const MaskType mask,
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#else
template<typename T, typename V, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==3, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const int (&maska)[V::Size],
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#endif

#ifdef FASTOR_HAS_AVX512_MASKS
        V bmm0;
        bmm0.mask_load(&b[0],mask,false);
#else
        const V bmm0(maskload<V>(&b[0],maska));
#endif

        const V amm0(a[(j  )*K]);
        const V amm1(a[(j+1)*K]);
        const V amm2(a[(j+2)*K]);

        // row 0
        V omm0(amm0*bmm0);
        // row 1
        V omm1(amm1*bmm0);
        // row 2
        V omm2(amm2*bmm0);

        for (size_t i=1; i<K; ++i) {
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm0.mask_load(&b[i*N],mask,false);
#else
            const V bmm0(maskload<V>(&b[i*N],maska));
#endif
            const V amm0(a[(j  )*K+i]);
            const V amm1(a[(j+1)*K+i]);
            const V amm2(a[(j+2)*K+i]);

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            // row 1
            omm1  = fmadd(amm1,bmm0,omm1);
            // row 2
            omm2  = fmadd(amm2,bmm0,omm2);
        }

        omm0.store(&out[(j  )*N],false);
        omm1.store(&out[(j+1)*N],false);
#ifdef FASTOR_HAS_AVX512_MASKS
        omm2.mask_store(&out[(j+2)*N],mask,false);
#else
        maskstore(&out[(j+2)*N],maska,omm2);
#endif
        return;

}


#ifdef FASTOR_HAS_AVX512_MASKS
template<typename T, typename V, typename MaskType, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==2, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const MaskType mask,
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#else
template<typename T, typename V, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==2, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const int (&maska)[V::Size],
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#endif

#ifdef FASTOR_HAS_AVX512_MASKS
        V bmm0;
        bmm0.mask_load(&b[0],mask,false);
#else
        const V bmm0(maskload<V>(&b[0],maska));
#endif

        const V amm0(a[(j  )*K]);
        const V amm1(a[(j+1)*K]);

        // row 0
        V omm0(amm0*bmm0);
        // row 1
        V omm1(amm1*bmm0);

        for (size_t i=1; i<K; ++i) {
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm0.mask_load(&b[i*N],mask,false);
#else
            const V bmm0(maskload<V>(&b[i*N],maska));
#endif
            const V amm0(a[(j  )*K+i]);
            const V amm1(a[(j+1)*K+i]);

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            // row 1
            omm1  = fmadd(amm1,bmm0,omm1);
        }

        omm0.store(&out[(j  )*N],false);
#ifdef FASTOR_HAS_AVX512_MASKS
        omm1.mask_store(&out[(j+1)*N],mask,false);
#else
        maskstore(&out[(j+1)*N],maska,omm1);
#endif
        return;

}


#ifdef FASTOR_HAS_AVX512_MASKS
template<typename T, typename V, typename MaskType, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==1, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const MaskType mask,
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#else
template<typename T, typename V, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==1, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const int (&maska)[V::Size],
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#endif

#ifdef FASTOR_HAS_AVX512_MASKS
        V bmm0;
        bmm0.mask_load(&b[0],mask,false);
#else
        const V bmm0(maskload<V>(&b[0],maska));
#endif

        const V amm0(a[(j  )*K]);

        // row 0
        V omm0(amm0*bmm0);

        for (size_t i=1; i<K; ++i) {
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm0.mask_load(&b[i*N],mask,false);
#else
            const V bmm0(maskload<V>(&b[i*N],maska));
#endif
            const V amm0(a[(j  )*K+i]);

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
        }

#ifdef FASTOR_HAS_AVX512_MASKS
        omm0.mask_store(&out[(j )*N],mask,false);
#else
        maskstore(&out[(j )*N],maska,omm0);
#endif
        return;

}



#ifdef FASTOR_HAS_AVX512_MASKS
template<typename T, typename V, typename MaskType, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==0, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const MaskType mask,
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#else
template<typename T, typename V, size_t K, size_t N, size_t remainder, enable_if_t_<remainder==0, bool> = false>
FASTOR_INLINE void matmul_mk_uptosimd_remainder_kernel(const size_t j, const int (&maska)[V::Size],
    const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b, T* FASTOR_RESTRICT out) {
#endif
    return;
}


template<typename T, size_t M, size_t K, size_t N,
         enable_if_t_<is_less<N, choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size>::value,bool> = 0>
FASTOR_INLINE
void _matmul_mk_smalln(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    using V = typename choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;
    // using V = SIMDVector<T,DEFAULT_ABI>;
    // Unroll a by 10
    constexpr size_t unrollOuterloop = 10UL;
    constexpr size_t M0 = M / unrollOuterloop * unrollOuterloop;

    // Number of columns of c (N) that can be safely unrolled with V::Size
    constexpr size_t N1 = N / V::Size * V::Size;

    int maska[V::Size];
    std::fill(maska,&maska[V::Size], -1);
    for (size_t jj=0; jj < V::Size - (N-N1); ++jj) maska[jj] = 0;
#ifdef FASTOR_HAS_AVX512_MASKS
    const auto mask = array_to_mask(maska);
    V bmm0;
#endif

    size_t j=0;
    for (; j<M0; j+=10UL) {

#ifdef FASTOR_HAS_AVX512_MASKS
        bmm0.mask_load(&b[0],mask,false);
#else
        const V bmm0(maskload<V>(&b[0],maska));
#endif

        const V amm0(a[(j  )*K]);
        const V amm1(a[(j+1)*K]);
        const V amm2(a[(j+2)*K]);
        const V amm3(a[(j+3)*K]);
        const V amm4(a[(j+4)*K]);
        const V amm5(a[(j+5)*K]);
        const V amm6(a[(j+6)*K]);
        const V amm7(a[(j+7)*K]);
        const V amm8(a[(j+8)*K]);
        const V amm9(a[(j+9)*K]);

        // row 0
        V omm0(amm0*bmm0);
        // row 1
        V omm1(amm1*bmm0);
        // row 2
        V omm2(amm2*bmm0);
        // row 3
        V omm3(amm3*bmm0);
        // row 4
        V omm4(amm4*bmm0);
        // row 5
        V omm5(amm5*bmm0);
        // row 6
        V omm6(amm6*bmm0);
        // row 7
        V omm7(amm7*bmm0);
        // row 8
        V omm8(amm8*bmm0);
        // row 9
        V omm9(amm9*bmm0);

        for (size_t i=1; i<K; ++i) {
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm0.mask_load(&b[i*N],mask,false);
#else
            const V bmm0(maskload<V>(&b[i*N],maska));
#endif
            const V amm0(a[(j  )*K+i]);
            const V amm1(a[(j+1)*K+i]);
            const V amm2(a[(j+2)*K+i]);
            const V amm3(a[(j+3)*K+i]);
            const V amm4(a[(j+4)*K+i]);
            const V amm5(a[(j+5)*K+i]);
            const V amm6(a[(j+6)*K+i]);
            const V amm7(a[(j+7)*K+i]);
            const V amm8(a[(j+8)*K+i]);
            const V amm9(a[(j+9)*K+i]);

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            // row 1
            omm1  = fmadd(amm1,bmm0,omm1);
            // row 2
            omm2  = fmadd(amm2,bmm0,omm2);
            // row 3
            omm3  = fmadd(amm3,bmm0,omm3);
            // row 4
            omm4  = fmadd(amm4,bmm0,omm4);
            // row 5
            omm5  = fmadd(amm5,bmm0,omm5);
            // row 6
            omm6  = fmadd(amm6,bmm0,omm6);
            // row 7
            omm7  = fmadd(amm7,bmm0,omm7);
            // row 8
            omm8  = fmadd(amm8,bmm0,omm8);
            // row 9
            omm9  = fmadd(amm9,bmm0,omm9);
        }

        omm0.store(&out[(j  )*N],false);
        omm1.store(&out[(j+1)*N],false);
        omm2.store(&out[(j+2)*N],false);
        omm3.store(&out[(j+3)*N],false);
        omm4.store(&out[(j+4)*N],false);
        omm5.store(&out[(j+5)*N],false);
        omm6.store(&out[(j+6)*N],false);
        omm7.store(&out[(j+7)*N],false);
        omm8.store(&out[(j+8)*N],false);
#ifdef FASTOR_HAS_AVX512_MASKS
        omm9.mask_store(&out[(j+9)*N],mask,false);
#else
        maskstore(&out[(j+9)*N],maska,omm9);
#endif
    }

#ifdef FASTOR_HAS_AVX512_MASKS
    matmul_mk_uptosimd_remainder_kernel<T,V,decltype(mask),K,N,M-M0>(j,mask,a,b,out);
#else
    matmul_mk_uptosimd_remainder_kernel<T,V,K,N,M-M0>(j,maska,a,b,out);
#endif
}
//-----------------------------------------------------------------------------------------------------------




// Take care of N==V::Size
//-----------------------------------------------------------------------------------------------------------
// Compile time loop unroller
template<size_t from, size_t to>
struct matmul_inner_loop_unroller {
    template<size_t M, size_t K, typename T, typename ABI>
    static FASTOR_INLINE void fmadd_(const size_t i, const T * FASTOR_RESTRICT a,
        const SIMDVector<T,ABI> &bmm0, SIMDVector<T,ABI> (&c_ij)[M]) {
        const SIMDVector<T,ABI> amm0(a[from*K+i]);
        c_ij[from] = fmadd(amm0,bmm0,c_ij[from]);
        matmul_inner_loop_unroller<from+1,to>::template fmadd_<M,K,T,ABI>(i, a, bmm0, c_ij);
    }

    template<size_t M, typename T, typename ABI>
    static FASTOR_INLINE void store_(const SIMDVector<T,ABI> (&c_ij)[M], T* FASTOR_RESTRICT out) {
        c_ij[from].store(&out[from*SIMDVector<T,ABI>::Size]);
        matmul_inner_loop_unroller<from+1,to>::template store_<M,T,ABI>(c_ij, out);
    }
};

template<size_t from>
struct matmul_inner_loop_unroller<from,from> {

    template<size_t M, size_t K, typename T, typename ABI>
    static FASTOR_INLINE void fmadd_(const size_t i, const T * FASTOR_RESTRICT a,
        const SIMDVector<T,ABI> &bmm0, SIMDVector<T,ABI> (&c_ij)[M]) {
        const SIMDVector<T,ABI> amm0(a[from*K+i]);
        c_ij[from] = fmadd(amm0,bmm0,c_ij[from]);
    }

    template<size_t M, typename T, typename ABI>
    static FASTOR_INLINE void store_(const SIMDVector<T,ABI> (&c_ij)[M], T* FASTOR_RESTRICT out) {
        c_ij[from].store(&out[from*SIMDVector<T,ABI>::Size],false);
    }
};


// This implementation is based on 2k2/3k3/4k4 but generalised for all Ms that is
// mk2/mk3/mk4/mk8 using compile time template unrolling. It uses an array of Vs
// instead of registers to generalise on M. While the loops are completely unrolled
// at compile time both clang and gcc fetch the first operand of (v)fmadd213ps for the
// first iteration of the loop from the cache/memory instead of operating on registers directly
// the remaining of the unrolled loop is on registers. However this has very little effect on
// performance of small tensors. This method although generalises on M it is designed for M<V::size
// in mind and works best for small Ms as unrolling beyond a certain size certainly hurts the performance
//------------------------------------------------------------------------------------------------//
template<typename T, size_t M, size_t K, size_t N,
        typename std::enable_if<is_less_equal<M,16UL>::value &&
        N==choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size &&
        1!=choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size,bool>::type = 0>
FASTOR_INLINE
void _matmul_mk_smalln(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;
    V c_ij[M];

    for (size_t i=0; i<K; ++i) {
        const V bmm0(&b[i*V::Size],false);
        matmul_inner_loop_unroller<0,M-1>::template fmadd_<M,K,T,typename V::abi_type>(i, a, bmm0, c_ij);
    }
    matmul_inner_loop_unroller<0,M-1>::template store_<M,T,typename V::abi_type>(c_ij, out);
}

#if 0
// This is the non-unrolled version of the above
template<typename T, size_t M, size_t K, size_t N,
        typename std::enable_if<is_less_equal<M,16UL>::value &&
        internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::Size==N,bool>::type = 0>
FASTOR_INLINE
void _matmul_mk_smalln(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;
    V c_ij[V::Size];

    for (size_t j=0; j<M; ++j) {
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*V::Size]);
            const V amm0(a[j*K+i]);
            c_ij[j] = fmadd(amm0,bmm0,c_ij[j]);
        }
        c_ij[j].store(&out[j*V::Size]);
    }
}
#endif


template<typename T, size_t M, size_t K, size_t N,
        typename std::enable_if<is_greater<M,16UL>::value &&
        N==choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size,bool>::type = 0>
FASTOR_INLINE
void _matmul_mk_smalln(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;

    constexpr size_t unrollOuterloop = 5UL;
    constexpr size_t M0 = M / unrollOuterloop * unrollOuterloop;
    // constexpr size_t remainder = M < unrollOuterloop ? 0 : M0-unrollOuterloop;
    constexpr bool isBAligned = false;
    constexpr bool isCAligned = false;

    size_t j=0;
    for (; j<M0; j+=unrollOuterloop) {
        V omm0, omm1, omm2, omm3, omm4;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N],isBAligned);

            const T amm0       = a[j*K+i];
            const T amm1       = a[(j+1)*K+i];
            const T amm2       = a[(j+2)*K+i];
            const T amm3       = a[(j+3)*K+i];
            const T amm4       = a[(j+4)*K+i];

            // row 0
            V a_vec0(amm0);
            omm0  = fmadd(a_vec0,bmm0,omm0);
            // row 1
            V a_vec1(amm1);
            omm1  = fmadd(a_vec1,bmm0,omm1);
            // row 2
            V a_vec2(amm2);
            omm2  = fmadd(a_vec2,bmm0,omm2);
            // row 3
            V a_vec3(amm3);
            omm3  = fmadd(a_vec3,bmm0,omm3);
            // row 4
            V a_vec4(amm4);
            omm4  = fmadd(a_vec4,bmm0,omm4);
        }

        omm0.store(&out[(j+0)*N],isCAligned);
        omm1.store(&out[(j+1)*N],isCAligned);
        omm2.store(&out[(j+2)*N],isCAligned);
        omm3.store(&out[(j+3)*N],isCAligned);
        omm4.store(&out[(j+4)*N],isCAligned);
    }

    // Remainder M-M0 rows
    // Explicitly unroll remaining loops, there is going to be atmost 4
    FASTOR_IF_CONSTEXPR (M-M0==4) {
        V omm0, omm1, omm2, omm3;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N],isBAligned);

            const T amm0       = a[j*K+i];
            const T amm1       = a[(j+1)*K+i];
            const T amm2       = a[(j+2)*K+i];
            const T amm3       = a[(j+3)*K+i];

            // row 0
            V a_vec0(amm0);
            omm0  = fmadd(a_vec0,bmm0,omm0);
            // row 1
            V a_vec1(amm1);
            omm1  = fmadd(a_vec1,bmm0,omm1);
            // row 2
            V a_vec2(amm2);
            omm2  = fmadd(a_vec2,bmm0,omm2);
            // row 3
            V a_vec3(amm3);
            omm3  = fmadd(a_vec3,bmm0,omm3);
        }

        omm0.store(&out[(j+0)*N],isCAligned);
        omm1.store(&out[(j+1)*N],isCAligned);
        omm2.store(&out[(j+2)*N],isCAligned);
        omm3.store(&out[(j+3)*N],isCAligned);
    }

    else FASTOR_IF_CONSTEXPR (M-M0==3) {
        V omm0, omm1, omm2;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N],isBAligned);

            const T amm0       = a[j*K+i];
            const T amm1       = a[(j+1)*K+i];
            const T amm2       = a[(j+2)*K+i];

            // row 0
            V a_vec0(amm0);
            omm0  = fmadd(a_vec0,bmm0,omm0);
            // row 1
            V a_vec1(amm1);
            omm1  = fmadd(a_vec1,bmm0,omm1);
            // row 2
            V a_vec2(amm2);
            omm2  = fmadd(a_vec2,bmm0,omm2);
        }

        omm0.store(&out[(j+0)*N],isCAligned);
        omm1.store(&out[(j+1)*N],isCAligned);
        omm2.store(&out[(j+2)*N],isCAligned);
    }

    else FASTOR_IF_CONSTEXPR (M-M0==2) {
        V omm0, omm1;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N],isBAligned);

            const T amm0       = a[j*K+i];
            const T amm1       = a[(j+1)*K+i];

            // row 0
            V a_vec0(amm0);
            omm0  = fmadd(a_vec0,bmm0,omm0);
            // row 1
            V a_vec1(amm1);
            omm1  = fmadd(a_vec1,bmm0,omm1);
        }

        omm0.store(&out[(j+0)*N],isCAligned);
        omm1.store(&out[(j+1)*N],isCAligned);
    }

    else FASTOR_IF_CONSTEXPR (M-M0==1) {
        V omm0;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N],isBAligned);

            const T amm0       = a[j*K+i];

            // row 0
            V a_vec0(amm0);
            omm0  = fmadd(a_vec0,bmm0,omm0);
        }

        omm0.store(&out[(j+0)*N],isCAligned);
    }
#if 0
    else {
        V c_ij[M-M0];
        for (size_t j=M0; j<M; ++j) {
            for (size_t i=0; i<K; ++i) {
                const V bmm0(&b[i*N]);
                const V amm0(a[j*K+i]);
                c_ij[j] = fmadd(amm0,bmm0,c_ij[j]);
            }
            maskstore(&out[j*N],maska,c_ij[j]);
        }
    }
#endif
}
//-----------------------------------------------------------------------------------------------------------




// Take care of [V::Size < N < 2*V::Size]
// The function implements standard loop unrolling over M. It uses conditional
// loads and store using masks and requires at least AVX. The efficiency of the method comes from
// the fact that it attempts to achieve exact two FMA per load. Both GCC and Clang emit excellent
// code for this at O3
// A recursive implementation of this using compile time unrolling is available at:
// https://gist.github.com/romeric/a176e28127a8348c3c37c5a369051451
//-----------------------------------------------------------------------------------------------------------
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<
            (is_less<N,2*choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size>::value &&
            is_greater<N,choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size>::value),bool>::type = 0>
FASTOR_INLINE
void _matmul_mk_smalln(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;
    // We unroll a by 5 and load 2 simd wide columns of b to get two FMA per load
    // Unrolling by 5 does not hurt as the fall back cases 4,3,2,1 are also explicitly
    // unrolled
    constexpr size_t unrollOuterloop = 5UL;
    constexpr size_t M0 = M / unrollOuterloop * unrollOuterloop;
    //constexpr size_t remainder = M < unrollOuterloop ? 0 : M0-unrollOuterloop;

    // Number of columns of c (N) that can be safely unrolled with V::Size
    constexpr size_t N1 = N / V::Size * V::Size;

    int maska[V::Size];
    std::fill(maska,&maska[V::Size], -1);
    for (size_t jj=0; jj < V::Size - (N-N1); ++jj) maska[jj] = 0;
#ifdef FASTOR_HAS_AVX512_MASKS
    const auto mask = array_to_mask(maska);
    V bmm1;
#endif

    size_t j=0;
    for (; j<M0; j+=unrollOuterloop) {
        const V bmm0(&b[0], false);
#ifdef FASTOR_HAS_AVX512_MASKS
        bmm1.mask_load(&b[V::Size],mask,false);
#else
        const V bmm1(maskload<V>(&b[V::Size],maska));
#endif

        const V amm0(a[(j  )*K]);
        const V amm1(a[(j+1)*K]);
        const V amm2(a[(j+2)*K]);
        const V amm3(a[(j+3)*K]);
        const V amm4(a[(j+4)*K]);

        // row 0
        V omm0(amm0*bmm0);
        V omm1(amm0*bmm1);
        // row 1
        V omm2(amm1*bmm0);
        V omm3(amm1*bmm1);
        // row 2
        V omm4(amm2*bmm0);
        V omm5(amm2*bmm1);
        // row 3
        V omm6(amm3*bmm0);
        V omm7(amm3*bmm1);
        // row 4
        V omm8(amm4*bmm0);
        V omm9(amm4*bmm1);

        for (size_t i=1; i<K; ++i) {
            const V bmm0(&b[i*N], false);
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm1.mask_load(&b[i*N+V::Size],mask,false);
#else
            const V bmm1(maskload<V>(&b[i*N+V::Size],maska));
#endif

            const V amm0(a[(j  )*K+i]);
            const V amm1(a[(j+1)*K+i]);
            const V amm2(a[(j+2)*K+i]);
            const V amm3(a[(j+3)*K+i]);
            const V amm4(a[(j+4)*K+i]);

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            omm1  = fmadd(amm0,bmm1,omm1);
            // row 1
            omm2  = fmadd(amm1,bmm0,omm2);
            omm3  = fmadd(amm1,bmm1,omm3);
            // row 2
            omm4  = fmadd(amm2,bmm0,omm4);
            omm5  = fmadd(amm2,bmm1,omm5);
            // row 3
            omm6  = fmadd(amm3,bmm0,omm6);
            omm7  = fmadd(amm3,bmm1,omm7);
            // row 4
            omm8  = fmadd(amm4,bmm0,omm8);
            omm9  = fmadd(amm4,bmm1,omm9);
        }

        omm0.store(&out[(j  )*N],false);
        omm1.store(&out[(j  )*N+V::Size],false);
        omm2.store(&out[(j+1)*N],false);
        omm3.store(&out[(j+1)*N+V::Size],false);
        omm4.store(&out[(j+2)*N],false);
        omm5.store(&out[(j+2)*N+V::Size],false);
        omm6.store(&out[(j+3)*N],false);
        omm7.store(&out[(j+3)*N+V::Size],false);
        omm8.store(&out[(j+4)*N],false);
#ifdef FASTOR_HAS_AVX512_MASKS
        omm9.mask_store(&out[(j+4)*N+V::Size],mask,false);
#else
        maskstore(&out[(j+4)*N+V::Size],maska,omm9);
#endif
    }

    // Remainder M-M0 rows
    // Explicitly unroll remaining loops, there is going to be atmost 4
    FASTOR_IF_CONSTEXPR (M-M0==4) {
        const V bmm0(&b[0], false);
#ifdef FASTOR_HAS_AVX512_MASKS
        bmm1.mask_load(&b[V::Size],mask,false);
#else
        const V bmm1(maskload<V>(&b[V::Size],maska));
#endif

        const V amm0(a[(j  )*K]);
        const V amm1(a[(j+1)*K]);
        const V amm2(a[(j+2)*K]);
        const V amm3(a[(j+3)*K]);

        // row 0
        V omm0(amm0*bmm0);
        V omm1(amm0*bmm1);
        // row 1
        V omm2(amm1*bmm0);
        V omm3(amm1*bmm1);
        // row 2
        V omm4(amm2*bmm0);
        V omm5(amm2*bmm1);
        // row 3
        V omm6(amm3*bmm0);
        V omm7(amm3*bmm1);

        for (size_t i=1; i<K; ++i) {
            const V bmm0(&b[i*N], false);
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm1.mask_load(&b[i*N+V::Size],mask,false);
#else
            const V bmm1(maskload<V>(&b[i*N+V::Size],maska));
#endif

            const V amm0(a[(j  )*K+i]);
            const V amm1(a[(j+1)*K+i]);
            const V amm2(a[(j+2)*K+i]);
            const V amm3(a[(j+3)*K+i]);

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            omm1  = fmadd(amm0,bmm1,omm1);
            // row 1
            omm2  = fmadd(amm1,bmm0,omm2);
            omm3  = fmadd(amm1,bmm1,omm3);
            // row 2
            omm4  = fmadd(amm2,bmm0,omm4);
            omm5  = fmadd(amm2,bmm1,omm5);
            // row 3
            omm6  = fmadd(amm3,bmm0,omm6);
            omm7  = fmadd(amm3,bmm1,omm7);
        }

        omm0.store(&out[(j  )*N],false);
        omm1.store(&out[(j  )*N+V::Size],false);
        omm2.store(&out[(j+1)*N],false);
        omm3.store(&out[(j+1)*N+V::Size],false);
        omm4.store(&out[(j+2)*N],false);
        omm5.store(&out[(j+2)*N+V::Size],false);
        omm6.store(&out[(j+3)*N],false);
#ifdef FASTOR_HAS_AVX512_MASKS
        omm7.mask_store(&out[(j+3)*N+V::Size],mask,false);
#else
        maskstore(&out[(j+3)*N+V::Size],maska,omm7);
#endif
    }

    else FASTOR_IF_CONSTEXPR (M-M0==3) {
        const V bmm0(&b[0], false);
#ifdef FASTOR_HAS_AVX512_MASKS
        bmm1.mask_load(&b[V::Size],mask,false);
#else
        const V bmm1(maskload<V>(&b[V::Size],maska));
#endif

        const V amm0(a[(j  )*K]);
        const V amm1(a[(j+1)*K]);
        const V amm2(a[(j+2)*K]);

        // row 0
        V omm0(amm0*bmm0);
        V omm1(amm0*bmm1);
        // row 1
        V omm2(amm1*bmm0);
        V omm3(amm1*bmm1);
        // row 2
        V omm4(amm2*bmm0);
        V omm5(amm2*bmm1);

        for (size_t i=1; i<K; ++i) {
            const V bmm0(&b[i*N], false);
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm1.mask_load(&b[i*N+V::Size],mask,false);
#else
            const V bmm1(maskload<V>(&b[i*N+V::Size],maska));
#endif

            const V amm0(a[(j  )*K+i]);
            const V amm1(a[(j+1)*K+i]);
            const V amm2(a[(j+2)*K+i]);

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            omm1  = fmadd(amm0,bmm1,omm1);
            // row 1
            omm2  = fmadd(amm1,bmm0,omm2);
            omm3  = fmadd(amm1,bmm1,omm3);
            // row 2
            omm4  = fmadd(amm2,bmm0,omm4);
            omm5  = fmadd(amm2,bmm1,omm5);
        }

        omm0.store(&out[(j  )*N],false);
        omm1.store(&out[(j  )*N+V::Size],false);
        omm2.store(&out[(j+1)*N],false);
        omm3.store(&out[(j+1)*N+V::Size],false);
        omm4.store(&out[(j+2)*N],false);
#ifdef FASTOR_HAS_AVX512_MASKS
        omm5.mask_store(&out[(j+2)*N+V::Size],mask,false);
#else
        maskstore(&out[(j+2)*N+V::Size],maska,omm5);
#endif
    }

    else FASTOR_IF_CONSTEXPR (M-M0==2) {
        const V bmm0(&b[0], false);
#ifdef FASTOR_HAS_AVX512_MASKS
        bmm1.mask_load(&b[V::Size],mask,false);
#else
        const V bmm1(maskload<V>(&b[V::Size],maska));
#endif

        const V amm0(a[(j  )*K]);
        const V amm1(a[(j+1)*K]);

        // row 0
        V omm0(amm0*bmm0);
        V omm1(amm0*bmm1);
        // row 1
        V omm2(amm1*bmm0);
        V omm3(amm1*bmm1);

        for (size_t i=1; i<K; ++i) {
            const V bmm0(&b[i*N], false);
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm1.mask_load(&b[i*N+V::Size],mask,false);
#else
            const V bmm1(maskload<V>(&b[i*N+V::Size],maska));
#endif

            const V amm0(a[(j  )*K+i]);
            const V amm1(a[(j+1)*K+i]);

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            omm1  = fmadd(amm0,bmm1,omm1);
            // row 1
            omm2  = fmadd(amm1,bmm0,omm2);
            omm3  = fmadd(amm1,bmm1,omm3);
        }

        omm0.store(&out[(j  )*N],false);
        omm1.store(&out[(j  )*N+V::Size],false);
        omm2.store(&out[(j+1)*N],false);
#ifdef FASTOR_HAS_AVX512_MASKS
        omm3.mask_store(&out[(j+1)*N+V::Size],mask,false);
#else
        maskstore(&out[(j+1)*N+V::Size],maska,omm3);
#endif
    }

    else FASTOR_IF_CONSTEXPR (M-M0==1) {
        const V bmm0(&b[0], false);
#ifdef FASTOR_HAS_AVX512_MASKS
        bmm1.mask_load(&b[V::Size],mask,false);
#else
        const V bmm1(maskload<V>(&b[V::Size],maska));
#endif

        const V amm0(a[(j  )*K]);

        // row 0
        V omm0(amm0*bmm0);
        V omm1(amm0*bmm1);

        for (size_t i=1; i<K; ++i) {
            const V bmm0(&b[i*N], false);
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm1.mask_load(&b[i*N+V::Size],mask,false);
#else
            const V bmm1(maskload<V>(&b[i*N+V::Size],maska));
#endif

            const V amm0(a[(j  )*K+i]);

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            omm1  = fmadd(amm0,bmm1,omm1);
        }

        omm0.store(&out[(j  )*N],false);
#ifdef FASTOR_HAS_AVX512_MASKS
        omm1.mask_store(&out[(j)*N+V::Size],mask,false);
#else
        maskstore(&out[(j)*N+V::Size],maska,omm1);
#endif
    }
}
//-----------------------------------------------------------------------------------------------------------



// Take care of 2*V::Size cases
//-----------------------------------------------------------------------------------------------------------
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<N==2*internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size,bool>::type = 0>
FASTOR_INLINE
void _matmul_mk_smalln(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;
    constexpr size_t unrollOuterloop = 5UL;
    constexpr size_t M0 = M / unrollOuterloop * unrollOuterloop;
    // constexpr size_t remainder = M < unrollOuterloop ? 0 : M0-unrollOuterloop;

    size_t j=0;
    for (; j<M0; j+=unrollOuterloop) {
        const size_t from = j;
        V omm0, omm1, omm2, omm3, omm4, omm5, omm6, omm7;
        V omm8, omm9;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N], false);
            const V bmm1((&b[i*N+V::Size]),false);

            const T amm0       = a[from*K+i];
            const T amm1       = a[(from+1)*K+i];
            const T amm2       = a[(from+2)*K+i];
            const T amm3       = a[(from+3)*K+i];
            const T amm4       = a[(from+4)*K+i];

            // row 0
            V a_vec0(amm0);
            omm0  = fmadd(a_vec0,bmm0,omm0);
            omm1  = fmadd(a_vec0,bmm1,omm1);
            // row 1
            V a_vec1(amm1);
            omm2  = fmadd(a_vec1,bmm0,omm2);
            omm3  = fmadd(a_vec1,bmm1,omm3);
            // row 2
            V a_vec2(amm2);
            omm4  = fmadd(a_vec2,bmm0,omm4);
            omm5  = fmadd(a_vec2,bmm1,omm5);
            // row 3
            V a_vec3(amm3);
            omm6  = fmadd(a_vec3,bmm0,omm6);
            omm7  = fmadd(a_vec3,bmm1,omm7);
            // row 4
            V a_vec4(amm4);
            omm8  = fmadd(a_vec4,bmm0,omm8);
            omm9  = fmadd(a_vec4,bmm1,omm9);
        }

        omm0.store(&out[from*N],false);
        omm1.store(&out[from*N+V::Size],false);
        omm2.store(&out[(from+1)*N],false);
        omm3.store(&out[(from+1)*N+V::Size],false);
        omm4.store(&out[(from+2)*N],false);
        omm5.store(&out[(from+2)*N+V::Size],false);
        omm6.store(&out[(from+3)*N],false);
        omm7.store(&out[(from+3)*N+V::Size],false);
        omm8.store(&out[(from+4)*N],false);
        omm9.store(&out[(from+4)*N+V::Size],false);
    }

    // Remainder M-M0 rows
    // Explicitly unroll remaining loops, there is going to be atmost 4
    FASTOR_IF_CONSTEXPR(M-M0 == 4) {
        const size_t from = j;
        V omm0, omm1, omm2, omm3, omm4, omm5, omm6, omm7;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N], false);
            const V bmm1((&b[i*N+V::Size]),false);

            const T amm0       = a[from*K+i];
            const T amm1       = a[(from+1)*K+i];
            const T amm2       = a[(from+2)*K+i];
            const T amm3       = a[(from+3)*K+i];

            // row 0
            V a_vec0(amm0);
            omm0  = fmadd(a_vec0,bmm0,omm0);
            omm1  = fmadd(a_vec0,bmm1,omm1);
            // row 1
            V a_vec1(amm1);
            omm2  = fmadd(a_vec1,bmm0,omm2);
            omm3  = fmadd(a_vec1,bmm1,omm3);
            // row 2
            V a_vec2(amm2);
            omm4  = fmadd(a_vec2,bmm0,omm4);
            omm5  = fmadd(a_vec2,bmm1,omm5);
            // row 3
            V a_vec3(amm3);
            omm6  = fmadd(a_vec3,bmm0,omm6);
            omm7  = fmadd(a_vec3,bmm1,omm7);
        }

        omm0.store(&out[from*N],false);
        omm1.store(&out[from*N+V::Size],false);
        omm2.store(&out[(from+1)*N],false);
        omm3.store(&out[(from+1)*N+V::Size],false);
        omm4.store(&out[(from+2)*N],false);
        omm5.store(&out[(from+2)*N+V::Size],false);
        omm6.store(&out[(from+3)*N],false);
        omm7.store(&out[(from+3)*N+V::Size],false);
    }

    else FASTOR_IF_CONSTEXPR (M-M0==3) {
        V omm0, omm1, omm2, omm3, omm4, omm5;
        constexpr size_t from = M0;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N], false);
            const V bmm1((&b[i*N+V::Size]),false);

            const T amm0       = a[from*K+i];
            const T amm1       = a[(from+1)*K+i];
            const T amm2       = a[(from+2)*K+i];

            // row 0
            V a_vec0(amm0);
            omm0  = fmadd(a_vec0,bmm0,omm0);
            omm1  = fmadd(a_vec0,bmm1,omm1);
            // row 1
            V a_vec1(amm1);
            omm2  = fmadd(a_vec1,bmm0,omm2);
            omm3  = fmadd(a_vec1,bmm1,omm3);
            // row 2
            V a_vec2(amm2);
            omm4  = fmadd(a_vec2,bmm0,omm4);
            omm5  = fmadd(a_vec2,bmm1,omm5);
        }

        omm0.store(&out[from*N],false);
        omm1.store(&out[from*N+V::Size],false);
        omm2.store(&out[(from+1)*N],false);
        omm3.store(&out[(from+1)*N+V::Size],false);
        omm4.store(&out[(from+2)*N],false);
        omm5.store(&out[(from+2)*N+V::Size],false);
    }

    else FASTOR_IF_CONSTEXPR (M-M0==2) {
        V omm0, omm1, omm2, omm3;
        constexpr size_t from = M0;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N], false);
            const V bmm1((&b[i*N+V::Size]),false);

            const T amm0       = a[from*K+i];
            const T amm1       = a[(from+1)*K+i];

            // row 0
            V a_vec0(amm0);
            omm0  = fmadd(a_vec0,bmm0,omm0);
            omm1  = fmadd(a_vec0,bmm1,omm1);
            // row 1
            V a_vec1(amm1);
            omm2  = fmadd(a_vec1,bmm0,omm2);
            omm3  = fmadd(a_vec1,bmm1,omm3);
        }

        omm0.store(&out[from*N],false);
        omm1.store(&out[from*N+V::Size],false);
        omm2.store(&out[(from+1)*N],false);
        omm3.store(&out[(from+1)*N+V::Size],false);
    }

    else FASTOR_IF_CONSTEXPR (M-M0==1) {
        V omm0, omm1;
        constexpr size_t from = M0;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N], false);
            const V bmm1((&b[i*N+V::Size]),false);

            const T amm0       = a[from*K+i];

            // row 0
            V a_vec0(amm0);
            omm0  = fmadd(a_vec0,bmm0,omm0);
            omm1  = fmadd(a_vec0,bmm1,omm1);
        }

        omm0.store(&out[from*N],false);
        omm1.store(&out[from*N+V::Size],false);
    }
#if 0
    else {
        V c_ij[M-M0][2];
        for (size_t j=M0; j<M; ++j) {
            for (size_t i=0; i<K; ++i) {
                const V bmm0(&b[i*N], false);
                const V bmm1((&b[i*N+V::Size]),false);
                const V amm0(a[j*K+i]);
                c_ij[j][0] = fmadd(amm0,bmm0,c_ij[j][0]);
                c_ij[j][1] = fmadd(amm0,bmm1,c_ij[j][1]);
            }
            c_ij[j][0].store(&out[j*N],false);
            c_ij[j][1].store(&out[j*N+V::Size],false);
        }
    }
#endif
}
//-----------------------------------------------------------------------------------------------------------




// Take care of [2*V::Size < N < 3*V::Size] cases
//-----------------------------------------------------------------------------------------------------------
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<
            (is_greater<N,2*choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size>::value &&
            is_less<N,3*choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size>::value),bool>::type = 0>
FASTOR_INLINE
void _matmul_mk_smalln(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {


    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;
    constexpr size_t unrollOuterloop = 2UL;
    constexpr size_t M0 = M / unrollOuterloop * unrollOuterloop;
    // constexpr size_t remainder = M < unrollOuterloop ? 0 : M0-unrollOuterloop;
    constexpr bool isBAligned = false;
    constexpr bool isCAligned = false;

    // Number of columns of c (N) that can be safely unrolled with V::Size
    constexpr size_t N1 = N / V::Size * V::Size;

    int maska[V::Size];
    std::fill(maska,&maska[V::Size], -1);
    for (size_t jj=0; jj < V::Size - (N-N1); ++jj) maska[jj] = 0;
#ifdef FASTOR_HAS_AVX512_MASKS
    const auto mask = array_to_mask(maska);
    V bmm2;
#endif

    size_t j=0;
    for (; j<M0; j+=unrollOuterloop) {

        const V amm0(a[j*K]);
        const V amm1(a[(j+1)*K]);

        const V bmm0(&b[0], isBAligned);
        const V bmm1((&b[V::Size]),isBAligned);
#ifdef FASTOR_HAS_AVX512_MASKS
        bmm2.mask_load(&b[2*V::Size],mask,false);
#else
        const V bmm2(maskload<V>(&b[2*V::Size],maska));
#endif

        // row 0
        V omm0(amm0*bmm0);
        V omm1(amm0*bmm1);
        V omm2(amm0*bmm2);
        // row 1
        V omm3(amm1*bmm0);
        V omm4(amm1*bmm1);
        V omm5(amm1*bmm2);

        for (size_t i=1; i<K; ++i) {
            const V bmm0(&b[i*N], isBAligned);
            const V bmm1((&b[i*N+V::Size]),isBAligned);
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm2.mask_load(&b[i*N+2*V::Size],mask,false);
#else
            const V bmm2(maskload<V>(&b[i*N+2*V::Size],maska));
#endif

            const V amm0(a[j*K+i]);
            const V amm1(a[(j+1)*K+i]);

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            omm1  = fmadd(amm0,bmm1,omm1);
            omm2  = fmadd(amm0,bmm2,omm2);
            // row 1
            omm3  = fmadd(amm1,bmm0,omm3);
            omm4  = fmadd(amm1,bmm1,omm4);
            omm5  = fmadd(amm1,bmm2,omm5);
        }

        omm0.store(&out[j*N],isCAligned);
        omm1.store(&out[j*N+V::Size],isCAligned);
        omm2.store(&out[j*N+2*V::Size],isCAligned);

        omm3.store(&out[(j+1)*N],isCAligned);
        omm4.store(&out[(j+1)*N+V::Size],isCAligned);
#ifdef FASTOR_HAS_AVX512_MASKS
        omm5.mask_store(&out[(j+1)*N+2*V::Size],mask,false);
#else
        maskstore(&out[(j+1)*N+2*V::Size],maska,omm5);
#endif
    }

    FASTOR_IF_CONSTEXPR (M-M0==1) {

        const V amm0(a[j*K]);

        const V bmm0(&b[0], isBAligned);
        const V bmm1((&b[V::Size]),isBAligned);
#ifdef FASTOR_HAS_AVX512_MASKS
        bmm2.mask_load(&b[2*V::Size],mask,false);
#else
        const V bmm2(maskload<V>(&b[2*V::Size],maska));
#endif

        // row 0
        V omm0(amm0*bmm0);
        V omm1(amm0*bmm1);
        V omm2(amm0*bmm2);

        for (size_t i=1; i<K; ++i) {
            const V bmm0(&b[i*N], isBAligned);
            const V bmm1((&b[i*N+V::Size]),isBAligned);
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm2.mask_load(&b[i*N+2*V::Size],mask,false);
#else
            const V bmm2(maskload<V>(&b[i*N+2*V::Size],maska));
#endif

            const V amm0(a[j*K+i]);
            const V amm1(a[(j+1)*K+i]);

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            omm1  = fmadd(amm0,bmm1,omm1);
            omm2  = fmadd(amm0,bmm2,omm2);
        }

        omm0.store(&out[j*N],isCAligned);
        omm1.store(&out[j*N+V::Size],isCAligned);

#ifdef FASTOR_HAS_AVX512_MASKS
        omm2.mask_store(&out[j*N+2*V::Size],mask,false);
#else
        maskstore(&out[j*N+2*V::Size],maska,omm2);
#endif
    }
}
//-----------------------------------------------------------------------------------------------------------



// Take care of 3*V::Size cases
// Note that you get the exact same performance by tuning _matmul_base parameters as
// [unrollOuterloop = 2, nSIMDRows=1 and nSIMDCols=3] however these parameters affect
// the universal behaviour of that method specially for big matrices
// This method unrolls M by 2 only as when N==3*V::Size unrolling the entire column of c
// and doing 2 rows at a time is the most benificial. Unrolling M any further hurts the
// performance really bad
//-----------------------------------------------------------------------------------------------------------
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<N==3*choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size,bool>::type = 0>
FASTOR_INLINE
void _matmul_mk_smalln(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;
    constexpr size_t unrollOuterloop = 2UL;
    constexpr size_t M0 = M / unrollOuterloop * unrollOuterloop;
    // constexpr size_t remainder = M < unrollOuterloop ? 0 : M0-unrollOuterloop;
    constexpr bool isBAligned = false;
    constexpr bool isCAligned = false;

    size_t j=0;
    for (; j<M0; j+=unrollOuterloop) {
        const size_t from = j;
        V omm0, omm1, omm2, omm3, omm4, omm5, omm6;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N], isBAligned);
            const V bmm1((&b[i*N+V::Size]),isBAligned);
            const V bmm2((&b[i*N+2*V::Size]),isBAligned);

            const T amm0       = a[from*K+i];
            const T amm1       = a[(from+1)*K+i];

            // row 0
            V a_vec0(amm0);
            omm0  = fmadd(a_vec0,bmm0,omm0);
            omm1  = fmadd(a_vec0,bmm1,omm1);
            omm2  = fmadd(a_vec0,bmm2,omm2);
            // row 1
            V a_vec1(amm1);
            omm4  = fmadd(a_vec1,bmm0,omm4);
            omm5  = fmadd(a_vec1,bmm1,omm5);
            omm6  = fmadd(a_vec1,bmm2,omm6);
        }

        omm0.store(&out[from*N],isCAligned);
        omm1.store(&out[from*N+V::Size],isCAligned);
        omm2.store(&out[from*N+2*V::Size],isCAligned);

        omm4.store(&out[(from+1)*N],isCAligned);
        omm5.store(&out[(from+1)*N+V::Size],isCAligned);
        omm6.store(&out[(from+1)*N+2*V::Size],isCAligned);
    }

    FASTOR_IF_CONSTEXPR (M-M0==1) {
        const size_t from = j;
        V omm0, omm1, omm2;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N], isBAligned);
            const V bmm1((&b[i*N+V::Size]),isBAligned);
            const V bmm2((&b[i*N+2*V::Size]),isBAligned);

            const T amm0       = a[from*K+i];

            // row 0
            V a_vec0(amm0);
            omm0  = fmadd(a_vec0,bmm0,omm0);
            omm1  = fmadd(a_vec0,bmm1,omm1);
            omm2  = fmadd(a_vec0,bmm2,omm2);
        }

        omm0.store(&out[from*N],isCAligned);
        omm1.store(&out[from*N+V::Size],isCAligned);
        omm2.store(&out[from*N+2*V::Size],isCAligned);
    }
}
//-----------------------------------------------------------------------------------------------------------




// Take care of [3*V::Size < N < 4*V::Size] cases
//-----------------------------------------------------------------------------------------------------------

template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<
            (is_greater<N,3*choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size>::value &&
            is_less<N,4*choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size>::value),bool>::type = 0>
FASTOR_INLINE
void _matmul_mk_smalln(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;
    constexpr size_t unrollOuterloop = 2UL;
    constexpr size_t M0 = M / unrollOuterloop * unrollOuterloop;
    // constexpr size_t remainder = M < unrollOuterloop ? 0 : M0-unrollOuterloop;
    constexpr bool isBAligned = false;
    constexpr bool isCAligned = false;

    // Number of columns of c (N) that can be safely unrolled with V::Size
    constexpr size_t N1 = N / V::Size * V::Size;

    int maska[V::Size];
    std::fill(maska,&maska[V::Size], -1);
    for (size_t jj=0; jj < V::Size - (N-N1); ++jj) maska[jj] = 0;
#ifdef FASTOR_HAS_AVX512_MASKS
    const auto mask = array_to_mask(maska);
    V bmm3;
#endif

    size_t j=0;
    for (; j<M0; j+=unrollOuterloop) {
        const size_t from = j;
        V omm0, omm1, omm2, omm3, omm4, omm5, omm6, omm7;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N], isBAligned);
            const V bmm1((&b[i*N+V::Size]),isBAligned);
            const V bmm2((&b[i*N+2*V::Size]),isBAligned);
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm3.mask_load(&b[i*N+3*V::Size],mask,false);
#else
            const V bmm3(maskload<V>(&b[i*N+3*V::Size],maska));
#endif

            const V amm0       = a[from*K+i];
            const V amm1       = a[(from+1)*K+i];

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            omm1  = fmadd(amm0,bmm1,omm1);
            omm2  = fmadd(amm0,bmm2,omm2);
            omm3  = fmadd(amm0,bmm3,omm3);
            // row 1
            omm4  = fmadd(amm1,bmm0,omm4);
            omm5  = fmadd(amm1,bmm1,omm5);
            omm6  = fmadd(amm1,bmm2,omm6);
            omm7  = fmadd(amm1,bmm3,omm7);
        }

        omm0.store(&out[from*N],isCAligned);
        omm1.store(&out[from*N+V::Size],isCAligned);
        omm2.store(&out[from*N+2*V::Size],isCAligned);
        omm3.store(&out[from*N+3*V::Size],isCAligned);

        omm4.store(&out[(from+1)*N],isCAligned);
        omm5.store(&out[(from+1)*N+V::Size],isCAligned);
        omm6.store(&out[(from+1)*N+2*V::Size],isCAligned);
#ifdef FASTOR_HAS_AVX512_MASKS
        omm7.mask_store(&out[(from+1)*N+3*V::Size],mask,false);
#else
        maskstore(&out[(from+1)*N+3*V::Size],maska,omm7);
#endif
    }

    FASTOR_IF_CONSTEXPR (M-M0==1) {
        const size_t from = j;
        V omm0, omm1, omm2, omm3, omm4, omm5, omm6, omm7;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N], isBAligned);
            const V bmm1((&b[i*N+V::Size]),isBAligned);
            const V bmm2((&b[i*N+2*V::Size]),isBAligned);
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm3.mask_load(&b[i*N+3*V::Size],mask,false);
#else
            const V bmm3(maskload<V>(&b[i*N+3*V::Size],maska));
#endif

            const V amm0       = a[from*K+i];

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            omm1  = fmadd(amm0,bmm1,omm1);
            omm2  = fmadd(amm0,bmm2,omm2);
            omm3  = fmadd(amm0,bmm3,omm3);
        }

        omm0.store(&out[from*N],isCAligned);
        omm1.store(&out[from*N+V::Size],isCAligned);
        omm2.store(&out[from*N+2*V::Size],isCAligned);
#ifdef FASTOR_HAS_AVX512_MASKS
        omm3.mask_store(&out[from*N+3*V::Size],mask,false);
#else
        maskstore(&out[from*N+3*V::Size],maska,omm3);
#endif
    }
}
//-----------------------------------------------------------------------------------------------------------




// Take care of 4*V::Size cases
// Note that you get the exact same performance by tuning _matmul_base parameters as
// [unrollOuterloop = 2, nSIMDRows=1 and nSIMDCols=4] however these parameters affect
// the universal behaviour of that method specially for big matrices
// This method unrolls M by 2 only as when N==4*V::Size unrolling the entire column of c
// and doing 2 rows at a time is the most benificial. Unrolling M any further hurts the
// performance really bad
//-----------------------------------------------------------------------------------------------------------
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<N==4*internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size,bool>::type = 0>
FASTOR_INLINE
void _matmul_mk_smalln(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {


    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;
    constexpr size_t unrollOuterloop = 2UL;
    constexpr size_t M0 = M / unrollOuterloop * unrollOuterloop;
    // constexpr size_t remainder = M < unrollOuterloop ? 0 : M0-unrollOuterloop;
    constexpr bool isBAligned = false;
    constexpr bool isCAligned = false;

    size_t j=0;
    for (; j<M0; j+=unrollOuterloop) {

        const V amm0(a[j*K]);
        const V amm1(a[(j+1)*K]);

        const V bmm0(&b[0], isBAligned);
        const V bmm1((&b[V::Size]),isBAligned);
        const V bmm2((&b[2*V::Size]),isBAligned);
        const V bmm3((&b[3*V::Size]),isBAligned);

        // row 0
        V omm0(amm0*bmm0);
        V omm1(amm0*bmm1);
        V omm2(amm0*bmm2);
        V omm3(amm0*bmm3);
        // row 1
        V omm5(amm1*bmm0);
        V omm6(amm1*bmm1);
        V omm7(amm1*bmm2);
        V omm8(amm1*bmm3);


        for (size_t i=1; i<K; ++i) {
            const V bmm0(&b[i*N], isBAligned);
            const V bmm1((&b[i*N+V::Size]),isBAligned);
            const V bmm2((&b[i*N+2*V::Size]),isBAligned);
            const V bmm3((&b[i*N+3*V::Size]),isBAligned);

            const V amm0       = a[j*K+i];
            const V amm1       = a[(j+1)*K+i];

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            omm1  = fmadd(amm0,bmm1,omm1);
            omm2  = fmadd(amm0,bmm2,omm2);
            omm3  = fmadd(amm0,bmm3,omm3);
            // row 1
            omm5  = fmadd(amm1,bmm0,omm5);
            omm6  = fmadd(amm1,bmm1,omm6);
            omm7  = fmadd(amm1,bmm2,omm7);
            omm8  = fmadd(amm1,bmm3,omm8);
        }

        omm0.store(&out[j*N],isCAligned);
        omm1.store(&out[j*N+V::Size],isCAligned);
        omm2.store(&out[j*N+2*V::Size],isCAligned);
        omm3.store(&out[j*N+3*V::Size],isCAligned);

        omm5.store(&out[(j+1)*N],isCAligned);
        omm6.store(&out[(j+1)*N+V::Size],isCAligned);
        omm7.store(&out[(j+1)*N+2*V::Size],isCAligned);
        omm8.store(&out[(j+1)*N+3*V::Size],isCAligned);
    }

    FASTOR_IF_CONSTEXPR (M-M0==1) {
        const V amm0(a[j*K]);

        const V bmm0(&b[0], isBAligned);
        const V bmm1((&b[V::Size]),isBAligned);
        const V bmm2((&b[2*V::Size]),isBAligned);
        const V bmm3((&b[3*V::Size]),isBAligned);

        // row 0
        V omm0(amm0*bmm0);
        V omm1(amm0*bmm1);
        V omm2(amm0*bmm2);
        V omm3(amm0*bmm3);

        for (size_t i=1; i<K; ++i) {
            const V bmm0(&b[i*N], isBAligned);
            const V bmm1((&b[i*N+V::Size]),isBAligned);
            const V bmm2((&b[i*N+2*V::Size]),isBAligned);
            const V bmm3((&b[i*N+3*V::Size]),isBAligned);

            const V amm0       = a[j*K+i];

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            omm1  = fmadd(amm0,bmm1,omm1);
            omm2  = fmadd(amm0,bmm2,omm2);
            omm3  = fmadd(amm0,bmm3,omm3);
        }

        omm0.store(&out[j*N],isCAligned);
        omm1.store(&out[j*N+V::Size],isCAligned);
        omm2.store(&out[j*N+2*V::Size],isCAligned);
        omm3.store(&out[j*N+3*V::Size],isCAligned);
    }
}


// Take care of [4*V::Size < N < 5*V::Size] cases
//-----------------------------------------------------------------------------------------------------------
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<
            (is_greater<N,4*choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size>::value &&
            is_less<N,5*choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size>::value),bool>::type = 0>
FASTOR_INLINE
void _matmul_mk_smalln(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {


    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;
    constexpr size_t unrollOuterloop = 2UL;
    constexpr size_t M0 = M / unrollOuterloop * unrollOuterloop;
    // constexpr size_t remainder = M < unrollOuterloop ? 0 : M0-unrollOuterloop;
    constexpr bool isBAligned = false;
    constexpr bool isCAligned = false;

    // Number of columns of c (N) that can be safely unrolled with V::Size
    constexpr size_t N1 = N / V::Size * V::Size;

    int maska[V::Size];
    std::fill(maska,&maska[V::Size], -1);
    for (size_t jj=0; jj < V::Size - (N-N1); ++jj) maska[jj] = 0;
#ifdef FASTOR_HAS_AVX512_MASKS
    const auto mask = array_to_mask(maska);
    V bmm4;
#endif

    size_t j=0;
    for (; j<M0; j+=unrollOuterloop) {

        V amm0(a[j*K]);
        V amm1(a[(j+1)*K]);

        const V bmm0(&b[0], isBAligned);
        const V bmm1((&b[V::Size]),isBAligned);
        const V bmm2((&b[2*V::Size]),isBAligned);
        const V bmm3((&b[3*V::Size]),isBAligned);
#ifdef FASTOR_HAS_AVX512_MASKS
        bmm4.mask_load(&b[4*V::Size],mask,false);
#else
        const V bmm4(maskload<V>(&b[4*V::Size],maska));
#endif

        // row 0
        V omm0(amm0*bmm0);
        V omm1(amm0*bmm1);
        V omm2(amm0*bmm2);
        V omm3(amm0*bmm3);
        V omm4(amm0*bmm4);
        // row 1
        V omm5(amm1*bmm0);
        V omm6(amm1*bmm1);
        V omm7(amm1*bmm2);
        V omm8(amm1*bmm3);
        V omm9(amm1*bmm4);


        for (size_t i=1; i<K; ++i) {
            const V bmm0(&b[i*N], isBAligned);
            const V bmm1((&b[i*N+V::Size]),isBAligned);
            const V bmm2((&b[i*N+2*V::Size]),isBAligned);
            const V bmm3((&b[i*N+3*V::Size]),isBAligned);
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm4.mask_load(&b[i*N+4*V::Size],mask,false);
#else
            const V bmm4(maskload<V>(&b[i*N+4*V::Size],maska));
#endif

            const V amm0       = a[j*K+i];
            const V amm1       = a[(j+1)*K+i];

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            omm1  = fmadd(amm0,bmm1,omm1);
            omm2  = fmadd(amm0,bmm2,omm2);
            omm3  = fmadd(amm0,bmm3,omm3);
            omm4  = fmadd(amm0,bmm4,omm4);
            // row 1
            omm5  = fmadd(amm1,bmm0,omm5);
            omm6  = fmadd(amm1,bmm1,omm6);
            omm7  = fmadd(amm1,bmm2,omm7);
            omm8  = fmadd(amm1,bmm3,omm8);
            omm9  = fmadd(amm1,bmm4,omm9);
        }

        omm0.store(&out[j*N],isCAligned);
        omm1.store(&out[j*N+V::Size],isCAligned);
        omm2.store(&out[j*N+2*V::Size],isCAligned);
        omm3.store(&out[j*N+3*V::Size],isCAligned);
        omm4.store(&out[j*N+4*V::Size],isCAligned);

        omm5.store(&out[(j+1)*N],isCAligned);
        omm6.store(&out[(j+1)*N+V::Size],isCAligned);
        omm7.store(&out[(j+1)*N+2*V::Size],isCAligned);
        omm8.store(&out[(j+1)*N+3*V::Size],isCAligned);
#ifdef FASTOR_HAS_AVX512_MASKS
        omm9.mask_store(&out[(j+1)*N+4*V::Size],mask,false);
#else
        maskstore(&out[(j+1)*N+4*V::Size],maska,omm9);
#endif
    }

    FASTOR_IF_CONSTEXPR (M-M0==1) {
        V amm0(a[j*K]);

        const V bmm0(&b[0], isBAligned);
        const V bmm1((&b[V::Size]),isBAligned);
        const V bmm2((&b[2*V::Size]),isBAligned);
        const V bmm3((&b[3*V::Size]),isBAligned);
#ifdef FASTOR_HAS_AVX512_MASKS
        bmm4.mask_load(&b[4*V::Size],mask,false);
#else
        const V bmm4(maskload<V>(&b[4*V::Size],maska));
#endif

        // row 0
        V omm0(amm0*bmm0);
        V omm1(amm0*bmm1);
        V omm2(amm0*bmm2);
        V omm3(amm0*bmm3);
        V omm4(amm0*bmm4);

        for (size_t i=1; i<K; ++i) {
            const V bmm0(&b[i*N], isBAligned);
            const V bmm1((&b[i*N+V::Size]),isBAligned);
            const V bmm2((&b[i*N+2*V::Size]),isBAligned);
            const V bmm3((&b[i*N+3*V::Size]),isBAligned);
#ifdef FASTOR_HAS_AVX512_MASKS
            bmm4.mask_load(&b[i*N+4*V::Size],mask,false);
#else
            const V bmm4(maskload<V>(&b[i*N+4*V::Size],maska));
#endif

            const V amm0       = a[j*K+i];

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            omm1  = fmadd(amm0,bmm1,omm1);
            omm2  = fmadd(amm0,bmm2,omm2);
            omm3  = fmadd(amm0,bmm3,omm3);
            omm4  = fmadd(amm0,bmm4,omm4);
        }

        omm0.store(&out[j*N],isCAligned);
        omm1.store(&out[j*N+V::Size],isCAligned);
        omm2.store(&out[j*N+2*V::Size],isCAligned);
        omm3.store(&out[j*N+3*V::Size],isCAligned);
#ifdef FASTOR_HAS_AVX512_MASKS
        omm4.mask_store(&out[j*N+4*V::Size],mask,false);
#else
        maskstore(&out[j*N+4*V::Size],maska,omm4);
#endif
    }
}
//-----------------------------------------------------------------------------------------------------------



// N==5*V::Size case
//-----------------------------------------------------------------------------------------------------------
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<N==5*internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size,bool>::type = 0>
FASTOR_INLINE
void _matmul_mk_smalln(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {


    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;
    constexpr size_t unrollOuterloop = 2UL;
    constexpr size_t M0 = M / unrollOuterloop * unrollOuterloop;
    // constexpr size_t remainder = M < unrollOuterloop ? 0 : M0-unrollOuterloop;
    constexpr bool isBAligned = false;
    constexpr bool isCAligned = false;

    size_t j=0;
    for (; j<M0; j+=unrollOuterloop) {

        V amm0(a[j*K]);
        V amm1(a[(j+1)*K]);

        const V bmm0(&b[0], isBAligned);
        const V bmm1((&b[V::Size]),isBAligned);
        const V bmm2((&b[2*V::Size]),isBAligned);
        const V bmm3((&b[3*V::Size]),isBAligned);
        const V bmm4((&b[4*V::Size]),isBAligned);

        // row 0
        V omm0(amm0*bmm0);
        V omm1(amm0*bmm1);
        V omm2(amm0*bmm2);
        V omm3(amm0*bmm3);
        V omm4(amm0*bmm4);
        // row 1
        V omm5(amm1*bmm0);
        V omm6(amm1*bmm1);
        V omm7(amm1*bmm2);
        V omm8(amm1*bmm3);
        V omm9(amm1*bmm4);


        for (size_t i=1; i<K; ++i) {
            const V bmm0(&b[i*N], isBAligned);
            const V bmm1((&b[i*N+V::Size]),isBAligned);
            const V bmm2((&b[i*N+2*V::Size]),isBAligned);
            const V bmm3((&b[i*N+3*V::Size]),isBAligned);
            const V bmm4((&b[i*N+4*V::Size]),isBAligned);

            const V amm0       = a[j*K+i];
            const V amm1       = a[(j+1)*K+i];

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            omm1  = fmadd(amm0,bmm1,omm1);
            omm2  = fmadd(amm0,bmm2,omm2);
            omm3  = fmadd(amm0,bmm3,omm3);
            omm4  = fmadd(amm0,bmm4,omm4);
            // row 1
            omm5  = fmadd(amm1,bmm0,omm5);
            omm6  = fmadd(amm1,bmm1,omm6);
            omm7  = fmadd(amm1,bmm2,omm7);
            omm8  = fmadd(amm1,bmm3,omm8);
            omm9  = fmadd(amm1,bmm4,omm9);
        }

        omm0.store(&out[j*N],isCAligned);
        omm1.store(&out[j*N+V::Size],isCAligned);
        omm2.store(&out[j*N+2*V::Size],isCAligned);
        omm3.store(&out[j*N+3*V::Size],isCAligned);
        omm4.store(&out[j*N+4*V::Size],isCAligned);

        omm5.store(&out[(j+1)*N],isCAligned);
        omm6.store(&out[(j+1)*N+V::Size],isCAligned);
        omm7.store(&out[(j+1)*N+2*V::Size],isCAligned);
        omm8.store(&out[(j+1)*N+3*V::Size],isCAligned);
        omm9.store(&out[(j+1)*N+4*V::Size],isCAligned);
    }

    FASTOR_IF_CONSTEXPR (M-M0==1) {
        V amm0(a[j*K]);

        const V bmm0(&b[0], isBAligned);
        const V bmm1((&b[V::Size]),isBAligned);
        const V bmm2((&b[2*V::Size]),isBAligned);
        const V bmm3((&b[3*V::Size]),isBAligned);
        const V bmm4((&b[4*V::Size]),isBAligned);

        // row 0
        V omm0(amm0*bmm0);
        V omm1(amm0*bmm1);
        V omm2(amm0*bmm2);
        V omm3(amm0*bmm3);
        V omm4(amm0*bmm4);

        for (size_t i=1; i<K; ++i) {
            const V bmm0(&b[i*N], isBAligned);
            const V bmm1((&b[i*N+V::Size]),isBAligned);
            const V bmm2((&b[i*N+2*V::Size]),isBAligned);
            const V bmm3((&b[i*N+3*V::Size]),isBAligned);
            const V bmm4((&b[i*N+4*V::Size]),isBAligned);

            const V amm0       = a[j*K+i];

            // row 0
            omm0  = fmadd(amm0,bmm0,omm0);
            omm1  = fmadd(amm0,bmm1,omm1);
            omm2  = fmadd(amm0,bmm2,omm2);
            omm3  = fmadd(amm0,bmm3,omm3);
            omm4  = fmadd(amm0,bmm4,omm4);
        }

        omm0.store(&out[j*N],isCAligned);
        omm1.store(&out[j*N+V::Size],isCAligned);
        omm2.store(&out[j*N+2*V::Size],isCAligned);
        omm3.store(&out[j*N+3*V::Size],isCAligned);
        omm4.store(&out[j*N+4*V::Size],isCAligned);
    }
}

//-----------------------------------------------------------------------------------------------------------



template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<
            is_greater<N,5*choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size>::value,bool>::type = 0>
FASTOR_INLINE
void _matmul_mk_smalln(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {
    _matmul_base_masked<T,M,K,N>(a,b,out);
}



} // internal

} // Fastor


#endif // MATMUL_MK_SMALLODDN_H