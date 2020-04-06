#ifndef MATMUL_MK_SMALLODDN_H
#define MATMUL_MK_SMALLODDN_H


#include "Fastor/simd_vector/simd_vector_abi.h"
#include "Fastor/simd_vector/SIMDVector.h"

namespace Fastor {

namespace internal {


// This implementation covers all matrix-matrix multiplications with any M and K and
// and N<2*SIMDVector::Size. Given that it uses choose_best_simd_type it can switche
// between SSE, AVX and AVX512 to cover all ranges of N.
//
// For SSE: it covers [5 <= N < 8] for single and does not cover double
// For AVX: it covers [5 <= N < 15 && N!=8] for single and [5 <= N < 8] for double
// For AVX512: it covers [5 <= N < 31 && N!=8 && N!=16] for single and [5 <= N < 15 && N!=8] for double
//
// The function implements standard loop unrolling over M. It uses conditional
// loads and store using masks and requires at least AVX. The efficiency of the method comes from
// the fact that it attempts to achieve exact two FMA per load. Both GCC and Clang emit excellent
// code for this at O3
// A recursive implementation of this using compile time unrolling is available at:
// https://gist.github.com/romeric/a176e28127a8348c3c37c5a369051451
//
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<
            (is_less<N,2*choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size>::value &&
            N!=choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size &&
            is_greater_equal<N,5>::value),bool>::type = 0>
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

    size_t j=0;
    for (; j<M0; j+=unrollOuterloop) {
        const size_t from = j;
        V omm0, omm1, omm2, omm3, omm4, omm5, omm6, omm7, omm8, omm9;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N], false);
            const V bmm1(maskload<V>(&b[i*N+V::Size],maska));

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
        maskstore(&out[(from+4)*N+V::Size],maska,omm9);
    }

    // Remainder M-M0 rows
    // Explicitly unroll remaining loops, there is going to be atmost 4
    FASTOR_IF_CONSTEXPR (M-M0==4) {
        const size_t from = M0;
        V omm0, omm1, omm2, omm3, omm4, omm5, omm6, omm7;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N], false);
            const V bmm1(maskload<V>(&b[i*N+V::Size],maska));

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
        maskstore(&out[(from+3)*N+V::Size],maska,omm7);
    }

    else FASTOR_IF_CONSTEXPR (M-M0==3) {
        V omm0, omm1, omm2, omm3, omm4, omm5;
        constexpr size_t from = M0;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N], false);
            const V bmm1(maskload<V>(&b[i*N+V::Size],maska));

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
        maskstore(&out[(from+2)*N+V::Size],maska,omm5);
    }

    else FASTOR_IF_CONSTEXPR (M-M0==2) {
        V omm0, omm1, omm2, omm3;
        constexpr size_t from = M0;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N], false);
            const V bmm1(maskload<V>(&b[i*N+V::Size],maska));

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
        maskstore(&out[(from+1)*N+V::Size],maska,omm3);
    }

    else FASTOR_IF_CONSTEXPR (M-M0==1) {
        V omm0, omm1;
        constexpr size_t from = M0;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N], false);
            const V bmm1(maskload<V>(&b[i*N+V::Size],maska));

            const T amm0       = a[from*K+i];

            // row 0
            V a_vec0(amm0);
            omm0  = fmadd(a_vec0,bmm0,omm0);
            omm1  = fmadd(a_vec0,bmm1,omm1);
        }

        omm0.store(&out[from*N],false);
        maskstore(&out[from*N+V::Size],maska,omm1);
    }
#if 0
    else {
        V c_ij[M-M0][2];
        for (size_t j=M0; j<M; ++j) {
            for (size_t i=0; i<K; ++i) {
                const V bmm0(&b[i*N], false);
                const V bmm1(maskload<V>(&b[i*N+V::Size],maska));
                const V amm0(a[j*K+i]);
                c_ij[j][0] = fmadd(amm0,bmm0,c_ij[j][0]);
                c_ij[j][1] = fmadd(amm0,bmm1,c_ij[j][1]);
            }
            c_ij[j][0].store(&out[j*N],false);
            maskstore(&out[j*N+V::Size],maska,c_ij[j][1]);
        }
    }
#endif
}



// Cover the N==2 and N==3 case
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<N==3 || (N==2 && choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size!=2),bool>::type = 0>
FASTOR_INLINE
void _matmul_mk_smalln(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    using V = typename std::conditional<sizeof(T)==8, SIMDVector<T,simd_abi::avx>,
                typename std::conditional<sizeof(T)==4, SIMDVector<T,simd_abi::sse>,
                    SIMDVector<T,simd_abi::fixed_size<4>>
                >::type
              >::type;

    constexpr size_t unrollOuterloop = 5UL;
    constexpr size_t M0 = M / unrollOuterloop * unrollOuterloop;
    // constexpr size_t remainder = M < unrollOuterloop ? 0 : M0-unrollOuterloop;

    // Number of columns of c (N) that can be safely unrolled with V::Size
    constexpr size_t N1 = N / V::Size * V::Size;

    int maska[V::Size];
    std::fill(maska,&maska[V::Size], -1);
    for (size_t jj=0; jj < V::Size - (N-N1); ++jj) maska[jj] = 0;

    size_t j=0;
    for (; j<M0; j+=unrollOuterloop) {
        V omm0, omm1, omm2, omm3, omm4;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(maskload<V>(&b[i*N],maska));

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

        omm0.store(&out[(j+0)*N],false);
        omm1.store(&out[(j+1)*N],false);
        omm2.store(&out[(j+2)*N],false);
        omm3.store(&out[(j+3)*N],false);
        maskstore(&out[(j+4)*N],maska,omm4);
    }

    // Remainder M-M0 rows
    // Explicitly unroll remaining loops, there is going to be atmost 4
    FASTOR_IF_CONSTEXPR (M-M0==4) {
        V omm0, omm1, omm2, omm3;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(maskload<V>(&b[i*N],maska));

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

        omm0.store(&out[(j+0)*N],false);
        omm1.store(&out[(j+1)*N],false);
        omm2.store(&out[(j+2)*N],false);
        maskstore(&out[(j+3)*N],maska,omm3);
    }

    else FASTOR_IF_CONSTEXPR (M-M0==3) {
        V omm0, omm1, omm2;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(maskload<V>(&b[i*N],maska));

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

        omm0.store(&out[(j+0)*N],false);
        omm1.store(&out[(j+1)*N],false);
        maskstore(&out[(j+2)*N],maska,omm2);
    }

    else FASTOR_IF_CONSTEXPR (M-M0==2) {
        V omm0, omm1;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(maskload<V>(&b[i*N],maska));

            const T amm0       = a[j*K+i];
            const T amm1       = a[(j+1)*K+i];

            // row 0
            V a_vec0(amm0);
            omm0  = fmadd(a_vec0,bmm0,omm0);
            // row 1
            V a_vec1(amm1);
            omm1  = fmadd(a_vec1,bmm0,omm1);
        }

        omm0.store(&out[(j+0)*N],false);
        maskstore(&out[(j+1)*N],maska,omm1);
    }

    else FASTOR_IF_CONSTEXPR (M-M0==1) {
        V omm0;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(maskload<V>(&b[i*N],maska));

            const T amm0       = a[j*K+i];

            // row 0
            V a_vec0(amm0);
            omm0  = fmadd(a_vec0,bmm0,omm0);
        }

        maskstore(&out[(j+0)*N],maska,omm0);
    }
#if 0
    // This is never hit but kept for debugging
    else {
        V c_ij[M-M0];
        for (size_t j=M0; j<M; ++j) {
            for (size_t i=0; i<K; ++i) {
                const V bmm0(maskload<V>(&b[i*N],maska));
                const V amm0(a[j*K+i]);
                c_ij[j] = fmadd(amm0,bmm0,c_ij[j]);
            }
            maskstore(&out[j*N],maska,c_ij[j]);
        }
    }
#endif
}



// Take care of 2*V::Size cases
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
        c_ij[from].store(&out[from*SIMDVector<T,ABI>::Size]);
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
        choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size==N,bool>::type = 0>
FASTOR_INLINE
void _matmul_mk_smalln(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;
    V c_ij[M];

    for (size_t i=0; i<K; ++i) {
        const V bmm0(&b[i*V::Size]);
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
        choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size==N,bool>::type = 0>
FASTOR_INLINE
void _matmul_mk_smalln(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;

    constexpr size_t unrollOuterloop = 5UL;
    constexpr size_t M0 = M / unrollOuterloop * unrollOuterloop;
    // constexpr size_t remainder = M < unrollOuterloop ? 0 : M0-unrollOuterloop;

    size_t j=0;
    for (; j<M0; j+=unrollOuterloop) {
        V omm0, omm1, omm2, omm3, omm4;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N]);

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

        omm0.store(&out[(j+0)*N],false);
        omm1.store(&out[(j+1)*N],false);
        omm2.store(&out[(j+2)*N],false);
        omm3.store(&out[(j+3)*N],false);
        omm4.store(&out[(j+4)*N],false);
    }

    // Remainder M-M0 rows
    // Explicitly unroll remaining loops, there is going to be atmost 4
    FASTOR_IF_CONSTEXPR (M-M0==4) {
        V omm0, omm1, omm2, omm3;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N]);

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

        omm0.store(&out[(j+0)*N],false);
        omm1.store(&out[(j+1)*N],false);
        omm2.store(&out[(j+2)*N],false);
        omm3.store(&out[(j+3)*N],false);
    }

    else FASTOR_IF_CONSTEXPR (M-M0==3) {
        V omm0, omm1, omm2;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N]);

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

        omm0.store(&out[(j+0)*N],false);
        omm1.store(&out[(j+1)*N],false);
        omm2.store(&out[(j+2)*N],false);
    }

    else FASTOR_IF_CONSTEXPR (M-M0==2) {
        V omm0, omm1;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N]);

            const T amm0       = a[j*K+i];
            const T amm1       = a[(j+1)*K+i];

            // row 0
            V a_vec0(amm0);
            omm0  = fmadd(a_vec0,bmm0,omm0);
            // row 1
            V a_vec1(amm1);
            omm1  = fmadd(a_vec1,bmm0,omm1);
        }

        omm0.store(&out[(j+0)*N],false);
        omm1.store(&out[(j+1)*N],false);
    }

    else FASTOR_IF_CONSTEXPR (M-M0==1) {
        V omm0;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N]);

            const T amm0       = a[j*K+i];

            // row 0
            V a_vec0(amm0);
            omm0  = fmadd(a_vec0,bmm0,omm0);
        }

        omm0.store(&out[(j+0)*N],false);
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




// Take care of 4*V::Size cases
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
        V omm0, omm1, omm2, omm3, omm4, omm5, omm6, omm7;
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
        V omm0, omm1, omm2, omm3;
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


// Take care of 4*V::Size cases
// Note that you get the exact same performance by tuning _matmul_base parameters as
// [unrollOuterloop = 2, nSIMDRows=1 and nSIMDCols=4] however these parameters affect
// the universal behaviour of that method specially for big matrices
// This method unrolls M by 2 only as when N==4*V::Size unrolling the entire column of c
// and doing 2 rows at a time is the most benificial. Unrolling M any further hurts the
// performance really bad
//-----------------------------------------------------------------------------------------------------------
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<N==4*choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size,bool>::type = 0>
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
        V omm0, omm1, omm2, omm3, omm4, omm5, omm6, omm7;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N], isBAligned);
            const V bmm1((&b[i*N+V::Size]),isBAligned);
            const V bmm2((&b[i*N+2*V::Size]),isBAligned);
            const V bmm3((&b[i*N+3*V::Size]),isBAligned);

            const T amm0       = a[from*K+i];
            const T amm1       = a[(from+1)*K+i];

            // row 0
            V a_vec0(amm0);
            omm0  = fmadd(a_vec0,bmm0,omm0);
            omm1  = fmadd(a_vec0,bmm1,omm1);
            omm2  = fmadd(a_vec0,bmm2,omm2);
            omm3  = fmadd(a_vec0,bmm3,omm3);
            // row 1
            V a_vec1(amm1);
            omm4  = fmadd(a_vec1,bmm0,omm4);
            omm5  = fmadd(a_vec1,bmm1,omm5);
            omm6  = fmadd(a_vec1,bmm2,omm6);
            omm7  = fmadd(a_vec1,bmm3,omm7);
        }

        omm0.store(&out[from*N],isCAligned);
        omm1.store(&out[from*N+V::Size],isCAligned);
        omm2.store(&out[from*N+2*V::Size],isCAligned);
        omm3.store(&out[from*N+3*V::Size],isCAligned);

        omm4.store(&out[(from+1)*N],isCAligned);
        omm5.store(&out[(from+1)*N+V::Size],isCAligned);
        omm6.store(&out[(from+1)*N+2*V::Size],isCAligned);
        omm7.store(&out[(from+1)*N+3*V::Size],isCAligned);
    }

    FASTOR_IF_CONSTEXPR (M-M0==1) {
        const size_t from = j;
        V omm0, omm1, omm2, omm3;
        for (size_t i=0; i<K; ++i) {
            const V bmm0(&b[i*N], isBAligned);
            const V bmm1((&b[i*N+V::Size]),isBAligned);
            const V bmm2((&b[i*N+2*V::Size]),isBAligned);
            const V bmm3((&b[i*N+3*V::Size]),isBAligned);

            const T amm0       = a[from*K+i];

            // row 0
            V a_vec0(amm0);
            omm0  = fmadd(a_vec0,bmm0,omm0);
            omm1  = fmadd(a_vec0,bmm1,omm1);
            omm2  = fmadd(a_vec0,bmm2,omm2);
            omm3  = fmadd(a_vec0,bmm3,omm3);
        }

        omm0.store(&out[from*N],isCAligned);
        omm1.store(&out[from*N+V::Size],isCAligned);
        omm2.store(&out[from*N+2*V::Size],isCAligned);
        omm3.store(&out[from*N+3*V::Size],isCAligned);
    }
}



//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------



template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<
            (
            is_greater<N,2*choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size>::value &&
            N!=3*choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size &&
            N!=4*choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type::Size
            )
             || N==1,bool>::type = 0>
FASTOR_INLINE
void _matmul_mk_smalln(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {
    _matmul_base_masked<T,M,K,N>(a,b,out);
}



} // internal

} // Fastor


#endif // MATMUL_MK_SMALLODDN_H