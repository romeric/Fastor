#ifndef MATMUL_KERNELS_H
#define MATMUL_KERNELS_H


#include "Fastor/config/config.h"
#include "Fastor/simd_vector/extintrin.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/meta/tensor_meta.h"


namespace Fastor {

namespace internal {


//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A set of helper functions for the inner blocks of matmul. Almost all compilers (GCC/CLang/Intel)
// unroll the inner-most loop (on unrollOuterloop)
//-----------------------------------------------------------------------------------------------------------
template<typename T, typename V, size_t M, size_t K, size_t N, size_t unrollOuterloop, size_t numSIMDRows, size_t numSIMDCols,
    typename std::enable_if<numSIMDCols==1,bool>::type = false>
FASTOR_INLINE
void interior_block_matmul_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j) {

    for (size_t ii = 0; ii < numSIMDRows; ++ii) {

        V c_ij[unrollOuterloop*numSIMDCols];
        // Loop over columns of a (rows of b)
        for (size_t k = 0; k < K; ++k) {

            const V bmm0(&b[k*N+j],false);

            for (size_t n = 0; n < unrollOuterloop; ++n) {

                const V amm0 = a[(i+ii*unrollOuterloop+n)*K+k];

                c_ij[n]                    = fmadd(amm0,bmm0,c_ij[n]);
            }
        }
        for (size_t n = 0; n < unrollOuterloop; ++n) {
            c_ij[n].store(&c[(i+ii*unrollOuterloop+n)*N+j],false);
        }
    }
}

template<typename T, typename V, size_t M, size_t K, size_t N, size_t unrollOuterloop, size_t numSIMDRows, size_t numSIMDCols,
    typename std::enable_if<numSIMDCols==2,bool>::type = false>
FASTOR_INLINE
void interior_block_matmul_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j) {

    for (size_t ii = 0; ii < numSIMDRows; ++ii) {

        V c_ij[unrollOuterloop*numSIMDCols];
        // Loop over columns of a (rows of b)
        for (size_t k = 0; k < K; ++k) {

            const V bmm0(&b[k*N+j],false);
            const V bmm1(&b[k*N+j+V::Size],false);

            for (size_t n = 0; n < unrollOuterloop; ++n) {

                const V amm0 = a[(i+ii*unrollOuterloop+n)*K+k];

                c_ij[n]                    = fmadd(amm0,bmm0,c_ij[n]);
                c_ij[n+unrollOuterloop]    = fmadd(amm0,bmm1,c_ij[n+unrollOuterloop]);
            }
        }
        for (size_t n = 0; n < unrollOuterloop; ++n) {
            c_ij[n].store(&c[(i+ii*unrollOuterloop+n)*N+j],false);
            c_ij[n+unrollOuterloop].store(&c[(i+ii*unrollOuterloop+n)*N+j+V::Size],false);
        }
    }
}


template<typename T, typename V, size_t M, size_t K, size_t N, size_t unrollOuterloop, size_t numSIMDRows, size_t numSIMDCols,
    typename std::enable_if<numSIMDCols==3,bool>::type = false>
FASTOR_INLINE
void interior_block_matmul_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j) {

    for (size_t ii = 0; ii < numSIMDRows; ++ii) {

        V c_ij[unrollOuterloop*numSIMDCols];
        // Loop over columns of a (rows of b)
        for (size_t k = 0; k < K; ++k) {

            const V bmm0(&b[k*N+j],false);
            const V bmm1(&b[k*N+j+V::Size],false);
            const V bmm2(&b[k*N+j+2*V::Size],false);

            for (size_t n = 0; n < unrollOuterloop; ++n) {

                const V amm0 = a[(i+ii*unrollOuterloop+n)*K+k];

                c_ij[n]                    = fmadd(amm0,bmm0,c_ij[n]);
                c_ij[n+unrollOuterloop]    = fmadd(amm0,bmm1,c_ij[n+unrollOuterloop]);
                c_ij[n+2*unrollOuterloop]  = fmadd(amm0,bmm2,c_ij[n+2*unrollOuterloop]);
            }
        }
        for (size_t n = 0; n < unrollOuterloop; ++n) {
            c_ij[n].store(&c[(i+ii*unrollOuterloop+n)*N+j],false);
            c_ij[n+unrollOuterloop].store(&c[(i+ii*unrollOuterloop+n)*N+j+V::Size],false);
            c_ij[n+2*unrollOuterloop].store(&c[(i+ii*unrollOuterloop+n)*N+j+2*V::Size],false);
        }
    }
}


template<typename T, typename V, size_t M, size_t K, size_t N, size_t unrollOuterloop, size_t numSIMDRows, size_t numSIMDCols,
    typename std::enable_if<numSIMDCols==4,bool>::type = false>
FASTOR_INLINE
void interior_block_matmul_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j) {

    for (size_t ii = 0; ii < numSIMDRows; ++ii) {

        V c_ij[unrollOuterloop*numSIMDCols];
        // Loop over columns of a (rows of b)
        for (size_t k = 0; k < K; ++k) {

            const V bmm0(&b[k*N+j],false);
            const V bmm1(&b[k*N+j+V::Size],false);
            const V bmm2(&b[k*N+j+2*V::Size],false);
            const V bmm3(&b[k*N+j+3*V::Size],false);

            for (size_t n = 0; n < unrollOuterloop; ++n) {

                const V amm0 = a[(i+ii*unrollOuterloop+n)*K+k];

                c_ij[n]                    = fmadd(amm0,bmm0,c_ij[n]);
                c_ij[n+unrollOuterloop]    = fmadd(amm0,bmm1,c_ij[n+unrollOuterloop]);
                c_ij[n+2*unrollOuterloop]  = fmadd(amm0,bmm2,c_ij[n+2*unrollOuterloop]);
                c_ij[n+3*unrollOuterloop]  = fmadd(amm0,bmm3,c_ij[n+3*unrollOuterloop]);
            }
        }
        for (size_t n = 0; n < unrollOuterloop; ++n) {
            c_ij[n].store(&c[(i+ii*unrollOuterloop+n)*N+j],false);
            c_ij[n+unrollOuterloop].store(&c[(i+ii*unrollOuterloop+n)*N+j+V::Size],false);
            c_ij[n+2*unrollOuterloop].store(&c[(i+ii*unrollOuterloop+n)*N+j+2*V::Size],false);
            c_ij[n+3*unrollOuterloop].store(&c[(i+ii*unrollOuterloop+n)*N+j+3*V::Size],false);
        }
    }
}


template<typename T, typename V, size_t M, size_t K, size_t N, size_t unrollOuterloop, size_t numSIMDRows, size_t numSIMDCols,
    typename std::enable_if<numSIMDCols==5,bool>::type = false>
FASTOR_INLINE
void interior_block_matmul_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j) {

    for (size_t ii = 0; ii < numSIMDRows; ++ii) {

        V c_ij[unrollOuterloop*numSIMDCols];
        // Loop over columns of a (rows of b)
        for (size_t k = 0; k < K; ++k) {

            const V bmm0(&b[k*N+j],false);
            const V bmm1(&b[k*N+j+V::Size],false);
            const V bmm2(&b[k*N+j+2*V::Size],false);
            const V bmm3(&b[k*N+j+3*V::Size],false);
            const V bmm4(&b[k*N+j+4*V::Size],false);

            for (size_t n = 0; n < unrollOuterloop; ++n) {

                const V amm0 = a[(i+ii*unrollOuterloop+n)*K+k];

                c_ij[n]                    = fmadd(amm0,bmm0,c_ij[n]);
                c_ij[n+unrollOuterloop]    = fmadd(amm0,bmm1,c_ij[n+unrollOuterloop]);
                c_ij[n+2*unrollOuterloop]  = fmadd(amm0,bmm2,c_ij[n+2*unrollOuterloop]);
                c_ij[n+3*unrollOuterloop]  = fmadd(amm0,bmm3,c_ij[n+3*unrollOuterloop]);
                c_ij[n+4*unrollOuterloop]  = fmadd(amm0,bmm3,c_ij[n+4*unrollOuterloop]);
            }
        }
        for (size_t n = 0; n < unrollOuterloop; ++n) {
            c_ij[n].store(&c[(i+ii*unrollOuterloop+n)*N+j],false);
            c_ij[n+unrollOuterloop].store(&c[(i+ii*unrollOuterloop+n)*N+j+V::Size],false);
            c_ij[n+2*unrollOuterloop].store(&c[(i+ii*unrollOuterloop+n)*N+j+2*V::Size],false);
            c_ij[n+3*unrollOuterloop].store(&c[(i+ii*unrollOuterloop+n)*N+j+3*V::Size],false);
            c_ij[n+4*unrollOuterloop].store(&c[(i+ii*unrollOuterloop+n)*N+j+4*V::Size],false);
        }
    }
}


template<typename T, typename V, size_t M, size_t K, size_t N, size_t unrollOuterloop, size_t numSIMDRows, size_t numSIMDCols,
    typename std::enable_if<numSIMDCols==1,bool>::type = false>
FASTOR_INLINE
void interior_block_matmul_scalar_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j) {

    for (size_t ii = 0; ii < numSIMDRows; ++ii) {

        T c_ij[unrollOuterloop*numSIMDCols] = {};
        // Loop over columns of a (rows of b)
        for (size_t k = 0; k < K; ++k) {
            const T bmm0(b[k*N+j]);
            for (size_t n = 0; n < unrollOuterloop; ++n) {

                const T amm0 = a[(i+ii*unrollOuterloop+n)*K+k];

                c_ij[n]                                  += amm0*bmm0;
            }
        }
        for (size_t n = 0; n < unrollOuterloop; ++n) {
            c[(i+ii*unrollOuterloop+n)*N+j]           = c_ij[n];
        }
    }
}


template<typename T, typename V, size_t M, size_t K, size_t N, size_t unrollOuterloop, size_t numSIMDRows, size_t numSIMDCols,
    typename std::enable_if<numSIMDCols==1,bool>::type = false>
FASTOR_INLINE
void interior_block_matmul_mask_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j, const int (&maska)[V::Size]) {

    for (size_t ii = 0; ii < numSIMDRows; ++ii) {

        V c_ij[unrollOuterloop*numSIMDCols];
        // Loop over columns of a (rows of b)
        for (size_t k = 0; k < K; ++k) {

            const V bmm0(maskload<V>(&b[k*N+j],maska));

            for (size_t n = 0; n < unrollOuterloop; ++n) {

                const V amm0 = a[(i+ii*unrollOuterloop+n)*K+k];

                c_ij[n]                    = fmadd(amm0,bmm0,c_ij[n]);
            }
        }
        for (size_t n = 0; n < unrollOuterloop; ++n) {
            maskstore(&c[(i+ii*unrollOuterloop+n)*N+j],maska,c_ij[n]);
        }
    }
}


template<typename T, typename MaskT, typename V, size_t M, size_t K, size_t N, size_t unrollOuterloop, size_t numSIMDRows, size_t numSIMDCols,
    typename std::enable_if<numSIMDCols==1,bool>::type = false>
FASTOR_INLINE
void interior_block_matmul_mask_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j, const MaskT mask) {

    V bmm0;
    for (size_t ii = 0; ii < numSIMDRows; ++ii) {

        V c_ij[unrollOuterloop*numSIMDCols];
        // Loop over columns of a (rows of b)
        for (size_t k = 0; k < K; ++k) {

            bmm0.mask_load(&b[k*N+j],mask,false);

            for (size_t n = 0; n < unrollOuterloop; ++n) {

                const V amm0 = a[(i+ii*unrollOuterloop+n)*K+k];

                c_ij[n]                    = fmadd(amm0,bmm0,c_ij[n]);
            }
        }

        for (size_t n = 0; n < unrollOuterloop; ++n) {
            c_ij[n].mask_store(&c[(i+ii*unrollOuterloop+n)*N+j],mask,false);
        }
    }
}
//-----------------------------------------------------------------------------------------------------------



//-----------------------------------------------------------------------------------------------------------
// This is the base implementation of matrix-matrix multiplication for all 2D tensors and
// higher order tensor products that can be expressed as gemm
// The function uses two level unrolling one based on block sizes and one based on register widths
// with any remainder left treated in a scalar fashion
template<typename T, size_t M, size_t K, size_t N>
FASTOR_INLINE
void _matmul_base(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {

    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;

    // This parameter can be adjusted and does not need to be 4UL/8UL etc
    // constexpr size_t unrollOuterloop = M % 5UL == 0 ? 5UL : 4UL;
    constexpr size_t unrollOuterloop = 4UL;

#ifndef FASTOR_MATMUL_OUTER_BLOCK_SIZE
    // Unroll the rows of (a and c) (M) by [numSIMDRows * V::Size]
    constexpr size_t numSIMDRows = M % (unrollOuterloop * 3UL) == 0 ? 3UL : (M < 2UL*V::Size ? 1UL : 2UL);
#else
    constexpr size_t numSIMDRows = FASTOR_MATMUL_OUTER_BLOCK_SIZE;
#endif
#ifndef FASTOR_MATMUL_INNER_BLOCK_SIZE
    // Unroll the columns of (b and c) (N) by [numSIMDCols * V::Size]
    constexpr size_t numSIMDCols = (N % (V::Size * 3UL) == 0 && M % (V::Size * 3UL) == 0 && N > 24UL) ? 3UL : 2UL;
#else
    constexpr size_t numSIMDCols = FASTOR_MATMUL_INNER_BLOCK_SIZE;
#endif

    // The goal is to get 10 parallel independent chains of accumulators
    // to saturate the pipeline by having a completely unrolled block of
    // [(unrollOuterloop) * (numSIMDCols)] at a time. A minimum value of
    // unrollOuterloop=4 ensures a minimum of 8 independent parallel chains
    // while a maximum of 12 i.e. for numSIMDCols=2 and numSIMDCols=3 respectively.
    // However, most recent X86/64 architectures can do 2 FMAs per load so
    // so unrolling with numSIMDCols > 2 is not beneficial

    constexpr size_t unrollOuterBlock = numSIMDRows*unrollOuterloop;
    // Number of rows of c (M) that can be safely unrolled with this block size.
    constexpr size_t M0 = M / unrollOuterBlock * unrollOuterBlock;

    constexpr size_t unrollInnerBlock = numSIMDCols*V::Size;
    // Number of columns of c (N) that can be safely unrolled with this block size
    constexpr size_t N0 = N / unrollInnerBlock * unrollInnerBlock;

    // Number of columns of c (N) that can be safely unrolled with V::Size
    constexpr size_t N1 = N / V::Size * V::Size;

    size_t i = 0;
    for (; i < M0; i += unrollOuterBlock) {
        size_t j = 0;
        for (; j < N0; j += unrollInnerBlock) {
            interior_block_matmul_impl<T,V,M,K,N,unrollOuterloop,numSIMDRows,numSIMDCols>(a,b,c,i,j);
        }

        // Remaining N - N0 columns
        for (; j < N1; j += V::Size) {
            interior_block_matmul_impl<T,V,M,K,N,unrollOuterloop,numSIMDRows,1>(a,b,c,i,j);
        }

        // Remaining N - N1 columns
        for (; j < N; ++j) {
            interior_block_matmul_scalar_impl<T,V,M,K,N,unrollOuterloop,numSIMDRows,1>(a,b,c,i,j);
        }

    }

    // The remaining M-M0 rows are now unrolled yet again by unrollOuterloop.
    // This is necessary as for small sizes the earlier block loop may not be
    // triggered if the size of the block is bigger than the number of rows of
    // (a and c) i.e. M
    constexpr size_t M1 = (M / unrollOuterloop * unrollOuterloop);
    for (; i < M1; i += unrollOuterloop) {
        size_t j = 0;
        for (; j < N0; j += unrollInnerBlock) {
            interior_block_matmul_impl<T,V,M,K,N,unrollOuterloop,1,numSIMDCols>(a,b,c,i,j);
        }

        // Remaining N - N0 columns
        for (; j < N1; j += V::Size) {
            V c_ij[unrollOuterloop];
            for (size_t k = 0; k < K; ++k) {
                for (size_t n = 0; n < unrollOuterloop; ++n) {
                    c_ij[n] = fmadd(V(a[(i + n)*K+k]), V(&b[k*N+j],false), c_ij[n]);
                }
            }
            for (size_t n = 0; n < unrollOuterloop; ++n) {
                c_ij[n].store(&c[(i + n)*N+j],false);
            }
        }

        // Remaining N - N1 columns
        for (; j < N; ++j) {
            T c_ij[unrollOuterloop] = {};
            for (size_t k = 0; k < K; ++k) {
                for (size_t n = 0; n < unrollOuterloop; ++n) {
                    c_ij[n] += a[(i + n)*K+k] * b[k*N+j];
                }
            }
            for (size_t n = 0; n < unrollOuterloop; ++n) {
                c[(i + n)*N+j] = c_ij[n];
            }
        }
    }

    // Now treat the remaining M-M1 rows
    FASTOR_IF_CONSTEXPR (M-M1 > 0) {
        // Hack to get around zero length array issue
        constexpr size_t MM1 = M-M1 != 0 ? M-M1 : 1;
        size_t j = 0;
        for (; j < N0; j += unrollInnerBlock) {
            // If MM1==0 the function never gets invoked anyway
            interior_block_matmul_impl<T,V,M,K,N,MM1,1,numSIMDCols>(a,b,c,i,j);
        }

        // Remaining N - N0 columns
        for (; j < N1; j += V::Size) {
            V c_ij[MM1];
            for (size_t k = 0; k < K; ++k) {
                for (size_t n = M1; n < M; ++n) {
                    c_ij[n-M1] = fmadd(V(a[n*K+k]), V(&b[k*N+j],false), c_ij[n-M1]);
                    c_ij[n-M1].store(&c[n*N+j],false);
                }
            }
            for (size_t n = M1; n < M; ++n) {
                c_ij[n-M1].store(&c[n*N+j],false);
            }
        }

        // Remaining N - N1 columns
        for (; j < N; ++j) {
            T c_ij[MM1] = {};
            for (size_t k = 0; k < K; ++k) {
                for (size_t n = M1; n < M; ++n) {
                    c_ij[n-M1] += a[n*K+k] * b[k*N+j];
                    c[n*N+j] = c_ij[n-M1];
                }
            }
            for (size_t n = M1; n < M; ++n) {
                c[n*N+j] = c_ij[n-M1];
            }
        }
    }
}
//-----------------------------------------------------------------------------------------------------------



//-----------------------------------------------------------------------------------------------------------
// This is the base implementation of matrix-matrix multiplication for all 2D tensors and
// higher order tensor products that can be expressed as gemm
// The function uses two level unrolling one based on block sizes and one based on register widths
// with any remainder left treated in vector mode with masked and conditional load/stores.
// Note that conditional load/store requires at least AVX intrinsics
template<typename T, size_t M, size_t K, size_t N>
FASTOR_INLINE
void _matmul_base_masked(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {

    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;

    // This parameter can be adjusted and does not need to be 4UL/8UL etc
    // constexpr size_t unrollOuterloop = M % 5UL == 0 ? 5UL : 4UL;
    constexpr size_t unrollOuterloop = 4UL;

#ifndef FASTOR_MATMUL_OUTER_BLOCK_SIZE
    // Unroll the rows of (a and c) (M) by [numSIMDRows * V::Size]
    constexpr size_t numSIMDRows = M % (unrollOuterloop * 3UL) == 0 ? 3UL : (M < 2UL*V::Size ? 1UL : 2UL);
#else
    constexpr size_t numSIMDRows = FASTOR_MATMUL_OUTER_BLOCK_SIZE;
#endif
#ifndef FASTOR_MATMUL_INNER_BLOCK_SIZE
    // Unroll the columns of (b and c) (N) by [numSIMDCols * V::Size]
    constexpr size_t numSIMDCols = (N % (V::Size * 3UL) == 0 && M % (V::Size * 3UL) == 0 && N > 24UL) ? 3UL : 2UL;
#else
    constexpr size_t numSIMDCols = FASTOR_MATMUL_INNER_BLOCK_SIZE;
#endif

    // The goal is to get 10 parallel independent chains of accumulators
    // to saturate the pipeline by having a completely unrolled block of
    // [(unrollOuterloop) * (numSIMDCols)] at a time. A minimum value of
    // unrollOuterloop=4 ensures a minimum of 8 independent parallel chains
    // while a maximum of 12 i.e. for numSIMDCols=2 and numSIMDCols=3 respectively.
    // However, most recent X86/64 architectures can do 2 FMAs per load so
    // so unrolling with numSIMDCols > 2 is not beneficial

    constexpr size_t unrollOuterBlock = numSIMDRows*unrollOuterloop;
    // Number of rows of c (M) that can be safely unrolled with this block size.
    constexpr size_t M0 = M / unrollOuterBlock * unrollOuterBlock;

    constexpr size_t unrollInnerBlock = numSIMDCols*V::Size;
    // Number of columns of c (N) that can be safely unrolled with this block size
    constexpr size_t N0 = N / unrollInnerBlock * unrollInnerBlock;

    // Number of columns of c (N) that can be safely unrolled with V::Size
    constexpr size_t N1 = N / V::Size * V::Size;

    int maska[V::Size];
    std::fill(maska,&maska[V::Size], -1);
    for (size_t jj=0; jj < V::Size - (N-N1); ++jj) maska[jj] = 0;
#ifdef FASTOR_HAS_AVX512_MASKS
    const auto mask = array_to_mask(maska);
#endif

    size_t i = 0;
    for (; i < M0; i += unrollOuterBlock) {
        size_t j = 0;
        for (; j < N0; j += unrollInnerBlock) {
            interior_block_matmul_impl<T,V,M,K,N,unrollOuterloop,numSIMDRows,numSIMDCols>(a,b,c,i,j);
        }

        // Remaining N - N0 columns
        for (; j < N1; j += V::Size) {
            interior_block_matmul_impl<T,V,M,K,N,unrollOuterloop,numSIMDRows,1>(a,b,c,i,j);
        }

        // Remaining N - N1 columns
        for (; j < N; j+= N-N1) {
#ifdef FASTOR_HAS_AVX512_MASKS
            interior_block_matmul_mask_impl<T,decltype(mask),V,M,K,N,unrollOuterloop,numSIMDRows,1>(a,b,c,i,j,mask);
#else
            interior_block_matmul_mask_impl<T,V,M,K,N,unrollOuterloop,numSIMDRows,1>(a,b,c,i,j,maska);
#endif
        }
    }

    // The remaining M-M0 rows are now unrolled yet again by unrollOuterloop.
    // This is necessary as for small sizes the earlier block loop may not be
    // triggered if the size of the block is bigger than the number of rows of
    // (a and c) i.e. M
    constexpr size_t M1 = (M / unrollOuterloop * unrollOuterloop);
    for (; i < M1; i += unrollOuterloop) {
        size_t j = 0;
        for (; j < N0; j += unrollInnerBlock) {
            interior_block_matmul_impl<T,V,M,K,N,unrollOuterloop,1,numSIMDCols>(a,b,c,i,j);
        }

        // Remaining N - N0 columns
        for (; j < N1; j += V::Size) {
            V c_ij[unrollOuterloop];
            for (size_t k = 0; k < K; ++k) {
                for (size_t n = 0; n < unrollOuterloop; ++n) {
                    c_ij[n] = fmadd(V(a[(i + n)*K+k]), V(&b[k*N+j],false), c_ij[n]);
                }
            }
            for (size_t n = 0; n < unrollOuterloop; ++n) {
                c_ij[n].store(&c[(i + n)*N+j],false);
            }
        }

        // Remaining N - N1 columns
        for (; j < N; j+=N-N1) {
            V c_ij[unrollOuterloop];
            for (size_t k = 0; k < K; ++k) {
                for (size_t n = 0; n < unrollOuterloop; ++n) {
#ifdef FASTOR_HAS_AVX512_MASKS
                    V bmm0; bmm0.mask_load(&b[k*N+j],mask);
#else
                    const V bmm0(maskload<V>(&b[k*N+j],maska));
#endif
                    const V amm0 = a[(i + n)*K+k];
                    c_ij[n] = fmadd(amm0,bmm0,c_ij[n]);
                }
            }
            for (size_t n = 0; n < unrollOuterloop; ++n) {
#ifdef FASTOR_HAS_AVX512_MASKS
                c_ij[n].mask_store(&c[(i+n)*N+j],mask,false);
#else
                maskstore(&c[(i+n)*N+j],maska,c_ij[n]);
#endif
            }
        }
    }

    // Now treat the remaining M-M1 rows
    FASTOR_IF_CONSTEXPR (M-M1 > 0) {
        // Hack to get around zero length array issue
        constexpr size_t MM1 = M-M1 != 0 ? M-M1 : 1;
        size_t j = 0;
        for (; j < N0; j += unrollInnerBlock) {
            // If MM1==0 the function never gets invoked anyway
            interior_block_matmul_impl<T,V,M,K,N,MM1,1,numSIMDCols>(a,b,c,i,j);
        }

        // Remaining N - N0 columns
        for (; j < N1; j += V::Size) {
            V c_ij[MM1];
            for (size_t k = 0; k < K; ++k) {
                for (size_t n = M1; n < M; ++n) {
                    c_ij[n-M1] = fmadd(V(a[n*K+k]), V(&b[k*N+j],false), c_ij[n-M1]);
                    c_ij[n-M1].store(&c[n*N+j],false);
                }
            }
            for (size_t n = M1; n < M; ++n) {
                c_ij[n-M1].store(&c[n*N+j],false);
            }
        }

        // Remaining N - N1 columns
        for (; j < N; j+=N-N1) {
            V c_ij[MM1] = {};
            for (size_t k = 0; k < K; ++k) {
                for (size_t n = M1; n < M; ++n) {
#ifdef FASTOR_HAS_AVX512_MASKS
                    V bmm0; bmm0.mask_load(&b[k*N+j],mask);
#else
                    const V bmm0(maskload<V>(&b[k*N+j],maska));
#endif
                    const V amm0 = a[n*K+k];
                    c_ij[n-M1] = fmadd(amm0,bmm0,c_ij[n-M1]);
                }
            }
            for (size_t n = M1; n < M; ++n) {
#ifdef FASTOR_HAS_AVX512_MASKS
                c_ij[n-M1].mask_store(&c[n*N+j],mask,false);
#else
                maskstore(&c[n*N+j],maska,c_ij[n-M1]);
#endif
            }
        }
    }
}
//-----------------------------------------------------------------------------------------------------------




//-----------------------------------------------------------------------------------------------------------
// matmul kernel for non-fundamental types
// The assumption here is that non-fundamental types are not SIMD vectorisable for instance
// Tensor<std::vector<T>,3,3> or Tensor<Tensor<...>,...> plus they cannot fuse [do fused-add-multiply]
// so operations like [c += a*b] or potentially [c = c + a*b] might introduce multiple copies in
// the inner most loops of matmul
template<typename T, size_t M, size_t K, size_t N>
FASTOR_INLINE
void _matmul_base_non_primitive(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {
    // There is no SIMD here as V::Size == 1 anyway
    // No outer loop unrolling otherwise the innermost loop
    // will create unnecessary temporaries
    for (size_t i=0; i<M; ++i) {
        // V::Size == 1 so this loop can't be unrolled
        for (size_t j=0; j<N; ++j) {
            // This could potentially cost as opposed to directly writing in to c
            T tmp {};
            for (size_t k=0; k<K; ++k) {
                tmp += a[i*K+k]*b[k*N+j];
            }
            c[i*N+j] = tmp;
        }
    }
}
//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------























// Other variants and slightly older implementations
//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------
// This is the same implementation as the above case but does not unroll on block sizes and does not require
// the registers to be zeroed out but K must be !=1
template<typename T, size_t M, size_t K, size_t N>
FASTOR_INLINE
void _matmul_mkn_square(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {

    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;

    // Get 10 parallel independent chains of accumulators for bigger matrices
    constexpr size_t unrollOuterloop = M >= 64 ? 10UL : (M % 8 == 0 ? 8UL : V::Size);

    // The row index (for a and c) is unrolled using the unrollOuterloop stride. Therefore
    // the last rows may need special treatment if M is not a multiple of unrollOuterloop.
    // M0 is the number of rows that can safely be iterated with a stride of
    // unrollOuterloop.
    constexpr size_t M0 = M / unrollOuterloop * unrollOuterloop;
    for (size_t i = 0; i < M0; i += unrollOuterloop) {
        // The iteration over the column index of b and c uses a stride of V::Size. This
        // enables row-vector loads (from b) and stores (to c). The matrix storage is
        // padded accordingly, ensuring correct bounds and alignment.
        for (size_t j = 0; j < N; j += V::Size) {
            // This temporary variables are used to accumulate the results of the products
            // producing the new values for the c matrix. This variable is necessary
            // because we need a V object for data-parallel accumulation. Storing to c
            // directly stores to scalar objects and thus would drop the ability for
            // data-parallel (SIMD) addition.
            V c_ij[unrollOuterloop];
            for (size_t n = 0; n < unrollOuterloop; ++n) {
                c_ij[n] = a[(i + n)*K]*V(&b[j]);
            }
            for (size_t k = 1; k < K - 1; ++k) {
                for (size_t n = 0; n < unrollOuterloop; ++n) {
                    c_ij[n] += a[(i + n)*K+k] * V(&b[k*N+j]);
                }
            }
            for (size_t n = 0; n < unrollOuterloop; ++n) {
                c_ij[n] += a[(i + n)*K+(K - 1)] * V(&b[(K - 1)*N+j]);
                c_ij[n].store(&c[(i + n)*N+j]);
            }
        }
    }
}

// This is the same implementation as the above case but does not unroll on block sizes and does not require
// the registers to be zeroed out but K must be !=1
template<typename T, size_t M, size_t K, size_t N>
FASTOR_INLINE
void _matmul_mkn_non_square(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {

    // This variant strictly cannot deal outer-product i.e. with K==1

    using V = typename internal::choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;

    // Get 10 parallel independent chains of accumulators for bigger matrices
    constexpr size_t unrollOuterloop = M < V::Size ? 1UL :
        (( M >= 64 && K > 10 && N > V::Size ) ? 10UL : (M % 8 == 0 && N > V::Size ? 8UL : V::Size));
    constexpr bool isPadded = N % V::Size == 0;

    // The row index (for a and c) is unrolled using the unrollOuterloop stride. Therefore
    // the last rows may need special treatment if M is not a multiple of unrollOuterloop.
    // M0 is the number of rows that can safely be iterated with a stride of
    // unrollOuterloop.
    constexpr size_t M0 = M / unrollOuterloop * unrollOuterloop;
    constexpr size_t N0 = N / V::Size * V::Size;

    for (size_t i = 0; i < M0; i += unrollOuterloop) {
        // The iteration over the column index of b and c uses a stride of V::size(). This
        // enables row-vector loads (from b) and stores (to c). The matrix storage is
        // padded accordingly, ensuring correct bounds and alignment.
        size_t j = 0;
        for (; j < N0; j += V::Size) {
            // This temporary variables are used to accumulate the results of the products
            // producing the new values for the c matrix. This variable is necessary
            // because we need a V object for data-parallel accumulation. Storing to c
            // directly stores to scalar objects and thus would drop the ability for
            // data-parallel (SIMD) addition.
            V c_ij[unrollOuterloop];
            for (size_t n = 0; n < unrollOuterloop; ++n) { // correct
                c_ij[n] = a[(i + n)*K]*V(&b[j], isPadded);
            }
            for (size_t k = 1; k < K - 1; ++k) { // correct
                for (size_t n = 0; n < unrollOuterloop; ++n) {
                    c_ij[n] += a[(i + n)*K+k] * V(&b[k*N+j], false);
                }
            }
            for (size_t n = 0; n < unrollOuterloop; ++n) { // correct
                c_ij[n] += a[(i + n)*K+(K - 1)] * V(&b[(K - 1)*N+j], false);
                c_ij[n].store(&c[(i + n)*N+j], isPadded);
            }
        }

        // Remainder N - N0 columns
        for (; j < N; ++j) {
            T c_ij[unrollOuterloop];
            for (size_t n = 0; n < unrollOuterloop; ++n) { // correct
                c_ij[n] = a[(i + n)*K]*b[j];
            }
            for (size_t k = 1; k < K - 1; ++k) { // correct
                for (size_t n = 0; n < unrollOuterloop; ++n) {
                    c_ij[n] += a[(i + n)*K+k] * b[k*N+j];
                }
            }
            for (size_t n = 0; n < unrollOuterloop; ++n) { // correct
                c_ij[n] += a[(i + n)*K+(K - 1)] * b[(K - 1)*N+j];
                c[(i + n)*N+j] = c_ij[n];
            }
        }
    }

    // This final loop treats the remaining M - M0 rows.
    size_t j = 0;
    for (; j < N0; j += V::Size) {
        V c_ij[M-M0];
        for (size_t n = M0; n < M; ++n) { // correct
            c_ij[n - M0] = a[n*K] * V(&b[j], isPadded);
        }
        for (size_t k = 1; k < K - 1; ++k) { // correct
            for (size_t n = M0; n < M; ++n) { // correct
                c_ij[n - M0] += a[n*K+k] * V(&b[k*N+j], false);
            }
        }
        for (size_t n = M0; n < M; ++n) { // correct
            c_ij[n - M0] += a[n*K+(K - 1)] * V(&b[(K - 1)*N+j], false);
            c_ij[n - M0].store(&c[n*N+j], isPadded);
        }
    }

    for (; j < N; ++j) {
        T c_ij[M-M0];
        for (size_t n = M0; n < M; ++n) { // correct
            c_ij[n - M0] = a[n*K] * b[j];
        }
        for (size_t k = 1; k < K - 1; ++k) { // correct
            for (size_t n = M0; n < M; ++n) { // correct
                c_ij[n - M0] += a[n*K+k] * b[k*N+j];
            }
        }
        for (size_t n = M0; n < M; ++n) { // correct
            c_ij[n - M0] += a[n*K+(K - 1)] * b[(K - 1)*N+j];
            c[n*N+j] = c_ij[n - M0];
        }
    }

}
//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

} // end of namespace internal

} // end of namespace Fastor

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------
#include "Fastor/backend/matmul/matmul_mk_smalln.h"
//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------


#endif // MATMUL_KERNELS_H
