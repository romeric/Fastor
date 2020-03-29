#ifndef MATMUL_KERNELS_H
#define MATMUL_KERNELS_H


#include "Fastor/commons/commons.h"
#include "Fastor/extended_intrinsics/extintrin.h"
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
template<typename T, size_t M, size_t K, size_t N, size_t unrollOuterloop, size_t numSIMDCols,
    typename std::enable_if<numSIMDCols==1,bool>::type = false>
FASTOR_INLINE
void interior_block_matmul_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j, const size_t ii) {

    using V = SIMDVector<T,DEFAULT_ABI>;

    V c_ij[unrollOuterloop*numSIMDCols];
    // Loop over columns of a (rows of b)
    for (size_t k = 0; k < K; ++k) {
        for (size_t n = 0; n < unrollOuterloop; ++n) {
            const V bmm0(&b[k*N+j],false);

            const V amm0 = a[(i+ii*unrollOuterloop+n)*K+k];

            c_ij[n]                    = fmadd(amm0,bmm0,c_ij[n]);

            c_ij[n].store(&c[(i+ii*unrollOuterloop+n)*N+j],false);
        }
    }
}

template<typename T, size_t M, size_t K, size_t N, size_t unrollOuterloop, size_t numSIMDCols,
    typename std::enable_if<numSIMDCols==2,bool>::type = false>
FASTOR_INLINE
void interior_block_matmul_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j, const size_t ii) {

    using V = SIMDVector<T,DEFAULT_ABI>;

    V c_ij[unrollOuterloop*numSIMDCols];
    // Loop over columns of a (rows of b)
    for (size_t k = 0; k < K; ++k) {
        for (size_t n = 0; n < unrollOuterloop; ++n) {
            const V bmm0(&b[k*N+j],false);
            const V bmm1(&b[k*N+j+V::Size],false);

            const V amm0 = a[(i+ii*unrollOuterloop+n)*K+k];

            c_ij[n]                    = fmadd(amm0,bmm0,c_ij[n]);
            c_ij[n+unrollOuterloop]    = fmadd(amm0,bmm1,c_ij[n+unrollOuterloop]);

            c_ij[n].store(&c[(i+ii*unrollOuterloop+n)*N+j],false);
            c_ij[n+unrollOuterloop].store(&c[(i+ii*unrollOuterloop+n)*N+j+V::Size],false);
        }
    }
}


template<typename T, size_t M, size_t K, size_t N, size_t unrollOuterloop, size_t numSIMDCols,
    typename std::enable_if<numSIMDCols==3,bool>::type = false>
FASTOR_INLINE
void interior_block_matmul_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j, const size_t ii) {

    using V = SIMDVector<T,DEFAULT_ABI>;

    V c_ij[unrollOuterloop*numSIMDCols];
    // Loop over columns of a (rows of b)
    for (size_t k = 0; k < K; ++k) {
        for (size_t n = 0; n < unrollOuterloop; ++n) {
            const V bmm0(&b[k*N+j],false);
            const V bmm1(&b[k*N+j+V::Size],false);
            const V bmm2(&b[k*N+j+2*V::Size],false);

            const V amm0 = a[(i+ii*unrollOuterloop+n)*K+k];

            c_ij[n]                    = fmadd(amm0,bmm0,c_ij[n]);
            c_ij[n+unrollOuterloop]    = fmadd(amm0,bmm1,c_ij[n+unrollOuterloop]);
            c_ij[n+2*unrollOuterloop]  = fmadd(amm0,bmm2,c_ij[n+2*unrollOuterloop]);

            c_ij[n].store(&c[(i+ii*unrollOuterloop+n)*N+j],false);
            c_ij[n+unrollOuterloop].store(&c[(i+ii*unrollOuterloop+n)*N+j+V::Size],false);
            c_ij[n+2*unrollOuterloop].store(&c[(i+ii*unrollOuterloop+n)*N+j+2*V::Size],false);
        }
    }
}


template<typename T, size_t M, size_t K, size_t N, size_t unrollOuterloop, size_t numSIMDCols,
    typename std::enable_if<numSIMDCols==4,bool>::type = false>
FASTOR_INLINE
void interior_block_matmul_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j, const size_t ii) {

    using V = SIMDVector<T,DEFAULT_ABI>;

    V c_ij[unrollOuterloop*numSIMDCols];
    // Loop over columns of a (rows of b)
    for (size_t k = 0; k < K; ++k) {
        for (size_t n = 0; n < unrollOuterloop; ++n) {
            const V bmm0(&b[k*N+j],false);
            const V bmm1(&b[k*N+j+V::Size],false);
            const V bmm2(&b[k*N+j+2*V::Size],false);
            const V bmm3(&b[k*N+j+3*V::Size],false);

            const V amm0 = a[(i+ii*unrollOuterloop+n)*K+k];

            c_ij[n]                    = fmadd(amm0,bmm0,c_ij[n]);
            c_ij[n+unrollOuterloop]    = fmadd(amm0,bmm1,c_ij[n+unrollOuterloop]);
            c_ij[n+2*unrollOuterloop]  = fmadd(amm0,bmm2,c_ij[n+2*unrollOuterloop]);
            c_ij[n+3*unrollOuterloop]  = fmadd(amm0,bmm3,c_ij[n+3*unrollOuterloop]);

            c_ij[n].store(&c[(i+ii*unrollOuterloop+n)*N+j],false);
            c_ij[n+unrollOuterloop].store(&c[(i+ii*unrollOuterloop+n)*N+j+V::Size],false);
            c_ij[n+2*unrollOuterloop].store(&c[(i+ii*unrollOuterloop+n)*N+j+2*V::Size],false);
            c_ij[n+3*unrollOuterloop].store(&c[(i+ii*unrollOuterloop+n)*N+j+3*V::Size],false);
        }
    }
}


template<typename T, size_t M, size_t K, size_t N, size_t unrollOuterloop, size_t numSIMDCols,
    typename std::enable_if<numSIMDCols==1,bool>::type = false>
FASTOR_INLINE
void interior_block_matmul_scalar_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j, const size_t ii) {

    T c_ij[unrollOuterloop*numSIMDCols] = {};
    // Loop over columns of a (rows of b)
    for (size_t k = 0; k < K; ++k) {
        for (size_t n = 0; n < unrollOuterloop; ++n) {
            const T bmm0(b[k*N+j]);

            const T amm0 = a[(i+ii*unrollOuterloop+n)*K+k];

            c_ij[n]                                  += amm0*bmm0;

            c[(i+ii*unrollOuterloop+n)*N+j]           = c_ij[n];
        }
    }
}
//-----------------------------------------------------------------------------------------------------------



//-----------------------------------------------------------------------------------------------------------
// This the base implementation of matrix-matrix multiplication in Fastor for all 2D tensors and
// high order tensor products that can be expressed as gemm
// The function uses two level unrolling one bases on block sizes and one based on register widths
template<typename T, size_t M, size_t K, size_t N>
FASTOR_INLINE
void _matmul_base(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {

    using V = SIMDVector<T,DEFAULT_ABI>;

    // This parameter can be adjusted and does not need to be 4UL/8UL etc
    constexpr size_t unrollOuterloop = M % 5UL == 0 ? 5UL : 4UL;

    // Unroll the rows of (a and c) (M) by [numSIMDRows * V::Size]
    constexpr size_t numSIMDRows = M % (unrollOuterloop * 3UL) == 0 ? 3UL : 2UL;

    // Unroll the columns of (b and c) (N) by [numSIMDCols * V::Size]
    // constexpr size_t numSIMDCols = N % (V::Size * 3UL) == 0 ? 3UL : 2UL;
    constexpr size_t numSIMDCols = (N % (V::Size * 3UL) == 0 && N > 24UL) ? 3UL : 2UL;

    // The goal is to get 10 parallel independent chains of accumulators
    // to saturate the pipeline by having a completely unrolled block of
    // [(unrollOuterloop) * (numSIMDCols)] at a time. A minimum value of
    // unrollOuterloop=4 ensures a minimum 8 independent parallel chains
    // while a maximum of 12.

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
            for (size_t ii = 0; ii < numSIMDRows; ++ii) {
                interior_block_matmul_impl<T,M,K,N,unrollOuterloop,numSIMDCols>(a,b,c,i,j,ii);
            }
        }

        // Remaining N - N0 columns
        for (; j < N1; j += V::Size) {
            for (size_t ii = 0; ii < numSIMDRows; ++ii) {
                interior_block_matmul_impl<T,M,K,N,unrollOuterloop,1>(a,b,c,i,j,ii);
            }
        }

        // Remaining N - N1 columns
        for (; j < N; ++j) {
            for (size_t ii = 0; ii < numSIMDRows; ++ii) {
                interior_block_matmul_scalar_impl<T,M,K,N,unrollOuterloop,1>(a,b,c,i,j,ii);
            }
        }

    }

    // The remaining M-M0 rows are now unrolled yet again by unrollOuterloop.
    // This is necessary as for small sizes the earlier block loop may not be
    // triggered if the size of the block is bigger than the number of rows of
    // (a and c) i.e. M
    constexpr size_t M1 = (M / unrollOuterloop * unrollOuterloop);
    for (; i < M1; i += unrollOuterloop) {
        size_t j = 0;
        for (; j < N1; j += V::Size) {
            V c_ij[unrollOuterloop];
            for (size_t k = 0; k < K; ++k) {
                for (size_t n = 0; n < unrollOuterloop; ++n) {
                    c_ij[n] += a[(i + n)*K+k] * V(&b[k*N+j],false);
                    c_ij[n].store(&c[(i + n)*N+j],false);
                }
            }
        }

        // Remaining N - N1 columns
        for (; j < N; ++j) {
            T c_ij[unrollOuterloop] = {};
            for (size_t k = 0; k < K; ++k) {
                for (size_t n = 0; n < unrollOuterloop; ++n) {
                    c_ij[n] += a[(i + n)*K+k] * b[k*N+j];
                    c[(i + n)*N+j] = c_ij[n];
                }
            }
        }
    }

    // Now treat the remaining M-M1 rows
    size_t j = 0;
    for (; j < N1; j += V::Size) {
        V c_ij[M-M1];
        for (size_t k = 0; k < K; ++k) {
            for (size_t n = M1; n < M; ++n) {
                c_ij[n-M1] += a[n*K+k] * V(&b[k*N+j],false);
                c_ij[n-M1].store(&c[n*N+j],false);
            }
        }
    }

    for (; j < N; ++j) {
        T c_ij[M-M1] = {};
        for (size_t k = 0; k < K; ++k) {
            for (size_t n = M1; n < M; ++n) {
                c_ij[n-M1] += a[n*K+k] * b[k*N+j];
                c[n*N+j] = c_ij[n-M1];
            }
        }
    }
}

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------































// Other used/unused variants and slightly older implementations
//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------
// This is the same implementation as the above case but does not unroll on block sizes and does not require
// the registers to be zeroed out but K must be !=1
template<typename T, size_t M, size_t K, size_t N>
FASTOR_INLINE
void _matmul_mkn_square(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {

    using V = SIMDVector<T,DEFAULT_ABI>;
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

    using V = SIMDVector<T,DEFAULT_ABI>;
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




// This is a generic version of matmul based on 2k2/3k3/4k4 variants. It builds on the same philosophy
// and in fact for very large Ks it is as performant as libxsmm and Eigen
//-----------------------------------------------------------------------------------------------------------
template<typename T, size_t M, size_t K, size_t N>
void _matmul_mKn(const T * FASTOR_RESTRICT a_data, const T * FASTOR_RESTRICT b_data, T * FASTOR_RESTRICT out_data) {

    using V = SIMDVector<T,DEFAULT_ABI>;
    // constexpr size_t UNROLL_LENGTH = 4UL;
    constexpr size_t UNROLL_LENGTH = 5UL;
    constexpr size_t ROUND_M = (M / UNROLL_LENGTH) * UNROLL_LENGTH;
    constexpr size_t ROUND_N = (N / V::Size) * V::Size;
    std::array<V, M > ymm_a;
    std::array<std::array<V, M >, N / V::Size > ymm_o;
    std::array<std::array<T, M >, N % V::Size > t_o = {};
    for (size_t i=0; i<K; ++i) {
        size_t k=0;
        size_t counter = 0;
        for (; k<ROUND_N; k+=V::Size) {
            const V ymm0 = V(&b_data[i*N+k],false);
            size_t j=0;
            for (; j<ROUND_M; j+=UNROLL_LENGTH) {
                ymm_a[j+0] = V(a_data[j*K+i]);
                ymm_a[j+1] = V(a_data[(j+1)*K+i]);
                ymm_a[j+2] = V(a_data[(j+2)*K+i]);
                ymm_a[j+3] = V(a_data[(j+3)*K+i]);
                ymm_a[j+4] = V(a_data[(j+4)*K+i]);

                ymm_o[counter][j+0] = fmadd(ymm0,ymm_a[j+0],ymm_o[counter][j+0]);
                ymm_o[counter][j+1] = fmadd(ymm0,ymm_a[j+1],ymm_o[counter][j+1]);
                ymm_o[counter][j+2] = fmadd(ymm0,ymm_a[j+2],ymm_o[counter][j+2]);
                ymm_o[counter][j+3] = fmadd(ymm0,ymm_a[j+3],ymm_o[counter][j+3]);
                ymm_o[counter][j+4] = fmadd(ymm0,ymm_a[j+4],ymm_o[counter][j+4]);
            }

            for (; j<M; ++j) {
                const V ymm1 = V(a_data[j*K+i]);
                ymm_o[counter][j] = fmadd(ymm0,ymm1,ymm_o[counter][j]);
            }
            counter++;
        }

        counter = 0;
        for (; k<N; ++k) {
            const T t0 = b_data[i*N+k];
            size_t j=0;
            for (; j<ROUND_M; j+=UNROLL_LENGTH) {
                const T t1 = a_data[j*K+i];
                const T t2 = a_data[(j+1)*K+i];
                const T t3 = a_data[(j+2)*K+i];
                const T t4 = a_data[(j+3)*K+i];
                const T t5 = a_data[(j+4)*K+i];

                t_o[counter][j+0] += t0*t1;
                t_o[counter][j+1] += t0*t2;
                t_o[counter][j+2] += t0*t3;
                t_o[counter][j+3] += t0*t4;
                t_o[counter][j+4] += t0*t5;
            }
            for (; j<M; ++j) {
                const T t1 = a_data[j*K+i];
                t_o[counter][j] += t0*t1;
            }
            counter++;
        }
    }


    for (size_t k=0; k< N / V::Size; ++k) {
        for (size_t j=0; j<M; ++j) {
            ymm_o[k][j].store(&out_data[j*N+k*V::Size],false);
        }
    }

    for (size_t k=0; k< N % V::Size; ++k) {
        for (size_t j=0; j<M; ++j) {
            out_data[j*N+ROUND_N+k] = t_o[k][j];
        }
    }
}

//-----------------------------------------------------------------------------------------------------------

} // end of namespace internal

} // end of namespace Fastor


#endif // MATMUL_KERNELS_H
