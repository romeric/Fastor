#ifndef MATMUL_KERNELS2_H
#define MATMUL_KERNELS2_H


#include "Fastor/commons/commons.h"
#include "Fastor/simd_vector/extintrin.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/meta/tensor_meta.h"


namespace Fastor {

namespace internal {

// TRMM implementation of Fastor - matrix-matrix multiplication when either or both operands are
// lower or upper triangular. The matrices do not need to be square and trapezoidal cases are also
// covered. For big matrices the speed-up is 2X or even better over matmul for when one operand is
// is triangular and nearly 4X for when both operands are triangular.
// For small matrices due to aggressive unrolling for SIMD the matrices cannot be exactly traversed in
// within their triangular part(s) and a bit the non-triangular part(s) need(s) to be loaded as well,
// hence the performance may not be exactly 2X over the the general matmul case


// The functions here are exact replica of those in matmul_kernels.h and will be eventually
// merged together as these variants have no associated overhead for the general case for matmul
//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------
/*

For triangular matmul only the iteration span of K is modified
using the following logic [lt = lower_tri, ut=upper_tri]

// if lhs == lt
const size_t kfirst = 0;
const size_t klast  = min(i+1,K);  // or min(i+unrollOuterloop,K);

// if lhs == ut
const size_t kfirst = i;
const size_t klast  = K;

// if rhs == lt
const size_t kfirst = j;
const size_t klast  = K;

// if rhs == ut
const size_t kfirst = 0;
const size_t klast  = min(j+1,K);   // or min(j+unrollOuterloop,K);


// both lower
const size_t kfirst = j;
const size_t klast  = min(i+1,K);    // or min(i+unrollOuterloop,K);

// if lhs == lt && rhs == ut
const size_t kfirst = 0;
const size_t klast  = min(min(i+1,j+1),K);  // or min(min(i+unrollOuterloop,j+unrollInnerloop),K);

// if lhs == ut && rhs == lt
const size_t kfirst = max(i,j);
const size_t klast  = K;

// if both upper
const size_t kfirst = i;
const size_t klast  = min(j+1,K);     // or min(j+unrollInnerloop,K);

*/

template<typename T> constexpr FASTOR_INLINE T __min(const T a, const T b) {return a < b ? a : b;}
template<typename T> constexpr FASTOR_INLINE T __max(const T a, const T b) {return a > b ? a : b;}

template<typename T, T K, T unrollOuterloop=1,T unrollInnerloop=1, typename LhsType = matrix_type::general, typename RhsType = matrix_type::general>
constexpr FASTOR_INLINE T find_kfirst(const T i, const T j) {
    return is_same_v_<LhsType,matrix_type::lower_tri> || is_same_v_<LhsType,matrix_type::general> ?
        ( is_same_v_<RhsType,matrix_type::lower_tri> ? j : 0UL ) :
            (is_same_v_<LhsType,matrix_type::upper_tri> ? ( is_same_v_<RhsType,matrix_type::lower_tri> ? __max(i,j) : i ) : 0UL );
}
template<typename T, T K, T unrollOuterloop=1,T unrollInnerloop=1, typename LhsType = matrix_type::general, typename RhsType = matrix_type::general>
constexpr FASTOR_INLINE T find_klast(const T i, const T j) {
    return is_same_v_<LhsType,matrix_type::lower_tri> ?
        ( is_same_v_<RhsType,matrix_type::upper_tri> ? __min(__min(i+unrollOuterloop,j+unrollInnerloop),K) : __min(i+unrollOuterloop,K) ) :
            (is_same_v_<LhsType,matrix_type::upper_tri> || (is_same_v_<LhsType,matrix_type::general>) ?
                ( is_same_v_<RhsType,matrix_type::upper_tri> ? __min(j+unrollInnerloop,K) : K ) : K );
}


//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

// A set of helper functions for the inner blocks of matmul. Almost all compilers (GCC/CLang/Intel)
// unroll the inner-most loop (on unrollOuterloop)
//-----------------------------------------------------------------------------------------------------------
template<typename T, typename V, size_t M, size_t K, size_t N, size_t unrollOuterloop, size_t numSIMDRows, size_t numSIMDCols,
    typename LhsType = matrix_type::general, typename RhsType = matrix_type::general,
    typename std::enable_if<numSIMDCols==1,bool>::type = false>
FASTOR_INLINE
void interior_block_tmatmul_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j) {

    const size_t kfirst = find_kfirst<size_t,K,unrollOuterloop*numSIMDRows,numSIMDCols*V::Size,LhsType,RhsType>(i,j);
    const size_t klast  = find_klast <size_t,K,unrollOuterloop*numSIMDRows,numSIMDCols*V::Size,LhsType,RhsType>(i,j);

    for (size_t ii = 0; ii < numSIMDRows; ++ii) {

        V c_ij[unrollOuterloop*numSIMDCols];
        // Loop over columns of a (rows of b)
        for (size_t k = kfirst; k < klast; ++k) {

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
    typename LhsType = matrix_type::general, typename RhsType = matrix_type::general,
    typename std::enable_if<numSIMDCols==2,bool>::type = false>
FASTOR_INLINE
void interior_block_tmatmul_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j) {

    const size_t kfirst = find_kfirst<size_t,K,unrollOuterloop*numSIMDRows,numSIMDCols*V::Size,LhsType,RhsType>(i,j);
    const size_t klast  = find_klast <size_t,K,unrollOuterloop*numSIMDRows,numSIMDCols*V::Size,LhsType,RhsType>(i,j);

    for (size_t ii = 0; ii < numSIMDRows; ++ii) {

        V c_ij[unrollOuterloop*numSIMDCols];
        // Loop over columns of a (rows of b)
        for (size_t k = kfirst; k < klast; ++k) {

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
    typename LhsType = matrix_type::general, typename RhsType = matrix_type::general,
    typename std::enable_if<numSIMDCols==3,bool>::type = false>
FASTOR_INLINE
void interior_block_tmatmul_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j) {

    const size_t kfirst = find_kfirst<size_t,K,unrollOuterloop*numSIMDRows,numSIMDCols*V::Size,LhsType,RhsType>(i,j);
    const size_t klast  = find_klast <size_t,K,unrollOuterloop*numSIMDRows,numSIMDCols*V::Size,LhsType,RhsType>(i,j);

    for (size_t ii = 0; ii < numSIMDRows; ++ii) {

        V c_ij[unrollOuterloop*numSIMDCols];
        // Loop over columns of a (rows of b)
        for (size_t k = kfirst; k < klast; ++k) {

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
    typename LhsType = matrix_type::general, typename RhsType = matrix_type::general,
    typename std::enable_if<numSIMDCols==4,bool>::type = false>
FASTOR_INLINE
void interior_block_tmatmul_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j) {

    const size_t kfirst = find_kfirst<size_t,K,unrollOuterloop*numSIMDRows,numSIMDCols*V::Size,LhsType,RhsType>(i,j);
    const size_t klast  = find_klast <size_t,K,unrollOuterloop*numSIMDRows,numSIMDCols*V::Size,LhsType,RhsType>(i,j);

    for (size_t ii = 0; ii < numSIMDRows; ++ii) {

        V c_ij[unrollOuterloop*numSIMDCols];
        // Loop over columns of a (rows of b)
        for (size_t k = kfirst; k < klast; ++k) {

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
    typename LhsType = matrix_type::general, typename RhsType = matrix_type::general,
    typename std::enable_if<numSIMDCols==5,bool>::type = false>
FASTOR_INLINE
void interior_block_tmatmul_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j) {

    const size_t kfirst = find_kfirst<size_t,K,unrollOuterloop*numSIMDRows,numSIMDCols*V::Size,LhsType,RhsType>(i,j);
    const size_t klast  = find_klast <size_t,K,unrollOuterloop*numSIMDRows,numSIMDCols*V::Size,LhsType,RhsType>(i,j);

    for (size_t ii = 0; ii < numSIMDRows; ++ii) {

        V c_ij[unrollOuterloop*numSIMDCols];
        // Loop over columns of a (rows of b)
        for (size_t k = kfirst; k < klast; ++k) {

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
    typename LhsType = matrix_type::general, typename RhsType = matrix_type::general,
    typename std::enable_if<numSIMDCols==1,bool>::type = false>
FASTOR_INLINE
void interior_block_tmatmul_scalar_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j) {

    const size_t kfirst = find_kfirst<size_t,K,unrollOuterloop*numSIMDRows,numSIMDCols,LhsType,RhsType>(i,j);
    const size_t klast  = find_klast <size_t,K,unrollOuterloop*numSIMDRows,numSIMDCols,LhsType,RhsType>(i,j);

    for (size_t ii = 0; ii < numSIMDRows; ++ii) {

        T c_ij[unrollOuterloop*numSIMDCols] = {};
        // Loop over columns of a (rows of b)
        for (size_t k = kfirst; k < klast; ++k) {
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
    typename LhsType = matrix_type::general, typename RhsType = matrix_type::general,
    typename std::enable_if<numSIMDCols==1,bool>::type = false>
FASTOR_INLINE
void interior_block_tmatmul_mask_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j, const int (&maska)[V::Size]) {

    const size_t kfirst = find_kfirst<size_t,K,unrollOuterloop*numSIMDRows,numSIMDCols*V::Size,LhsType,RhsType>(i,j);
    const size_t klast  = find_klast <size_t,K,unrollOuterloop*numSIMDRows,numSIMDCols*V::Size,LhsType,RhsType>(i,j);

    for (size_t ii = 0; ii < numSIMDRows; ++ii) {

        V c_ij[unrollOuterloop*numSIMDCols];
        // Loop over columns of a (rows of b)
        for (size_t k = kfirst; k < klast; ++k) {

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
    typename LhsType = matrix_type::general, typename RhsType = matrix_type::general,
    typename std::enable_if<numSIMDCols==1,bool>::type = false>
FASTOR_INLINE
void interior_block_tmatmul_mask_impl(
    const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c,
    const size_t i, const size_t j, const MaskT mask) {

    V bmm0;
    const size_t kfirst = find_kfirst<size_t,K,unrollOuterloop*numSIMDRows,numSIMDCols*V::Size,LhsType,RhsType>(i,j);
    const size_t klast  = find_klast <size_t,K,unrollOuterloop*numSIMDRows,numSIMDCols*V::Size,LhsType,RhsType>(i,j);

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
// This is the base implementation of triangular matrix-matrix multiplication for all 2D tensors and
// higher order tensor products that can be expressed as trmm
// The function uses two level unrolling one based on block sizes and one based on register widths
// with any remainder left treated in a scalar fashion
template<typename T, size_t M, size_t K, size_t N, typename LhsType = matrix_type::general, typename RhsType = matrix_type::general>
FASTOR_INLINE
void _tmatmul_base(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {

    using V = choose_best_simd_t<SIMDVector<T,DEFAULT_ABI>,N>;

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
            interior_block_tmatmul_impl<T,V,M,K,N,unrollOuterloop,numSIMDRows,numSIMDCols,LhsType,RhsType>(a,b,c,i,j);
        }

        // Remaining N - N0 columns
        for (; j < N1; j += V::Size) {
            interior_block_tmatmul_impl<T,V,M,K,N,unrollOuterloop,numSIMDRows,1,LhsType,RhsType>(a,b,c,i,j);
        }

        // Remaining N - N1 columns
        for (; j < N; ++j) {
            interior_block_tmatmul_scalar_impl<T,V,M,K,N,unrollOuterloop,numSIMDRows,1,LhsType,RhsType>(a,b,c,i,j);
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
            interior_block_tmatmul_impl<T,V,M,K,N,unrollOuterloop,1,numSIMDCols,LhsType,RhsType>(a,b,c,i,j);
        }

        // Remaining N - N0 columns
        for (; j < N1; j += V::Size) {

            const size_t kfirst = find_kfirst<size_t,K,unrollOuterloop,V::Size,LhsType,RhsType>(i,j);
            const size_t klast  = find_klast <size_t,K,unrollOuterloop,V::Size,LhsType,RhsType>(i,j);

            V c_ij[unrollOuterloop];
            for (size_t k = kfirst; k < klast; ++k) {
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

            const size_t kfirst = find_kfirst<size_t,K,unrollOuterloop,1,LhsType,RhsType>(i,j);
            const size_t klast  = find_klast <size_t,K,unrollOuterloop,1,LhsType,RhsType>(i,j);

            T c_ij[unrollOuterloop] = {};
            for (size_t k = kfirst; k < K; ++k) {
                for (size_t n = 0; n < unrollOuterloop; ++n) {
                    c_ij[n] += a[(i + n)*K+k] * b[k*N+j];
                }
            }
            for (size_t n = 0; n < unrollOuterloop; ++n) {
                c[(i + n)*N+j] = c_ij[n];
            }
        }
    }

    // Now treat the remaining M-M1 rows - here the klast - kfirst range is not used
    // so the implementation is exactly the same as matmul_base
    FASTOR_IF_CONSTEXPR (M-M1 > 0) {
        // Hack to get around zero length array issue
        constexpr size_t MM1 = M-M1 != 0 ? M-M1 : 1;
        size_t j = 0;
        for (; j < N0; j += unrollInnerBlock) {
            // If MM1==0 the function never gets invoked anyway
            interior_block_tmatmul_impl<T,V,M,K,N,MM1,1,numSIMDCols>(a,b,c,i,j);
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
// This the base implementation of triangular matrix-matrix multiplication for all 2D tensors and
// higher order tensor products that can be expressed as trmm
// The function uses two level unrolling one based on block sizes and one based on register widths
// with any remainder left treated in vector mode with masked and conditional load/stores.
// Note that conditional load/store requires at least AVX intrinsics
template<typename T, size_t M, size_t K, size_t N, typename LhsType = matrix_type::general, typename RhsType = matrix_type::general>
FASTOR_INLINE
void _tmatmul_base_masked(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {

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
            interior_block_tmatmul_impl<T,V,M,K,N,unrollOuterloop,numSIMDRows,numSIMDCols>(a,b,c,i,j);
        }

        // Remaining N - N0 columns
        for (; j < N1; j += V::Size) {
            interior_block_tmatmul_impl<T,V,M,K,N,unrollOuterloop,numSIMDRows,1>(a,b,c,i,j);
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
            interior_block_tmatmul_impl<T,V,M,K,N,unrollOuterloop,1,numSIMDCols>(a,b,c,i,j);
        }

        // Remaining N - N0 columns
        for (; j < N1; j += V::Size) {

            const size_t kfirst = find_kfirst<size_t,K,unrollOuterloop,V::Size,LhsType,RhsType>(i,j);
            const size_t klast  = find_klast <size_t,K,unrollOuterloop,V::Size,LhsType,RhsType>(i,j);

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

            const size_t kfirst = find_kfirst<size_t,K,unrollOuterloop,V::Size,LhsType,RhsType>(i,j);
            const size_t klast  = find_klast <size_t,K,unrollOuterloop,V::Size,LhsType,RhsType>(i,j);

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

    // Now treat the remaining M-M1 rows - here the klast - kfirst range is not used
    // so the implementation is exactly the same as matmul_base
    FASTOR_IF_CONSTEXPR (M-M1 > 0) {
        // Hack to get around zero length array issue
        constexpr size_t MM1 = M-M1 != 0 ? M-M1 : 1;
        size_t j = 0;
        for (; j < N0; j += unrollInnerBlock) {
            // If MM1==0 the function never gets invoked anyway
            interior_block_tmatmul_impl<T,V,M,K,N,MM1,1,numSIMDCols>(a,b,c,i,j);
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
//-----------------------------------------------------------------------------------------------------------

} // end of namespace internal


//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------
// Backend tmatmul function
template<typename T, size_t M, size_t K, size_t N, typename LhsType = matrix_type::general, typename RhsType = matrix_type::general>
FASTOR_INLINE
void _tmatmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    using nativeV = SIMDVector<T,DEFAULT_ABI>;
    using V = choose_best_simd_t<nativeV,N>;

    // Use specialised kernels
#if defined(FASTOR_AVX2_IMPL) || defined(FASTOR_HAS_AVX512_MASKS)
    FASTOR_IF_CONSTEXPR(N % V::Size <= 1UL) {
        internal::_tmatmul_base<T,M,K,N,LhsType,RhsType>(a,b,out);
        return;
    }
    else {
        internal::_tmatmul_base_masked<T,M,K,N,LhsType,RhsType>(a,b,out);
        return;
    }
#else
    internal::_tmatmul_base<T,M,K,N,LhsType,RhsType>(a,b,out);
        return;
#endif
}
//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------
} // end of namespace Fastor


#endif // MATMUL_KERNELS_H
