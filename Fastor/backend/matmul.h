#ifndef MATMUL_H
#define MATMUL_H

#include "Fastor/commons/commons.h"
#include "Fastor/extended_intrinsics/extintrin.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/meta/tensor_meta.h"

#ifdef FASTOR_USE_LIBXSMM
#include "libxsmm_backend.h"
#endif

namespace Fastor {


// For square matrices with N == multiple of SIMD register width
//-----------------------------------------------------------------------------------------------------------
#ifndef FASTOR_USE_LIBXSMM
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<M==N && M==K && N % SIMDVector<T,DEFAULT_ABI>::Size ==0,bool>::type = 0>
#else
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<M==N && M==K && N % SIMDVector<T,DEFAULT_ABI>::Size ==0
         && is_less_equal<N,BLAS_SWITCH_MATRIX_SIZE_S>::value,bool>::type = 0>
#endif
FASTOR_INLINE
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {

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
#ifdef FASTOR_USE_LIBXSMM
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<M==N && M==K && N % SIMDVector<T,DEFAULT_ABI>::Size ==0
         && is_greater<N,BLAS_SWITCH_MATRIX_SIZE_S>::value,bool>::type = 0>
FASTOR_INLINE
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {
    blas::matmul_libxsmm<T,M,K,N>(a,b,c);
}
#endif
//-----------------------------------------------------------------------------------------------------------


// For general non-square matrices with arbitrary alignment/padding excluding outer product (K==1)
//-----------------------------------------------------------------------------------------------------------
template<typename T, size_t M, size_t K, size_t N>
FASTOR_INLINE
void _matmul_mkn(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {

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
    constexpr size_t i1 = N / V::Size * V::Size;

    for (size_t i = 0; i < M0; i += unrollOuterloop) {
        // The iteration over the column index of b and c uses a stride of V::size(). This
        // enables row-vector loads (from b) and stores (to c). The matrix storage is
        // padded accordingly, ensuring correct bounds and alignment.
        size_t j = 0;
        for (; j < i1; j += V::Size) {
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

        // Remainder N - i1 columns
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
    for (; j < i1; j += V::Size) {
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
// and outerperforms the main version in almost all cases if K is large enough and M and N are small.
// In fact for large Ks it is as performant as libxsmm and Eigen
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



#ifdef __SSE4_2__

// (2xk) x (kx2) matrices
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<(M!=K && M==N && M==2 && std::is_same<T,double>::value),bool>::type = 0>
FASTOR_INLINE
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    __m128d out_row0 = ZEROPD;
    __m128d out_row1 = ZEROPD;

    for (size_t i=0; i<K; ++i) {
        __m128d brow = _mm_loadu_pd(&b[i*2]);
#ifndef __FMA__
        // row 0
        __m128d a_vec0 = _mm_set1_pd(a[i]);
        out_row0 = _mm_add_pd(out_row0,_mm_mul_pd(a_vec0,brow));
        // row 1
        __m128d a_vec1 = _mm_set1_pd(a[K+i]);
        out_row1 = _mm_add_pd(out_row1,_mm_mul_pd(a_vec1,brow));
#else
        // row 0
        __m128d a_vec0 = _mm_set1_pd(a[i]);
        out_row0 = _mm_fmadd_pd(a_vec0,brow,out_row0);
        // row 1
        __m128d a_vec1 = _mm_set1_pd(a[K+i]);
        out_row1 = _mm_fmadd_pd(a_vec1,brow,out_row1);
#endif
    }
    _mm_store_pd(out,out_row0);
    _mm_storeu_pd(out+2,out_row1);
}


// (2xk) x (kx2) matrices
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<(M!=K && M==N && M==2 && std::is_same<T,float>::value),bool>::type = 0>
FASTOR_INLINE
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    __m128 out_row0 = ZEROPS;
    __m128 out_row1 = ZEROPS;

    for (size_t i=0; i<K; i++) {
        __m128 brow = _mm_loadu_ps(&b[i*2]);
#ifndef __FMA__
        // row 0
        __m128 a_vec0 = _mm_set1_ps(a[i]);
        out_row0 = _mm_add_ps(out_row0,_mm_mul_ps(a_vec0,brow));
        // row 1
        __m128 a_vec1 = _mm_set1_ps(a[K+i]);
        out_row1 = _mm_add_ps(out_row1,_mm_mul_ps(a_vec1,brow));
#else
        // row 0
        __m128 a_vec0 = _mm_set1_ps(a[i]);
        out_row0 = _mm_fmadd_ps(a_vec0,brow,out_row0);
        // row 1
        __m128 a_vec1 = _mm_set1_ps(a[K+i]);
        out_row1 = _mm_fmadd_ps(a_vec1,brow,out_row1);
#endif
    }
    _mm_store_ps(out,_mm_shuffle_ps(out_row0,out_row1,_MM_SHUFFLE(1,0,1,0)));
}

#else
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<M!=K && M==N && M==2,bool>::type = 0>
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {
    _matmul_mKn<T,M,K,N>(a,b,out);
}
#endif

#ifdef __AVX__

// (3xk) x (kx3) matrices
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<(M!=K && M==N && M==3 && std::is_same<T,double>::value),bool>::type = 0>
FASTOR_INLINE
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    __m256d out_row0 = VZEROPD;
    __m256d out_row1 = VZEROPD;
    __m256d out_row2 = VZEROPD;

    for (size_t i=0; i<K; ++i) {
        __m256d brow = _mm256_loadul3_pd(&b[i*3]);
#ifndef __FMA__
        // row 0
        __m256d a_vec0 = _mm256_set1_pd(a[i]);
        out_row0 = _mm256_add_pd(out_row0,_mm256_mul_pd(a_vec0,brow));
        // row 1
        __m256d a_vec1 = _mm256_set1_pd(a[K+i]);
        out_row1 = _mm256_add_pd(out_row1,_mm256_mul_pd(a_vec1,brow));
        // row 2
        __m256d a_vec2 = _mm256_set1_pd(a[2*K+i]);
        out_row2 = _mm256_add_pd(out_row2,_mm256_mul_pd(a_vec2,brow));
#else
        // row 0
        __m256d a_vec0 = _mm256_set1_pd(a[i]);
        out_row0 = _mm256_fmadd_pd(a_vec0,brow,out_row0);
        // row 1
        __m256d a_vec1 = _mm256_set1_pd(a[K+i]);
        out_row1 = _mm256_fmadd_pd(a_vec1,brow,out_row1);
        // row 2
        __m256d a_vec2 = _mm256_set1_pd(a[2*K+i]);
        out_row2 = _mm256_fmadd_pd(a_vec2,brow,out_row2);
#endif
    }
    _mm256_store_pd(out,out_row0);
    _mm256_storeu_pd(out+3,out_row1);
    // Causees crash for the last 8byte
    // _mm256_storeu_pd(out+6,out_row2);

    _mm_storeu_pd(out+6, _mm256_castpd256_pd128(out_row2));
    _mm_store_sd (out+8, _mm256_extractf128_pd(out_row2, 1));
}

#else
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<M!=K && M==N && M==3 && std::is_same<T,double>::value,bool>::type = 0>
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {
    _matmul_mKn<T,M,K,N>(a,b,out);
}
#endif

#ifdef __SSE4_2__

// (3xk) x (kx3) matrices
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<(M!=K && M==N && M==3 && std::is_same<T,float>::value),bool>::type = 0>
FASTOR_INLINE
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    __m128 out_row0 = ZEROPS;
    __m128 out_row1 = ZEROPS;
    __m128 out_row2 = ZEROPS;

    for (size_t i=0; i<K; ++i) {
        __m128 brow = _mm_loadul3_ps(&b[i*3]);
#ifndef __FMA__
        // row 0
        __m128 a_vec0 = _mm_set1_ps(a[i]);
        out_row0 = _mm_add_ps(out_row0,_mm_mul_ps(a_vec0,brow));
        // row 1
        __m128 a_vec1 = _mm_set1_ps(a[K+i]);
        out_row1 = _mm_add_ps(out_row1,_mm_mul_ps(a_vec1,brow));
        // row 2
        __m128 a_vec2 = _mm_set1_ps(a[2*K+i]);
        out_row2 = _mm_add_ps(out_row2,_mm_mul_ps(a_vec2,brow));
#else
        // row 0
        __m128 a_vec0 = _mm_set1_ps(a[i]);
        out_row0 = _mm_fmadd_ps(a_vec0,brow,out_row0);
        // row 1
        __m128 a_vec1 = _mm_set1_ps(a[K+i]);
        out_row1 = _mm_fmadd_ps(a_vec1,brow,out_row1);
        // row 2
        __m128 a_vec2 = _mm_set1_ps(a[2*K+i]);
        out_row2 = _mm_fmadd_ps(a_vec2,brow,out_row2);
#endif

    }
    _mm_store_ps(out,out_row0);
    _mm_storeu_ps(out+3,out_row1);
    _mm_storeu_ps(out+6,out_row2);
}

#else
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<(M!=K && M==N && M==3 && std::is_same<T,float>::value),bool>::type = 0>
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {
    _matmul_mKn<T,M,K,N>(a,b,out);
}
#endif

#ifdef __AVX__

// (4xk) x (kx4) matrices
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<(M!=K && M==N && M==4 && std::is_same<T,double>::value),bool>::type = 0>
FASTOR_INLINE
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    __m256d out_row0 = VZEROPD;
    __m256d out_row1 = VZEROPD;
    __m256d out_row2 = VZEROPD;
    __m256d out_row3 = VZEROPD;

    for (size_t i=0; i<K; ++i) {
        __m256d brow = _mm256_load_pd(&b[i*4]);
#ifndef __FMA__
        // row 0
        __m256d a_vec0 = _mm256_set1_pd(a[i]);
        out_row0 = _mm256_add_pd(out_row0,_mm256_mul_pd(a_vec0,brow));
        // row 1
        __m256d a_vec1 = _mm256_set1_pd(a[K+i]);
        out_row1 = _mm256_add_pd(out_row1,_mm256_mul_pd(a_vec1,brow));
        // row 2
        __m256d a_vec2 = _mm256_set1_pd(a[2*K+i]);
        out_row2 = _mm256_add_pd(out_row2,_mm256_mul_pd(a_vec2,brow));
        // row 3
        __m256d a_vec3 = _mm256_set1_pd(a[3*K+i]);
        out_row3 = _mm256_add_pd(out_row3,_mm256_mul_pd(a_vec3,brow));
#else
        // row 0
        __m256d a_vec0 = _mm256_set1_pd(a[i]);
        out_row0 = _mm256_fmadd_pd(a_vec0,brow,out_row0);
        // row 1
        __m256d a_vec1 = _mm256_set1_pd(a[K+i]);
        out_row1 = _mm256_fmadd_pd(a_vec1,brow,out_row1);
        // row 2
        __m256d a_vec2 = _mm256_set1_pd(a[2*K+i]);
        out_row2 = _mm256_fmadd_pd(a_vec2,brow,out_row2);
        // row 3
        __m256d a_vec3 = _mm256_set1_pd(a[3*K+i]);
        out_row3 = _mm256_fmadd_pd(a_vec3,brow,out_row3);
#endif
    }
    _mm256_store_pd(out,out_row0);
    _mm256_store_pd(&out[4],out_row1);
    _mm256_store_pd(&out[8],out_row2);
    _mm256_store_pd(&out[12],out_row3);
}

#else
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<M!=K && M==N && M==4 && std::is_same<T,double>::value,bool>::type = 0>
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {
    _matmul_mKn<T,M,K,N>(a,b,out);
}
#endif


#ifdef __SSE4_2__

// (4xk) x (kx4) matrices
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<(M!=K && M==N && M==4 && std::is_same<T,float>::value),bool>::type = 0>
FASTOR_INLINE
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    __m128 out_row0 = ZEROPS;
    __m128 out_row1 = ZEROPS;
    __m128 out_row2 = ZEROPS;
    __m128 out_row3 = ZEROPS;

    for (size_t i=0; i<K; ++i) {
        __m128 brow = _mm_load_ps(&b[i*4]);
#ifndef __FMA__
        // row 0
        __m128 a_vec0 = _mm_set1_ps(a[i]);
        out_row0 = _mm_add_ps(out_row0,_mm_mul_ps(a_vec0,brow));
        // row 1
        __m128 a_vec1 = _mm_set1_ps(a[K+i]);
        out_row1 = _mm_add_ps(out_row1,_mm_mul_ps(a_vec1,brow));
        // row 2
        __m128 a_vec2 = _mm_set1_ps(a[2*K+i]);
        out_row2 = _mm_add_ps(out_row2,_mm_mul_ps(a_vec2,brow));
        // row 3
        __m128 a_vec3 = _mm_set1_ps(a[3*K+i]);
        out_row3 = _mm_add_ps(out_row3,_mm_mul_ps(a_vec3,brow));
#else
        // row 0
        __m128 a_vec0 = _mm_set1_ps(a[i]);
        out_row0 = _mm_fmadd_ps(a_vec0,brow,out_row0);
        // row 1
        __m128 a_vec1 = _mm_set1_ps(a[K+i]);
        out_row1 = _mm_fmadd_ps(a_vec1,brow,out_row1);
        // row 2
        __m128 a_vec2 = _mm_set1_ps(a[2*K+i]);
        out_row2 = _mm_fmadd_ps(a_vec2,brow,out_row2);
        // row 3
        __m128 a_vec3 = _mm_set1_ps(a[3*K+i]);
        out_row3 = _mm_fmadd_ps(a_vec3,brow,out_row3);
#endif
    }
    _mm_store_ps(out,out_row0);
    _mm_store_ps(&out[4],out_row1);
    _mm_store_ps(&out[8],out_row2);
    _mm_store_ps(&out[12],out_row3);
}

#else
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<M!=K && M==N && M==4 && std::is_same<T,float>::value,bool>::type = 0>
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {
    _matmul_mKn<T,M,K,N>(a,b,out);
}
#endif



// Forward declare
template<typename T, size_t M, size_t N>
void _matvecmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out);
//



// Non-sqaure matrices
#ifndef FASTOR_USE_LIBXSMM
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<(M==K && K!=N) || (M!=K && K==N) || (M!=K && K!=N && M!=N)
                                 || (M!=K && M==N && M!=2 && M!=3 && M!=4)
                                 || ((M==N && M==K) && N % SIMDVector<T,DEFAULT_ABI>::Size !=0),bool>::type = 0>
#else
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<((M==K && K!=N) || (M!=K && K==N) || (M!=K && K!=N && M!=N)
                                 || (M!=K && M==N && M!=2 && M!=3 && M!=4)
                                 || ((M==N && M==K) && N % SIMDVector<T,DEFAULT_ABI>::Size !=0))
                                 && is_less_equal<M*N*K/BLAS_SWITCH_MATRIX_SIZE_NS/BLAS_SWITCH_MATRIX_SIZE_NS/BLAS_SWITCH_MATRIX_SIZE_NS,
                                 1>::value ,bool>::type = 0>
#endif
FASTOR_INLINE
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    // Matrix-vector specialisation
    FASTOR_IF_CONSTEXPR (N==1) {
        _matvecmul<T,M,K>(a,b,out);
        return;
    }

    using V = SIMDVector<T,DEFAULT_ABI>;
    // Use new faster matmul variants - this hueristics need to be changed
    // if matmul implementation changes
    FASTOR_IF_CONSTEXPR(M>=V::Size && K>=V::Size && N>1) {
        _matmul_mkn<T,M,K,N>(a,b,out);
        return;
    }

    // constexpr bool should_be_dispatched = (K > 32 ||
    //     (N==4 && std::is_same<T,double>::value)   ||
    //     (N==8 && std::is_same<T,float>::value)      ) ? true : false;
    // FASTOR_IF_CONSTEXPR(should_be_dispatched) {
    //     _matmul_mKn<T,M,K,N>(a,b,out);
    //     return;
    // }

    constexpr int SIZE_ = V::Size;
    constexpr int ROUND_ = ROUND_DOWN(N,SIZE_);

    for (size_t j=0; j<M; ++j) {
        size_t k=0;
        for (; k<ROUND_; k+=SIZE_) {
            V out_row;
            for (size_t i=0; i<K; ++i) {
                V brow(&b[i*N+k],false);
                V vec_a(a[j*K+i]);
#ifndef __FMA__
                out_row += vec_a*brow;
#else
                out_row = fmadd(vec_a,brow,out_row);
#endif
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


#ifdef FASTOR_USE_LIBXSMM
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<((M==K && K!=N) || (M!=K && K==N) || (M!=K && K!=N && M!=N)
                                 || (M!=K && M==N && M!=2 && M!=3 && M!=4)
                                 || ((M==N && M==K) && N % SIMDVector<T,DEFAULT_ABI>::Size !=0))
                                 && is_greater<M*N*K/BLAS_SWITCH_MATRIX_SIZE_NS/BLAS_SWITCH_MATRIX_SIZE_NS/BLAS_SWITCH_MATRIX_SIZE_NS,1>::value,
                                 bool>::type = 0>
FASTOR_INLINE
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {
    blas::matmul_libxsmm<T,M,K,N>(a,b,c);
}
#endif



#ifdef __SSE4_2__

template<>
FASTOR_INLINE
void _matmul<float,2,2,2>(const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {

    // 17 OPS
    __m128 ar = _mm_load_ps(a);
    __m128 br = _mm_load_ps(b);
    __m128 ar0 = _mm_shuffle_ps(ar,ar,_MM_SHUFFLE(2,2,0,0));
    __m128 ar1 = _mm_shuffle_ps(ar,ar,_MM_SHUFFLE(3,3,1,1));
    __m128 br0 = _mm_shuffle_ps(br,br,_MM_SHUFFLE(1,0,1,0));
    __m128 br1 = _mm_shuffle_ps(br,br,_MM_SHUFFLE(3,2,3,2));
    __m128 res = _mm_add_ps(_mm_mul_ps(ar0,br0),_mm_mul_ps(ar1,br1));
    _mm_store_ps(out,res);
}

template<>
FASTOR_INLINE
void _matmul<float,3,3,3>(const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {

    // 63 OPS + 3 OPS
    // This is a completely vectorised approach that reduces
    // (27 scalar mul + 18 scalar add) to (9 SSE mul + 6 SEE add)

    __m128 brow0 = _mm_loadl3_ps(b);
    __m128 brow1 = _mm_loadul3_ps(b+3);
    __m128 brow2 = _mm_loadul3_ps(b+6);

    {
        __m128 ai0 = _mm_set1_ps(a[0]);
        __m128 ai1 = _mm_set1_ps(a[1]);
        __m128 ai2 = _mm_set1_ps(a[2]);

        ai0 = _mm_mul_ps(ai0,brow0);
        ai1 = _mm_mul_ps(ai1,brow1);
        ai2 = _mm_mul_ps(ai2,brow2);
        _mm_store_ps(out,_mm_add_ps(ai0,_mm_add_ps(ai1,ai2)));
    }

    {
        __m128 ai0 = _mm_set1_ps(a[3]);
        __m128 ai1 = _mm_set1_ps(a[4]);
        __m128 ai2 = _mm_set1_ps(a[5]);

        ai0 = _mm_mul_ps(ai0,brow0);
        ai1 = _mm_mul_ps(ai1,brow1);
        ai2 = _mm_mul_ps(ai2,brow2);
        _mm_storeu_ps(out+3,_mm_add_ps(ai0,_mm_add_ps(ai1,ai2)));
    }

    {
        __m128 ai0 = _mm_set1_ps(a[6]);
        __m128 ai1 = _mm_set1_ps(a[7]);
        __m128 ai2 = _mm_set1_ps(a[8]);

        ai0 = _mm_mul_ps(ai0,brow0);
        ai1 = _mm_mul_ps(ai1,brow1);
        ai2 = _mm_mul_ps(ai2,brow2);
        _mm_storeu_ps(out+6,_mm_add_ps(ai0,_mm_add_ps(ai1,ai2)));
    }
}


template<>
FASTOR_INLINE void _matmul<float,4,4,4>(const float * FASTOR_RESTRICT b, const float * FASTOR_RESTRICT a, float * FASTOR_RESTRICT out) {

    // Note that a and b are swapped here
    // 16 SSE mul + 12 SSE add + 16 shuffles
    // Haswell 132 cycle
    // Skylake 116 cycle

    __m128 a0 = _mm_load_ps(a);
    __m128 a1 = _mm_load_ps(a+4);
    __m128 a2 = _mm_load_ps(a+8);
    __m128 a3 = _mm_load_ps(a+12);

    __m128 b0 = _mm_load_ps(b);
    __m128 b1 = _mm_load_ps(b+4);
    __m128 b2 = _mm_load_ps(b+8);
    __m128 b3 = _mm_load_ps(b+12);

    {
        __m128 e0 = _mm_shuffle_ps(b0, b0, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 e1 = _mm_shuffle_ps(b0, b0, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 e2 = _mm_shuffle_ps(b0, b0, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 e3 = _mm_shuffle_ps(b0, b0, _MM_SHUFFLE(3, 3, 3, 3));

        __m128 m0 = _mm_mul_ps(a0, e0);
        __m128 m1 = _mm_mul_ps(a1, e1);
        __m128 m2 = _mm_mul_ps(a2, e2);
        __m128 m3 = _mm_mul_ps(a3, e3);

        __m128 c0 = _mm_add_ps(m0, m1);
        __m128 c1 = _mm_add_ps(m2, m3);
        __m128 c2 = _mm_add_ps(c0, c1);

        _mm_store_ps(out,c2);
    }

    {
        __m128 e0 = _mm_shuffle_ps(b1, b1, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 e1 = _mm_shuffle_ps(b1, b1, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 e2 = _mm_shuffle_ps(b1, b1, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 e3 = _mm_shuffle_ps(b1, b1, _MM_SHUFFLE(3, 3, 3, 3));

        __m128 m0 = _mm_mul_ps(a0, e0);
        __m128 m1 = _mm_mul_ps(a1, e1);
        __m128 m2 = _mm_mul_ps(a2, e2);
        __m128 m3 = _mm_mul_ps(a3, e3);

        __m128 c0 = _mm_add_ps(m0, m1);
        __m128 c1 = _mm_add_ps(m2, m3);
        __m128 c2 = _mm_add_ps(c0, c1);

        _mm_store_ps(out+4,c2);
    }

    {
        __m128 e0 = _mm_shuffle_ps(b2, b2, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 e1 = _mm_shuffle_ps(b2, b2, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 e2 = _mm_shuffle_ps(b2, b2, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 e3 = _mm_shuffle_ps(b2, b2, _MM_SHUFFLE(3, 3, 3, 3));

        __m128 m0 = _mm_mul_ps(a0, e0);
        __m128 m1 = _mm_mul_ps(a1, e1);
        __m128 m2 = _mm_mul_ps(a2, e2);
        __m128 m3 = _mm_mul_ps(a3, e3);

        __m128 c0 = _mm_add_ps(m0, m1);
        __m128 c1 = _mm_add_ps(m2, m3);
        __m128 c2 = _mm_add_ps(c0, c1);

        _mm_store_ps(out+8,c2);
    }

    {
        __m128 e0 = _mm_shuffle_ps(b3, b3, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 e1 = _mm_shuffle_ps(b3, b3, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 e2 = _mm_shuffle_ps(b3, b3, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 e3 = _mm_shuffle_ps(b3, b3, _MM_SHUFFLE(3, 3, 3, 3));

        __m128 m0 = _mm_mul_ps(a0, e0);
        __m128 m1 = _mm_mul_ps(a1, e1);
        __m128 m2 = _mm_mul_ps(a2, e2);
        __m128 m3 = _mm_mul_ps(a3, e3);

        __m128 c0 = _mm_add_ps(m0, m1);
        __m128 c1 = _mm_add_ps(m2, m3);
        __m128 c2 = _mm_add_ps(c0, c1);

        _mm_store_ps(out+12,c2);
    }
}
#endif


#ifdef __AVX__
template<>
FASTOR_INLINE
void _matmul<double,2,2,2>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {

    const double a0 = a[0], a1=a[1], a2=a[2], a3=a[3];
    __m256d ar0 = _mm256_setr_pd(a0,a0,a2,a2);
    __m256d ar1 = _mm256_setr_pd(a1,a1,a3,a3);
    __m128d brl = _mm_load_pd(b);
    __m256d br0 = _mm256_castpd128_pd256(brl);
    br0 = _mm256_insertf128_pd(br0,brl,0x1);
    __m128d brh = _mm_load_pd(b+2);
    __m256d br1 = _mm256_castpd128_pd256(brh);
    br1 = _mm256_insertf128_pd(br1,brh,0x1);

    __m256d res = _mm256_add_pd(_mm256_mul_pd(ar0,br0),_mm256_mul_pd(ar1,br1));

    _mm256_store_pd(out,res);
}

template<>
FASTOR_INLINE
void _matmul<double,3,3,3>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {

    // 63 OPS + (3 OPS IVY)/(9 OPS HW)
    // This is a completely vectorised approach that reduces
    // (27 scalar mul + 18 scalar add) to (9 SSE mul + 6 SEE add)

    __m256d brow0 = _mm256_loadl3_pd(b);
    __m256d brow1 = _mm256_loadul3_pd(b+3);
    __m256d brow2 = _mm256_loadul3_pd(b+6);

    {
        __m256d ai0 = _mm256_set1_pd(a[0]);
        __m256d ai1 = _mm256_set1_pd(a[1]);
        __m256d ai2 = _mm256_set1_pd(a[2]);

        ai0 = _mm256_mul_pd(ai0,brow0);
        ai1 = _mm256_mul_pd(ai1,brow1);
        ai2 = _mm256_mul_pd(ai2,brow2);
        _mm256_store_pd(out,_mm256_add_pd(ai0,_mm256_add_pd(ai1,ai2)));
    }

    {
        __m256d ai0 = _mm256_set1_pd(a[3]);
        __m256d ai1 = _mm256_set1_pd(a[4]);
        __m256d ai2 = _mm256_set1_pd(a[5]);

        ai0 = _mm256_mul_pd(ai0,brow0);
        ai1 = _mm256_mul_pd(ai1,brow1);
        ai2 = _mm256_mul_pd(ai2,brow2);
        _mm256_storeu_pd(out+3,_mm256_add_pd(ai0,_mm256_add_pd(ai1,ai2)));
    }

    {
        __m256d ai0 = _mm256_set1_pd(a[6]);
        __m256d ai1 = _mm256_set1_pd(a[7]);
        __m256d ai2 = _mm256_set1_pd(a[8]);

        ai0 = _mm256_mul_pd(ai0,brow0);
        ai1 = _mm256_mul_pd(ai1,brow1);
        ai2 = _mm256_mul_pd(ai2,brow2);
        _mm256_storeu_pd(out+6,_mm256_add_pd(ai0,_mm256_add_pd(ai1,ai2)));
    }
}
#endif










//!----------------------------------------------------------------------
//! Matrix-vector multiplication

// Don't call this function directly as it's name is unconventional
// It gets called from within matmul anyway so always call matmul
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
    // of accumulators
    V _vec_a0, _vec_a1, _vec_b;
    int i=0;
    for (; i<ROUND_DOWN(M,2); i+=2) {
        V _vec_out0;
        V _vec_out1;
        int j = 0;
        for (; j< ROUND; j+=V::Size) {
            _vec_a0.load(&a[i*N+j]);
            _vec_a1.load(&a[(i+1)*N+j]);
            _vec_b.load(&b[j]);
            // _vec_out0 += _vec_a0*_vec_b;
            // _vec_out1 += _vec_a1*_vec_b;
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
            _vec_a0.load(&a[i*N+j]);
            _vec_b.load(&b[j]);
            // _vec_out0 += _vec_a0*_vec_b;
            _vec_out0 = fmadd(_vec_a0,_vec_b,_vec_out0);
        }
        T out_s0 = 0;
        for (; j< N; j+=1) {
            out_s0 += a[i*N+j]*b[j];
        }
        out[i]= _vec_out0.sum() + out_s0;
    }
}


#ifdef __SSE4_2__
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
    // 47 OPS
    __m128 a0 = _mm_load_ps(a);
    __m128 row0 = _mm_shift1_ps(a0);
    __m128 a1 = _mm_load_ps(a+4);
    __m128 row1 = _mm_shift1_ps(_mm_reverse_ps(_mm_shift1_ps(_mm_shuffle_ps(a1,a0,_MM_SHUFFLE(3,3,1,0)))));

    __m128 a2 = _mm_load_ss(a+8);
    __m128 row2 = _mm_shift1_ps(_mm_shuffle_ps(a1,a2,_MM_SHUFFLE(1,0,3,2)));

    __m128 vec_b = _mm_shift1_ps(_mm_load_ps(b));

    __m128 c0 = _add_ps(_mm_mul_ps(row0,vec_b));
    __m128 c1 = _add_ps(_mm_mul_ps(row1,vec_b));
    __m128 c2 = _add_ps(_mm_mul_ps(row2,vec_b));

    _mm_store_ss(out,c0);
    _mm_store_ss(out+1,c1);
    _mm_store_ss(out+2,c2);
}
#endif
#ifdef __AVX__
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
}
#endif



} // end of namespace

#endif // MATMUL_H

