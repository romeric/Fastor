#ifndef MATMUL_H
#define MATMUL_H

#include "commons/commons.h"
#include "extended_intrinsics/extintrin.h"
#include "simd_vector/SIMDVector.h"

namespace Fastor {



// For square matrices
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<M==N && M==K && N % SIMDVector<T,DEFAULT_ABI>::Size ==0,bool>::type = 0>
FASTOR_INLINE
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT c) {

    using V = SIMDVector<T,DEFAULT_ABI>;
    constexpr size_t UnrollOuterloop = V::size();

    // The row index (for a and c) is unrolled using the UnrollOuterloop stride. Therefore
    // the last rows may need special treatment if N is not a multiple of UnrollOuterloop.
    // N0 is the number of rows that can safely be iterated with a stride of
    // UnrollOuterloop.
    constexpr size_t i0 = N / UnrollOuterloop * UnrollOuterloop;
    for (size_t i = 0; i < i0; i += UnrollOuterloop) {
        // The iteration over the column index of b and c uses a stride of V::size(). This
        // enables row-vector loads (from b) and stores (to c). The matrix storage is
        // padded accordingly, ensuring correct bounds and alignment.
        for (size_t j = 0; j < N; j += V::size()) {
            // This temporary variables are used to accumulate the results of the products
            // producing the new values for the c matrix. This variable is necessary
            // because we need a V object for data-parallel accumulation. Storing to c
            // directly stores to scalar objects and thus would drop the ability for
            // data-parallel (SIMD) addition.
            V c_ij[UnrollOuterloop];
            for (size_t n = 0; n < UnrollOuterloop; ++n) {
                c_ij[n] = a[(i + n)*N]*V(&b[j]);
            }
            for (size_t k = 1; k < N - 1; ++k) {
                for (size_t n = 0; n < UnrollOuterloop; ++n) {
                    c_ij[n] += a[(i + n)*N+k] * V(&b[k*N+j]);
                }
            }
            for (size_t n = 0; n < UnrollOuterloop; ++n) {
                c_ij[n] += a[(i + n)*N+(N - 1)] * V(&b[(N - 1)*N+j]);
                c_ij[n].store(&c[(i + n)*N+j]);
            }
        }
    }
    // This final loop treats the remaining NN - N0 rows.
    for (size_t j = 0; j < N; j += V::size()) {
        V c_ij[UnrollOuterloop];
        for (size_t n = i0; n < N; ++n) {
            c_ij[n - i0] = a[n*N] * V(&b[j]);
        }
        for (size_t k = 1; k < N - 1; ++k) {
            for (size_t n = i0; n < N; ++n) {
                c_ij[n - i0] += a[n*N+k] * V(&b[k*N+j]);
            }
        }
        for (size_t n = i0; n < N; ++n) {
            c_ij[n - i0] += a[n*N+(N - 1)] * V(&b[(N - 1)*N+j]);
            c_ij[n - i0].store(&c[n*N+j]);
        }
    }
}


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
    _mm256_storeu_pd(out+6,out_row2);
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
    _mm256_store_pd(out+4,out_row1);
    _mm256_store_pd(out+8,out_row2);
    _mm256_store_pd(out+12,out_row3);
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
        out_row3 = _mm_fmadd_ps(out_row3,brow,a_vec3);
#endif
    }
    _mm_store_ps(out,out_row0);
    _mm_store_ps(out+4,out_row1);
    _mm_store_ps(out+8,out_row2);
    _mm_store_ps(out+12,out_row3);


    /* 
    // AVX version - process two rows in one AVX register - doesn't pay off
    // as on FMA based architecture increases the cycle count by 4
    __m256 out_row01 = VZEROPS;
    __m256 out_row23 = VZEROPS;

    for (size_t i=0; i<K; ++i) {
        __m128 br = _mm_load_ps(&b[i*4]);
        __m256 brow = _mm256_castps128_ps256(br);
        brow = _mm256_insertf128_ps(brow,br,0x1);
        // row  0 & 1
        __m128 a_vec0 = _mm_set1_ps(a[i]);
        __m128 a_vec1 = _mm_set1_ps(a[K+i]);
        __m256 a_vec01 = _mm256_castps128_ps256(a_vec0);
        a_vec01 = _mm256_insertf128_ps(a_vec01,a_vec1,0x1);
#ifndef __FMA__
        out_row01 = _mm256_add_ps(out_row01,_mm256_mul_ps(a_vec01,brow));
#else
        out_row01 = _mm256_fmadd_ps(a_vec01,brow,out_row01);
#endif
        // row 2 & 3
        __m128 a_vec2 = _mm_set1_ps(a[2*K+i]);
        __m128 a_vec3 = _mm_set1_ps(a[3*K+i]);
        __m256 a_vec23 = _mm256_castps128_ps256(a_vec2);
        a_vec23 = _mm256_insertf128_ps(a_vec23,a_vec3,0x1);
#ifndef __FMA__
        out_row23 = _mm256_add_ps(out_row23,_mm256_mul_ps(a_vec23,brow));
#else
        out_row23 = _mm256_fmadd_ps(a_vec23,brow,out_row23);
#endif
    }
    _mm256_store_ps(out,out_row01);
    _mm256_store_ps(out+8,out_row23);
    */
}

#endif


// (2x2) x (2xn) matrices
template<typename T, size_t M, size_t N>
FASTOR_INLINE
void _matmul_2x2xn(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {


    using V256 = SIMDVector<T,256>;
    using V128 = SIMDVector<T,128>;

    constexpr int SIZE_AVX = V256::Size;
    constexpr int SIZE_SSE = V128::Size;
    constexpr int ROUND_AVX = ROUND_DOWN(N,(int)SIZE_AVX);
    constexpr int ROUND_SSE = ROUND_DOWN(N,(int)SIZE_SSE);

    size_t k=0;
    for (; k<ROUND_AVX; k+=SIZE_AVX) {

        V256 out_row0, out_row1, vec_a0, vec_a1;
        for (size_t i=0; i<M; ++i) {
            V256 brow; brow.load(&b[i*N+k],false);
            vec_a0.set(a[i]);
            vec_a1.set(a[i+M]);
#ifndef __FMA__
            out_row0 += vec_a0*brow;
            out_row1 += vec_a1*brow;
#else
            out_row0 = fmadd(vec_a0,brow,out_row0);
            out_row1 = fmadd(vec_a1,brow,out_row1);
#endif            
        }
        out_row0.store(out+k,false);
        out_row1.store(out+N+k,false);
    }

    for (; k<ROUND_SSE; k+=SIZE_SSE) {
        V128 out_row0, out_row1, vec_a0, vec_a1, brow;
        for (size_t i=0; i<M; ++i) {
            V128 brow; brow.load(&b[i*N+k],false);
            vec_a0.set(a[i]);
            vec_a1.set(a[i+M]);
#ifndef __FMA__
            out_row0 += vec_a0*brow;
            out_row1 += vec_a1*brow;
#else
            out_row0 = fmadd(vec_a0,brow,out_row0);
            out_row1 = fmadd(vec_a1,brow,out_row1);
#endif 
        }
        out_row0.store(out+k,false);
        out_row1.store(out+N+k,false);
    }

    for (; k<N; k++) {
        T out_row0=0., out_row1=0.;
        for (size_t i=0; i<M; ++i) {
            T brow = b[i*N+k];
            out_row0 += a[i]*brow;
            out_row1 += a[i+M]*brow;
        }
        out[k] = out_row0;
        out[N+k] = out_row1;
    }
}


// (3x3) x (3xn) matrices
template<typename T, size_t M, size_t N>
FASTOR_INLINE
void _matmul_3x3xn(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {


    using V256 = SIMDVector<T,256>;
    using V128 = SIMDVector<T,128>;

    constexpr int SIZE_AVX = V256::Size;
    constexpr int SIZE_SSE = V128::Size;
    constexpr int ROUND_AVX = ROUND_DOWN(N,(int)SIZE_AVX);
    constexpr int ROUND_SSE = ROUND_DOWN(N,(int)SIZE_SSE);

    size_t k=0;
    for (; k<ROUND_AVX; k+=SIZE_AVX) {

        V256 out_row0, out_row1, out_row2, vec_a0, vec_a1, vec_a2;
        for (size_t i=0; i<M; ++i) {
            V256 brow; brow.load(&b[i*N+k],false);
            vec_a0.set(a[i]);
            vec_a1.set(a[i+M]);
            vec_a2.set(a[i+2*M]);
#ifndef __FMA__
            out_row0 += vec_a0*brow;
            out_row1 += vec_a1*brow;
            out_row2 += vec_a2*brow;
#else
            out_row0 = fmadd(vec_a0,brow,out_row0);
            out_row1 = fmadd(vec_a1,brow,out_row1);
            out_row2 = fmadd(vec_a2,brow,out_row2);
#endif            
        }
        out_row0.store(out+k,false);
        out_row1.store(out+N+k,false);
        out_row2.store(out+2*N+k,false);
    }

    for (; k<ROUND_SSE; k+=SIZE_SSE) {
        V128 out_row0, out_row1, out_row2, vec_a0, vec_a1, vec_a2;
        for (size_t i=0; i<M; ++i) {
            V128 brow; brow.load(&b[i*N+k],false);
            vec_a0.set(a[i]);
            vec_a1.set(a[i+M]);
            vec_a2.set(a[i+2*M]);
#ifndef __FMA__
            out_row0 += vec_a0*brow;
            out_row1 += vec_a1*brow;
            out_row2 += vec_a2*brow;
#else
            out_row0 = fmadd(vec_a0,brow,out_row0);
            out_row1 = fmadd(vec_a1,brow,out_row1);
            out_row2 = fmadd(vec_a2,brow,out_row2);
#endif 
        }
        out_row0.store(out+k,false);
        out_row1.store(out+N+k,false);
        out_row2.store(out+2*N+k,false);
    }

    for (; k<N; k++) {
        T out_row0=0., out_row1=0., out_row2=0.;
        for (size_t i=0; i<M; ++i) {
            T brow = b[i*N+k];
            out_row0 += a[i]*brow;
            out_row1 += a[i+M]*brow;
            out_row2 += a[i+2*M]*brow;
        }
        out[k] = out_row0;
        out[N+k] = out_row1;
        out[2*N+k] = out_row2;
    }
}




// Non-sqaure matrices
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<(M==K && K!=N) || (M!=K && K==N) || (M!=K && K!=N && M!=N)
                                 || (M!=K && M==N && M!=2 && M!=3 && M!=4)
                                 || ((M==N && M==K) && N % SIMDVector<T,DEFAULT_ABI>::Size !=0),bool>::type = 0>
FASTOR_INLINE
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    // The following specialisations don't make much of a difference (at least for SP)
    // Need thorough performance checks
    if (M==3 && K==3 && N!=K) {
        _matmul_3x3xn<T,M,N>(a,b,out);
        return;
    }
    else if (M==2 && K==2 && N!=K) {
        _matmul_2x2xn<T,M,N>(a,b,out);
        return;
    }

    using V256 = SIMDVector<T,256>;
    using V128 = SIMDVector<T,128>;

    constexpr int SIZE_AVX = V256::Size;
    constexpr int SIZE_SSE = V128::Size;
    constexpr int ROUND_AVX = ROUND_DOWN(N,(int)SIZE_AVX);
    constexpr int ROUND_SSE = ROUND_DOWN(N,(int)SIZE_SSE);


#if FASTOR_MATMUL_UNROLL_LENGTH==2

    size_t j=0;
    for (; j<ROUND_DOWN(M,2); j+=2) {
        int k=0;
        for (; k<ROUND_AVX; k+=SIZE_AVX) {
            V256 out_row0, out_row1, vec_a0, vec_a1;
            for (size_t i=0; i<K; ++i) {
                V256 brow; brow.load(&b[i*N+k],false);
                vec_a0.set(a[j*K+i]);
                vec_a1.set(a[(j+1)*K+i]);
#ifndef __FMA__
                out_row0 += vec_a0*brow;
                out_row1 += vec_a1*brow;
#else
                out_row0 = fmadd(vec_a0,brow,out_row0);
                out_row1 = fmadd(vec_a1,brow,out_row1);
#endif
            }
            out_row0.store(out+k+N*j,false);
            out_row1.store(out+k+N*(j+1),false);
        }

        for (; k<ROUND_SSE; k+=SIZE_SSE) {
            V128 out_row0, out_row1, vec_a0, vec_a1;
            for (size_t i=0; i<K; ++i) {
                V128 brow; brow.load(&b[i*N+k],false);
                vec_a0.set(a[j*K+i]);
                vec_a1.set(a[(j+1)*K+i]);
#ifndef __FMA__
                out_row0 += vec_a0*brow;
                out_row1 += vec_a1*brow;
#else
                out_row0 = fmadd(vec_a0,brow,out_row0);
                out_row1 = fmadd(vec_a1,brow,out_row1);
#endif
            }
            out_row0.store(out+k+N*j,false);
            out_row1.store(out+k+N*(j+1),false);
        }

        for (; k<N; k++) {
            T out_row0 = 0., out_row1 = 0.;
            for (size_t i=0; i<K; ++i) {
                T brow = b[i*N+k];
                out_row0 += a[j*K+i]*brow;
                out_row1 += a[(j+1)*K+i]*brow;
            }
            out[k+N*j] = out_row0;
            out[k+N*(j+1)] = out_row1;
        }
    }

    for (; j<M; ++j) {
        int k=0;
        for (; k<ROUND_AVX; k+=SIZE_AVX) {
            V256 out_row, vec_a;
            for (size_t i=0; i<K; ++i) {
                V256 brow; brow.load(&b[i*N+k],false);
                vec_a.set(a[j*K+i]);
                out_row += vec_a*brow;
#ifndef __FMA__
                out_row += vec_a*brow;
#else
                out_row = fmadd(vec_a,brow,out_row);
#endif
            }
            out_row.store(out+k+N*j,false);
        }

        for (; k<ROUND_SSE; k+=SIZE_SSE) {
            V128 out_row, vec_a;
            for (size_t i=0; i<K; ++i) {
                V128 brow; brow.load(&b[i*N+k],false);
                vec_a.set(a[j*K+i]);
#ifndef __FMA__
                out_row += vec_a*brow;
#else
                out_row = fmadd(vec_a,brow,out_row);
#endif
            }
            out_row.store(out+k+N*j,false);
        }

        for (; k<N; k++) {
            T out_row = 0.;
            for (size_t i=0; i<K; ++i) {
                out_row += a[j*K+i]*b[i*N+k];
            }
            out[k+N*j] = out_row;
        }
    }


#elif FASTOR_MATMUL_UNROLL_LENGTH==4

    size_t j=0;
    for (; j<ROUND_DOWN(M,4); j+=4) {
        int k=0;
        for (; k<ROUND_AVX; k+=SIZE_AVX) {
            V256 out_row0, out_row1, out_row2, out_row3, vec_a0, vec_a1, vec_a2, vec_a3;
            for (size_t i=0; i<K; ++i) {
                V256 brow; brow.load(&b[i*N+k],false);
                vec_a0.set(a[j*K+i]);
                vec_a1.set(a[(j+1)*K+i]);
                vec_a2.set(a[(j+2)*K+i]);
                vec_a3.set(a[(j+3)*K+i]);
                out_row0 += vec_a0*brow;
                out_row1 += vec_a1*brow;
                out_row2 += vec_a2*brow;
                out_row3 += vec_a3*brow;
#ifndef __FMA__
                out_row0 += vec_a0*brow;
                out_row1 += vec_a1*brow;
                out_row2 += vec_a2*brow;
                out_row3 += vec_a3*brow;
#else
                out_row0 = fmadd(vec_a0,brow,out_row0);
                out_row1 = fmadd(vec_a1,brow,out_row1);
                out_row2 = fmadd(vec_a2,brow,out_row2);
                out_row3 = fmadd(vec_a3,brow,out_row3);
#endif
            }
            out_row0.store(out+k+N*j,false);
            out_row1.store(out+k+N*(j+1),false);
            out_row2.store(out+k+N*(j+2),false);
            out_row3.store(out+k+N*(j+3),false);
        }

        for (; k<ROUND_SSE; k+=SIZE_SSE) {
            V128 out_row0, out_row1, out_row2, out_row3, vec_a0, vec_a1, vec_a2, vec_a3;
            for (int i=0; i<K; ++i) {
                V128 brow; brow.load(&b[i*N+k],false);
                vec_a0.set(a[j*K+i]);
                vec_a1.set(a[(j+1)*K+i]);
                vec_a2.set(a[(j+2)*K+i]);
                vec_a3.set(a[(j+3)*K+i]);
#ifndef __FMA__
                out_row0 += vec_a0*brow;
                out_row1 += vec_a1*brow;
                out_row2 += vec_a2*brow;
                out_row3 += vec_a3*brow;
#else
                out_row0 = fmadd(vec_a0,brow,out_row0);
                out_row1 = fmadd(vec_a1,brow,out_row1);
                out_row2 = fmadd(vec_a2,brow,out_row2);
                out_row3 = fmadd(vec_a3,brow,out_row3);
#endif
            }
            out_row0.store(out+k+N*j,false);
            out_row1.store(out+k+N*(j+1),false);
            out_row2.store(out+k+N*(j+2),false);
            out_row3.store(out+k+N*(j+3),false);
        }

        for (; k<N; k++) {
            T out_row0 = 0., out_row1 = 0., out_row2 = 0., out_row3 = 0.;
            for (size_t i=0; i<K; ++i) {
                T brow = b[i*N+k];
                out_row0 += a[j*K+i]*brow;
                out_row1 += a[(j+1)*K+i]*brow;
                out_row2 += a[(j+2)*K+i]*brow;
                out_row3 += a[(j+3)*K+i]*brow;
            }
            out[k+N*j] = out_row0;
            out[k+N*(j+1)] = out_row1;
            out[k+N*(j+2)] = out_row2;
            out[k+N*(j+3)] = out_row3;
        }
    }

    for (; j<M; ++j) {
        int k=0;
        for (; k<ROUND_AVX; k+=SIZE_AVX) {
            V256 out_row, vec_a;
            for (int i=0; i<K; ++i) {
                V256 brow; brow.load(&b[i*N+k],false);
                vec_a.set(a[j*K+i]);
#ifndef __FMA__
                out_row += vec_a*brow;
#else
                out_row = fmadd(vec_a,brow,out_row);
#endif
            }
            out_row.store(out+k+N*j,false);
        }

        for (; k<ROUND_SSE; k+=SIZE_SSE) {
            V128 out_row, vec_a;
            for (int i=0; i<K; ++i) {
                V128 brow; brow.load(&b[i*N+k],false);
                vec_a.set(a[j*K+i]);
#ifndef __FMA__
                out_row += vec_a*brow;
#else
                out_row = fmadd(vec_a,brow,out_row);
#endif
            }
            out_row.store(out+k+N*j,false);
        }

        for (; k<N; k++) {
            T out_row = 0.;
            for (int i=0; i<K; ++i) {
                out_row += a[j*K+i]*b[i*N+k];
            }
            out[k+N*j] = out_row;
        }
    }

#else

    for (size_t j=0; j<M; ++j) {
        size_t k=0;
        for (; k<ROUND_AVX; k+=SIZE_AVX) {
            V256 out_row, vec_a;
            for (size_t i=0; i<K; ++i) {
                V256 brow; brow.load(&b[i*N+k],false);
                vec_a.set(a[j*K+i]);
#ifndef __FMA__
                out_row += vec_a*brow;
#else
                out_row = fmadd(vec_a,brow,out_row);
#endif
            }
            out_row.store(out+k+N*j,false);
        }

        for (; k<ROUND_SSE; k+=SIZE_SSE) {
            V128 out_row, vec_a;
            for (size_t i=0; i<K; ++i) {
                V128 brow; brow.load(&b[i*N+k],false);
                vec_a.set(a[j*K+i]);
#ifndef __FMA__
                out_row += vec_a*brow;
#else
                out_row = fmadd(vec_a,brow,out_row);
#endif
            }
            out_row.store(out+k+N*j,false);
        }

        for (; k<N; k++) {
            T out_row = 0.;
            for (size_t i=0; i<K; ++i) {
                out_row += a[j*K+i]*b[i*N+k];
            }
            out[N*j+k] = out_row;
        }
    }

#endif

#ifdef FASTOR_MATMUL_UNROLL_INNER
#ifndef FASTOR_MATMUL_UNROLL_LENGTH

    constexpr int INNER_UNROLL = ROUND_DOWN(K,2);

    for (size_t j=0; j<M; ++j) {
        int k=0;
        for (; k<ROUND_AVX; k+=SIZE_AVX) {
            V256 out_row, out_row0, out_row1, vec_a0, vec_a1;
            int i=0;
            for (; i<INNER_UNROLL; i+=2) {
                V256 brow0; brow0.load(&b[i*N+k],false);
                V256 brow1; brow1.load(&b[(i+1)*N+k],false);
                vec_a0.set(a[j*K+i]);
                vec_a1.set(a[j*K+i+1]);
#ifndef __FMA__
                out_row0 += vec_a0*brow0;
                out_row1 += vec_a1*brow1;
#else
                out_row0 = fmadd(vec_a0,brow0,out_row0);
                out_row1 = fmadd(vec_a1,brow1,out_row1)
#endif
            }
            for (; i<K; ++i) {
                V256 brow; brow.load(&b[i*N+k],false);
                vec_a0.set(a[j*K+i]);
                out_row += vec_a0*brow;
            }
            out_row += out_row0 + out_row1;
            out_row.store(out+k+N*j,false);
        }

        for (; k<ROUND_SSE; k+=SIZE_SSE) {
            V128 out_row, out_row0, out_row1, vec_a0, vec_a1;
            int i=0;
            for (; i<INNER_UNROLL; i+=2) {
                V128 brow0; brow0.load(&b[i*N+k],true);
                V128 brow1; brow1.load(&b[(i+1)*N+k],false);
                vec_a0.set(a[j*K+i]);
                vec_a1.set(a[j*K+i+1]);
#ifndef __FMA__
                out_row0 += vec_a0*brow0;
                out_row1 += vec_a1*brow1;
#else
                out_row0 = fmadd(vec_a0,brow0,out_row0);
                out_row1 = fmadd(vec_a1,brow1,out_row1)
#endif
            }
            for (; i<K; ++i) {
                V128 brow; brow.load(&b[i*N+k],false);
                vec_a0.set(a[j*K+i]);
                out_row += vec_a0*brow;
            }
            out_row += out_row0 + out_row1;
            out_row.store(out+k+N*j,false);
        }

        for (; k<N; k++) {
            T out_row = 0.;
            for (size_t i=0; i<K; ++i) {
                out_row += a[j*K+i]*b[i*N+k];
            }
            out[N*j+k] = out_row;
        }
    }
#endif
#endif
}



#ifdef __SSE4_2__

template<>
FASTOR_INLINE
void _matmul<float,2,2,2>(const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {

#ifndef USE_OLD_VERSION
    // 17 OPS
    __m128 ar = _mm_load_ps(a);
    __m128 br = _mm_load_ps(b);
    __m128 ar0 = _mm_shuffle_ps(ar,ar,_MM_SHUFFLE(2,2,0,0));
    __m128 ar1 = _mm_shuffle_ps(ar,ar,_MM_SHUFFLE(3,3,1,1));
    __m128 br0 = _mm_shuffle_ps(br,br,_MM_SHUFFLE(1,0,1,0));
    __m128 br1 = _mm_shuffle_ps(br,br,_MM_SHUFFLE(3,2,3,2));
    __m128 res = _mm_add_ps(_mm_mul_ps(ar0,br0),_mm_mul_ps(ar1,br1));
    _mm_store_ps(out,res);
#else
    // 24 OPS
    __m128 ar = _mm_load_ps(a);
    __m128 br = _mm_load_ps(b);
    __m128 r0 = _mm_shuffle_ps(br,br,_MM_SHUFFLE(3,1,2,0));
    __m128 r1 = _mm_shuffle_ps(ar,ar,_MM_SHUFFLE(1,0,3,2));
    __m128 c0 = _mm_mul_ps(ar,r0);
    __m128 c1 = _mm_mul_ps(r1,r0);
    c0 = _mm_hadd_ps(c0,c0);
    c1 = _mm_hadd_ps(c1,c1);
    __m128 c = _mm_shuffle_ps(c0,c1,_MM_SHUFFLE(1,0,1,0));
    c = _mm_shuffle_ps(c,c,_MM_SHUFFLE(1,2,3,0));

    _mm_store_ps(out,c);
#endif
}

template<>
FASTOR_INLINE
void _matmul<float,3,3,3>(const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {

#ifndef USE_OLD_VERSION
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

#else

    // 144 OPS with store_ss
    // 150 OPS with store_ps
    __m128 arow0 = _mm_shift1_ps(_mm_load_ps(a));
    __m128 arow1 = _mm_shift1_ps(_mm_loadu_ps(a+3));
    __m128 arow2 = _mm_shift1_ps(_mm_loadu_ps(a+6));

    __m128 brow0 = _mm_load_ps(b);
    __m128 brow1 = _mm_load_ps(b+4);
    __m128 brow2 = _mm_load_ss(b+8);

    __m128 r0 = _mm_shuffle_ps(brow0,brow1,_MM_SHUFFLE(1,2,3,0));
    __m128 bcol0 = _mm_shift1_ps(r0);
    __m128 r1 = _mm_shuffle_ps(brow0,brow1,_MM_SHUFFLE(3,0,3,1));
    __m128 bcol1 = _mm_shift1_ps(_mm_shuffle_ps(r1,r1,_MM_SHUFFLE(1,3,2,0)));
    __m128 r2 = _mm_shuffle_ps(brow0,brow1,_MM_SHUFFLE(3,1,3,2));
    __m128 bcol2 = _mm_shift1_ps(_mm_shuffle_ps(r2,brow2,_MM_SHUFFLE(3,0,2,0)));

    // Using horizontal addition hadd instrunction (This is slow even in Skylake)
#ifdef USE_HADD
    // Row 0
    __m128 out_00 = _mm_mul_ps(arow0,bcol0);
    out_00 = _mm_hadd_ps(out_00,out_00);
    out_00 = _mm_hadd_ps(out_00,out_00);
    __m128 out_01 = _mm_mul_ps(arow0,bcol1);
    out_01 = _mm_hadd_ps(out_01,out_01);
    out_01 = _mm_hadd_ps(out_01,out_01);
    __m128 out_02 = _mm_mul_ps(arow0,bcol2);
    out_02 = _mm_hadd_ps(out_02,out_02);
    out_02 = _mm_hadd_ps(out_02,out_02);
    // Row 1
    __m128 out_10 = _mm_mul_ps(arow1,bcol0);
    out_10 = _mm_hadd_ps(out_10,out_10);
    out_10 = _mm_hadd_ps(out_10,out_10);
    __m128 out_11 = _mm_mul_ps(arow1,bcol1);
    out_11 = _mm_hadd_ps(out_11,out_11);
    out_11 = _mm_hadd_ps(out_11,out_11);
    __m128 out_12 = _mm_mul_ps(arow1,bcol2);
    out_12 = _mm_hadd_ps(out_12,out_12);
    out_12 = _mm_hadd_ps(out_12,out_12);
    // Row 2
    __m128 out_20 = _mm_mul_ps(arow2,bcol0);
    out_20 = _mm_hadd_ps(out_20,out_20);
    out_20 = _mm_hadd_ps(out_20,out_20);
    __m128 out_21 = _mm_mul_ps(arow2,bcol1);
    out_21 = _mm_hadd_ps(out_21,out_21);
    out_21 = _mm_hadd_ps(out_21,out_21);
    __m128 out_22 = _mm_mul_ps(arow2,bcol2);
    out_22 = _mm_hadd_ps(out_22,out_22);
    out_22 = _mm_hadd_ps(out_22,out_22);
#else
    // Row 0
    __m128 out_00 = _mm_mul_ps(arow0,bcol0);
    __m128 shuf = _mm_movehdup_ps(out_00);        // line up elements 3,1 with 2,0
    __m128 sums = _mm_add_ps(out_00, shuf);
    shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
    out_00 = _mm_add_ss(sums, shuf);

    __m128 out_01 = _mm_mul_ps(arow0,bcol1);
    shuf = _mm_movehdup_ps(out_01);
    sums = _mm_add_ps(out_01, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    out_01 = _mm_add_ss(sums, shuf);

    __m128 out_02 = _mm_mul_ps(arow0,bcol2);
    shuf = _mm_movehdup_ps(out_02);
    sums = _mm_add_ps(out_02, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    out_02 = _mm_add_ss(sums, shuf);
    // Row 1
    __m128 out_10 = _mm_mul_ps(arow1,bcol0);
    shuf = _mm_movehdup_ps(out_10);
    sums = _mm_add_ps(out_10, shuf);
    shuf        = _mm_movehl_ps(shuf, sums);
    out_10        = _mm_add_ss(sums, shuf);
    __m128 out_11 = _mm_mul_ps(arow1,bcol1);
    shuf = _mm_movehdup_ps(out_11);
    sums = _mm_add_ps(out_11, shuf);
    shuf        = _mm_movehl_ps(shuf, sums);
    out_11        = _mm_add_ss(sums, shuf);
    __m128 out_12 = _mm_mul_ps(arow1,bcol2);
    shuf = _mm_movehdup_ps(out_12);
    sums = _mm_add_ps(out_12, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    out_12 = _mm_add_ss(sums, shuf);
    // Row 2
    __m128 out_20 = _mm_mul_ps(arow2,bcol0);
    shuf = _mm_movehdup_ps(out_20);
    sums = _mm_add_ps(out_20, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    out_20 = _mm_add_ss(sums, shuf);
    __m128 out_21 = _mm_mul_ps(arow2,bcol1);
    shuf = _mm_movehdup_ps(out_21);
    sums = _mm_add_ps(out_21, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    out_21 = _mm_add_ss(sums, shuf);
    __m128 out_22 = _mm_mul_ps(arow2,bcol2);
    shuf = _mm_movehdup_ps(out_22);
    sums = _mm_add_ps(out_22, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    out_22 = _mm_add_ss(sums, shuf);
#endif

    // This is equally fast
    _mm_store_ss(out,out_00);
    _mm_store_ss(out+1,out_01);
    _mm_store_ss(out+2,out_02);
    _mm_store_ss(out+3,out_10);
    _mm_store_ss(out+4,out_11);
    _mm_store_ss(out+5,out_12);
    _mm_store_ss(out+6,out_20);
    _mm_store_ss(out+7,out_21);
    _mm_store_ss(out+8,out_22);

#endif

}

#endif

#ifdef __AVX__
template<>
FASTOR_INLINE
void _matmul<double,2,2,2>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {

#ifndef USE_OLD_VERSION

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

#else
    __m256d ar = _mm256_load_pd(a);
    __m256d br = _mm256_load_pd(b);

    // arrage as [b11 b21 b12 b22]
    __m128d d1 = _mm256_castpd256_pd128(br);
    __m128d d2 = _mm256_extractf128_pd(br, 0x1);
    auto dd1 = _mm_shuffle_pd(d1,d2,0x0);
    auto dd2 = _mm_shuffle_pd(d1,d2,0x3);
    br = _mm256_castpd128_pd256(dd1);
    br = _mm256_insertf128_pd(br,dd2,1);

    auto mul0 = _mm256_mul_pd(ar,br);
    ar = _mm256_permute2f128_pd(ar,ar,0x1);
    auto mul1 = _mm256_mul_pd(ar,br);
    mul0 = _mm256_hadd_pd(mul0,mul0);
    mul1 = _mm256_hadd_pd(mul1,mul1);

    d1 = _mm256_castpd256_pd128(mul0);
    d2 = _mm256_extractf128_pd(mul0,1);
    _mm_store_sd(out,d1);
    _mm_store_sd(out+3,d2);

    d1 = _mm256_castpd256_pd128(mul1);
    d2 = _mm256_extractf128_pd(mul1,1);
    _mm_store_sd(out+2,d1);
    _mm_store_sd(out+1,d2);

#endif
}

template<>
FASTOR_INLINE
void _matmul<double,3,3,3>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {


#ifndef USE_OLD_VERSION
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

#else

    // IVY 135 OPS / HW 162 OPS
    __m256d arow0 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_load_pd(a)),_mm_load_sd(a+2),0x1);
    __m256d arow1 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(a+3)),_mm_load_sd(a+5),0x1);
    __m256d arow2 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(a+6)),_mm_load_sd(a+8),0x1);

    __m128d b00 = _mm_load_sd(b);
    __m128d b01 = _mm_load_sd(b+1);
    __m128d b02 = _mm_load_sd(b+2);
    __m128d b10 = _mm_load_sd(b+3);
    __m128d b11 = _mm_load_sd(b+4);
    __m128d b12 = _mm_load_sd(b+5);
    __m128d b20 = _mm_load_sd(b+6);
    __m128d b21 = _mm_load_sd(b+7);
    __m128d b22 = _mm_load_sd(b+8);

    __m128d r0 = _mm_shuffle_pd(b00,b10,0x0);
    __m256d bcol0 = _mm256_castpd128_pd256(r0);
    bcol0 = _mm256_insertf128_pd(bcol0,b20,0x1);
    __m128d r1 = _mm_shuffle_pd(b01,b11,0x0);
    __m256d bcol1 = _mm256_castpd128_pd256(r1);
    bcol1 = _mm256_insertf128_pd(bcol1,b21,0x1);
    __m128d r2 = _mm_shuffle_pd(b02,b12,0x0);
    __m256d bcol2 = _mm256_castpd128_pd256(r2);
    bcol2 = _mm256_insertf128_pd(bcol2,b22,0x1);
    // Row 0
    __m256d vout_00 = _mm256_mul_pd(arow0,bcol0);
    __m128d out_00 = _add_pd(vout_00);
    __m256d vout_01 = _mm256_mul_pd(arow0,bcol1);
    __m128d out_01 = _add_pd(vout_01);
    __m256d vout_02 = _mm256_mul_pd(arow0,bcol2);
    __m128d out_02 = _add_pd(vout_02);
    // Row 1
    __m256d vout_10 = _mm256_mul_pd(arow1,bcol0);
    __m128d out_10 = _add_pd(vout_10);
    __m256d vout_11 = _mm256_mul_pd(arow1,bcol1);
    __m128d out_11 = _add_pd(vout_11);
    __m256d vout_12 = _mm256_mul_pd(arow1,bcol2);
    __m128d out_12 = _add_pd(vout_12);
    // Row 2
    __m256d vout_20 = _mm256_mul_pd(arow2,bcol0);
    __m128d out_20 = _add_pd(vout_20);
    __m256d vout_21 = _mm256_mul_pd(arow2,bcol1);
    __m128d out_21 = _add_pd(vout_21);
    __m256d vout_22 = _mm256_mul_pd(arow2,bcol2);
    __m128d out_22 = _add_pd(vout_22);
    // Store
    _mm_store_sd(out,out_00);
    _mm_store_sd(out+1,out_01);
    _mm_store_sd(out+2,out_02);
    _mm_store_sd(out+3,out_10);
    _mm_store_sd(out+4,out_11);
    _mm_store_sd(out+5,out_12);
    _mm_store_sd(out+6,out_20);
    _mm_store_sd(out+7,out_21);
    _mm_store_sd(out+8,out_22);
#endif

}
#endif

#ifdef __SSE4_2__
template<>
FASTOR_INLINE void _matmul<float,4,4,4>(const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {

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













//!----------------------------------------------------------------------
//! Matrix-vector multiplication
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

