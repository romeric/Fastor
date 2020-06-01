#ifndef MATMUL_SPECIALISATIONS_KERNELS_H
#define MATMUL_SPECIALISATIONS_KERNELS_H


// Forward declare
//-----------------------------------------------------------------------------------------------------------
namespace internal {
template<typename T, size_t M, size_t K, size_t N,
    typename std::enable_if<M==N && M==8 && std::is_same<T,float>::value,bool>::type = 0 >
FASTOR_INLINE
void _matmul8k8_float(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out);

template<typename T, size_t M, size_t K, size_t N,
    typename std::enable_if<M==N && M==8 && std::is_same<T,double>::value,bool>::type = 0 >
FASTOR_INLINE
void _matmul8k8_double(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out);

} // internal
//-----------------------------------------------------------------------------------------------------------



#ifdef FASTOR_SSE2_IMPL
// (2xk) x (kx2) matrices
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<(M!=K && M==N && M==2 && std::is_same<T,double>::value),bool>::type = 0>
FASTOR_INLINE
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    __m128d out_row0 = ZEROPD;
    __m128d out_row1 = ZEROPD;

    for (size_t i=0; i<K; ++i) {
        __m128d brow = _mm_loadu_pd(&b[i*2]);
#ifndef FASTOR_FMA_IMPL
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
    _mm_storeu_pd(out  ,out_row0);
    _mm_storeu_pd(out+2,out_row1);
}

// (2xk) x (kx2) matrices
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<(M!=K && M==N && M==2 && std::is_same<T,float>::value),bool>::type = 0>
FASTOR_INLINE
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    __m128 out_row0 = ZEROPS;
    __m128 out_row1 = ZEROPS;

    __m128 brow = ZEROPS;
    for (size_t i=0; i<K; i++) {
        // __m128 brow = _mm_loadu_ps(&b[i*2]);
        brow = _mm_loadl_pi(brow,(__m64*)&b[i*2]);
#ifndef FASTOR_FMA_IMPL
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
    _mm_storeu_ps(out,_mm_shuffle_ps(out_row0,out_row1,_MM_SHUFFLE(1,0,1,0)));
}
#else
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<M!=K && M==N && M==2,bool>::type = 0>
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {
    internal::_matmul_base<T,M,K,N>(a,b,out);
}
#endif

#ifdef FASTOR_AVX_IMPL
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
#ifndef FASTOR_FMA_IMPL
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
    _mm256_storeu_pd  (out,out_row0);
    _mm256_storeu_pd  (out+3,out_row1);
    _mm256_storeul3_pd(out+6,out_row2);
}

#else
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<M!=K && M==N && M==3 && std::is_same<T,double>::value,bool>::type = 0>
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {
    internal::_matmul_base<T,M,K,N>(a,b,out);
}
#endif

#ifdef FASTOR_SSE2_IMPL
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
#ifndef FASTOR_FMA_IMPL
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
    _mm_storeu_ps  (out,out_row0);
    _mm_storeu_ps  (&out[3],out_row1);
    _mm_storeul3_ps(&out[6],out_row2);
}
#else
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<(M!=K && M==N && M==3 && std::is_same<T,float>::value),bool>::type = 0>
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {
    internal::_matmul_base<T,M,K,N>(a,b,out);
}
#endif

#ifdef FASTOR_AVX_IMPL
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
        __m256d brow = _mm256_loadu_pd(&b[i*4]);
#ifndef FASTOR_FMA_IMPL
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
    _mm256_storeu_pd(&out[0] , out_row0);
    _mm256_storeu_pd(&out[4] , out_row1);
    _mm256_storeu_pd(&out[8] , out_row2);
    _mm256_storeu_pd(&out[12], out_row3);
}
#else
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<M!=K && M==N && M==4 && std::is_same<T,double>::value,bool>::type = 0>
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {
    internal::_matmul_base<T,M,K,N>(a,b,out);
}
#endif

#ifdef FASTOR_SSE2_IMPL
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
        __m128 brow = _mm_loadu_ps(&b[i*4]);
#ifndef FASTOR_FMA_IMPL
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
    _mm_storeu_ps(out,out_row0);
    _mm_storeu_ps(&out[4],out_row1);
    _mm_storeu_ps(&out[8],out_row2);
    _mm_storeu_ps(&out[12],out_row3);
}
#else
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<M!=K && M==N && M==4 && std::is_same<T,float>::value,bool>::type = 0>
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {
    internal::_matmul_base<T,M,K,N>(a,b,out);
}
#endif


// (8xk) x (kx8) matrices
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<M!=K && M==N && M==8 && std::is_same<T,double>::value,bool>::type = 0>
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {
#ifdef FASTOR_AVX512_IMPL
    FASTOR_IF_CONSTEXPR(K<=64)
        internal::_matmul8k8_double<T,M,K,N>(a,b,out);
    else
        internal::_matmul_mk_smalln<T,M,K,N>(a,b,out);
#else
    internal::_matmul_mk_smalln<T,M,K,N>(a,b,out);
#endif
}

template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<M!=K && M==N && M==8 && std::is_same<T,float>::value,bool>::type = 0>
void _matmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {
#ifdef FASTOR_AVX_IMPL
    internal::_matmul8k8_float<T,M,K,N>(a,b,out);
#else
    internal::_matmul_base<T,M,K,N>(a,b,out);
#endif
}





#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE
void _matmul<float,2,2,2>(const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {

    // Ivy 17 - Sky 16 [12 if fused] OPS
    __m128 ar  = _mm_loadu_ps(a);
    __m128 br  = _mm_loadu_ps(b);
    __m128 ar0 = _mm_shuffle_ps(ar,ar,_MM_SHUFFLE(2,2,0,0));
    __m128 ar1 = _mm_shuffle_ps(ar,ar,_MM_SHUFFLE(3,3,1,1));
    __m128 br0 = _mm_shuffle_ps(br,br,_MM_SHUFFLE(1,0,1,0));
    __m128 br1 = _mm_shuffle_ps(br,br,_MM_SHUFFLE(3,2,3,2));
    __m128 res = _mm_add_ps(_mm_mul_ps(ar0,br0),_mm_mul_ps(ar1,br1));
    _mm_storeu_ps(out,res);
}

template<>
FASTOR_INLINE
void _matmul<float,3,3,3>(const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {

    // Ivy 63 - HW 45 - Sky 36 OPS
    // Fully vectorised approach that reduces
    // (27 scalar mul + 18 scalar add) to
    // (9 SSE mul + 6 SEE add) [non-FMA]
    // (3 SSE mul + 6 SEE FMA) [    FMA]

    __m128 bmm0 = _mm_loadu_ps  (b  );
    __m128 bmm1 = _mm_loadu_ps  (b+3);
    __m128 bmm2 = _mm_loadul3_ps(b+6);

    __m128 omm0(_mm_mul_ps(_mm_set1_ps(a[0]),bmm0));
    __m128 omm1(_mm_mul_ps(_mm_set1_ps(a[3]),bmm0));
    __m128 omm2(_mm_mul_ps(_mm_set1_ps(a[6]),bmm0));
#ifndef FASTOR_FMA_IMPL
    omm0 = _mm_add_ps(omm0,_mm_mul_ps(_mm_set1_ps(a[1]),bmm1));
    omm1 = _mm_add_ps(omm1,_mm_mul_ps(_mm_set1_ps(a[4]),bmm1));
    omm2 = _mm_add_ps(omm2,_mm_mul_ps(_mm_set1_ps(a[7]),bmm1));

    omm0 = _mm_add_ps(omm0,_mm_mul_ps(_mm_set1_ps(a[2]),bmm2));
    omm1 = _mm_add_ps(omm1,_mm_mul_ps(_mm_set1_ps(a[5]),bmm2));
    omm2 = _mm_add_ps(omm2,_mm_mul_ps(_mm_set1_ps(a[8]),bmm2));
#else
    omm0 = _mm_fmadd_ps(_mm_set1_ps(a[1]),bmm1,omm0);
    omm1 = _mm_fmadd_ps(_mm_set1_ps(a[4]),bmm1,omm1);
    omm2 = _mm_fmadd_ps(_mm_set1_ps(a[7]),bmm1,omm2);

    omm0 = _mm_fmadd_ps(_mm_set1_ps(a[2]),bmm2,omm0);
    omm1 = _mm_fmadd_ps(_mm_set1_ps(a[5]),bmm2,omm1);
    omm2 = _mm_fmadd_ps(_mm_set1_ps(a[8]),bmm2,omm2);
#endif

    _mm_storeu_ps  (out  , omm0);
    _mm_storeu_ps  (out+3, omm1);
    _mm_storeul3_ps(out+6, omm2);
}

template<>
FASTOR_INLINE void _matmul<float,4,4,4>(const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {

    // Ivy 116 - HW 80 - Sky 64 OPS
    // Fully vectorised approach that reduces
    // (64 scalar mul + 48 scalar add) to
    // (16 SSE mul + 12 SEE add) [non-FMA]
    // (4  SSE mul + 12 SEE FMA) [    FMA]

    __m128 bmm0 = _mm_loadu_ps  (b   );
    __m128 bmm1 = _mm_loadu_ps  (b+4 );
    __m128 bmm2 = _mm_loadu_ps  (b+8 );
    __m128 bmm3 = _mm_loadu_ps  (b+12);

    __m128 omm0(_mm_mul_ps(_mm_set1_ps(a[0 ]),bmm0));
    __m128 omm1(_mm_mul_ps(_mm_set1_ps(a[4 ]),bmm0));
    __m128 omm2(_mm_mul_ps(_mm_set1_ps(a[8 ]),bmm0));
    __m128 omm3(_mm_mul_ps(_mm_set1_ps(a[12]),bmm0));
#ifndef FASTOR_FMA_IMPL
    omm0 = _mm_add_ps(omm0,_mm_mul_ps(_mm_set1_ps(a[1 ]),bmm1));
    omm1 = _mm_add_ps(omm1,_mm_mul_ps(_mm_set1_ps(a[5 ]),bmm1));
    omm2 = _mm_add_ps(omm2,_mm_mul_ps(_mm_set1_ps(a[9 ]),bmm1));
    omm3 = _mm_add_ps(omm3,_mm_mul_ps(_mm_set1_ps(a[13]),bmm1));

    omm0 = _mm_add_ps(omm0,_mm_mul_ps(_mm_set1_ps(a[2 ]),bmm2));
    omm1 = _mm_add_ps(omm1,_mm_mul_ps(_mm_set1_ps(a[6 ]),bmm2));
    omm2 = _mm_add_ps(omm2,_mm_mul_ps(_mm_set1_ps(a[10]),bmm2));
    omm3 = _mm_add_ps(omm3,_mm_mul_ps(_mm_set1_ps(a[14]),bmm2));

    omm0 = _mm_add_ps(omm0,_mm_mul_ps(_mm_set1_ps(a[3 ]),bmm3));
    omm1 = _mm_add_ps(omm1,_mm_mul_ps(_mm_set1_ps(a[7 ]),bmm3));
    omm2 = _mm_add_ps(omm2,_mm_mul_ps(_mm_set1_ps(a[11]),bmm3));
    omm3 = _mm_add_ps(omm3,_mm_mul_ps(_mm_set1_ps(a[15]),bmm3));
#else
    omm0 = _mm_fmadd_ps(_mm_set1_ps(a[1 ]),bmm1,omm0);
    omm1 = _mm_fmadd_ps(_mm_set1_ps(a[5 ]),bmm1,omm1);
    omm2 = _mm_fmadd_ps(_mm_set1_ps(a[9 ]),bmm1,omm2);
    omm3 = _mm_fmadd_ps(_mm_set1_ps(a[13]),bmm1,omm3);

    omm0 = _mm_fmadd_ps(_mm_set1_ps(a[2 ]),bmm2,omm0);
    omm1 = _mm_fmadd_ps(_mm_set1_ps(a[6 ]),bmm2,omm1);
    omm2 = _mm_fmadd_ps(_mm_set1_ps(a[10]),bmm2,omm2);
    omm3 = _mm_fmadd_ps(_mm_set1_ps(a[14]),bmm2,omm3);

    omm0 = _mm_fmadd_ps(_mm_set1_ps(a[3 ]),bmm3,omm0);
    omm1 = _mm_fmadd_ps(_mm_set1_ps(a[7 ]),bmm3,omm1);
    omm2 = _mm_fmadd_ps(_mm_set1_ps(a[11]),bmm3,omm2);
    omm3 = _mm_fmadd_ps(_mm_set1_ps(a[15]),bmm3,omm3);
#endif

    _mm_storeu_ps  (out   , omm0);
    _mm_storeu_ps  (out+4 , omm1);
    _mm_storeu_ps  (out+8 , omm2);
    _mm_storeu_ps  (out+12, omm3);
}
#endif

#ifdef FASTOR_AVX_IMPL
namespace internal {
// This is the common interface for 8k8 matmul and not only for 888 so do not
// make it specific to 888 floats
template<typename T, size_t M, size_t K, size_t N,
    typename std::enable_if<M==N && M==8 && std::is_same<T,float>::value,bool>::type>
FASTOR_INLINE
void _matmul8k8_float(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    __m256 out_row0 = VZEROPS;
    __m256 out_row1 = VZEROPS;
    __m256 out_row2 = VZEROPS;
    __m256 out_row3 = VZEROPS;
    __m256 out_row4 = VZEROPS;
    __m256 out_row5 = VZEROPS;
    __m256 out_row6 = VZEROPS;
    __m256 out_row7 = VZEROPS;

    for (size_t i=0; i<K; ++i) {
        __m256 brow = _mm256_load_ps(&b[i*8]);
#ifndef FASTOR_FMA_IMPL
        // row 0
        __m256 a_vec0 = _mm256_set1_ps(a[i]);
        out_row0 = _mm256_add_ps(out_row0,_mm256_mul_ps(a_vec0,brow));
        // row 1
        __m256 a_vec1 = _mm256_set1_ps(a[K+i]);
        out_row1 = _mm256_add_ps(out_row1,_mm256_mul_ps(a_vec1,brow));
        // row 2
        __m256 a_vec2 = _mm256_set1_ps(a[2*K+i]);
        out_row2 = _mm256_add_ps(out_row2,_mm256_mul_ps(a_vec2,brow));
        // row 3
        __m256 a_vec3 = _mm256_set1_ps(a[3*K+i]);
        out_row3 = _mm256_add_ps(out_row3,_mm256_mul_ps(a_vec3,brow));
        // row 4
        __m256 a_vec4 = _mm256_set1_ps(a[4*K+i]);
        out_row4 = _mm256_add_ps(out_row4,_mm256_mul_ps(a_vec4,brow));
        // row 5
        __m256 a_vec5 = _mm256_set1_ps(a[5*K+i]);
        out_row5 = _mm256_add_ps(out_row5,_mm256_mul_ps(a_vec5,brow));
        // row 6
        __m256 a_vec6 = _mm256_set1_ps(a[6*K+i]);
        out_row6 = _mm256_add_ps(out_row6,_mm256_mul_ps(a_vec6,brow));
        // row 7
        __m256 a_vec7 = _mm256_set1_ps(a[7*K+i]);
        out_row7 = _mm256_add_ps(out_row7,_mm256_mul_ps(a_vec7,brow));
#else
        // row 0
        __m256 a_vec0 = _mm256_set1_ps(a[i]);
        out_row0 = _mm256_fmadd_ps(a_vec0,brow,out_row0);
        // row 1
        __m256 a_vec1 = _mm256_set1_ps(a[K+i]);
        out_row1 = _mm256_fmadd_ps(a_vec1,brow,out_row1);
        // row 2
        __m256 a_vec2 = _mm256_set1_ps(a[2*K+i]);
        out_row2 = _mm256_fmadd_ps(a_vec2,brow,out_row2);
        // row 3
        __m256 a_vec3 = _mm256_set1_ps(a[3*K+i]);
        out_row3 = _mm256_fmadd_ps(a_vec3,brow,out_row3);
        // row 4
        __m256 a_vec4 = _mm256_set1_ps(a[4*K+i]);
        out_row4 = _mm256_fmadd_ps(a_vec4,brow,out_row4);
        // row 5
        __m256 a_vec5 = _mm256_set1_ps(a[5*K+i]);
        out_row5 = _mm256_fmadd_ps(a_vec5,brow,out_row5);
        // row 6
        __m256 a_vec6 = _mm256_set1_ps(a[6*K+i]);
        out_row6 = _mm256_fmadd_ps(a_vec6,brow,out_row6);
        // row 7
        __m256 a_vec7 = _mm256_set1_ps(a[7*K+i]);
        out_row7 = _mm256_fmadd_ps(a_vec7,brow,out_row7);
#endif
    }
    _mm256_store_ps(out,out_row0);
    _mm256_store_ps(&out[8],out_row1);
    _mm256_store_ps(&out[16],out_row2);
    _mm256_store_ps(&out[24],out_row3);
    _mm256_store_ps(&out[32],out_row4);
    _mm256_store_ps(&out[40],out_row5);
    _mm256_store_ps(&out[48],out_row6);
    _mm256_store_ps(&out[56],out_row7);
}

} // internal

template<>
FASTOR_INLINE void _matmul<float,8,8,8>(const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {
    internal::_matmul8k8_float<float,8,8,8>(a,b,out);
    return;
}
#endif


#ifdef FASTOR_AVX512_IMPL
namespace internal {
template<typename T, size_t M, size_t K, size_t N,
    typename std::enable_if<M==N && M==8 && std::is_same<T,double>::value,bool>::type>
FASTOR_INLINE
void _matmul8k8_double(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    __m512d out_row0 = _mm512_setzero_pd();
    __m512d out_row1 = _mm512_setzero_pd();
    __m512d out_row2 = _mm512_setzero_pd();
    __m512d out_row3 = _mm512_setzero_pd();
    __m512d out_row4 = _mm512_setzero_pd();
    __m512d out_row5 = _mm512_setzero_pd();
    __m512d out_row6 = _mm512_setzero_pd();
    __m512d out_row7 = _mm512_setzero_pd();

    for (size_t i=0; i<K; ++i) {
        __m512d brow = _mm512_loadu_pd(&b[i*8]);
        // row 0
        __m512d a_vec0 = _mm512_set1_pd(a[i]);
        out_row0 = _mm512_fmadd_pd(a_vec0,brow,out_row0);
        // row 1
        __m512d a_vec1 = _mm512_set1_pd(a[K+i]);
        out_row1 = _mm512_fmadd_pd(a_vec1,brow,out_row1);
        // row 2
        __m512d a_vec2 = _mm512_set1_pd(a[2*K+i]);
        out_row2 = _mm512_fmadd_pd(a_vec2,brow,out_row2);
        // row 3
        __m512d a_vec3 = _mm512_set1_pd(a[3*K+i]);
        out_row3 = _mm512_fmadd_pd(a_vec3,brow,out_row3);
        // row 4
        __m512d a_vec4 = _mm512_set1_pd(a[4*K+i]);
        out_row4 = _mm512_fmadd_pd(a_vec4,brow,out_row4);
        // row 5
        __m512d a_vec5 = _mm512_set1_pd(a[5*K+i]);
        out_row5 = _mm512_fmadd_pd(a_vec5,brow,out_row5);
        // row 6
        __m512d a_vec6 = _mm512_set1_pd(a[6*K+i]);
        out_row6 = _mm512_fmadd_pd(a_vec6,brow,out_row6);
        // row 7
        __m512d a_vec7 = _mm512_set1_pd(a[7*K+i]);
        out_row7 = _mm512_fmadd_pd(a_vec7,brow,out_row7);
    }
    _mm512_storeu_pd(out,out_row0);
    _mm512_storeu_pd(&out[8],out_row1);
    _mm512_storeu_pd(&out[16],out_row2);
    _mm512_storeu_pd(&out[24],out_row3);
    _mm512_storeu_pd(&out[32],out_row4);
    _mm512_storeu_pd(&out[40],out_row5);
    _mm512_storeu_pd(&out[48],out_row6);
    _mm512_storeu_pd(&out[56],out_row7);
}
} // internal

template<>
FASTOR_INLINE void _matmul<double,8,8,8>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {
    internal::_matmul8k8_double<double,8,8,8>(a,b,out);
    return;
}
#endif


#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE
void _matmul<double,2,2,2>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {

    // Ivy 26 - HW 20 - Sky 16 OPS

    __m128d bmm0 = _mm_loadu_pd  (b  );
    __m128d bmm1 = _mm_loadu_pd  (b+2);

    __m128d omm0(_mm_mul_pd(_mm_set1_pd(a[0]),bmm0));
    __m128d omm1(_mm_mul_pd(_mm_set1_pd(a[2]),bmm0));
#ifndef FASTOR_FMA_IMPL
    omm0 = _mm_add_pd(omm0,_mm_mul_pd(_mm_set1_pd(a[1]),bmm1));
    omm1 = _mm_add_pd(omm1,_mm_mul_pd(_mm_set1_pd(a[3]),bmm1));
#else
    omm0 = _mm_fmadd_pd(_mm_set1_pd(a[1]),bmm1,omm0);
    omm1 = _mm_fmadd_pd(_mm_set1_pd(a[3]),bmm1,omm1);
#endif
    _mm_storeu_pd  (out  , omm0);
    _mm_storeu_pd  (out+2, omm1);
}
#endif

#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE
void _matmul<double,3,3,3>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {

    // Ivy 63 - HW 45 - Sky 36 OPS
    // Fully vectorised approach that reduces
    // (27 scalar mul + 18 scalar add) to
    // (9 AVX mul + 6 AVX add) [non-FMA]
    // (3 AVX mul + 6 AVX FMA) [    FMA]

    __m256d bmm0 = _mm256_loadu_pd  (b  );
    __m256d bmm1 = _mm256_loadu_pd  (b+3);
    __m256d bmm2 = _mm256_loadul3_pd(b+6);

    __m256d omm0(_mm256_mul_pd(_mm256_set1_pd(a[0]),bmm0));
    __m256d omm1(_mm256_mul_pd(_mm256_set1_pd(a[3]),bmm0));
    __m256d omm2(_mm256_mul_pd(_mm256_set1_pd(a[6]),bmm0));
#ifndef FASTOR_FMA_IMPL
    omm0 = _mm256_add_pd(omm0,_mm256_mul_pd(_mm256_set1_pd(a[1]),bmm1));
    omm1 = _mm256_add_pd(omm1,_mm256_mul_pd(_mm256_set1_pd(a[4]),bmm1));
    omm2 = _mm256_add_pd(omm2,_mm256_mul_pd(_mm256_set1_pd(a[7]),bmm1));

    omm0 = _mm256_add_pd(omm0,_mm256_mul_pd(_mm256_set1_pd(a[2]),bmm2));
    omm1 = _mm256_add_pd(omm1,_mm256_mul_pd(_mm256_set1_pd(a[5]),bmm2));
    omm2 = _mm256_add_pd(omm2,_mm256_mul_pd(_mm256_set1_pd(a[8]),bmm2));
#else
    omm0 = _mm256_fmadd_pd(_mm256_set1_pd(a[1]),bmm1,omm0);
    omm1 = _mm256_fmadd_pd(_mm256_set1_pd(a[4]),bmm1,omm1);
    omm2 = _mm256_fmadd_pd(_mm256_set1_pd(a[7]),bmm1,omm2);

    omm0 = _mm256_fmadd_pd(_mm256_set1_pd(a[2]),bmm2,omm0);
    omm1 = _mm256_fmadd_pd(_mm256_set1_pd(a[5]),bmm2,omm1);
    omm2 = _mm256_fmadd_pd(_mm256_set1_pd(a[8]),bmm2,omm2);
#endif

    _mm256_storeu_pd  (out  , omm0);
    _mm256_storeu_pd  (out+3, omm1);
    _mm256_storeul3_pd(out+6, omm2);
}

template<>
FASTOR_INLINE
void _matmul<double,4,4,4>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {

    // Ivy 116 - HW 80 - Sky 64 OPS
    // Fully vectorised approach that reduces
    // (64 scalar mul + 48 scalar add) to
    // (16 SSE mul + 12 SEE add) [non-FMA]
    // (4  SSE mul + 12 SEE FMA) [    FMA]

    __m256d bmm0 = _mm256_loadu_pd  (b   );
    __m256d bmm1 = _mm256_loadu_pd  (b+4 );
    __m256d bmm2 = _mm256_loadu_pd  (b+8 );
    __m256d bmm3 = _mm256_loadu_pd  (b+12);

    __m256d omm0(_mm256_mul_pd(_mm256_set1_pd(a[0 ]),bmm0));
    __m256d omm1(_mm256_mul_pd(_mm256_set1_pd(a[4 ]),bmm0));
    __m256d omm2(_mm256_mul_pd(_mm256_set1_pd(a[8 ]),bmm0));
    __m256d omm3(_mm256_mul_pd(_mm256_set1_pd(a[12]),bmm0));
#ifndef FASTOR_FMA_IMPL
    omm0 = _mm256_add_pd(omm0,_mm256_mul_pd(_mm256_set1_pd(a[1 ]),bmm1));
    omm1 = _mm256_add_pd(omm1,_mm256_mul_pd(_mm256_set1_pd(a[5 ]),bmm1));
    omm2 = _mm256_add_pd(omm2,_mm256_mul_pd(_mm256_set1_pd(a[9 ]),bmm1));
    omm3 = _mm256_add_pd(omm3,_mm256_mul_pd(_mm256_set1_pd(a[13]),bmm1));

    omm0 = _mm256_add_pd(omm0,_mm256_mul_pd(_mm256_set1_pd(a[2 ]),bmm2));
    omm1 = _mm256_add_pd(omm1,_mm256_mul_pd(_mm256_set1_pd(a[6 ]),bmm2));
    omm2 = _mm256_add_pd(omm2,_mm256_mul_pd(_mm256_set1_pd(a[10]),bmm2));
    omm3 = _mm256_add_pd(omm3,_mm256_mul_pd(_mm256_set1_pd(a[14]),bmm2));

    omm0 = _mm256_add_pd(omm0,_mm256_mul_pd(_mm256_set1_pd(a[3 ]),bmm3));
    omm1 = _mm256_add_pd(omm1,_mm256_mul_pd(_mm256_set1_pd(a[7 ]),bmm3));
    omm2 = _mm256_add_pd(omm2,_mm256_mul_pd(_mm256_set1_pd(a[11]),bmm3));
    omm3 = _mm256_add_pd(omm3,_mm256_mul_pd(_mm256_set1_pd(a[15]),bmm3));
#else
    omm0 = _mm256_fmadd_pd(_mm256_set1_pd(a[1 ]),bmm1,omm0);
    omm1 = _mm256_fmadd_pd(_mm256_set1_pd(a[5 ]),bmm1,omm1);
    omm2 = _mm256_fmadd_pd(_mm256_set1_pd(a[9 ]),bmm1,omm2);
    omm3 = _mm256_fmadd_pd(_mm256_set1_pd(a[13]),bmm1,omm3);

    omm0 = _mm256_fmadd_pd(_mm256_set1_pd(a[2 ]),bmm2,omm0);
    omm1 = _mm256_fmadd_pd(_mm256_set1_pd(a[6 ]),bmm2,omm1);
    omm2 = _mm256_fmadd_pd(_mm256_set1_pd(a[10]),bmm2,omm2);
    omm3 = _mm256_fmadd_pd(_mm256_set1_pd(a[14]),bmm2,omm3);

    omm0 = _mm256_fmadd_pd(_mm256_set1_pd(a[3 ]),bmm3,omm0);
    omm1 = _mm256_fmadd_pd(_mm256_set1_pd(a[7 ]),bmm3,omm1);
    omm2 = _mm256_fmadd_pd(_mm256_set1_pd(a[11]),bmm3,omm2);
    omm3 = _mm256_fmadd_pd(_mm256_set1_pd(a[15]),bmm3,omm3);
#endif

    _mm256_storeu_pd  (out   , omm0);
    _mm256_storeu_pd  (out+4 , omm1);
    _mm256_storeu_pd  (out+8 , omm2);
    _mm256_storeu_pd  (out+12, omm3);
}
#endif





//! Matrix-vector multiplication
//-----------------------------------------------------------------------------------------------------------
// Don't call this function directly as it's name is unconventional
// It gets called from within matmul anyway so always call matmul
namespace internal {

template<typename T, size_t M, size_t N>
FASTOR_INLINE
void _matvecmul(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    using V = typename choose_best_simd_type<SIMDVector<T,DEFAULT_ABI>,N>::type;
    constexpr size_t unrollOuterloop = 8UL;
    constexpr size_t M0 = M / unrollOuterloop * unrollOuterloop;

    // Number of columns of c (N) that can be safely unrolled with V::Size
    constexpr size_t N1 = N / V::Size * V::Size;

    constexpr bool isAAligned = false;
    constexpr bool isBAligned = false;

    size_t i=0;
    FASTOR_IF_CONSTEXPR(N < V::Size) {

        for (; i<ROUND_DOWN(M,2UL); i+=2UL) {
            V omm0, omm1;
            size_t j = 0;
            for (; j< N1; j+=V::Size) {
                V amm0(&a[(i    )*N+j],false);
                V amm1(&a[(i+1UL)*N+j],false);
                V bmm0(&b[j],false);

                omm0 = fmadd(amm0,bmm0,omm0);
                omm1 = fmadd(amm1,bmm0,omm1);
            }
            T out_s0 = 0;
            T out_s1 = 0;
            for (; j< N; ++j) {
                out_s0 += a[i*N+j]*b[j];
                out_s1 += a[(i+1)*N+j]*b[j];
            }
            out[i]   = omm0.sum() + out_s0;
            out[i+1] = omm1.sum() + out_s1;
        }

        for (; i<M; ++i) {
            V omm0;
            size_t j = 0;
            for (; j< N1; j+=V::Size) {
                const V amm0(&a[i*N+j],false);
                const V bmm0(&b[j],false);

                omm0 = fmadd(amm0,bmm0,omm0);
            }
            T out_s0 = 0;
            for (; j< N; j+=1) {
                out_s0 += a[i*N+j]*b[j];
            }
            out[i]  = omm0.sum() + out_s0;
        }
        return;
    }

    else
    {

        for (; i<M0; i+=unrollOuterloop) {

            const V bmm0(&b[0],isBAligned);

            V omm0(V(&a[(i    )*N],isAAligned)*bmm0);
            V omm1(V(&a[(i+1UL)*N],isAAligned)*bmm0);
            V omm2(V(&a[(i+2UL)*N],isAAligned)*bmm0);
            V omm3(V(&a[(i+3UL)*N],isAAligned)*bmm0);
            V omm4(V(&a[(i+4UL)*N],isAAligned)*bmm0);
            V omm5(V(&a[(i+5UL)*N],isAAligned)*bmm0);
            V omm6(V(&a[(i+6UL)*N],isAAligned)*bmm0);
            V omm7(V(&a[(i+7UL)*N],isAAligned)*bmm0);

            size_t j = V::Size;
            for (; j< N1; j+=V::Size) {

                const V bmm0(&b[j],isBAligned);

                const V amm0(&a[(i    )*N+j],isAAligned);
                const V amm1(&a[(i+1UL)*N+j],isAAligned);
                const V amm2(&a[(i+2UL)*N+j],isAAligned);
                const V amm3(&a[(i+3UL)*N+j],isAAligned);
                const V amm4(&a[(i+4UL)*N+j],isAAligned);
                const V amm5(&a[(i+5UL)*N+j],isAAligned);
                const V amm6(&a[(i+6UL)*N+j],isAAligned);
                const V amm7(&a[(i+7UL)*N+j],isAAligned);

                omm0 = fmadd(amm0,bmm0,omm0);
                omm1 = fmadd(amm1,bmm0,omm1);
                omm2 = fmadd(amm2,bmm0,omm2);
                omm3 = fmadd(amm3,bmm0,omm3);
                omm4 = fmadd(amm4,bmm0,omm4);
                omm5 = fmadd(amm5,bmm0,omm5);
                omm6 = fmadd(amm6,bmm0,omm6);
                omm7 = fmadd(amm7,bmm0,omm7);
            }

            out[i    ] = omm0.sum();
            out[i+1UL] = omm1.sum();
            out[i+2UL] = omm2.sum();
            out[i+3UL] = omm3.sum();
            out[i+4UL] = omm4.sum();
            out[i+5UL] = omm5.sum();
            out[i+6UL] = omm6.sum();
            out[i+7UL] = omm7.sum();

            for (; j< N; ++j) {
                const T bmm0(b[j]);
                out[i    ] += a[(i    )*N+j]*bmm0;
                out[i+1UL] += a[(i+1UL)*N+j]*bmm0;
                out[i+2UL] += a[(i+2UL)*N+j]*bmm0;
                out[i+3UL] += a[(i+3UL)*N+j]*bmm0;
                out[i+4UL] += a[(i+4UL)*N+j]*bmm0;
                out[i+5UL] += a[(i+5UL)*N+j]*bmm0;
                out[i+6UL] += a[(i+6UL)*N+j]*bmm0;
                out[i+7UL] += a[(i+7UL)*N+j]*bmm0;
            }
        }

        FASTOR_IF_CONSTEXPR (M - M0 == 7) {

            const V bmm0(&b[0],isBAligned);

            V omm0(V(&a[(i    )*N],isAAligned)*bmm0);
            V omm1(V(&a[(i+1UL)*N],isAAligned)*bmm0);
            V omm2(V(&a[(i+2UL)*N],isAAligned)*bmm0);
            V omm3(V(&a[(i+3UL)*N],isAAligned)*bmm0);
            V omm4(V(&a[(i+4UL)*N],isAAligned)*bmm0);
            V omm5(V(&a[(i+5UL)*N],isAAligned)*bmm0);
            V omm6(V(&a[(i+6UL)*N],isAAligned)*bmm0);

            size_t j = V::Size;
            for (; j< N1; j+=V::Size) {

                const V bmm0(&b[j],isBAligned);

                const V amm0(&a[(i    )*N+j],isAAligned);
                const V amm1(&a[(i+1UL)*N+j],isAAligned);
                const V amm2(&a[(i+2UL)*N+j],isAAligned);
                const V amm3(&a[(i+3UL)*N+j],isAAligned);
                const V amm4(&a[(i+4UL)*N+j],isAAligned);
                const V amm5(&a[(i+5UL)*N+j],isAAligned);
                const V amm6(&a[(i+6UL)*N+j],isAAligned);

                omm0 = fmadd(amm0,bmm0,omm0);
                omm1 = fmadd(amm1,bmm0,omm1);
                omm2 = fmadd(amm2,bmm0,omm2);
                omm3 = fmadd(amm3,bmm0,omm3);
                omm4 = fmadd(amm4,bmm0,omm4);
                omm5 = fmadd(amm5,bmm0,omm5);
                omm6 = fmadd(amm6,bmm0,omm6);
            }

            out[i    ] = omm0.sum();
            out[i+1UL] = omm1.sum();
            out[i+2UL] = omm2.sum();
            out[i+3UL] = omm3.sum();
            out[i+4UL] = omm4.sum();
            out[i+5UL] = omm5.sum();
            out[i+6UL] = omm6.sum();

            for (; j< N; ++j) {
                const T bmm0(b[j]);
                out[i    ] += a[(i    )*N+j]*bmm0;
                out[i+1UL] += a[(i+1UL)*N+j]*bmm0;
                out[i+2UL] += a[(i+2UL)*N+j]*bmm0;
                out[i+3UL] += a[(i+3UL)*N+j]*bmm0;
                out[i+4UL] += a[(i+4UL)*N+j]*bmm0;
                out[i+5UL] += a[(i+5UL)*N+j]*bmm0;
                out[i+6UL] += a[(i+6UL)*N+j]*bmm0;
            }
        }

        else FASTOR_IF_CONSTEXPR (M - M0 == 6) {

            const V bmm0(&b[0],isBAligned);

            V omm0(V(&a[(i    )*N],isAAligned)*bmm0);
            V omm1(V(&a[(i+1UL)*N],isAAligned)*bmm0);
            V omm2(V(&a[(i+2UL)*N],isAAligned)*bmm0);
            V omm3(V(&a[(i+3UL)*N],isAAligned)*bmm0);
            V omm4(V(&a[(i+4UL)*N],isAAligned)*bmm0);
            V omm5(V(&a[(i+5UL)*N],isAAligned)*bmm0);

            size_t j = V::Size;
            for (; j< N1; j+=V::Size) {

                const V bmm0(&b[j],isBAligned);

                const V amm0(&a[(i    )*N+j],isAAligned);
                const V amm1(&a[(i+1UL)*N+j],isAAligned);
                const V amm2(&a[(i+2UL)*N+j],isAAligned);
                const V amm3(&a[(i+3UL)*N+j],isAAligned);
                const V amm4(&a[(i+4UL)*N+j],isAAligned);
                const V amm5(&a[(i+5UL)*N+j],isAAligned);

                omm0 = fmadd(amm0,bmm0,omm0);
                omm1 = fmadd(amm1,bmm0,omm1);
                omm2 = fmadd(amm2,bmm0,omm2);
                omm3 = fmadd(amm3,bmm0,omm3);
                omm4 = fmadd(amm4,bmm0,omm4);
                omm5 = fmadd(amm5,bmm0,omm5);
            }

            out[i    ] = omm0.sum();
            out[i+1UL] = omm1.sum();
            out[i+2UL] = omm2.sum();
            out[i+3UL] = omm3.sum();
            out[i+4UL] = omm4.sum();
            out[i+5UL] = omm5.sum();

            for (; j< N; ++j) {
                const T bmm0(b[j]);
                out[i    ] += a[(i    )*N+j]*bmm0;
                out[i+1UL] += a[(i+1UL)*N+j]*bmm0;
                out[i+2UL] += a[(i+2UL)*N+j]*bmm0;
                out[i+3UL] += a[(i+3UL)*N+j]*bmm0;
                out[i+4UL] += a[(i+4UL)*N+j]*bmm0;
                out[i+5UL] += a[(i+5UL)*N+j]*bmm0;
            }
        }

        else FASTOR_IF_CONSTEXPR (M - M0 == 5) {

            const V bmm0(&b[0],isBAligned);

            V omm0(V(&a[(i    )*N],isAAligned)*bmm0);
            V omm1(V(&a[(i+1UL)*N],isAAligned)*bmm0);
            V omm2(V(&a[(i+2UL)*N],isAAligned)*bmm0);
            V omm3(V(&a[(i+3UL)*N],isAAligned)*bmm0);
            V omm4(V(&a[(i+4UL)*N],isAAligned)*bmm0);

            size_t j = V::Size;
            for (; j< N1; j+=V::Size) {

                const V bmm0(&b[j],isBAligned);

                const V amm0(&a[(i    )*N+j],isAAligned);
                const V amm1(&a[(i+1UL)*N+j],isAAligned);
                const V amm2(&a[(i+2UL)*N+j],isAAligned);
                const V amm3(&a[(i+3UL)*N+j],isAAligned);
                const V amm4(&a[(i+4UL)*N+j],isAAligned);

                omm0 = fmadd(amm0,bmm0,omm0);
                omm1 = fmadd(amm1,bmm0,omm1);
                omm2 = fmadd(amm2,bmm0,omm2);
                omm3 = fmadd(amm3,bmm0,omm3);
                omm4 = fmadd(amm4,bmm0,omm4);
            }

            out[i    ] = omm0.sum();
            out[i+1UL] = omm1.sum();
            out[i+2UL] = omm2.sum();
            out[i+3UL] = omm3.sum();
            out[i+4UL] = omm4.sum();

            for (; j< N; ++j) {
                const T bmm0(b[j]);
                out[i    ] += a[(i    )*N+j]*bmm0;
                out[i+1UL] += a[(i+1UL)*N+j]*bmm0;
                out[i+2UL] += a[(i+2UL)*N+j]*bmm0;
                out[i+3UL] += a[(i+3UL)*N+j]*bmm0;
                out[i+4UL] += a[(i+4UL)*N+j]*bmm0;
            }
        }

        else FASTOR_IF_CONSTEXPR (M - M0 == 4) {

            const V bmm0(&b[0],isBAligned);

            V omm0(V(&a[(i    )*N],isAAligned)*bmm0);
            V omm1(V(&a[(i+1UL)*N],isAAligned)*bmm0);
            V omm2(V(&a[(i+2UL)*N],isAAligned)*bmm0);
            V omm3(V(&a[(i+3UL)*N],isAAligned)*bmm0);

            size_t j = V::Size;
            for (; j< N1; j+=V::Size) {

                const V bmm0(&b[j],isBAligned);

                const V amm0(&a[(i    )*N+j],isAAligned);
                const V amm1(&a[(i+1UL)*N+j],isAAligned);
                const V amm2(&a[(i+2UL)*N+j],isAAligned);
                const V amm3(&a[(i+3UL)*N+j],isAAligned);

                omm0 = fmadd(amm0,bmm0,omm0);
                omm1 = fmadd(amm1,bmm0,omm1);
                omm2 = fmadd(amm2,bmm0,omm2);
                omm3 = fmadd(amm3,bmm0,omm3);
            }

            out[i    ] = omm0.sum();
            out[i+1UL] = omm1.sum();
            out[i+2UL] = omm2.sum();
            out[i+3UL] = omm3.sum();

            for (; j< N; ++j) {
                const T bmm0(b[j]);
                out[i    ] += a[(i    )*N+j]*bmm0;
                out[i+1UL] += a[(i+1UL)*N+j]*bmm0;
                out[i+2UL] += a[(i+2UL)*N+j]*bmm0;
                out[i+3UL] += a[(i+3UL)*N+j]*bmm0;
            }
        }

        else FASTOR_IF_CONSTEXPR (M - M0 == 3) {

            const V bmm0(&b[0],isBAligned);

            V omm0(V(&a[(i    )*N],isAAligned)*bmm0);
            V omm1(V(&a[(i+1UL)*N],isAAligned)*bmm0);
            V omm2(V(&a[(i+2UL)*N],isAAligned)*bmm0);

            size_t j = V::Size;
            for (; j< N1; j+=V::Size) {

                const V bmm0(&b[j],isBAligned);

                const V amm0(&a[(i    )*N+j],isAAligned);
                const V amm1(&a[(i+1UL)*N+j],isAAligned);
                const V amm2(&a[(i+2UL)*N+j],isAAligned);

                omm0 = fmadd(amm0,bmm0,omm0);
                omm1 = fmadd(amm1,bmm0,omm1);
                omm2 = fmadd(amm2,bmm0,omm2);
            }

            out[i    ] = omm0.sum();
            out[i+1UL] = omm1.sum();
            out[i+2UL] = omm2.sum();

            for (; j< N; ++j) {
                const T bmm0(b[j]);
                out[i    ] += a[(i    )*N+j]*bmm0;
                out[i+1UL] += a[(i+1UL)*N+j]*bmm0;
                out[i+2UL] += a[(i+2UL)*N+j]*bmm0;
            }
        }

        else FASTOR_IF_CONSTEXPR (M - M0 == 2) {

            const V bmm0(&b[0],isBAligned);

            V omm0(V(&a[(i    )*N],isAAligned)*bmm0);
            V omm1(V(&a[(i+1UL)*N],isAAligned)*bmm0);

            size_t j = V::Size;
            for (; j< N1; j+=V::Size) {

                const V bmm0(&b[j],isBAligned);

                const V amm0(&a[(i    )*N+j],isAAligned);
                const V amm1(&a[(i+1UL)*N+j],isAAligned);

                omm0 = fmadd(amm0,bmm0,omm0);
                omm1 = fmadd(amm1,bmm0,omm1);
            }

            out[i    ] = omm0.sum();
            out[i+1UL] = omm1.sum();

            for (; j< N; ++j) {
                const T bmm0(b[j]);
                out[i    ] += a[(i    )*N+j]*bmm0;
                out[i+1UL] += a[(i+1UL)*N+j]*bmm0;
            }
        }

        else FASTOR_IF_CONSTEXPR (M - M0 == 1) {

            const V bmm0(&b[0],isBAligned);

            V omm0(V(&a[(i    )*N],isAAligned)*bmm0);

            size_t j = V::Size;
            for (; j< N1; j+=V::Size) {

                const V bmm0(&b[j],isBAligned);

                const V amm0(&a[(i    )*N+j],isAAligned);

                omm0 = fmadd(amm0,bmm0,omm0);
            }

            out[i    ] = omm0.sum();

            for (; j< N; ++j) {
                const T bmm0(b[j]);
                out[i    ] += a[(i    )*N+j]*bmm0;
            }
        }
    }
}

}


#ifdef FASTOR_SSE2_IMPL
template<>
FASTOR_INLINE void _matmul<float,2,2,1>(const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {
    // 11 OPS
    __m128 amm = _mm_loadu_ps(a);
    __m128 bmm = ZEROPS;
    bmm = _mm_loadl_pi(bmm, (__m64*)b);

    __m128 res   = _mm_mul_ps(amm,_mm_movelh_ps(bmm,bmm));
    __m128 res2  = _mm_shuffle_ps(res ,res ,_MM_SHUFFLE(2,3,0,1));
    res2         = _mm_add_ps(res,res2);
    __m128 res3  = _mm_shuffle_ps(res2,res2,_MM_SHUFFLE(3,1,2,0));
    _mm_storel_pi((__m64*) out,res3);
}

template<>
FASTOR_INLINE void _matmul<float,3,3,1>(const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {
    // IVY/HW 47 OPS
    // Mask loads on AVX and AVX512 otherwise 3 single loads each
    __m128 amm0 = _mm_loadul3_ps(a);
    __m128 amm1 = _mm_loadul3_ps(&a[3]);
    __m128 amm2 = _mm_loadul3_ps(&a[6]);
    __m128 bmm  = _mm_loadul3_ps(b);

    out[0] = _mm_sum_ps(_mm_mul_ps(amm0,bmm));
    out[1] = _mm_sum_ps(_mm_mul_ps(amm1,bmm));
    out[2] = _mm_sum_ps(_mm_mul_ps(amm2,bmm));
}

template<>
FASTOR_INLINE void _matmul<double,2,2,1>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {
    // IVY/HW/SKY 19 OPS
    __m128d amm0 = _mm_loadu_pd(a  );
    __m128d amm1 = _mm_loadu_pd(a+2);
    __m128d bmm0 = _mm_loadu_pd(b  );

    __m128d omm0 = _mm_mul_pd(amm0,bmm0);
    __m128d omm1 = _mm_mul_pd(amm1,bmm0);
    omm0 = _mm_add_pd(omm0, _mm_reverse_pd(omm0));
    omm1 = _mm_add_pd(omm1, _mm_reverse_pd(omm1));

    _mm_storeu_pd(out,_mm_shuffle_pd(omm0,omm1,0x1));
}
#endif


#ifdef FASTOR_AVX_IMPL
template<>
FASTOR_INLINE void _matmul<double,3,3,1>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {
    // SKY 49 - ICE 87 OPS
    // Mask loads on AVX and AVX512
    __m256d amm0 = _mm256_loadul3_pd(a);
    __m256d amm1 = _mm256_loadul3_pd(&a[3]);
    __m256d amm2 = _mm256_loadul3_pd(&a[6]);
    __m256d bmm  = _mm256_loadul3_pd(b);

    out[0] = _mm256_sum_pd(_mm256_mul_pd(amm0,bmm));
    out[1] = _mm256_sum_pd(_mm256_mul_pd(amm1,bmm));
    out[2] = _mm256_sum_pd(_mm256_mul_pd(amm2,bmm));
}
#endif

#endif // MATMUL_SPECIALISATIONS_KERNELS_H
