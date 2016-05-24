#ifndef TRANSPOSE_H
#define TRANSPOSE_H

template<typename T, size_t M, size_t N>
void _transpose(const T * __restrict__ a, T * __restrict__ out) {
    for (size_t i=0; i< M; ++i)
        for (size_t j=0; j<N; ++j)
            a[i*N+j] = a[j*N+i];
}

template<>
void _transpose<float,2,2>(const float * __restrict__ a, float * __restrict__ out) {
    __m128 a_reg = _mm_load_ps(a);
    _mm_store_ps(out,_mm_shuffle_ps(a_reg,a_reg,_MM_SHUFFLE(3,1,2,0)));
}

template<>
void _transpose<double,2,2>(const double* __restrict__ a, double* __restrict__ out) {
    // IVY 4 OPS / HW 8 OPS
    __m256d a1 =  _mm256_load_pd(a);
    __m128d a2 =  _mm256_castpd256_pd128(a1);
    __m128d a3 =  _mm256_extractf128_pd(a1,0x1);
    a2 = _mm_shuffle_pd(a2,a3,0x1);
    a3 = _mm_shuffle_pd(a2,a3,0x2);
    a1 = _mm256_castpd128_pd256(a2);
    a1 = _mm256_insertf128_pd(a1,a3,0x1);
    _mm256_store_pd(out,a1);
}

template<>
void _transpose<double,3,3>(const double* __restrict__ a, double* __restrict__ out) {
    // Note that for single precision there is a direct way of doing this
    // by using _MM_TRANSPOSE4_PS


    /*-------------------------------------------------------*/
    // SSE VERSION - Requires 32byte alignment
    // all loads are 16 byte aligned if a is 32byte aligned
    __m128d a11 = _mm_load_pd(a);
    __m128d a12 = _mm_load_pd(a+2);
    __m128d a21 = _mm_load_pd(a+4);
    __m128d a22 = _mm_load_pd(a+6);

    // all stores are aligned
    _mm_store_pd(out,_mm_shuffle_pd(a11,a12,0x2));
    _mm_storer_pd(out+2,_mm_shuffle_pd(a11,a22,0x1));
    _mm_store_pd(out+4,_mm_shuffle_pd(a21,a22,0x2));
    _mm_store_pd(out+6,_mm_shuffle_pd(a12,a21,0x2));
    _mm_store_sd(out+8,_mm_load_sd(a+8));
    /*-------------------------------------------------------*/

//    /*-------------------------------------------------------*/
//    // AVX VERSION - NOTE THAT AVX cannot shuffle across 128bit boundaries
//    // so the AVX version requires more instruction although number of load/store
//    // is reduced to 6
//    __m256d row1 = _mm256_load_pd(a);
//    __m256d row2 = _mm256_load_pd(a+4);

//    __m128d a11 = _mm256_extractf128_pd(row1,0x0);
//    __m128d a12 = _mm256_extractf128_pd(row1,0x1);
//    __m128d a21 = _mm256_extractf128_pd(row2,0x0);
//    __m128d a22 = _mm256_extractf128_pd(row2,0x1);

//    row1 = _mm256_insertf128_pd(row1,_mm_shuffle_pd(a11,a12,0x2),0x0);
//    row1 = _mm256_insertf128_pd(row1,_mm_shuffle_pd(a22,a11,0x2),0x1);
//    row2 = _mm256_insertf128_pd(row2,_mm_shuffle_pd(a21,a22,0x2),0x0);
//    row2 = _mm256_insertf128_pd(row2,_mm_shuffle_pd(a12,a21,0x2),0x1);

//    _mm256_store_pd(out,row1);
//    _mm256_store_pd(out+4,row2);
//    _mm_store_sd(out+8,_mm_load_sd(a+8));
//    /*-------------------------------------------------------*/
}

#endif // TRANSPOSE_H

