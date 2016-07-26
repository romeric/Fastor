#ifndef MATMUL_H
#define MATMUL_H

#include "commons/commons.h"
#include "extended_intrinsics/extintrin.h"
#include "simd_vector/SIMDVector.h"

namespace Fastor {



// For square matrices
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<M==N && M==K && N % SIMDVector<T>::Size ==0,bool>::type = 0>
FASTOR_INLINE
void _matmul(const T * __restrict__ a, const T * __restrict__ b, T * __restrict__ c) {

    using V = SIMDVector<T>;
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

// Non-sqaure matrices
template<typename T, size_t M, size_t K, size_t N,
         typename std::enable_if<(M==K && K!=N) || (M!=K && K==N) || (M!=K && K!=N)
                                 || ((M==N && M==K) && N % SIMDVector<T>::Size !=0),bool>::type = 0>
FASTOR_INLINE
void _matmul(const T * __restrict__ a, const T * __restrict__ b, T * __restrict__ out) {

    // The branches are optimised away
    constexpr size_t stride_sse = N % SIMDVector<T,128>::Size;
    constexpr size_t stride_avx = N % SIMDVector<T>::Size;

    if (stride_sse == 0 && stride_avx != 0) {
        // SSE
        using V = SIMDVector<T,128>;
        constexpr size_t stride = V::Size;

        V _vec_a;
        for (size_t i=0; i<M*N; i+=stride) {
            _vec_a.store(&out[i+stride]);
        }

        for (size_t i=0; i<M; ++i) {
            for (size_t j=0; j<K; ++j) {
                _vec_a.set(a[i*K+j]);
                for (size_t k=0; k<N; k+=stride) {
                    V _vec_out = _vec_a*V(&b[j*N+k]) +  V(&out[i*N+k]);
                    _vec_out.store(&out[i*N+k]);
                }
            }
        }
//        print("sse");
    }

    else if (stride_sse == 0 && stride_avx == 0) {
        // AVX
        using V = SIMDVector<T>;
//        using V = SIMDVector<T,128>;
        constexpr size_t stride = V::Size;
//        print(stride);

        V _vec_a;
        for (size_t i=0; i<M*N; i+=stride) {
            _vec_a.store(&out[i+stride]);
        }

        for (size_t i=0; i<M; ++i) {
            for (size_t j=0; j<K; ++j) {
                _vec_a.set(a[i*K+j]);
                for (size_t k=0; k<N; k+=stride) {
                    V _vec_out = _vec_a*V(&b[j*N+k]) +  V(&out[i*N+k]);
//                    V _vec_out;
//                    if (j*N+k==0)
//                        _vec_out = _vec_a*V(_mm256_set_pd(3,2,1,0)) +  V(&out[i*N+k]);
//                    else
//                        _vec_out = _vec_a*V(&b[j*N+k]) +  V(&out[i*N+k]);
//                    print(_vec_a, V(&b[j*N+k]), _vec_a*V(&b[j*N+k]));
//                    std::cout << i+j+k << " " << _vec_a << " " << V(&b[j*N+k]) << " " << _vec_a*V(&b[j*N+k]) << "\n";
                    _vec_out.store(&out[i*N+k]);
                }
            }
        }
//        print("avx");
    }
    else {
        // Scalar
        for (size_t i=0; i<M; ++i) {
            for (size_t k=0; k<N; ++k ) {
                out[i*N+k] = a[i*K]*b[k];
            }
            for (size_t j=1; j<K; ++j) {
                for (size_t k=0; k<N; ++k ) {
                    out[i*N+k] += a[i*K+j]*b[j*N+k];
                }
            }
        }
//        print("scalar");
    }
}











template<>
FASTOR_INLINE
void _matmul<float,2,2,2>(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ out) {
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
}

template<>
FASTOR_INLINE
void _matmul<float,3,3,3>(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ out) {
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

    // blend is not correct check
//    __m128 twos0 = _mm_blend_ps(out_00,out_01,0x2);
//    __m128 twos1 = _mm_blend_ps(out_02,out_10,0x2);
//    __m128 fours0 = _mm_shuffle_ps(twos0,twos1,_MM_SHUFFLE(1,0,1,0));
//    _mm_store_ps(out,fours0);
//    __m128 twos2 = _mm_blend_ps(out_11,out_12,0x2);
//    __m128 twos3 = _mm_blend_ps(out_20,out_21,0x2);
//    __m128 fours1 = _mm_shuffle_ps(twos2,twos3,_MM_SHUFFLE(1,0,1,0));
//    _mm_store_ps(out+4,fours1);
//    _mm_store_ss(out+8,out_22);

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
}


template<>
FASTOR_INLINE
void _matmul<double,2,2,2>(const double * __restrict__ a, const double * __restrict__ b, double * __restrict__ out) {

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
}

template<>
FASTOR_INLINE
void _matmul<double,3,3,3>(const double * __restrict__ a, const double * __restrict__ b, double * __restrict__ out) {
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
}



template<>
FASTOR_INLINE void _matmul<float,4,4,4>(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ out) {

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















//!----------------------------------------------------------------------
//! Matrix-vector multiplication

template<>
FASTOR_INLINE
void _matmul<float,2,2,1>(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ out) {
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
void _matmul<float,3,3,1>(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ out) {
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


template<>
void _matmul<double,2,2,1>(const double * __restrict__ a, const double * __restrict__ b, double * __restrict__ out) {
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
void _matmul<double,3,3,1>(const double * __restrict__ a, const double * __restrict__ b, double * __restrict__ out) {
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


} // end of namespace

#endif // MATMUL_H

