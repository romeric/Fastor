#ifndef DYADIC_H
#define DYADIC_H


#include "Fastor/simd_vector/SIMDVector.h"

namespace Fastor {

// The non-voigt version of outer product

// dyadic template parameters are based on size
// of the two tensors

template<typename T, size_t size0, size_t size1>
FASTOR_INLINE
void _dyadic(const T * FASTOR_RESTRICT a, const T * FASTOR_RESTRICT b, T * FASTOR_RESTRICT out) {

    using V  = SIMDVector<T,DEFAULT_ABI>;
    constexpr int VSIZE = static_cast<int>(V::Size);

    constexpr int A_DIM = size0;
    constexpr int B_DIM = size1;
    constexpr int VROUND_INNER = ROUND_DOWN(B_DIM,VSIZE);

#if FASTOR_MATMUL_UNROLL_LENGTH==4
    // Unrolling by 4 hurts the performance
    constexpr int VROUND_OUTER = ROUND_DOWN(A_DIM,VSIZE);

    int i=0;
    V vec_out, vec_out0, vec_out1, vec_out2, vec_out3;
    for (; i<VROUND_OUTER; i+=VSIZE) {
        T a0=a[i], a1=a[i+1], a2=a[i+2], a3=a[i+3];
        V vec_a0(a0), vec_a1(a1), vec_a2(a2), vec_a3(a3);
        int j=0;
        for (; j<VROUND_INNER; j+=VSIZE) {
            V brow = V(&b[j]);
            vec_out0 = vec_a0*brow;
            vec_out1 = vec_a1*brow;
            vec_out2 = vec_a2*brow;
            vec_out3 = vec_a3*brow;

            vec_out0.store(out+j+i*B_DIM,true);
            vec_out1.store(out+j+(i+1)*B_DIM,false);
            vec_out2.store(out+j+(i+2)*B_DIM,false);
            vec_out3.store(out+j+(i+3)*B_DIM,false);
        }
        for (; j<B_DIM; ++j) {
            T brow = b[j];
            out[j+i*B_DIM] = a0*brow;
            out[j+(i+1)*B_DIM] = a1*brow;
            out[j+(i+2)*B_DIM] = a2*brow;
            out[j+(i+3)*B_DIM] = a3*brow;
        }
    }

    for (; i<A_DIM; ++i) {
        T a0 = a[i];
        V vec_a(a0);
        int j=0;
        for (; j<VROUND_INNER; j+=VSIZE) {
            vec_out = vec_a*V(&b[j]);
            vec_out.store(&out[j+i*B_DIM]);
        }
        for (; j<B_DIM; ++j) {
            out[j+i*B_DIM] = a0*b[j];
        }
    }

#else

    V vec_out;
    for (int i=0; i<A_DIM; ++i) {
        T a0 = a[i];
        V vec_a(a0);
        int j=0;
        for (; j<VROUND_INNER; j+=VSIZE) {
            vec_out = vec_a*V(&b[j]);
            vec_out.store(&out[i*B_DIM+j],false);
        }
        for (; j<B_DIM; ++j) {
            out[i*B_DIM+j] = a0*b[j];
        }
    }

#endif

}




#ifdef FASTOR_AVX_IMPL

// Outer product (2x2) x (2x2)
template<>
FASTOR_INLINE
void _dyadic<float,4,4>(const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {

    __m128 vec_a = _mm_load_ps(a);
    __m128 vec_b = _mm_load_ps(b);

    __m128 a00 = _mm_shuffle_ps(vec_a,vec_a,_MM_SHUFFLE(0,0,0,0));
    __m128 a01 = _mm_shuffle_ps(vec_a,vec_a,_MM_SHUFFLE(1,1,1,1));
    __m128 a10 = _mm_shuffle_ps(vec_a,vec_a,_MM_SHUFFLE(2,2,2,2));
    __m128 a11 = _mm_shuffle_ps(vec_a,vec_a,_MM_SHUFFLE(3,3,3,3));

    __m256 a0001 = _mm256_castps128_ps256(a00);
    a0001 = _mm256_insertf128_ps(a0001,a01,0x1);
    __m256 a1011 = _mm256_castps128_ps256(a10);
    a1011 = _mm256_insertf128_ps(a1011,a11,0x1);

    __m256 vec_b2 = _mm256_castps128_ps256(vec_b);
    vec_b2 = _mm256_insertf128_ps(vec_b2,vec_b,0x1);

    _mm256_store_ps(out,_mm256_mul_ps(a0001,vec_b2));
    _mm256_store_ps(out+8,_mm256_mul_ps(a1011,vec_b2));
}


// Outer product (2x2) x (2x2)
template<>
FASTOR_INLINE
void _dyadic<double,4,4>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {

    __m256d vec_b = _mm256_load_pd(b);

    __m256d a00 = _mm256_set1_pd(a[0]);
    __m256d a01 = _mm256_set1_pd(a[1]);
    __m256d a10 = _mm256_set1_pd(a[2]);
    __m256d a11 = _mm256_set1_pd(a[3]);

    _mm256_store_pd(out,_mm256_mul_pd(a00,vec_b));
    _mm256_store_pd(out+4,_mm256_mul_pd(a01,vec_b));
    _mm256_store_pd(out+8,_mm256_mul_pd(a10,vec_b));
    _mm256_store_pd(out+12,_mm256_mul_pd(a11,vec_b));
}



// Outer product (1x2) x (1x2) [for vectors]
template<>
FASTOR_INLINE
void _dyadic<float,2,2>(const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {
    // 7 OPS
    __m128 vec_a = _mm_load_ps(a);
    __m128 vec_b = _mm_load_ps(b);

    vec_a = _mm_shuffle_ps(vec_a,vec_a,_MM_SHUFFLE(1,1,0,0));
    vec_b = _mm_shuffle_ps(vec_b,vec_b,_MM_SHUFFLE(1,0,1,0));

    _mm_store_ps(out,_mm_mul_ps(vec_a,vec_b));
}


// Outer product (1x2) x (1x2) [for vectors]
template<>
FASTOR_INLINE
void _dyadic<double,2,2>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {
    // IVY 9 OPS / HW 13 OPS
    __m128d vec_a = _mm_load_pd(a);
    __m128d vec_b = _mm_load_pd(b);

    __m128d a0 = _mm_shuffle_pd(vec_a,vec_a,0x0);
    __m128d a1 = _mm_shuffle_pd(vec_a,vec_a,0x3);
    __m256d as = _mm256_castpd128_pd256(a0);
    as = _mm256_insertf128_pd(as,a1,0x1);

    __m256d bs = _mm256_castpd128_pd256(vec_b);
    bs = _mm256_insertf128_pd(bs,vec_b,0x1);

    _mm256_store_pd(out,_mm256_mul_pd(as,bs));
}


// Outer product (1x3) x (1x3) [for vectors]
template<>
FASTOR_INLINE
void _dyadic<float,3,3>(const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {
    // 18 OPS
    __m128 vec_a = _mm_load_ps(a);
    __m128 vec_b = _mm_load_ps(b);

    __m128 a0 = _mm_shuffle_ps(vec_a,vec_a,_MM_SHUFFLE(0,0,0,0));
    __m128 a1 = _mm_shuffle_ps(vec_a,vec_a,_MM_SHUFFLE(1,1,1,1));
    __m128 a2 = _mm_shuffle_ps(vec_a,vec_a,_MM_SHUFFLE(2,2,2,2));

    _mm_store_ps(out,_mm_mul_ps(a0,vec_b));
    _mm_storeu_ps(out+3,_mm_mul_ps(a1,vec_b));
    _mm_storeu_ps(out+6,_mm_mul_ps(a2,vec_b));
}


// Outer product (1x3) x (1x3) [for vectors]
template<>
FASTOR_INLINE
void _dyadic<double,3,3>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {
    // 15 OPS + set OPS
    __m256d vec_b = _mm256_load_pd(b);
    __m256d a0 = _mm256_set1_pd(a[0]);
    __m256d a1 = _mm256_set1_pd(a[1]);
    __m256d a2 = _mm256_set1_pd(a[2]);

    _mm256_store_pd(out,_mm256_mul_pd(a0,vec_b));
    _mm256_storeu_pd(out+3,_mm256_mul_pd(a1,vec_b));
    _mm256_storeu_pd(out+6,_mm256_mul_pd(a2,vec_b));
}

#endif



// Outer product of scalars
template<>
FASTOR_INLINE
void _dyadic<double,1,1>(const double * FASTOR_RESTRICT a, const double * FASTOR_RESTRICT b, double * FASTOR_RESTRICT out) {
    out[0] = a[0]*b[0];
}
template<>
FASTOR_INLINE
void _dyadic<float,1,1>(const float * FASTOR_RESTRICT a, const float * FASTOR_RESTRICT b, float * FASTOR_RESTRICT out) {
    out[0] = a[0]*b[0];
}

}

#endif // DYADIC_H
