#ifndef INVERSE_H
#define INVERSE_H

#include "Fastor/config/config.h"
#include "Fastor/meta/meta.h"
#include "Fastor/simd_vector/extintrin.h"

namespace Fastor {

template<typename T, size_t N, enable_if_t_<is_greater_v_<N,4>, bool> = false>
FASTOR_INLINE void _inverse(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst);

template<typename T, size_t N, enable_if_t_<is_equal_v_<N,1>, bool> = false>
FASTOR_INLINE void _inverse(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst) {
    *dst = T(1) / (*src);
}

#ifdef FASTOR_SSE2_IMPL
template<typename T, size_t N, enable_if_t_<is_equal_v_<N,2> && !is_same_v_<T,float> && !is_same_v_<T,double>, bool> = false>
#else
template<typename T, size_t N, enable_if_t_<is_equal_v_<N,2>, bool> = false>
#endif
FASTOR_INLINE void _inverse(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst)
{
    T det;

    T src0 = src[0];
    T src1 = src[1];
    T src2 = src[2];
    T src3 = src[3];

    /* Compute adjoint: */
    dst[0] = + src3;
    dst[1] = - src1;
    dst[2] = - src2;
    dst[3] = + src0;

    /* Compute determinant: */
    det = src0 * dst[0] + src1 * dst[2];

    /* Multiply adjoint with reciprocal of determinant: */
    det = T(1.0) / det;
    dst[0] *= det;
    dst[1] *= det;
    dst[2] *= det;
    dst[3] *= det;
}

#ifdef FASTOR_SSE2_IMPL
template<typename T, size_t N, enable_if_t_<is_equal_v_<N,2> && is_same_v_<T,float>, bool> = false>
FASTOR_INLINE void _inverse(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst)
{
    // This is much superior to the scalar code as
    // gcc/clang can't auto-vectorise the scalar code

    // 6 shuffles + 1 add + 1 mul + 1 div
    // Sky 6 + 4 + 4 + 11     = 25

    __m128 mat  = _mm_loadu_ps(src);
    // xor to swap off-diagonals sings
    __m128 nmat = _mm_neg_ps(mat);
    // two shuffles to get adjoint
    __m128 adj  = _mm_shuffle_ps(mat, nmat, 0x009C );
    adj         = _mm_shuffle_ps(adj, adj , 0x39   );

    // compute determinat
    __m128 tmp0 = _mm_shuffle_ps(mat , mat , 0x00D8);
    tmp0        = _mm_mul_ps    (adj , tmp0        );
    __m128 tmp1 = _mm_shuffle_ps(tmp0, tmp0, 0x1   );
    __m128 det  = _mm_div_ss    (ONEPS, _mm_add_ss(tmp0,tmp1));
    // broadcast det to all elements of __m128
    det         = _mm_shuffle_ps(det, det,  0x0    );
    // divide adjoint by determinant
    __m128 inv  = _mm_mul_ps    (adj, det);

    _mm_storeu_ps(dst, inv);
}

template<typename T, size_t N, enable_if_t_<is_equal_v_<N,2> && is_same_v_<T,double>, bool> = false>
FASTOR_INLINE void _inverse(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst)
{
    // This is much superior to the scalar code as
    // gcc/clang can't auto-vectorise the scalar code

    // 8 shuffles + 1 add + 3 mul + 1 div
    // Sky 8 + 4 + 12 + 14     = 38

    __m128d row0   = _mm_loadu_pd(src);
    __m128d row1   = _mm_loadu_pd(src+2);

    __m128d tmp    = row0;
    row0           = _mm_shuffle_pd(row0,_mm_neg_pd(row0),0x2);
    row1           = _mm_shuffle_pd(_mm_neg_pd(row1),row1,0x2);
    // these two registers hold the adjoint
    __m128d irow0  = _mm_shuffle_pd(row1,row0,0x3);
    __m128d irow1  = _mm_shuffle_pd(row1,row0,0x0);
    // dot product to compute determinant
    __m128d det    = _mm_mul_pd(tmp,_mm_reverse_pd(row1));
    det            = _mm_add_pd(det,_mm_reverse_pd(det));
    // one by determinant
    __m128d invdet = _mm_div_pd(_mm_set1_pd(1.0),det);
    // scale
    irow0          = _mm_mul_pd(irow0,invdet);
    irow1          = _mm_mul_pd(irow1,invdet);

    _mm_storeu_pd(dst  ,irow0);
    _mm_storeu_pd(dst+2,irow1);
}
#endif

template<typename T, size_t N, enable_if_t_<is_equal_v_<N,3>, bool> = false>
FASTOR_INLINE void _inverse(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst)
{
    T det;

    T src0 = src[0];
    T src1 = src[1];
    T src2 = src[2];
    T src3 = src[3];
    T src4 = src[4];
    T src5 = src[5];
    T src6 = src[6];
    T src7 = src[7];
    T src8 = src[8];

    /* Compute adjoint: */
    dst[0] = + src4 * src8 - src5 * src7;
    dst[1] = - src1 * src8 + src2 * src7;
    dst[2] = + src1 * src5 - src2 * src4;
    dst[3] = - src3 * src8 + src5 * src6;
    dst[4] = + src0 * src8 - src2 * src6;
    dst[5] = - src0 * src5 + src2 * src3;
    dst[6] = + src3 * src7 - src4 * src6;
    dst[7] = - src0 * src7 + src1 * src6;
    dst[8] = + src0 * src4 - src1 * src3;

    /* Compute determinant: */
    det = src0 * dst[0] + src1 * dst[3] + src2 * dst[6];

    /* Multiply adjoint with reciprocal of determinant: */
    det = T(1.0) / det;

    dst[0] *= det;
    dst[1] *= det;
    dst[2] *= det;
    dst[3] *= det;
    dst[4] *= det;
    dst[5] *= det;
    dst[6] *= det;
    dst[7] *= det;
    dst[8] *= det;
}


#ifdef FASTOR_SSE2_IMPL
template<typename T, size_t N, enable_if_t_<is_equal_v_<N,4> && !is_same_v_<T,float> && !is_same_v_<T,double>, bool> = false>
#else
template<typename T, size_t N, enable_if_t_<is_equal_v_<N,4>, bool> = false>
#endif
FASTOR_INLINE void _inverse(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst)
{
   T t1 = src[2*4+2]*src[3*4+3] - src[2*4+3]*src[3*4+2];
   T t2 = src[2*4+1]*src[3*4+3] - src[2*4+3]*src[3*4+1];
   T t3 = src[2*4+1]*src[3*4+2] - src[2*4+2]*src[3*4+1];

   dst[0]  = src[1*4+1]*t1 - src[1*4+2]*t2 + src[1*4+3]*t3;
   dst[1]  = src[0*4+2]*t2 - src[0*4+1]*t1 - src[0*4+3]*t3;

   T t4 = src[2*4+0]*src[3*4+3] - src[2*4+3]*src[3*4+0];
   T t5 = src[2*4+0]*src[3*4+2] - src[2*4+2]*src[3*4+0];

   dst[4]  = src[1*4+2]*t4 - src[1*4+0]*t1 - src[1*4+3]*t5;
   dst[5]  = src[0*4+0]*t1 - src[0*4+2]*t4 + src[0*4+3]*t5;

   t1 = src[2*4+0]*src[3*4+1] - src[2*4+1]*src[3*4+0];

   dst[8]  = src[1*4+0]*t2 - src[1*4+1]*t4 + src[1*4+3]*t1;
   dst[9]  = src[0*4+1]*t4 - src[0*4+0]*t2 - src[0*4+3]*t1;
   dst[12] = src[1*4+1]*t5 - src[1*4+0]*t3 - src[1*4+2]*t1;
   dst[13] = src[0*4+0]*t3 - src[0*4+1]*t5 + src[0*4+2]*t1;

   t1 = src[0*4+2]*src[1*4+3] - src[0*4+3]*src[1*4+2];
   t2 = src[0*4+1]*src[1*4+3] - src[0*4+3]*src[1*4+1];
   t3 = src[0*4+1]*src[1*4+2] - src[0*4+2]*src[1*4+1];

   dst[2]  = src[3*4+1]*t1 - src[3*4+2]*t2 + src[3*4+3]*t3;
   dst[3]  = src[2*4+2]*t2 - src[2*4+1]*t1 - src[2*4+3]*t3;

   t4 = src[0*4+0]*src[1*4+3] - src[0*4+3]*src[1*4+0];
   t5 = src[0*4+0]*src[1*4+2] - src[0*4+2]*src[1*4+0];

   dst[6]  = src[3*4+2]*t4 - src[3*4+0]*t1 - src[3*4+3]*t5;
   dst[7]  = src[2*4+0]*t1 - src[2*4+2]*t4 + src[2*4+3]*t5;

   t1 = src[0*4+0]*src[1*4+1] - src[0*4+1]*src[1*4+0];

   dst[10] = src[3*4+0]*t2 - src[3*4+1]*t4 + src[3*4+3]*t1;
   dst[11] = src[2*4+1]*t4 - src[2*4+0]*t2 - src[2*4+3]*t1;
   dst[14] = src[3*4+1]*t5 - src[3*4+0]*t3 - src[3*4+2]*t1;
   dst[15] = src[2*4+0]*t3 - src[2*4+1]*t5 + src[2*4+2]*t1;

   const T __det = src[0]*dst[0] + src[1]*dst[4] + src[2]*dst[8] + src[3]*dst[12];
   const T __invdet =  T(1)/__det;
   for (int i=0; i<16; ++i)
        dst[i] *= __invdet;
}


#ifdef FASTOR_SSE2_IMPL
template<typename T, size_t N, enable_if_t_<is_equal_v_<N,4> && is_same_v_<T,float>, bool> = false>
FASTOR_INLINE void _inverse(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst)
{
    // From Intel's SSE matrix library
    // The inverse is calculated using "Divide and Conquer" technique. The
    // original matrix is divide into four 2x2 sub-matrices. Since each
    // register of the matrix holds two elements, the smaller matrices are
    // consisted of two registers. Hence we get a better locality of the
    // calculations.

    const __m128 p4f_sign_PNNP = _mm_castsi128_ps(_mm_set_epi32(0x00000000, 0x80000000, 0x80000000, 0x00000000));

    // Load the full matrix into registers
    __m128 _L1 = _mm_loadu_ps(src +  0);
    __m128 _L2 = _mm_loadu_ps(src +  4);
    __m128 _L3 = _mm_loadu_ps(src +  8);
    __m128 _L4 = _mm_loadu_ps(src + 12);

    __m128 A, B, C, D; // the four sub-matrices

    A = _mm_movelh_ps(_L1, _L2);
    B = _mm_movehl_ps(_L2, _L1);
    C = _mm_movelh_ps(_L3, _L4);
    D = _mm_movehl_ps(_L4, _L3);

    // partial inverse of the sub-matrices
    __m128 iA, iB, iC, iD, DC, AB;
    __m128 dA, dB, dC, dD;                 // determinant of the sub-matrices
    __m128 det, d, d1, d2;
    __m128 rd;                             // reciprocal of the determinant

    //  AB = A# * B
    AB = _mm_mul_ps(_mm_shuffle_ps(A,A,0x0F), B);
    AB = _mm_sub_ps(AB,_mm_mul_ps(_mm_shuffle_ps(A,A,0xA5), _mm_shuffle_ps(B,B,0x4E)));
    //  DC = D# * C
    DC = _mm_mul_ps(_mm_shuffle_ps(D,D,0x0F), C);
    DC = _mm_sub_ps(DC,_mm_mul_ps(_mm_shuffle_ps(D,D,0xA5), _mm_shuffle_ps(C,C,0x4E)));

    //  dA = |A|
    dA = _mm_mul_ps(_mm_shuffle_ps(A, A, 0x5F),A);
    dA = _mm_sub_ss(dA, _mm_movehl_ps(dA,dA));
    //  dB = |B|
    dB = _mm_mul_ps(_mm_shuffle_ps(B, B, 0x5F),B);
    dB = _mm_sub_ss(dB, _mm_movehl_ps(dB,dB));

    //  dC = |C|
    dC = _mm_mul_ps(_mm_shuffle_ps(C, C, 0x5F),C);
    dC = _mm_sub_ss(dC, _mm_movehl_ps(dC,dC));
    //  dD = |D|
    dD = _mm_mul_ps(_mm_shuffle_ps(D, D, 0x5F),D);
    dD = _mm_sub_ss(dD, _mm_movehl_ps(dD,dD));

    //  d = trace(AB*DC) = trace(A#*B*D#*C)
    d = _mm_mul_ps(_mm_shuffle_ps(DC,DC,0xD8),AB);

    //  iD = C*A#*B
    iD = _mm_mul_ps(_mm_shuffle_ps(C,C,0xA0), _mm_movelh_ps(AB,AB));
    iD = _mm_add_ps(iD,_mm_mul_ps(_mm_shuffle_ps(C,C,0xF5), _mm_movehl_ps(AB,AB)));
    //  iA = B*D#*C
    iA = _mm_mul_ps(_mm_shuffle_ps(B,B,0xA0), _mm_movelh_ps(DC,DC));
    iA = _mm_add_ps(iA,_mm_mul_ps(_mm_shuffle_ps(B,B,0xF5), _mm_movehl_ps(DC,DC)));

    //  d = trace(AB*DC) = trace(A#*B*D#*C) [continue]
    d  = _mm_add_ps(d, _mm_movehl_ps(d, d));
    d  = _mm_add_ss(d, _mm_shuffle_ps(d, d, 1));
    d1 = _mm_mul_ss(dA,dD);
    d2 = _mm_mul_ss(dB,dC);

    //  iD = D*|A| - C*A#*B
    iD = _mm_sub_ps(_mm_mul_ps(D,_mm_shuffle_ps(dA,dA,0)), iD);

    //  iA = A*|D| - B*D#*C;
    iA = _mm_sub_ps(_mm_mul_ps(A,_mm_shuffle_ps(dD,dD,0)), iA);

    //  det = |A|*|D| + |B|*|C| - trace(A#*B*D#*C)
    det = _mm_sub_ss(_mm_add_ss(d1,d2),d);
    rd  = _mm_div_ss(_mm_set_ss(1.0f), det);

    //  iB = D * (A#B)# = D*B#*A
    iB = _mm_mul_ps(D, _mm_shuffle_ps(AB,AB,0x33));
    iB = _mm_sub_ps(iB, _mm_mul_ps(_mm_shuffle_ps(D,D,0xB1), _mm_shuffle_ps(AB,AB,0x66)));
    //  iC = A * (D#C)# = A*C#*D
    iC = _mm_mul_ps(A, _mm_shuffle_ps(DC,DC,0x33));
    iC = _mm_sub_ps(iC, _mm_mul_ps(_mm_shuffle_ps(A,A,0xB1), _mm_shuffle_ps(DC,DC,0x66)));

    rd = _mm_shuffle_ps(rd,rd,0);
    rd = _mm_xor_ps(rd, p4f_sign_PNNP);

    //  iB = C*|B| - D*B#*A
    iB = _mm_sub_ps(_mm_mul_ps(C,_mm_shuffle_ps(dB,dB,0)), iB);

    //  iC = B*|C| - A*C#*D;
    iC = _mm_sub_ps(_mm_mul_ps(B,_mm_shuffle_ps(dC,dC,0)), iC);

    //  iX = iX / det
    iA = _mm_mul_ps(rd,iA);
    iB = _mm_mul_ps(rd,iB);
    iC = _mm_mul_ps(rd,iC);
    iD = _mm_mul_ps(rd,iD);

    _mm_storeu_ps(dst+0,  _mm_shuffle_ps(iA,iB,0x77));
    _mm_storeu_ps(dst+4,  _mm_shuffle_ps(iA,iB,0x22));
    _mm_storeu_ps(dst+8,  _mm_shuffle_ps(iC,iD,0x77));
    _mm_storeu_ps(dst+12, _mm_shuffle_ps(iC,iD,0x22));
}


template<typename T, size_t N, enable_if_t_<is_equal_v_<N,4> && is_same_v_<T,double>, bool> = false>
FASTOR_INLINE void _inverse(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst)
{
    // From Intel's SSE matrix library
    // The inverse is calculated using "Divide and Conquer" technique. The
    // original matrix is divide into four 2x2 sub-matrices. Since each
    // register of the matrix holds two elements, the smaller matrices are
    // consisted of two registers. Hence we get a better locality of the
    // calculations.

    const __m128d _Sign_NP = _mm_castsi128_pd(_mm_set_epi32(0x0,0x0,0x80000000,0x0));
    const __m128d _Sign_PN = _mm_castsi128_pd(_mm_set_epi32(0x80000000,0x0,0x0,0x0));

    // the four sub-matrices
    __m128d A1, A2, B1, B2, C1, C2, D1, D2;

    A1 = _mm_loadu_pd(src + 0); B1 = _mm_loadu_pd(src + 2);
    A2 = _mm_loadu_pd(src + 4); B2 = _mm_loadu_pd(src + 6);
    C1 = _mm_loadu_pd(src + 8); D1 = _mm_loadu_pd(src +10);
    C2 = _mm_loadu_pd(src +12); D2 = _mm_loadu_pd(src +14);

    // partial inverse of the sub-matrices
    __m128d iA1, iA2, iB1, iB2, iC1, iC2, iD1, iD2, DC1, DC2, AB1, AB2;
    __m128d dA, dB, dC, dD;     // determinant of the sub-matrices
    __m128d det, d1, d2, rd;

    //  dA = |A|
    dA = _mm_shuffle_pd(A2, A2, 1);
    dA = _mm_mul_pd(A1, dA);
    dA = _mm_sub_sd(dA, _mm_shuffle_pd(dA,dA,3));
    //  dB = |B|
    dB = _mm_shuffle_pd(B2, B2, 1);
    dB = _mm_mul_pd(B1, dB);
    dB = _mm_sub_sd(dB, _mm_shuffle_pd(dB,dB,3));

    //  AB = A# * B
    AB1 = _mm_mul_pd(B1, _mm_shuffle_pd(A2,A2,3));
    AB2 = _mm_mul_pd(B2, _mm_shuffle_pd(A1,A1,0));
    AB1 = _mm_sub_pd(AB1, _mm_mul_pd(B2, _mm_shuffle_pd(A1,A1,3)));
    AB2 = _mm_sub_pd(AB2, _mm_mul_pd(B1, _mm_shuffle_pd(A2,A2,0)));

    //  dC = |C|
    dC = _mm_shuffle_pd(C2, C2, 1);
    dC = _mm_mul_pd(C1, dC);
    dC = _mm_sub_sd(dC, _mm_shuffle_pd(dC,dC,3));
    //  dD = |D|
    dD = _mm_shuffle_pd(D2, D2, 1);
    dD = _mm_mul_pd(D1, dD);
    dD = _mm_sub_sd(dD, _mm_shuffle_pd(dD,dD,3));

    //  DC = D# * C
    DC1 = _mm_mul_pd(C1, _mm_shuffle_pd(D2,D2,3));
    DC2 = _mm_mul_pd(C2, _mm_shuffle_pd(D1,D1,0));
    DC1 = _mm_sub_pd(DC1, _mm_mul_pd(C2, _mm_shuffle_pd(D1,D1,3)));
    DC2 = _mm_sub_pd(DC2, _mm_mul_pd(C1, _mm_shuffle_pd(D2,D2,0)));

    //  rd = trace(AB*DC) = trace(A#*B*D#*C)
    d1 = _mm_mul_pd(AB1, _mm_shuffle_pd(DC1, DC2, 0));
    d2 = _mm_mul_pd(AB2, _mm_shuffle_pd(DC1, DC2, 3));
    rd = _mm_add_pd(d1, d2);
    rd = _mm_add_sd(rd, _mm_shuffle_pd(rd, rd,3));

    //  iD = C*A#*B
    iD1 = _mm_mul_pd(AB1, _mm_shuffle_pd(C1,C1,0));
    iD2 = _mm_mul_pd(AB1, _mm_shuffle_pd(C2,C2,0));
    iD1 = _mm_add_pd(iD1, _mm_mul_pd(AB2, _mm_shuffle_pd(C1,C1,3)));
    iD2 = _mm_add_pd(iD2, _mm_mul_pd(AB2, _mm_shuffle_pd(C2,C2,3)));

    //  iA = B*D#*C
    iA1 = _mm_mul_pd(DC1, _mm_shuffle_pd(B1,B1,0));
    iA2 = _mm_mul_pd(DC1, _mm_shuffle_pd(B2,B2,0));
    iA1 = _mm_add_pd(iA1, _mm_mul_pd(DC2, _mm_shuffle_pd(B1,B1,3)));
    iA2 = _mm_add_pd(iA2, _mm_mul_pd(DC2, _mm_shuffle_pd(B2,B2,3)));

    //  iD = D*|A| - C*A#*B
    dA = _mm_shuffle_pd(dA,dA,0);
    iD1 = _mm_sub_pd(_mm_mul_pd(D1, dA), iD1);
    iD2 = _mm_sub_pd(_mm_mul_pd(D2, dA), iD2);

    //  iA = A*|D| - B*D#*C;
    dD = _mm_shuffle_pd(dD,dD,0);
    iA1 = _mm_sub_pd(_mm_mul_pd(A1, dD), iA1);
    iA2 = _mm_sub_pd(_mm_mul_pd(A2, dD), iA2);

    d1 = _mm_mul_sd(dA, dD);
    d2 = _mm_mul_sd(dB, dC);

    //  iB = D * (A#B)# = D*B#*A
    iB1 = _mm_mul_pd(D1, _mm_shuffle_pd(AB2,AB1,1));
    iB2 = _mm_mul_pd(D2, _mm_shuffle_pd(AB2,AB1,1));
    iB1 = _mm_sub_pd(iB1, _mm_mul_pd(_mm_shuffle_pd(D1,D1,1), _mm_shuffle_pd(AB2,AB1,2)));
    iB2 = _mm_sub_pd(iB2, _mm_mul_pd(_mm_shuffle_pd(D2,D2,1), _mm_shuffle_pd(AB2,AB1,2)));

    //  det = |A|*|D| + |B|*|C| - trace(A#*B*D#*C)
    det = _mm_add_sd(d1, d2);
    det = _mm_sub_sd(det, rd);

    //  iC = A * (D#C)# = A*C#*D
    iC1 = _mm_mul_pd(A1, _mm_shuffle_pd(DC2,DC1,1));
    iC2 = _mm_mul_pd(A2, _mm_shuffle_pd(DC2,DC1,1));
    iC1 = _mm_sub_pd(iC1, _mm_mul_pd(_mm_shuffle_pd(A1,A1,1), _mm_shuffle_pd(DC2,DC1,2)));
    iC2 = _mm_sub_pd(iC2, _mm_mul_pd(_mm_shuffle_pd(A2,A2,1), _mm_shuffle_pd(DC2,DC1,2)));

    rd = _mm_div_sd(_mm_set_sd(1.0), det);
    rd = _mm_shuffle_pd(rd,rd,0);

    //  iB = C*|B| - D*B#*A
    dB = _mm_shuffle_pd(dB,dB,0);
    iB1 = _mm_sub_pd(_mm_mul_pd(C1, dB), iB1);
    iB2 = _mm_sub_pd(_mm_mul_pd(C2, dB), iB2);

    d1 = _mm_xor_pd(rd, _Sign_PN);
    d2 = _mm_xor_pd(rd, _Sign_NP);

    //  iC = B*|C| - A*C#*D;
    dC = _mm_shuffle_pd(dC,dC,0);
    iC1 = _mm_sub_pd(_mm_mul_pd(B1, dC), iC1);
    iC2 = _mm_sub_pd(_mm_mul_pd(B2, dC), iC2);

    _mm_storeu_pd(dst+0,    _mm_mul_pd(_mm_shuffle_pd(iA2, iA1, 3), d1));
    _mm_storeu_pd(dst+4,    _mm_mul_pd(_mm_shuffle_pd(iA2, iA1, 0), d2));
    _mm_storeu_pd(dst+2,    _mm_mul_pd(_mm_shuffle_pd(iB2, iB1, 3), d1));
    _mm_storeu_pd(dst+4+2,  _mm_mul_pd(_mm_shuffle_pd(iB2, iB1, 0), d2));
    _mm_storeu_pd(dst+2*4,  _mm_mul_pd(_mm_shuffle_pd(iC2, iC1, 3), d1));
    _mm_storeu_pd(dst+3*4,  _mm_mul_pd(_mm_shuffle_pd(iC2, iC1, 0), d2));
    _mm_storeu_pd(dst+2*4+2,_mm_mul_pd(_mm_shuffle_pd(iD2, iD1, 3), d1));
    _mm_storeu_pd(dst+3*4+2,_mm_mul_pd(_mm_shuffle_pd(iD2, iD1, 0), d2));
}
#endif


} // end of namespace Fastor

#endif // INVERSE_H
