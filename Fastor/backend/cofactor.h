#ifndef COFACTOR_H
#define COFACTOR_H

#include "Fastor/config/config.h"
#include "Fastor/meta/meta.h"
#include "Fastor/simd_vector/extintrin.h"

namespace Fastor {

template<typename T, size_t N, enable_if_t_<is_greater_v_<N,4>, bool> = false>
FASTOR_INLINE void _cofactor(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst);

template<typename T, size_t N, enable_if_t_<is_equal_v_<N,1>, bool> = false>
FASTOR_INLINE void _cofactor(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst) {
    *dst = *src;
}

template<typename T, size_t N, enable_if_t_<is_equal_v_<N,2>, bool> = false>
FASTOR_INLINE void _cofactor(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst)
{
    T src0 = src[0];
    T src1 = src[1];
    T src2 = src[2];
    T src3 = src[3];

    /* Compute cofactor: */
    dst[0] = + src3;
    dst[1] = - src2;
    dst[2] = - src1;
    dst[3] = + src0;
}

template<typename T, size_t N, enable_if_t_<is_equal_v_<N,3>, bool> = false>
FASTOR_INLINE void _cofactor(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst)
{
    T src0 = src[0];
    T src1 = src[1];
    T src2 = src[2];
    T src3 = src[3];
    T src4 = src[4];
    T src5 = src[5];
    T src6 = src[6];
    T src7 = src[7];
    T src8 = src[8];

    /* Compute cofactor: */
    dst[0] = + src4 * src8 - src5 * src7;
    dst[1] = - src3 * src8 + src5 * src6;
    dst[2] = + src3 * src7 - src4 * src6;
    dst[3] = - src1 * src8 + src2 * src7;
    dst[4] = + src0 * src8 - src2 * src6;
    dst[5] = - src0 * src7 + src1 * src6;
    dst[6] = + src1 * src5 - src2 * src4;
    dst[7] = - src0 * src5 + src2 * src3;
    dst[8] = + src0 * src4 - src1 * src3;
}

template<typename T, size_t N, enable_if_t_<is_equal_v_<N,4>, bool> = false>
FASTOR_INLINE void _cofactor(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst)
{
    /* Compute cofactor: */
   T t1 = src[2*4+2]*src[3*4+3] - src[2*4+3]*src[3*4+2];
   T t2 = src[2*4+1]*src[3*4+3] - src[2*4+3]*src[3*4+1];
   T t3 = src[2*4+1]*src[3*4+2] - src[2*4+2]*src[3*4+1];

   dst[0]  = src[1*4+1]*t1 - src[1*4+2]*t2 + src[1*4+3]*t3;
   dst[4]  = src[0*4+2]*t2 - src[0*4+1]*t1 - src[0*4+3]*t3;

   T t4 = src[2*4+0]*src[3*4+3] - src[2*4+3]*src[3*4+0];
   T t5 = src[2*4+0]*src[3*4+2] - src[2*4+2]*src[3*4+0];

   dst[1]  = src[1*4+2]*t4 - src[1*4+0]*t1 - src[1*4+3]*t5;
   dst[5]  = src[0*4+0]*t1 - src[0*4+2]*t4 + src[0*4+3]*t5;

   t1 = src[2*4+0]*src[3*4+1] - src[2*4+1]*src[3*4+0];

   dst[2]  = src[1*4+0]*t2 - src[1*4+1]*t4 + src[1*4+3]*t1;
   dst[6]  = src[0*4+1]*t4 - src[0*4+0]*t2 - src[0*4+3]*t1;
   dst[3]  = src[1*4+1]*t5 - src[1*4+0]*t3 - src[1*4+2]*t1;
   dst[7]  = src[0*4+0]*t3 - src[0*4+1]*t5 + src[0*4+2]*t1;

   t1 = src[0*4+2]*src[1*4+3] - src[0*4+3]*src[1*4+2];
   t2 = src[0*4+1]*src[1*4+3] - src[0*4+3]*src[1*4+1];
   t3 = src[0*4+1]*src[1*4+2] - src[0*4+2]*src[1*4+1];

   dst[8]  = src[3*4+1]*t1 - src[3*4+2]*t2 + src[3*4+3]*t3;
   dst[12] = src[2*4+2]*t2 - src[2*4+1]*t1 - src[2*4+3]*t3;

   t4 = src[0*4+0]*src[1*4+3] - src[0*4+3]*src[1*4+0];
   t5 = src[0*4+0]*src[1*4+2] - src[0*4+2]*src[1*4+0];

   dst[9]  = src[3*4+2]*t4 - src[3*4+0]*t1 - src[3*4+3]*t5;
   dst[13] = src[2*4+0]*t1 - src[2*4+2]*t4 + src[2*4+3]*t5;

   t1 = src[0*4+0]*src[1*4+1] - src[0*4+1]*src[1*4+0];

   dst[10] = src[3*4+0]*t2 - src[3*4+1]*t4 + src[3*4+3]*t1;
   dst[14] = src[2*4+1]*t4 - src[2*4+0]*t2 - src[2*4+3]*t1;
   dst[11] = src[3*4+1]*t5 - src[3*4+0]*t3 - src[3*4+2]*t1;
   dst[15] = src[2*4+0]*t3 - src[2*4+1]*t5 + src[2*4+2]*t1;
}

}

#endif // COFACTOR_H
