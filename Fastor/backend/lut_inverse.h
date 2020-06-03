#ifndef LOWUNITRI_INVERSE_H
#define LOWUNITRI_INVERSE_H

#include "Fastor/config/config.h"
#include "Fastor/meta/meta.h"

namespace Fastor {

template<typename T, size_t N, enable_if_t_<is_greater_v_<N,4>, bool> = false>
FASTOR_INLINE void _lowunitri_inverse(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst);

template<typename T, size_t N, enable_if_t_<is_equal_v_<N,1>, bool> = false>
FASTOR_INLINE void _lowunitri_inverse(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst) {
    *dst = T(1);
}

template<typename T, size_t N, enable_if_t_<is_equal_v_<N,2>, bool> = false>
FASTOR_INLINE void _lowunitri_inverse(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst)
{
    /* Compute adjoint: */
    dst[0] = 1;
    dst[1] = 0;
    dst[2] = - src[2];
    dst[3] = 1;
}

template<typename T, size_t N, enable_if_t_<is_equal_v_<N,3>, bool> = false>
FASTOR_INLINE void _lowunitri_inverse(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst)
{
    T src3 = src[3];
    T src6 = src[6];
    T src7 = src[7];
    T src8 = src[8];

    /* Compute adjoint: */
    dst[0] = 1;
    dst[1] = 0;
    dst[2] = 0;
    dst[3] = - src3 * src8;
    dst[4] = 1;
    dst[5] = 0;
    dst[6] = + src3 * src7 - src6;
    dst[7] = - src7;
    dst[8] = 1;
}


template<typename T, size_t N, enable_if_t_<is_equal_v_<N,4>, bool> = false>
FASTOR_INLINE void _lowunitri_inverse(const T *FASTOR_RESTRICT src, T *FASTOR_RESTRICT dst)
{
   dst[0]  = 1;
   dst[1]  = 0;
   dst[2]  = 0;
   dst[3]  = 0;

   dst[4]  = - src[1*4+0];
   dst[5]  = 1;
   dst[6]  = 0;
   dst[7]  = 0;

   T t2 = src[2*4+1];
   T t3 = src[2*4+1]*src[3*4+2] - src[3*4+1];

   T t4 = src[2*4+0];
   T t5 = src[2*4+0]*src[3*4+2] - src[3*4+0];

   dst[8]  = src[1*4+0]*t2 - t4;
   dst[9]  = - t2;
   dst[10] = 1;
   dst[11] = 0;

   dst[12] = t5 - src[1*4+0]*t3;
   dst[13] = t3;
   dst[14] = - src[3*4+2];
   dst[15] = 1;
}

} // end of namespace Fastor

#endif // LOWUNITRI_INVERSE_H
