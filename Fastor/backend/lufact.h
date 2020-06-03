#ifndef LUFACT_H
#define LUFACT_H

#include "Fastor/meta/meta.h"
#include "Fastor/config/config.h"
#include "Fastor/simd_vector/extintrin.h"

namespace Fastor {

template<typename T, size_t N, enable_if_t_<is_greater_v_<N,8>, bool> = false>
FASTOR_INLINE void _lufact(const T *FASTOR_RESTRICT a, T *FASTOR_RESTRICT l, T *FASTOR_RESTRICT u);

template<typename T, size_t N, enable_if_t_<is_equal_v_<N,1>, bool> = false>
FASTOR_INLINE void _lufact(const T *FASTOR_RESTRICT a, T *FASTOR_RESTRICT l, T *FASTOR_RESTRICT u) {
    *l = 1;
    *u = *a;
}

template<typename T, size_t N, enable_if_t_<is_equal_v_<N,2>, bool> = false>
FASTOR_INLINE void _lufact(const T *FASTOR_RESTRICT A, T *FASTOR_RESTRICT L, T *FASTOR_RESTRICT U) {

    // [a11 a12]   [1   0] [u11 u12]
    // [a21 a22]   [l21 1] [0   u22]

    const T L21 = A[2]/A[0];

    L[0] = 1;
    L[1] = 0;
    L[2] = L21;
    L[3] = 1;


    U[0] = A[0];
    U[1] = A[1];
    U[2] = 0;
    U[3] = A[3] - L21 * A[1];
}

template<typename T, size_t N, enable_if_t_<is_equal_v_<N,3>, bool> = false>
FASTOR_INLINE void _lufact(const T *FASTOR_RESTRICT A, T *FASTOR_RESTRICT L, T *FASTOR_RESTRICT U) {

    const T A00 = A[0];
    const T L21 = A[3]/A00;
    const T L31 = A[6]/A00;

    const T U22 = A[4] - L21 * A[1];
    const T U23 = A[5] - L21 * A[2];

    const T L32 = (A[7] - L31 * A[1]) / U22;

    const T U33 = A[8] - L31 * A[2] - L32 * U23;

    L[0] = 1;
    L[1] = 0;
    L[2] = 0;
    L[3] = L21;
    L[4] = 1;
    L[5] = 0;
    L[6] = L31;
    L[7] = L32;
    L[8] = 1;

    U[0] = A00;
    U[1] = A[1];
    U[2] = A[2];
    U[3] = 0;
    U[4] = U22;
    U[5] = U23;
    U[6] = 0;
    U[7] = 0;
    U[8] = U33;
}

template<typename T, size_t N, enable_if_t_<is_equal_v_<N,4>, bool> = false>
FASTOR_INLINE void _lufact(const T *FASTOR_RESTRICT A, T *FASTOR_RESTRICT L, T *FASTOR_RESTRICT U) {

    const T A00 = A[0];
    const T L21 = A[4] /A00;
    const T L31 = A[8] /A00;
    const T L41 = A[12]/A00;

    const T U22 = A[5] - L21 * A[1];
    const T U23 = A[6] - L21 * A[2];
    const T U24 = A[7] - L21 * A[3];

    const T L32 = (A[9 ] - L31 * A[1]) / U22;
    const T L42 = (A[13] - L41 * A[1]) / U22;

    const T U33 = A[10] - L31 * A[2] - L32 * U23;
    const T U34 = A[11] - L31 * A[3] - L32 * U24;

    const T L43 = (A[14] - L41 * A[2] - L42 * U23) / U33;

    const T U44 = A[15] - L41 * A[3] - L42 * U24 - L43 * U34;

    L[0]  = 1;
    L[1]  = 0;
    L[2]  = 0;
    L[3]  = 0;
    L[4]  = L21;
    L[5]  = 1;
    L[6]  = 0;
    L[7]  = 0;
    L[8]  = L31;
    L[9]  = L32;
    L[10] = 1;
    L[11] = 0;
    L[12] = L41;
    L[13] = L42;
    L[14] = L43;
    L[15] = 1;

    U[0]  = A00;
    U[1]  = A[1];
    U[2]  = A[2];
    U[3]  = A[3];
    U[4]  = 0;
    U[5]  = U22;
    U[6]  = U23;
    U[7]  = U24;
    U[8]  = 0;
    U[9]  = 0;
    U[10] = U33;
    U[11] = U34;
    U[12] = 0;
    U[13] = 0;
    U[14] = 0;
    U[15] = U44;
}

template<typename T, size_t N, enable_if_t_<is_equal_v_<N,5>, bool> = false>
FASTOR_INLINE void _lufact(const T *FASTOR_RESTRICT A, T *FASTOR_RESTRICT L, T *FASTOR_RESTRICT U) {

    const T A00 = A[0*N];
    const T L21 = A[1*N] /A00;
    const T L31 = A[2*N] /A00;
    const T L41 = A[3*N] /A00;
    const T L51 = A[4*N] /A00;

    const T U22 = A[1*N+1] - L21 * A[1];
    const T U23 = A[1*N+2] - L21 * A[2];
    const T U24 = A[1*N+3] - L21 * A[3];
    const T U25 = A[1*N+4] - L21 * A[4];

    const T L32 = (A[2*N+1] - L31 * A[1]) / U22;
    const T L42 = (A[3*N+1] - L41 * A[1]) / U22;
    const T L52 = (A[4*N+1] - L51 * A[1]) / U22;

    const T U33 = A[2*N+2] - L31 * A[2] - L32 * U23;
    const T U34 = A[2*N+3] - L31 * A[3] - L32 * U24;
    const T U35 = A[2*N+4] - L31 * A[4] - L32 * U25;

    const T L43 = (A[3*N+2] - L41 * A[2] - L42 * U23) / U33;
    const T L53 = (A[4*N+2] - L51 * A[2] - L52 * U23) / U33;

    const T U44 = A[3*N+3] - L41 * A[3] - L42 * U24 - L43 * U34;
    const T U45 = A[3*N+4] - L41 * A[4] - L42 * U25 - L43 * U35;

    const T L54 = (A[4*N+3] - L51 * A[3] - L52 * U24 - L53 * U34) / U44;

    const T U55 = A[4*N+4] - L51 * A[4] - L52 * U25 - L53 * U35 - L54 * U45;

    // L
    L[0*N+0]  = 1;
    L[0*N+1]  = 0;
    L[0*N+2]  = 0;
    L[0*N+3]  = 0;
    L[0*N+4]  = 0;

    L[1*N+0]  = L21;
    L[1*N+1]  = 1;
    L[1*N+2]  = 0;
    L[1*N+3]  = 0;
    L[1*N+4]  = 0;

    L[2*N+0]  = L31;
    L[2*N+1]  = L32;
    L[2*N+2]  = 1;
    L[2*N+3]  = 0;
    L[2*N+4]  = 0;

    L[3*N+0]  = L41;
    L[3*N+1]  = L42;
    L[3*N+2]  = L43;
    L[3*N+3]  = 1;
    L[3*N+4]  = 0;

    L[4*N+0]  = L51;
    L[4*N+1]  = L52;
    L[4*N+2]  = L53;
    L[4*N+3]  = L54;
    L[4*N+4]  = 1;

    // U
    U[0*N+0]  = A00;
    U[0*N+1]  = A[1];
    U[0*N+2]  = A[2];
    U[0*N+3]  = A[3];
    U[0*N+4]  = A[4];

    U[1*N+0]  = 0;
    U[1*N+1]  = U22;
    U[1*N+2]  = U23;
    U[1*N+3]  = U24;
    U[1*N+4]  = U25;

    U[2*N+0]  = 0;
    U[2*N+1]  = 0;
    U[2*N+2]  = U33;
    U[2*N+3]  = U34;
    U[2*N+4]  = U35;

    U[3*N+0]  = 0;
    U[3*N+1]  = 0;
    U[3*N+2]  = 0;
    U[3*N+3]  = U44;
    U[3*N+4]  = U45;

    U[4*N+0]  = 0;
    U[4*N+1]  = 0;
    U[4*N+2]  = 0;
    U[4*N+3]  = 0;
    U[4*N+4]  = U55;
}

template<typename T, size_t N, enable_if_t_<is_equal_v_<N,6>, bool> = false>
FASTOR_INLINE void _lufact(const T *FASTOR_RESTRICT A, T *FASTOR_RESTRICT L, T *FASTOR_RESTRICT U) {

    const T A00 = A[0*N];
    const T L21 = A[1*N] /A00;
    const T L31 = A[2*N] /A00;
    const T L41 = A[3*N] /A00;
    const T L51 = A[4*N] /A00;
    const T L61 = A[5*N] /A00;

    const T U22 = A[1*N+1] - L21 * A[1];
    const T U23 = A[1*N+2] - L21 * A[2];
    const T U24 = A[1*N+3] - L21 * A[3];
    const T U25 = A[1*N+4] - L21 * A[4];
    const T U26 = A[1*N+5] - L21 * A[5];

    const T L32 = (A[2*N+1] - L31 * A[1]) / U22;
    const T L42 = (A[3*N+1] - L41 * A[1]) / U22;
    const T L52 = (A[4*N+1] - L51 * A[1]) / U22;
    const T L62 = (A[5*N+1] - L61 * A[1]) / U22;

    const T U33 = A[2*N+2] - L31 * A[2] - L32 * U23;
    const T U34 = A[2*N+3] - L31 * A[3] - L32 * U24;
    const T U35 = A[2*N+4] - L31 * A[4] - L32 * U25;
    const T U36 = A[2*N+5] - L31 * A[5] - L32 * U26;

    const T L43 = (A[3*N+2] - L41 * A[2] - L42 * U23) / U33;
    const T L53 = (A[4*N+2] - L51 * A[2] - L52 * U23) / U33;
    const T L63 = (A[5*N+2] - L61 * A[2] - L62 * U23) / U33;

    const T U44 = A[3*N+3] - L41 * A[3] - L42 * U24 - L43 * U34;
    const T U45 = A[3*N+4] - L41 * A[4] - L42 * U25 - L43 * U35;
    const T U46 = A[3*N+5] - L41 * A[5] - L42 * U26 - L43 * U36;

    const T L54 = (A[4*N+3] - L51 * A[3] - L52 * U24 - L53 * U34) / U44;
    const T L64 = (A[5*N+3] - L61 * A[3] - L62 * U24 - L63 * U34) / U44;

    const T U55 = A[4*N+4] - L51 * A[4] - L52 * U25 - L53 * U35 - L54 * U45;
    const T U56 = A[4*N+5] - L51 * A[5] - L52 * U26 - L53 * U36 - L54 * U46;

    const T L65 = (A[5*N+4] - L61 * A[4] - L62 * U25 - L63 * U35 - L64 * U45) / U55;

    const T U66 = A[5*N+5] - L61 * A[5] - L62 * U26 - L63 * U36 - L64 * U46 - L65 * U56;

    // L
    L[0*N+0]  = 1;
    L[0*N+1]  = 0;
    L[0*N+2]  = 0;
    L[0*N+3]  = 0;
    L[0*N+4]  = 0;
    L[0*N+5]  = 0;

    L[1*N+0]  = L21;
    L[1*N+1]  = 1;
    L[1*N+2]  = 0;
    L[1*N+3]  = 0;
    L[1*N+4]  = 0;
    L[1*N+5]  = 0;

    L[2*N+0]  = L31;
    L[2*N+1]  = L32;
    L[2*N+2]  = 1;
    L[2*N+3]  = 0;
    L[2*N+4]  = 0;
    L[2*N+5]  = 0;

    L[3*N+0]  = L41;
    L[3*N+1]  = L42;
    L[3*N+2]  = L43;
    L[3*N+3]  = 1;
    L[3*N+4]  = 0;
    L[3*N+5]  = 0;

    L[4*N+0]  = L51;
    L[4*N+1]  = L52;
    L[4*N+2]  = L53;
    L[4*N+3]  = L54;
    L[4*N+4]  = 1;
    L[4*N+5]  = 0;

    L[5*N+0]  = L61;
    L[5*N+1]  = L62;
    L[5*N+2]  = L63;
    L[5*N+3]  = L64;
    L[5*N+4]  = L65;
    L[5*N+5]  = 1;

    // U
    U[0*N+0]  = A00;
    U[0*N+1]  = A[1];
    U[0*N+2]  = A[2];
    U[0*N+3]  = A[3];
    U[0*N+4]  = A[4];
    U[0*N+5]  = A[5];

    U[1*N+0]  = 0;
    U[1*N+1]  = U22;
    U[1*N+2]  = U23;
    U[1*N+3]  = U24;
    U[1*N+4]  = U25;
    U[1*N+5]  = U26;

    U[2*N+0]  = 0;
    U[2*N+1]  = 0;
    U[2*N+2]  = U33;
    U[2*N+3]  = U34;
    U[2*N+4]  = U35;
    U[2*N+5]  = U36;

    U[3*N+0]  = 0;
    U[3*N+1]  = 0;
    U[3*N+2]  = 0;
    U[3*N+3]  = U44;
    U[3*N+4]  = U45;
    U[3*N+5]  = U46;

    U[4*N+0]  = 0;
    U[4*N+1]  = 0;
    U[4*N+2]  = 0;
    U[4*N+3]  = 0;
    U[4*N+4]  = U55;
    U[4*N+5]  = U56;

    U[5*N+0]  = 0;
    U[5*N+1]  = 0;
    U[5*N+2]  = 0;
    U[5*N+3]  = 0;
    U[5*N+4]  = 0;
    U[5*N+5]  = U66;
}

template<typename T, size_t N, enable_if_t_<is_equal_v_<N,7>, bool> = false>
FASTOR_INLINE void _lufact(const T *FASTOR_RESTRICT A, T *FASTOR_RESTRICT L, T *FASTOR_RESTRICT U) {

    const T A00 = A[0*N];
    const T L21 = A[1*N] /A00;
    const T L31 = A[2*N] /A00;
    const T L41 = A[3*N] /A00;
    const T L51 = A[4*N] /A00;
    const T L61 = A[5*N] /A00;
    const T L71 = A[6*N] /A00;

    const T U22 = A[1*N+1] - L21 * A[1];
    const T U23 = A[1*N+2] - L21 * A[2];
    const T U24 = A[1*N+3] - L21 * A[3];
    const T U25 = A[1*N+4] - L21 * A[4];
    const T U26 = A[1*N+5] - L21 * A[5];
    const T U27 = A[1*N+6] - L21 * A[6];

    const T L32 = (A[2*N+1] - L31 * A[1]) / U22;
    const T L42 = (A[3*N+1] - L41 * A[1]) / U22;
    const T L52 = (A[4*N+1] - L51 * A[1]) / U22;
    const T L62 = (A[5*N+1] - L61 * A[1]) / U22;
    const T L72 = (A[6*N+1] - L71 * A[1]) / U22;

    const T U33 = A[2*N+2] - L31 * A[2] - L32 * U23;
    const T U34 = A[2*N+3] - L31 * A[3] - L32 * U24;
    const T U35 = A[2*N+4] - L31 * A[4] - L32 * U25;
    const T U36 = A[2*N+5] - L31 * A[5] - L32 * U26;
    const T U37 = A[2*N+6] - L31 * A[6] - L32 * U27;

    const T L43 = (A[3*N+2] - L41 * A[2] - L42 * U23) / U33;
    const T L53 = (A[4*N+2] - L51 * A[2] - L52 * U23) / U33;
    const T L63 = (A[5*N+2] - L61 * A[2] - L62 * U23) / U33;
    const T L73 = (A[6*N+2] - L71 * A[2] - L72 * U23) / U33;

    const T U44 = A[3*N+3] - L41 * A[3] - L42 * U24 - L43 * U34;
    const T U45 = A[3*N+4] - L41 * A[4] - L42 * U25 - L43 * U35;
    const T U46 = A[3*N+5] - L41 * A[5] - L42 * U26 - L43 * U36;
    const T U47 = A[3*N+6] - L41 * A[6] - L42 * U27 - L43 * U37;

    const T L54 = (A[4*N+3] - L51 * A[3] - L52 * U24 - L53 * U34) / U44;
    const T L64 = (A[5*N+3] - L61 * A[3] - L62 * U24 - L63 * U34) / U44;
    const T L74 = (A[6*N+3] - L71 * A[3] - L72 * U24 - L73 * U34) / U44;

    const T U55 = A[4*N+4] - L51 * A[4] - L52 * U25 - L53 * U35 - L54 * U45;
    const T U56 = A[4*N+5] - L51 * A[5] - L52 * U26 - L53 * U36 - L54 * U46;
    const T U57 = A[4*N+6] - L51 * A[6] - L52 * U27 - L53 * U37 - L54 * U47;

    const T L65 = (A[5*N+4] - L61 * A[4] - L62 * U25 - L63 * U35 - L64 * U45) / U55;
    const T L75 = (A[6*N+4] - L71 * A[4] - L72 * U25 - L73 * U35 - L74 * U45) / U55;

    const T U66 = A[5*N+5] - L61 * A[5] - L62 * U26 - L63 * U36 - L64 * U46 - L65 * U56;
    const T U67 = A[5*N+6] - L61 * A[6] - L62 * U27 - L63 * U37 - L64 * U47 - L65 * U57;

    const T L76 = (A[6*N+5] - L71 * A[5] - L72 * U26 - L73 * U36 - L74 * U46 - L75 * U56) / U66;

    const T U77 = A[6*N+6] - L71 * A[6] - L72 * U27 - L73 * U37 - L74 * U47 - L75 * U57 - L76 * U67;

    // L
    L[0*N+0]  = 1;
    L[0*N+1]  = 0;
    L[0*N+2]  = 0;
    L[0*N+3]  = 0;
    L[0*N+4]  = 0;
    L[0*N+5]  = 0;
    L[0*N+6]  = 0;

    L[1*N+0]  = L21;
    L[1*N+1]  = 1;
    L[1*N+2]  = 0;
    L[1*N+3]  = 0;
    L[1*N+4]  = 0;
    L[1*N+5]  = 0;
    L[1*N+6]  = 0;

    L[2*N+0]  = L31;
    L[2*N+1]  = L32;
    L[2*N+2]  = 1;
    L[2*N+3]  = 0;
    L[2*N+4]  = 0;
    L[2*N+5]  = 0;
    L[2*N+6]  = 0;

    L[3*N+0]  = L41;
    L[3*N+1]  = L42;
    L[3*N+2]  = L43;
    L[3*N+3]  = 1;
    L[3*N+4]  = 0;
    L[3*N+5]  = 0;
    L[3*N+6]  = 0;

    L[4*N+0]  = L51;
    L[4*N+1]  = L52;
    L[4*N+2]  = L53;
    L[4*N+3]  = L54;
    L[4*N+4]  = 1;
    L[4*N+5]  = 0;
    L[4*N+6]  = 0;

    L[5*N+0]  = L61;
    L[5*N+1]  = L62;
    L[5*N+2]  = L63;
    L[5*N+3]  = L64;
    L[5*N+4]  = L65;
    L[5*N+5]  = 1;
    L[5*N+6]  = 0;

    L[6*N+0]  = L71;
    L[6*N+1]  = L72;
    L[6*N+2]  = L73;
    L[6*N+3]  = L74;
    L[6*N+4]  = L75;
    L[6*N+5]  = L76;
    L[6*N+6]  = 1;

    // U
    U[0*N+0]  = A00;
    U[0*N+1]  = A[1];
    U[0*N+2]  = A[2];
    U[0*N+3]  = A[3];
    U[0*N+4]  = A[4];
    U[0*N+5]  = A[5];
    U[0*N+6]  = A[6];

    U[1*N+0]  = 0;
    U[1*N+1]  = U22;
    U[1*N+2]  = U23;
    U[1*N+3]  = U24;
    U[1*N+4]  = U25;
    U[1*N+5]  = U26;
    U[1*N+6]  = U27;

    U[2*N+0]  = 0;
    U[2*N+1]  = 0;
    U[2*N+2]  = U33;
    U[2*N+3]  = U34;
    U[2*N+4]  = U35;
    U[2*N+5]  = U36;
    U[2*N+6]  = U37;

    U[3*N+0]  = 0;
    U[3*N+1]  = 0;
    U[3*N+2]  = 0;
    U[3*N+3]  = U44;
    U[3*N+4]  = U45;
    U[3*N+5]  = U46;
    U[3*N+6]  = U47;

    U[4*N+0]  = 0;
    U[4*N+1]  = 0;
    U[4*N+2]  = 0;
    U[4*N+3]  = 0;
    U[4*N+4]  = U55;
    U[4*N+5]  = U56;
    U[4*N+6]  = U57;

    U[5*N+0]  = 0;
    U[5*N+1]  = 0;
    U[5*N+2]  = 0;
    U[5*N+3]  = 0;
    U[5*N+4]  = 0;
    U[5*N+5]  = U66;
    U[5*N+6]  = U67;

    U[6*N+0]  = 0;
    U[6*N+1]  = 0;
    U[6*N+2]  = 0;
    U[6*N+3]  = 0;
    U[6*N+4]  = 0;
    U[6*N+5]  = 0;
    U[6*N+6]  = U77;
}

template<typename T, size_t N, enable_if_t_<is_equal_v_<N,8>, bool> = false>
FASTOR_INLINE void _lufact(const T *FASTOR_RESTRICT A, T *FASTOR_RESTRICT L, T *FASTOR_RESTRICT U) {

    const T A00 = A[0*N];
    const T L21 = A[1*N] /A00;
    const T L31 = A[2*N] /A00;
    const T L41 = A[3*N] /A00;
    const T L51 = A[4*N] /A00;
    const T L61 = A[5*N] /A00;
    const T L71 = A[6*N] /A00;
    const T L81 = A[7*N] /A00;

    const T U22 = A[1*N+1] - L21 * A[1];
    const T U23 = A[1*N+2] - L21 * A[2];
    const T U24 = A[1*N+3] - L21 * A[3];
    const T U25 = A[1*N+4] - L21 * A[4];
    const T U26 = A[1*N+5] - L21 * A[5];
    const T U27 = A[1*N+6] - L21 * A[6];
    const T U28 = A[1*N+7] - L21 * A[7];

    const T L32 = (A[2*N+1] - L31 * A[1]) / U22;
    const T L42 = (A[3*N+1] - L41 * A[1]) / U22;
    const T L52 = (A[4*N+1] - L51 * A[1]) / U22;
    const T L62 = (A[5*N+1] - L61 * A[1]) / U22;
    const T L72 = (A[6*N+1] - L71 * A[1]) / U22;
    const T L82 = (A[7*N+1] - L81 * A[1]) / U22;

    const T U33 = A[2*N+2] - L31 * A[2] - L32 * U23;
    const T U34 = A[2*N+3] - L31 * A[3] - L32 * U24;
    const T U35 = A[2*N+4] - L31 * A[4] - L32 * U25;
    const T U36 = A[2*N+5] - L31 * A[5] - L32 * U26;
    const T U37 = A[2*N+6] - L31 * A[6] - L32 * U27;
    const T U38 = A[2*N+7] - L31 * A[7] - L32 * U28;

    const T L43 = (A[3*N+2] - L41 * A[2] - L42 * U23) / U33;
    const T L53 = (A[4*N+2] - L51 * A[2] - L52 * U23) / U33;
    const T L63 = (A[5*N+2] - L61 * A[2] - L62 * U23) / U33;
    const T L73 = (A[6*N+2] - L71 * A[2] - L72 * U23) / U33;
    const T L83 = (A[7*N+2] - L81 * A[2] - L82 * U23) / U33;

    const T U44 = A[3*N+3] - L41 * A[3] - L42 * U24 - L43 * U34;
    const T U45 = A[3*N+4] - L41 * A[4] - L42 * U25 - L43 * U35;
    const T U46 = A[3*N+5] - L41 * A[5] - L42 * U26 - L43 * U36;
    const T U47 = A[3*N+6] - L41 * A[6] - L42 * U27 - L43 * U37;
    const T U48 = A[3*N+7] - L41 * A[7] - L42 * U28 - L43 * U38;

    const T L54 = (A[4*N+3] - L51 * A[3] - L52 * U24 - L53 * U34) / U44;
    const T L64 = (A[5*N+3] - L61 * A[3] - L62 * U24 - L63 * U34) / U44;
    const T L74 = (A[6*N+3] - L71 * A[3] - L72 * U24 - L73 * U34) / U44;
    const T L84 = (A[7*N+3] - L81 * A[3] - L82 * U24 - L83 * U34) / U44;

    const T U55 = A[4*N+4] - L51 * A[4] - L52 * U25 - L53 * U35 - L54 * U45;
    const T U56 = A[4*N+5] - L51 * A[5] - L52 * U26 - L53 * U36 - L54 * U46;
    const T U57 = A[4*N+6] - L51 * A[6] - L52 * U27 - L53 * U37 - L54 * U47;
    const T U58 = A[4*N+7] - L51 * A[7] - L52 * U28 - L53 * U38 - L54 * U48;

    const T L65 = (A[5*N+4] - L61 * A[4] - L62 * U25 - L63 * U35 - L64 * U45) / U55;
    const T L75 = (A[6*N+4] - L71 * A[4] - L72 * U25 - L73 * U35 - L74 * U45) / U55;
    const T L85 = (A[7*N+4] - L81 * A[4] - L82 * U25 - L83 * U35 - L84 * U45) / U55;

    const T U66 = A[5*N+5] - L61 * A[5] - L62 * U26 - L63 * U36 - L64 * U46 - L65 * U56;
    const T U67 = A[5*N+6] - L61 * A[6] - L62 * U27 - L63 * U37 - L64 * U47 - L65 * U57;
    const T U68 = A[5*N+7] - L61 * A[7] - L62 * U28 - L63 * U38 - L64 * U48 - L65 * U58;

    const T L76 = (A[6*N+5] - L71 * A[5] - L72 * U26 - L73 * U36 - L74 * U46 - L75 * U56) / U66;
    const T L86 = (A[7*N+5] - L81 * A[5] - L82 * U26 - L83 * U36 - L84 * U46 - L85 * U56) / U66;

    const T U77 = A[6*N+6] - L71 * A[6] - L72 * U27 - L73 * U37 - L74 * U47 - L75 * U57 - L76 * U67;
    const T U78 = A[6*N+7] - L71 * A[7] - L72 * U28 - L73 * U38 - L74 * U48 - L75 * U58 - L76 * U68;

    const T L87 = (A[7*N+6] - L81 * A[6] - L82 * U27 - L83 * U37 - L84 * U47 - L85 * U57 - L86 * U67) / U77;

    const T U88 = A[7*N+7] - L81 * A[7] - L82 * U28 - L83 * U38 - L84 * U48 - L85 * U58 - L86 * U68 - L87 * U78;

    // L
    L[0*N+0]  = 1;
    L[0*N+1]  = 0;
    L[0*N+2]  = 0;
    L[0*N+3]  = 0;
    L[0*N+4]  = 0;
    L[0*N+5]  = 0;
    L[0*N+6]  = 0;
    L[0*N+7]  = 0;

    L[1*N+0]  = L21;
    L[1*N+1]  = 1;
    L[1*N+2]  = 0;
    L[1*N+3]  = 0;
    L[1*N+4]  = 0;
    L[1*N+5]  = 0;
    L[1*N+6]  = 0;
    L[1*N+7]  = 0;

    L[2*N+0]  = L31;
    L[2*N+1]  = L32;
    L[2*N+2]  = 1;
    L[2*N+3]  = 0;
    L[2*N+4]  = 0;
    L[2*N+5]  = 0;
    L[2*N+6]  = 0;
    L[2*N+7]  = 0;

    L[3*N+0]  = L41;
    L[3*N+1]  = L42;
    L[3*N+2]  = L43;
    L[3*N+3]  = 1;
    L[3*N+4]  = 0;
    L[3*N+5]  = 0;
    L[3*N+6]  = 0;
    L[3*N+7]  = 0;

    L[4*N+0]  = L51;
    L[4*N+1]  = L52;
    L[4*N+2]  = L53;
    L[4*N+3]  = L54;
    L[4*N+4]  = 1;
    L[4*N+5]  = 0;
    L[4*N+6]  = 0;
    L[4*N+7]  = 0;

    L[5*N+0]  = L61;
    L[5*N+1]  = L62;
    L[5*N+2]  = L63;
    L[5*N+3]  = L64;
    L[5*N+4]  = L65;
    L[5*N+5]  = 1;
    L[5*N+6]  = 0;
    L[5*N+7]  = 0;

    L[6*N+0]  = L71;
    L[6*N+1]  = L72;
    L[6*N+2]  = L73;
    L[6*N+3]  = L74;
    L[6*N+4]  = L75;
    L[6*N+5]  = L76;
    L[6*N+6]  = 1;
    L[6*N+7]  = 0;

    L[7*N+0]  = L81;
    L[7*N+1]  = L82;
    L[7*N+2]  = L83;
    L[7*N+3]  = L84;
    L[7*N+4]  = L85;
    L[7*N+5]  = L86;
    L[7*N+6]  = L87;
    L[7*N+7]  = 1;

    // U
    U[0*N+0]  = A00;
    U[0*N+1]  = A[1];
    U[0*N+2]  = A[2];
    U[0*N+3]  = A[3];
    U[0*N+4]  = A[4];
    U[0*N+5]  = A[5];
    U[0*N+6]  = A[6];
    U[0*N+7]  = A[7];

    U[1*N+0]  = 0;
    U[1*N+1]  = U22;
    U[1*N+2]  = U23;
    U[1*N+3]  = U24;
    U[1*N+4]  = U25;
    U[1*N+5]  = U26;
    U[1*N+6]  = U27;
    U[1*N+7]  = U28;

    U[2*N+0]  = 0;
    U[2*N+1]  = 0;
    U[2*N+2]  = U33;
    U[2*N+3]  = U34;
    U[2*N+4]  = U35;
    U[2*N+5]  = U36;
    U[2*N+6]  = U37;
    U[2*N+7]  = U38;

    U[3*N+0]  = 0;
    U[3*N+1]  = 0;
    U[3*N+2]  = 0;
    U[3*N+3]  = U44;
    U[3*N+4]  = U45;
    U[3*N+5]  = U46;
    U[3*N+6]  = U47;
    U[3*N+7]  = U48;

    U[4*N+0]  = 0;
    U[4*N+1]  = 0;
    U[4*N+2]  = 0;
    U[4*N+3]  = 0;
    U[4*N+4]  = U55;
    U[4*N+5]  = U56;
    U[4*N+6]  = U57;
    U[4*N+7]  = U58;

    U[5*N+0]  = 0;
    U[5*N+1]  = 0;
    U[5*N+2]  = 0;
    U[5*N+3]  = 0;
    U[5*N+4]  = 0;
    U[5*N+5]  = U66;
    U[5*N+6]  = U67;
    U[5*N+7]  = U68;

    U[6*N+0]  = 0;
    U[6*N+1]  = 0;
    U[6*N+2]  = 0;
    U[6*N+3]  = 0;
    U[6*N+4]  = 0;
    U[6*N+5]  = 0;
    U[6*N+6]  = U77;
    U[6*N+7]  = U78;

    U[7*N+0]  = 0;
    U[7*N+1]  = 0;
    U[7*N+2]  = 0;
    U[7*N+3]  = 0;
    U[7*N+4]  = 0;
    U[7*N+5]  = 0;
    U[7*N+6]  = 0;
    U[7*N+7]  = U88;
}

} // end of namespace Fastor

#endif // LUFACT_H
