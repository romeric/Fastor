#ifndef COMMONS_H
#define COMMONS_H

#if __cplusplus < 201103
#if (defined Vc_MSVC && Vc_MSVC >= 160000000)
// these compilers still work, even if they don't define __cplusplus as expected
#else
#error "Fastor requires support for C++11."
#endif
#elif __cplusplus >= 201402L
# define Vc_CXX14 1
#endif

#if defined(_MSC_VER)
    #if _MSC_VER < 1800
       #error SIMDTensor needs a C++11 compliant compiler
    #endif
#elif defined(__GNUC__) || defined(__GNUG__)
    #if __cplusplus <= 199711L
        #error SIMDTensor needs a C++11 compliant compiler
    #endif
#endif

#if defined(__GNUC__) || defined(__GNUG__)
    #define FASTOR_INLINE inline __attribute__((always_inline))
    #define FASTOR_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
    #define FASTOR_INLINE __forceinline
    #define FASTOR_NOINLINE __declspec(noinline)
#endif

#if defined(__GNUC__) || defined(__GNUG__)
    #define FASTOR_ALIGN __attribute__((aligned(0x20)))
#elif defined(_MSC_VER)
    #define FASTOR_ALIGN __declspec(align(32))
#endif

// Define this if hadd seems beneficial
//#define USE_HADD

// Bounds check - on by default
#define BOUNDSCHECK

#include <cstdlib>
#include <cassert>

#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif


// FASTOR CONSTRUCTS
#define Symmetric -100
#define NonSymmetric -101
#define AntiSymmetric -102
#define Identity -103
#define One -104
#define Zero -105
#define Voigt -106

#define ThreeD -150
#define TwoD -151
#define PlaneStrain -152
#define PlaneStress -153

#define DepthFirst -200
#define NoDepthFirst -201


#define SSE 128
#define AVX 256
#define Scalar 64
#define Double 64
#define Single 32


#ifdef __SSE4_2__
#define ZEROPS (_mm_set1_ps(0.f))
#define ZEROPD (_mm_set1_pd(0.0))
// minus/negative version
#define MZEROPS (_mm_set1_ps(-0.f))
#define MZEROPD (_mm_set1_pd(-0.0))
#define ONEPS (_mm_set1_ps(1.f))
#define ONEPD (_mm_set1_pd(1.0))
#define HALFPS (_mm_set1_ps(0.5f))
#define HALFPD (_mm_set1_pd(0.5))
#endif
#ifdef __AVX__
#define VZEROPS (_mm256_set1_ps(0.f))
#define VZEROPD (_mm256_set1_pd(0.0))
// minus/negative version
#define MVZEROPS (_mm256_set1_ps(-0.f))
#define MVZEROPD (_mm256_set1_pd(-0.0))
#define VONEPS (_mm256_set1_ps(1.f))
#define VONEPD (_mm256_set1_pd(1.0))
#define VHALFPS (_mm256_set1_ps(0.5f))
#define VHALFPD (_mm256_set1_pd(0.5))
#endif

using FASTOR_INDEX = size_t;
using Int64 = long long int;


#define PRECI_TOL 1e-14

void FASTOR_ASSERT(bool cond, const std::string &x) {
    if (cond==true) {
        return;
    }
    else {
        std::cout << x << std::endl;
        exit(EXIT_FAILURE);
    }
}

void FASTOR_WARN(bool cond, const std::string &x) {
    if (cond==true) {
        return;
    }
    else {
        std::cout << x << std::endl;
    }
}


#include "extended_algorithms.h"


#endif // COMMONS_H

