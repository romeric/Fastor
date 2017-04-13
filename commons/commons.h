#ifndef COMMONS_H
#define COMMONS_H


#ifdef __GNUC__
    #ifndef __clang__
        #ifndef __INTEL_COMPILER
            #define FASTOR_GCC
        #endif
    #endif
#endif

#ifdef __INTEL_COMPILER
    #define FASTOR_INTEL
#endif

#ifdef __clang__
    #define FASTOR_CLANG
#endif

#if defined(_MSC_VER)
    #define FASTOR_MSC
#endif

#if defined(_MSC_VER)
    #if _MSC_VER < 1800
       #error FASTOR REQUIRES AN ISO C++11 COMPLIANT COMPILER
    #endif
#elif defined(__GNUC__) || defined(__GNUG__)
    #if __cplusplus <= 199711L
        #error FASTOR REQUIRES AN ISO C++11 COMPLIANT COMPILER
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

#if !defined(__FMA__) && defined(__AVX2__)
    #define __FMA__ 1
#endif


// Only for GCC & Clang, affects tensor divisions.
// ICC's default option is fast anyway (i.e. -fp-model fast=1)
// but it does not define the __FAST_MATH__ macro
#if defined(__FAST_MATH__)
#define FASTOR_FAST_MATH
#endif

// Define this if hadd seems beneficial
//#define USE_HADD

// ADDITIONAL MACROS DEFINED THROUGHOUT FASTOR
//-----------------------------------------------
// Bounds checking - on by default
#ifndef NDEBUG
#define BOUNDSCHECK
#define SHAPE_CHECK
#endif
//#define FASTOR_DONT_VECTORISE
//#define FASTOR_DONT_PERFORM_OP_MIN
//#define COPY_SMART_EXPR
#define FASTOR_MATMUL_UNROLL_LENGTH 1
//#define FASTOR_MATMUL_UNROLL_INNER
//#define FASTOR_USE_OLD_OUTER
//#define USE_OLD_VERSION // TO USE SOME OLD VERSIONS OF INTRINSICS
//#define FASTOR_USE_VECTORISED_EXPR_ASSIGN  // TO USE VECTORISED EXPRESSION ASSIGNMENT

#ifndef FASTOR_NO_ALIAS
#define FASTOR_DISALLOW_ALIASING
#endif

#define DepthFirst -200
#define NoDepthFirst -201
//-----------------------------------------------

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

#define SSE 128
#define AVX 256
#define AVX512 512
#define Scalar 64
#define Double 64
#define Single 32

#ifdef __SSE4_2__
    #ifdef __AVX__
        #ifdef __AVX512F__
            #define DEFAULT_ABI AVX512
        #else
            #define DEFAULT_ABI AVX
        #endif
    #else
        #define DEFAULT_ABI SSE
    #endif
#else
    // Define the largest float size as vector size
    #define DEFAULT_ABI Double
#endif


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
#define TWOPS (_mm_set1_ps(2.0f))
#define TOWPD (_mm_set1_pd(2.0))
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
#define VTWOPS (_mm256_set1_ps(2.0f))
#define VTOWPD (_mm256_set1_pd(2.0))
#endif

using FASTOR_INDEX = size_t;
using Int64 = long long int;
using DEFAULT_FLOAT_TYPE = double;
using DFT = DEFAULT_FLOAT_TYPE;
using FASTOR_VINDEX = volatile size_t;


#define ROUND_DOWN(x, s) ((x) & ~((s)-1))
#define PRECI_TOL 1e-14

#ifndef NDEBUG
void FASTOR_ASSERT(bool cond, const std::string &x) {
    if (cond==true) {
        return;
    }
    else {
        std::cout << x << std::endl;
        exit(EXIT_FAILURE);
    }
}
#else
void FASTOR_ASSERT(bool, const std::string&) {}
#endif

void FASTOR_WARN(bool cond, const std::string &x) {
    if (cond==true) {
        return;
    }
    else {
        std::cout << x << std::endl;
    }
}

#define _FASTOR_TOSTRING(X) #X
#define FASTOR_TOSTRING(X) _FASTOR_TOSTRING(X)


#if defined(__GNUC__)
    #define DEPRECATE(foo, msg) foo __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
    #define DEPRECATE(foo, msg) __declspec(deprecated(msg)) foo
#else
    #error FASTOR STATIC WARNING DOES NOT SUPPORT THIS COMPILER
#endif

#define PP_CAT(x,y) PP_CAT1(x,y)
#define PP_CAT1(x,y) x##y

namespace useless
{
    struct true_type {};
    struct false_type {};
    template <int test> struct converter : public true_type {};
    template <> struct converter<0> : public false_type {};
}

#define FASTOR_STATIC_WARN(cond, msg) \
struct PP_CAT(static_warning,__LINE__) { \
  DEPRECATE(void _(::useless::false_type const& ),msg) {}; \
  void _(::useless::true_type const& ) {}; \
  PP_CAT(static_warning,__LINE__)() {_(::useless::converter<(cond)>());} \
}



//
#define FASTOR_ISALIGNED(POINTER, BYTE_COUNT) \
    (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0)

// asm comment
#define FASTOR_ASM(STR) asm(STR ::)


#include "extended_algorithms.h"


#endif // COMMONS_H

