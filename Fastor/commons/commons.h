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

#if defined(__cplusplus)
    #if __cplusplus == 201103L
        #define FASTOR_CXX_VERSION 2011
    #elif __cplusplus == 201402L
        #define FASTOR_CXX_VERSION 2014
    #elif __cplusplus == 201703L
        #define FASTOR_CXX_VERSION 2017
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
#define FASTOR_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define FASTOR_RESTRICT __restrict
#endif

#if defined(__GNUC__) || defined(__GNUG__)
    #define FASTOR_ALIGN __attribute__((aligned(0x20)))
#elif defined(_MSC_VER)
    #define FASTOR_ALIGN __declspec(align(32))
#endif

#if !defined(__FMA__) && defined(__AVX2__)
    #define __FMA__ 1
#endif

// Traditional inline which works will helps the compiler
// eliminate a lot of code
#define FASTOR_HINT_INLINE inline


// C++17 if constexpr
#if FASTOR_CXX_VERSION == 2017
    #define FASTOR_HAS_IF_CONSTEXPR 1
    #define FASTOR_IF_CONSTEXPR if constexpr
#else
    #define FASTOR_HAS_IF_CONSTEXPR 0
    #define FASTOR_IF_CONSTEXPR if
#endif


// ICC's default option is fast anyway (i.e. -fp-model fast=1)
// but it does not define the __FAST_MATH__ macro
#if defined(__FAST_MATH__)
#define FASTOR_FAST_MATH
// Use the following for unsafe math
// Only for GCC & Clang, affects tensor divisions.
// #define FASTOR_UNSAFE_MATH
#endif


// ADDITIONAL MACROS DEFINED THROUGHOUT FASTOR
//-----------------------------------------------
// Bounds checking - on by default
#ifndef NDEBUG
#define BOUNDSCHECK
#define SHAPE_CHECK
#endif
//#define FASTOR_DONT_VECTORISE
//#define FASTOR_DONT_PERFORM_OP_MIN
//#define FASTOR_USE_OLD_OUTER
//#define USE_OLD_VERSION // TO USE SOME OLD VERSIONS OF INTRINSICS
//#define FASTOR_USE_VECTORISED_EXPR_ASSIGN  // TO USE VECTORISED EXPRESSION ASSIGNMENT
//#define FASTOR_ZERO_INITIALISE
//#define FASTOR_USE_OLD_NDVIEWS
//#define FASTOR_DISPATCH_DIV_TO_MUL_EXPR // CHANGE BINARY_DIV_OP TO BINARY_MUL_OP
//#define FASTOR_DISABLE_SPECIALISED_CTR

#ifndef BLAS_SWITCH_MATRIX_SIZE_NS
#define BLAS_SWITCH_MATRIX_SIZE_NS 13
#endif

#ifndef BLAS_SWITCH_MATRIX_SIZE_S
#define BLAS_SWITCH_MATRIX_SIZE_S 16
#endif

// This changes the behaviour of all expression templates (apart from views)
#ifdef FASTOR_COPY_EXPR
// ALL SMART EXPRESSION TEMPLATES WITH MAKE A COPY OF UNDERLYING TENSORS
#define COPY_SMART_EXPR
#endif

// Define this if hadd seems beneficial
//#define USE_HADD

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
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif

// FASTOR CONSTRUCTS
#define FASTOR_Symmetric -100
#define FASTOR_NonSymmetric -101
#define FASTOR_AntiSymmetric -102
#define FASTOR_Identity -103
#define FASTOR_One -104
#define FASTOR_Zero -105

#define FASTOR_ThreeD -150
#define FASTOR_TwoD -151
#define FASTOR_PlaneStrain -152
#define FASTOR_PlaneStress -153
#define FASTOR_Voigt -106

#define FASTOR_SSE 128
#define FASTOR_AVX 256
#define FASTOR_AVX512 512
#define FASTOR_Double 64
#define FASTOR_Single 32
#ifndef FASTOR_Scalar
#define FASTOR_Scalar 64
#endif

#ifdef __SSE4_2__
    #ifdef __AVX__
        #ifdef __AVX512F__
            #define DEFAULT_ABI FASTOR_AVX512
        #else
            #define DEFAULT_ABI FASTOR_AVX
        #endif
    #else
        #define DEFAULT_ABI FASTOR_SSE
    #endif
#else
    // Define the largest floating point size as vector size
    #define DEFAULT_ABI FASTOR_Scalar
#endif

// Conservative alignment for SIMD
#ifdef __SSE4_2__
    #ifdef __AVX__
        #ifdef FASTOR_CONSERVATIVE_ALIGN
            #define IS_ALIGNED false
        #else
            #define IS_ALIGNED true
        #endif
    #else
        #define IS_ALIGNED true
    #endif
#else
    #define IS_ALIGNED true
#endif


#define ROUND_DOWN2(x, s) ((x) & ~((s)-1))
#define ROUND_DOWN(x, s) ROUND_DOWN2(x,s)
#define PRECI_TOL 1e-14

namespace Fastor {

using FASTOR_INDEX = size_t;
using Int64 = long long int;
using DEFAULT_FLOAT_TYPE = double;
using DFT = DEFAULT_FLOAT_TYPE;
using FASTOR_VINDEX = volatile size_t;

constexpr int RowMajor = 0;
constexpr int ColumnMajor = 1;


#ifndef NDEBUG
#define FASTOR_ASSERT(COND, MESSAGE) assert(COND && MESSAGE)
#else
#define FASTOR_ASSERT(COND, MESSAGE)
#endif
// The following assert is provided for cases where despite
// the DNDEBUG one might want the code to stop at failure
FASTOR_INLINE void FASTOR_EXIT_ASSERT(bool cond, const std::string &x="") {
    if (cond==true) {
        return;
    }
    else {
        std::cout << x << '\n';
        exit(EXIT_FAILURE);
    }
}

FASTOR_INLINE void FASTOR_WARN(bool cond, const std::string &x) {
    if (cond==true) {
        return;
    }
    else {
        std::cout << x << std::endl;
    }
}

} // end of namespace Fastor

#define _FASTOR_TOSTRING(X) #X
#define FASTOR_TOSTRING(X) _FASTOR_TOSTRING(X)


#ifndef FASTOR_NO_STATIC_WARNING
#if defined(__GNUC__)
    #define FASTOR_DEPRECATE(foo, msg) foo __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
    #define FASTOR_DEPRECATE(foo, msg) __declspec(deprecated(msg)) foo
#else
    #error FASTOR STATIC WARNING DOES NOT SUPPORT THIS COMPILER
#endif

#define FASTOR_CAT(x,y) _FASTOR_CAT1(x,y)
#define _FASTOR_CAT1(x,y) x##y


namespace Fastor {

namespace useless
{
    struct true_type {};
    struct false_type {};
    template <int test> struct converter : public true_type {};
    template <> struct converter<0> : public false_type {};
}

#define FASTOR_STATIC_WARN(cond, msg) \
struct FASTOR_CAT(static_warning,__LINE__) { \
  FASTOR_DEPRECATE(void _(::useless::false_type const& ),msg) {}; \
  void _(::useless::true_type const& ) {}; \
  FASTOR_CAT(static_warning,__LINE__)() {_(::useless::converter<(cond)>());} \
}

} // end of namespace Fastor
#endif

//
#define FASTOR_ISALIGNED(POINTER, BYTE_COUNT) \
    (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0)

// asm comment
#define FASTOR_ASM(STR) asm(STR ::)


#include "extended_algorithms.h"


#endif // COMMONS_H