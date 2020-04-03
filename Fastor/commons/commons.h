/*  This file is part of the FASTOR library. {{{
Copyright (c) 2016 Roman Poya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

}}}*/

#ifndef COMMONS_H
#define COMMONS_H

#include <cstdlib>
#include <cassert>

//------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------//

// Error out for pre-C++11 compiler versions
//------------------------------------------------------------------------------------------------//
#if defined(_MSC_VER)
    #if _MSC_VER < 1800
       #error FASTOR REQUIRES AN ISO C++11 COMPLIANT COMPILER
    #endif
#elif defined(__GNUC__) || defined(__GNUG__)
    #if __cplusplus <= 199711L
        #error FASTOR REQUIRES AN ISO C++11 COMPLIANT COMPILER
    #endif
#endif
//------------------------------------------------------------------------------------------------//


// Compiler define macros
//------------------------------------------------------------------------------------------------//
#ifdef __INTEL_COMPILER
#define FASTOR_INTEL __INTEL_COMPILER_BUILD_DATE
#elif defined(__clang__) && defined(__apple_build_version__)
#define FASTOR_APPLECLANG (__clang_major__ * 0x10000 + __clang_minor__ * 0x100 + __clang_patchlevel__)
#elif defined(__clang__)
#define FASTOR_CLANG (__clang_major__ * 0x10000 + __clang_minor__ * 0x100 + __clang_patchlevel__)
#elif defined(__GNUC__)
#define FASTOR_GCC (__GNUC__ * 0x10000 + __GNUC_MINOR__ * 0x100 + __GNUC_PATCHLEVEL__)
#elif defined(_MSC_VER)
#define FASTOR_MSVC _MSC_FULL_VER
#else
#define FASTOR_UNSUPPORTED_COMPILER 1
#endif
//------------------------------------------------------------------------------------------------//


// Determine CXX version
//------------------------------------------------------------------------------------------------//
#if defined(__cplusplus)
    #if __cplusplus == 201103L
        #define FASTOR_CXX_VERSION 2011
    #elif __cplusplus == 201402L
        #define FASTOR_CXX_VERSION 2014
    #elif __cplusplus == 201703L
        #define FASTOR_CXX_VERSION 2017
    #endif
#endif
//------------------------------------------------------------------------------------------------//


// Force inline and no-inline macros
//------------------------------------------------------------------------------------------------//
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

// Traditional inline which works will and helps the compiler
// eliminate a lot of code
#define FASTOR_HINT_INLINE inline
//------------------------------------------------------------------------------------------------//


// C++17 [if constexpr] define
//------------------------------------------------------------------------------------------------//
#if FASTOR_CXX_VERSION == 2017
    #define FASTOR_HAS_IF_CONSTEXPR 1
    #define FASTOR_IF_CONSTEXPR if constexpr
#else
    #define FASTOR_HAS_IF_CONSTEXPR 0
    #define FASTOR_IF_CONSTEXPR if
#endif
//------------------------------------------------------------------------------------------------//



// Intrinsics defines
//------------------------------------------------------------------------------------------------//
#ifdef FASTOR_MSVC
    #ifdef _M_IX86_FP
        #if _M_IX86_FP >= 1
            #ifndef __SSE__
                #define __SSE__ 1
            #endif
        #endif
        #if _M_IX86_FP >= 2
            #ifndef __SSE2__
                #define __SSE2__ 1
            #endif
        #endif
    #elif defined(_M_AMD64)
        #ifndef __SSE__
            #define __SSE__ 1
        #endif
        #ifndef __SSE2__
            #define __SSE2__ 1
        #endif
    #endif
#endif

#if defined(__MIC__)
    #define FASTOR_MIC_IMPL 1
#endif
#if defined(__AVX512F__)
    #define FASTOR_AVX512F_IMPL 1
#endif
#if defined(__AVX512CD__)
    #define FASTOR_AVX512CD_IMPL 1
#endif
#if defined(__AVX2__)
    #define FASTOR_AVX2_IMPL 1
#endif
#if defined(__AVX__)
    #define FASTOR_AVX_IMPL 1
#endif
#if defined(__SSE4_2__)
    #define FASTOR_SSE4_2_IMPL 1
#endif
#if defined(__SSE4_1__)
    #define FASTOR_SSE4_1_IMPL 1
#endif
#if defined(__SSE3__)
    #define FASTOR_SSE3_IMPL 1
#endif
#if defined(__SSSE3__)
    #define FASTOR_SSSE3_IMPL 1
#endif
#if defined(__SSE2__)
    #define FASTOR_SSE2_IMPL 1
#endif

#if defined(__FMA4__)
    #define FASTOR_FMA4_IMPL 1
#endif
#if defined(__FMA__)
    #define FASTOR_FMA_IMPL 1
#endif
#if !defined(__FMA__) && defined(__AVX2__)
    #define __FMA__ 1
#endif


#if defined(FASTOR_AVX512F_IMPL) || defined(FASTOR_AVX512CD_IMPL)
    #define FASTOR_AVX512_IMPL 1
#endif
#if defined(FASTOR_AVX2_IMPL) && !defined(FASTOR_AVX_IMPL)
    #define FASTOR_AVX_IMPL 1
#endif
#if defined(FASTOR_SSE2_IMPL) || defined(FASTOR_SSSE3_IMPL) || defined(FASTOR_SSE3_IMPL) || defined(FASTOR_SSE4_1_IMPL) || defined(FASTOR_SSE4_2_IMPL)
    #define FASTOR_SSE_IMPL 1
#endif

#if !defined(FASTOR_MIC_IMPL) && !defined(FASTOR_AVX512_IMPL) && !defined(FASTOR_AVX_IMPL) && !defined(FASTOR_SSE_IMPL)
#define FASTOR_SCALAR_IMPL 1
#endif


#ifdef FASTOR_SSE2_IMPL
#include <emmintrin.h>
#endif
#ifdef FASTOR_SSE4_1_IMPL
#include <smmintrin.h>
#endif
#ifdef FASTOR_AVX_IMPL
#include <immintrin.h>
#endif


#define FASTOR_AVX512_BITSIZE 512
#define FASTOR_AVX_BITSIZE 256
#define FASTOR_SSE_BITSIZE 128
#define FASTOR_DOUBLE_BITSIZE (sizeof(double)*8)
#define FASTOR_SINGLE_BITSIZE (sizeof(float)*8)
#ifndef FASTOR_SCALAR_BITSIZE
#define FASTOR_SCALAR_BITSIZE FASTOR_DOUBLE_SIZE
#endif


// Conservative alignment for SIMD
#if defined(FASTOR_CONSERVATIVE_ALIGN) || defined(FASTOR_DONT_VECTORISE)
    #define FASTOR_ALIGNED false
#else
    #define FASTOR_ALIGNED true
#endif

#define FASTOR_ISALIGNED(POINTER, BYTE_COUNT) \
    (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0)

//------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------//




// Fastor internal defines
//------------------------------------------------------------------------------------------------//
// ICC's default option is fast anyway (i.e. -fp-model fast=1)
// but it does not define the __FAST_MATH__ macro
#if defined(__FAST_MATH__)
#define FASTOR_FAST_MATH
// Use the following for unsafe math
// Only for GCC & Clang, activates fast div/rcp
//#define FASTOR_UNSAFE_MATH
#endif

// Bounds and shape checking - ON by default
#ifndef NDEBUG
#ifndef FASTOR_BOUNDS_CHECK
#define FASTOR_BOUNDS_CHECK 1
#endif
#ifndef FASTOR_SHAPE_CHECK
#define FASTOR_SHAPE_CHECK 1
#endif
#else
#ifndef FASTOR_BOUNDS_CHECK
#define FASTOR_BOUNDS_CHECK 0
#endif
#ifndef FASTOR_SHAPE_CHECK
#define FASTOR_SHAPE_CHECK 0
#endif
#endif

//#define FASTOR_DONT_VECTORISE
//#define FASTOR_DONT_PERFORM_OP_MIN
//#define FASTOR_USE_OLD_OUTER
//#define FASTOR_USE_OLD_INTRINSICS
//#define FASTOR_USE_VECTORISED_EXPR_ASSIGN  // To use vectorised expression assignment
//#define FASTOR_ZERO_INITIALISE
//#define FASTOR_USE_OLD_NDVIEWS
//#define FASTOR_DISPATCH_DIV_TO_MUL_EXPR // Change BINARY_DIV_OP to BINARY_MUL_OP for Expression/Number
//#define FASTOR_DISABLE_SPECIALISED_CTR

//FASTOR_MATMUL_OUTER_BLOCK_SIZE 2
//FASTOR_MATMUL_INNER_BLOCK_SIZE 2

//FASTOR_TRANS_OUTER_BLOCK_SIZE 2
//FASTOR_TRANS_INNER_BLOCK_SIZE 2

#ifndef FASTOR_BLAS_SWITCH_MATRIX_SIZE
#define FASTOR_BLAS_SWITCH_MATRIX_SIZE 16
#endif

// This changes the behaviour of all expression templates (apart from views)
#ifdef FASTOR_COPY_EXPR
#define COPY_SMART_EXPR
#endif

// Define this if hadd seems beneficial
//#define FASTOR_USE_HADD

#ifndef FASTOR_NO_ALIAS
#define FASTOR_DISALLOW_ALIASING
#endif

#define DepthFirst -200
#define NoDepthFirst -201
//------------------------------------------------------------------------------------------------//




// FASTOR CONSTRUCTS
//------------------------------------------------------------------------------------------------//
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


// asm comment
#define FASTOR_ASM(STR) asm(STR ::)
//------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------//

#include "extended_algorithms.h"

//------------------------------------------------------------------------------------------------//


#endif // COMMONS_H
