/*  This file is part of the FASTOR library. {{{
Copyright (c) 2020 Roman Poya

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

#ifndef CONFIG_H
#define CONFIG_H

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

//------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------//

// Error out for pre-C++14 compiler versions
//------------------------------------------------------------------------------------------------//
#if defined(_MSC_VER)
    #if _MSC_VER < 1920
       #error FASTOR REQUIRES AN ISO C++14 COMPLIANT COMPILER
    #endif
#elif defined(__GNUC__) || defined(__GNUG__)
    #if __cplusplus < 201402L
        #error FASTOR REQUIRES AN ISO C++14 COMPLIANT COMPILER
    #endif
#endif
//------------------------------------------------------------------------------------------------//


// Compiler define macros
//------------------------------------------------------------------------------------------------//
#ifdef __INTEL_COMPILER
#define FASTOR_INTEL __INTEL_COMPILER_BUILD_DATE
#elif defined(__clang__) && defined(__apple_build_version__)
#define FASTOR_APPLECLANG (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#elif defined(__clang__)
#define FASTOR_CLANG (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#elif defined(__GNUC__)
#define FASTOR_GCC (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#elif defined(_MSC_VER)
#define FASTOR_MSVC _MSC_FULL_VER
#else
#define FASTOR_UNSUPPORTED_COMPILER 1
#endif
//------------------------------------------------------------------------------------------------//


// Operating system define macros
//------------------------------------------------------------------------------------------------//
#if defined(_WIN32)
#define FASTOR_WINDOWS32_OS 1
#endif
#if defined(_WIN64)
#define FASTOR_WINDOWS64_OS 1
#endif
#if defined(_WIN32) || defined(_WIN64)
#define FASTOR_WINDOWS_OS 1
#endif
#if defined(__unix__) || defined(__unix)
#define FASTOR_UNIX_OS 1
#endif
#if defined(__linux__)
#define FASTOR_LINUX_OS 1
#endif
#if defined(__APPLE__) || defined(__MACH__)
#define FASTOR_APPLE_OS 1
#endif
#if defined(__ANDROID__) || defined(ANDROID)
#define FASTOR_ANDROID_OS 1
#endif
#if defined(__CYGWIN__)
#define FASTOR_CYGWIN_OS 1
#endif
//------------------------------------------------------------------------------------------------//


// Determine CXX version
//------------------------------------------------------------------------------------------------//
#if defined(FASTOR_MSVC)
#if defined(_MSVC_LANG)
    #if _MSVC_LANG == 199711L
        #define FASTOR_CXX_VERSION 1998
    #elif _MSVC_LANG == 201103L
        #define FASTOR_CXX_VERSION 2011
    #elif _MSVC_LANG == 201402L
        #define FASTOR_CXX_VERSION 2014
    #elif _MSVC_LANG == 201703L
        #define FASTOR_CXX_VERSION 2017
    #elif _MSVC_LANG > 201703L
        #define FASTOR_CXX_VERSION 2020
    #endif
#endif
#else
#if defined(__cplusplus)
    #if __cplusplus == 199711L
        #define FASTOR_CXX_VERSION 1998
    #elif __cplusplus == 201103L
        #define FASTOR_CXX_VERSION 2011
    #elif __cplusplus == 201402L
        #define FASTOR_CXX_VERSION 2014
    #elif __cplusplus == 201703L
        #define FASTOR_CXX_VERSION 2017
    #elif __cplusplus > 201703L
        #define FASTOR_CXX_VERSION 2020
    #endif
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
#if defined(__AVX512BW__)
    #define FASTOR_AVX512BW_IMPL 1
#endif
#if defined(__AVX512DQ__)
    #define FASTOR_AVX512DQ_IMPL 1
#endif
#if defined(__AVX512VL__)
    #define FASTOR_AVX512VL_IMPL 1
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
// #if !defined(__FMA__) && defined(__AVX2__)
//     #define __FMA__ 1
// #endif


// Get around MSVC issue
#if defined(FASTOR_WINDOWS_OS) || defined(FASTOR_MSVC)
    #if defined(FASTOR_AVX512F_IMPL)
        #if !defined(FASTOR_SSE2_IMPL)
            #define FASTOR_SSE2_IMPL 1
        #endif
        #if !defined(FASTOR_SSE3_IMPL)
            #define FASTOR_SSE3_IMPL 1
        #endif
        #if !defined(FASTOR_SSSE3_IMPL)
            #define FASTOR_SSSE3_IMPL 1
        #endif
        #if !defined(FASTOR_SSE4_1_IMPL)
            #define FASTOR_SSE4_1_IMPL 1
        #endif
        #if !defined(FASTOR_SSE4_2_IMPL)
            #define FASTOR_SSE4_2_IMPL 1
        #endif
        #if !defined(FASTOR_AVX_IMPL)
            #define FASTOR_AVX_IMPL 1
        #endif
        #if !defined(FASTOR_AVX2_IMPL)
            #define FASTOR_AVX2_IMPL 1
        #endif
    #elif defined(FASTOR_AVX_IMPL) || defined(FASTOR_AVX2_IMPL)
        #if !defined(FASTOR_SSE2_IMPL)
            #define FASTOR_SSE2_IMPL 1
        #endif
        #if !defined(FASTOR_SSE3_IMPL)
            #define FASTOR_SSE3_IMPL 1
        #endif
        #if !defined(FASTOR_SSSE3_IMPL)
            #define FASTOR_SSSE3_IMPL 1
        #endif
        #if !defined(FASTOR_SSE4_1_IMPL)
            #define FASTOR_SSE4_1_IMPL 1
        #endif
        #if !defined(FASTOR_SSE4_2_IMPL)
            #define FASTOR_SSE4_2_IMPL 1
        #endif
    #endif
#endif


#if defined(FASTOR_AVX512F_IMPL) || defined(FASTOR_AVX512CD_IMPL) || defined(FASTOR_AVX512BW_IMPL) || defined(FASTOR_AVX512DQ_IMPL) || defined(FASTOR_AVX512VL_IMPL)
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
#ifdef FASTOR_SSE3_IMPL
#include <pmmintrin.h>
#endif
#ifdef FASTOR_SSSE3_IMPL
#include <tmmintrin.h>
#endif
#ifdef FASTOR_SSE4_1_IMPL
#include <smmintrin.h>
#endif
#ifdef FASTOR_SSE4_2_IMPL
#include <nmmintrin.h>
#endif
#ifdef FASTOR_AVX_IMPL
#include <immintrin.h>
#endif


// Mask loading
//------------------------------------------------------------------------------------------------//
#if defined(FASTOR_AVX512F_IMPL) && defined(FASTOR_AVX512VL_IMPL)
    #define FASTOR_HAS_AVX512_MASKS 1
#endif
//------------------------------------------------------------------------------------------------//

// Horizontal add
//------------------------------------------------------------------------------------------------//
#if defined(FASTOR_AVX512F_IMPL)
#if defined(FASTOR_INTEL)
    #define FASTOR_HAS_AVX512_REDUCE_ADD 1
#elif defined (FASTOR_GCC) && __GNUC__ >= 7
    #define FASTOR_HAS_AVX512_REDUCE_ADD 1
#elif defined (FASTOR_GCC) && __clang_major__ >= 4
    #define FASTOR_HAS_AVX512_REDUCE_ADD 1
#endif
#endif
//------------------------------------------------------------------------------------------------//

// AVX512 abs
//------------------------------------------------------------------------------------------------//
#if defined(FASTOR_AVX512F_IMPL)
#if defined(FASTOR_INTEL)
    #define FASTOR_HAS_AVX512_ABS 1
#elif defined (FASTOR_GCC) && __GNUC__ >= 7 && __GNUC_MINOR__ >= 4
    #define FASTOR_HAS_AVX512_ABS 1
#elif defined (FASTOR_GCC) && __clang_major__ >= 4
    #define FASTOR_HAS_AVX512_ABS 1
#endif
#endif
//------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------//

#include "Fastor/config/macros.h"

//------------------------------------------------------------------------------------------------//

#endif // CONFIG_H
