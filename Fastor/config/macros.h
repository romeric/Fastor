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

#ifndef FASTOR_MACROS_H
#define FASTOR_MACROS_H


/* This file contains only the set of macros that Fastor defines and can be used or altered by the user
*/
//------------------------------------------------------------------------------------------------//
// Compiler version
//------------------------------------------------------------------------------------------------//
#define FASTOR_MAJOR 0
#define FASTOR_MINOR 6
#define FASTOR_PATCHLEVEL 4
#define FASTOR_VERSION (FASTOR_MAJOR * 10000 + FASTOR_MINOR * 100 + FASTOR_PATCHLEVEL)
//------------------------------------------------------------------------------------------------//

//------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------//
// ICC's default option is fast anyway (i.e. -fp-model fast=1)
// but it does not define the __FAST_MATH__ macro
#if defined(__FAST_MATH__)
#define FASTOR_FAST_MATH
// Use the following for unsafe math
// Only for GCC & Clang, activates fast div/rcp
//#define FASTOR_UNSAFE_MATH
#endif

// Runtime checks - off by default under release
//------------------------------------------------------------------------------------------------//
#ifndef FASTOR_ENABLE_RUNTIME_CHECKS
#define FASTOR_ENABLE_RUNTIME_CHECKS 0
#endif
//------------------------------------------------------------------------------------------------//

// Bounds and shape checking - ON by default under debug
//------------------------------------------------------------------------------------------------//
#if !defined(NDEBUG) || FASTOR_ENABLE_RUNTIME_CHECKS
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
//------------------------------------------------------------------------------------------------//


// Aliasing
//------------------------------------------------------------------------------------------------//
#ifndef FASTOR_NO_ALIAS
#define FASTOR_NO_ALIAS 0
#endif
//------------------------------------------------------------------------------------------------//


// Macros used throughout Fastor
//------------------------------------------------------------------------------------------------//
//#define FASTOR_COPY_EXPR
//#define FASTOR_DONT_VECTORISE
//#define FASTOR_DONT_PERFORM_OP_MIN
//#define FASTOR_USE_OLD_OUTER
//#define FASTOR_USE_OLD_INTRINSICS
//#define FASTOR_USE_HADD
//#define FASTOR_USE_VECTORISED_EXPR_ASSIGN  // to use vectorised expression assignment
//#define FASTOR_ZERO_INITIALISE
//#define FASTOR_USE_OLD_NDVIEWS
//#define FASTOR_DISPATCH_DIV_TO_MUL_EXPR // change BINARY_DIV_OP to BINARY_MUL_OP for Expression/Number
//#define FASTOR_DISABLE_SPECIALISED_CTR

//FASTOR_MATMUL_OUTER_BLOCK_SIZE 2
//FASTOR_MATMUL_INNER_BLOCK_SIZE 2

//FASTOR_TRANS_OUTER_BLOCK_SIZE 2
//FASTOR_TRANS_INNER_BLOCK_SIZE 2

#ifndef FASTOR_BLAS_SWITCH_MATRIX_SIZE
#define FASTOR_BLAS_SWITCH_MATRIX_SIZE 16
#endif

// FASTOR_NIL
//------------------------------------------------------------------------------------------------//
#define FASTOR_NIL 0
//------------------------------------------------------------------------------------------------//


// Bit sizes
//------------------------------------------------------------------------------------------------//
#define FASTOR_AVX512_BITSIZE 512
#define FASTOR_AVX_BITSIZE 256
#define FASTOR_SSE_BITSIZE 128
#define FASTOR_DOUBLE_BITSIZE (sizeof(double)*8)
#define FASTOR_SINGLE_BITSIZE (sizeof(float)*8)
#ifndef FASTOR_SCALAR_BITSIZE
#define FASTOR_SCALAR_BITSIZE FASTOR_DOUBLE_BITSIZE
#endif
//------------------------------------------------------------------------------------------------//


// Alignment
//------------------------------------------------------------------------------------------------//
#ifndef FASTOR_MEMORY_ALIGNMENT_VALUE
#if defined(FASTOR_AVX512_IMPL)
#define FASTOR_MEMORY_ALIGNMENT_VALUE 64
#elif defined(FASTOR_AVX_IMPL)
#define FASTOR_MEMORY_ALIGNMENT_VALUE 32
#elif defined(FASTOR_SSE_IMPL)
#define FASTOR_MEMORY_ALIGNMENT_VALUE 16
#else
#define FASTOR_MEMORY_ALIGNMENT_VALUE 8
#endif
#endif

/* User controllable alignment for Fastor containers */
#define FASTOR_ALIGN alignas((FASTOR_MEMORY_ALIGNMENT_VALUE))
/* Strict non-controllable alignment for Fastor's internal use */
#define FASTOR_ARCH_ALIGN alignas((FASTOR_MEMORY_ALIGNMENT_VALUE))

// Conservative alignment for SIMD
#if defined(FASTOR_DONT_ALIGN) || defined(FASTOR_DONT_VECTORISE)
    #define FASTOR_ALIGNED false
#else
    #define FASTOR_ALIGNED true
#endif

#define FASTOR_ISALIGNED(POINTER, BYTE_COUNT) \
    (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0)
//------------------------------------------------------------------------------------------------//


// Assertions
//------------------------------------------------------------------------------------------------//
namespace Fastor {
// Strong unconditional assert
FASTOR_INLINE void FASTOR_EXIT_ASSERT(bool cond, const std::string &msg="") {
    if (cond==false) {
        throw std::runtime_error(msg);
    }
}

#if !defined(NDEBUG) || FASTOR_ENABLE_RUNTIME_CHECKS
#define FASTOR_ASSERT(COND, MESSAGE) FASTOR_EXIT_ASSERT(COND, MESSAGE)
#else
#define FASTOR_ASSERT(COND, MESSAGE)
#endif

// Warn
FASTOR_INLINE void FASTOR_WARN(bool cond, const std::string &x) {
    if (cond==false) {
        std::cout << x << std::endl;
    }
}
} // end of namespace Fastor
//------------------------------------------------------------------------------------------------//


// Warnings
//------------------------------------------------------------------------------------------------//
// Static warn
#ifndef FASTOR_NO_STATIC_WARNING
#if defined(__GNUC__)
    #define FASTOR_DEPRECATE(foo, msg) foo __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
    #define FASTOR_DEPRECATE(foo, msg) __declspec(deprecated(msg)) foo
#else
    #error FASTOR STATIC WARNING DOES NOT SUPPORT THIS COMPILER
#endif

#define FASTOR_CAT(x,y) FASTOR_CAT1_(x,y)
#define FASTOR_CAT1_(x,y) x##y


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
#endif // FASTOR_NO_STATIC_WARNING
//------------------------------------------------------------------------------------------------//


// Token to string
//------------------------------------------------------------------------------------------------//
#define FASTOR_TOSTRING_(X) #X
#define FASTOR_TOSTRING(X) FASTOR_TOSTRING_(X)
//------------------------------------------------------------------------------------------------//

// asm comment
//------------------------------------------------------------------------------------------------//
#define FASTOR_ASM(STR) asm(STR ::)
//------------------------------------------------------------------------------------------------//

// unused
//------------------------------------------------------------------------------------------------//
namespace Fastor {
// clobber
template <typename T> void unused(T &&x) {
#ifndef _WIN32
    asm("" ::"m"(x));
#endif
}
template <typename T, typename ... U> void unused(T&& x, U&& ...y) { unused(x); unused(y...); }
} // end of namespace Fastor
//------------------------------------------------------------------------------------------------//


// SIMD round-down
//------------------------------------------------------------------------------------------------//
#define ROUND_DOWN2(x, s) ((x) & ~((s)-1))
#define ROUND_DOWN(x, s) ROUND_DOWN2(x,s)
//------------------------------------------------------------------------------------------------//


// FASTOR CONSTRUCTS
//------------------------------------------------------------------------------------------------//
#include <cstdint>

namespace Fastor {

using FASTOR_INDEX = size_t;
using Int64 = int64_t;
using DEFAULT_FLOAT_TYPE = double;
using DFT = DEFAULT_FLOAT_TYPE;
using FASTOR_VINDEX = volatile size_t;

constexpr int RowMajor = 0;
constexpr int ColumnMajor = 1;


constexpr int FASTOR_Symmetric = -100;
constexpr int FASTOR_NonSymmetric = -101;
constexpr int FASTOR_AntiSymmetric = -102;
constexpr int FASTOR_Identity = -103;
constexpr int FASTOR_One = -104;
constexpr int FASTOR_Zero = -105;

constexpr int FASTOR_ThreeD = -150;
constexpr int FASTOR_TwoD = -151;
constexpr int FASTOR_PlaneStrain = -152;
constexpr int FASTOR_PlaneStress = -153;
constexpr int FASTOR_Voigt = -106;

constexpr int DepthFirst = -200;
constexpr int NoDepthFirst = -201;

constexpr double PRECI_TOL  = 1e-14;

}
//------------------------------------------------------------------------------------------------//

#endif // FASTOR_MACROS_H
