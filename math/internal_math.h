#ifndef INTERNAL_MATH_H
#define INTERNAL_MATH_H

#include "extended_intrinsics/extintrin.h"
#include "meta/tensor_meta.h"

namespace Fastor {

//#define HAS_VDT

#ifdef HAS_VDT
#include <vdt/vdtMath.h>

__m256 internal_exp(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       out[i] = vdt::fast_expf(a[i]);
   }
   return out;
}
__m256d internal_exp(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       out[i] = vdt::fast_exp(a[i]);
   }
   return out;
}

__m256 internal_log(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       out[i] = vdt::fast_logf(a[i]);
   }
   return out;
}
__m256d internal_log(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       out[i] = vdt::fast_log(a[i]);
   }
   return out;
}

// This can give inaccurate solution
//__m256 internal_pow(__m256 a, __m256 b) {
//   __m256 out;
//   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
//       out[i] = vdt::fast_expf(a[i]*vdt::fast_logf(b[i]));
//   }
//   return out;
//}
//__m256d internal_pow(__m256d a, __m256d b) {
//   __m256d out;
//   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
//       out[i] = vdt::fast_exp(a[i]*vdt::fast_log(b[i]));
//   }
//   return out;
//}

__m256 internal_sin(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       out[i] = vdt::fast_sinf(a[i]);
   }
   return out;
}
__m256d internal_sin(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       out[i] = vdt::fast_sin(a[i]);
   }
   return out;
}

__m256 internal_cos(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       out[i] = vdt::fast_cosf(a[i]);
   }
   return out;
}
__m256d internal_cos(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       out[i] = vdt::fast_cos(a[i]);
   }
   return out;
}

__m256 internal_tan(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       out[i] = vdt::fast_tanf(a[i]);
   }
   return out;
}
__m256d internal_tan(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       out[i] = vdt::fast_tan(a[i]);
   }
   return out;
}

__m256 internal_asin(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       out[i] = vdt::fast_asinf(a[i]);
   }
   return out;
}
__m256d internal_asin(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       out[i] = vdt::fast_asin(a[i]);
   }
   return out;
}

__m256 internal_acos(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       out[i] = vdt::fast_acosf(a[i]);
   }
   return out;
}
__m256d internal_acos(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       out[i] = vdt::fast_acos(a[i]);
   }
   return out;
}

__m256 internal_atan(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       out[i] = vdt::fast_atanf(a[i]);
   }
   return out;
}
__m256d internal_atan(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       out[i] = vdt::fast_atan(a[i]);
   }
   return out;
}

#else

// SHUT GCC6 -Wignored-attributes WARNINGS
#ifdef __GNUC__
#if __GNUC__==6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#endif




template<typename T>
T internal_exp(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::exp(a[i]);
   }
   return out;
}

template<typename T>
T internal_log(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::log(a[i]);
   }
   return out;
}

template<typename T>
T internal_sin(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::sin(a[i]);
   }
   return out;
}

template<typename T>
T internal_cos(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::cos(a[i]);
   }
   return out;
}

template<typename T>
T internal_tan(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::tan(a[i]);
   }
   return out;
}

template<typename T>
T internal_asin(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::asin(a[i]);
   }
   return out;
}

template<typename T>
T internal_acos(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::acos(a[i]);
   }
   return out;
}

template<typename T>
T internal_atan(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::atan(a[i]);
   }
   return out;
}

#endif


// not available in vdt
template<typename T, typename U>
T internal_pow(T a, U b) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::pow(a[i],b[i]);
   }
   return out;
}

template<typename T>
T internal_sinh(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::sinh(a[i]);
   }
   return out;
}

template<typename T>
T internal_cosh(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::cosh(a[i]);
   }
   return out;
}

template<typename T>
T internal_tanh(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::tanh(a[i]);
   }
   return out;
}


}

#endif // INTERNAL_MATH_H

