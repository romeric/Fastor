#ifndef INTERNAL_MATH_H
#define INTERNAL_MATH_H

#include "Fastor/extended_intrinsics/extintrin.h"
#include "Fastor/meta/tensor_meta.h"

namespace Fastor {

//#define HAS_VDT

#ifdef HAS_VDT
#include <vdt/vdtMath.h>

#ifdef FASTOR_SSE4_2_IMPL
inline __m128 internal_exp(__m128 a) {
   __m128 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
      ((float*)&out)[i] = vdt::fast_expf(((float*)&a)[i]);
   }
   return out;
}
inline __m128 internal_log(__m128 a) {
   __m128 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
      ((float*)&out)[i] = vdt::fast_logf(((float*)&a)[i]);
   }
   return out;
}
inline __m128 internal_sin(__m128 a) {
   __m128 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       ((float*)&out)[i] = vdt::fast_sinf(((float*)&a)[i]);
   }
   return out;
}
inline __m128 internal_cos(__m128 a) {
   __m128 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       ((float*)&out)[i] = vdt::fast_cosf(((float*)&a)[i]);
   }
   return out;
}
inline __m128 internal_tan(__m128 a) {
   __m128 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       ((float*)&out)[i] = vdt::fast_tanf(((float*)&a)[i]);
   }
   return out;
}
inline __m128 internal_asin(__m128 a) {
   __m128 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       ((float*)&out)[i] = vdt::fast_asinf(((float*)&a)[i]);
   }
   return out;
}
inline __m128 internal_acos(__m128 a) {
   __m128 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       ((float*)&out)[i] = vdt::fast_acosf(((float*)&a)[i]);
   }
   return out;
}
inline __m128 internal_atan(__m128 a) {
   __m128 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       ((float*)&out)[i] = vdt::fast_atanf(((float*)&a)[i]);
   }
   return out;
}



inline __m128d internal_exp(__m128d a) {
   __m128d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
      ((double*)&out)[i] = vdt::fast_exp(((double*)&a)[i]);
   }
   return out;
}
inline __m128d internal_log(__m128d a) {
   __m128d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
      ((double*)&out)[i] = vdt::fast_log(((double*)&a)[i]);
   }
   return out;
}
inline __m128d internal_sin(__m128d a) {
   __m128d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       ((double*)&out)[i] = vdt::fast_sin(((double*)&a)[i]);
   }
   return out;
}
inline __m128d internal_cos(__m128d a) {
   __m128d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       ((double*)&out)[i] = vdt::fast_cos(((double*)&a)[i]);
   }
   return out;
}
inline __m128d internal_tan(__m128d a) {
   __m128d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       ((double*)&out)[i] = vdt::fast_tan(((double*)&a)[i]);
   }
   return out;
}
inline __m128d internal_asin(__m128d a) {
   __m128d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       ((double*)&out)[i] = vdt::fast_asin(((double*)&a)[i]);
   }
   return out;
}
inline __m128d internal_acos(__m128d a) {
   __m128d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       ((double*)&out)[i] = vdt::fast_acos(((double*)&a)[i]);
   }
   return out;
}
inline __m128d internal_atan(__m128d a) {
   __m128d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       ((double*)&out)[i] = vdt::fast_atan(((double*)&a)[i]);
   }
   return out;
}
#endif

#ifdef FASTOR_AVX_IMPL
inline __m256 internal_exp(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       ((float*)&out)[i] = vdt::fast_expf(((float*)&a)[i]);
   }
   return out;
}
inline __m256 internal_log(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       ((float*)&out)[i] = vdt::fast_logf(((float*)&a)[i]);
   }
   return out;
}
// This can give inaccurate results
//inline __m256 internal_pow(__m256 a, __m256 b) {
//   __m256 out;
//   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
//       out[i] = vdt::fast_expf(a[i]*vdt::fast_logf(b[i]));
//   }
//   return out;
//}
inline __m256 internal_sin(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       ((float*)&out)[i] = vdt::fast_sinf(((float*)&a)[i]);
   }
   return out;
}
inline __m256 internal_cos(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       ((float*)&out)[i] = vdt::fast_cosf(((float*)&a)[i]);
   }
   return out;
}
inline __m256 internal_tan(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       ((float*)&out)[i] = vdt::fast_tanf(((float*)&a)[i]);
   }
   return out;
}
inline __m256 internal_asin(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       ((float*)&out)[i] = vdt::fast_asinf(((float*)&a)[i]);
   }
   return out;
}
inline __m256 internal_acos(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       ((float*)&out)[i] = vdt::fast_acosf(((float*)&a)[i]);
   }
   return out;
}
inline __m256 internal_atan(__m256 a) {
   __m256 out;
   for (FASTOR_INDEX i=0; i<stride_finder<float>::Stride; i++) {
       ((float*)&out)[i] = vdt::fast_atanf(((float*)&a)[i]);
   }
   return out;
}


inline __m256d internal_exp(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       ((double*)&out)[i] = vdt::fast_exp(((double*)&a)[i]);
   }
   return out;
}
__m256d internal_log(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       ((double*)&out)[i] = vdt::fast_log(((double*)&a)[i]);
   }
   return out;
}
// This can give inaccurate results
//inline __m256d internal_pow(__m256d a, __m256d b) {
//   __m256d out;
//   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
//       out[i] = vdt::fast_exp(a[i]*vdt::fast_log(b[i]));
//   }
//   return out;
//}
inline __m256d internal_sin(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       ((double*)&out)[i] = vdt::fast_sin(((double*)&a)[i]);
   }
   return out;
}
inline __m256d internal_cos(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       ((double*)&out)[i] = vdt::fast_cos(((double*)&a)[i]);
   }
   return out;
}
inline __m256d internal_tan(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       ((double*)&out)[i] = vdt::fast_tan(((double*)&a)[i]);
   }
   return out;
}
inline __m256d internal_asin(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       ((double*)&out)[i] = vdt::fast_asin(((double*)&a)[i]);
   }
   return out;
}
inline __m256d internal_acos(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       ((double*)&out)[i] = vdt::fast_acos(((double*)&a)[i]);
   }
   return out;
}
inline __m256d internal_atan(__m256d a) {
   __m256d out;
   for (FASTOR_INDEX i=0; i<stride_finder<double>::Stride; i++) {
       ((double*)&out)[i] = vdt::fast_atan(((double*)&a)[i]);
   }
   return out;
}
#endif

#else

// SHUT GCC6 -Wignored-attributes WARNINGS
#ifdef __GNUC__
#if __GNUC__==6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#endif


template<typename T>
inline T internal_exp(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::exp(a[i]);
   }
   return out;
}

template<typename T>
inline T internal_log(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::log(a[i]);
   }
   return out;
}

template<typename T>
inline T internal_sin(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::sin(a[i]);
   }
   return out;
}

template<typename T>
inline T internal_cos(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::cos(a[i]);
   }
   return out;
}

template<typename T>
inline T internal_tan(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::tan(a[i]);
   }
   return out;
}

template<typename T>
inline T internal_asin(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::asin(a[i]);
   }
   return out;
}

template<typename T>
inline T internal_acos(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::acos(a[i]);
   }
   return out;
}

template<typename T>
inline T internal_atan(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::atan(a[i]);
   }
   return out;
}

#ifdef FASTOR_SSE4_2_IMPL
template<>
inline __m128 internal_exp(__m128 a) {
   __m128 out;

   for (FASTOR_INDEX i=0; i<stride_finder<float>::value; i++) {
       ((float*)&out)[i] = std::exp(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m128 internal_log(__m128 a) {
   __m128 out;

   for (FASTOR_INDEX i=0; i<stride_finder<float>::value; i++) {
       ((float*)&out)[i] = std::log(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m128 internal_sin(__m128 a) {
   __m128 out;

   for (FASTOR_INDEX i=0; i<stride_finder<float>::value; i++) {
       ((float*)&out)[i] = std::sin(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m128 internal_cos(__m128 a) {
   __m128 out;

   for (FASTOR_INDEX i=0; i<stride_finder<float>::value; i++) {
       ((float*)&out)[i] = std::cos(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m128 internal_tan(__m128 a) {
   __m128 out;

   for (FASTOR_INDEX i=0; i<stride_finder<float>::value; i++) {
       ((float*)&out)[i] = std::tan(((float*)&a)[i]);
   }
   return out;
}
inline __m128 internal_asin(__m128 a) {
   __m128 out;

   for (FASTOR_INDEX i=0; i<stride_finder<float>::value; i++) {
       ((float*)&out)[i] = std::asin(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m128 internal_acos(__m128 a) {
   __m128 out;

   for (FASTOR_INDEX i=0; i<stride_finder<float>::value; i++) {
       ((float*)&out)[i] = std::acos(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m128 internal_atan(__m128 a) {
   __m128 out;

   for (FASTOR_INDEX i=0; i<stride_finder<float>::value; i++) {
       ((float*)&out)[i] = std::atan(((float*)&a)[i]);
   }
   return out;
}


template<>
inline __m128d internal_exp(__m128d a) {
   __m128d out;

   for (FASTOR_INDEX i=0; i<stride_finder<double>::value; i++) {
       ((double*)&out)[i] = std::exp(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m128d internal_log(__m128d a) {
   __m128d out;

   for (FASTOR_INDEX i=0; i<stride_finder<double>::value; i++) {
       ((double*)&out)[i] = std::log(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m128d internal_sin(__m128d a) {
   __m128d out;

   for (FASTOR_INDEX i=0; i<stride_finder<double>::value; i++) {
       ((double*)&out)[i] = std::sin(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m128d internal_cos(__m128d a) {
   __m128d out;

   for (FASTOR_INDEX i=0; i<stride_finder<double>::value; i++) {
       ((double*)&out)[i] = std::cos(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m128d internal_tan(__m128d a) {
   __m128d out;

   for (FASTOR_INDEX i=0; i<stride_finder<double>::value; i++) {
       ((double*)&out)[i] = std::tan(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m128d internal_asin(__m128d a) {
   __m128d out;

   for (FASTOR_INDEX i=0; i<stride_finder<double>::value; i++) {
       ((double*)&out)[i] = std::asin(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m128d internal_acos(__m128d a) {
   __m128d out;

   for (FASTOR_INDEX i=0; i<stride_finder<double>::value; i++) {
       ((double*)&out)[i] = std::acos(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m128d internal_atan(__m128d a) {
   __m128d out;

   for (FASTOR_INDEX i=0; i<stride_finder<double>::value; i++) {
       ((double*)&out)[i] = std::atan(((double*)&a)[i]);
   }
   return out;
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
inline __m256 internal_exp(__m256 a) {
   __m256 out;

   for (FASTOR_INDEX i=0; i<stride_finder<float>::value; i++) {
       ((float*)&out)[i] = std::exp(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m256 internal_log(__m256 a) {
   __m256 out;

   for (FASTOR_INDEX i=0; i<stride_finder<float>::value; i++) {
       ((float*)&out)[i] = std::log(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m256 internal_sin(__m256 a) {
   __m256 out;

   for (FASTOR_INDEX i=0; i<stride_finder<float>::value; i++) {
       ((float*)&out)[i] = std::sin(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m256 internal_cos(__m256 a) {
   __m256 out;

   for (FASTOR_INDEX i=0; i<stride_finder<float>::value; i++) {
       ((float*)&out)[i] = std::cos(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m256 internal_tan(__m256 a) {
   __m256 out;

   for (FASTOR_INDEX i=0; i<stride_finder<float>::value; i++) {
       ((float*)&out)[i] = std::tan(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m256 internal_asin(__m256 a) {
   __m256 out;

   for (FASTOR_INDEX i=0; i<stride_finder<float>::value; i++) {
       ((float*)&out)[i] = std::asin(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m256 internal_acos(__m256 a) {
   __m256 out;

   for (FASTOR_INDEX i=0; i<stride_finder<float>::value; i++) {
       ((float*)&out)[i] = std::acos(((float*)&a)[i]);
   }
   return out;
}
template<>
inline __m256 internal_atan(__m256 a) {
   __m256 out;

   for (FASTOR_INDEX i=0; i<stride_finder<float>::value; i++) {
       ((float*)&out)[i] = std::atan(((float*)&a)[i]);
   }
   return out;
}


template<>
inline __m256d internal_exp(__m256d a) {
   __m256d out;

   for (FASTOR_INDEX i=0; i<stride_finder<double>::value; i++) {
       ((double*)&out)[i] = std::exp(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m256d internal_log(__m256d a) {
   __m256d out;

   for (FASTOR_INDEX i=0; i<stride_finder<double>::value; i++) {
       ((double*)&out)[i] = std::log(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m256d internal_sin(__m256d a) {
   __m256d out;

   for (FASTOR_INDEX i=0; i<stride_finder<double>::value; i++) {
       ((double*)&out)[i] = std::sin(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m256d internal_cos(__m256d a) {
   __m256d out;

   for (FASTOR_INDEX i=0; i<stride_finder<double>::value; i++) {
       ((double*)&out)[i] = std::cos(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m256d internal_tan(__m256d a) {
   __m256d out;

   for (FASTOR_INDEX i=0; i<stride_finder<double>::value; i++) {
       ((double*)&out)[i] = std::tan(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m256d internal_asin(__m256d a) {
   __m256d out;

   for (FASTOR_INDEX i=0; i<stride_finder<double>::value; i++) {
       ((double*)&out)[i] = std::asin(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m256d internal_acos(__m256d a) {
   __m256d out;

   for (FASTOR_INDEX i=0; i<stride_finder<double>::value; i++) {
       ((double*)&out)[i] = std::acos(((double*)&a)[i]);
   }
   return out;
}
template<>
inline __m256d internal_atan(__m256d a) {
   __m256d out;

   for (FASTOR_INDEX i=0; i<stride_finder<double>::value; i++) {
       ((double*)&out)[i] = std::atan(((double*)&a)[i]);
   }
   return out;
}
#endif
#endif


// not available in vdt
template<typename T, typename U>
inline T internal_pow(T a, U b) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::pow(a[i],b[i]);
   }
   return out;
}

template<typename T>
inline T internal_sinh(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::sinh(a[i]);
   }
   return out;
}

template<typename T>
inline T internal_cosh(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::cosh(a[i]);
   }
   return out;
}

template<typename T>
inline T internal_tanh(T a) {
   T out;
   for (FASTOR_INDEX i=0; i<stride_finder<typename std::remove_reference<decltype(a[0])>::type>::value; i++) {
       out[i] = std::tanh(a[i]);
   }
   return out;
}









// specialisation for doubles - necessary for SIMDVector<T,32>
template<>
inline float internal_exp(float a) {
  return std::exp(a);
}
template<>
inline float internal_log(float a) {
  return std::log(a);
}
template<>
inline float internal_sin(float a) {
  return std::sin(a);
}
template<>
inline float internal_cos(float a) {
  return std::cos(a);
}
template<>
inline float internal_tan(float a) {
  return std::tan(a);
}
template<>
inline float internal_asin(float a) {
  return std::asin(a);
}
template<>
inline float internal_acos(float a) {
  return std::acos(a);
}
template<>
inline float internal_atan(float a) {
  return std::atan(a);
}
template<>
inline float internal_sinh(float a) {
  return std::sinh(a);
}
template<>
inline float internal_cosh(float a) {
  return std::cosh(a);
}
template<>
inline float internal_tanh(float a) {
  return std::tanh(a);
}
template<>
inline float internal_pow(float a, float b) {
  return std::pow(a,b);
}
template<>
inline float internal_pow(float a, double b) {
  return std::pow(a,b);
}
template<>
inline float internal_pow(float a, int b) {
  return std::pow(a,b);
}



// specialisation for doubles - necessary for SIMDVector<T,64>
template<>
inline double internal_exp(double a) {
  return std::exp(a);
}
template<>
inline double internal_log(double a) {
  return std::log(a);
}
template<>
inline double internal_sin(double a) {
  return std::sin(a);
}
template<>
inline double internal_cos(double a) {
  return std::cos(a);
}
template<>
inline double internal_tan(double a) {
  return std::tan(a);
}
template<>
inline double internal_asin(double a) {
  return std::asin(a);
}
template<>
inline double internal_acos(double a) {
  return std::acos(a);
}
template<>
inline double internal_atan(double a) {
  return std::atan(a);
}
template<>
inline double internal_sinh(double a) {
  return std::sinh(a);
}
template<>
inline double internal_cosh(double a) {
  return std::cosh(a);
}
template<>
inline double internal_tanh(double a) {
  return std::tanh(a);
}
template<>
inline double internal_pow(double a, double b) {
  return std::pow(a,b);
}
template<>
inline double internal_pow(double a, float b) {
  return std::pow(a,b);
}
template<>
inline double internal_pow(double a, int b) {
  return std::pow(a,b);
}



}

#endif // INTERNAL_MATH_H

