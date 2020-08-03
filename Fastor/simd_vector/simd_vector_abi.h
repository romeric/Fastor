#ifndef FASTOR_SIMD_VECTOR_ABI
#define FASTOR_SIMD_VECTOR_ABI

#include "Fastor/meta/meta.h"
#include "Fastor/config/config.h"
#include <complex>
#include <type_traits>

namespace Fastor {

namespace simd_abi {

struct scalar {};
struct sse {};
struct avx {};
struct avx512 {};
struct mic {};
template<size_t N> struct fixed_size {};

#ifndef FASTOR_DONT_VECTORISE
#if defined(FASTOR_AVX512_IMPL)
using native = simd_abi::avx512;
#elif defined(FASTOR_AVX_IMPL)
using native = simd_abi::avx;
#elif defined(FASTOR_SSE2_IMPL)
using native = simd_abi::sse;
#else
using native = simd_abi::scalar;
#endif
#else
using native = simd_abi::scalar;
#endif

}

// Definition of DEFAULT_ABI is here
//--------------------------------------------------------------------------------------------------------------//
#define DEFAULT_ABI simd_abi::native
//--------------------------------------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------------------------------------//
namespace internal {

template<class __svec>
struct get_simd_vector_size;
template<template<typename, typename> class __svec, typename T, typename ABI>
struct get_simd_vector_size<__svec<T,ABI>> {
    static constexpr size_t bitsize = std::is_same<ABI,simd_abi::avx512>::value
                                                ? FASTOR_AVX512_BITSIZE : (std::is_same<ABI,simd_abi::avx>::value
                                                    ? FASTOR_AVX_BITSIZE : (std::is_same<ABI,simd_abi::sse>::value
                                                        ? FASTOR_SSE_BITSIZE : sizeof(T)*8));

    // Size should be at least 1UL
    static constexpr size_t value = (bitsize / sizeof(T) / 8UL) != 0 ? (bitsize / sizeof(T) / 8UL) : 1UL;
};
// Specialisation for fixed size simd vectors
template<template<typename, typename> class __svec, typename T, size_t N>
struct get_simd_vector_size<__svec<T,simd_abi::fixed_size<N> > > {
    static constexpr size_t bitsize = N*8UL;
    static constexpr size_t value = N;
};
// Specialisation for complex simd vectors
template<template<typename, typename> class __svec, typename ABI>
struct get_simd_vector_size<__svec<std::complex<float>,ABI>> {
    using T = float;
    static constexpr size_t bitsize = std::is_same<ABI,simd_abi::avx512>::value
                                                ? FASTOR_AVX512_BITSIZE : (std::is_same<ABI,simd_abi::avx>::value
                                                    ? FASTOR_AVX_BITSIZE : (std::is_same<ABI,simd_abi::sse>::value
                                                        ? FASTOR_SSE_BITSIZE : sizeof(T)*8));

    // Size should be at least 1UL
    static constexpr size_t value = (bitsize / sizeof(T) / 8UL) != 0 ? (bitsize / sizeof(T) / 8UL) : 1UL;
};
template<template<typename, typename> class __svec, typename ABI>
struct get_simd_vector_size<__svec<std::complex<double>,ABI>> {
    using T = double;
    static constexpr size_t bitsize = std::is_same<ABI,simd_abi::avx512>::value
                                                ? FASTOR_AVX512_BITSIZE : (std::is_same<ABI,simd_abi::avx>::value
                                                    ? FASTOR_AVX_BITSIZE : (std::is_same<ABI,simd_abi::sse>::value
                                                        ? FASTOR_SSE_BITSIZE : sizeof(T)*8));

    // Size should be at least 1UL
    static constexpr size_t value = (bitsize / sizeof(T) / 8UL) != 0 ? (bitsize / sizeof(T) / 8UL) : 1UL;
};


template<class __svec>
struct get_half_simd_type;
template<template<typename, typename> class __svec, typename T, typename ABI>
struct get_half_simd_type<__svec<T,ABI>> {
    // If not a half of simd we give back the actual incoming type
    using actual_type    = __svec<T,ABI>;
    using type = typename std::conditional< std::is_same<ABI,simd_abi::avx512>::value, __svec<T,simd_abi::avx>,
                        typename std::conditional< std::is_same<ABI,simd_abi::avx>::value, __svec<T,simd_abi::sse>, actual_type>::type
                      >::type;
};


template<class __svec>
struct get_quarter_simd_type;
template<template<typename, typename> class __svec, typename T, typename ABI>
struct get_quarter_simd_type<__svec<T,ABI>> {
    // If not a quarter of simd we give back the actual incoming type
    using actual_type    = __svec<T,ABI>;
    using type = typename std::conditional< std::is_same<ABI,simd_abi::avx512>::value, __svec<T,simd_abi::sse>, actual_type>::type;
};


template<class __svec, size_t N>
struct is_exact_multiple_of_smaller_simd;
template<template<typename, typename> class __svec, typename T, typename ABI, size_t N>
struct is_exact_multiple_of_smaller_simd<__svec<T,ABI>,N> {
    // If not a fraction of simd we give back the actual incoming type
    using actual_type    = __svec<T,ABI>;
    // if N is half simd which=2, if it is a 1/4th which=4, else which=1
    static constexpr int which = get_simd_vector_size<__svec<T,ABI>>::value / N == 2UL ? 2UL
                                 : (get_simd_vector_size<__svec<T,ABI>>::value / N == 4UL ? 4UL : 1UL);
    static constexpr bool value = which != 1UL && !std::is_same<ABI,simd_abi::sse>::value ? true : false;

    static constexpr bool is_half_of_avx512 = std::is_same<ABI,simd_abi::avx512>::value && which==2UL;
    static constexpr bool is_half_of_avx = std::is_same<ABI,simd_abi::avx>::value && which==2UL;
    static constexpr bool is_4th_of_avx512 = std::is_same<ABI,simd_abi::avx512>::value && which==4UL;
    // static constexpr bool is_4th_of_avx = std::is_same<ABI,simd_abi::avx>::value && which==4UL;

    static constexpr bool is_half_type = is_half_of_avx512 || is_half_of_avx;
    static constexpr bool is_quarter_type = is_4th_of_avx512;

    using half_type = typename std::conditional< is_half_of_avx512, __svec<T,simd_abi::avx>,
                        typename std::conditional< is_half_of_avx, __svec<T,simd_abi::sse>, actual_type>::type
                      >::type;
    using quarter_type = typename std::conditional< is_4th_of_avx512, __svec<T,simd_abi::sse>, actual_type>::type;

    using type = typename std::conditional<is_half_type, half_type,
                    typename std::conditional<is_quarter_type, quarter_type, actual_type>::type
                 >::type;
};

template<class __svec, size_t N>
struct choose_best_simd_type;
template<template<typename, typename> class __svec, typename TT, typename ABI, size_t N>
struct choose_best_simd_type<__svec<TT,ABI>,N> {
    using T = remove_cv_ref_t<TT>;
    using actual_type = __svec<T,ABI>;
    static constexpr size_t _vec_size = get_simd_vector_size<__svec<T,ABI>>::value;
    // For exact fractions simd gets proper speed up for instance for matmul
    using exact_multiple_t = typename is_exact_multiple_of_smaller_simd<__svec<T,ABI>,N>::type;
    static constexpr bool is_exact_multiple = is_exact_multiple_of_smaller_simd<__svec<T,ABI>,N>::value;

#if defined(FASTOR_AVX2_IMPL) || defined(FASTOR_HAS_AVX512_MASKS)
    using size_based_type = typename std::conditional<is_exact_multiple, exact_multiple_t, actual_type>::type;
#else
    using size_based_type = typename std::conditional<is_exact_multiple, exact_multiple_t,
                        typename std::conditional<is_less<N,_vec_size>::value, typename get_half_simd_type<__svec<T,ABI>>::type, actual_type
                        >::type
                     >::type;
#endif

    // using size_based_type = typename std::conditional<is_exact_multiple, typename is_exact_multiple_of_smaller_simd<__svec<T,ABI>,N>::type,
    //                 typename std::conditional<is_greater<_vec_size,2UL*N>::value, typename get_quarter_simd_type<__svec<T,ABI>>::type,
    //                     typename std::conditional<is_greater<_vec_size,N>::value, typename get_half_simd_type<__svec<T,ABI>>::type, actual_type
    //                     >::type
    //                 >::type
    //              >::type;

    // // For other fractions masking might be a better idea than, hence this special logic for remainder using is_less.
    // // For no special logic use the above case
    // using size_based_type = typename std::conditional<is_exact_multiple, typename is_exact_multiple_of_smaller_simd<__svec<T,ABI>,N>::type,
    //                 typename std::conditional<is_greater<_vec_size,2UL*N>::value && is_greater<_vec_size % N,1UL>::value,
    //                     typename get_quarter_simd_type<__svec<T,ABI>>::type,
    //                     typename std::conditional<is_greater<_vec_size,N>::value && is_greater<_vec_size % N,1UL>::value,
    //                         typename get_half_simd_type<__svec<T,ABI>>::type, actual_type
    //                     >::type
    //                 >::type
    //              >::type;

    using type = typename std::conditional< std::is_same<T,float>::value                    ||
                                            std::is_same<T,double>::value                   ||
                                            std::is_same<T,std::complex<float>>::value      ||
                                            std::is_same<T,std::complex<double>>::value     ||
                                            std::is_same<T,int32_t>::value                  ||
                                            std::is_same<T,int64_t>::value,
                                            size_based_type,
                                            __svec<T,simd_abi::scalar>
                >::type;
};

} // end of namesapce internal


// This is specifically for matmul and other backend tensor kernels that need
// to choose/switch between best simd types for utmost performance
template<class __svec, size_t N>
using choose_best_simd_t = typename internal::choose_best_simd_type<__svec,N>::type;
//--------------------------------------------------------------------------------------------------------------//


// For switching between complex and non-complex implementation of simd we need different return types
//--------------------------------------------------------------------------------------------------------------//
template<typename T, typename T2 = void>
struct get_simd_cmplx_value_type;
template<typename T>
struct get_simd_cmplx_value_type<T, enable_if_t_<!is_complex_v_<T> > > {
    using type = T;
};
template<typename T>
struct get_simd_cmplx_value_type<T, enable_if_t_<is_complex_v_<T> > > {
    using type = typename T::value_type;
};

template<class VecType>
struct simd_cmplx_value_type;
template<template<typename, typename> class __svec, typename T, typename ABI>
struct simd_cmplx_value_type< __svec<T,ABI> > {
    using type = typename get_simd_cmplx_value_type<T>::type;
};
template<class VecType>
using simd_cmplx_value_t = typename simd_cmplx_value_type<VecType>::type;
//--------------------------------------------------------------------------------------------------------------//



} // end of namesapce Fastor

#endif // FASTOR_SIMD_VECTOR_ABI
