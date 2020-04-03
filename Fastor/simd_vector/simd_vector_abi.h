#ifndef FASTOR_SIMD_VECTOR_ABI
#define FASTOR_SIMD_VECTOR_ABI

#include "Fastor/commons/commons.h"
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

#if defined(FASTOR_AVX512_IMPL)
using native = simd_abi::avx512;
#elif defined(FASTOR_AVX_IMPL)
using native = simd_abi::avx;
#elif defined(FASTOR_SSE_IMPL)
using native = simd_abi::sse;
#else
using native = simd_abi::scalar;
#endif

}

// Definition of DEFAULT_ABI is here
//--------------------------------------------------------------------------------------------------------------//
#define DEFAULT_ABI simd_abi::native
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

    static constexpr size_t value = bitsize / sizeof(T) / 8UL;
};
template<template<typename, typename> class __svec, typename T, size_t N>
struct get_simd_vector_size<__svec<T,simd_abi::fixed_size<N> > > {
    static constexpr size_t bitsize = N*8UL;
    static constexpr size_t value = N;
};

} // end of namesapce internal

} // end of namesapce Fastor

#endif // FASTOR_SIMD_VECTOR_ABI
