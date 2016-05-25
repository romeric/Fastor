#ifndef SIMD_VECTOR_INT_H
#define SIMD_VECTOR_INT_H

#include "simd_vector_base.h"

namespace Fastor {

//! SIMDVector<int> is a wrapper over __m128i, as most integer arithmatics
//! are only possible with AVX2

template <>
struct SIMDVector<int> {
    static constexpr FASTOR_INDEX Size = get_vector_size<int>::size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return get_vector_size<int>::size;}
    static constexpr int unroll_size(FASTOR_INDEX size) {return (static_cast<int>(size) - static_cast<int>(Size));}

    FASTOR_INLINE SIMDVector() : value(_mm_setzero_si128()) {}
    FASTOR_INLINE SIMDVector(int num) : value(_mm_set1_epi32(num)) {}
    FASTOR_INLINE SIMDVector(__m128i regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const SIMDVector<int> &a) : value(a.value) {}
    FASTOR_INLINE SIMDVector(const int *data) : value(_mm_load_si128((__m128i*)data)) {}
    FASTOR_INLINE SIMDVector(int *data) : value(_mm_load_si128((__m128i*)data)) {}

    FASTOR_INLINE SIMDVector<int> operator=(int num) {
        value = _mm_set1_epi32(num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<int> operator=(__m128i regi) {
        value = regi;
        return *this;
    }
    FASTOR_INLINE SIMDVector<int> operator=(const SIMDVector<int> &a) {
        value = a.value;
        return *this;
    }

    FASTOR_INLINE void load(const int *data, bool Aligned=true) {
        if (Aligned)
            value =_mm_load_si128((__m128i*)data);
        else
            value =_mm_loadu_si128((__m128i*)data);
    }
    FASTOR_INLINE void store(int *data, bool Aligned=true) {
        if (Aligned)
            _mm_store_si128((__m128i*)data,value);
        else
            _mm_storeu_si128((__m128i*)data,value);
    }

    FASTOR_INLINE int operator[](FASTOR_INDEX i) {return value[i];}
    FASTOR_INLINE int operator()(FASTOR_INDEX i) {return value[i];}

    FASTOR_INLINE void set(int num0, int num1, int num2, int num3) {
        value = _mm_setr_epi32(num0,num1,num2,num3);
    }
    FASTOR_INLINE void set_sequential(int num0) {
        value = _mm_setr_epi32(num0,num0+1,num0+2,num0+3);
    }

    // In-place operators
    FASTOR_INLINE void operator+=(int num) {
        value = _mm_add_epi32(value,_mm_set1_epi32(num));
    }
    FASTOR_INLINE void operator+=(__m128i regi) {
        value = _mm_add_epi32(value,regi);
    }
    FASTOR_INLINE void operator+=(const SIMDVector<int> &a) {
        value = _mm_add_epi32(value,a.value);
    }

    FASTOR_INLINE void operator-=(int num) {
        value = _mm_sub_epi32(value,_mm_set1_epi32(num));
    }
    FASTOR_INLINE void operator-=(__m128i regi) {
        value = _mm_sub_epi32(value,regi);
    }
    FASTOR_INLINE void operator-=(const SIMDVector<int> &a) {
        value = _mm_sub_epi32(value,a.value);
    }

    FASTOR_INLINE void operator*=(int num) {
        value = _mm_mul_epi32(value,_mm_set1_epi32(num));
    }
    FASTOR_INLINE void operator*=(__m128i regi) {
        value = _mm_mul_epi32(value,regi);
    }
    FASTOR_INLINE void operator*=(const SIMDVector<int> &a) {
        value = _mm_mul_epi32(value,a.value);
    }
    // end of in-place operators

    __m128i value;
};

}

#endif // SIMD_VECTOR_INT_H

