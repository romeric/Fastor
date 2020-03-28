#ifndef SIMD_VECTOR_INT_H
#define SIMD_VECTOR_INT_H

#include "simd_vector_base.h"

namespace Fastor {


// AVX VERSION
//-----------------------------------------------------------------------------------------------

#ifdef FASTOR_AVX2_IMPL

template<>
struct SIMDVector<int,simd_abi::avx> {
    using value_type = __m256i;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<int,simd_abi::avx>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<int,simd_abi::avx>>::value;}
    static constexpr int unroll_size(FASTOR_INDEX size) {return (static_cast<int>(size) - static_cast<int>(Size));}

    FASTOR_INLINE SIMDVector() : value(_mm256_setzero_si256()) {}
    FASTOR_INLINE SIMDVector(int num) : value(_mm256_set1_epi32(num)) {}
    FASTOR_INLINE SIMDVector(__m256i regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const SIMDVector<int,simd_abi::avx> &a) : value(a.value) {}
    FASTOR_INLINE SIMDVector(const int *data, bool Aligned=true) {
        if (Aligned)
            value =_mm256_load_si256((__m256i*)data);
        else
            value = _mm256_loadu_si256((__m256i*)data);
    }
    FASTOR_INLINE SIMDVector(int *data, bool Aligned=true) {
        if (Aligned)
            value =_mm256_load_si256((__m256i*)data);
        else
            value = _mm256_loadu_si256((__m256i*)data);
    }

    FASTOR_INLINE SIMDVector<int,simd_abi::avx> operator=(int num) {
        value = _mm256_set1_epi32(num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<int,simd_abi::avx> operator=(__m256i regi) {
        value = regi;
        return *this;
    }
    FASTOR_INLINE SIMDVector<int,simd_abi::avx> operator=(const SIMDVector<int,simd_abi::avx> &a) {
        value = a.value;
        return *this;
    }

    FASTOR_INLINE void load(const int *data, bool Aligned=true) {
        if (Aligned)
            value =_mm256_load_si256((__m256i*)data);
        else
            value = _mm256_loadu_si256((__m256i*)data);
    }
    FASTOR_INLINE void store(int *data, bool Aligned=true) const {
        if (Aligned)
            _mm256_store_si256((__m256i*)data,value);
        else
            _mm256_storeu_si256((__m256i*)data,value);
    }

    FASTOR_INLINE void aligned_load(const int *data) {
        value =_mm256_load_si256((__m256i*)data);
    }
    FASTOR_INLINE void aligned_store(int *data) const {
        _mm256_store_si256((__m256i*)data,value);
    }

    FASTOR_INLINE int operator[](FASTOR_INDEX i) const {return reinterpret_cast<const int*>(&value)[i];}
    FASTOR_INLINE int operator()(FASTOR_INDEX i) const {return reinterpret_cast<const int*>(&value)[i];}

    FASTOR_INLINE void set(int num) {
        value = _mm256_set1_epi32(num);
    }
    FASTOR_INLINE void set(int num0, int num1, int num2, int num3, int num4, int num5, int num6, int num7) {
        value = _mm256_set_epi32(num0,num1,num2,num3,num4,num5,num6,num7);
    }
    FASTOR_INLINE void set_sequential(int num0) {
        value = _mm256_setr_epi32(num0,num0+1,num0+2,num0+3,num0+4,num0+5,num0+6,num0+7);
    }

    // In-place operators
    FASTOR_INLINE void operator+=(int num) {
        value = _mm256_add_epi32x(value,_mm256_set1_epi32(num));

    }
    FASTOR_INLINE void operator+=(__m256i regi) {
        value = _mm256_add_epi32x(value,regi);
    }
    FASTOR_INLINE void operator+=(const SIMDVector<int,simd_abi::avx> &a) {
        value = _mm256_add_epi32x(value,a.value);
    }

    FASTOR_INLINE void operator-=(int num) {
        value = _mm256_sub_epi32x(value,_mm256_set1_epi32(num));
    }
    FASTOR_INLINE void operator-=(__m256i regi) {
        value = _mm256_sub_epi32x(value,regi);
    }
    FASTOR_INLINE void operator-=(const SIMDVector<int,simd_abi::avx> &a) {
        value = _mm256_sub_epi32x(value,a.value);
    }

    FASTOR_INLINE void operator*=(int num) {
        value = _mm256_mul_epi32x(value,_mm256_set1_epi32(num));
    }
    FASTOR_INLINE void operator*=(__m256i regi) {
        value = _mm256_mul_epi32x(value,regi);
    }
    FASTOR_INLINE void operator*=(const SIMDVector<int,simd_abi::avx> &a) {
        value = _mm256_mul_epi32x(value,a.value);
    }

    FASTOR_INLINE void operator/=(int num) {
        int val[Size]; _mm256_storeu_si256((__m256i*)val, value);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= num;
        }
        value = _mm256_loadu_si256((__m256i*)val);
    }
    FASTOR_INLINE void operator/=(__m256i regi) {
        int val[Size]; _mm256_storeu_si256((__m256i*)val, value);
        int val_num[Size]; _mm256_storeu_si256((__m256i*)val_num, regi);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= val_num[i];
        }
        value = _mm256_loadu_si256((__m256i*)val);
    }
    FASTOR_INLINE void operator/=(const SIMDVector<int,simd_abi::avx> &a) {
        int val[Size]; _mm256_storeu_si256((__m256i*)val, value);
        int val_a[Size]; _mm256_storeu_si256((__m256i*)val_a, a.value);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= val_a[i];
        }
        value = _mm256_loadu_si256((__m256i*)val);
    }

    FASTOR_INLINE int minimum() {
        int *vals = (int*)&value;
        int quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            if (vals[i]<quan)
                quan = vals[i];
        return quan;
    }
    FASTOR_INLINE int maximum() {
        int *vals = (int*)&value;
        int quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            if (vals[i]>quan)
                quan = vals[i];
        return quan;
    }
    FASTOR_INLINE SIMDVector<int,simd_abi::avx> reverse() {
        SIMDVector<int,simd_abi::avx> out;
        out.value = _mm256_reverse_epi32(value);
        return out;
    }

    FASTOR_INLINE int sum() {
        int vals[Size]; _mm256_storeu_si256((__m256i*)vals, value);
        int quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            quan += vals[i];
        return quan;
    }

    FASTOR_INLINE int dot(const SIMDVector<int,simd_abi::avx> &other) {
        int vals0[Size]; _mm256_storeu_si256((__m256i*)vals0, value);
        int vals1[Size]; _mm256_storeu_si256((__m256i*)vals1, other.value);
        int quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            quan += vals0[i]*vals1[i];
        return quan;
    }

    __m256i value;
};

FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<int,simd_abi::avx> a) {
    const int *value = (int*) &a.value;
    os << "[" << value[0] <<  " " << value[1] << " " << value[2] << " " << value[3]
       << " " << value[4] <<  " " << value[5] << " " << value[6] << " " << value[7] << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<int,simd_abi::avx> operator+(const SIMDVector<int,simd_abi::avx> &a, const SIMDVector<int,simd_abi::avx> &b) {
    SIMDVector<int,simd_abi::avx> out;
    out.value = _mm256_add_epi32x(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::avx> operator+(const SIMDVector<int,simd_abi::avx> &a, int b) {
    SIMDVector<int,simd_abi::avx> out;
    out.value = _mm256_add_epi32x(a.value,_mm256_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::avx> operator+(int a, const SIMDVector<int,simd_abi::avx> &b) {
    SIMDVector<int,simd_abi::avx> out;
    out.value = _mm256_add_epi32x(_mm256_set1_epi32(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::avx> operator+(const SIMDVector<int,simd_abi::avx> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<int,simd_abi::avx> operator-(const SIMDVector<int,simd_abi::avx> &a, const SIMDVector<int,simd_abi::avx> &b) {
    SIMDVector<int,simd_abi::avx> out;
    out.value = _mm256_sub_epi32x(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::avx> operator-(const SIMDVector<int,simd_abi::avx> &a, int b) {
    SIMDVector<int,simd_abi::avx> out;
    out.value = _mm256_sub_epi32x(a.value,_mm256_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::avx> operator-(int a, const SIMDVector<int,simd_abi::avx> &b) {
    SIMDVector<int,simd_abi::avx> out;
    out.value = _mm256_sub_epi32x(_mm256_set1_epi32(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::avx> operator-(const SIMDVector<int,simd_abi::avx> &b) {
    SIMDVector<int,simd_abi::avx> out;
    out.value = _mm256_castps_si256(_mm256_neg_ps(_mm256_castsi256_ps(b.value)));
    return out;
}

FASTOR_INLINE SIMDVector<int,simd_abi::avx> operator*(const SIMDVector<int,simd_abi::avx> &a, const SIMDVector<int,simd_abi::avx> &b) {
    SIMDVector<int,simd_abi::avx> out;
    out.value = _mm256_mul_epi32x(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::avx> operator*(const SIMDVector<int,simd_abi::avx> &a, int b) {
    SIMDVector<int,simd_abi::avx> out;
    out.value = _mm256_mul_epi32x(a.value,_mm256_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::avx> operator*(int a, const SIMDVector<int,simd_abi::avx> &b) {
    SIMDVector<int,simd_abi::avx> out;
    out.value = _mm256_mul_epi32x(_mm256_set1_epi32(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<int,simd_abi::avx> operator/(const SIMDVector<int,simd_abi::avx> &a, const SIMDVector<int,simd_abi::avx> &b) {
    SIMDVector<int,simd_abi::avx> out;
    int val[out.size()];   _mm256_storeu_si256((__m256i*)val, out.value);
    int val_a[out.size()]; _mm256_storeu_si256((__m256i*)val_a, a.value);
    int val_b[out.size()]; _mm256_storeu_si256((__m256i*)val_b, b.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = val_a[i] / val_b[i];
    }
    out.value = _mm256_loadu_si256((__m256i*)val);
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::avx> operator/(const SIMDVector<int,simd_abi::avx> &a, int b) {
    SIMDVector<int,simd_abi::avx> out;
    int val[out.size()];   _mm256_storeu_si256((__m256i*)val, out.value);
    int val_a[out.size()]; _mm256_storeu_si256((__m256i*)val_a, a.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = val_a[i] / b;
    }
    out.value = _mm256_loadu_si256((__m256i*)val);
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::avx> operator/(int a, const SIMDVector<int,simd_abi::avx> &b) {
    SIMDVector<int,simd_abi::avx> out;
    int val[out.size()];   _mm256_storeu_si256((__m256i*)val, out.value);
    int val_b[out.size()]; _mm256_storeu_si256((__m256i*)val_b, b.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = a / val_b[i];
    }
    out.value = _mm256_loadu_si256((__m256i*)val);
    return out;
}

FASTOR_INLINE SIMDVector<int,simd_abi::avx> abs(const SIMDVector<int,simd_abi::avx> &a) {
    SIMDVector<int,simd_abi::avx> out;
#ifdef __AVX2__
    out.value = _mm256_abs_epi32(a.value);
#else
    // THIS IS ALSO AVX2 VERSION!
    // __m128i lo = _mm_abs_epi32(_mm256_castsi256_si128(a.value));
    // __m128i hi = _mm_abs_epi32(_mm256_extracti128_si256(a.value,0x1));
    // out.value = _mm256_castsi128_si256(lo);
    // out.value = _mm256_insertf128_si256(out.value,hi,0x1);

    int *value = (int*) &a.value;
    for (int i=0; i<8; ++i) {
        value[i] = std::abs(value[i]);
    }
#endif
    return out;
}


#endif


// SSE VERSION
//-----------------------------------------------------------------------------------------------

#ifdef FASTOR_SSE4_2_IMPL

template<>
struct SIMDVector<int,simd_abi::sse> {
    using value_type = __m128i;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<int,simd_abi::sse>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<int,simd_abi::sse>>::value;}
    static constexpr int unroll_size(FASTOR_INDEX size) {return (static_cast<int>(size) - static_cast<int>(Size));}

    FASTOR_INLINE SIMDVector() : value(_mm_setzero_si128()) {}
    FASTOR_INLINE SIMDVector(int num) : value(_mm_set1_epi32(num)) {}
    FASTOR_INLINE SIMDVector(__m128i regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const SIMDVector<int,simd_abi::sse> &a) : value(a.value) {}
    FASTOR_INLINE SIMDVector(const int *data, bool Aligned=true) {
        if (Aligned)
            value =_mm_load_si128((__m128i*)data);
        else
            value = _mm_loadu_si128((__m128i*)data);
    }
    FASTOR_INLINE SIMDVector(int *data, bool Aligned=true) {
        if (Aligned)
            value =_mm_load_si128((__m128i*)data);
        else
            value = _mm_loadu_si128((__m128i*)data);
    }

    FASTOR_INLINE SIMDVector<int,simd_abi::sse> operator=(int num) {
        value = _mm_set1_epi32(num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<int,simd_abi::sse> operator=(__m128i regi) {
        value = regi;
        return *this;
    }
    FASTOR_INLINE SIMDVector<int,simd_abi::sse> operator=(const SIMDVector<int,simd_abi::sse> &a) {
        value = a.value;
        return *this;
    }

    FASTOR_INLINE void load(const int *data, bool Aligned=true) {
        if (Aligned)
            value =_mm_load_si128((__m128i*)data);
        else
            value = _mm_loadu_si128((__m128i*)data);
    }
    FASTOR_INLINE void store(int *data, bool Aligned=true) const {
        if (Aligned)
            _mm_store_si128((__m128i*)data,value);
        else
            _mm_storeu_si128((__m128i*)data,value);
    }

    FASTOR_INLINE void aligned_load(const int *data) {
        value =_mm_load_si128((__m128i*)data);
    }
    FASTOR_INLINE void aligned_store(int *data) const {
        _mm_store_si128((__m128i*)data,value);
    }

    FASTOR_INLINE int operator[](FASTOR_INDEX i) const {return reinterpret_cast<const int*>(&value)[i];}
    FASTOR_INLINE int operator()(FASTOR_INDEX i) const {return reinterpret_cast<const int*>(&value)[i];}

    FASTOR_INLINE void set(int num) {
        value = _mm_set1_epi32(num);
    }
    FASTOR_INLINE void set(int num0, int num1, int num2, int num3) {
        value = _mm_set_epi32(num0,num1,num2,num3);
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
    FASTOR_INLINE void operator+=(const SIMDVector<int,simd_abi::sse> &a) {
        value = _mm_add_epi32(value,a.value);
    }

    FASTOR_INLINE void operator-=(int num) {
        value = _mm_sub_epi32(value,_mm_set1_epi32(num));
    }
    FASTOR_INLINE void operator-=(__m128i regi) {
        value = _mm_sub_epi32(value,regi);
    }
    FASTOR_INLINE void operator-=(const SIMDVector<int,simd_abi::sse> &a) {
        value = _mm_sub_epi32(value,a.value);
    }

    FASTOR_INLINE void operator*=(int num) {
        value = _mm_mul_epi32(value,_mm_set1_epi32(num));
    }
    FASTOR_INLINE void operator*=(__m128i regi) {
        value = _mm_mul_epi32(value,regi);
    }
    FASTOR_INLINE void operator*=(const SIMDVector<int,simd_abi::sse> &a) {
        value = _mm_mul_epi32(value,a.value);
    }

    FASTOR_INLINE void operator/=(int num) {
        int val[Size]; _mm_storeu_si128((__m128i*)val, value);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= num;
        }
        value = _mm_loadu_si128((__m128i*)val);
    }
    FASTOR_INLINE void operator/=(__m128i regi) {
        int val[Size]; _mm_storeu_si128((__m128i*)val, value);
        int val_num[Size]; _mm_storeu_si128((__m128i*)val_num, regi);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= val_num[i];
        }
        value = _mm_loadu_si128((__m128i*)val);
    }
    FASTOR_INLINE void operator/=(const SIMDVector<int,simd_abi::sse> &a) {
        int val[Size]; _mm_storeu_si128((__m128i*)val, value);
        int val_a[Size]; _mm_storeu_si128((__m128i*)val_a, a.value);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= val_a[i];
        }
        value = _mm_loadu_si128((__m128i*)val);
    }

    FASTOR_INLINE int minimum() {
        int *vals = (int*)&value;
        int quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            if (vals[i]<quan)
                quan = vals[i];
        return quan;
    }
    FASTOR_INLINE int maximum() {
        int *vals = (int*)&value;
        int quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            if (vals[i]>quan)
                quan = vals[i];
        return quan;
    }
    FASTOR_INLINE SIMDVector<int,simd_abi::sse> reverse() {
        SIMDVector<int,simd_abi::sse> out;
        out.value = _mm_reverse_epi32(value);
        return out;
    }

    FASTOR_INLINE int sum() {
        int vals[Size]; _mm_storeu_si128((__m128i*)vals, value);
        int quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            quan += vals[i];
        return quan;
    }

    FASTOR_INLINE int dot(const SIMDVector<int,simd_abi::sse> &other) {
        int *vals0 = (int*)&value;
        int *vals1 = (int*)&other.value;
        int quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            quan += vals0[i]*vals1[i];
        return quan;
    }

    __m128i value;
};

FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<int,simd_abi::sse> a) {
    const int *value = (int*) &a.value;
    os << "[" << value[0] <<  " " << value[1] << " " << value[2] << " " << value[3] << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<int,simd_abi::sse> operator+(const SIMDVector<int,simd_abi::sse> &a, const SIMDVector<int,simd_abi::sse> &b) {
    SIMDVector<int,simd_abi::sse> out;
    out.value = _mm_add_epi32(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::sse> operator+(const SIMDVector<int,simd_abi::sse> &a, int b) {
    SIMDVector<int,simd_abi::sse> out;
    out.value = _mm_add_epi32(a.value,_mm_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::sse> operator+(int a, const SIMDVector<int,simd_abi::sse> &b) {
    SIMDVector<int,simd_abi::sse> out;
    out.value = _mm_add_epi32(_mm_set1_epi32(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::sse> operator+(const SIMDVector<int,simd_abi::sse> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<int,simd_abi::sse> operator-(const SIMDVector<int,simd_abi::sse> &a, const SIMDVector<int,simd_abi::sse> &b) {
    SIMDVector<int,simd_abi::sse> out;
    out.value = _mm_sub_epi32(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::sse> operator-(const SIMDVector<int,simd_abi::sse> &a, int b) {
    SIMDVector<int,simd_abi::sse> out;
    out.value = _mm_sub_epi32(a.value,_mm_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::sse> operator-(int a, const SIMDVector<int,simd_abi::sse> &b) {
    SIMDVector<int,simd_abi::sse> out;
    out.value = _mm_sub_epi32(_mm_set1_epi32(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::sse> operator-(const SIMDVector<int,simd_abi::sse> &b) {
    SIMDVector<int,simd_abi::sse> out;
    out.value = _mm_castps_si128(_mm_neg_ps(_mm_castsi128_ps(b.value)));
    return out;
}

FASTOR_INLINE SIMDVector<int,simd_abi::sse> operator*(const SIMDVector<int,simd_abi::sse> &a, const SIMDVector<int,simd_abi::sse> &b) {
    SIMDVector<int,simd_abi::sse> out;
    out.value = _mm_mul_epi32x(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::sse> operator*(const SIMDVector<int,simd_abi::sse> &a, int b) {
    SIMDVector<int,simd_abi::sse> out;
    out.value = _mm_mul_epi32x(a.value,_mm_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::sse> operator*(int a, const SIMDVector<int,simd_abi::sse> &b) {
    SIMDVector<int,simd_abi::sse> out;
    out.value = _mm_mul_epi32x(_mm_set1_epi32(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<int,simd_abi::sse> operator/(const SIMDVector<int,simd_abi::sse> &a, const SIMDVector<int,simd_abi::sse> &b) {
    SIMDVector<int,simd_abi::sse> out;
    int val[out.size()];   _mm_storeu_si128((__m128i*)val, out.value);
    int val_a[out.size()]; _mm_storeu_si128((__m128i*)val_a, a.value);
    int val_b[out.size()]; _mm_storeu_si128((__m128i*)val_b, b.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = val_a[i] / val_b[i];
    }
    out.value = _mm_loadu_si128((__m128i*)val);
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::sse> operator/(const SIMDVector<int,simd_abi::sse> &a, int b) {
    SIMDVector<int,simd_abi::sse> out;
    int val[out.size()];   _mm_storeu_si128((__m128i*)val, out.value);
    int val_a[out.size()]; _mm_storeu_si128((__m128i*)val_a, a.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = val_a[i] / b;
    }
    out.value = _mm_loadu_si128((__m128i*)val);
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::sse> operator/(int a, const SIMDVector<int,simd_abi::sse> &b) {
    SIMDVector<int,simd_abi::sse> out;
    int val[out.size()];   _mm_storeu_si128((__m128i*)val, out.value);
    int val_b[out.size()]; _mm_storeu_si128((__m128i*)val_b, b.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = a / val_b[i];
    }
    out.value = _mm_loadu_si128((__m128i*)val);
    return out;
}

FASTOR_INLINE SIMDVector<int,simd_abi::sse> abs(const SIMDVector<int,simd_abi::sse> &a) {
    SIMDVector<int,simd_abi::sse> out;
    out.value = _mm_abs_epi32(a.value);
    return out;
}



#endif


// SCALAR VERSION
//------------------------------------------------------------------------------------------------------------
template <>
struct SIMDVector<int, simd_abi::scalar> {
    using value_type = int;
    static constexpr FASTOR_INDEX Size = 1;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return 1;}
    static constexpr int unroll_size(FASTOR_INDEX size) {return (static_cast<int>(size) - 1);}

    FASTOR_INLINE SIMDVector() : value(0) {}
    FASTOR_INLINE SIMDVector(int num) : value(num) {}
    FASTOR_INLINE SIMDVector(const SIMDVector<int,simd_abi::scalar> &a) : value(a.value) {}
    FASTOR_INLINE SIMDVector(const int *data,bool Aligned=true) : value(*data) {}
    FASTOR_INLINE SIMDVector(int *data,bool Aligned=true) : value(*data) {}

    FASTOR_INLINE SIMDVector<int,simd_abi::scalar> operator=(int num) {
        value = num;
        return *this;
    }
    FASTOR_INLINE SIMDVector<int,simd_abi::scalar> operator=(const SIMDVector<int,simd_abi::scalar> &a) {
        value = a.value;
        return *this;
    }

    FASTOR_INLINE void load(const int *data, bool ) {
        value = *data;
    }
    FASTOR_INLINE void store(int *data, bool ) const {
        data[0] = value;
    }

    FASTOR_INLINE void load(const int *data) {
        value = *data;
    }
    FASTOR_INLINE void store(int *data) const {
        data[0] = value;
    }

    FASTOR_INLINE void aligned_load(const int *data) {
        value = *data;
    }
    FASTOR_INLINE void aligned_store(int *data) const {
        data[0] = value;
    }

    FASTOR_INLINE int operator[](FASTOR_INDEX) const {return value;}
    FASTOR_INLINE int operator()(FASTOR_INDEX) const {return value;}

    FASTOR_INLINE void set(int num) {
        value = num;
    }

    FASTOR_INLINE void set_sequential(int num) {
        value = num;
    }

    // In-place operators
    FASTOR_INLINE void operator+=(int num) {
        value += num;
    }
    FASTOR_INLINE void operator+=(const SIMDVector<int,simd_abi::scalar> &a) {
        value += a.value;
    }

    FASTOR_INLINE void operator-=(int num) {
        value -= num;
    }
    FASTOR_INLINE void operator-=(const SIMDVector<int,simd_abi::scalar> &a) {
        value -= a.value;
    }

    FASTOR_INLINE void operator*=(int num) {
        value *= num;
    }
    FASTOR_INLINE void operator*=(const SIMDVector<int,simd_abi::scalar> &a) {
        value *= a.value;
    }

    FASTOR_INLINE void operator/=(int num) {
        value /= num;
    }
    FASTOR_INLINE void operator/=(const SIMDVector<int,simd_abi::scalar> &a) {
        value /= a.value;
    }
    // end of in-place operators

    FASTOR_INLINE SIMDVector<int,simd_abi::scalar> shift(FASTOR_INDEX) {
        return *this;
    }
    FASTOR_INLINE int minimum() {return value;}
    FASTOR_INLINE int maximum() {return value;}
    FASTOR_INLINE SIMDVector<int,simd_abi::scalar> reverse() {SIMDVector<int,simd_abi::scalar> out; out.value = value; return out;}

    FASTOR_INLINE int sum() {return value;}
    FASTOR_INLINE int dot(const SIMDVector<int,simd_abi::scalar> &other) {
        return value*other.value;
    }

    int value;
};


FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<int,simd_abi::scalar> a) {
    os << "[" << a.value << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<int,simd_abi::scalar> operator+(const SIMDVector<int,simd_abi::scalar> &a, const SIMDVector<int,simd_abi::scalar> &b) {
    SIMDVector<int,simd_abi::scalar> out;
    out.value = a.value+b.value;
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::scalar> operator+(const SIMDVector<int,simd_abi::scalar> &a, int b) {
    SIMDVector<int,simd_abi::scalar> out;
    out.value = a.value+b;
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::scalar> operator+(int a, const SIMDVector<int,simd_abi::scalar> &b) {
    SIMDVector<int,simd_abi::scalar> out;
    out.value = a+b.value;
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::scalar> operator+(const SIMDVector<int,simd_abi::scalar> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<int,simd_abi::scalar> operator-(const SIMDVector<int,simd_abi::scalar> &a, const SIMDVector<int,simd_abi::scalar> &b) {
    SIMDVector<int,simd_abi::scalar> out;
    out.value = a.value-b.value;
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::scalar> operator-(const SIMDVector<int,simd_abi::scalar> &a, int b) {
    SIMDVector<int,simd_abi::scalar> out;
    out.value = a.value-b;
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::scalar> operator-(int a, const SIMDVector<int,simd_abi::scalar> &b) {
    SIMDVector<int,simd_abi::scalar> out;
    out.value = a-b.value;
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::scalar> operator-(const SIMDVector<int,simd_abi::scalar> &b) {
    SIMDVector<int,simd_abi::scalar> out;
    out.value = -b.value;
    return out;
}

FASTOR_INLINE SIMDVector<int,simd_abi::scalar> operator*(const SIMDVector<int,simd_abi::scalar> &a, const SIMDVector<int,simd_abi::scalar> &b) {
    SIMDVector<int,simd_abi::scalar> out;
    out.value = a.value*b.value;
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::scalar> operator*(const SIMDVector<int,simd_abi::scalar> &a, int b) {
    SIMDVector<int,simd_abi::scalar> out;
    out.value = a.value*b;
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::scalar> operator*(int a, const SIMDVector<int,simd_abi::scalar> &b) {
    SIMDVector<int,simd_abi::scalar> out;
    out.value = a*b.value;
    return out;
}

FASTOR_INLINE SIMDVector<int,simd_abi::scalar> operator/(const SIMDVector<int,simd_abi::scalar> &a, const SIMDVector<int,simd_abi::scalar> &b) {
    SIMDVector<int,simd_abi::scalar> out;
    out.value = a.value/b.value;
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::scalar> operator/(const SIMDVector<int,simd_abi::scalar> &a, int b) {
    SIMDVector<int,simd_abi::scalar> out;
    out.value = a.value/b;
    return out;
}
FASTOR_INLINE SIMDVector<int,simd_abi::scalar> operator/(int a, const SIMDVector<int,simd_abi::scalar> &b) {
    SIMDVector<int,simd_abi::scalar> out;
    out.value = a/b.value;
    return out;
}

FASTOR_INLINE SIMDVector<int,simd_abi::scalar> sqrt(const SIMDVector<int,simd_abi::scalar> &a) {
    return std::sqrt(a.value);
}

FASTOR_INLINE SIMDVector<int,simd_abi::scalar> abs(const SIMDVector<int,simd_abi::scalar> &a) {
    return std::abs(a.value);
}


}


#endif // SIMD_VECTOR_INT_H
