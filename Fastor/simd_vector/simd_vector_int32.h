#ifndef SIMD_VECTOR_INT_H
#define SIMD_VECTOR_INT_H

#include "Fastor/simd_vector/simd_vector_base.h"
#include <cstdint>

namespace Fastor {


// AVX512 VERSION
//-----------------------------------------------------------------------------------------------

#ifdef FASTOR_AVX512F_IMPL

template<>
struct SIMDVector<int32_t,simd_abi::avx512> {
    using value_type = __m512i;
    using scalar_value_type = int32_t;
    using abi_type = simd_abi::avx512;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<int32_t,simd_abi::avx512>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<int32_t,simd_abi::avx512>>::value;}

    FASTOR_INLINE SIMDVector() : value(_mm512_setzero_si512()) {}
    FASTOR_INLINE SIMDVector(int32_t num) : value(_mm512_set1_epi32(num)) {}
    FASTOR_INLINE SIMDVector(__m512i regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const int32_t *data, bool Aligned=true) {
        if (Aligned)
            value =_mm512_load_si512((__m512i*)data);
        else
            value = _mm512_loadu_si512((__m512i*)data);
    }

    FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> operator=(int32_t num) {
        value = _mm512_set1_epi32(num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> operator=(__m512i regi) {
        value = regi;
        return *this;
    }

    FASTOR_INLINE void load(const int32_t *data, bool Aligned=true) {
        if (Aligned)
            value =_mm512_load_si512((__m512i*)data);
        else
            value = _mm512_loadu_si512((__m512i*)data);
    }
    FASTOR_INLINE void store(int32_t *data, bool Aligned=true) const {
        if (Aligned)
            _mm512_store_si512((__m512i*)data,value);
        else
            _mm512_storeu_si512((__m512i*)data,value);
    }

    FASTOR_INLINE void aligned_load(const int32_t *data) {
        value =_mm512_load_si512((__m512i*)data);
    }
    FASTOR_INLINE void aligned_store(int32_t *data) const {
        _mm512_store_si512((__m512i*)data,value);
    }

    FASTOR_INLINE void mask_load(const scalar_value_type *a, uint8_t mask, bool Aligned=false) {
#ifdef FASTOR_HAS_AVX512_MASKS
        if (!Aligned)
            value = _mm512_mask_loadu_epi32(value, mask, a);
        else
            value = _mm512_mask_load_epi32(value, mask, a);
#else
        // perhaps very inefficient but they never get used
        int maska[Size];
        mask_to_array(mask,maska);
        value = _mm512_setzero_si512();
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                ((scalar_value_type*)&value)[Size - i - 1] = a[Size - i - 1];
            }
        }
        unused(Aligned);
#endif
    }
    FASTOR_INLINE void mask_store(scalar_value_type *a, uint8_t mask, bool Aligned=false) const {
#ifdef FASTOR_HAS_AVX512_MASKS
        if (!Aligned)
            _mm512_mask_storeu_epi32(a, mask, value);
        else
            _mm512_mask_store_epi32(a, mask, value);
#else
        // perhaps very inefficient but they never get used
        int maska[Size];
        mask_to_array(mask,maska);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                a[Size - i - 1] = ((const scalar_value_type*)&value)[Size - i - 1];
            }
            else {
                a[Size - i - 1] = 0;
            }
        }
        unused(Aligned);
#endif
    }

    FASTOR_INLINE int32_t operator[](FASTOR_INDEX i) const {return reinterpret_cast<const int32_t*>(&value)[i];}
    FASTOR_INLINE int32_t operator()(FASTOR_INDEX i) const {return reinterpret_cast<const int32_t*>(&value)[i];}

    FASTOR_INLINE void set(int32_t num) {
        value = _mm512_set1_epi32(num);
    }
    FASTOR_INLINE void set(int32_t num0, int32_t num1, int32_t num2, int32_t num3, int32_t num4, int32_t num5, int32_t num6, int32_t num7,
                           int32_t num8, int32_t num9, int32_t num10, int32_t num11, int32_t num12, int32_t num13, int32_t num14, int32_t num15) {
        value = _mm512_set_epi32(num0,num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,num13,num14,num15);
    }
    FASTOR_INLINE void set_sequential(int32_t num0) {
        value = _mm512_setr_epi32(num0,num0+1,num0+2,num0+3,num0+4,num0+5,num0+6,num0+7,
                                    num0+8,num0+9,num0+10,num0+11,num0+12,num0+13,num0+14,num0+15);
    }

    // In-place operators
    FASTOR_INLINE void operator+=(int32_t num) {
        value = _mm512_add_epi32(value,_mm512_set1_epi32(num));

    }
    FASTOR_INLINE void operator+=(__m512i regi) {
        value = _mm512_add_epi32(value,regi);
    }
    FASTOR_INLINE void operator+=(const SIMDVector<int32_t,simd_abi::avx512> &a) {
        value = _mm512_add_epi32(value,a.value);
    }

    FASTOR_INLINE void operator-=(int32_t num) {
        value = _mm512_sub_epi32(value,_mm512_set1_epi32(num));
    }
    FASTOR_INLINE void operator-=(__m512i regi) {
        value = _mm512_sub_epi32(value,regi);
    }
    FASTOR_INLINE void operator-=(const SIMDVector<int32_t,simd_abi::avx512> &a) {
        value = _mm512_sub_epi32(value,a.value);
    }

    FASTOR_INLINE void operator*=(int32_t num) {
        value = _mm512_mullo_epi32(value,_mm512_set1_epi32(num));
    }
    FASTOR_INLINE void operator*=(__m512i regi) {
        value = _mm512_mullo_epi32(value,regi);
    }
    FASTOR_INLINE void operator*=(const SIMDVector<int32_t,simd_abi::avx512> &a) {
        value = _mm512_mullo_epi32(value,a.value);
    }

    FASTOR_INLINE void operator/=(int32_t num) {
#ifdef FASTOR_INTEL
        value = _mm512_div_epi32(value,_mm512_set1_epi32(num));
#else
        int32_t val[Size]; _mm512_storeu_si512((__m512i*)val, value);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= num;
        }
        value = _mm512_loadu_si512((__m512i*)val);
#endif
    }
    FASTOR_INLINE void operator/=(__m512i regi) {
#ifdef FASTOR_INTEL
        value = _mm512_div_epi32(value,regi);
#else
        int32_t val[Size]; _mm512_storeu_si512((__m512i*)val, value);
        int32_t val_num[Size]; _mm512_storeu_si512((__m512i*)val_num, regi);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= val_num[i];
        }
        value = _mm512_loadu_si512((__m512i*)val);
#endif
    }
    FASTOR_INLINE void operator/=(const SIMDVector<int32_t,simd_abi::avx512> &a) {
#ifdef FASTOR_INTEL
        value = _mm512_div_epi32(value,a.value);
#else
        int32_t val[Size]; _mm512_storeu_si512((__m512i*)val, value);
        int32_t val_a[Size]; _mm512_storeu_si512((__m512i*)val_a, a.value);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= val_a[i];
        }
        value = _mm512_loadu_si512((__m512i*)val);
#endif
    }

    FASTOR_INLINE int32_t minimum() {
        int32_t *vals = (int32_t*)&value;
        int32_t quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            if (vals[i]<quan)
                quan = vals[i];
        return quan;
    }
    FASTOR_INLINE int32_t maximum() {
        int32_t *vals = (int32_t*)&value;
        int32_t quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            if (vals[i]>quan)
                quan = vals[i];
        return quan;
    }
    FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> reverse() {
        return _mm512_reverse_epi32(value);
    }

    FASTOR_INLINE int32_t sum() {
#ifdef FASTOR_HAS_AVX512_REDUCE_ADD
        return _mm512_reduce_add_epi32(value);
#else
        int32_t vals[Size]; _mm512_storeu_si512((__m512i*)vals, value);
        int32_t quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            quan += vals[i];
        return quan;
#endif
    }

    FASTOR_INLINE int32_t product() {
        int32_t vals[Size]; _mm512_storeu_si512((__m512i*)vals, value);
        int32_t quan = 1;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            quan *= vals[i];
        return quan;
    }

    FASTOR_INLINE int32_t dot(const SIMDVector<int32_t,simd_abi::avx512> &other) {
#ifdef FASTOR_HAS_AVX512_REDUCE_ADD
        return _mm512_reduce_add_epi32(_mm512_mullo_epi32(value,other.value));
#else
        return SIMDVector<int32_t,simd_abi::avx512>(_mm512_mullo_epi32(value,other.value)).sum();
#endif
    }

    __m512i value;
};

FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<int32_t,simd_abi::avx512> a) {
    const int32_t *value = (int32_t*) &a.value;
    os << "["
       << value[0]  << " " << value[1]  << " "
       << value[2]  << " " << value[3]  << " "
       << value[4]  << " " << value[5]  << " "
       << value[6]  << " " << value[7]  << " "
       << value[8]  << " " << value[9]  << " "
       << value[10] << " " << value[11] << " "
       << value[12] << " " << value[13] << " "
       << value[14] << " " << value[15] << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> operator+(const SIMDVector<int32_t,simd_abi::avx512> &a, const SIMDVector<int32_t,simd_abi::avx512> &b) {
    SIMDVector<int32_t,simd_abi::avx512> out;
    out.value = _mm512_add_epi32(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> operator+(const SIMDVector<int32_t,simd_abi::avx512> &a, int32_t b) {
    SIMDVector<int32_t,simd_abi::avx512> out;
    out.value = _mm512_add_epi32(a.value,_mm512_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> operator+(int32_t a, const SIMDVector<int32_t,simd_abi::avx512> &b) {
    SIMDVector<int32_t,simd_abi::avx512> out;
    out.value = _mm512_add_epi32(_mm512_set1_epi32(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> operator+(const SIMDVector<int32_t,simd_abi::avx512> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> operator-(const SIMDVector<int32_t,simd_abi::avx512> &a, const SIMDVector<int32_t,simd_abi::avx512> &b) {
    SIMDVector<int32_t,simd_abi::avx512> out;
    out.value = _mm512_sub_epi32(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> operator-(const SIMDVector<int32_t,simd_abi::avx512> &a, int32_t b) {
    SIMDVector<int32_t,simd_abi::avx512> out;
    out.value = _mm512_sub_epi32(a.value,_mm512_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> operator-(int32_t a, const SIMDVector<int32_t,simd_abi::avx512> &b) {
    SIMDVector<int32_t,simd_abi::avx512> out;
    out.value = _mm512_sub_epi32(_mm512_set1_epi32(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> operator-(const SIMDVector<int32_t,simd_abi::avx512> &b) {
    return _mm512_castps_si512(_mm512_neg_ps(_mm512_castsi512_ps(b.value)));
}

FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> operator*(const SIMDVector<int32_t,simd_abi::avx512> &a, const SIMDVector<int32_t,simd_abi::avx512> &b) {
    SIMDVector<int32_t,simd_abi::avx512> out;
    out.value = _mm512_mullo_epi32(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> operator*(const SIMDVector<int32_t,simd_abi::avx512> &a, int32_t b) {
    SIMDVector<int32_t,simd_abi::avx512> out;
    out.value = _mm512_mullo_epi32(a.value,_mm512_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> operator*(int32_t a, const SIMDVector<int32_t,simd_abi::avx512> &b) {
    SIMDVector<int32_t,simd_abi::avx512> out;
    out.value = _mm512_mullo_epi32(_mm512_set1_epi32(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> operator/(const SIMDVector<int32_t,simd_abi::avx512> &a, const SIMDVector<int32_t,simd_abi::avx512> &b) {
    SIMDVector<int32_t,simd_abi::avx512> out;
#ifdef FASTOR_INTEL
    out.value = _mm512_div_epi32(a.value,b.value);
#else
    int32_t val[out.size()];   _mm512_storeu_si512((__m512i*)val, out.value);
    int32_t val_a[out.size()]; _mm512_storeu_si512((__m512i*)val_a, a.value);
    int32_t val_b[out.size()]; _mm512_storeu_si512((__m512i*)val_b, b.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = val_a[i] / val_b[i];
    }
    out.value = _mm512_loadu_si512((__m512i*)val);
#endif
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> operator/(const SIMDVector<int32_t,simd_abi::avx512> &a, int32_t b) {
    SIMDVector<int32_t,simd_abi::avx512> out;
#ifdef FASTOR_INTEL
    out.value = _mm512_div_epi32(a.value,_mm512_set1_epi32(b));
#else
    int32_t val[out.size()];   _mm512_storeu_si512((__m512i*)val, out.value);
    int32_t val_a[out.size()]; _mm512_storeu_si512((__m512i*)val_a, a.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = val_a[i] / b;
    }
    out.value = _mm512_loadu_si512((__m512i*)val);
#endif
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> operator/(int32_t a, const SIMDVector<int32_t,simd_abi::avx512> &b) {
    SIMDVector<int32_t,simd_abi::avx512> out;
#ifdef FASTOR_INTEL
    out.value = _mm512_div_epi32(_mm512_set1_epi32(a),b.value);
#else
    int32_t val[out.size()];   _mm512_storeu_si512((__m512i*)val, out.value);
    int32_t val_b[out.size()]; _mm512_storeu_si512((__m512i*)val_b, b.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = a / val_b[i];
    }
    out.value = _mm512_loadu_si512((__m512i*)val);
#endif
    return out;
}

FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx512> abs(const SIMDVector<int32_t,simd_abi::avx512> &a) {
    SIMDVector<int32_t,simd_abi::avx512> out;
#ifdef FASTOR_HAS_AVX512_ABS
    out.value = _mm512_abs_epi32(a.value);
#else
    for (FASTOR_INDEX i=0UL; i<16UL; ++i) {
       ((int32_t*)&out.value)[i] = std::abs(((int32_t*)&a.value)[i]);
    }
#endif
    return out;
}


#endif


// AVX VERSION
//-----------------------------------------------------------------------------------------------

#ifdef FASTOR_AVX2_IMPL

template<>
struct SIMDVector<int32_t,simd_abi::avx> {
    using value_type = __m256i;
    using scalar_value_type = int32_t;
    using abi_type = simd_abi::avx;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<int32_t,simd_abi::avx>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<int32_t,simd_abi::avx>>::value;}

    FASTOR_INLINE SIMDVector() : value(_mm256_setzero_si256()) {}
    FASTOR_INLINE SIMDVector(int32_t num) : value(_mm256_set1_epi32(num)) {}
    FASTOR_INLINE SIMDVector(__m256i regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const int32_t *data, bool Aligned=true) {
        if (Aligned)
            value =_mm256_load_si256((__m256i*)data);
        else
            value = _mm256_loadu_si256((__m256i*)data);
    }

    FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> operator=(int32_t num) {
        value = _mm256_set1_epi32(num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> operator=(__m256i regi) {
        value = regi;
        return *this;
    }

    FASTOR_INLINE void load(const int32_t *data, bool Aligned=true) {
        if (Aligned)
            value =_mm256_load_si256((__m256i*)data);
        else
            value = _mm256_loadu_si256((__m256i*)data);
    }
    FASTOR_INLINE void store(int32_t *data, bool Aligned=true) const {
        if (Aligned)
            _mm256_store_si256((__m256i*)data,value);
        else
            _mm256_storeu_si256((__m256i*)data,value);
    }

    FASTOR_INLINE void aligned_load(const int32_t *data) {
        value =_mm256_load_si256((__m256i*)data);
    }
    FASTOR_INLINE void aligned_store(int32_t *data) const {
        _mm256_store_si256((__m256i*)data,value);
    }

    FASTOR_INLINE void mask_load(const scalar_value_type *a, uint8_t mask, bool Aligned=false) {
#ifdef FASTOR_HAS_AVX512_MASKS
        if (!Aligned)
            value = _mm256_mask_loadu_epi32(value, mask, a);
        else
            value = _mm256_mask_load_epi32(value, mask, a);
#else
        // perhaps very inefficient but they never get used
        int maska[Size];
        mask_to_array(mask,maska);
        value = _mm256_setzero_si256();
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                ((scalar_value_type*)&value)[Size - i - 1] = a[Size - i - 1];
            }
        }
        unused(Aligned);
#endif
    }
    FASTOR_INLINE void mask_store(scalar_value_type *a, uint8_t mask, bool Aligned=false) const {
#ifdef FASTOR_HAS_AVX512_MASKS
        if (!Aligned)
            _mm256_mask_storeu_epi32(a, mask, value);
        else
            _mm256_mask_store_epi32(a, mask, value);
#else
        // perhaps very inefficient but they never get used
        int maska[Size];
        mask_to_array(mask,maska);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                a[Size - i - 1] = ((const scalar_value_type*)&value)[Size - i - 1];
            }
            else {
                a[Size - i - 1] = 0;
            }
        }
        unused(Aligned);
#endif
    }

    FASTOR_INLINE int32_t operator[](FASTOR_INDEX i) const {return reinterpret_cast<const int32_t*>(&value)[i];}
    FASTOR_INLINE int32_t operator()(FASTOR_INDEX i) const {return reinterpret_cast<const int32_t*>(&value)[i];}

    FASTOR_INLINE void set(int32_t num) {
        value = _mm256_set1_epi32(num);
    }
    FASTOR_INLINE void set(int32_t num0, int32_t num1, int32_t num2, int32_t num3, int32_t num4, int32_t num5, int32_t num6, int32_t num7) {
        value = _mm256_set_epi32(num0,num1,num2,num3,num4,num5,num6,num7);
    }
    FASTOR_INLINE void set_sequential(int32_t num0) {
        value = _mm256_setr_epi32(num0,num0+1,num0+2,num0+3,num0+4,num0+5,num0+6,num0+7);
    }

    // In-place operators
    FASTOR_INLINE void operator+=(int32_t num) {
        value = _mm256_add_epi32x(value,_mm256_set1_epi32(num));

    }
    FASTOR_INLINE void operator+=(__m256i regi) {
        value = _mm256_add_epi32x(value,regi);
    }
    FASTOR_INLINE void operator+=(const SIMDVector<int32_t,simd_abi::avx> &a) {
        value = _mm256_add_epi32x(value,a.value);
    }

    FASTOR_INLINE void operator-=(int32_t num) {
        value = _mm256_sub_epi32x(value,_mm256_set1_epi32(num));
    }
    FASTOR_INLINE void operator-=(__m256i regi) {
        value = _mm256_sub_epi32x(value,regi);
    }
    FASTOR_INLINE void operator-=(const SIMDVector<int32_t,simd_abi::avx> &a) {
        value = _mm256_sub_epi32x(value,a.value);
    }

    FASTOR_INLINE void operator*=(int32_t num) {
        value = _mm256_mul_epi32x(value,_mm256_set1_epi32(num));
    }
    FASTOR_INLINE void operator*=(__m256i regi) {
        value = _mm256_mul_epi32x(value,regi);
    }
    FASTOR_INLINE void operator*=(const SIMDVector<int32_t,simd_abi::avx> &a) {
        value = _mm256_mul_epi32x(value,a.value);
    }

    FASTOR_INLINE void operator/=(int32_t num) {
        int32_t val[Size]; _mm256_storeu_si256((__m256i*)val, value);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= num;
        }
        value = _mm256_loadu_si256((__m256i*)val);
    }
    FASTOR_INLINE void operator/=(__m256i regi) {
        int32_t val[Size]; _mm256_storeu_si256((__m256i*)val, value);
        int32_t val_num[Size]; _mm256_storeu_si256((__m256i*)val_num, regi);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= val_num[i];
        }
        value = _mm256_loadu_si256((__m256i*)val);
    }
    FASTOR_INLINE void operator/=(const SIMDVector<int32_t,simd_abi::avx> &a) {
        int32_t val[Size]; _mm256_storeu_si256((__m256i*)val, value);
        int32_t val_a[Size]; _mm256_storeu_si256((__m256i*)val_a, a.value);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= val_a[i];
        }
        value = _mm256_loadu_si256((__m256i*)val);
    }

    FASTOR_INLINE int32_t minimum() {
        int32_t *vals = (int32_t*)&value;
        int32_t quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            if (vals[i]<quan)
                quan = vals[i];
        return quan;
    }
    FASTOR_INLINE int32_t maximum() {
        int32_t *vals = (int32_t*)&value;
        int32_t quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            if (vals[i]>quan)
                quan = vals[i];
        return quan;
    }
    FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> reverse() {
        SIMDVector<int32_t,simd_abi::avx> out;
        out.value = _mm256_reverse_epi32(value);
        return out;
    }

    FASTOR_INLINE int32_t sum() {
        int32_t vals[Size]; _mm256_storeu_si256((__m256i*)vals, value);
        int32_t quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            quan += vals[i];
        return quan;
    }

    FASTOR_INLINE int32_t dot(const SIMDVector<int32_t,simd_abi::avx> &other) {
        int32_t vals0[Size]; _mm256_storeu_si256((__m256i*)vals0, value);
        int32_t vals1[Size]; _mm256_storeu_si256((__m256i*)vals1, other.value);
        int32_t quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            quan += vals0[i]*vals1[i];
        return quan;
    }

    __m256i value;
};

FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<int32_t,simd_abi::avx> a) {
    const int32_t *value = (int32_t*) &a.value;
    os << "[" << value[0] <<  " " << value[1] << " " << value[2] << " " << value[3]
       << " " << value[4] <<  " " << value[5] << " " << value[6] << " " << value[7] << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> operator+(const SIMDVector<int32_t,simd_abi::avx> &a, const SIMDVector<int32_t,simd_abi::avx> &b) {
    SIMDVector<int32_t,simd_abi::avx> out;
    out.value = _mm256_add_epi32x(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> operator+(const SIMDVector<int32_t,simd_abi::avx> &a, int32_t b) {
    SIMDVector<int32_t,simd_abi::avx> out;
    out.value = _mm256_add_epi32x(a.value,_mm256_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> operator+(int32_t a, const SIMDVector<int32_t,simd_abi::avx> &b) {
    SIMDVector<int32_t,simd_abi::avx> out;
    out.value = _mm256_add_epi32x(_mm256_set1_epi32(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> operator+(const SIMDVector<int32_t,simd_abi::avx> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> operator-(const SIMDVector<int32_t,simd_abi::avx> &a, const SIMDVector<int32_t,simd_abi::avx> &b) {
    SIMDVector<int32_t,simd_abi::avx> out;
    out.value = _mm256_sub_epi32x(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> operator-(const SIMDVector<int32_t,simd_abi::avx> &a, int32_t b) {
    SIMDVector<int32_t,simd_abi::avx> out;
    out.value = _mm256_sub_epi32x(a.value,_mm256_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> operator-(int32_t a, const SIMDVector<int32_t,simd_abi::avx> &b) {
    SIMDVector<int32_t,simd_abi::avx> out;
    out.value = _mm256_sub_epi32x(_mm256_set1_epi32(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> operator-(const SIMDVector<int32_t,simd_abi::avx> &b) {
    return _mm256_castps_si256(_mm256_neg_ps(_mm256_castsi256_ps(b.value)));
}

FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> operator*(const SIMDVector<int32_t,simd_abi::avx> &a, const SIMDVector<int32_t,simd_abi::avx> &b) {
    SIMDVector<int32_t,simd_abi::avx> out;
    out.value = _mm256_mul_epi32x(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> operator*(const SIMDVector<int32_t,simd_abi::avx> &a, int32_t b) {
    SIMDVector<int32_t,simd_abi::avx> out;
    out.value = _mm256_mul_epi32x(a.value,_mm256_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> operator*(int32_t a, const SIMDVector<int32_t,simd_abi::avx> &b) {
    SIMDVector<int32_t,simd_abi::avx> out;
    out.value = _mm256_mul_epi32x(_mm256_set1_epi32(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> operator/(const SIMDVector<int32_t,simd_abi::avx> &a, const SIMDVector<int32_t,simd_abi::avx> &b) {
    SIMDVector<int32_t,simd_abi::avx> out;
    int32_t val[out.size()];   _mm256_storeu_si256((__m256i*)val, out.value);
    int32_t val_a[out.size()]; _mm256_storeu_si256((__m256i*)val_a, a.value);
    int32_t val_b[out.size()]; _mm256_storeu_si256((__m256i*)val_b, b.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = val_a[i] / val_b[i];
    }
    out.value = _mm256_loadu_si256((__m256i*)val);
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> operator/(const SIMDVector<int32_t,simd_abi::avx> &a, int32_t b) {
    SIMDVector<int32_t,simd_abi::avx> out;
    int32_t val[out.size()];   _mm256_storeu_si256((__m256i*)val, out.value);
    int32_t val_a[out.size()]; _mm256_storeu_si256((__m256i*)val_a, a.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = val_a[i] / b;
    }
    out.value = _mm256_loadu_si256((__m256i*)val);
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> operator/(int32_t a, const SIMDVector<int32_t,simd_abi::avx> &b) {
    SIMDVector<int32_t,simd_abi::avx> out;
    int32_t val[out.size()];   _mm256_storeu_si256((__m256i*)val, out.value);
    int32_t val_b[out.size()]; _mm256_storeu_si256((__m256i*)val_b, b.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = a / val_b[i];
    }
    out.value = _mm256_loadu_si256((__m256i*)val);
    return out;
}

FASTOR_INLINE SIMDVector<int32_t,simd_abi::avx> abs(const SIMDVector<int32_t,simd_abi::avx> &a) {
    SIMDVector<int32_t,simd_abi::avx> out;
#ifdef __AVX2__
    out.value = _mm256_abs_epi32(a.value);
#else
    // THIS IS ALSO AVX2 VERSION!
    // __m128i lo = _mm_abs_epi32(_mm256_castsi256_si128(a.value));
    // __m128i hi = _mm_abs_epi32(_mm256_extracti128_si256(a.value,0x1));
    // out.value = _mm256_castsi128_si256(lo);
    // out.value = _mm256_insertf128_si256(out.value,hi,0x1);

    int32_t *value = (int32_t*) &a.value;
    for (int32_t i=0; i<8; ++i) {
        value[i] = std::abs(value[i]);
    }
#endif
    return out;
}


#endif


// SSE VERSION
//-----------------------------------------------------------------------------------------------

#ifdef FASTOR_SSE2_IMPL

template<>
struct SIMDVector<int32_t,simd_abi::sse> {
    using value_type = __m128i;
    using scalar_value_type = int32_t;
    using abi_type = simd_abi::sse;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<int32_t,simd_abi::sse>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<int32_t,simd_abi::sse>>::value;}

    FASTOR_INLINE SIMDVector() : value(_mm_setzero_si128()) {}
    FASTOR_INLINE SIMDVector(int32_t num) : value(_mm_set1_epi32(num)) {}
    FASTOR_INLINE SIMDVector(__m128i regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const int32_t *data, bool Aligned=true) {
        if (Aligned)
            value =_mm_load_si128((__m128i*)data);
        else
            value = _mm_loadu_si128((__m128i*)data);
    }

    FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> operator=(int32_t num) {
        value = _mm_set1_epi32(num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> operator=(__m128i regi) {
        value = regi;
        return *this;
    }

    FASTOR_INLINE void load(const int32_t *data, bool Aligned=true) {
        if (Aligned)
            value =_mm_load_si128((__m128i*)data);
        else
            value = _mm_loadu_si128((__m128i*)data);
    }
    FASTOR_INLINE void store(int32_t *data, bool Aligned=true) const {
        if (Aligned)
            _mm_store_si128((__m128i*)data,value);
        else
            _mm_storeu_si128((__m128i*)data,value);
    }

    FASTOR_INLINE void aligned_load(const int32_t *data) {
        value =_mm_load_si128((__m128i*)data);
    }
    FASTOR_INLINE void aligned_store(int32_t *data) const {
        _mm_store_si128((__m128i*)data,value);
    }

    FASTOR_INLINE void mask_load(const scalar_value_type *a, uint8_t mask, bool Aligned=false) {
#ifdef FASTOR_HAS_AVX512_MASKS
        if (!Aligned)
            value = _mm_mask_loadu_epi32(value, mask, a);
        else
            value = _mm_mask_load_epi32(value, mask, a);
#else
        // perhaps very inefficient but they never get used
        int maska[Size];
        mask_to_array(mask,maska);
        value = _mm_setzero_si128();
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                ((scalar_value_type*)&value)[Size - i - 1] = a[Size - i - 1];
            }
        }
        unused(Aligned);
#endif
    }
    FASTOR_INLINE void mask_store(scalar_value_type *a, uint8_t mask, bool Aligned=false) const {
#ifdef FASTOR_HAS_AVX512_MASKS
        if (!Aligned)
            _mm_mask_storeu_epi32(a, mask, value);
        else
            _mm_mask_store_epi32(a, mask, value);
#else
        // perhaps very inefficient but they never get used
        int maska[Size];
        mask_to_array(mask,maska);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                a[Size - i - 1] = ((const scalar_value_type*)&value)[Size - i - 1];
            }
            else {
                a[Size - i - 1] = 0;
            }
        }
        unused(Aligned);
#endif
    }

    FASTOR_INLINE int32_t operator[](FASTOR_INDEX i) const {return reinterpret_cast<const int32_t*>(&value)[i];}
    FASTOR_INLINE int32_t operator()(FASTOR_INDEX i) const {return reinterpret_cast<const int32_t*>(&value)[i];}

    FASTOR_INLINE void set(int32_t num) {
        value = _mm_set1_epi32(num);
    }
    FASTOR_INLINE void set(int32_t num0, int32_t num1, int32_t num2, int32_t num3) {
        value = _mm_set_epi32(num0,num1,num2,num3);
    }
    FASTOR_INLINE void set_sequential(int32_t num0) {
        value = _mm_setr_epi32(num0,num0+1,num0+2,num0+3);
    }

    // In-place operators
    FASTOR_INLINE void operator+=(int32_t num) {
        value = _mm_add_epi32(value,_mm_set1_epi32(num));

    }
    FASTOR_INLINE void operator+=(__m128i regi) {
        value = _mm_add_epi32(value,regi);
    }
    FASTOR_INLINE void operator+=(const SIMDVector<int32_t,simd_abi::sse> &a) {
        value = _mm_add_epi32(value,a.value);
    }

    FASTOR_INLINE void operator-=(int32_t num) {
        value = _mm_sub_epi32(value,_mm_set1_epi32(num));
    }
    FASTOR_INLINE void operator-=(__m128i regi) {
        value = _mm_sub_epi32(value,regi);
    }
    FASTOR_INLINE void operator-=(const SIMDVector<int32_t,simd_abi::sse> &a) {
        value = _mm_sub_epi32(value,a.value);
    }

    FASTOR_INLINE void operator*=(int32_t num) {
        value = _mm_mul_epi32x(value,_mm_set1_epi32(num));
    }
    FASTOR_INLINE void operator*=(__m128i regi) {
        value = _mm_mul_epi32x(value,regi);
    }
    FASTOR_INLINE void operator*=(const SIMDVector<int32_t,simd_abi::sse> &a) {
        value = _mm_mul_epi32x(value,a.value);
    }

    FASTOR_INLINE void operator/=(int32_t num) {
        int32_t val[Size]; _mm_storeu_si128((__m128i*)val, value);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= num;
        }
        value = _mm_loadu_si128((__m128i*)val);
    }
    FASTOR_INLINE void operator/=(__m128i regi) {
        int32_t val[Size]; _mm_storeu_si128((__m128i*)val, value);
        int32_t val_num[Size]; _mm_storeu_si128((__m128i*)val_num, regi);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= val_num[i];
        }
        value = _mm_loadu_si128((__m128i*)val);
    }
    FASTOR_INLINE void operator/=(const SIMDVector<int32_t,simd_abi::sse> &a) {
        int32_t val[Size]; _mm_storeu_si128((__m128i*)val, value);
        int32_t val_a[Size]; _mm_storeu_si128((__m128i*)val_a, a.value);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= val_a[i];
        }
        value = _mm_loadu_si128((__m128i*)val);
    }

    FASTOR_INLINE int32_t minimum() {
        int32_t *vals = (int32_t*)&value;
        int32_t quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            if (vals[i]<quan)
                quan = vals[i];
        return quan;
    }
    FASTOR_INLINE int32_t maximum() {
        int32_t *vals = (int32_t*)&value;
        int32_t quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            if (vals[i]>quan)
                quan = vals[i];
        return quan;
    }
    FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> reverse() {
        return _mm_reverse_epi32(value);
    }

    FASTOR_INLINE int32_t sum() {return _mm_sum_epi32(value);}
    FASTOR_INLINE int32_t product() {return _mm_prod_epi32(value);}

    FASTOR_INLINE int32_t dot(const SIMDVector<int32_t,simd_abi::sse> &other) {
        return _mm_sum_epi32(_mm_mul_epi32x(value,other.value));
    }

    __m128i value;
};

FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<int32_t,simd_abi::sse> a) {
    const int32_t *value = (int32_t*) &a.value;
    os << "[" << value[0] <<  " " << value[1] << " " << value[2] << " " << value[3] << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> operator+(const SIMDVector<int32_t,simd_abi::sse> &a, const SIMDVector<int32_t,simd_abi::sse> &b) {
    SIMDVector<int32_t,simd_abi::sse> out;
    out.value = _mm_add_epi32(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> operator+(const SIMDVector<int32_t,simd_abi::sse> &a, int32_t b) {
    SIMDVector<int32_t,simd_abi::sse> out;
    out.value = _mm_add_epi32(a.value,_mm_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> operator+(int32_t a, const SIMDVector<int32_t,simd_abi::sse> &b) {
    SIMDVector<int32_t,simd_abi::sse> out;
    out.value = _mm_add_epi32(_mm_set1_epi32(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> operator+(const SIMDVector<int32_t,simd_abi::sse> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> operator-(const SIMDVector<int32_t,simd_abi::sse> &a, const SIMDVector<int32_t,simd_abi::sse> &b) {
    SIMDVector<int32_t,simd_abi::sse> out;
    out.value = _mm_sub_epi32(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> operator-(const SIMDVector<int32_t,simd_abi::sse> &a, int32_t b) {
    SIMDVector<int32_t,simd_abi::sse> out;
    out.value = _mm_sub_epi32(a.value,_mm_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> operator-(int32_t a, const SIMDVector<int32_t,simd_abi::sse> &b) {
    SIMDVector<int32_t,simd_abi::sse> out;
    out.value = _mm_sub_epi32(_mm_set1_epi32(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> operator-(const SIMDVector<int32_t,simd_abi::sse> &b) {
    return _mm_castps_si128(_mm_neg_ps(_mm_castsi128_ps(b.value)));
}

FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> operator*(const SIMDVector<int32_t,simd_abi::sse> &a, const SIMDVector<int32_t,simd_abi::sse> &b) {
    SIMDVector<int32_t,simd_abi::sse> out;
    out.value = _mm_mul_epi32x(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> operator*(const SIMDVector<int32_t,simd_abi::sse> &a, int32_t b) {
    SIMDVector<int32_t,simd_abi::sse> out;
    out.value = _mm_mul_epi32x(a.value,_mm_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> operator*(int32_t a, const SIMDVector<int32_t,simd_abi::sse> &b) {
    SIMDVector<int32_t,simd_abi::sse> out;
    out.value = _mm_mul_epi32x(_mm_set1_epi32(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> operator/(const SIMDVector<int32_t,simd_abi::sse> &a, const SIMDVector<int32_t,simd_abi::sse> &b) {
    SIMDVector<int32_t,simd_abi::sse> out;
    int32_t val[out.size()];   _mm_storeu_si128((__m128i*)val, out.value);
    int32_t val_a[out.size()]; _mm_storeu_si128((__m128i*)val_a, a.value);
    int32_t val_b[out.size()]; _mm_storeu_si128((__m128i*)val_b, b.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = val_a[i] / val_b[i];
    }
    out.value = _mm_loadu_si128((__m128i*)val);
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> operator/(const SIMDVector<int32_t,simd_abi::sse> &a, int32_t b) {
    SIMDVector<int32_t,simd_abi::sse> out;
    int32_t val[out.size()];   _mm_storeu_si128((__m128i*)val, out.value);
    int32_t val_a[out.size()]; _mm_storeu_si128((__m128i*)val_a, a.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = val_a[i] / b;
    }
    out.value = _mm_loadu_si128((__m128i*)val);
    return out;
}
FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> operator/(int32_t a, const SIMDVector<int32_t,simd_abi::sse> &b) {
    SIMDVector<int32_t,simd_abi::sse> out;
    int32_t val[out.size()];   _mm_storeu_si128((__m128i*)val, out.value);
    int32_t val_b[out.size()]; _mm_storeu_si128((__m128i*)val_b, b.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = a / val_b[i];
    }
    out.value = _mm_loadu_si128((__m128i*)val);
    return out;
}

FASTOR_INLINE SIMDVector<int32_t,simd_abi::sse> abs(const SIMDVector<int32_t,simd_abi::sse> &a) {
    SIMDVector<int32_t,simd_abi::sse> out;
#ifdef FASTOR_SSSE3_IMPL
    out.value = _mm_abs_epi32(a.value);
#else // SSE2
    __m128i sign = _mm_srai_epi32(a.value, 31);
    __m128i inv = _mm_xor_si128(a.value, sign);
    out.value = _mm_sub_epi32(inv, sign);
#endif
    return out;
}


#endif


} // end of namespace Fastor


#endif // SIMD_VECTOR_INT_H
