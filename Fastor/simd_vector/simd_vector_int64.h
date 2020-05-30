#ifndef SIMD_VECTOR_INT64_H
#define SIMD_VECTOR_INT64_H

#include "Fastor/simd_vector/simd_vector_base.h"
#include <cstdint>

namespace Fastor {


// AVX512 VERSION
//-----------------------------------------------------------------------------------------------

#ifdef FASTOR_AVX512F_IMPL

template<>
struct SIMDVector<int64_t,simd_abi::avx512> {
    using value_type = __m512i;
    using scalar_value_type = int64_t;
    using abi_type = simd_abi::avx512;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<int64_t,simd_abi::avx512>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<int64_t,simd_abi::avx512>>::value;}

    FASTOR_INLINE SIMDVector() : value(_mm512_setzero_si512()) {}
    FASTOR_INLINE SIMDVector(int64_t num) : value(_mm512_set1_epi64(num)) {}
    FASTOR_INLINE SIMDVector(__m512i regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const int64_t *data, bool Aligned=true) {
        if (Aligned)
            value =_mm512_load_si512((__m512i*)data);
        else
            value = _mm512_loadu_si512((__m512i*)data);
    }

    FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> operator=(int64_t num) {
        value = _mm512_set1_epi64(num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> operator=(__m512i regi) {
        value = regi;
        return *this;
    }

    FASTOR_INLINE void load(const int64_t *data, bool Aligned=true) {
        if (Aligned)
            value =_mm512_load_si512((__m512i*)data);
        else
            value = _mm512_loadu_si512((__m512i*)data);
    }
    FASTOR_INLINE void store(int64_t *data, bool Aligned=true) const {
        if (Aligned)
            _mm512_store_si512((__m512i*)data,value);
        else
            _mm512_storeu_si512((__m512i*)data,value);
    }

    FASTOR_INLINE void aligned_load(const int64_t *data) {
        value =_mm512_load_si512((__m512i*)data);
    }
    FASTOR_INLINE void aligned_store(int64_t *data) const {
        _mm512_store_si512((__m512i*)data,value);
    }

    FASTOR_INLINE int64_t operator[](FASTOR_INDEX i) const {return reinterpret_cast<const int64_t*>(&value)[i];}
    FASTOR_INLINE int64_t operator()(FASTOR_INDEX i) const {return reinterpret_cast<const int64_t*>(&value)[i];}

    FASTOR_INLINE void mask_load(const scalar_value_type *a, uint8_t mask, bool Aligned=false) {
#ifdef FASTOR_HAS_AVX512_MASKS
        if (!Aligned)
            value = _mm512_mask_loadu_epi64(value, mask, a);
        else
            value = _mm512_mask_load_epi64(value, mask, a);
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
            _mm512_mask_storeu_epi64(a, mask, value);
        else
            _mm512_mask_store_epi64(a, mask, value);
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

    FASTOR_INLINE void set(int64_t num) {
        value = _mm512_set1_epi64(num);
    }
    FASTOR_INLINE void set(int64_t num0, int64_t num1, int64_t num2, int64_t num3, int64_t num4, int64_t num5, int64_t num6, int64_t num7) {
        value = _mm512_set_epi64(num0,num1,num2,num3,num4,num5,num6,num7);
    }
    FASTOR_INLINE void set_sequential(int64_t num0) {
        value = _mm512_setr_epi64(num0,num0+1,num0+2,num0+3,num0+4,num0+5,num0+6,num0+7);
    }

    // In-place operators
    FASTOR_INLINE void operator+=(int64_t num) {
        value = _mm512_add_epi64(value,_mm512_set1_epi64(num));

    }
    FASTOR_INLINE void operator+=(__m512i regi) {
        value = _mm512_add_epi64(value,regi);
    }
    FASTOR_INLINE void operator+=(const SIMDVector<int64_t,simd_abi::avx512> &a) {
        value = _mm512_add_epi64(value,a.value);
    }

    FASTOR_INLINE void operator-=(int64_t num) {
        value = _mm512_sub_epi64(value,_mm512_set1_epi64(num));
    }
    FASTOR_INLINE void operator-=(__m512i regi) {
        value = _mm512_sub_epi64(value,regi);
    }
    FASTOR_INLINE void operator-=(const SIMDVector<int64_t,simd_abi::avx512> &a) {
        value = _mm512_sub_epi64(value,a.value);
    }

    FASTOR_INLINE void operator*=(int64_t num) {
#ifdef FASTOR_AVX512DQ_IMPL
        value = _mm512_mullo_epi64(value,_mm512_set1_epi64(num));
#else
        for (FASTOR_INDEX i=0; i<Size; i++) {
            ((int64_t*)&value)[i] *= num;
        }
#endif
    }
    FASTOR_INLINE void operator*=(__m512i regi) {
#ifdef FASTOR_AVX512DQ_IMPL
        value = _mm512_mullo_epi64(value,regi);
#else
        for (FASTOR_INDEX i=0; i<Size; i++) {
            ((int64_t*)&value)[i] *= (((const int64_t*)&regi)[i]);
        }
#endif
    }
    FASTOR_INLINE void operator*=(const SIMDVector<int64_t,simd_abi::avx512> &a) {
#ifdef FASTOR_AVX512DQ_IMPL
        value = _mm512_mullo_epi64(value,a.value);
#else
        for (FASTOR_INDEX i=0; i<Size; i++) {
            ((int64_t*)&value)[i] *= (((const int64_t*)&(a.value))[i]);
        }
#endif
    }

    FASTOR_INLINE void operator/=(int64_t num) {
#ifdef FASTOR_INTEL
        value = _mm512_div_epi64(value,_mm512_set1_epi64(num));
#else
        int64_t val[Size]; _mm512_storeu_si512((__m512i*)val, value);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= num;
        }
        value = _mm512_loadu_si512((__m512i*)val);
#endif
    }
    FASTOR_INLINE void operator/=(__m512i regi) {
#ifdef FASTOR_INTEL
        value = _mm512_div_epi64(value,regi);
#else
        int64_t val[Size]; _mm512_storeu_si512((__m512i*)val, value);
        int64_t val_num[Size]; _mm512_storeu_si512((__m512i*)val_num, regi);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= val_num[i];
        }
        value = _mm512_loadu_si512((__m512i*)val);
#endif
    }
    FASTOR_INLINE void operator/=(const SIMDVector<int64_t,simd_abi::avx512> &a) {
#ifdef FASTOR_INTEL
        value = _mm512_div_epi64(value,a.value);
#else
        int64_t val[Size]; _mm512_storeu_si512((__m512i*)val, value);
        int64_t val_a[Size]; _mm512_storeu_si512((__m512i*)val_a, a.value);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= val_a[i];
        }
        value = _mm512_loadu_si512((__m512i*)val);
#endif
    }

    FASTOR_INLINE int64_t minimum() {
        const int64_t *vals = reinterpret_cast<const int64_t*>(&value);
        int64_t quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            if (vals[i]<quan)
                quan = vals[i];
        return quan;
    }
    FASTOR_INLINE int64_t maximum() {
        const int64_t *vals = reinterpret_cast<const int64_t*>(&value);
        int64_t quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            if (vals[i]>quan)
                quan = vals[i];
        return quan;
    }
    FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> reverse() {
        return _mm512_reverse_epi64(value);
    }

    FASTOR_INLINE int64_t sum() {
#ifdef FASTOR_HAS_AVX512_REDUCE_ADD
        return _mm512_reduce_add_epi64(value);
#else
        const int64_t *vals = reinterpret_cast<const int64_t*>(&value);
        int64_t quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            quan += vals[i];
        return quan;
#endif
    }

    FASTOR_INLINE int64_t product() {
        const int64_t *vals = reinterpret_cast<const int64_t*>(&value);
        int64_t quan = 1;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            quan *= vals[i];
        return quan;
    }

    FASTOR_INLINE int64_t dot(const SIMDVector<int64_t,simd_abi::avx512> &other) {
        return (*this * other).sum();
    }

    __m512i value;
};

FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<int64_t,simd_abi::avx512> a) {
    const int64_t *value = reinterpret_cast<const int64_t*>(&a.value);
    os << "[" << value[0] <<  " " << value[1] << " " << value[2] << " " << value[3]
       << " " << value[4] <<  " " << value[5] << " " << value[6] << " " << value[7] << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> operator+(const SIMDVector<int64_t,simd_abi::avx512> &a, const SIMDVector<int64_t,simd_abi::avx512> &b) {
    SIMDVector<int64_t,simd_abi::avx512> out;
    out.value = _mm512_add_epi64(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> operator+(const SIMDVector<int64_t,simd_abi::avx512> &a, int64_t b) {
    SIMDVector<int64_t,simd_abi::avx512> out;
    out.value = _mm512_add_epi64(a.value,_mm512_set1_epi64(b));
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> operator+(int64_t a, const SIMDVector<int64_t,simd_abi::avx512> &b) {
    SIMDVector<int64_t,simd_abi::avx512> out;
    out.value = _mm512_add_epi64(_mm512_set1_epi64(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> operator+(const SIMDVector<int64_t,simd_abi::avx512> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> operator-(const SIMDVector<int64_t,simd_abi::avx512> &a, const SIMDVector<int64_t,simd_abi::avx512> &b) {
    SIMDVector<int64_t,simd_abi::avx512> out;
    out.value = _mm512_sub_epi64(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> operator-(const SIMDVector<int64_t,simd_abi::avx512> &a, int64_t b) {
    SIMDVector<int64_t,simd_abi::avx512> out;
    out.value = _mm512_sub_epi64(a.value,_mm512_set1_epi64(b));
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> operator-(int64_t a, const SIMDVector<int64_t,simd_abi::avx512> &b) {
    SIMDVector<int64_t,simd_abi::avx512> out;
    out.value = _mm512_sub_epi64(_mm512_set1_epi64(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> operator-(const SIMDVector<int64_t,simd_abi::avx512> &b) {
    return _mm512_castps_si512(_mm512_neg_ps(_mm512_castsi512_ps(b.value)));
}

FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> operator*(const SIMDVector<int64_t,simd_abi::avx512> &a, const SIMDVector<int64_t,simd_abi::avx512> &b) {
    SIMDVector<int64_t,simd_abi::avx512> out;
#ifdef FASTOR_AVX512DQ_IMPL
    out.value = _mm512_mullo_epi64(a.value,b.value);
#else
    for (FASTOR_INDEX i=0; i<out.size(); i++) {
       ((int64_t*)&out.value)[i] = (((int64_t*)&a.value)[i])*(((int64_t*)&b.value)[i]);
    }
#endif
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> operator*(const SIMDVector<int64_t,simd_abi::avx512> &a, int64_t b) {
    SIMDVector<int64_t,simd_abi::avx512> out;
#ifdef FASTOR_AVX512DQ_IMPL
    out.value = _mm512_mullo_epi64(a.value,_mm512_set1_epi64(b));
#else
    for (FASTOR_INDEX i=0; i<out.size(); i++) {
       ((int64_t*)&out.value)[i] = (((int64_t*)&a.value)[i])*b;
    }
#endif
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> operator*(int64_t a, const SIMDVector<int64_t,simd_abi::avx512> &b) {
    SIMDVector<int64_t,simd_abi::avx512> out;
#ifdef FASTOR_AVX512DQ_IMPL
    out.value = _mm512_mullo_epi64(_mm512_set1_epi64(a),b.value);
#else
    for (FASTOR_INDEX i=0; i<out.size(); i++) {
       ((int64_t*)&out.value)[i] = a*(((int64_t*)&b.value)[i]);
    }
#endif
    return out;
}

FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> operator/(const SIMDVector<int64_t,simd_abi::avx512> &a, const SIMDVector<int64_t,simd_abi::avx512> &b) {
    SIMDVector<int64_t,simd_abi::avx512> out;
#ifdef FASTOR_INTEL
    out.value = _mm512_div_epi64(a.value,b.value);
#else
    int64_t val[out.size()];   _mm512_storeu_si512((__m512i*)val, out.value);
    int64_t val_a[out.size()]; _mm512_storeu_si512((__m512i*)val_a, a.value);
    int64_t val_b[out.size()]; _mm512_storeu_si512((__m512i*)val_b, b.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = val_a[i] / val_b[i];
    }
    out.value = _mm512_loadu_si512((__m512i*)val);
#endif
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> operator/(const SIMDVector<int64_t,simd_abi::avx512> &a, int64_t b) {
    SIMDVector<int64_t,simd_abi::avx512> out;
#ifdef FASTOR_INTEL
    out.value = _mm512_div_epi64(a.value,_mm512_set1_epi64(b));
#else
    int64_t val[out.size()];   _mm512_storeu_si512((__m512i*)val, out.value);
    int64_t val_a[out.size()]; _mm512_storeu_si512((__m512i*)val_a, a.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = val_a[i] / b;
    }
    out.value = _mm512_loadu_si512((__m512i*)val);
#endif
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> operator/(int64_t a, const SIMDVector<int64_t,simd_abi::avx512> &b) {
    SIMDVector<int64_t,simd_abi::avx512> out;
#ifdef FASTOR_INTEL
    out.value = _mm512_div_epi64(_mm512_set1_epi64(a),b.value);
#else
    int64_t val[out.size()];   _mm512_storeu_si512((__m512i*)val, out.value);
    int64_t val_b[out.size()]; _mm512_storeu_si512((__m512i*)val_b, b.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = a / val_b[i];
    }
    out.value = _mm512_loadu_si512((__m512i*)val);
#endif
    return out;
}

FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx512> abs(const SIMDVector<int64_t,simd_abi::avx512> &a) {
    SIMDVector<int64_t,simd_abi::avx512> out;
#ifdef FASTOR_HAS_AVX512_ABS
    out.value = _mm512_abs_epi64(a.value);
#else
    for (FASTOR_INDEX i=0UL; i<8UL; ++i) {
       ((int64_t*)&out.value)[i] = std::abs(((int64_t*)&a.value)[i]);
    }
#endif
    return out;
}

#endif




// AVX VERSION
//-----------------------------------------------------------------------------------------------

#ifdef FASTOR_AVX2_IMPL

template<>
struct SIMDVector<int64_t,simd_abi::avx> {
    using value_type = __m256i;
    using scalar_value_type = int64_t;
    using abi_type = simd_abi::avx;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<int64_t,simd_abi::avx>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<int64_t,simd_abi::avx>>::value;}

    FASTOR_INLINE SIMDVector() : value(_mm256_setzero_si256()) {}
    FASTOR_INLINE SIMDVector(int64_t num) : value(_mm256_set1_epi64x(num)) {}
    FASTOR_INLINE SIMDVector(__m256i regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const int64_t *data, bool Aligned=true) {
        if (Aligned)
            value =_mm256_load_si256((__m256i*)data);
        else
            value = _mm256_loadu_si256((__m256i*)data);
    }

    FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx> operator=(int64_t num) {
        value = _mm256_set1_epi64x(num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx> operator=(__m256i regi) {
        value = regi;
        return *this;
    }

    FASTOR_INLINE void load(const int64_t *data, bool Aligned=true) {
        if (Aligned)
            value =_mm256_load_si256((__m256i*)data);
        else
            value = _mm256_loadu_si256((__m256i*)data);
    }
    FASTOR_INLINE void store(int64_t *data, bool Aligned=true) const {
        if (Aligned)
            _mm256_store_si256((__m256i*)data,value);
        else
            _mm256_storeu_si256((__m256i*)data,value);
    }

    FASTOR_INLINE void aligned_load(const int64_t *data) {
        value =_mm256_load_si256((__m256i*)data);
    }
    FASTOR_INLINE void aligned_store(int64_t *data) const {
        _mm256_store_si256((__m256i*)data,value);
    }

    FASTOR_INLINE int64_t operator[](FASTOR_INDEX i) const {return reinterpret_cast<const int64_t*>(&value)[i];}
    FASTOR_INLINE int64_t operator()(FASTOR_INDEX i) const {return reinterpret_cast<const int64_t*>(&value)[i];}

    FASTOR_INLINE void mask_load(const scalar_value_type *a, uint8_t mask, bool Aligned=false) {
#ifdef FASTOR_HAS_AVX512_MASKS
        if (!Aligned)
            value = _mm256_mask_loadu_epi64(value, mask, a);
        else
            value = _mm256_mask_load_epi64(value, mask, a);
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
            _mm256_mask_storeu_epi64(a, mask, value);
        else
            _mm256_mask_store_epi64(a, mask, value);
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

    FASTOR_INLINE void set(int64_t num) {
        value = _mm256_set1_epi64x(num);
    }
    FASTOR_INLINE void set(int64_t num0, int64_t num1, int64_t num2, int64_t num3) {
        value = _mm256_set_epi64x(num0,num1,num2,num3);
    }
    FASTOR_INLINE void set_sequential(int64_t num0) {
        value = _mm256_setr_epi64x(num0,num0+1,num0+2,num0+3);
    }

    // In-place operators
    FASTOR_INLINE void operator+=(int64_t num) {
        value = _mm256_add_epi64x(value,_mm256_set1_epi64x(num));

    }
    FASTOR_INLINE void operator+=(__m256i regi) {
        value = _mm256_add_epi64x(value,regi);
    }
    FASTOR_INLINE void operator+=(const SIMDVector<int64_t,simd_abi::avx> &a) {
        value = _mm256_add_epi64x(value,a.value);
    }

    FASTOR_INLINE void operator-=(int64_t num) {
        value = _mm256_sub_epi64x(value,_mm256_set1_epi64x(num));
    }
    FASTOR_INLINE void operator-=(__m256i regi) {
        value = _mm256_sub_epi64x(value,regi);
    }
    FASTOR_INLINE void operator-=(const SIMDVector<int64_t,simd_abi::avx> &a) {
        value = _mm256_sub_epi64x(value,a.value);
    }

    FASTOR_INLINE void operator*=(int64_t num) {
        value = _mm256_mul_epi64x(value,_mm256_set1_epi64x(num));
    }
    FASTOR_INLINE void operator*=(__m256i regi) {
        value = _mm256_mul_epi64x(value,regi);
    }
    FASTOR_INLINE void operator*=(const SIMDVector<int64_t,simd_abi::avx> &a) {
        value = _mm256_mul_epi64x(value,a.value);
    }

    FASTOR_INLINE void operator/=(int64_t num) {
        int64_t val[Size]; _mm256_storeu_si256((__m256i*)val, value);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= num;
        }
        value = _mm256_loadu_si256((__m256i*)val);
    }
    FASTOR_INLINE void operator/=(__m256i regi) {
        int64_t val[Size]; _mm256_storeu_si256((__m256i*)val, value);
        int64_t val_num[Size]; _mm256_storeu_si256((__m256i*)val_num, regi);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= val_num[i];
        }
        value = _mm256_loadu_si256((__m256i*)val);
    }
    FASTOR_INLINE void operator/=(const SIMDVector<int64_t,simd_abi::avx> &a) {
        int64_t val[Size]; _mm256_storeu_si256((__m256i*)val, value);
        int64_t val_a[Size]; _mm256_storeu_si256((__m256i*)val_a, a.value);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= val_a[i];
        }
        value = _mm256_loadu_si256((__m256i*)val);
    }

    FASTOR_INLINE int64_t minimum() {
        const int64_t *vals = reinterpret_cast<const int64_t*>(&value);
        int64_t quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            if (vals[i]<quan)
                quan = vals[i];
        return quan;
    }
    FASTOR_INLINE int64_t maximum() {
        const int64_t *vals = reinterpret_cast<const int64_t*>(&value);
        int64_t quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            if (vals[i]>quan)
                quan = vals[i];
        return quan;
    }
    FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx> reverse() {
        // Reversing a 64 bit vector seems really expensive regardless
        // of which of the following methods being used
        // SIMDVector<int64_t,simd_abi::avx> out(_mm256_set_epi64x(value[0],value[1],value[2],value[3]));
        // SIMDVector<int64_t,simd_abi::avx> out; out.set(value[3],value[2],value[1],value[0]);
        // return out;
        SIMDVector<int64_t,simd_abi::avx> out;
        out.value = _mm256_reverse_epi64(value);
        return out;
    }

    FASTOR_INLINE int64_t sum() {
        const int64_t *vals = reinterpret_cast<const int64_t*>(&value);
        int64_t quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            quan += vals[i];
        return quan;
    }

    FASTOR_INLINE int64_t product() {
        const int64_t *vals = reinterpret_cast<const int64_t*>(&value);
        int64_t quan = 1;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            quan *= vals[i];
        return quan;
    }

    FASTOR_INLINE int64_t dot(const SIMDVector<int64_t,simd_abi::avx> &other) {
        const int64_t *vals0 = reinterpret_cast<const int64_t*>(&value);
        const int64_t *vals1 = reinterpret_cast<const int64_t*>(&other.value);
        int64_t quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            quan += vals0[i]*vals1[i];
        return quan;
    }

    __m256i value;
};

FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<int64_t,simd_abi::avx> a) {
    const int64_t *value = reinterpret_cast<const int64_t*>(&a.value);
    os << "[" << value[0] <<  " " << value[1] << " " << value[2] << " " << value[3] << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx> operator+(const SIMDVector<int64_t,simd_abi::avx> &a, const SIMDVector<int64_t,simd_abi::avx> &b) {
    SIMDVector<int64_t,simd_abi::avx> out;
    out.value = _mm256_add_epi64x(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx> operator+(const SIMDVector<int64_t,simd_abi::avx> &a, int64_t b) {
    SIMDVector<int64_t,simd_abi::avx> out;
    out.value = _mm256_add_epi64x(a.value,_mm256_set1_epi64x(b));
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx> operator+(int64_t a, const SIMDVector<int64_t,simd_abi::avx> &b) {
    SIMDVector<int64_t,simd_abi::avx> out;
    out.value = _mm256_add_epi64x(_mm256_set1_epi64x(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx> operator+(const SIMDVector<int64_t,simd_abi::avx> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx> operator-(const SIMDVector<int64_t,simd_abi::avx> &a, const SIMDVector<int64_t,simd_abi::avx> &b) {
    SIMDVector<int64_t,simd_abi::avx> out;
    out.value = _mm256_sub_epi64x(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx> operator-(const SIMDVector<int64_t,simd_abi::avx> &a, int64_t b) {
    SIMDVector<int64_t,simd_abi::avx> out;
    out.value = _mm256_sub_epi64x(a.value,_mm256_set1_epi64x(b));
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx> operator-(int64_t a, const SIMDVector<int64_t,simd_abi::avx> &b) {
    SIMDVector<int64_t,simd_abi::avx> out;
    out.value = _mm256_sub_epi64x(_mm256_set1_epi64x(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx> operator-(const SIMDVector<int64_t,simd_abi::avx> &b) {
    SIMDVector<int64_t,simd_abi::avx> out;
    out.value = _mm256_castpd_si256(_mm256_neg_pd(_mm256_castsi256_pd(b.value)));
    return out;
}

FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx> operator*(const SIMDVector<int64_t,simd_abi::avx> &a, const SIMDVector<int64_t,simd_abi::avx> &b) {
    SIMDVector<int64_t,simd_abi::avx> out;
    out.value = _mm256_mul_epi64x(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx> operator*(const SIMDVector<int64_t,simd_abi::avx> &a, int64_t b) {
    SIMDVector<int64_t,simd_abi::avx> out;
    out.value = _mm256_mul_epi64x(a.value,_mm256_set1_epi64x(b));
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx> operator*(int64_t a, const SIMDVector<int64_t,simd_abi::avx> &b) {
    SIMDVector<int64_t,simd_abi::avx> out;
    out.value = _mm256_mul_epi64x(_mm256_set1_epi64x(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx> operator/(const SIMDVector<int64_t,simd_abi::avx> &a, const SIMDVector<int64_t,simd_abi::avx> &b) {
    SIMDVector<int64_t,simd_abi::avx> out;
    int64_t val[out.size()];   _mm256_storeu_si256((__m256i*)val, out.value);
    int64_t val_a[out.size()]; _mm256_storeu_si256((__m256i*)val_a, a.value);
    int64_t val_b[out.size()]; _mm256_storeu_si256((__m256i*)val_b, b.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = val_a[i] / val_b[i];
    }
    out.value = _mm256_loadu_si256((__m256i*)val);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx> operator/(const SIMDVector<int64_t,simd_abi::avx> &a, int64_t b) {
    SIMDVector<int64_t,simd_abi::avx> out;
    int64_t val[out.size()];   _mm256_storeu_si256((__m256i*)val, out.value);
    int64_t val_a[out.size()]; _mm256_storeu_si256((__m256i*)val_a, a.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = val_a[i] / b;
    }
    out.value = _mm256_loadu_si256((__m256i*)val);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::avx> operator/(int64_t a, const SIMDVector<int64_t,simd_abi::avx> &b) {
    SIMDVector<int64_t,simd_abi::avx> out;
    int64_t val[out.size()];   _mm256_storeu_si256((__m256i*)val, out.value);
    int64_t val_b[out.size()]; _mm256_storeu_si256((__m256i*)val_b, b.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = a / val_b[i];
    }
    out.value = _mm256_loadu_si256((__m256i*)val);
    return out;
}

#endif



// SSE VERSION
//-----------------------------------------------------------------------------------------------

#ifdef FASTOR_SSE2_IMPL

template<>
struct SIMDVector<int64_t,simd_abi::sse> {
    using value_type = __m128i;
    using scalar_value_type = int64_t;
    using abi_type = simd_abi::sse;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<int64_t,simd_abi::sse>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<int64_t,simd_abi::sse>>::value;}

    FASTOR_INLINE SIMDVector() : value(_mm_setzero_si128()) {}
    FASTOR_INLINE SIMDVector(int64_t num) {
        value = _mm_set_epi64x(num,num);
    }
    FASTOR_INLINE SIMDVector(__m128i regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const int64_t *data, bool Aligned=true) {
        if (Aligned)
            value =_mm_load_si128((__m128i*)data);
        else
            value = _mm_loadu_si128((__m128i*)data);
    }

    FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> operator=(int64_t num) {
        value = _mm_set_epi64x(num,num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> operator=(__m128i regi) {
        value = regi;
        return *this;
    }

    FASTOR_INLINE void load(const int64_t *data, bool Aligned=true) {
        if (Aligned)
            value =_mm_load_si128((__m128i*)data);
        else
            value = _mm_loadu_si128((__m128i*)data);
    }
    FASTOR_INLINE void store(int64_t *data, bool Aligned=true) const {
        if (Aligned)
            _mm_store_si128((__m128i*)data,value);
        else
            _mm_storeu_si128((__m128i*)data,value);
    }

    FASTOR_INLINE void aligned_load(const int64_t *data) {
        value =_mm_load_si128((__m128i*)data);
    }
    FASTOR_INLINE void aligned_store(int64_t *data) const {
        _mm_store_si128((__m128i*)data,value);
    }

    FASTOR_INLINE void mask_load(const scalar_value_type *a, uint8_t mask, bool Aligned=false) {
#ifdef FASTOR_HAS_AVX512_MASKS
        if (!Aligned)
            value = _mm_mask_loadu_epi64(value, mask, a);
        else
            value = _mm_mask_load_epi64(value, mask, a);
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
            _mm_mask_storeu_epi64(a, mask, value);
        else
            _mm_mask_store_epi64(a, mask, value);
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

    FASTOR_INLINE int64_t operator[](FASTOR_INDEX i) const {return reinterpret_cast<const int64_t*>(&value)[i];}
    FASTOR_INLINE int64_t operator()(FASTOR_INDEX i) const {return reinterpret_cast<const int64_t*>(&value)[i];}

    FASTOR_INLINE void set(int64_t num) {
        value = _mm_set_epi64x(num,num);
    }
    FASTOR_INLINE void set(int64_t num0, int64_t num1) {
        value = _mm_set_epi64x(num0,num1);
    }
    FASTOR_INLINE void set_sequential(int64_t num0) {
        value = _mm_set_epi64x(num0+1,num0);
    }

    // In-place operators
    FASTOR_INLINE void operator+=(int64_t num) {
        auto numb = _mm_set_epi64x(num,num);
        value = _mm_add_epi64(value,numb);
    }
    FASTOR_INLINE void operator+=(__m128i regi) {
        value = _mm_add_epi64(value,regi);
    }
    FASTOR_INLINE void operator+=(const SIMDVector<int64_t,simd_abi::sse> &a) {
        value = _mm_add_epi64(value,a.value);
    }

    FASTOR_INLINE void operator-=(int64_t num) {
        auto numb = _mm_set_epi64x(num,num);
        value = _mm_sub_epi64(value,numb);
    }
    FASTOR_INLINE void operator-=(__m128i regi) {
        value = _mm_sub_epi64(value,regi);
    }
    FASTOR_INLINE void operator-=(const SIMDVector<int64_t,simd_abi::sse> &a) {
        value = _mm_sub_epi64(value,a.value);
    }

    FASTOR_INLINE void operator*=(int64_t num) {
        auto numb = _mm_set_epi64x(num,num);
        value = _mm_mul_epi64(value,numb);
    }
    FASTOR_INLINE void operator*=(__m128i regi) {
        value = _mm_mul_epi64(value,regi);
    }
    FASTOR_INLINE void operator*=(const SIMDVector<int64_t,simd_abi::sse> &a) {
        value = _mm_mul_epi64(value,a.value);
    }

    FASTOR_INLINE void operator/=(int64_t num) {
        int64_t val[Size]; _mm_storeu_si128((__m128i*)val, value);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= num;
        }
        value = _mm_loadu_si128((__m128i*)val);
    }
    FASTOR_INLINE void operator/=(__m128i regi) {
        int64_t val[Size]; _mm_storeu_si128((__m128i*)val, value);
        int64_t val_num[Size]; _mm_storeu_si128((__m128i*)val_num, regi);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= val_num[i];
        }
        value = _mm_loadu_si128((__m128i*)val);
    }
    FASTOR_INLINE void operator/=(const SIMDVector<int64_t,simd_abi::sse> &a) {
        int64_t val[Size]; _mm_storeu_si128((__m128i*)val, value);
        int64_t val_a[Size]; _mm_storeu_si128((__m128i*)val_a, a.value);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            val[i] /= val_a[i];
        }
        value = _mm_loadu_si128((__m128i*)val);
    }

    FASTOR_INLINE int64_t minimum() {
        const int64_t *vals = reinterpret_cast<const int64_t*>(&value);
        int64_t quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            if (vals[i]<quan)
                quan = vals[i];
        return static_cast<int64_t>(quan);
    }
    FASTOR_INLINE int64_t maximum() {
        const int64_t *vals = reinterpret_cast<const int64_t*>(&value);
        int64_t quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            if (vals[i]>quan)
                quan = vals[i];
        return static_cast<int64_t>(quan);
    }
    FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> reverse() {
        return _mm_reverse_epi64(value);
    }

    FASTOR_INLINE int64_t sum() {
        const int64_t *vals = reinterpret_cast<const int64_t*>(&value);
        int64_t quan = 0;
        for (FASTOR_INDEX i=0; i<2; ++i)
            quan += vals[i];
        return static_cast<int64_t>(quan);
    }
    FASTOR_INLINE int64_t product() {
        const int64_t *vals = reinterpret_cast<const int64_t*>(&value);
        int64_t quan = 1;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            quan *= vals[i];
        return quan;
    }

    FASTOR_INLINE int64_t dot(const SIMDVector<int64_t,simd_abi::sse> &other) {
        const int64_t *vals0 = reinterpret_cast<const int64_t*>(&value);
        const int64_t *vals1 = reinterpret_cast<const int64_t*>(&other.value);
        int64_t quan = 0;
        for (FASTOR_INDEX i=0; i<2; ++i)
            quan += vals0[i]*vals1[i];
        return static_cast<int64_t>(quan);
    }

    __m128i value;
};

FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<int64_t,simd_abi::sse> a) {
    const int64_t *value = reinterpret_cast<const int64_t*>(&a.value);
    os << "[" << value[0] <<  " " << value[1] << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> operator+(const SIMDVector<int64_t,simd_abi::sse> &a, const SIMDVector<int64_t,simd_abi::sse> &b) {
    SIMDVector<int64_t,simd_abi::sse> out;
    out.value = _mm_add_epi64(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> operator+(const SIMDVector<int64_t,simd_abi::sse> &a, int64_t b) {
    SIMDVector<int64_t,simd_abi::sse> out;
    auto numb = _mm_set_epi64x(b,b);
    out.value = _mm_add_epi64(a.value,numb);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> operator+(int64_t a, const SIMDVector<int64_t,simd_abi::sse> &b) {
    SIMDVector<int64_t,simd_abi::sse> out;
    auto numb = _mm_set_epi64x(a,a);
    out.value = _mm_add_epi64(numb,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> operator+(const SIMDVector<int64_t,simd_abi::sse> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> operator-(const SIMDVector<int64_t,simd_abi::sse> &a, const SIMDVector<int64_t,simd_abi::sse> &b) {
    SIMDVector<int64_t,simd_abi::sse> out;
    out.value = _mm_sub_epi64(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> operator-(const SIMDVector<int64_t,simd_abi::sse> &a, int64_t b) {
    SIMDVector<int64_t,simd_abi::sse> out;
    auto numb = _mm_set_epi64x(b,b);
    out.value = _mm_sub_epi64(a.value,numb);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> operator-(int64_t a, const SIMDVector<int64_t,simd_abi::sse> &b) {
    SIMDVector<int64_t,simd_abi::sse> out;
    auto numb = _mm_set_epi64x(a,a);
    out.value = _mm_sub_epi64(numb,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> operator-(const SIMDVector<int64_t,simd_abi::sse> &b) {
    SIMDVector<int64_t,simd_abi::sse> out;
    out.value = _mm_castpd_si128(_mm_neg_pd(_mm_castsi128_pd(b.value)));
    return out;
}

FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> operator*(const SIMDVector<int64_t,simd_abi::sse> &a, const SIMDVector<int64_t,simd_abi::sse> &b) {
    SIMDVector<int64_t,simd_abi::sse> out;
    out.value = _mm_mul_epi64(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> operator*(const SIMDVector<int64_t,simd_abi::sse> &a, int64_t b) {
    SIMDVector<int64_t,simd_abi::sse> out;
    auto numb = _mm_set_epi64x(b,b);
    out.value = _mm_mul_epi64(a.value,numb);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> operator*(int64_t a, const SIMDVector<int64_t,simd_abi::sse> &b) {
    SIMDVector<int64_t,simd_abi::sse> out;
    auto numb = _mm_set_epi64x(a,a);
    out.value = _mm_mul_epi64(numb,b.value);
    return out;
}

FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> operator/(const SIMDVector<int64_t,simd_abi::sse> &a, const SIMDVector<int64_t,simd_abi::sse> &b) {
    SIMDVector<int64_t,simd_abi::sse> out;
    int64_t val[out.size()];   _mm_storeu_si128((__m128i*)val, out.value);
    int64_t val_a[out.size()]; _mm_storeu_si128((__m128i*)val_a, a.value);
    int64_t val_b[out.size()]; _mm_storeu_si128((__m128i*)val_b, b.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = val_a[i] / val_b[i];
    }
    out.value = _mm_loadu_si128((__m128i*)val);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> operator/(const SIMDVector<int64_t,simd_abi::sse> &a, int64_t b) {
    SIMDVector<int64_t,simd_abi::sse> out;
    int64_t val[out.size()];   _mm_storeu_si128((__m128i*)val, out.value);
    int64_t val_a[out.size()]; _mm_storeu_si128((__m128i*)val_a, a.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = val_a[i] / b;
    }
    out.value = _mm_loadu_si128((__m128i*)val);
    return out;
}
FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> operator/(int64_t a, const SIMDVector<int64_t,simd_abi::sse> &b) {
    SIMDVector<int64_t,simd_abi::sse> out;
    int64_t val[out.size()];   _mm_storeu_si128((__m128i*)val, out.value);
    int64_t val_b[out.size()]; _mm_storeu_si128((__m128i*)val_b, b.value);
    for (FASTOR_INDEX i=0; i<out.size(); ++i) {
        val[i] = a / val_b[i];
    }
    out.value = _mm_loadu_si128((__m128i*)val);
    return out;
}

FASTOR_INLINE SIMDVector<int64_t,simd_abi::sse> abs(const SIMDVector<int64_t,simd_abi::sse> &a) {
    SIMDVector<int64_t,simd_abi::sse> out;
#ifdef FASTOR_AVX512VL_IMPL
    return _mm_abs_epi64(a.value);
#elif defined (FASTOR_SSE4_2_IMPL)
    __m128i sign = _mm_cmpgt_epi64(_mm_setzero_si128(), a.value); // 0 > a
    __m128i inv = _mm_xor_si128(a.value, sign);                   // invert bits if negative
    return _mm_sub_epi64(inv, sign);                              // add 1
#else // SSE2
    __m128i signh = _mm_srai_epi32(a.value, 31);                  // sign in high dword
    __m128i sign = _mm_shuffle_epi32(signh, 0xF5);                // copy sign to low dword
    __m128i inv = _mm_xor_si128(a.value, sign);                   // invert bits if negative
    return _mm_sub_epi64(inv, sign);                              // add 1
#endif
}


#endif


} // end of namespace Fastor


#endif // SIMD_VECTOR_INT64_H
