#ifndef SIMD_VECTOR_COMPLEX_FLOAT_H
#define SIMD_VECTOR_COMPLEX_FLOAT_H

#include "Fastor/util/extended_algorithms.h"
#include "Fastor/simd_vector/extintrin.h"
#include "Fastor/simd_vector/simd_vector_base.h"
#include "Fastor/simd_vector/simd_vector_float.h"
#include <cmath>
#include <complex>

namespace Fastor {


// AVX512 VERSION
//------------------------------------------------------------------------------------------------------------

#ifdef FASTOR_AVX512F_IMPL

template <>
struct SIMDVector<std::complex<float>, simd_abi::avx512> {
    using vector_type = SIMDVector<std::complex<float>, simd_abi::avx512>;
    using value_type = __m512;
    using scalar_value_type = std::complex<float>;
    using abi_type = simd_abi::avx512;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<float,simd_abi::avx512>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<float,simd_abi::avx512>>::value;}

    FASTOR_INLINE SIMDVector() : value_r(_mm512_setzero_ps()), value_i(_mm512_setzero_ps()) {}
    FASTOR_INLINE SIMDVector(std::complex<float> num) {
        value_r = _mm512_set1_ps(num.real());
        value_i = _mm512_set1_ps(num.imag());
    }
    FASTOR_INLINE SIMDVector(value_type reg0, value_type reg1) : value_r(reg0), value_i(reg1) {}
    FASTOR_INLINE SIMDVector(const std::complex<float> *data, bool Aligned=true) {
        if (Aligned)
            complex_aligned_load(data);
        else
            complex_unaligned_load(data);
    }

    FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512> operator=(std::complex<float> num) {
        value_r = _mm512_set1_ps(num.real());
        value_i = _mm512_set1_ps(num.imag());
        return *this;
    }

    FASTOR_INLINE void load(const std::complex<float> *data, bool Aligned=true) {
        if (Aligned)
            complex_aligned_load(data);
        else
            complex_unaligned_load(data);
    }
    FASTOR_INLINE void store(std::complex<float> *data, bool Aligned=true) const {
        if (Aligned)
            complex_aligned_store(data);
        else
            complex_unaligned_store(data);
    }

    FASTOR_INLINE void mask_load(const scalar_value_type *data, uint16_t mask, bool Aligned=false) {
        if (!Aligned)
            complex_mask_unaligned_load(data,mask);
        else
            complex_mask_aligned_load(data,mask);
    }
    FASTOR_INLINE void mask_store(scalar_value_type *data, uint16_t mask, bool Aligned=false) const {
        if (!Aligned)
            complex_mask_unaligned_store(data,mask);
        else
            complex_mask_aligned_store(data,mask);
    }

    FASTOR_INLINE scalar_value_type operator[](FASTOR_INDEX i) const {
        return scalar_value_type(reinterpret_cast<const float*>(&value_r)[i],reinterpret_cast<const float*>(&value_i)[i]);
    }

    FASTOR_INLINE SIMDVector<float,simd_abi::avx512> real() const {
        return value_r;
    }
    FASTOR_INLINE SIMDVector<float,simd_abi::avx512> imag() const {
        return value_i;
    }

    FASTOR_INLINE void set(std::complex<float> num) {
        value_r = _mm512_set1_ps(num.real());
        value_i = _mm512_set1_ps(num.imag());
    }
    FASTOR_INLINE void set(scalar_value_type num0, scalar_value_type num1,
                           scalar_value_type num2, scalar_value_type num3,
                           scalar_value_type num4, scalar_value_type num5,
                           scalar_value_type num6, scalar_value_type num7,
                           scalar_value_type num8, scalar_value_type num9,
                           scalar_value_type num10, scalar_value_type num11,
                           scalar_value_type num12, scalar_value_type num13,
                           scalar_value_type num14, scalar_value_type num15) {
        const scalar_value_type tmp[Size] = {num0,num1,num2,num3,num4,num5,num6,num7,
                                                num8,num9,num10,num11,num12,num13,num14,num15};
        complex_unaligned_load(tmp);
    }

    // In-place operators
    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator+=(U num) {
        value_r = _mm512_add_ps(value_r,_mm512_set1_ps(num));
    }
    FASTOR_INLINE void operator+=(scalar_value_type num) {
        *this += vector_type(num);
    }
    FASTOR_INLINE void operator+=(const vector_type &a) {
        value_r = _mm512_add_ps(value_r,a.value_r);
        value_i = _mm512_add_ps(value_i,a.value_i);
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator-=(U num) {
        value_r = _mm512_sub_ps(value_r,_mm512_set1_ps(num));
    }
    FASTOR_INLINE void operator-=(scalar_value_type num) {
        *this -= vector_type(num);
    }
    FASTOR_INLINE void operator-=(const vector_type &a) {
        value_r = _mm512_sub_ps(value_r,a.value_r);
        value_i = _mm512_sub_ps(value_i,a.value_i);
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator*=(U num) {
        __m512 val = _mm512_set1_ps(num);
        value_r = _mm512_mul_ps(value_r,val);
        value_i = _mm512_mul_ps(value_i,val);
    }
    FASTOR_INLINE void operator*=(scalar_value_type num) {
        *this *= vector_type(num);
    }
    FASTOR_INLINE void operator*=(const vector_type &a) {
#ifdef FASTOR_FMA_IMPL
        __m512 tmp = _mm512_fmsub_ps(value_r,a.value_r,_mm512_mul_ps(value_i,a.value_i));
        value_i     = _mm512_fmadd_ps(value_r,a.value_i,_mm512_mul_ps(value_i,a.value_r));
#else
        __m512 tmp = _mm512_sub_ps(_mm512_mul_ps(value_r,a.value_r),_mm512_mul_ps(value_i,a.value_i));
        value_i     = _mm512_add_ps(_mm512_mul_ps(value_r,a.value_i),_mm512_mul_ps(value_i,a.value_r));
#endif
        value_r = tmp;
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator/=(U num) {
        __m512 val = _mm512_set1_ps(num);
        value_r = _mm512_div_ps(value_r,val);
        value_i = _mm512_div_ps(value_i,val);
    }
    FASTOR_INLINE void operator/=(scalar_value_type num) {
        *this /= vector_type(num);
    }
    FASTOR_INLINE void operator/=(const vector_type &a) {
        __m512 tmp = value_r;
#ifdef FASTOR_FMA_IMPL
        value_r     = _mm512_fmadd_ps(value_r  , a.value_r, _mm512_mul_ps(value_i,a.value_i));
        value_i     = _mm512_fmsub_ps(value_i  , a.value_r, _mm512_mul_ps(tmp,a.value_i));
        __m512 den  = _mm512_fmadd_ps(a.value_r, a.value_r, _mm512_mul_ps(a.value_i,a.value_i));
#else
        value_r     = _mm512_add_ps(_mm512_mul_ps(value_r  , a.value_r), _mm512_mul_ps(value_i,a.value_i));
        value_i     = _mm512_sub_ps(_mm512_mul_ps(value_i  , a.value_r), _mm512_mul_ps(tmp,a.value_i));
        __m512 den  = _mm512_add_ps(_mm512_mul_ps(a.value_r, a.value_r), _mm512_mul_ps(a.value_i,a.value_i));
#endif
        value_r     = _mm512_div_ps(value_r,den);
        value_i     = _mm512_div_ps(value_i,den);
    }
    // end of in-place operators

    FASTOR_INLINE scalar_value_type sum() const {
#ifdef FASTOR_HAS_AVX512_REDUCE_ADD
        return scalar_value_type(_mm512_reduce_add_ps(value_r),_mm512_reduce_add_ps(value_i));
#else
        __m256 lor  = _mm512_castps512_ps256(value_r);
        __m256 hir  = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(value_r),0x1));
        __m256 loi  = _mm512_castps512_ps256(value_i);
        __m256 hii  = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(value_i),0x1));
        return scalar_value_type(_mm256_sum_ps(_mm256_add_ps(lor,hir)), _mm256_sum_ps(_mm256_add_ps(loi,hii)));
#endif
    }
    FASTOR_INLINE scalar_value_type product() const {
        vector_type tmp(*this);
        scalar_value_type out(tmp[0]);
        for (FASTOR_INDEX i=1; i<Size; ++i) out *= tmp[i];
        return out;
    }
    FASTOR_INLINE vector_type reverse() const {
        vector_type out;
        out.value_r = _mm512_reverse_ps(value_r);
        out.value_i = _mm512_reverse_ps(value_i);
        return out;
    }
    /* Actual magnitude - Note that this is a vertical operation */
    FASTOR_INLINE SIMDVector<float,simd_abi::avx512> magnitude() const {
#ifdef FASTOR_FMA_IMPL
        return _mm512_sqrt_ps(_mm512_fmadd_ps(value_r,value_r,_mm512_mul_ps(value_i,value_i)));
#else
        return _mm512_sqrt_ps(_mm512_add_ps(_mm512_mul_ps(value_r,value_r),_mm512_mul_ps(value_i,value_i)));
#endif
    }
    /* STL compliant squared norm - Note that this is a vertical operation */
    FASTOR_INLINE SIMDVector<float,simd_abi::avx512> norm() const {
#ifdef FASTOR_FMA_IMPL
        return _mm512_fmadd_ps(value_r,value_r,_mm512_mul_ps(value_i,value_i));
#else
        return _mm512_add_ps(_mm512_mul_ps(value_r,value_r),_mm512_mul_ps(value_i,value_i));
#endif
    }
    // Magnitude based minimum
    FASTOR_INLINE scalar_value_type minimum() const;
    // Magnitude based maximum
    FASTOR_INLINE scalar_value_type maximum() const;

    FASTOR_INLINE scalar_value_type dot(const vector_type &other) const {
        vector_type out(*this);
        out *= other;
        return out.sum();
    }

    value_type value_r;
    value_type value_i;

protected:
    FASTOR_INLINE void complex_aligned_load(const std::complex<float> *data) {
        __m512 lo = _mm512_load_ps(reinterpret_cast<const float*>(data  ));
        __m512 hi = _mm512_load_ps(reinterpret_cast<const float*>(data+8));
        arrange_from_load(value_r, value_i, lo, hi);
    }
    FASTOR_INLINE void complex_unaligned_load(const std::complex<float> *data) {
        __m512 lo = _mm512_loadu_ps(reinterpret_cast<const float*>(data  ));
        __m512 hi = _mm512_loadu_ps(reinterpret_cast<const float*>(data+8));
        arrange_from_load(value_r, value_i, lo, hi);
    }

    FASTOR_INLINE void complex_aligned_store(std::complex<float> *data) const {
        __m512 lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        _mm512_store_ps(reinterpret_cast<float*>(data  ), lo);
        _mm512_store_ps(reinterpret_cast<float*>(data+8), hi);
    }
    FASTOR_INLINE void complex_unaligned_store(std::complex<float> *data) const {
        __m512 lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        _mm512_storeu_ps(reinterpret_cast<float*>(data  ), lo);
        _mm512_storeu_ps(reinterpret_cast<float*>(data+8), hi);
    }

    FASTOR_INLINE void complex_mask_aligned_load(const scalar_value_type *data, uint16_t mask) {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m512 lo, hi;
        uint16_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        lo = _mm512_mask_load_ps(lo, mask0, reinterpret_cast<const float*>(data  ));
        hi = _mm512_mask_load_ps(hi, mask1, reinterpret_cast<const float*>(data+8));
        arrange_from_load(value_r, value_i, lo, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        value_r = _mm512_setzero_ps();
        value_i = _mm512_setzero_ps();
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                ((float*)&value_r)[Size - i - 1] = data[Size - i - 1].real();
                ((float*)&value_i)[Size - i - 1] = data[Size - i - 1].imag();
            }
        }
#endif
    }
    FASTOR_INLINE void complex_mask_unaligned_load(const scalar_value_type *data, uint16_t mask) {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m512 lo, hi;
        uint16_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        lo = _mm512_mask_loadu_ps(lo, mask0, reinterpret_cast<const float*>(data  ));
        hi = _mm512_mask_loadu_ps(hi, mask1, reinterpret_cast<const float*>(data+8));
        arrange_from_load(value_r, value_i, lo, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        value_r = _mm512_setzero_ps();
        value_i = _mm512_setzero_ps();
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                ((float*)&value_r)[Size - i - 1] = data[Size - i - 1].real();
                ((float*)&value_i)[Size - i - 1] = data[Size - i - 1].imag();
            }
        }
#endif
    }

    FASTOR_INLINE void complex_mask_aligned_store(scalar_value_type *data, uint16_t mask) const {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m512 lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        uint16_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        _mm512_mask_store_ps(reinterpret_cast<float*>(data  ), mask0, lo);
        _mm512_mask_store_ps(reinterpret_cast<float*>(data+8), mask1, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                float _real = ((const float*)&value_r)[Size - i - 1];
                float _imag = ((const float*)&value_i)[Size - i - 1];
                data[Size - i - 1] = std::complex<float>(_real,_imag);
            }
            else {
                data[Size - i - 1] = std::complex<float>(0,0);
            }
        }
#endif
    }
    FASTOR_INLINE void complex_mask_unaligned_store(scalar_value_type *data, uint16_t mask) const {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m512 lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        uint16_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        _mm512_mask_storeu_ps(reinterpret_cast<float*>(data  ), mask0, lo);
        _mm512_mask_storeu_ps(reinterpret_cast<float*>(data+8), mask1, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                float _real = ((const float*)&value_r)[Size - i - 1];
                float _imag = ((const float*)&value_i)[Size - i - 1];
                data[Size - i - 1] = std::complex<float>(_real,_imag);
            }
            else {
                data[Size - i - 1] = std::complex<float>(0,0);
            }
        }
#endif
    }
};

FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<std::complex<float>,simd_abi::avx512> a) {
    // ICC crashes without a copy
    const __m512 vr = a.value_r;
    const __m512 vi = a.value_i;
    const float* value_r = reinterpret_cast<const float*>(&vr);
    const float* value_i = reinterpret_cast<const float*>(&vi);
    os << "[" << value_r[0] <<  signum_string(value_i[0]) << std::abs(value_i[0]) << "j, "
              << value_r[1] <<  signum_string(value_i[1]) << std::abs(value_i[1]) << "j, "
              << value_r[2] <<  signum_string(value_i[2]) << std::abs(value_i[2]) << "j, "
              << value_r[3] <<  signum_string(value_i[3]) << std::abs(value_i[3]) << "j, "
              << value_r[4] <<  signum_string(value_i[4]) << std::abs(value_i[4]) << "j, "
              << value_r[5] <<  signum_string(value_i[5]) << std::abs(value_i[5]) << "j, "
              << value_r[6] <<  signum_string(value_i[6]) << std::abs(value_i[6]) << "j, "
              << value_r[7] <<  signum_string(value_i[7]) << std::abs(value_i[7]) << "j, "
              << value_r[8] <<  signum_string(value_i[8]) << std::abs(value_i[8]) << "j, "
              << value_r[9] <<  signum_string(value_i[9]) << std::abs(value_i[9]) << "j, "
              << value_r[10] <<  signum_string(value_i[10]) << std::abs(value_i[10]) << "j, "
              << value_r[11] <<  signum_string(value_i[11]) << std::abs(value_i[11]) << "j, "
              << value_r[12] <<  signum_string(value_i[12]) << std::abs(value_i[12]) << "j, "
              << value_r[13] <<  signum_string(value_i[13]) << std::abs(value_i[13]) << "j, "
              << value_r[14] <<  signum_string(value_i[14]) << std::abs(value_i[14]) << "j, "
              << value_r[15] <<  signum_string(value_i[15]) << std::abs(value_i[15]) << "j" << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator+(const SIMDVector<std::complex<float>,simd_abi::avx512> &a, const SIMDVector<std::complex<float>,simd_abi::avx512> &b) {
    SIMDVector<std::complex<float>,simd_abi::avx512> out;
    out.value_r = _mm512_add_ps(a.value_r,b.value_r);
    out.value_i = _mm512_add_ps(a.value_i,b.value_i);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator+(const SIMDVector<std::complex<float>,simd_abi::avx512> &a, std::complex<float> b) {
    return a + SIMDVector<std::complex<float>,simd_abi::avx512>(b);
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator+(std::complex<float> a, const SIMDVector<std::complex<float>,simd_abi::avx512> &b) {
    return SIMDVector<std::complex<float>,simd_abi::avx512>(a) + b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator+(const SIMDVector<std::complex<float>,simd_abi::avx512> &a, U b) {
    SIMDVector<std::complex<float>,simd_abi::avx512> out(a);
    out.value_r = _mm512_add_ps(a.value_r,_mm512_set1_ps(b));
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator+(U a, const SIMDVector<std::complex<float>,simd_abi::avx512> &b) {
    SIMDVector<std::complex<float>,simd_abi::avx512> out(b);
    out.value_r = _mm512_add_ps(_mm512_set1_ps(a), b.value_r);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator+(const SIMDVector<std::complex<float>,simd_abi::avx512> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator-(const SIMDVector<std::complex<float>,simd_abi::avx512> &a, const SIMDVector<std::complex<float>,simd_abi::avx512> &b) {
    SIMDVector<std::complex<float>,simd_abi::avx512> out;
    out.value_r = _mm512_sub_ps(a.value_r,b.value_r);
    out.value_i = _mm512_sub_ps(a.value_i,b.value_i);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator-(const SIMDVector<std::complex<float>,simd_abi::avx512> &a, std::complex<float> b) {
    return a - SIMDVector<std::complex<float>,simd_abi::avx512>(b);
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator-(std::complex<float> a, const SIMDVector<std::complex<float>,simd_abi::avx512> &b) {
    return SIMDVector<std::complex<float>,simd_abi::avx512>(a) - b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator-(const SIMDVector<std::complex<float>,simd_abi::avx512> &a, U b) {
    SIMDVector<std::complex<float>,simd_abi::avx512> out(a);
    out.value_r = _mm512_sub_ps(a.value_r,_mm512_set1_ps(b));
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator-(U a, const SIMDVector<std::complex<float>,simd_abi::avx512> &b) {
    return SIMDVector<std::complex<float>,simd_abi::avx512>(std::complex<float>(a,0)) - b;
}
/* This is negation and not complex conjugate  */
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator-(const SIMDVector<std::complex<float>,simd_abi::avx512> &a) {
    SIMDVector<std::complex<float>,simd_abi::avx512> out;
    out.value_r = _mm512_neg_ps(a.value_r);
    out.value_i = _mm512_neg_ps(a.value_i);
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator*(const SIMDVector<std::complex<float>,simd_abi::avx512> &a, const SIMDVector<std::complex<float>,simd_abi::avx512> &b) {
    SIMDVector<std::complex<float>,simd_abi::avx512> out;
#ifdef FASTOR_FMA_IMPL
    out.value_r = _mm512_fmsub_ps(a.value_r,b.value_r,_mm512_mul_ps(a.value_i,b.value_i));
    out.value_i = _mm512_fmadd_ps(a.value_r,b.value_i,_mm512_mul_ps(a.value_i,b.value_r));
#else
    out.value_r = _mm512_sub_ps(_mm512_mul_ps(a.value_r,b.value_r),_mm512_mul_ps(a.value_i,b.value_i));
    out.value_i = _mm512_add_ps(_mm512_mul_ps(a.value_r,b.value_i),_mm512_mul_ps(a.value_i,b.value_r));
#endif
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator*(const SIMDVector<std::complex<float>,simd_abi::avx512> &a, std::complex<float> b) {
    return a * SIMDVector<std::complex<float>,simd_abi::avx512>(b);
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator*(std::complex<float> a, const SIMDVector<std::complex<float>,simd_abi::avx512> &b) {
    return SIMDVector<std::complex<float>,simd_abi::avx512>(a) * b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator*(const SIMDVector<std::complex<float>,simd_abi::avx512> &a, U b) {
    SIMDVector<std::complex<float>,simd_abi::avx512> out;
    __m512 val = _mm512_set1_ps(b);
    out.value_r = _mm512_mul_ps(a.value_r,val);
    out.value_i = _mm512_mul_ps(a.value_i,val);
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator*(U a, const SIMDVector<std::complex<float>,simd_abi::avx512> &b) {
    SIMDVector<std::complex<float>,simd_abi::avx512> out;
    __m512 val = _mm512_set1_ps(a);
    out.value_r = _mm512_mul_ps(val,b.value_r);
    out.value_i = _mm512_mul_ps(val,b.value_i);
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator/(const SIMDVector<std::complex<float>,simd_abi::avx512> &a, const SIMDVector<std::complex<float>,simd_abi::avx512> &b) {
    SIMDVector<std::complex<float>,simd_abi::avx512> out;
#ifdef FASTOR_FMA_IMPL
    out.value_r = _mm512_fmadd_ps(a.value_r,b.value_r,_mm512_mul_ps(a.value_i,b.value_i));
    out.value_i = _mm512_fmsub_ps(a.value_i,b.value_r,_mm512_mul_ps(a.value_r,b.value_i));
    __m512 den = _mm512_fmadd_ps(b.value_r,b.value_r,_mm512_mul_ps(b.value_i,b.value_i));
#else
    out.value_r = _mm512_add_ps(_mm512_mul_ps(a.value_r,b.value_r),_mm512_mul_ps(a.value_i,b.value_i));
    out.value_i = _mm512_sub_ps(_mm512_mul_ps(a.value_i,b.value_r),_mm512_mul_ps(a.value_r,b.value_i));
    __m512 den = _mm512_add_ps(_mm512_mul_ps(b.value_r,b.value_r),_mm512_mul_ps(b.value_i,b.value_i));
#endif
    out.value_r = _mm512_div_ps(out.value_r,den);
    out.value_i = _mm512_div_ps(out.value_i,den);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator/(const SIMDVector<std::complex<float>,simd_abi::avx512> &a, std::complex<float> b) {
    return a / SIMDVector<std::complex<float>,simd_abi::avx512>(b);
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator/(std::complex<float> a, const SIMDVector<std::complex<float>,simd_abi::avx512> &b) {
    return SIMDVector<std::complex<float>,simd_abi::avx512>(a) / b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator/(const SIMDVector<std::complex<float>,simd_abi::avx512> &a, U b) {
    SIMDVector<std::complex<float>,simd_abi::avx512> out;
    __m512 val = _mm512_set1_ps(b);
    out.value_r = _mm512_div_ps(a.value_r,val);
    out.value_i = _mm512_div_ps(a.value_i,val);
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
operator/(U a, const SIMDVector<std::complex<float>,simd_abi::avx512> &b) {
    return SIMDVector<std::complex<float>,simd_abi::avx512>(std::complex<float>(a,0)) / b;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
rcp(const SIMDVector<std::complex<float>,simd_abi::avx512> &a) {
    SIMDVector<std::complex<float>,simd_abi::avx512> out;
#ifdef FASTOR_FMA_IMPL
    __m512 den = _mm512_fmadd_ps(a.value_r,a.value_r,_mm512_mul_ps(a.value_i,a.value_i));
#else
    __m512 den = _mm512_add_ps(_mm512_mul_ps(a.value_r,a.value_r),_mm512_mul_ps(a.value_i,a.value_i));
#endif
    out.value_r = _mm512_div_ps(out.value_r,den);
    out.value_i = _mm512_neg_ps(_mm512_div_ps(out.value_i,den));
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
sqrt(const SIMDVector<std::complex<float>,simd_abi::avx512> &a) = delete;

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
rsqrt(const SIMDVector<std::complex<float>,simd_abi::avx512> &a) = delete;

/* This intentionally return a complex vector */
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
abs(const SIMDVector<std::complex<float>,simd_abi::avx512> &a) {
    SIMDVector<std::complex<float>,simd_abi::avx512> out;
    out.value_r = a.magnitude().value;
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
conj(const SIMDVector<std::complex<float>,simd_abi::avx512> &a) {
    SIMDVector<std::complex<float>,simd_abi::avx512> out(a);
    out.value_i = _mm512_neg_ps(out.value_i);
    return out;
}

/* Argument or phase angle */
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx512>
arg(const SIMDVector<std::complex<float>,simd_abi::avx512> &a) {
    SIMDVector<std::complex<float>,simd_abi::avx512> out(a);
    for (FASTOR_INDEX i=0UL; i<16UL; ++i) {
       ((float*)&out.value_r)[i] = std::atan2(((float*)&a.value_i)[i],((float*)&a.value_r)[i]);
    }
    return out;
}
//------------------------------------------------------------------------------------------------------------

#endif



// AVX VERSION
//------------------------------------------------------------------------------------------------------------

#ifdef FASTOR_AVX_IMPL

template <>
struct SIMDVector<std::complex<float>, simd_abi::avx> {
    using vector_type = SIMDVector<std::complex<float>, simd_abi::avx>;
    using value_type = __m256;
    using scalar_value_type = std::complex<float>;
    using abi_type = simd_abi::avx;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<float,simd_abi::avx>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<float,simd_abi::avx>>::value;}

    FASTOR_INLINE SIMDVector() : value_r(_mm256_setzero_ps()), value_i(_mm256_setzero_ps()) {}
    FASTOR_INLINE SIMDVector(std::complex<float> num) {
        value_r = _mm256_set1_ps(num.real());
        value_i = _mm256_set1_ps(num.imag());
    }
    FASTOR_INLINE SIMDVector(value_type reg0, value_type reg1) : value_r(reg0), value_i(reg1) {}
    FASTOR_INLINE SIMDVector(const std::complex<float> *data, bool Aligned=true) {
        if (Aligned)
            complex_aligned_load(data);
        else
            complex_unaligned_load(data);
    }

    FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx> operator=(std::complex<float> num) {
        value_r = _mm256_set1_ps(num.real());
        value_i = _mm256_set1_ps(num.imag());
        return *this;
    }

    FASTOR_INLINE void load(const std::complex<float> *data, bool Aligned=true) {
        if (Aligned)
            complex_aligned_load(data);
        else
            complex_unaligned_load(data);
    }
    FASTOR_INLINE void store(std::complex<float> *data, bool Aligned=true) const {
        if (Aligned)
            complex_aligned_store(data);
        else
            complex_unaligned_store(data);
    }

    FASTOR_INLINE void mask_load(const scalar_value_type *data, uint8_t mask, bool Aligned=false) {
        if (!Aligned)
            complex_mask_unaligned_load(data,mask);
        else
            complex_mask_aligned_load(data,mask);
    }
    FASTOR_INLINE void mask_store(scalar_value_type *data, uint8_t mask, bool Aligned=false) const {
        if (!Aligned)
            complex_mask_unaligned_store(data,mask);
        else
            complex_mask_aligned_store(data,mask);
    }

    FASTOR_INLINE scalar_value_type operator[](FASTOR_INDEX i) const {
        if      (i == 0) { return scalar_value_type(_mm256_get0_ps(value_r), _mm256_get0_ps(value_i)); }
        else if (i == 1) { return scalar_value_type(_mm256_get1_ps(value_r), _mm256_get1_ps(value_i)); }
        else if (i == 2) { return scalar_value_type(_mm256_get2_ps(value_r), _mm256_get2_ps(value_i)); }
        else if (i == 3) { return scalar_value_type(_mm256_get3_ps(value_r), _mm256_get3_ps(value_i)); }
        else if (i == 4) { return scalar_value_type(_mm256_get4_ps(value_r), _mm256_get4_ps(value_i)); }
        else if (i == 5) { return scalar_value_type(_mm256_get5_ps(value_r), _mm256_get5_ps(value_i)); }
        else if (i == 6) { return scalar_value_type(_mm256_get6_ps(value_r), _mm256_get6_ps(value_i)); }
        else             { return scalar_value_type(_mm256_get7_ps(value_r), _mm256_get7_ps(value_i)); }
    }

    FASTOR_INLINE SIMDVector<float,simd_abi::avx> real() const {
        return value_r;
    }
    FASTOR_INLINE SIMDVector<float,simd_abi::avx> imag() const {
        return value_i;
    }

    FASTOR_INLINE void set(std::complex<float> num) {
        value_r = _mm256_set1_ps(num.real());
        value_i = _mm256_set1_ps(num.imag());
    }
    FASTOR_INLINE void set(scalar_value_type num0, scalar_value_type num1,
                           scalar_value_type num2, scalar_value_type num3,
                           scalar_value_type num4, scalar_value_type num5,
                           scalar_value_type num6, scalar_value_type num7) {
        const scalar_value_type tmp[Size] = {num0,num1,num2,num3,num4,num5,num6,num7};
        complex_unaligned_load(tmp);
    }

    // In-place operators
    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator+=(U num) {
        value_r = _mm256_add_ps(value_r,_mm256_set1_ps(num));
    }
    FASTOR_INLINE void operator+=(scalar_value_type num) {
        *this += vector_type(num);
    }
    FASTOR_INLINE void operator+=(const vector_type &a) {
        value_r = _mm256_add_ps(value_r,a.value_r);
        value_i = _mm256_add_ps(value_i,a.value_i);
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator-=(U num) {
        value_r = _mm256_sub_ps(value_r,_mm256_set1_ps(num));
    }
    FASTOR_INLINE void operator-=(scalar_value_type num) {
        *this -= vector_type(num);
    }
    FASTOR_INLINE void operator-=(const vector_type &a) {
        value_r = _mm256_sub_ps(value_r,a.value_r);
        value_i = _mm256_sub_ps(value_i,a.value_i);
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator*=(U num) {
        __m256 val = _mm256_set1_ps(num);
        value_r = _mm256_mul_ps(value_r,val);
        value_i = _mm256_mul_ps(value_i,val);
    }
    FASTOR_INLINE void operator*=(scalar_value_type num) {
        *this *= vector_type(num);
    }
    FASTOR_INLINE void operator*=(const vector_type &a) {
#ifdef FASTOR_FMA_IMPL
        __m256 tmp = _mm256_fmsub_ps(value_r,a.value_r,_mm256_mul_ps(value_i,a.value_i));
        value_i     = _mm256_fmadd_ps(value_r,a.value_i,_mm256_mul_ps(value_i,a.value_r));
#else
        __m256 tmp = _mm256_sub_ps(_mm256_mul_ps(value_r,a.value_r),_mm256_mul_ps(value_i,a.value_i));
        value_i     = _mm256_add_ps(_mm256_mul_ps(value_r,a.value_i),_mm256_mul_ps(value_i,a.value_r));
#endif
        value_r = tmp;
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator/=(U num) {
        __m256 val = _mm256_set1_ps(num);
        value_r = _mm256_div_ps(value_r,val);
        value_i = _mm256_div_ps(value_i,val);
    }
    FASTOR_INLINE void operator/=(scalar_value_type num) {
        *this /= vector_type(num);
    }
    FASTOR_INLINE void operator/=(const vector_type &a) {
        __m256 tmp = value_r;
#ifdef FASTOR_FMA_IMPL
        value_r     = _mm256_fmadd_ps(value_r  , a.value_r, _mm256_mul_ps(value_i,a.value_i));
        value_i     = _mm256_fmsub_ps(value_i  , a.value_r, _mm256_mul_ps(tmp,a.value_i));
        __m256 den = _mm256_fmadd_ps(a.value_r, a.value_r, _mm256_mul_ps(a.value_i,a.value_i));
#else
        value_r     = _mm256_add_ps(_mm256_mul_ps(value_r  , a.value_r), _mm256_mul_ps(value_i,a.value_i));
        value_i     = _mm256_sub_ps(_mm256_mul_ps(value_i  , a.value_r), _mm256_mul_ps(tmp,a.value_i));
        __m256 den = _mm256_add_ps(_mm256_mul_ps(a.value_r, a.value_r), _mm256_mul_ps(a.value_i,a.value_i));
#endif
        value_r     = _mm256_div_ps(value_r,den);
        value_i     = _mm256_div_ps(value_i,den);
    }
    // end of in-place operators

    FASTOR_INLINE scalar_value_type sum() const {
        return scalar_value_type(_mm256_sum_ps(value_r),_mm256_sum_ps(value_i));
    }
    FASTOR_INLINE scalar_value_type product() const {
        vector_type tmp(*this);
        return tmp[0]*tmp[1]*tmp[2]*tmp[3]*tmp[4]*tmp[5]*tmp[6]*tmp[7];
    }
    FASTOR_INLINE vector_type reverse() const {
        vector_type out;
        out.value_r = _mm256_reverse_ps(value_r);
        out.value_i = _mm256_reverse_ps(value_i);
        return out;
    }
    /* Actual magnitude - Note that this is a vertical operation */
    FASTOR_INLINE SIMDVector<float,simd_abi::avx> magnitude() const {
#ifdef FASTOR_FMA_IMPL
        return _mm256_sqrt_ps(_mm256_fmadd_ps(value_r,value_r,_mm256_mul_ps(value_i,value_i)));
#else
        return _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(value_r,value_r),_mm256_mul_ps(value_i,value_i)));
#endif
    }
    /* STL compliant squared norm - Note that this is a vertical operation */
    FASTOR_INLINE SIMDVector<float,simd_abi::avx> norm() const {
#ifdef FASTOR_FMA_IMPL
        return _mm256_fmadd_ps(value_r,value_r,_mm256_mul_ps(value_i,value_i));
#else
        return _mm256_add_ps(_mm256_mul_ps(value_r,value_r),_mm256_mul_ps(value_i,value_i));
#endif
    }
    // Magnitude based minimum
    FASTOR_INLINE scalar_value_type minimum() const;
    // Magnitude based maximum
    FASTOR_INLINE scalar_value_type maximum() const;

    FASTOR_INLINE scalar_value_type dot(const vector_type &other) const {
        vector_type out(*this);
        out *= other;
        return out.sum();
    }

    value_type value_r;
    value_type value_i;

protected:
    FASTOR_INLINE void complex_aligned_load(const std::complex<float> *data) {
        __m256 lo = _mm256_load_ps(reinterpret_cast<const float*>(data  ));
        __m256 hi = _mm256_load_ps(reinterpret_cast<const float*>(data+4));
        arrange_from_load(value_r, value_i, lo, hi);
    }
    FASTOR_INLINE void complex_unaligned_load(const std::complex<float> *data) {
        __m256 lo = _mm256_loadu_ps(reinterpret_cast<const float*>(data  ));
        __m256 hi = _mm256_loadu_ps(reinterpret_cast<const float*>(data+4));
        arrange_from_load(value_r, value_i, lo, hi);
    }

    FASTOR_INLINE void complex_aligned_store(std::complex<float> *data) const {
        __m256 lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        _mm256_store_ps(reinterpret_cast<float*>(data  ), lo);
        _mm256_store_ps(reinterpret_cast<float*>(data+4), hi);
    }
    FASTOR_INLINE void complex_unaligned_store(std::complex<float> *data) const {
        __m256 lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        _mm256_storeu_ps(reinterpret_cast<float*>(data  ), lo);
        _mm256_storeu_ps(reinterpret_cast<float*>(data+4), hi);
    }

    FASTOR_INLINE void complex_mask_aligned_load(const scalar_value_type *data, uint8_t mask) {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m256 lo, hi;
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        lo = _mm256_mask_load_ps(lo, mask0, reinterpret_cast<const float*>(data  ));
        hi = _mm256_mask_load_ps(hi, mask1, reinterpret_cast<const float*>(data+4));
        arrange_from_load(value_r, value_i, lo, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        value_r = _mm256_setzero_ps();
        value_i = _mm256_setzero_ps();
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                ((float*)&value_r)[Size - i - 1] = data[Size - i - 1].real();
                ((float*)&value_i)[Size - i - 1] = data[Size - i - 1].imag();
            }
        }
#endif
    }
    FASTOR_INLINE void complex_mask_unaligned_load(const scalar_value_type *data, uint8_t mask) {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m256 lo, hi;
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        lo = _mm256_mask_loadu_ps(lo, mask0, reinterpret_cast<const float*>(data  ));
        hi = _mm256_mask_loadu_ps(hi, mask1, reinterpret_cast<const float*>(data+4));
        arrange_from_load(value_r, value_i, lo, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        value_r = _mm256_setzero_ps();
        value_i = _mm256_setzero_ps();
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                ((float*)&value_r)[Size - i - 1] = data[Size - i - 1].real();
                ((float*)&value_i)[Size - i - 1] = data[Size - i - 1].imag();
            }
        }
#endif
    }

    FASTOR_INLINE void complex_mask_aligned_store(scalar_value_type *data, uint8_t mask) const {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m256 lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        _mm256_mask_store_ps(reinterpret_cast<float*>(data  ), mask0, lo);
        _mm256_mask_store_ps(reinterpret_cast<float*>(data+4), mask1, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                float _real = ((const float*)&value_r)[Size - i - 1];
                float _imag = ((const float*)&value_i)[Size - i - 1];
                data[Size - i - 1] = std::complex<float>(_real,_imag);
            }
            else {
                data[Size - i - 1] = std::complex<float>(0,0);
            }
        }
#endif
    }
    FASTOR_INLINE void complex_mask_unaligned_store(scalar_value_type *data, uint8_t mask) const {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m256 lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        _mm256_mask_storeu_ps(reinterpret_cast<float*>(data  ), mask0, lo);
        _mm256_mask_storeu_ps(reinterpret_cast<float*>(data+4), mask1, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                float _real = ((const float*)&value_r)[Size - i - 1];
                float _imag = ((const float*)&value_i)[Size - i - 1];
                data[Size - i - 1] = std::complex<float>(_real,_imag);
            }
            else {
                data[Size - i - 1] = std::complex<float>(0,0);
            }
        }
#endif
    }
};

FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<std::complex<float>,simd_abi::avx> a) {
    // ICC crashes without a copy
    const __m256 vr = a.value_r;
    const __m256 vi = a.value_i;
    const float* value_r = reinterpret_cast<const float*>(&vr);
    const float* value_i = reinterpret_cast<const float*>(&vi);
    os << "[" << value_r[0] <<  signum_string(value_i[0]) << std::abs(value_i[0]) << "j, "
              << value_r[1] <<  signum_string(value_i[1]) << std::abs(value_i[1]) << "j, "
              << value_r[2] <<  signum_string(value_i[2]) << std::abs(value_i[2]) << "j, "
              << value_r[3] <<  signum_string(value_i[3]) << std::abs(value_i[3]) << "j, "
              << value_r[4] <<  signum_string(value_i[4]) << std::abs(value_i[4]) << "j, "
              << value_r[5] <<  signum_string(value_i[5]) << std::abs(value_i[5]) << "j, "
              << value_r[6] <<  signum_string(value_i[6]) << std::abs(value_i[6]) << "j, "
              << value_r[7] <<  signum_string(value_i[7]) << std::abs(value_i[7]) << "j" << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator+(const SIMDVector<std::complex<float>,simd_abi::avx> &a, const SIMDVector<std::complex<float>,simd_abi::avx> &b) {
    SIMDVector<std::complex<float>,simd_abi::avx> out;
    out.value_r = _mm256_add_ps(a.value_r,b.value_r);
    out.value_i = _mm256_add_ps(a.value_i,b.value_i);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator+(const SIMDVector<std::complex<float>,simd_abi::avx> &a, std::complex<float> b) {
    return a + SIMDVector<std::complex<float>,simd_abi::avx>(b);
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator+(std::complex<float> a, const SIMDVector<std::complex<float>,simd_abi::avx> &b) {
    return SIMDVector<std::complex<float>,simd_abi::avx>(a) + b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator+(const SIMDVector<std::complex<float>,simd_abi::avx> &a, U b) {
    SIMDVector<std::complex<float>,simd_abi::avx> out(a);
    out.value_r = _mm256_add_ps(a.value_r,_mm256_set1_ps(b));
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator+(U a, const SIMDVector<std::complex<float>,simd_abi::avx> &b) {
    SIMDVector<std::complex<float>,simd_abi::avx> out(b);
    out.value_r = _mm256_add_ps(_mm256_set1_ps(a), b.value_r);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator+(const SIMDVector<std::complex<float>,simd_abi::avx> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator-(const SIMDVector<std::complex<float>,simd_abi::avx> &a, const SIMDVector<std::complex<float>,simd_abi::avx> &b) {
    SIMDVector<std::complex<float>,simd_abi::avx> out;
    out.value_r = _mm256_sub_ps(a.value_r,b.value_r);
    out.value_i = _mm256_sub_ps(a.value_i,b.value_i);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator-(const SIMDVector<std::complex<float>,simd_abi::avx> &a, std::complex<float> b) {
    return a - SIMDVector<std::complex<float>,simd_abi::avx>(b);
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator-(std::complex<float> a, const SIMDVector<std::complex<float>,simd_abi::avx> &b) {
    return SIMDVector<std::complex<float>,simd_abi::avx>(a) - b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator-(const SIMDVector<std::complex<float>,simd_abi::avx> &a, U b) {
    SIMDVector<std::complex<float>,simd_abi::avx> out(a);
    out.value_r = _mm256_sub_ps(a.value_r,_mm256_set1_ps(b));
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator-(U a, const SIMDVector<std::complex<float>,simd_abi::avx> &b) {
    return SIMDVector<std::complex<float>,simd_abi::avx>(std::complex<float>(a,0)) - b;
}
/* This is negation and not complex conjugate  */
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator-(const SIMDVector<std::complex<float>,simd_abi::avx> &a) {
    SIMDVector<std::complex<float>,simd_abi::avx> out;
    out.value_r = _mm256_neg_ps(a.value_r);
    out.value_i = _mm256_neg_ps(a.value_i);
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator*(const SIMDVector<std::complex<float>,simd_abi::avx> &a, const SIMDVector<std::complex<float>,simd_abi::avx> &b) {
    SIMDVector<std::complex<float>,simd_abi::avx> out;
#ifdef FASTOR_FMA_IMPL
    out.value_r = _mm256_fmsub_ps(a.value_r,b.value_r,_mm256_mul_ps(a.value_i,b.value_i));
    out.value_i = _mm256_fmadd_ps(a.value_r,b.value_i,_mm256_mul_ps(a.value_i,b.value_r));
#else
    out.value_r = _mm256_sub_ps(_mm256_mul_ps(a.value_r,b.value_r),_mm256_mul_ps(a.value_i,b.value_i));
    out.value_i = _mm256_add_ps(_mm256_mul_ps(a.value_r,b.value_i),_mm256_mul_ps(a.value_i,b.value_r));
#endif
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator*(const SIMDVector<std::complex<float>,simd_abi::avx> &a, std::complex<float> b) {
    return a * SIMDVector<std::complex<float>,simd_abi::avx>(b);
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator*(std::complex<float> a, const SIMDVector<std::complex<float>,simd_abi::avx> &b) {
    return SIMDVector<std::complex<float>,simd_abi::avx>(a) * b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator*(const SIMDVector<std::complex<float>,simd_abi::avx> &a, U b) {
    SIMDVector<std::complex<float>,simd_abi::avx> out;
    __m256 val = _mm256_set1_ps(b);
    out.value_r = _mm256_mul_ps(a.value_r,val);
    out.value_i = _mm256_mul_ps(a.value_i,val);
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator*(U a, const SIMDVector<std::complex<float>,simd_abi::avx> &b) {
    SIMDVector<std::complex<float>,simd_abi::avx> out;
    __m256 val = _mm256_set1_ps(a);
    out.value_r = _mm256_mul_ps(val,b.value_r);
    out.value_i = _mm256_mul_ps(val,b.value_i);
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator/(const SIMDVector<std::complex<float>,simd_abi::avx> &a, const SIMDVector<std::complex<float>,simd_abi::avx> &b) {
    SIMDVector<std::complex<float>,simd_abi::avx> out;
#ifdef FASTOR_FMA_IMPL
    out.value_r = _mm256_fmadd_ps(a.value_r,b.value_r,_mm256_mul_ps(a.value_i,b.value_i));
    out.value_i = _mm256_fmsub_ps(a.value_i,b.value_r,_mm256_mul_ps(a.value_r,b.value_i));
    __m256 den = _mm256_fmadd_ps(b.value_r,b.value_r,_mm256_mul_ps(b.value_i,b.value_i));
#else
    out.value_r = _mm256_add_ps(_mm256_mul_ps(a.value_r,b.value_r),_mm256_mul_ps(a.value_i,b.value_i));
    out.value_i = _mm256_sub_ps(_mm256_mul_ps(a.value_i,b.value_r),_mm256_mul_ps(a.value_r,b.value_i));
    __m256 den = _mm256_add_ps(_mm256_mul_ps(b.value_r,b.value_r),_mm256_mul_ps(b.value_i,b.value_i));
#endif
    out.value_r = _mm256_div_ps(out.value_r,den);
    out.value_i = _mm256_div_ps(out.value_i,den);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator/(const SIMDVector<std::complex<float>,simd_abi::avx> &a, std::complex<float> b) {
    return a / SIMDVector<std::complex<float>,simd_abi::avx>(b);
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator/(std::complex<float> a, const SIMDVector<std::complex<float>,simd_abi::avx> &b) {
    return SIMDVector<std::complex<float>,simd_abi::avx>(a) / b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator/(const SIMDVector<std::complex<float>,simd_abi::avx> &a, U b) {
    SIMDVector<std::complex<float>,simd_abi::avx> out;
    __m256 val = _mm256_set1_ps(b);
    out.value_r = _mm256_div_ps(a.value_r,val);
    out.value_i = _mm256_div_ps(a.value_i,val);
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
operator/(U a, const SIMDVector<std::complex<float>,simd_abi::avx> &b) {
    return SIMDVector<std::complex<float>,simd_abi::avx>(std::complex<float>(a,0)) / b;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
rcp(const SIMDVector<std::complex<float>,simd_abi::avx> &a) {
    SIMDVector<std::complex<float>,simd_abi::avx> out;
#ifdef FASTOR_FMA_IMPL
    __m256 den = _mm256_fmadd_ps(a.value_r,a.value_r,_mm256_mul_ps(a.value_i,a.value_i));
#else
    __m256 den = _mm256_add_ps(_mm256_mul_ps(a.value_r,a.value_r),_mm256_mul_ps(a.value_i,a.value_i));
#endif
    out.value_r = _mm256_div_ps(out.value_r,den);
    out.value_i = _mm256_neg_ps(_mm256_div_ps(out.value_i,den));
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
sqrt(const SIMDVector<std::complex<float>,simd_abi::avx> &a) = delete;

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
rsqrt(const SIMDVector<std::complex<float>,simd_abi::avx> &a) = delete;

/* This intentionally return a complex vector */
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
abs(const SIMDVector<std::complex<float>,simd_abi::avx> &a) {
    SIMDVector<std::complex<float>,simd_abi::avx> out;
    out.value_r = a.magnitude().value;
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
conj(const SIMDVector<std::complex<float>,simd_abi::avx> &a) {
    SIMDVector<std::complex<float>,simd_abi::avx> out(a);
    out.value_i = _mm256_neg_ps(out.value_i);
    return out;
}

/* Argument or phase angle */
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::avx>
arg(const SIMDVector<std::complex<float>,simd_abi::avx> &a) {
    SIMDVector<std::complex<float>,simd_abi::avx> out(a);
    for (FASTOR_INDEX i=0UL; i<8UL; ++i) {
       ((float*)&out.value_r)[i] = std::atan2(((float*)&a.value_i)[i],((float*)&a.value_r)[i]);
    }
    return out;
}
//------------------------------------------------------------------------------------------------------------

#endif




// SSE VERSION
//------------------------------------------------------------------------------------------------------------

#ifdef FASTOR_SSE2_IMPL

template <>
struct SIMDVector<std::complex<float>, simd_abi::sse> {
    using vector_type = SIMDVector<std::complex<float>, simd_abi::sse>;
    using value_type = __m128;
    using scalar_value_type = std::complex<float>;
    using abi_type = simd_abi::sse;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<float,simd_abi::sse>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<float,simd_abi::sse>>::value;}

    FASTOR_INLINE SIMDVector() : value_r(_mm_setzero_ps()), value_i(_mm_setzero_ps()) {}
    FASTOR_INLINE SIMDVector(std::complex<float> num) {
        value_r = _mm_set1_ps(num.real());
        value_i = _mm_set1_ps(num.imag());
    }
    FASTOR_INLINE SIMDVector(value_type reg0, value_type reg1) : value_r(reg0), value_i(reg1) {}
    FASTOR_INLINE SIMDVector(const std::complex<float> *data, bool Aligned=true) {
        if (Aligned)
            complex_aligned_load(data);
        else
            complex_unaligned_load(data);
    }

    FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse> operator=(std::complex<float> num) {
        value_r = _mm_set1_ps(num.real());
        value_i = _mm_set1_ps(num.imag());
        return *this;
    }

    FASTOR_INLINE void load(const std::complex<float> *data, bool Aligned=true) {
        if (Aligned)
            complex_aligned_load(data);
        else
            complex_unaligned_load(data);
    }
    FASTOR_INLINE void store(std::complex<float> *data, bool Aligned=true) const {
        if (Aligned)
            complex_aligned_store(data);
        else
            complex_unaligned_store(data);
    }

    FASTOR_INLINE void mask_load(const scalar_value_type *data, uint8_t mask, bool Aligned=false) {
        if (!Aligned)
            complex_mask_unaligned_load(data,mask);
        else
            complex_mask_aligned_load(data,mask);
    }
    FASTOR_INLINE void mask_store(scalar_value_type *data, uint8_t mask, bool Aligned=false) const {
        if (!Aligned)
            complex_mask_unaligned_store(data,mask);
        else
            complex_mask_aligned_store(data,mask);
    }

    FASTOR_INLINE scalar_value_type operator[](FASTOR_INDEX i) const {
        if      (i == 0) { return scalar_value_type(_mm_get0_ps(value_r), _mm_get0_ps(value_i)); }
        else if (i == 1) { return scalar_value_type(_mm_get1_ps(value_r), _mm_get1_ps(value_i)); }
        else if (i == 2) { return scalar_value_type(_mm_get2_ps(value_r), _mm_get2_ps(value_i)); }
        else             { return scalar_value_type(_mm_get3_ps(value_r), _mm_get3_ps(value_i)); }
    }

    FASTOR_INLINE SIMDVector<float,simd_abi::sse> real() const {
        return value_r;
    }
    FASTOR_INLINE SIMDVector<float,simd_abi::sse> imag() const {
        return value_i;
    }

    FASTOR_INLINE void set(std::complex<float> num) {
        value_r = _mm_set1_ps(num.real());
        value_i = _mm_set1_ps(num.imag());
    }
    FASTOR_INLINE void set(scalar_value_type num0, scalar_value_type num1,
                           scalar_value_type num2, scalar_value_type num3) {
        const scalar_value_type tmp[Size] = {num0,num1,num2,num3};
        complex_unaligned_load(tmp);
    }

    // In-place operators
    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator+=(U num) {
        value_r = _mm_add_ps(value_r,_mm_set1_ps(num));
    }
    FASTOR_INLINE void operator+=(scalar_value_type num) {
        *this += vector_type(num);
    }
    FASTOR_INLINE void operator+=(const vector_type &a) {
        value_r = _mm_add_ps(value_r,a.value_r);
        value_i = _mm_add_ps(value_i,a.value_i);
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator-=(U num) {
        value_r = _mm_sub_ps(value_r,_mm_set1_ps(num));
    }
    FASTOR_INLINE void operator-=(scalar_value_type num) {
        *this -= vector_type(num);
    }
    FASTOR_INLINE void operator-=(const vector_type &a) {
        value_r = _mm_sub_ps(value_r,a.value_r);
        value_i = _mm_sub_ps(value_i,a.value_i);
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator*=(U num) {
        __m128 val = _mm_set1_ps(num);
        value_r = _mm_mul_ps(value_r,val);
        value_i = _mm_mul_ps(value_i,val);
    }
    FASTOR_INLINE void operator*=(scalar_value_type num) {
        *this *= vector_type(num);
    }
    FASTOR_INLINE void operator*=(const vector_type &a) {
#ifdef FASTOR_FMA_IMPL
        __m128 tmp = _mm_fmsub_ps(value_r,a.value_r,_mm_mul_ps(value_i,a.value_i));
        value_i     = _mm_fmadd_ps(value_r,a.value_i,_mm_mul_ps(value_i,a.value_r));
#else
        __m128 tmp = _mm_sub_ps(_mm_mul_ps(value_r,a.value_r),_mm_mul_ps(value_i,a.value_i));
        value_i     = _mm_add_ps(_mm_mul_ps(value_r,a.value_i),_mm_mul_ps(value_i,a.value_r));
#endif
        value_r = tmp;
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator/=(U num) {
        __m128 val = _mm_set1_ps(num);
        value_r = _mm_div_ps(value_r,val);
        value_i = _mm_div_ps(value_i,val);
    }
    FASTOR_INLINE void operator/=(scalar_value_type num) {
        *this /= vector_type(num);
    }
    FASTOR_INLINE void operator/=(const vector_type &a) {
        __m128 tmp = value_r;
#ifdef FASTOR_FMA_IMPL
        value_r     = _mm_fmadd_ps(value_r  , a.value_r, _mm_mul_ps(value_i,a.value_i));
        value_i     = _mm_fmsub_ps(value_i  , a.value_r, _mm_mul_ps(tmp,a.value_i));
        __m128 den = _mm_fmadd_ps(a.value_r, a.value_r, _mm_mul_ps(a.value_i,a.value_i));
#else
        value_r     = _mm_add_ps(_mm_mul_ps(value_r  , a.value_r), _mm_mul_ps(value_i,a.value_i));
        value_i     = _mm_sub_ps(_mm_mul_ps(value_i  , a.value_r), _mm_mul_ps(tmp,a.value_i));
        __m128 den = _mm_add_ps(_mm_mul_ps(a.value_r, a.value_r), _mm_mul_ps(a.value_i,a.value_i));
#endif
        value_r     = _mm_div_ps(value_r,den);
        value_i     = _mm_div_ps(value_i,den);
    }
    // end of in-place operators

    FASTOR_INLINE scalar_value_type sum() const {
        return scalar_value_type(_mm_sum_ps(value_r),_mm_sum_ps(value_i));
    }
    FASTOR_INLINE scalar_value_type product() const {
        vector_type tmp(*this);
        return tmp[0]*tmp[1]*tmp[2]*tmp[3];
    }
    FASTOR_INLINE vector_type reverse() const {
        vector_type out;
        out.value_r = _mm_reverse_ps(value_r);
        out.value_i = _mm_reverse_ps(value_i);
        return out;
    }
    /* Actual magnitude - Note that this is a vertical operation */
    FASTOR_INLINE SIMDVector<float,simd_abi::sse> magnitude() const {
#ifdef FASTOR_FMA_IMPL
        return _mm_sqrt_ps(_mm_fmadd_ps(value_r,value_r,_mm_mul_ps(value_i,value_i)));
#else
        return _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(value_r,value_r),_mm_mul_ps(value_i,value_i)));
#endif
    }
    /* STL compliant squared norm - Note that this is a vertical operation */
    FASTOR_INLINE SIMDVector<float,simd_abi::sse> norm() const {
#ifdef FASTOR_FMA_IMPL
        return _mm_fmadd_ps(value_r,value_r,_mm_mul_ps(value_i,value_i));
#else
        return _mm_add_ps(_mm_mul_ps(value_r,value_r),_mm_mul_ps(value_i,value_i));
#endif
    }
    // Magnitude based minimum
    FASTOR_INLINE scalar_value_type minimum() const;
    // Magnitude based maximum
    FASTOR_INLINE scalar_value_type maximum() const;

    FASTOR_INLINE scalar_value_type dot(const vector_type &other) const {
        vector_type out(*this);
        out *= other;
        return out.sum();
    }

    value_type value_r;
    value_type value_i;

protected:
    FASTOR_INLINE void complex_aligned_load(const std::complex<float> *data) {
        __m128 lo = _mm_load_ps(reinterpret_cast<const float*>(data  ));
        __m128 hi = _mm_load_ps(reinterpret_cast<const float*>(data+2));
        arrange_from_load(value_r, value_i, lo, hi);
    }
    FASTOR_INLINE void complex_unaligned_load(const std::complex<float> *data) {
        __m128 lo = _mm_loadu_ps(reinterpret_cast<const float*>(data  ));
        __m128 hi = _mm_loadu_ps(reinterpret_cast<const float*>(data+2));
        arrange_from_load(value_r, value_i, lo, hi);
    }

    FASTOR_INLINE void complex_aligned_store(std::complex<float> *data) const {
        __m128 lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        _mm_store_ps(reinterpret_cast<float*>(data  ),lo);
        _mm_store_ps(reinterpret_cast<float*>(data+2),hi);
    }
    FASTOR_INLINE void complex_unaligned_store(std::complex<float> *data) const {
        __m128 lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        _mm_storeu_ps(reinterpret_cast<float*>(data  ),lo);
        _mm_storeu_ps(reinterpret_cast<float*>(data+2),hi);
    }

    FASTOR_INLINE void complex_mask_aligned_load(const scalar_value_type *data, uint8_t mask) {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m128 lo, hi;
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        lo = _mm_mask_load_ps(lo, mask0, reinterpret_cast<const float*>(data  ));
        hi = _mm_mask_load_ps(hi, mask1, reinterpret_cast<const float*>(data+2));
        arrange_from_load(value_r, value_i, lo, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        value_r = _mm_setzero_ps();
        value_i = _mm_setzero_ps();
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                ((float*)&value_r)[Size - i - 1] = data[Size - i - 1].real();
                ((float*)&value_i)[Size - i - 1] = data[Size - i - 1].imag();
            }
        }
#endif
    }
    FASTOR_INLINE void complex_mask_unaligned_load(const scalar_value_type *data, uint8_t mask) {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m128 lo, hi;
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        lo = _mm_mask_loadu_ps(lo, mask0, reinterpret_cast<const float*>(data  ));
        hi = _mm_mask_loadu_ps(hi, mask1, reinterpret_cast<const float*>(data+2));
        arrange_from_load(value_r, value_i, lo, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        value_r = _mm_setzero_ps();
        value_i = _mm_setzero_ps();
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                ((float*)&value_r)[Size - i - 1] = data[Size - i - 1].real();
                ((float*)&value_i)[Size - i - 1] = data[Size - i - 1].imag();
            }
        }
#endif
    }

    FASTOR_INLINE void complex_mask_aligned_store(scalar_value_type *data, uint8_t mask) const {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m128 lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        _mm_mask_store_ps(reinterpret_cast<float*>(data  ), mask0, lo);
        _mm_mask_store_ps(reinterpret_cast<float*>(data+2), mask1, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                float _real = ((const float*)&value_r)[Size - i - 1];
                float _imag = ((const float*)&value_i)[Size - i - 1];
                data[Size - i - 1] = std::complex<float>(_real,_imag);
            }
            else {
                data[Size - i - 1] = std::complex<float>(0,0);
            }
        }
#endif
    }
    FASTOR_INLINE void complex_mask_unaligned_store(scalar_value_type *data, uint8_t mask) const {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m128 lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        _mm_mask_storeu_ps(reinterpret_cast<float*>(data  ), mask0, lo);
        _mm_mask_storeu_ps(reinterpret_cast<float*>(data+2), mask1, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                float _real = ((const float*)&value_r)[Size - i - 1];
                float _imag = ((const float*)&value_i)[Size - i - 1];
                data[Size - i - 1] = std::complex<float>(_real,_imag);
            }
            else {
                data[Size - i - 1] = std::complex<float>(0,0);
            }
        }
#endif
    }

};

FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<std::complex<float>,simd_abi::sse> a) {
    // ICC crashes without a copy
    const __m128 vr = a.value_r;
    const __m128 vi = a.value_i;
    const float* value_r = reinterpret_cast<const float*>(&vr);
    const float* value_i = reinterpret_cast<const float*>(&vi);
    os << "[" << value_r[0] <<  signum_string(value_i[0]) << std::abs(value_i[0]) << "j, "
              << value_r[1] <<  signum_string(value_i[1]) << std::abs(value_i[1]) << "j, "
              << value_r[2] <<  signum_string(value_i[2]) << std::abs(value_i[2]) << "j, "
              << value_r[3] <<  signum_string(value_i[3]) << std::abs(value_i[3]) << "j" << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator+(const SIMDVector<std::complex<float>,simd_abi::sse> &a, const SIMDVector<std::complex<float>,simd_abi::sse> &b) {
    SIMDVector<std::complex<float>,simd_abi::sse> out;
    out.value_r = _mm_add_ps(a.value_r,b.value_r);
    out.value_i = _mm_add_ps(a.value_i,b.value_i);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator+(const SIMDVector<std::complex<float>,simd_abi::sse> &a, std::complex<float> b) {
    return a + SIMDVector<std::complex<float>,simd_abi::sse>(b);
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator+(std::complex<float> a, const SIMDVector<std::complex<float>,simd_abi::sse> &b) {
    return SIMDVector<std::complex<float>,simd_abi::sse>(a) + b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator+(const SIMDVector<std::complex<float>,simd_abi::sse> &a, U b) {
    SIMDVector<std::complex<float>,simd_abi::sse> out(a);
    out.value_r = _mm_add_ps(a.value_r,_mm_set1_ps(b));
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator+(U a, const SIMDVector<std::complex<float>,simd_abi::sse> &b) {
    SIMDVector<std::complex<float>,simd_abi::sse> out(b);
    out.value_r = _mm_add_ps(_mm_set1_ps(a), b.value_r);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator+(const SIMDVector<std::complex<float>,simd_abi::sse> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator-(const SIMDVector<std::complex<float>,simd_abi::sse> &a, const SIMDVector<std::complex<float>,simd_abi::sse> &b) {
    SIMDVector<std::complex<float>,simd_abi::sse> out;
    out.value_r = _mm_sub_ps(a.value_r,b.value_r);
    out.value_i = _mm_sub_ps(a.value_i,b.value_i);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator-(const SIMDVector<std::complex<float>,simd_abi::sse> &a, std::complex<float> b) {
    return a - SIMDVector<std::complex<float>,simd_abi::sse>(b);
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator-(std::complex<float> a, const SIMDVector<std::complex<float>,simd_abi::sse> &b) {
    return SIMDVector<std::complex<float>,simd_abi::sse>(a) - b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator-(const SIMDVector<std::complex<float>,simd_abi::sse> &a, U b) {
    SIMDVector<std::complex<float>,simd_abi::sse> out(a);
    out.value_r = _mm_sub_ps(a.value_r,_mm_set1_ps(b));
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator-(U a, const SIMDVector<std::complex<float>,simd_abi::sse> &b) {
    return SIMDVector<std::complex<float>,simd_abi::sse>(std::complex<float>(a,0)) - b;
}
/* This is negation and not complex conjugate  */
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator-(const SIMDVector<std::complex<float>,simd_abi::sse> &a) {
    SIMDVector<std::complex<float>,simd_abi::sse> out;
    out.value_r = _mm_neg_ps(a.value_r);
    out.value_i = _mm_neg_ps(a.value_i);
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator*(const SIMDVector<std::complex<float>,simd_abi::sse> &a, const SIMDVector<std::complex<float>,simd_abi::sse> &b) {
    SIMDVector<std::complex<float>,simd_abi::sse> out;
#ifdef FASTOR_FMA_IMPL
    out.value_r = _mm_fmsub_ps(a.value_r,b.value_r,_mm_mul_ps(a.value_i,b.value_i));
    out.value_i = _mm_fmadd_ps(a.value_r,b.value_i,_mm_mul_ps(a.value_i,b.value_r));
#else
    out.value_r = _mm_sub_ps(_mm_mul_ps(a.value_r,b.value_r),_mm_mul_ps(a.value_i,b.value_i));
    out.value_i = _mm_add_ps(_mm_mul_ps(a.value_r,b.value_i),_mm_mul_ps(a.value_i,b.value_r));
#endif
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator*(const SIMDVector<std::complex<float>,simd_abi::sse> &a, std::complex<float> b) {
    return a * SIMDVector<std::complex<float>,simd_abi::sse>(b);
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator*(std::complex<float> a, const SIMDVector<std::complex<float>,simd_abi::sse> &b) {
    return SIMDVector<std::complex<float>,simd_abi::sse>(a) * b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator*(const SIMDVector<std::complex<float>,simd_abi::sse> &a, U b) {
    SIMDVector<std::complex<float>,simd_abi::sse> out;
    __m128 val = _mm_set1_ps(b);
    out.value_r = _mm_mul_ps(a.value_r,val);
    out.value_i = _mm_mul_ps(a.value_i,val);
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator*(U a, const SIMDVector<std::complex<float>,simd_abi::sse> &b) {
    SIMDVector<std::complex<float>,simd_abi::sse> out;
    __m128 val = _mm_set1_ps(a);
    out.value_r = _mm_mul_ps(val,b.value_r);
    out.value_i = _mm_mul_ps(val,b.value_i);
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator/(const SIMDVector<std::complex<float>,simd_abi::sse> &a, const SIMDVector<std::complex<float>,simd_abi::sse> &b) {
    SIMDVector<std::complex<float>,simd_abi::sse> out;
#ifdef FASTOR_FMA_IMPL
    out.value_r = _mm_fmadd_ps(a.value_r,b.value_r,_mm_mul_ps(a.value_i,b.value_i));
    out.value_i = _mm_fmsub_ps(a.value_i,b.value_r,_mm_mul_ps(a.value_r,b.value_i));
    __m128 den = _mm_fmadd_ps(b.value_r,b.value_r,_mm_mul_ps(b.value_i,b.value_i));
#else
    out.value_r = _mm_add_ps(_mm_mul_ps(a.value_r,b.value_r),_mm_mul_ps(a.value_i,b.value_i));
    out.value_i = _mm_sub_ps(_mm_mul_ps(a.value_i,b.value_r),_mm_mul_ps(a.value_r,b.value_i));
    __m128 den = _mm_add_ps(_mm_mul_ps(b.value_r,b.value_r),_mm_mul_ps(b.value_i,b.value_i));
#endif
    out.value_r = _mm_div_ps(out.value_r,den);
    out.value_i = _mm_div_ps(out.value_i,den);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator/(const SIMDVector<std::complex<float>,simd_abi::sse> &a, std::complex<float> b) {
    return a / SIMDVector<std::complex<float>,simd_abi::sse>(b);
}
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator/(std::complex<float> a, const SIMDVector<std::complex<float>,simd_abi::sse> &b) {
    return SIMDVector<std::complex<float>,simd_abi::sse>(a) / b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator/(const SIMDVector<std::complex<float>,simd_abi::sse> &a, U b) {
    SIMDVector<std::complex<float>,simd_abi::sse> out;
    __m128 val = _mm_set1_ps(b);
    out.value_r = _mm_div_ps(a.value_r,val);
    out.value_i = _mm_div_ps(a.value_i,val);
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
operator/(U a, const SIMDVector<std::complex<float>,simd_abi::sse> &b) {
    return SIMDVector<std::complex<float>,simd_abi::sse>(std::complex<float>(a,0)) / b;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
rcp(const SIMDVector<std::complex<float>,simd_abi::sse> &a) {
    SIMDVector<std::complex<float>,simd_abi::sse> out;
#ifdef FASTOR_FMA_IMPL
    __m128 den = _mm_fmadd_ps(a.value_r,a.value_r,_mm_mul_ps(a.value_i,a.value_i));
#else
    __m128 den = _mm_add_ps(_mm_mul_ps(a.value_r,a.value_r),_mm_mul_ps(a.value_i,a.value_i));
#endif
    out.value_r = _mm_div_ps(out.value_r,den);
    out.value_i = _mm_neg_ps(_mm_div_ps(out.value_i,den));
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
sqrt(const SIMDVector<std::complex<float>,simd_abi::sse> &a) = delete;

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
rsqrt(const SIMDVector<std::complex<float>,simd_abi::sse> &a) = delete;

/* This intentionally return a complex vector */
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
abs(const SIMDVector<std::complex<float>,simd_abi::sse> &a) {
    SIMDVector<std::complex<float>,simd_abi::sse> out;
    out.value_r = a.magnitude().value;
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
conj(const SIMDVector<std::complex<float>,simd_abi::sse> &a) {
    SIMDVector<std::complex<float>,simd_abi::sse> out(a);
    out.value_i = _mm_neg_ps(out.value_i);
    return out;
}

/* Argument or phase angle */
FASTOR_INLINE SIMDVector<std::complex<float>,simd_abi::sse>
arg(const SIMDVector<std::complex<float>,simd_abi::sse> &a) {
    SIMDVector<std::complex<float>,simd_abi::sse> out(a);
    for (FASTOR_INDEX i=0UL; i<4UL; ++i) {
       ((float*)&out.value_r)[i] = std::atan2(((float*)&a.value_i)[i],((float*)&a.value_r)[i]);
    }
    return out;
}
//------------------------------------------------------------------------------------------------------------

#endif


} // end of namespace Fastor

#endif // SIMD_VECTOR_COMPLEX_FLOAT_H
