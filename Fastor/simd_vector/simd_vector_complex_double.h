#ifndef SIMD_VECTOR_COMPLEX_DOUBLE_H
#define SIMD_VECTOR_COMPLEX_DOUBLE_H

#include "Fastor/util/extended_algorithms.h"
#include "Fastor/simd_vector/extintrin.h"
#include "Fastor/simd_vector/simd_vector_base.h"
#include "Fastor/simd_vector/simd_vector_double.h"
#include <cmath>
#include <complex>

namespace Fastor {


// AVX512 VERSION
//------------------------------------------------------------------------------------------------------------

#ifdef FASTOR_AVX512F_IMPL

template <>
struct SIMDVector<std::complex<double>, simd_abi::avx512> {
    using vector_type = SIMDVector<std::complex<double>, simd_abi::avx512>;
    using value_type = __m512d;
    using scalar_value_type = std::complex<double>;
    using abi_type = simd_abi::avx512;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<double,simd_abi::avx512>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<double,simd_abi::avx512>>::value;}

    FASTOR_INLINE SIMDVector() : value_r(_mm512_setzero_pd()), value_i(_mm512_setzero_pd()) {}
    FASTOR_INLINE SIMDVector(std::complex<double> num) {
        value_r = _mm512_set1_pd(num.real());
        value_i = _mm512_set1_pd(num.imag());
    }
    FASTOR_INLINE SIMDVector(value_type reg0, value_type reg1) : value_r(reg0), value_i(reg1) {}
    FASTOR_INLINE SIMDVector(const std::complex<double> *data, bool Aligned=true) {
        if (Aligned)
            complex_aligned_load(data);
        else
            complex_unaligned_load(data);
    }

    FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512> operator=(std::complex<double> num) {
        value_r = _mm512_set1_pd(num.real());
        value_i = _mm512_set1_pd(num.imag());
        return *this;
    }

    FASTOR_INLINE void load(const std::complex<double> *data, bool Aligned=true) {
        if (Aligned)
            complex_aligned_load(data);
        else
            complex_unaligned_load(data);
    }
    FASTOR_INLINE void store(std::complex<double> *data, bool Aligned=true) const {
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
        return scalar_value_type(reinterpret_cast<const double*>(&value_r)[i],reinterpret_cast<const double*>(&value_i)[i]);
    }

    FASTOR_INLINE SIMDVector<double,simd_abi::avx512> real() const {
        return value_r;
    }
    FASTOR_INLINE SIMDVector<double,simd_abi::avx512> imag() const {
        return value_i;
    }

    FASTOR_INLINE void set(std::complex<double> num) {
        value_r = _mm512_set1_pd(num.real());
        value_i = _mm512_set1_pd(num.imag());
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
        value_r = _mm512_add_pd(value_r,_mm512_set1_pd(num));
    }
    FASTOR_INLINE void operator+=(scalar_value_type num) {
        *this += vector_type(num);
    }
    FASTOR_INLINE void operator+=(const vector_type &a) {
        value_r = _mm512_add_pd(value_r,a.value_r);
        value_i = _mm512_add_pd(value_i,a.value_i);
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator-=(U num) {
        value_r = _mm512_sub_pd(value_r,_mm512_set1_pd(num));
    }
    FASTOR_INLINE void operator-=(scalar_value_type num) {
        *this -= vector_type(num);
    }
    FASTOR_INLINE void operator-=(const vector_type &a) {
        value_r = _mm512_sub_pd(value_r,a.value_r);
        value_i = _mm512_sub_pd(value_i,a.value_i);
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator*=(U num) {
        __m512d val = _mm512_set1_pd(num);
        value_r = _mm512_mul_pd(value_r,val);
        value_i = _mm512_mul_pd(value_i,val);
    }
    FASTOR_INLINE void operator*=(scalar_value_type num) {
        *this *= vector_type(num);
    }
    FASTOR_INLINE void operator*=(const vector_type &a) {
#ifdef FASTOR_FMA_IMPL
        __m512d tmp = _mm512_fmsub_pd(value_r,a.value_r,_mm512_mul_pd(value_i,a.value_i));
        value_i     = _mm512_fmadd_pd(value_r,a.value_i,_mm512_mul_pd(value_i,a.value_r));
#else
        __m512d tmp = _mm512_sub_pd(_mm512_mul_pd(value_r,a.value_r),_mm512_mul_pd(value_i,a.value_i));
        value_i     = _mm512_add_pd(_mm512_mul_pd(value_r,a.value_i),_mm512_mul_pd(value_i,a.value_r));
#endif
        value_r = tmp;
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator/=(U num) {
        __m512d val = _mm512_set1_pd(num);
        value_r = _mm512_div_pd(value_r,val);
        value_i = _mm512_div_pd(value_i,val);
    }
    FASTOR_INLINE void operator/=(scalar_value_type num) {
        *this /= vector_type(num);
    }
    FASTOR_INLINE void operator/=(const vector_type &a) {
        __m512d tmp = value_r;
#ifdef FASTOR_FMA_IMPL
        value_r     = _mm512_fmadd_pd(value_r  , a.value_r, _mm512_mul_pd(value_i,a.value_i));
        value_i     = _mm512_fmsub_pd(value_i  , a.value_r, _mm512_mul_pd(tmp,a.value_i));
        __m512d den = _mm512_fmadd_pd(a.value_r, a.value_r, _mm512_mul_pd(a.value_i,a.value_i));
#else
        value_r     = _mm512_add_pd(_mm512_mul_pd(value_r  , a.value_r), _mm512_mul_pd(value_i,a.value_i));
        value_i     = _mm512_sub_pd(_mm512_mul_pd(value_i  , a.value_r), _mm512_mul_pd(tmp,a.value_i));
        __m512d den = _mm512_add_pd(_mm512_mul_pd(a.value_r, a.value_r), _mm512_mul_pd(a.value_i,a.value_i));
#endif
        value_r     = _mm512_div_pd(value_r,den);
        value_i     = _mm512_div_pd(value_i,den);
    }
    // end of in-place operators

    FASTOR_INLINE scalar_value_type sum() const {
#ifdef FASTOR_HAS_AVX512_REDUCE_ADD
        return scalar_value_type(_mm512_reduce_add_pd(value_r),_mm512_reduce_add_pd(value_i));
#else
        __m256d lor  = _mm512_castpd512_pd256(value_r);
        __m256d hir  = _mm512_extractf64x4_pd(value_r,0x1);
        __m256d loi  = _mm512_castpd512_pd256(value_i);
        __m256d hii  = _mm512_extractf64x4_pd(value_i,0x1);
        return scalar_value_type(_mm256_sum_pd(_mm256_add_pd(lor,hir)), _mm256_sum_pd(_mm256_add_pd(loi,hii)));
#endif
    }
    FASTOR_INLINE scalar_value_type product() const {
        vector_type tmp(*this);
        return tmp[0]*tmp[1]*tmp[2]*tmp[3]*tmp[4]*tmp[5]*tmp[6]*tmp[7];
    }
    FASTOR_INLINE vector_type reverse() const {
        vector_type out;
        out.value_r = _mm512_reverse_pd(value_r);
        out.value_i = _mm512_reverse_pd(value_i);
        return out;
    }
    /* Actual magnitude - Note that this is a vertical operation */
    FASTOR_INLINE SIMDVector<double,simd_abi::avx512> magnitude() const {
#ifdef FASTOR_FMA_IMPL
        return _mm512_sqrt_pd(_mm512_fmadd_pd(value_r,value_r,_mm512_mul_pd(value_i,value_i)));
#else
        return _mm512_sqrt_pd(_mm512_add_pd(_mm512_mul_pd(value_r,value_r),_mm512_mul_pd(value_i,value_i)));
#endif
    }
    /* STL compliant squared norm - Note that this is a vertical operation */
    FASTOR_INLINE SIMDVector<double,simd_abi::avx512> norm() const {
#ifdef FASTOR_FMA_IMPL
        return _mm512_fmadd_pd(value_r,value_r,_mm512_mul_pd(value_i,value_i));
#else
        return _mm512_add_pd(_mm512_mul_pd(value_r,value_r),_mm512_mul_pd(value_i,value_i));
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
    FASTOR_INLINE void complex_aligned_load(const std::complex<double> *data) {
        __m512d lo = _mm512_load_pd(reinterpret_cast<const double*>(data  ));
        __m512d hi = _mm512_load_pd(reinterpret_cast<const double*>(data+4));
        arrange_from_load(value_r, value_i, lo, hi);
    }
    FASTOR_INLINE void complex_unaligned_load(const std::complex<double> *data) {
        __m512d lo = _mm512_loadu_pd(reinterpret_cast<const double*>(data  ));
        __m512d hi = _mm512_loadu_pd(reinterpret_cast<const double*>(data+4));
        arrange_from_load(value_r, value_i, lo, hi);
    }

    FASTOR_INLINE void complex_aligned_store(std::complex<double> *data) const {
        __m512d lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        _mm512_store_pd(reinterpret_cast<double*>(data  ), lo);
        _mm512_store_pd(reinterpret_cast<double*>(data+4), hi);
    }
    FASTOR_INLINE void complex_unaligned_store(std::complex<double> *data) const {
        __m512d lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        _mm512_storeu_pd(reinterpret_cast<double*>(data  ), lo);
        _mm512_storeu_pd(reinterpret_cast<double*>(data+4), hi);
    }

    FASTOR_INLINE void complex_mask_aligned_load(const scalar_value_type *data, uint8_t mask) {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m512d lo, hi;
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        lo = _mm512_mask_load_pd(lo, mask0, reinterpret_cast<const double*>(data  ));
        hi = _mm512_mask_load_pd(hi, mask1, reinterpret_cast<const double*>(data+4));
        arrange_from_load(value_r, value_i, lo, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        value_r = _mm512_setzero_pd();
        value_i = _mm512_setzero_pd();
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                ((double*)&value_r)[Size - i - 1] = data[Size - i - 1].real();
                ((double*)&value_i)[Size - i - 1] = data[Size - i - 1].imag();
            }
        }
#endif
    }
    FASTOR_INLINE void complex_mask_unaligned_load(const scalar_value_type *data, uint8_t mask) {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m512d lo, hi;
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        lo = _mm512_mask_loadu_pd(lo, mask0, reinterpret_cast<const double*>(data  ));
        hi = _mm512_mask_loadu_pd(hi, mask1, reinterpret_cast<const double*>(data+4));
        arrange_from_load(value_r, value_i, lo, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        value_r = _mm512_setzero_pd();
        value_i = _mm512_setzero_pd();
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                ((double*)&value_r)[Size - i - 1] = data[Size - i - 1].real();
                ((double*)&value_i)[Size - i - 1] = data[Size - i - 1].imag();
            }
        }
#endif
    }

    FASTOR_INLINE void complex_mask_aligned_store(scalar_value_type *data, uint8_t mask) const {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m512d lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        _mm512_mask_store_pd(reinterpret_cast<double*>(data  ), mask0, lo);
        _mm512_mask_store_pd(reinterpret_cast<double*>(data+4), mask1, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                double _real = ((const double*)&value_r)[Size - i - 1];
                double _imag = ((const double*)&value_i)[Size - i - 1];
                data[Size - i - 1] = std::complex<double>(_real,_imag);
            }
            else {
                data[Size - i - 1] = std::complex<double>(0,0);
            }
        }
#endif
    }
    FASTOR_INLINE void complex_mask_unaligned_store(scalar_value_type *data, uint8_t mask) const {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m512d lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        _mm512_mask_storeu_pd(reinterpret_cast<double*>(data  ), mask0, lo);
        _mm512_mask_storeu_pd(reinterpret_cast<double*>(data+4), mask1, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                double _real = ((const double*)&value_r)[Size - i - 1];
                double _imag = ((const double*)&value_i)[Size - i - 1];
                data[Size - i - 1] = std::complex<double>(_real,_imag);
            }
            else {
                data[Size - i - 1] = std::complex<double>(0,0);
            }
        }
#endif
    }
};

FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<std::complex<double>,simd_abi::avx512> a) {
    // ICC crashes without a copy
    const __m512d vr = a.value_r;
    const __m512d vi = a.value_i;
    const double* value_r = reinterpret_cast<const double*>(&vr);
    const double* value_i = reinterpret_cast<const double*>(&vi);
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

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator+(const SIMDVector<std::complex<double>,simd_abi::avx512> &a, const SIMDVector<std::complex<double>,simd_abi::avx512> &b) {
    SIMDVector<std::complex<double>,simd_abi::avx512> out;
    out.value_r = _mm512_add_pd(a.value_r,b.value_r);
    out.value_i = _mm512_add_pd(a.value_i,b.value_i);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator+(const SIMDVector<std::complex<double>,simd_abi::avx512> &a, std::complex<double> b) {
    return a + SIMDVector<std::complex<double>,simd_abi::avx512>(b);
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator+(std::complex<double> a, const SIMDVector<std::complex<double>,simd_abi::avx512> &b) {
    return SIMDVector<std::complex<double>,simd_abi::avx512>(a) + b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator+(const SIMDVector<std::complex<double>,simd_abi::avx512> &a, U b) {
    SIMDVector<std::complex<double>,simd_abi::avx512> out(a);
    out.value_r = _mm512_add_pd(a.value_r,_mm512_set1_pd(b));
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator+(U a, const SIMDVector<std::complex<double>,simd_abi::avx512> &b) {
    SIMDVector<std::complex<double>,simd_abi::avx512> out(b);
    out.value_r = _mm512_add_pd(_mm512_set1_pd(a), b.value_r);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator+(const SIMDVector<std::complex<double>,simd_abi::avx512> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator-(const SIMDVector<std::complex<double>,simd_abi::avx512> &a, const SIMDVector<std::complex<double>,simd_abi::avx512> &b) {
    SIMDVector<std::complex<double>,simd_abi::avx512> out;
    out.value_r = _mm512_sub_pd(a.value_r,b.value_r);
    out.value_i = _mm512_sub_pd(a.value_i,b.value_i);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator-(const SIMDVector<std::complex<double>,simd_abi::avx512> &a, std::complex<double> b) {
    return a - SIMDVector<std::complex<double>,simd_abi::avx512>(b);
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator-(std::complex<double> a, const SIMDVector<std::complex<double>,simd_abi::avx512> &b) {
    return SIMDVector<std::complex<double>,simd_abi::avx512>(a) - b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator-(const SIMDVector<std::complex<double>,simd_abi::avx512> &a, U b) {
    SIMDVector<std::complex<double>,simd_abi::avx512> out(a);
    out.value_r = _mm512_sub_pd(a.value_r,_mm512_set1_pd(b));
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator-(U a, const SIMDVector<std::complex<double>,simd_abi::avx512> &b) {
    return SIMDVector<std::complex<double>,simd_abi::avx512>(std::complex<double>(a,0)) - b;
}
/* This is negation and not complex conjugate  */
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator-(const SIMDVector<std::complex<double>,simd_abi::avx512> &a) {
    SIMDVector<std::complex<double>,simd_abi::avx512> out;
    out.value_r = _mm512_neg_pd(a.value_r);
    out.value_i = _mm512_neg_pd(a.value_i);
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator*(const SIMDVector<std::complex<double>,simd_abi::avx512> &a, const SIMDVector<std::complex<double>,simd_abi::avx512> &b) {
    SIMDVector<std::complex<double>,simd_abi::avx512> out;
#ifdef FASTOR_FMA_IMPL
    out.value_r = _mm512_fmsub_pd(a.value_r,b.value_r,_mm512_mul_pd(a.value_i,b.value_i));
    out.value_i = _mm512_fmadd_pd(a.value_r,b.value_i,_mm512_mul_pd(a.value_i,b.value_r));
#else
    out.value_r = _mm512_sub_pd(_mm512_mul_pd(a.value_r,b.value_r),_mm512_mul_pd(a.value_i,b.value_i));
    out.value_i = _mm512_add_pd(_mm512_mul_pd(a.value_r,b.value_i),_mm512_mul_pd(a.value_i,b.value_r));
#endif
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator*(const SIMDVector<std::complex<double>,simd_abi::avx512> &a, std::complex<double> b) {
    return a * SIMDVector<std::complex<double>,simd_abi::avx512>(b);
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator*(std::complex<double> a, const SIMDVector<std::complex<double>,simd_abi::avx512> &b) {
    return SIMDVector<std::complex<double>,simd_abi::avx512>(a) * b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator*(const SIMDVector<std::complex<double>,simd_abi::avx512> &a, U b) {
    SIMDVector<std::complex<double>,simd_abi::avx512> out;
    __m512d val = _mm512_set1_pd(b);
    out.value_r = _mm512_mul_pd(a.value_r,val);
    out.value_i = _mm512_mul_pd(a.value_i,val);
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator*(U a, const SIMDVector<std::complex<double>,simd_abi::avx512> &b) {
    SIMDVector<std::complex<double>,simd_abi::avx512> out;
    __m512d val = _mm512_set1_pd(a);
    out.value_r = _mm512_mul_pd(val,b.value_r);
    out.value_i = _mm512_mul_pd(val,b.value_i);
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator/(const SIMDVector<std::complex<double>,simd_abi::avx512> &a, const SIMDVector<std::complex<double>,simd_abi::avx512> &b) {
    SIMDVector<std::complex<double>,simd_abi::avx512> out;
#ifdef FASTOR_FMA_IMPL
    out.value_r = _mm512_fmadd_pd(a.value_r,b.value_r,_mm512_mul_pd(a.value_i,b.value_i));
    out.value_i = _mm512_fmsub_pd(a.value_i,b.value_r,_mm512_mul_pd(a.value_r,b.value_i));
    __m512d den = _mm512_fmadd_pd(b.value_r,b.value_r,_mm512_mul_pd(b.value_i,b.value_i));
#else
    out.value_r = _mm512_add_pd(_mm512_mul_pd(a.value_r,b.value_r),_mm512_mul_pd(a.value_i,b.value_i));
    out.value_i = _mm512_sub_pd(_mm512_mul_pd(a.value_i,b.value_r),_mm512_mul_pd(a.value_r,b.value_i));
    __m512d den = _mm512_add_pd(_mm512_mul_pd(b.value_r,b.value_r),_mm512_mul_pd(b.value_i,b.value_i));
#endif
    out.value_r = _mm512_div_pd(out.value_r,den);
    out.value_i = _mm512_div_pd(out.value_i,den);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator/(const SIMDVector<std::complex<double>,simd_abi::avx512> &a, std::complex<double> b) {
    return a / SIMDVector<std::complex<double>,simd_abi::avx512>(b);
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator/(std::complex<double> a, const SIMDVector<std::complex<double>,simd_abi::avx512> &b) {
    return SIMDVector<std::complex<double>,simd_abi::avx512>(a) / b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator/(const SIMDVector<std::complex<double>,simd_abi::avx512> &a, U b) {
    SIMDVector<std::complex<double>,simd_abi::avx512> out;
    __m512d val = _mm512_set1_pd(b);
    out.value_r = _mm512_div_pd(a.value_r,val);
    out.value_i = _mm512_div_pd(a.value_i,val);
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
operator/(U a, const SIMDVector<std::complex<double>,simd_abi::avx512> &b) {
    return SIMDVector<std::complex<double>,simd_abi::avx512>(std::complex<double>(a,0)) / b;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
rcp(const SIMDVector<std::complex<double>,simd_abi::avx512> &a) {
    SIMDVector<std::complex<double>,simd_abi::avx512> out;
#ifdef FASTOR_FMA_IMPL
    __m512d den = _mm512_fmadd_pd(a.value_r,a.value_r,_mm512_mul_pd(a.value_i,a.value_i));
#else
    __m512d den = _mm512_add_pd(_mm512_mul_pd(a.value_r,a.value_r),_mm512_mul_pd(a.value_i,a.value_i));
#endif
    out.value_r = _mm512_div_pd(out.value_r,den);
    out.value_i = _mm512_neg_pd(_mm512_div_pd(out.value_i,den));
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
sqrt(const SIMDVector<std::complex<double>,simd_abi::avx512> &a) = delete;

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
rsqrt(const SIMDVector<std::complex<double>,simd_abi::avx512> &a) = delete;

/* This intentionally return a complex vector */
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
abs(const SIMDVector<std::complex<double>,simd_abi::avx512> &a) {
    SIMDVector<std::complex<double>,simd_abi::avx512> out;
    out.value_r = a.magnitude().value;
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
conj(const SIMDVector<std::complex<double>,simd_abi::avx512> &a) {
    SIMDVector<std::complex<double>,simd_abi::avx512> out(a);
    out.value_i = _mm512_neg_pd(out.value_i);
    return out;
}

/* Argument or phase angle */
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx512>
arg(const SIMDVector<std::complex<double>,simd_abi::avx512> &a) {
    SIMDVector<std::complex<double>,simd_abi::avx512> out(a);
    for (FASTOR_INDEX i=0UL; i<8UL; ++i) {
       ((double*)&out.value_r)[i] = std::atan2(((double*)&a.value_i)[i],((double*)&a.value_r)[i]);
    }
    return out;
}
//------------------------------------------------------------------------------------------------------------

#endif



// AVX VERSION
//------------------------------------------------------------------------------------------------------------

#ifdef FASTOR_AVX_IMPL

template <>
struct SIMDVector<std::complex<double>, simd_abi::avx> {
    using vector_type = SIMDVector<std::complex<double>, simd_abi::avx>;
    using value_type = __m256d;
    using scalar_value_type = std::complex<double>;
    using abi_type = simd_abi::avx;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<double,simd_abi::avx>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<double,simd_abi::avx>>::value;}

    FASTOR_INLINE SIMDVector() : value_r(_mm256_setzero_pd()), value_i(_mm256_setzero_pd()) {}
    FASTOR_INLINE SIMDVector(std::complex<double> num) {
        value_r = _mm256_set1_pd(num.real());
        value_i = _mm256_set1_pd(num.imag());
    }
    FASTOR_INLINE SIMDVector(value_type reg0, value_type reg1) : value_r(reg0), value_i(reg1) {}
    FASTOR_INLINE SIMDVector(const std::complex<double> *data, bool Aligned=true) {
        if (Aligned)
            complex_aligned_load(data);
        else
            complex_unaligned_load(data);
    }

    FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx> operator=(std::complex<double> num) {
        value_r = _mm256_set1_pd(num.real());
        value_i = _mm256_set1_pd(num.imag());
        return *this;
    }

    FASTOR_INLINE void load(const std::complex<double> *data, bool Aligned=true) {
        if (Aligned)
            complex_aligned_load(data);
        else
            complex_unaligned_load(data);
    }
    FASTOR_INLINE void store(std::complex<double> *data, bool Aligned=true) const {
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
        if      (i == 0) { return scalar_value_type(_mm256_get0_pd(value_r), _mm256_get0_pd(value_i)); }
        else if (i == 1) { return scalar_value_type(_mm256_get1_pd(value_r), _mm256_get1_pd(value_i)); }
        else if (i == 2) { return scalar_value_type(_mm256_get2_pd(value_r), _mm256_get2_pd(value_i)); }
        else             { return scalar_value_type(_mm256_get3_pd(value_r), _mm256_get3_pd(value_i)); }
    }

    FASTOR_INLINE SIMDVector<double,simd_abi::avx> real() const {
        return value_r;
    }
    FASTOR_INLINE SIMDVector<double,simd_abi::avx> imag() const {
        return value_i;
    }

    FASTOR_INLINE void set(std::complex<double> num) {
        value_r = _mm256_set1_pd(num.real());
        value_i = _mm256_set1_pd(num.imag());
    }
    FASTOR_INLINE void set(scalar_value_type num0, scalar_value_type num1,
                                      scalar_value_type num2, scalar_value_type num3) {
        const scalar_value_type tmp[Size] = {num0,num1,num2,num3};
        complex_unaligned_load(tmp);
    }

    // In-place operators
    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator+=(U num) {
        value_r = _mm256_add_pd(value_r,_mm256_set1_pd(num));
    }
    FASTOR_INLINE void operator+=(scalar_value_type num) {
        *this += vector_type(num);
    }
    FASTOR_INLINE void operator+=(const vector_type &a) {
        value_r = _mm256_add_pd(value_r,a.value_r);
        value_i = _mm256_add_pd(value_i,a.value_i);
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator-=(U num) {
        value_r = _mm256_sub_pd(value_r,_mm256_set1_pd(num));
    }
    FASTOR_INLINE void operator-=(scalar_value_type num) {
        *this -= vector_type(num);
    }
    FASTOR_INLINE void operator-=(const vector_type &a) {
        value_r = _mm256_sub_pd(value_r,a.value_r);
        value_i = _mm256_sub_pd(value_i,a.value_i);
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator*=(U num) {
        __m256d val = _mm256_set1_pd(num);
        value_r = _mm256_mul_pd(value_r,val);
        value_i = _mm256_mul_pd(value_i,val);
    }
    FASTOR_INLINE void operator*=(scalar_value_type num) {
        *this *= vector_type(num);
    }
    FASTOR_INLINE void operator*=(const vector_type &a) {
#ifdef FASTOR_FMA_IMPL
        __m256d tmp = _mm256_fmsub_pd(value_r,a.value_r,_mm256_mul_pd(value_i,a.value_i));
        value_i     = _mm256_fmadd_pd(value_r,a.value_i,_mm256_mul_pd(value_i,a.value_r));
#else
        __m256d tmp = _mm256_sub_pd(_mm256_mul_pd(value_r,a.value_r),_mm256_mul_pd(value_i,a.value_i));
        value_i     = _mm256_add_pd(_mm256_mul_pd(value_r,a.value_i),_mm256_mul_pd(value_i,a.value_r));
#endif
        value_r = tmp;
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator/=(U num) {
        __m256d val = _mm256_set1_pd(num);
        value_r = _mm256_div_pd(value_r,val);
        value_i = _mm256_div_pd(value_i,val);
    }
    FASTOR_INLINE void operator/=(scalar_value_type num) {
        *this /= vector_type(num);
    }
    FASTOR_INLINE void operator/=(const vector_type &a) {
        __m256d tmp = value_r;
#ifdef FASTOR_FMA_IMPL
        value_r     = _mm256_fmadd_pd(value_r  , a.value_r, _mm256_mul_pd(value_i,a.value_i));
        value_i     = _mm256_fmsub_pd(value_i  , a.value_r, _mm256_mul_pd(tmp,a.value_i));
        __m256d den = _mm256_fmadd_pd(a.value_r, a.value_r, _mm256_mul_pd(a.value_i,a.value_i));
#else
        value_r     = _mm256_add_pd(_mm256_mul_pd(value_r  , a.value_r), _mm256_mul_pd(value_i,a.value_i));
        value_i     = _mm256_sub_pd(_mm256_mul_pd(value_i  , a.value_r), _mm256_mul_pd(tmp,a.value_i));
        __m256d den = _mm256_add_pd(_mm256_mul_pd(a.value_r, a.value_r), _mm256_mul_pd(a.value_i,a.value_i));
#endif
        value_r     = _mm256_div_pd(value_r,den);
        value_i     = _mm256_div_pd(value_i,den);
    }
    // end of in-place operators

    FASTOR_INLINE scalar_value_type sum() const {
        return scalar_value_type(_mm256_sum_pd(value_r),_mm256_sum_pd(value_i));
    }
    FASTOR_INLINE scalar_value_type product() const {
        vector_type tmp(*this);
        return tmp[0]*tmp[1]*tmp[2]*tmp[3];
    }
    FASTOR_INLINE vector_type reverse() const {
        vector_type out;
        out.value_r = _mm256_reverse_pd(value_r);
        out.value_i = _mm256_reverse_pd(value_i);
        return out;
    }
    /* Actual magnitude - Note that this is a vertical operation */
    FASTOR_INLINE SIMDVector<double,simd_abi::avx> magnitude() const {
#ifdef FASTOR_FMA_IMPL
        return _mm256_sqrt_pd(_mm256_fmadd_pd(value_r,value_r,_mm256_mul_pd(value_i,value_i)));
#else
        return _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(value_r,value_r),_mm256_mul_pd(value_i,value_i)));
#endif
    }
    /* STL compliant squared norm - Note that this is a vertical operation */
    FASTOR_INLINE SIMDVector<double,simd_abi::avx> norm() const {
#ifdef FASTOR_FMA_IMPL
        return _mm256_fmadd_pd(value_r,value_r,_mm256_mul_pd(value_i,value_i));
#else
        return _mm256_add_pd(_mm256_mul_pd(value_r,value_r),_mm256_mul_pd(value_i,value_i));
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
    FASTOR_INLINE void complex_aligned_load(const std::complex<double> *data) {
        __m256d lo = _mm256_load_pd(reinterpret_cast<const double*>(data  ));
        __m256d hi = _mm256_load_pd(reinterpret_cast<const double*>(data+2));
        arrange_from_load(value_r, value_i, lo, hi);
    }
    FASTOR_INLINE void complex_unaligned_load(const std::complex<double> *data) {
        __m256d lo = _mm256_loadu_pd(reinterpret_cast<const double*>(data  ));
        __m256d hi = _mm256_loadu_pd(reinterpret_cast<const double*>(data+2));
        arrange_from_load(value_r, value_i, lo, hi);
    }

    FASTOR_INLINE void complex_aligned_store(std::complex<double> *data) const {
        __m256d lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        _mm256_store_pd(reinterpret_cast<double*>(data  ), lo);
        _mm256_store_pd(reinterpret_cast<double*>(data+2), hi);
    }
    FASTOR_INLINE void complex_unaligned_store(std::complex<double> *data) const {
        __m256d lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        _mm256_storeu_pd(reinterpret_cast<double*>(data  ), lo);
        _mm256_storeu_pd(reinterpret_cast<double*>(data+2), hi);
    }

    FASTOR_INLINE void complex_mask_aligned_load(const scalar_value_type *data, uint8_t mask) {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m256d lo, hi;
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        lo = _mm256_mask_load_pd(lo, mask0, reinterpret_cast<const double*>(data  ));
        hi = _mm256_mask_load_pd(hi, mask1, reinterpret_cast<const double*>(data+2));
        arrange_from_load(value_r, value_i, lo, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        value_r = _mm256_setzero_pd();
        value_i = _mm256_setzero_pd();
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                ((double*)&value_r)[Size - i - 1] = data[Size - i - 1].real();
                ((double*)&value_i)[Size - i - 1] = data[Size - i - 1].imag();
            }
        }
#endif
    }
    FASTOR_INLINE void complex_mask_unaligned_load(const scalar_value_type *data, uint8_t mask) {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m256d lo, hi;
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        lo = _mm256_mask_loadu_pd(lo, mask0, reinterpret_cast<const double*>(data  ));
        hi = _mm256_mask_loadu_pd(hi, mask1, reinterpret_cast<const double*>(data+2));
        arrange_from_load(value_r, value_i, lo, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        value_r = _mm256_setzero_pd();
        value_i = _mm256_setzero_pd();
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                ((double*)&value_r)[Size - i - 1] = data[Size - i - 1].real();
                ((double*)&value_i)[Size - i - 1] = data[Size - i - 1].imag();
            }
        }
#endif
    }

    FASTOR_INLINE void complex_mask_aligned_store(scalar_value_type *data, uint8_t mask) const {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m256d lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        _mm256_mask_store_pd(reinterpret_cast<double*>(data  ), mask0, lo);
        _mm256_mask_store_pd(reinterpret_cast<double*>(data+2), mask1, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                double _real = ((const double*)&value_r)[Size - i - 1];
                double _imag = ((const double*)&value_i)[Size - i - 1];
                data[Size - i - 1] = std::complex<double>(_real,_imag);
            }
            else {
                data[Size - i - 1] = std::complex<double>(0,0);
            }
        }
#endif
    }
    FASTOR_INLINE void complex_mask_unaligned_store(scalar_value_type *data, uint8_t mask) const {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m256d lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        _mm256_mask_storeu_pd(reinterpret_cast<double*>(data  ), mask0, lo);
        _mm256_mask_storeu_pd(reinterpret_cast<double*>(data+2), mask1, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                double _real = ((const double*)&value_r)[Size - i - 1];
                double _imag = ((const double*)&value_i)[Size - i - 1];
                data[Size - i - 1] = std::complex<double>(_real,_imag);
            }
            else {
                data[Size - i - 1] = std::complex<double>(0,0);
            }
        }
#endif
    }
};

FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<std::complex<double>,simd_abi::avx> a) {
    // ICC crashes without a copy
    const __m256d vr = a.value_r;
    const __m256d vi = a.value_i;
    const double* value_r = reinterpret_cast<const double*>(&vr);
    const double* value_i = reinterpret_cast<const double*>(&vi);
    os << "[" << value_r[0] <<  signum_string(value_i[0]) << std::abs(value_i[0]) << "j, "
              << value_r[1] <<  signum_string(value_i[1]) << std::abs(value_i[1]) << "j, "
              << value_r[2] <<  signum_string(value_i[2]) << std::abs(value_i[2]) << "j, "
              << value_r[3] <<  signum_string(value_i[3]) << std::abs(value_i[3]) << "j" << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator+(const SIMDVector<std::complex<double>,simd_abi::avx> &a, const SIMDVector<std::complex<double>,simd_abi::avx> &b) {
    SIMDVector<std::complex<double>,simd_abi::avx> out;
    out.value_r = _mm256_add_pd(a.value_r,b.value_r);
    out.value_i = _mm256_add_pd(a.value_i,b.value_i);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator+(const SIMDVector<std::complex<double>,simd_abi::avx> &a, std::complex<double> b) {
    return a + SIMDVector<std::complex<double>,simd_abi::avx>(b);
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator+(std::complex<double> a, const SIMDVector<std::complex<double>,simd_abi::avx> &b) {
    return SIMDVector<std::complex<double>,simd_abi::avx>(a) + b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator+(const SIMDVector<std::complex<double>,simd_abi::avx> &a, U b) {
    SIMDVector<std::complex<double>,simd_abi::avx> out(a);
    out.value_r = _mm256_add_pd(a.value_r,_mm256_set1_pd(b));
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator+(U a, const SIMDVector<std::complex<double>,simd_abi::avx> &b) {
    SIMDVector<std::complex<double>,simd_abi::avx> out(b);
    out.value_r = _mm256_add_pd(_mm256_set1_pd(a), b.value_r);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator+(const SIMDVector<std::complex<double>,simd_abi::avx> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator-(const SIMDVector<std::complex<double>,simd_abi::avx> &a, const SIMDVector<std::complex<double>,simd_abi::avx> &b) {
    SIMDVector<std::complex<double>,simd_abi::avx> out;
    out.value_r = _mm256_sub_pd(a.value_r,b.value_r);
    out.value_i = _mm256_sub_pd(a.value_i,b.value_i);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator-(const SIMDVector<std::complex<double>,simd_abi::avx> &a, std::complex<double> b) {
    return a - SIMDVector<std::complex<double>,simd_abi::avx>(b);
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator-(std::complex<double> a, const SIMDVector<std::complex<double>,simd_abi::avx> &b) {
    return SIMDVector<std::complex<double>,simd_abi::avx>(a) - b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator-(const SIMDVector<std::complex<double>,simd_abi::avx> &a, U b) {
    SIMDVector<std::complex<double>,simd_abi::avx> out(a);
    out.value_r = _mm256_sub_pd(a.value_r,_mm256_set1_pd(b));
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator-(U a, const SIMDVector<std::complex<double>,simd_abi::avx> &b) {
    return SIMDVector<std::complex<double>,simd_abi::avx>(std::complex<double>(a,0)) - b;
}
/* This is negation and not complex conjugate  */
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator-(const SIMDVector<std::complex<double>,simd_abi::avx> &a) {
    SIMDVector<std::complex<double>,simd_abi::avx> out;
    out.value_r = _mm256_neg_pd(a.value_r);
    out.value_i = _mm256_neg_pd(a.value_i);
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator*(const SIMDVector<std::complex<double>,simd_abi::avx> &a, const SIMDVector<std::complex<double>,simd_abi::avx> &b) {
    SIMDVector<std::complex<double>,simd_abi::avx> out;
#ifdef FASTOR_FMA_IMPL
    out.value_r = _mm256_fmsub_pd(a.value_r,b.value_r,_mm256_mul_pd(a.value_i,b.value_i));
    out.value_i = _mm256_fmadd_pd(a.value_r,b.value_i,_mm256_mul_pd(a.value_i,b.value_r));
#else
    out.value_r = _mm256_sub_pd(_mm256_mul_pd(a.value_r,b.value_r),_mm256_mul_pd(a.value_i,b.value_i));
    out.value_i = _mm256_add_pd(_mm256_mul_pd(a.value_r,b.value_i),_mm256_mul_pd(a.value_i,b.value_r));
#endif
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator*(const SIMDVector<std::complex<double>,simd_abi::avx> &a, std::complex<double> b) {
    return a * SIMDVector<std::complex<double>,simd_abi::avx>(b);
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator*(std::complex<double> a, const SIMDVector<std::complex<double>,simd_abi::avx> &b) {
    return SIMDVector<std::complex<double>,simd_abi::avx>(a) * b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator*(const SIMDVector<std::complex<double>,simd_abi::avx> &a, U b) {
    SIMDVector<std::complex<double>,simd_abi::avx> out;
    __m256d val = _mm256_set1_pd(b);
    out.value_r = _mm256_mul_pd(a.value_r,val);
    out.value_i = _mm256_mul_pd(a.value_i,val);
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator*(U a, const SIMDVector<std::complex<double>,simd_abi::avx> &b) {
    SIMDVector<std::complex<double>,simd_abi::avx> out;
    __m256d val = _mm256_set1_pd(a);
    out.value_r = _mm256_mul_pd(val,b.value_r);
    out.value_i = _mm256_mul_pd(val,b.value_i);
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator/(const SIMDVector<std::complex<double>,simd_abi::avx> &a, const SIMDVector<std::complex<double>,simd_abi::avx> &b) {
    SIMDVector<std::complex<double>,simd_abi::avx> out;
#ifdef FASTOR_FMA_IMPL
    out.value_r = _mm256_fmadd_pd(a.value_r,b.value_r,_mm256_mul_pd(a.value_i,b.value_i));
    out.value_i = _mm256_fmsub_pd(a.value_i,b.value_r,_mm256_mul_pd(a.value_r,b.value_i));
    __m256d den = _mm256_fmadd_pd(b.value_r,b.value_r,_mm256_mul_pd(b.value_i,b.value_i));
#else
    out.value_r = _mm256_add_pd(_mm256_mul_pd(a.value_r,b.value_r),_mm256_mul_pd(a.value_i,b.value_i));
    out.value_i = _mm256_sub_pd(_mm256_mul_pd(a.value_i,b.value_r),_mm256_mul_pd(a.value_r,b.value_i));
    __m256d den = _mm256_add_pd(_mm256_mul_pd(b.value_r,b.value_r),_mm256_mul_pd(b.value_i,b.value_i));
#endif
    out.value_r = _mm256_div_pd(out.value_r,den);
    out.value_i = _mm256_div_pd(out.value_i,den);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator/(const SIMDVector<std::complex<double>,simd_abi::avx> &a, std::complex<double> b) {
    return a / SIMDVector<std::complex<double>,simd_abi::avx>(b);
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator/(std::complex<double> a, const SIMDVector<std::complex<double>,simd_abi::avx> &b) {
    return SIMDVector<std::complex<double>,simd_abi::avx>(a) / b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator/(const SIMDVector<std::complex<double>,simd_abi::avx> &a, U b) {
    SIMDVector<std::complex<double>,simd_abi::avx> out;
    __m256d val = _mm256_set1_pd(b);
    out.value_r = _mm256_div_pd(a.value_r,val);
    out.value_i = _mm256_div_pd(a.value_i,val);
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
operator/(U a, const SIMDVector<std::complex<double>,simd_abi::avx> &b) {
    return SIMDVector<std::complex<double>,simd_abi::avx>(std::complex<double>(a,0)) / b;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
rcp(const SIMDVector<std::complex<double>,simd_abi::avx> &a) {
    SIMDVector<std::complex<double>,simd_abi::avx> out;
#ifdef FASTOR_FMA_IMPL
    __m256d den = _mm256_fmadd_pd(a.value_r,a.value_r,_mm256_mul_pd(a.value_i,a.value_i));
#else
    __m256d den = _mm256_add_pd(_mm256_mul_pd(a.value_r,a.value_r),_mm256_mul_pd(a.value_i,a.value_i));
#endif
    out.value_r = _mm256_div_pd(out.value_r,den);
    out.value_i = _mm256_neg_pd(_mm256_div_pd(out.value_i,den));
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
sqrt(const SIMDVector<std::complex<double>,simd_abi::avx> &a) = delete;

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
rsqrt(const SIMDVector<std::complex<double>,simd_abi::avx> &a) = delete;

/* This intentionally return a complex vector */
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
abs(const SIMDVector<std::complex<double>,simd_abi::avx> &a) {
    SIMDVector<std::complex<double>,simd_abi::avx> out;
    out.value_r = a.magnitude().value;
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
conj(const SIMDVector<std::complex<double>,simd_abi::avx> &a) {
    SIMDVector<std::complex<double>,simd_abi::avx> out(a);
    out.value_i = _mm256_neg_pd(out.value_i);
    return out;
}

/* Argument or phase angle */
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::avx>
arg(const SIMDVector<std::complex<double>,simd_abi::avx> &a) {
    SIMDVector<std::complex<double>,simd_abi::avx> out(a);
    for (FASTOR_INDEX i=0UL; i<4UL; ++i) {
       ((double*)&out.value_r)[i] = std::atan2(((double*)&a.value_i)[i],((double*)&a.value_r)[i]);
    }
    return out;
}
//------------------------------------------------------------------------------------------------------------

#endif




// SSE VERSION
//------------------------------------------------------------------------------------------------------------

#ifdef FASTOR_SSE2_IMPL

template <>
struct SIMDVector<std::complex<double>, simd_abi::sse> {
    using vector_type = SIMDVector<std::complex<double>, simd_abi::sse>;
    using value_type = __m128d;
    using scalar_value_type = std::complex<double>;
    using abi_type = simd_abi::sse;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<double,simd_abi::sse>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<double,simd_abi::sse>>::value;}

    FASTOR_INLINE SIMDVector() : value_r(_mm_setzero_pd()), value_i(_mm_setzero_pd()) {}
    FASTOR_INLINE SIMDVector(std::complex<double> num) {
        value_r = _mm_set1_pd(num.real());
        value_i = _mm_set1_pd(num.imag());
    }
    FASTOR_INLINE SIMDVector(value_type reg0, value_type reg1) : value_r(reg0), value_i(reg1) {}
    FASTOR_INLINE SIMDVector(const std::complex<double> *data, bool Aligned=true) {
        if (Aligned)
            complex_aligned_load(data);
        else
            complex_unaligned_load(data);
    }

    FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse> operator=(std::complex<double> num) {
        value_r = _mm_set1_pd(num.real());
        value_i = _mm_set1_pd(num.imag());
        return *this;
    }

    FASTOR_INLINE void load(const std::complex<double> *data, bool Aligned=true) {
        if (Aligned)
            complex_aligned_load(data);
        else
            complex_unaligned_load(data);
    }
    FASTOR_INLINE void store(std::complex<double> *data, bool Aligned=true) const {
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
        if      (i == 0) { return scalar_value_type(_mm_get0_pd(value_r), _mm_get0_pd(value_i)); }
        else             { return scalar_value_type(_mm_get1_pd(value_r), _mm_get1_pd(value_i)); }
    }

    FASTOR_INLINE SIMDVector<double,simd_abi::sse> real() const {
        return value_r;
    }
    FASTOR_INLINE SIMDVector<double,simd_abi::sse> imag() const {
        return value_i;
    }

    FASTOR_INLINE void set(std::complex<double> num) {
        value_r = _mm_set1_pd(num.real());
        value_i = _mm_set1_pd(num.imag());
    }
    FASTOR_INLINE void set(scalar_value_type num0, scalar_value_type num1) {
        const scalar_value_type tmp[Size] = {num0,num1};
        complex_unaligned_load(tmp);
    }

    // In-place operators
    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator+=(U num) {
        value_r = _mm_add_pd(value_r,_mm_set1_pd(num));
    }
    FASTOR_INLINE void operator+=(scalar_value_type num) {
        *this += vector_type(num);
    }
    FASTOR_INLINE void operator+=(const vector_type &a) {
        value_r = _mm_add_pd(value_r,a.value_r);
        value_i = _mm_add_pd(value_i,a.value_i);
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator-=(U num) {
        value_r = _mm_sub_pd(value_r,_mm_set1_pd(num));
    }
    FASTOR_INLINE void operator-=(scalar_value_type num) {
        *this -= vector_type(num);
    }
    FASTOR_INLINE void operator-=(const vector_type &a) {
        value_r = _mm_sub_pd(value_r,a.value_r);
        value_i = _mm_sub_pd(value_i,a.value_i);
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator*=(U num) {
        __m128d val = _mm_set1_pd(num);
        value_r = _mm_mul_pd(value_r,val);
        value_i = _mm_mul_pd(value_i,val);
    }
    FASTOR_INLINE void operator*=(scalar_value_type num) {
        *this *= vector_type(num);
    }
    FASTOR_INLINE void operator*=(const vector_type &a) {
#ifdef FASTOR_FMA_IMPL
        __m128d tmp = _mm_fmsub_pd(value_r,a.value_r,_mm_mul_pd(value_i,a.value_i));
        value_i     = _mm_fmadd_pd(value_r,a.value_i,_mm_mul_pd(value_i,a.value_r));
#else
        __m128d tmp = _mm_sub_pd(_mm_mul_pd(value_r,a.value_r),_mm_mul_pd(value_i,a.value_i));
        value_i     = _mm_add_pd(_mm_mul_pd(value_r,a.value_i),_mm_mul_pd(value_i,a.value_r));
#endif
        value_r = tmp;
    }

    template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator/=(U num) {
        __m128d val = _mm_set1_pd(num);
        value_r = _mm_div_pd(value_r,val);
        value_i = _mm_div_pd(value_i,val);
    }
    FASTOR_INLINE void operator/=(scalar_value_type num) {
        *this /= vector_type(num);
    }
    FASTOR_INLINE void operator/=(const vector_type &a) {
        __m128d tmp = value_r;
#ifdef FASTOR_FMA_IMPL
        value_r     = _mm_fmadd_pd(value_r  , a.value_r, _mm_mul_pd(value_i,a.value_i));
        value_i     = _mm_fmsub_pd(value_i  , a.value_r, _mm_mul_pd(tmp,a.value_i));
        __m128d den = _mm_fmadd_pd(a.value_r, a.value_r, _mm_mul_pd(a.value_i,a.value_i));
#else
        value_r     = _mm_add_pd(_mm_mul_pd(value_r  , a.value_r), _mm_mul_pd(value_i,a.value_i));
        value_i     = _mm_sub_pd(_mm_mul_pd(value_i  , a.value_r), _mm_mul_pd(tmp,a.value_i));
        __m128d den = _mm_add_pd(_mm_mul_pd(a.value_r, a.value_r), _mm_mul_pd(a.value_i,a.value_i));
#endif
        value_r     = _mm_div_pd(value_r,den);
        value_i     = _mm_div_pd(value_i,den);
    }
    // end of in-place operators

    FASTOR_INLINE scalar_value_type sum() const {
        return scalar_value_type(_mm_sum_pd(value_r),_mm_sum_pd(value_i));
    }
    FASTOR_INLINE scalar_value_type product() const {
        vector_type tmp(*this);
        // Multiply vector of complex numbers
        tmp *= tmp.reverse();
        // Alternatively if relied on the compiler
        // return tmp[0]*tmp[1];
        return tmp[0];
    }
    FASTOR_INLINE vector_type reverse() const {
        vector_type out;
        out.value_r = _mm_reverse_pd(value_r);
        out.value_i = _mm_reverse_pd(value_i);
        return out;
    }
    /* Actual magnitude - Note that this is a vertical operation */
    FASTOR_INLINE SIMDVector<double,simd_abi::sse> magnitude() const {
#ifdef FASTOR_FMA_IMPL
        return _mm_sqrt_pd(_mm_fmadd_pd(value_r,value_r,_mm_mul_pd(value_i,value_i)));
#else
        return _mm_sqrt_pd(_mm_add_pd(_mm_mul_pd(value_r,value_r),_mm_mul_pd(value_i,value_i)));
#endif
    }
    /* STL compliant squared norm - Note that this is a vertical operation */
    FASTOR_INLINE SIMDVector<double,simd_abi::sse> norm() const {
#ifdef FASTOR_FMA_IMPL
        return _mm_fmadd_pd(value_r,value_r,_mm_mul_pd(value_i,value_i));
#else
        return _mm_add_pd(_mm_mul_pd(value_r,value_r),_mm_mul_pd(value_i,value_i));
#endif
    }
    // Magnitude based minimum
    FASTOR_INLINE scalar_value_type minimum() const {
        vector_type out;
        SIMDVector<double,simd_abi::sse> tmp(this->norm());
        if ( _mm_get0_pd(tmp.value) < _mm_get1_pd(tmp.value) ) {
            return scalar_value_type(_mm_get0_pd(value_r),_mm_get0_pd(value_i));
        }
        else {
            return scalar_value_type(_mm_get1_pd(value_r),_mm_get1_pd(value_i));
        }
    }
    // Magnitude based maximum
    FASTOR_INLINE scalar_value_type maximum() const {
        vector_type out;
        SIMDVector<double,simd_abi::sse> tmp(this->norm());
        if ( _mm_get0_pd(tmp.value) > _mm_get1_pd(tmp.value) ) {
            return scalar_value_type(_mm_get0_pd(value_r),_mm_get0_pd(value_i));
        }
        else {
            return scalar_value_type(_mm_get1_pd(value_r),_mm_get1_pd(value_i));
        }
    }

    FASTOR_INLINE scalar_value_type dot(const vector_type &other) const {
        vector_type out(*this);
        out *= other;
        return out.sum();
    }

    value_type value_r;
    value_type value_i;

protected:
    FASTOR_INLINE void complex_aligned_load(const std::complex<double> *data) {
        __m128d lo = _mm_load_pd(reinterpret_cast<const double*>(data  ));
        __m128d hi = _mm_load_pd(reinterpret_cast<const double*>(data+1));
        arrange_from_load(value_r, value_i, lo, hi);
    }
    FASTOR_INLINE void complex_unaligned_load(const std::complex<double> *data) {
        __m128d lo = _mm_loadu_pd(reinterpret_cast<const double*>(data  ));
        __m128d hi = _mm_loadu_pd(reinterpret_cast<const double*>(data+1));
        arrange_from_load(value_r, value_i, lo, hi);
    }

    FASTOR_INLINE void complex_aligned_store(std::complex<double> *data) const {
        __m128d lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        _mm_store_pd(reinterpret_cast<double*>(data  ),lo);
        _mm_store_pd(reinterpret_cast<double*>(data+1),hi);
    }
    FASTOR_INLINE void complex_unaligned_store(std::complex<double> *data) const {
        __m128d lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        _mm_storeu_pd(reinterpret_cast<double*>(data  ),lo);
        _mm_storeu_pd(reinterpret_cast<double*>(data+1),hi);
    }

    FASTOR_INLINE void complex_mask_aligned_load(const scalar_value_type *data, uint8_t mask) {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m128d lo, hi;
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        lo = _mm_mask_load_pd(lo, mask0, reinterpret_cast<const double*>(data  ));
        hi = _mm_mask_load_pd(hi, mask1, reinterpret_cast<const double*>(data+1));
        arrange_from_load(value_r, value_i, lo, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        value_r = _mm_setzero_pd();
        value_i = _mm_setzero_pd();
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                ((double*)&value_r)[Size - i - 1] = data[Size - i - 1].real();
                ((double*)&value_i)[Size - i - 1] = data[Size - i - 1].imag();
            }
        }
#endif
    }
    FASTOR_INLINE void complex_mask_unaligned_load(const scalar_value_type *data, uint8_t mask) {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m128d lo, hi;
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        lo = _mm_mask_loadu_pd(lo, mask0, reinterpret_cast<const double*>(data  ));
        hi = _mm_mask_loadu_pd(hi, mask1, reinterpret_cast<const double*>(data+1));
        arrange_from_load(value_r, value_i, lo, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        value_r = _mm_setzero_pd();
        value_i = _mm_setzero_pd();
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                ((double*)&value_r)[Size - i - 1] = data[Size - i - 1].real();
                ((double*)&value_i)[Size - i - 1] = data[Size - i - 1].imag();
            }
        }
#endif
    }

    FASTOR_INLINE void complex_mask_aligned_store(scalar_value_type *data, uint8_t mask) const {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m128d lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        _mm_mask_store_pd(reinterpret_cast<double*>(data  ), mask0, lo);
        _mm_mask_store_pd(reinterpret_cast<double*>(data+1), mask1, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                double _real = ((const double*)&value_r)[Size - i - 1];
                double _imag = ((const double*)&value_i)[Size - i - 1];
                data[Size - i - 1] = std::complex<double>(_real,_imag);
            }
            else {
                data[Size - i - 1] = std::complex<double>(0,0);
            }
        }
#endif
    }
    FASTOR_INLINE void complex_mask_unaligned_store(scalar_value_type *data, uint8_t mask) const {
#ifdef FASTOR_HAS_AVX512_MASKS
        __m128d lo, hi;
        arrange_for_store(lo, hi, value_r, value_i);
        uint8_t mask0, mask1;
        split_mask<Size>(mask, mask0, mask1);
        _mm_mask_storeu_pd(reinterpret_cast<double*>(data  ), mask0, lo);
        _mm_mask_storeu_pd(reinterpret_cast<double*>(data+1), mask1, hi);
#else
        int maska[Size];
        mask_to_array(mask,maska);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                double _real = ((const double*)&value_r)[Size - i - 1];
                double _imag = ((const double*)&value_i)[Size - i - 1];
                data[Size - i - 1] = std::complex<double>(_real,_imag);
            }
            else {
                data[Size - i - 1] = std::complex<double>(0,0);
            }
        }
#endif
    }

};

FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<std::complex<double>,simd_abi::sse> a) {
    // ICC crashes without a copy
    const __m128d vr = a.value_r;
    const __m128d vi = a.value_i;
    const double* value_r = reinterpret_cast<const double*>(&vr);
    const double* value_i = reinterpret_cast<const double*>(&vi);
    os << "[" << value_r[0] <<  signum_string(value_i[0]) << std::abs(value_i[0]) << "j, "
              << value_r[1] <<  signum_string(value_i[1]) << std::abs(value_i[1]) << "j" << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator+(const SIMDVector<std::complex<double>,simd_abi::sse> &a, const SIMDVector<std::complex<double>,simd_abi::sse> &b) {
    SIMDVector<std::complex<double>,simd_abi::sse> out;
    out.value_r = _mm_add_pd(a.value_r,b.value_r);
    out.value_i = _mm_add_pd(a.value_i,b.value_i);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator+(const SIMDVector<std::complex<double>,simd_abi::sse> &a, std::complex<double> b) {
    return a + SIMDVector<std::complex<double>,simd_abi::sse>(b);
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator+(std::complex<double> a, const SIMDVector<std::complex<double>,simd_abi::sse> &b) {
    return SIMDVector<std::complex<double>,simd_abi::sse>(a) + b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator+(const SIMDVector<std::complex<double>,simd_abi::sse> &a, U b) {
    SIMDVector<std::complex<double>,simd_abi::sse> out(a);
    out.value_r = _mm_add_pd(a.value_r,_mm_set1_pd(b));
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator+(U a, const SIMDVector<std::complex<double>,simd_abi::sse> &b) {
    SIMDVector<std::complex<double>,simd_abi::sse> out(b);
    out.value_r = _mm_add_pd(_mm_set1_pd(a), b.value_r);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator+(const SIMDVector<std::complex<double>,simd_abi::sse> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator-(const SIMDVector<std::complex<double>,simd_abi::sse> &a, const SIMDVector<std::complex<double>,simd_abi::sse> &b) {
    SIMDVector<std::complex<double>,simd_abi::sse> out;
    out.value_r = _mm_sub_pd(a.value_r,b.value_r);
    out.value_i = _mm_sub_pd(a.value_i,b.value_i);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator-(const SIMDVector<std::complex<double>,simd_abi::sse> &a, std::complex<double> b) {
    return a - SIMDVector<std::complex<double>,simd_abi::sse>(b);
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator-(std::complex<double> a, const SIMDVector<std::complex<double>,simd_abi::sse> &b) {
    return SIMDVector<std::complex<double>,simd_abi::sse>(a) - b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator-(const SIMDVector<std::complex<double>,simd_abi::sse> &a, U b) {
    SIMDVector<std::complex<double>,simd_abi::sse> out(a);
    out.value_r = _mm_sub_pd(a.value_r,_mm_set1_pd(b));
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator-(U a, const SIMDVector<std::complex<double>,simd_abi::sse> &b) {
    return SIMDVector<std::complex<double>,simd_abi::sse>(std::complex<double>(a,0)) - b;
}
/* This is negation and not complex conjugate  */
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator-(const SIMDVector<std::complex<double>,simd_abi::sse> &a) {
    SIMDVector<std::complex<double>,simd_abi::sse> out;
    out.value_r = _mm_neg_pd(a.value_r);
    out.value_i = _mm_neg_pd(a.value_i);
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator*(const SIMDVector<std::complex<double>,simd_abi::sse> &a, const SIMDVector<std::complex<double>,simd_abi::sse> &b) {
    SIMDVector<std::complex<double>,simd_abi::sse> out;
#ifdef FASTOR_FMA_IMPL
    out.value_r = _mm_fmsub_pd(a.value_r,b.value_r,_mm_mul_pd(a.value_i,b.value_i));
    out.value_i = _mm_fmadd_pd(a.value_r,b.value_i,_mm_mul_pd(a.value_i,b.value_r));
#else
    out.value_r = _mm_sub_pd(_mm_mul_pd(a.value_r,b.value_r),_mm_mul_pd(a.value_i,b.value_i));
    out.value_i = _mm_add_pd(_mm_mul_pd(a.value_r,b.value_i),_mm_mul_pd(a.value_i,b.value_r));
#endif
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator*(const SIMDVector<std::complex<double>,simd_abi::sse> &a, std::complex<double> b) {
    return a * SIMDVector<std::complex<double>,simd_abi::sse>(b);
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator*(std::complex<double> a, const SIMDVector<std::complex<double>,simd_abi::sse> &b) {
    return SIMDVector<std::complex<double>,simd_abi::sse>(a) * b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator*(const SIMDVector<std::complex<double>,simd_abi::sse> &a, U b) {
    SIMDVector<std::complex<double>,simd_abi::sse> out;
    __m128d val = _mm_set1_pd(b);
    out.value_r = _mm_mul_pd(a.value_r,val);
    out.value_i = _mm_mul_pd(a.value_i,val);
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator*(U a, const SIMDVector<std::complex<double>,simd_abi::sse> &b) {
    SIMDVector<std::complex<double>,simd_abi::sse> out;
    __m128d val = _mm_set1_pd(a);
    out.value_r = _mm_mul_pd(val,b.value_r);
    out.value_i = _mm_mul_pd(val,b.value_i);
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator/(const SIMDVector<std::complex<double>,simd_abi::sse> &a, const SIMDVector<std::complex<double>,simd_abi::sse> &b) {
    SIMDVector<std::complex<double>,simd_abi::sse> out;
#ifdef FASTOR_FMA_IMPL
    out.value_r = _mm_fmadd_pd(a.value_r,b.value_r,_mm_mul_pd(a.value_i,b.value_i));
    out.value_i = _mm_fmsub_pd(a.value_i,b.value_r,_mm_mul_pd(a.value_r,b.value_i));
    __m128d den = _mm_fmadd_pd(b.value_r,b.value_r,_mm_mul_pd(b.value_i,b.value_i));
#else
    out.value_r = _mm_add_pd(_mm_mul_pd(a.value_r,b.value_r),_mm_mul_pd(a.value_i,b.value_i));
    out.value_i = _mm_sub_pd(_mm_mul_pd(a.value_i,b.value_r),_mm_mul_pd(a.value_r,b.value_i));
    __m128d den = _mm_add_pd(_mm_mul_pd(b.value_r,b.value_r),_mm_mul_pd(b.value_i,b.value_i));
#endif
    out.value_r = _mm_div_pd(out.value_r,den);
    out.value_i = _mm_div_pd(out.value_i,den);
    return out;
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator/(const SIMDVector<std::complex<double>,simd_abi::sse> &a, std::complex<double> b) {
    return a / SIMDVector<std::complex<double>,simd_abi::sse>(b);
}
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator/(std::complex<double> a, const SIMDVector<std::complex<double>,simd_abi::sse> &b) {
    return SIMDVector<std::complex<double>,simd_abi::sse>(a) / b;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator/(const SIMDVector<std::complex<double>,simd_abi::sse> &a, U b) {
    SIMDVector<std::complex<double>,simd_abi::sse> out;
    __m128d val = _mm_set1_pd(b);
    out.value_r = _mm_div_pd(a.value_r,val);
    out.value_i = _mm_div_pd(a.value_i,val);
    return out;
}
template<typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
operator/(U a, const SIMDVector<std::complex<double>,simd_abi::sse> &b) {
    return SIMDVector<std::complex<double>,simd_abi::sse>(std::complex<double>(a,0)) / b;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
rcp(const SIMDVector<std::complex<double>,simd_abi::sse> &a) {
    SIMDVector<std::complex<double>,simd_abi::sse> out;
#ifdef FASTOR_FMA_IMPL
    __m128d den = _mm_fmadd_pd(a.value_r,a.value_r,_mm_mul_pd(a.value_i,a.value_i));
#else
    __m128d den = _mm_add_pd(_mm_mul_pd(a.value_r,a.value_r),_mm_mul_pd(a.value_i,a.value_i));
#endif
    out.value_r = _mm_div_pd(out.value_r,den);
    out.value_i = _mm_neg_pd(_mm_div_pd(out.value_i,den));
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
sqrt(const SIMDVector<std::complex<double>,simd_abi::sse> &a) = delete;

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
rsqrt(const SIMDVector<std::complex<double>,simd_abi::sse> &a) = delete;

/* This intentionally return a complex vector */
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
abs(const SIMDVector<std::complex<double>,simd_abi::sse> &a) {
    SIMDVector<std::complex<double>,simd_abi::sse> out;
    out.value_r = a.magnitude().value;
    return out;
}

FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
conj(const SIMDVector<std::complex<double>,simd_abi::sse> &a) {
    SIMDVector<std::complex<double>,simd_abi::sse> out(a);
    out.value_i = _mm_neg_pd(out.value_i);
    return out;
}

/* Argument or phase angle */
FASTOR_INLINE SIMDVector<std::complex<double>,simd_abi::sse>
arg(const SIMDVector<std::complex<double>,simd_abi::sse> &a) {
    SIMDVector<std::complex<double>,simd_abi::sse> out(a);
    for (FASTOR_INDEX i=0UL; i<2UL; ++i) {
       ((double*)&out.value_r)[i] = std::atan2(((double*)&a.value_i)[i],((double*)&a.value_r)[i]);
    }
    return out;
}
//------------------------------------------------------------------------------------------------------------

#endif


} // end of namespace Fastor

#endif // SIMD_VECTOR_COMPLEX_DOUBLE_H
