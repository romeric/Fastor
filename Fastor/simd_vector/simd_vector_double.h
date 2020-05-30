#ifndef SIMD_VECTOR_DOUBLE_H
#define SIMD_VECTOR_DOUBLE_H

#include "Fastor/simd_vector/simd_vector_base.h"

namespace Fastor {

// AVX512 VERSION
//--------------------------------------------------------------------------------------------------
#ifdef FASTOR_AVX512_IMPL

template <>
struct SIMDVector<double, simd_abi::avx512> {
    using value_type = __m512d;
    using scalar_value_type = double;
    using abi_type = simd_abi::avx512;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<double,simd_abi::avx512>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<double,simd_abi::avx512>>::value;}

    FASTOR_INLINE SIMDVector() : value(_mm512_setzero_pd()) {}
    FASTOR_INLINE SIMDVector(double num) : value(_mm512_set1_pd(num)) {}
    FASTOR_INLINE SIMDVector(__m512d regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const double *data, bool Aligned=true) {
        if (Aligned)
            value =_mm512_load_pd(data);
        else
            value = _mm512_loadu_pd(data);
    }

    FASTOR_INLINE SIMDVector<double,simd_abi::avx512> operator=(double num) {
        value = _mm512_set1_pd(num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<double,simd_abi::avx512> operator=(__m512d regi) {
        value = regi;
        return *this;
    }

    FASTOR_INLINE void load(const double *data, bool Aligned=true) {
        if (Aligned)
            value =_mm512_load_pd(data);
        else
            value = _mm512_loadu_pd(data);
    }
    FASTOR_INLINE void store(double *data, bool Aligned=true) const {
        if (Aligned)
            _mm512_store_pd(data,value);
        else
            _mm512_storeu_pd(data,value);
    }

    FASTOR_INLINE void aligned_load(const double *data) {
        value =_mm512_load_pd(data);
    }
    FASTOR_INLINE void aligned_store(double *data) const {
        _mm512_store_pd(data,value);
    }

    FASTOR_INLINE void mask_load(const scalar_value_type *a, uint8_t mask, bool Aligned=false) {
#ifdef FASTOR_HAS_AVX512_MASKS
        if (!Aligned)
            value = _mm512_mask_loadu_pd(value, mask, a);
        else
            value = _mm512_mask_load_pd(value, mask, a);
#else
        // perhaps very inefficient but they never get used
        int maska[Size];
        mask_to_array(mask,maska);
        value = _mm512_setzero_pd();
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
            _mm512_mask_storeu_pd(a, mask, value);
        else
            _mm512_mask_store_pd(a, mask, value);
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

    FASTOR_INLINE double operator[](FASTOR_INDEX i) const {return reinterpret_cast<const double*>(&value)[i];}
    FASTOR_INLINE double operator()(FASTOR_INDEX i) const {return reinterpret_cast<const double*>(&value)[i];}

    FASTOR_INLINE void set(double num) {
        value = _mm512_set1_pd(num);
    }
    FASTOR_INLINE void set(double num0, double num1, double num2, double num3, double num4, double num5, double num6, double num7) {
        value = _mm512_set_pd(num0,num1,num2,num3,num4,num5,num6,num7);
    }
    FASTOR_INLINE void set_sequential(double num0) {
        value = _mm512_setr_pd(num0,num0+1.0,num0+2.0,num0+3.0,num0+4.0,num0+5.0,num0+6.0,num0+7.0);
    }
    FASTOR_INLINE void broadcast(const double *data) {
        // value = _mm512_broadcast_sd(data);
    }

    // In-place operators
    FASTOR_INLINE void operator+=(double num) {
        value = _mm512_add_pd(value,_mm512_set1_pd(num));
    }
    FASTOR_INLINE void operator+=(__m512d regi) {
        value = _mm512_add_pd(value,regi);
    }
    FASTOR_INLINE void operator+=(const SIMDVector<double,simd_abi::avx512> &a) {
        value = _mm512_add_pd(value,a.value);
    }

    FASTOR_INLINE void operator-=(double num) {
        value = _mm512_sub_pd(value,_mm512_set1_pd(num));
    }
    FASTOR_INLINE void operator-=(__m512d regi) {
        value = _mm512_sub_pd(value,regi);
    }
    FASTOR_INLINE void operator-=(const SIMDVector<double,simd_abi::avx512> &a) {
        value = _mm512_sub_pd(value,a.value);
    }

    FASTOR_INLINE void operator*=(double num) {
        value = _mm512_mul_pd(value,_mm512_set1_pd(num));
    }
    FASTOR_INLINE void operator*=(__m512d regi) {
        value = _mm512_mul_pd(value,regi);
    }
    FASTOR_INLINE void operator*=(const SIMDVector<double,simd_abi::avx512> &a) {
        value = _mm512_mul_pd(value,a.value);
    }

    FASTOR_INLINE void operator/=(double num) {
        value = _mm512_div_pd(value,_mm512_set1_pd(num));
    }
    FASTOR_INLINE void operator/=(__m512d regi) {
        value = _mm512_div_pd(value,regi);
    }
    FASTOR_INLINE void operator/=(const SIMDVector<double,simd_abi::avx512> &a) {
        value = _mm512_div_pd(value,a.value);
    }
    // end of in-place operators

    // FASTOR_INLINE SIMDVector<double,simd_abi::avx512> shift(FASTOR_INDEX i) {
    //     SIMDVector<double,simd_abi::avx512> out;
    //     if (i==1)
    //         out.value = _mm512_shift1_pd(value);
    //     else if (i==2)
    //         out.value = _mm512_shift2_pd(value);
    //     else if (i==3)
    //         out.value = _mm512_shift3_pd(value);
    //     return out;
    // }
    FASTOR_INLINE double sum() {
#ifdef FASTOR_HAS_AVX512_REDUCE_ADD
        return _mm512_reduce_add_pd(value);
#else
        __m256d low  = _mm512_castpd512_pd256(value);
        __m256d high = _mm512_extractf64x4_pd(value,0x1);
        return _mm256_sum_pd(_mm256_add_pd(low,high));
#endif
    }
    FASTOR_INLINE double product() {
        __m256d low  = _mm512_castpd512_pd256(value);
        __m256d high = _mm512_extractf64x4_pd(value,1);
        return _mm256_prod_pd(_mm256_mul_pd(low,high));
    }
    FASTOR_INLINE SIMDVector<double,simd_abi::avx512> reverse() {
        return _mm512_reverse_pd(value);
    }
    // FASTOR_INLINE double minimum() {return _mm512_hmin_pd(value);}
    // FASTOR_INLINE double maximum() {return _mm512_hmax_pd(value);}

    FASTOR_INLINE double dot(const SIMDVector<double,simd_abi::avx512> &other) {
        __m512d res =  _mm512_mul_pd(value,other.value);
        __m256d low  = _mm512_castpd512_pd256(res);
        __m256d high = _mm512_extractf64x4_pd(res,1);
        return _mm256_sum_pd(_mm256_add_pd(low,high));
    }

    __m512d value;
};


FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<double,simd_abi::avx512> a) {
    // ICC crashes without a copy
    const __m512d v = a.value;
    const double* value = reinterpret_cast<const double*>(&v);
    os << "[" << value[0] <<  " " << value[1] << " " << value[2] << " " << value[3]
       << " " << value[4] <<  " " << value[5] << " " << value[6] << " " << value[7] << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<double,simd_abi::avx512> operator+(
    const SIMDVector<double,simd_abi::avx512> &a, const SIMDVector<double,simd_abi::avx512> &b) {
    SIMDVector<double,simd_abi::avx512> out;
    out.value = _mm512_add_pd(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> operator+(const SIMDVector<double,simd_abi::avx512> &a, double b) {
    SIMDVector<double,simd_abi::avx512> out;
    out.value = _mm512_add_pd(a.value,_mm512_set1_pd(b));
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> operator+(double a, const SIMDVector<double,simd_abi::avx512> &b) {
    SIMDVector<double,simd_abi::avx512> out;
    out.value = _mm512_add_pd(_mm512_set1_pd(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> operator+(const SIMDVector<double,simd_abi::avx512> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<double,simd_abi::avx512> operator-(
    const SIMDVector<double,simd_abi::avx512> &a, const SIMDVector<double,simd_abi::avx512> &b) {
    SIMDVector<double,simd_abi::avx512> out;
    out.value = _mm512_sub_pd(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> operator-(const SIMDVector<double,simd_abi::avx512> &a, double b) {
    SIMDVector<double,simd_abi::avx512> out;
    out.value = _mm512_sub_pd(a.value,_mm512_set1_pd(b));
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> operator-(double a, const SIMDVector<double,simd_abi::avx512> &b) {
    SIMDVector<double,simd_abi::avx512> out;
    out.value = _mm512_sub_pd(_mm512_set1_pd(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> operator-(const SIMDVector<double,simd_abi::avx512> &b) {
    return _mm512_neg_pd(b.value);
}

FASTOR_INLINE SIMDVector<double,simd_abi::avx512> operator*(
    const SIMDVector<double,simd_abi::avx512> &a, const SIMDVector<double,simd_abi::avx512> &b) {
    SIMDVector<double,simd_abi::avx512> out;
    out.value = _mm512_mul_pd(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> operator*(const SIMDVector<double,simd_abi::avx512> &a, double b) {
    SIMDVector<double,simd_abi::avx512> out;
    out.value = _mm512_mul_pd(a.value,_mm512_set1_pd(b));
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> operator*(double a, const SIMDVector<double,simd_abi::avx512> &b) {
    SIMDVector<double,simd_abi::avx512> out;
    out.value = _mm512_mul_pd(_mm512_set1_pd(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<double,simd_abi::avx512> operator/(
    const SIMDVector<double,simd_abi::avx512> &a, const SIMDVector<double,simd_abi::avx512> &b) {
    SIMDVector<double,simd_abi::avx512> out;
    out.value = _mm512_div_pd(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> operator/(const SIMDVector<double,simd_abi::avx512> &a, double b) {
    SIMDVector<double,simd_abi::avx512> out;
    out.value = _mm512_div_pd(a.value,_mm512_set1_pd(b));
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx512> operator/(double a, const SIMDVector<double,simd_abi::avx512> &b) {
    SIMDVector<double,simd_abi::avx512> out;
    out.value = _mm512_div_pd(_mm512_set1_pd(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<double,simd_abi::avx512> rcp(const SIMDVector<double,simd_abi::avx512> &a) {
    SIMDVector<double,simd_abi::avx512> out;
    out.value = _mm512_rcp14_pd(a.value);
    return out;
}

FASTOR_INLINE SIMDVector<double,simd_abi::avx512> sqrt(const SIMDVector<double,simd_abi::avx512> &a) {
    SIMDVector<double,simd_abi::avx512> out;
    out.value = _mm512_sqrt_pd(a.value);
    return out;
}

FASTOR_INLINE SIMDVector<double,simd_abi::avx512> rsqrt(const SIMDVector<double,simd_abi::avx512> &a) {
    SIMDVector<double,simd_abi::avx512> out;
    out.value = _mm512_rsqrt14_pd(a.value);
    return out;
}

FASTOR_INLINE SIMDVector<double,simd_abi::avx512> abs(const SIMDVector<double,simd_abi::avx512> &a) {
    SIMDVector<double,simd_abi::avx512> out;
#ifdef FASTOR_HAS_AVX512_ABS
    out.value = _mm512_abs_pd(a.value);
#else
    for (FASTOR_INDEX i=0UL; i<8UL; ++i) {
       ((double*)&out.value)[i] = std::abs(((double*)&a.value)[i]);
    }
#endif
    return out;
}

#endif



// AVX VERSION
//--------------------------------------------------------------------------------------------------
#ifdef FASTOR_AVX_IMPL

template <>
struct SIMDVector<double, simd_abi::avx> {
    using value_type = __m256d;
    using scalar_value_type = double;
    using abi_type = simd_abi::avx;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<double,simd_abi::avx>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<double,simd_abi::avx>>::value;}

    FASTOR_INLINE SIMDVector() : value(_mm256_setzero_pd()) {}
    FASTOR_INLINE SIMDVector(double num) : value(_mm256_set1_pd(num)) {}
    FASTOR_INLINE SIMDVector(__m256d regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const double *data, bool Aligned=true) {
        if (Aligned)
            value =_mm256_load_pd(data);
        else
            value = _mm256_loadu_pd(data);
    }

    FASTOR_INLINE SIMDVector<double,simd_abi::avx> operator=(double num) {
        value = _mm256_set1_pd(num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<double,simd_abi::avx> operator=(__m256d regi) {
        value = regi;
        return *this;
    }

    FASTOR_INLINE void load(const double *data, bool Aligned=true) {
        if (Aligned)
            value =_mm256_load_pd(data);
        else
            value = _mm256_loadu_pd(data);
    }
    FASTOR_INLINE void store(double *data, bool Aligned=true) const {
        if (Aligned)
            _mm256_store_pd(data,value);
        else
            _mm256_storeu_pd(data,value);
    }

    FASTOR_INLINE void aligned_load(const double *data) {
        value =_mm256_load_pd(data);
    }
    FASTOR_INLINE void aligned_store(double *data) const {
        _mm256_store_pd(data,value);
    }

    FASTOR_INLINE void mask_load(const scalar_value_type *a, uint8_t mask, bool Aligned=false) {
#ifdef FASTOR_HAS_AVX512_MASKS
        if (!Aligned)
            value = _mm256_mask_loadu_pd(value, mask, a);
        else
            value = _mm256_mask_load_pd(value, mask, a);
#else
        // perhaps very inefficient but they never get used
        int maska[Size];
        mask_to_array(mask,maska);
        value = _mm256_setzero_pd();
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
            _mm256_mask_storeu_pd(a, mask, value);
        else
            _mm256_mask_store_pd(a, mask, value);
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

    FASTOR_INLINE double operator[](FASTOR_INDEX i) const {return reinterpret_cast<const double*>(&value)[i];}
    FASTOR_INLINE double operator()(FASTOR_INDEX i) const {return reinterpret_cast<const double*>(&value)[i];}

    FASTOR_INLINE void set(double num) {
        value = _mm256_set1_pd(num);
    }
    FASTOR_INLINE void set(double num0, double num1, double num2, double num3) {
        value = _mm256_set_pd(num0,num1,num2,num3);
    }
    FASTOR_INLINE void set_sequential(double num0) {
        value = _mm256_setr_pd(num0,num0+1.0,num0+2.0,num0+3.0);
    }
    FASTOR_INLINE void broadcast(const double *data) {
        value = _mm256_broadcast_sd(data);
    }

    // In-place operators
    FASTOR_INLINE void operator+=(double num) {
        value = _mm256_add_pd(value,_mm256_set1_pd(num));
    }
    FASTOR_INLINE void operator+=(__m256d regi) {
        value = _mm256_add_pd(value,regi);
    }
    FASTOR_INLINE void operator+=(const SIMDVector<double,simd_abi::avx> &a) {
        value = _mm256_add_pd(value,a.value);
    }

    FASTOR_INLINE void operator-=(double num) {
        value = _mm256_sub_pd(value,_mm256_set1_pd(num));
    }
    FASTOR_INLINE void operator-=(__m256d regi) {
        value = _mm256_sub_pd(value,regi);
    }
    FASTOR_INLINE void operator-=(const SIMDVector<double,simd_abi::avx> &a) {
        value = _mm256_sub_pd(value,a.value);
    }

    FASTOR_INLINE void operator*=(double num) {
        value = _mm256_mul_pd(value,_mm256_set1_pd(num));
    }
    FASTOR_INLINE void operator*=(__m256d regi) {
        value = _mm256_mul_pd(value,regi);
    }
    FASTOR_INLINE void operator*=(const SIMDVector<double,simd_abi::avx> &a) {
        value = _mm256_mul_pd(value,a.value);
    }

    FASTOR_INLINE void operator/=(double num) {
        value = _mm256_div_pd(value,_mm256_set1_pd(num));
    }
    FASTOR_INLINE void operator/=(__m256d regi) {
        value = _mm256_div_pd(value,regi);
    }
    FASTOR_INLINE void operator/=(const SIMDVector<double,simd_abi::avx> &a) {
        value = _mm256_div_pd(value,a.value);
    }
    // end of in-place operators

    FASTOR_INLINE SIMDVector<double,simd_abi::avx> shift(FASTOR_INDEX i) {
        SIMDVector<double,simd_abi::avx> out;
        if (i==1)
            out.value = _mm256_shift1_pd(value);
        else if (i==2)
            out.value = _mm256_shift2_pd(value);
        else if (i==3)
            out.value = _mm256_shift3_pd(value);
        return out;
    }
    FASTOR_INLINE double sum() {return _mm256_sum_pd(value);}
    FASTOR_INLINE double product() {return _mm256_prod_pd(value);}
    FASTOR_INLINE SIMDVector<double,simd_abi::avx> reverse() {
        SIMDVector<double,simd_abi::avx> out;
        out.value = _mm256_reverse_pd(value);
        return out;
    }
    FASTOR_INLINE double minimum() {return _mm256_hmin_pd(value);}
    FASTOR_INLINE double maximum() {return _mm256_hmax_pd(value);}

    FASTOR_INLINE double dot(const SIMDVector<double,simd_abi::avx> &other) {
        return _mm_cvtsd_f64(_mm256_dp_pd(value,other.value));
    }

    __m256d value;
};


FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<double,simd_abi::avx> a) {
    // ICC crashes without a copy
    const __m256d v = a.value;
    const double* value = reinterpret_cast<const double*>(&v);
    os << "[" << value[0] <<  " " << value[1] << " " << value[2] << " " << value[3] << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<double,simd_abi::avx> operator+(const SIMDVector<double,simd_abi::avx> &a, const SIMDVector<double,simd_abi::avx> &b) {
    SIMDVector<double,simd_abi::avx> out;
    out.value = _mm256_add_pd(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx> operator+(const SIMDVector<double,simd_abi::avx> &a, double b) {
    SIMDVector<double,simd_abi::avx> out;
    out.value = _mm256_add_pd(a.value,_mm256_set1_pd(b));
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx> operator+(double a, const SIMDVector<double,simd_abi::avx> &b) {
    SIMDVector<double,simd_abi::avx> out;
    out.value = _mm256_add_pd(_mm256_set1_pd(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx> operator+(const SIMDVector<double,simd_abi::avx> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<double,simd_abi::avx> operator-(const SIMDVector<double,simd_abi::avx> &a, const SIMDVector<double,simd_abi::avx> &b) {
    SIMDVector<double,simd_abi::avx> out;
    out.value = _mm256_sub_pd(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx> operator-(const SIMDVector<double,simd_abi::avx> &a, double b) {
    SIMDVector<double,simd_abi::avx> out;
    out.value = _mm256_sub_pd(a.value,_mm256_set1_pd(b));
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx> operator-(double a, const SIMDVector<double,simd_abi::avx> &b) {
    SIMDVector<double,simd_abi::avx> out;
    out.value = _mm256_sub_pd(_mm256_set1_pd(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx> operator-(const SIMDVector<double,simd_abi::avx> &b) {
    SIMDVector<double,simd_abi::avx> out;
    out.value = _mm256_neg_pd(b.value);
    return out;
}

FASTOR_INLINE SIMDVector<double,simd_abi::avx> operator*(const SIMDVector<double,simd_abi::avx> &a, const SIMDVector<double,simd_abi::avx> &b) {
    SIMDVector<double,simd_abi::avx> out;
    out.value = _mm256_mul_pd(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx> operator*(const SIMDVector<double,simd_abi::avx> &a, double b) {
    SIMDVector<double,simd_abi::avx> out;
    out.value = _mm256_mul_pd(a.value,_mm256_set1_pd(b));
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx> operator*(double a, const SIMDVector<double,simd_abi::avx> &b) {
    SIMDVector<double,simd_abi::avx> out;
    out.value = _mm256_mul_pd(_mm256_set1_pd(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<double,simd_abi::avx> operator/(const SIMDVector<double,simd_abi::avx> &a, const SIMDVector<double,simd_abi::avx> &b) {
    SIMDVector<double,simd_abi::avx> out;
    out.value = _mm256_div_pd(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx> operator/(const SIMDVector<double,simd_abi::avx> &a, double b) {
    SIMDVector<double,simd_abi::avx> out;
    out.value = _mm256_div_pd(a.value,_mm256_set1_pd(b));
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::avx> operator/(double a, const SIMDVector<double,simd_abi::avx> &b) {
    SIMDVector<double,simd_abi::avx> out;
    out.value = _mm256_div_pd(_mm256_set1_pd(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<double,simd_abi::avx> rcp(const SIMDVector<double,simd_abi::avx> &a) {
    SIMDVector<double,simd_abi::avx> out;
    // This is very inaccurate for double precision
    out.value = _mm256_cvtps_pd(_mm_rcp_ps(_mm256_cvtpd_ps(a.value)));
    return out;

    // // For making it more accurate using Newton Raphson use this
    // __m128d xmm0 = _mm256_cvtps_pd(_mm_rcp_ps(_mm256_cvtpd_ps(a.value)));
    // xmm0 = _mm256_mul_pd(xmm0,_mm256_sub_pd(VTWOPD,_mm256_mul_pd(x,xmm0)));
    // out.value = _mm256_mul_pd(xmm0,_mm256_sub_pd(VTWOPD,_mm256_mul_pd(x,xmm0)));
    // return out;

}

FASTOR_INLINE SIMDVector<double,simd_abi::avx> sqrt(const SIMDVector<double,simd_abi::avx> &a) {
    SIMDVector<double,simd_abi::avx> out;
    out.value = _mm256_sqrt_pd(a.value);
    return out;
}

FASTOR_INLINE SIMDVector<double,simd_abi::avx> rsqrt(const SIMDVector<double,simd_abi::avx> &a) {
    SIMDVector<double,simd_abi::avx> out;
   // This is very inaccurate for double precision
    out.value = _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(a.value)));
    return out;
}

FASTOR_INLINE SIMDVector<double,simd_abi::avx> abs(const SIMDVector<double,simd_abi::avx> &a) {
    SIMDVector<double,simd_abi::avx> out;
    out.value = _mm256_abs_pd(a.value);
    return out;
}

#endif







// SSE VERSION
//------------------------------------------------------------------------------------------------------------

#ifdef FASTOR_SSE2_IMPL

template <>
struct SIMDVector<double, simd_abi::sse> {
    using value_type = __m128d;
    using scalar_value_type = double;
    using abi_type = simd_abi::sse;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<double,simd_abi::sse>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<double,simd_abi::sse>>::value;}

    FASTOR_INLINE SIMDVector() : value(_mm_setzero_pd()) {}
    FASTOR_INLINE SIMDVector(double num) : value(_mm_set1_pd(num)) {}
    FASTOR_INLINE SIMDVector(__m128d regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const double *data, bool Aligned=true) {
        if (Aligned)
            value =_mm_load_pd(data);
        else
            value = _mm_loadu_pd(data);
    }

    FASTOR_INLINE SIMDVector<double,simd_abi::sse> operator=(double num) {
        value = _mm_set1_pd(num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<double,simd_abi::sse> operator=(__m128d regi) {
        value = regi;
        return *this;
    }

    FASTOR_INLINE void load(const double *data, bool Aligned=true) {
        if (Aligned)
            value =_mm_load_pd(data);
        else
            value = _mm_loadu_pd(data);
    }
    FASTOR_INLINE void store(double *data, bool Aligned=true) const {
        if (Aligned)
            _mm_store_pd(data,value);
        else
            _mm_storeu_pd(data,value);
    }

    FASTOR_INLINE void aligned_load(const double *data) {
        value =_mm_load_pd(data);
    }
    FASTOR_INLINE void aligned_store(double *data) const {
        _mm_store_pd(data,value);
    }

    FASTOR_INLINE void mask_load(const scalar_value_type *a, uint8_t mask, bool Aligned=false) {
#ifdef FASTOR_HAS_AVX512_MASKS
        if (!Aligned)
            value = _mm_mask_loadu_pd(value, mask, a);
        else
            value = _mm_mask_load_pd(value, mask, a);
#else
        // perhaps very inefficient but they never get used
        int maska[Size];
        mask_to_array(mask,maska);
        value = _mm_setzero_pd();
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
            _mm_mask_storeu_pd(a, mask, value);
        else
            _mm_mask_store_pd(a, mask, value);
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

    FASTOR_INLINE double operator[](FASTOR_INDEX i) const {return reinterpret_cast<const double*>(&value)[i];}
    FASTOR_INLINE double operator()(FASTOR_INDEX i) const {return reinterpret_cast<const double*>(&value)[i];}

    FASTOR_INLINE void set(double num) {
        value = _mm_set1_pd(num);
    }
    FASTOR_INLINE void set(double num0, double num1) {
        value = _mm_set_pd(num0,num1);
    }
    FASTOR_INLINE void set_sequential(double num0) {
        value = _mm_setr_pd(num0,num0+1.0);
    }
    FASTOR_INLINE void broadcast(const double *data) {
        value = _mm_load1_pd(data);
    }

    // In-place operators
    FASTOR_INLINE void operator+=(double num) {
        value = _mm_add_pd(value,_mm_set1_pd(num));
    }
    FASTOR_INLINE void operator+=(__m128d regi) {
        value = _mm_add_pd(value,regi);
    }
    FASTOR_INLINE void operator+=(const SIMDVector<double,simd_abi::sse> &a) {
        value = _mm_add_pd(value,a.value);
    }

    FASTOR_INLINE void operator-=(double num) {
        value = _mm_sub_pd(value,_mm_set1_pd(num));
    }
    FASTOR_INLINE void operator-=(__m128d regi) {
        value = _mm_sub_pd(value,regi);
    }
    FASTOR_INLINE void operator-=(const SIMDVector<double,simd_abi::sse> &a) {
        value = _mm_sub_pd(value,a.value);
    }

    FASTOR_INLINE void operator*=(double num) {
        value = _mm_mul_pd(value,_mm_set1_pd(num));
    }
    FASTOR_INLINE void operator*=(__m128d regi) {
        value = _mm_mul_pd(value,regi);
    }
    FASTOR_INLINE void operator*=(const SIMDVector<double,simd_abi::sse> &a) {
        value = _mm_mul_pd(value,a.value);
    }

    FASTOR_INLINE void operator/=(double num) {
        value = _mm_div_pd(value,_mm_set1_pd(num));
    }
    FASTOR_INLINE void operator/=(__m128d regi) {
        value = _mm_div_pd(value,regi);
    }
    FASTOR_INLINE void operator/=(const SIMDVector<double,simd_abi::sse> &a) {
        value = _mm_div_pd(value,a.value);
    }
    // end of in-place operators

    FASTOR_INLINE SIMDVector<double,simd_abi::sse> shift(FASTOR_INDEX i) {
        SIMDVector<double,simd_abi::sse> out;
        FASTOR_ASSERT(i==1,"INCORRECT SHIFT INDEX");
            out.value = _mm_shift1_pd(value);
        return out;
    }
    FASTOR_INLINE double sum() {return _mm_sum_pd(value);}
    FASTOR_INLINE double product() {return _mm_prod_pd(value);}
    FASTOR_INLINE SIMDVector<double,simd_abi::sse> reverse() {
        SIMDVector<double,simd_abi::sse> out;
        out.value = _mm_reverse_pd(value);
        return out;
    }
    FASTOR_INLINE double minimum() {return _mm_hmin_pd(value);}
    FASTOR_INLINE double maximum() {return _mm_hmax_pd(value);}

    FASTOR_INLINE double dot(const SIMDVector<double,simd_abi::sse> &other) {
#ifdef FASTOR_SSE4_1_IMPL
        return _mm_cvtsd_f64(_mm_dp_pd(value,other.value,0xff));
#else
        return _mm_sum_pd(_mm_mul_pd(value,other.value));
#endif
    }

    __m128d value;
};


FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<double,simd_abi::sse> a) {
    // ICC crashes without a copy
    const __m128d v = a.value;
    const double* value = reinterpret_cast<const double*>(&v);
    os << "[" << value[0] <<  " " << value[1] << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<double,simd_abi::sse> operator+(const SIMDVector<double,simd_abi::sse> &a, const SIMDVector<double,simd_abi::sse> &b) {
    SIMDVector<double,simd_abi::sse> out;
    out.value = _mm_add_pd(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::sse> operator+(const SIMDVector<double,simd_abi::sse> &a, double b) {
    SIMDVector<double,simd_abi::sse> out;
    out.value = _mm_add_pd(a.value,_mm_set1_pd(b));
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::sse> operator+(double a, const SIMDVector<double,simd_abi::sse> &b) {
    SIMDVector<double,simd_abi::sse> out;
    out.value = _mm_add_pd(_mm_set1_pd(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::sse> operator+(const SIMDVector<double,simd_abi::sse> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<double,simd_abi::sse> operator-(const SIMDVector<double,simd_abi::sse> &a, const SIMDVector<double,simd_abi::sse> &b) {
    SIMDVector<double,simd_abi::sse> out;
    out.value = _mm_sub_pd(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::sse> operator-(const SIMDVector<double,simd_abi::sse> &a, double b) {
    SIMDVector<double,simd_abi::sse> out;
    out.value = _mm_sub_pd(a.value,_mm_set1_pd(b));
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::sse> operator-(double a, const SIMDVector<double,simd_abi::sse> &b) {
    SIMDVector<double,simd_abi::sse> out;
    out.value = _mm_sub_pd(_mm_set1_pd(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::sse> operator-(const SIMDVector<double,simd_abi::sse> &b) {
    SIMDVector<double,simd_abi::sse> out;
    out.value = _mm_neg_pd(b.value);
    return out;
}

FASTOR_INLINE SIMDVector<double,simd_abi::sse> operator*(const SIMDVector<double,simd_abi::sse> &a, const SIMDVector<double,simd_abi::sse> &b) {
    SIMDVector<double,simd_abi::sse> out;
    out.value = _mm_mul_pd(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::sse> operator*(const SIMDVector<double,simd_abi::sse> &a, double b) {
    SIMDVector<double,simd_abi::sse> out;
    out.value = _mm_mul_pd(a.value,_mm_set1_pd(b));
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::sse> operator*(double a, const SIMDVector<double,simd_abi::sse> &b) {
    SIMDVector<double,simd_abi::sse> out;
    out.value = _mm_mul_pd(_mm_set1_pd(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<double,simd_abi::sse> operator/(const SIMDVector<double,simd_abi::sse> &a, const SIMDVector<double,simd_abi::sse> &b) {
    SIMDVector<double,simd_abi::sse> out;
    out.value = _mm_div_pd(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::sse> operator/(const SIMDVector<double,simd_abi::sse> &a, double b) {
    SIMDVector<double,simd_abi::sse> out;
    out.value = _mm_div_pd(a.value,_mm_set1_pd(b));
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::sse> operator/(double a, const SIMDVector<double,simd_abi::sse> &b) {
    SIMDVector<double,simd_abi::sse> out;
    out.value = _mm_div_pd(_mm_set1_pd(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<double,simd_abi::sse> rcp(const SIMDVector<double,simd_abi::sse> &a) {
    SIMDVector<double,simd_abi::sse> out;
    // This is very inaccurate for double precision
    out.value = _mm_cvtps_pd(_mm_rcp_ps(_mm_cvtpd_ps(a.value)));
    return out;

    /*
    // For making it more accurate using Newton Raphson use this
    __m128d xmm0 = _mm_cvtps_pd(_mm_rcp_ps(_mm_cvtpd_ps(a.value)));
    xmm0 = _mm_mul_pd(xmm0,_mm_sub_pd(TWOPD,_mm_mul_pd(x,xmm0)));
    out.value = _mm_mul_pd(xmm0,_mm_sub_pd(TWOPD,_mm_mul_pd(x,xmm0)));
    return out;
    */
}

FASTOR_INLINE SIMDVector<double,simd_abi::sse> sqrt(const SIMDVector<double,simd_abi::sse> &a) {
    SIMDVector<double,simd_abi::sse> out;
    out.value = _mm_sqrt_pd(a.value);
    return out;
}

FASTOR_INLINE SIMDVector<double,simd_abi::sse> rsqrt(const SIMDVector<double,simd_abi::sse> &a) {
    SIMDVector<double,simd_abi::sse> out;
    // This is very inaccurate for double precision
    out.value = _mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(a.value)));
    return out;
}

FASTOR_INLINE SIMDVector<double,simd_abi::sse> abs(const SIMDVector<double,simd_abi::sse> &a) {
    SIMDVector<double,simd_abi::sse> out;
    out.value = _mm_abs_pd(a.value);
    return out;
}


#endif


} // end of namespace Fastor

#endif // // SIMD_VECTOR_DOUBLE_H
