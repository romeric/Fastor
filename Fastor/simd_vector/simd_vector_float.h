#ifndef SIMD_VECTOR_FLOAT_H
#define SIMD_VECTOR_FLOAT_H

#include "simd_vector_base.h"

namespace Fastor {


// AVX VERSION
//--------------------------------------------------------------------------------------------------

#ifdef FASTOR_AVX_IMPL

template <>
struct SIMDVector<float,simd_abi::avx> {
    using value_type = __m256;
    using scalar_value_type = float;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<float,simd_abi::avx>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<float,simd_abi::avx>>::value;}
    static constexpr int unroll_size(FASTOR_INDEX size) {return (static_cast<int>(size) - static_cast<int>(Size));}

    FASTOR_INLINE SIMDVector() : value(_mm256_setzero_ps()) {}
    FASTOR_INLINE SIMDVector(float num) : value(_mm256_set1_ps(num)) {}
    FASTOR_INLINE SIMDVector(__m256 regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const SIMDVector<float,simd_abi::avx> &a) : value(a.value) {}
    FASTOR_INLINE SIMDVector(const float *data, bool Aligned=true) {
        if (Aligned)
            value =_mm256_load_ps(data);
        else
            value = _mm256_loadu_ps(data);
    }
    FASTOR_INLINE SIMDVector(float *data, bool Aligned=true) {
        if (Aligned)
            value =_mm256_load_ps(data);
        else
            value = _mm256_loadu_ps(data);
    }

    FASTOR_INLINE SIMDVector<float,simd_abi::avx> operator=(float num) {
        value = _mm256_set1_ps(num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<float,simd_abi::avx> operator=(__m256 regi) {
        value = regi;
        return *this;
    }
    FASTOR_INLINE SIMDVector<float,simd_abi::avx> operator=(const SIMDVector<float,simd_abi::avx> &a) {
        value = a.value;
        return *this;
    }

    FASTOR_INLINE void load(const float *data, bool Aligned=true) {
        if (Aligned)
            value =_mm256_load_ps(data);
        else
            value = _mm256_loadu_ps(data);
    }
    FASTOR_INLINE void store(float *data, bool Aligned=true) const {
        if (Aligned)
            _mm256_store_ps(data,value);
        else
            _mm256_storeu_ps(data,value);
    }

    FASTOR_INLINE void aligned_load(const float *data) {
        value =_mm256_load_ps(data);
    }
    FASTOR_INLINE void aligned_store(float *data) const {
        _mm256_store_ps(data,value);
    }

    FASTOR_INLINE float operator[](FASTOR_INDEX i) const {return reinterpret_cast<const float*>(&value)[i];}
    FASTOR_INLINE float operator()(FASTOR_INDEX i) const {return reinterpret_cast<const float*>(&value)[i];}

    FASTOR_INLINE void set(float num) {
        value = _mm256_set1_ps(num);
    }
    FASTOR_INLINE void set(float num0, float num1, float num2, float num3,
                             float num4, float num5, float num6, float num7) {
        value = _mm256_set_ps(num0,num1,num2,num3,num4,num5,num6,num7);
    }
    FASTOR_INLINE void set_sequential(float num0) {
        value = _mm256_setr_ps(num0,num0+1.f,num0+2.f,num0+3.f,num0+4.f,num0+5.f,num0+6.f,num0+7.f);
    }
    FASTOR_INLINE void broadcast(const float *data) {
        value = _mm256_broadcast_ss(data);
    }

    // In-place operators
    FASTOR_INLINE void operator+=(float num) {
        value = _mm256_add_ps(value,_mm256_set1_ps(num));
    }
    FASTOR_INLINE void operator+=(__m256 regi) {
        value = _mm256_add_ps(value,regi);
    }
    FASTOR_INLINE void operator+=(const SIMDVector<float,simd_abi::avx> &a) {
        value = _mm256_add_ps(value,a.value);
    }

    FASTOR_INLINE void operator-=(float num) {
        value = _mm256_sub_ps(value,_mm256_set1_ps(num));
    }
    FASTOR_INLINE void operator-=(__m256 regi) {
        value = _mm256_sub_ps(value,regi);
    }
    FASTOR_INLINE void operator-=(const SIMDVector<float,simd_abi::avx> &a) {
        value = _mm256_sub_ps(value,a.value);
    }

    FASTOR_INLINE void operator*=(float num) {
        value = _mm256_mul_ps(value,_mm256_set1_ps(num));
    }
    FASTOR_INLINE void operator*=(__m256 regi) {
        value = _mm256_mul_ps(value,regi);
    }
    FASTOR_INLINE void operator*=(const SIMDVector<float,simd_abi::avx> &a) {
        value = _mm256_mul_ps(value,a.value);
    }

    FASTOR_INLINE void operator/=(float num) {
        value = _mm256_div_ps(value,_mm256_set1_ps(num));
    }
    FASTOR_INLINE void operator/=(__m256 regi) {
        value = _mm256_div_ps(value,regi);
    }
    FASTOR_INLINE void operator/=(const SIMDVector<float,simd_abi::avx> &a) {
        value = _mm256_div_ps(value,a.value);
    }
    // end of in-place operators

    FASTOR_INLINE SIMDVector<float,simd_abi::avx> shift(FASTOR_INDEX i) {
        SIMDVector<float,simd_abi::avx> out;
        if (i==1)
            out.value = _mm256_shift1_ps(value);
        else if (i==2)
            out.value = _mm256_shift2_ps(value);
        else if (i==3)
            out.value = _mm256_shift3_ps(value);
        else if (i==4)
            out.value = _mm256_shift4_ps(value);
        else if (i==5)
            out.value = _mm256_shift5_ps(value);
        else if (i==6)
            out.value = _mm256_shift6_ps(value);
        else if (i==7)
            out.value = _mm256_shift7_ps(value);
        return out;
    }
    FASTOR_INLINE float sum() {return _mm256_sum_ps(value);}
    FASTOR_INLINE float product() {return _mm256_prod_ps(value);}
    FASTOR_INLINE SIMDVector<float,simd_abi::avx> reverse() {
        SIMDVector<float,simd_abi::avx> out;
        out.value = _mm256_reverse_ps(value);
        return out;
    }
    FASTOR_INLINE float minimum() {return _mm256_hmin_ps(value);}
    FASTOR_INLINE float maximum() {return _mm256_hmax_ps(value);}

    FASTOR_INLINE float dot(const SIMDVector<float,simd_abi::avx> &other) {
        __m256 tmp = _mm256_dp_ps(value,other.value,0xff);
        return _mm256_get0_ps(tmp)+_mm256_get4_ps(tmp);
    }

    __m256 value;
};


FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<float,simd_abi::avx> a) {
    // ICC crashes without a copy
    const __m256 v = a.value;
    const float* value = reinterpret_cast<const float*>(&v);
    os << "[" << value[0] <<  " " << value[1] << " "
       << value[2] << " " << value[3] << " "
       << value[4] << " " << value[5] << " "
       << value[6] << " " << value[7] << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<float,simd_abi::avx> operator+(const SIMDVector<float,simd_abi::avx> &a, const SIMDVector<float,simd_abi::avx> &b) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_add_ps(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::avx> operator+(const SIMDVector<float,simd_abi::avx> &a, float b) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_add_ps(a.value,_mm256_set1_ps(b));
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::avx> operator+(float a, const SIMDVector<float,simd_abi::avx> &b) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_add_ps(_mm256_set1_ps(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::avx> operator+(const SIMDVector<float,simd_abi::avx> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<float,simd_abi::avx> operator-(const SIMDVector<float,simd_abi::avx> &a, const SIMDVector<float,simd_abi::avx> &b) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_sub_ps(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::avx> operator-(const SIMDVector<float,simd_abi::avx> &a, float b) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_sub_ps(a.value,_mm256_set1_ps(b));
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::avx> operator-(float a, const SIMDVector<float,simd_abi::avx> &b) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_sub_ps(_mm256_set1_ps(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::avx> operator-(const SIMDVector<float,simd_abi::avx> &b) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_neg_ps(b.value);
    return out;
}

FASTOR_INLINE SIMDVector<float,simd_abi::avx> operator*(const SIMDVector<float,simd_abi::avx> &a, const SIMDVector<float,simd_abi::avx> &b) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_mul_ps(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::avx> operator*(const SIMDVector<float,simd_abi::avx> &a, float b) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_mul_ps(a.value,_mm256_set1_ps(b));
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::avx> operator*(float a, const SIMDVector<float,simd_abi::avx> &b) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_mul_ps(_mm256_set1_ps(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<float,simd_abi::avx> operator/(const SIMDVector<float,simd_abi::avx> &a, const SIMDVector<float,simd_abi::avx> &b) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_div_ps(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::avx> operator/(const SIMDVector<float,simd_abi::avx> &a, float b) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_div_ps(a.value,_mm256_set1_ps(b));
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::avx> operator/(float a, const SIMDVector<float,simd_abi::avx> &b) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_div_ps(_mm256_set1_ps(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<float,simd_abi::avx> rcp(const SIMDVector<float,simd_abi::avx> &a) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_rcp_ps(a.value);
    return out;
}

FASTOR_INLINE SIMDVector<float,simd_abi::avx> sqrt(const SIMDVector<float,simd_abi::avx> &a) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_sqrt_ps(a.value);
    return out;
}

FASTOR_INLINE SIMDVector<float,simd_abi::avx> rsqrt(const SIMDVector<float,simd_abi::avx> &a) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_rsqrt_ps(a.value);
    return out;
}

FASTOR_INLINE SIMDVector<float,simd_abi::avx> abs(const SIMDVector<float,simd_abi::avx> &a) {
    SIMDVector<float,simd_abi::avx> out;
    out.value = _mm256_abs_ps(a.value);
    return out;
}


#endif






// SSE VERSION
//--------------------------------------------------------------------------------------------------

#ifdef FASTOR_SSE4_2_IMPL

template <>
struct SIMDVector<float,simd_abi::sse> {
    using value_type = __m128;
    using scalar_value_type = float;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<float,simd_abi::sse>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<float,simd_abi::sse>>::value;}
    static constexpr int unroll_size(FASTOR_INDEX size) {return (static_cast<int>(size) - static_cast<int>(Size));}

    FASTOR_INLINE SIMDVector() : value(_mm_setzero_ps()) {}
    FASTOR_INLINE SIMDVector(float num) : value(_mm_set1_ps(num)) {}
    FASTOR_INLINE SIMDVector(__m128 regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const SIMDVector<float,simd_abi::sse> &a) : value(a.value) {}
    FASTOR_INLINE SIMDVector(const float *data, bool Aligned=true) {
        if (Aligned)
            value =_mm_load_ps(data);
        else
            value = _mm_loadu_ps(data);
    }
    FASTOR_INLINE SIMDVector(float *data, bool Aligned=true) {
        if (Aligned)
            value =_mm_load_ps(data);
        else
            value = _mm_loadu_ps(data);
    }

    FASTOR_INLINE SIMDVector<float,simd_abi::sse> operator=(float num) {
        value = _mm_set1_ps(num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<float,simd_abi::sse> operator=(__m128 regi) {
        value = regi;
        return *this;
    }
    FASTOR_INLINE SIMDVector<float,simd_abi::sse> operator=(const SIMDVector<float,simd_abi::sse> &a) {
        value = a.value;
        return *this;
    }

    FASTOR_INLINE void load(const float *data, bool Aligned=true) {
        if (Aligned)
            value =_mm_load_ps(data);
        else
            value = _mm_loadu_ps(data);
    }
    FASTOR_INLINE void store(float *data, bool Aligned=true) const {
        if (Aligned)
            _mm_store_ps(data,value);
        else
            _mm_storeu_ps(data,value);
    }

    FASTOR_INLINE void aligned_load(const float *data) {
        value =_mm_load_ps(data);
    }
    FASTOR_INLINE void aligned_store(float *data) const {
        _mm_store_ps(data,value);
    }

    FASTOR_INLINE float operator[](FASTOR_INDEX i) const {return reinterpret_cast<const float*>(&value)[i];}
    FASTOR_INLINE float operator()(FASTOR_INDEX i) const {return reinterpret_cast<const float*>(&value)[i];}

    FASTOR_INLINE void set(float num) {
        value = _mm_set1_ps(num);
    }
    FASTOR_INLINE void set(float num0, float num1, float num2, float num3) {
        value = _mm_set_ps(num0,num1,num2,num3);
    }
    FASTOR_INLINE void set_sequential(float num0) {
        value = _mm_setr_ps(num0,num0+1.f,num0+2.f,num0+3.f);
    }
    FASTOR_INLINE void broadcast(const float *data) {
        value = _mm_load1_ps(data);
    }

    // In-place operators
    FASTOR_INLINE void operator+=(float num) {
        value = _mm_add_ps(value,_mm_set1_ps(num));
    }
    FASTOR_INLINE void operator+=(__m128 regi) {
        value = _mm_add_ps(value,regi);
    }
    FASTOR_INLINE void operator+=(const SIMDVector<float,simd_abi::sse> &a) {
        value = _mm_add_ps(value,a.value);
    }

    FASTOR_INLINE void operator-=(float num) {
        value = _mm_sub_ps(value,_mm_set1_ps(num));
    }
    FASTOR_INLINE void operator-=(__m128 regi) {
        value = _mm_sub_ps(value,regi);
    }
    FASTOR_INLINE void operator-=(const SIMDVector<float,simd_abi::sse> &a) {
        value = _mm_sub_ps(value,a.value);
    }

    FASTOR_INLINE void operator*=(float num) {
        value = _mm_mul_ps(value,_mm_set1_ps(num));
    }
    FASTOR_INLINE void operator*=(__m128 regi) {
        value = _mm_mul_ps(value,regi);
    }
    FASTOR_INLINE void operator*=(const SIMDVector<float,simd_abi::sse> &a) {
        value = _mm_mul_ps(value,a.value);
    }

    FASTOR_INLINE void operator/=(float num) {
        value = _mm_div_ps(value,_mm_set1_ps(num));
    }
    FASTOR_INLINE void operator/=(__m128 regi) {
        value = _mm_div_ps(value,regi);
    }
    FASTOR_INLINE void operator/=(const SIMDVector<float,simd_abi::sse> &a) {
        value = _mm_div_ps(value,a.value);
    }
    // end of in-place operators

    FASTOR_INLINE SIMDVector<float,simd_abi::sse> shift(FASTOR_INDEX i) {
        SIMDVector<float,simd_abi::sse> out;
        if (i==1)
            out.value = _mm_shift1_ps(value);
        else if (i==2)
            out.value = _mm_shift2_ps(value);
        else if (i==3)
            out.value = _mm_shift3_ps(value);
        return out;
    }
    FASTOR_INLINE float sum() {return _mm_sum_ps(value);}
    FASTOR_INLINE float product() {return _mm_prod_ps(value);}
    FASTOR_INLINE SIMDVector<float,simd_abi::sse> reverse() {
        SIMDVector<float,simd_abi::sse> out;
        out.value = _mm_reverse_ps(value);
        return out;
    }
    FASTOR_INLINE float minimum() {return _mm_hmin_ps(value);}
    FASTOR_INLINE float maximum() {return _mm_hmax_ps(value);}

    FASTOR_INLINE float dot(const SIMDVector<float,simd_abi::sse> &other) {
#ifdef __SSE4_1__
        return _mm_cvtss_f32(_mm_dp_ps(value,other.value,0xff));
#else
        return _mm_sum_ps(_mm_mul_ps(value,other.value));
#endif
    }

    __m128 value;
};


FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<float,simd_abi::sse> a) {
    // ICC crashes without a copy
    const __m128 v = a.value;
    const float* value = reinterpret_cast<const float*>(&v);
    os << "[" << value[0] <<  " " << value[1] << " "
       << value[2] << " " << value[3] << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<float,simd_abi::sse> operator+(const SIMDVector<float,simd_abi::sse> &a, const SIMDVector<float,simd_abi::sse> &b) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_add_ps(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::sse> operator+(const SIMDVector<float,simd_abi::sse> &a, float b) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_add_ps(a.value,_mm_set1_ps(b));
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::sse> operator+(float a, const SIMDVector<float,simd_abi::sse> &b) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_add_ps(_mm_set1_ps(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::sse> operator+(const SIMDVector<float,simd_abi::sse> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<float,simd_abi::sse> operator-(const SIMDVector<float,simd_abi::sse> &a, const SIMDVector<float,simd_abi::sse> &b) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_sub_ps(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::sse> operator-(const SIMDVector<float,simd_abi::sse> &a, float b) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_sub_ps(a.value,_mm_set1_ps(b));
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::sse> operator-(float a, const SIMDVector<float,simd_abi::sse> &b) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_sub_ps(_mm_set1_ps(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::sse> operator-(const SIMDVector<float,simd_abi::sse> &b) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_neg_ps(b.value);
    return out;
}

FASTOR_INLINE SIMDVector<float,simd_abi::sse> operator*(const SIMDVector<float,simd_abi::sse> &a, const SIMDVector<float,simd_abi::sse> &b) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_mul_ps(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::sse> operator*(const SIMDVector<float,simd_abi::sse> &a, float b) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_mul_ps(a.value,_mm_set1_ps(b));
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::sse> operator*(float a, const SIMDVector<float,simd_abi::sse> &b) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_mul_ps(_mm_set1_ps(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<float,simd_abi::sse> operator/(const SIMDVector<float,simd_abi::sse> &a, const SIMDVector<float,simd_abi::sse> &b) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_div_ps(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::sse> operator/(const SIMDVector<float,simd_abi::sse> &a, float b) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_div_ps(a.value,_mm_set1_ps(b));
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::sse> operator/(float a, const SIMDVector<float,simd_abi::sse> &b) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_div_ps(_mm_set1_ps(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<float,simd_abi::sse> rcp(const SIMDVector<float,simd_abi::sse> &a) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_rcp_ps(a.value);
    return out;
}

FASTOR_INLINE SIMDVector<float,simd_abi::sse> sqrt(const SIMDVector<float,simd_abi::sse> &a) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_sqrt_ps(a.value);
    return out;
}

FASTOR_INLINE SIMDVector<float,simd_abi::sse> rsqrt(const SIMDVector<float,simd_abi::sse> &a) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_rsqrt_ps(a.value);
    return out;
}

FASTOR_INLINE SIMDVector<float,simd_abi::sse> abs(const SIMDVector<float,simd_abi::sse> &a) {
    SIMDVector<float,simd_abi::sse> out;
    out.value = _mm_abs_ps(a.value);
    return out;
}


#endif




// SCALAR VERSION
//------------------------------------------------------------------------------------------------------------
template <>
struct SIMDVector<float, simd_abi::scalar> {
    using value_type = float;
    using scalar_value_type = float;
    static constexpr FASTOR_INDEX Size = 1;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return 1;}
    static constexpr int unroll_size(FASTOR_INDEX size) {return (static_cast<int>(size) - 1);}

    FASTOR_INLINE SIMDVector() : value(0) {}
    FASTOR_INLINE SIMDVector(float num) : value(num) {}
    FASTOR_INLINE SIMDVector(const SIMDVector<float,simd_abi::scalar> &a) : value(a.value) {}
    FASTOR_INLINE SIMDVector(const float *data, bool Aligned=true) : value(*data) {}
    FASTOR_INLINE SIMDVector(float *data, bool Aligned=true) : value(*data) {}

    FASTOR_INLINE SIMDVector<float,simd_abi::scalar> operator=(float num) {
        value = num;
        return *this;
    }
    FASTOR_INLINE SIMDVector<float,simd_abi::scalar> operator=(const SIMDVector<float,simd_abi::scalar> &a) {
        value = a.value;
        return *this;
    }

    FASTOR_INLINE void load(const float *data, bool ) {
        value = *data;
    }
    FASTOR_INLINE void store(float *data, bool ) const {
        data[0] = value;
    }

    FASTOR_INLINE void load(const float *data) {
        value = *data;
    }
    FASTOR_INLINE void store(float *data) const {
        data[0] = value;
    }

    FASTOR_INLINE void aligned_load(const float *data) {
        value = *data;
    }
    FASTOR_INLINE void aligned_store(float *data) const {
        data[0] = value;
    }

    FASTOR_INLINE float operator[](FASTOR_INDEX) const {return value;}
    FASTOR_INLINE float operator()(FASTOR_INDEX) const {return value;}

    FASTOR_INLINE void set(float num) {
        value = num;
    }

    FASTOR_INLINE void set_sequential(float num) {
        value = num;
    }

    FASTOR_INLINE void broadcast(const float *data) {
        value = *data;
    }

    // In-place operators
    FASTOR_INLINE void operator+=(float num) {
        value += num;
    }
    FASTOR_INLINE void operator+=(const SIMDVector<float,simd_abi::scalar> &a) {
        value += a.value;
    }

    FASTOR_INLINE void operator-=(float num) {
        value -= num;
    }
    FASTOR_INLINE void operator-=(const SIMDVector<float,simd_abi::scalar> &a) {
        value -= a.value;
    }

    FASTOR_INLINE void operator*=(float num) {
        value *= num;
    }
    FASTOR_INLINE void operator*=(const SIMDVector<float,simd_abi::scalar> &a) {
        value *= a.value;
    }

    FASTOR_INLINE void operator/=(float num) {
        value /= num;
    }
    FASTOR_INLINE void operator/=(const SIMDVector<float,simd_abi::scalar> &a) {
        value /= a.value;
    }
    // end of in-place operators

    FASTOR_INLINE SIMDVector<float,simd_abi::scalar> shift(FASTOR_INDEX) {
        return *this;
    }
    FASTOR_INLINE float sum() {return value;}
    FASTOR_INLINE float product() {return value;}
    FASTOR_INLINE SIMDVector<float,simd_abi::scalar> reverse() {
        return *this;
    }
    FASTOR_INLINE float minimum() {return value;}
    FASTOR_INLINE float maximum() {return value;}

    FASTOR_INLINE float dot(const SIMDVector<float,simd_abi::scalar> &other) {
        return value*other.value;
    }

    float value;
};


FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<float,simd_abi::scalar> a) {
    os << "[" << a.value << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<float,simd_abi::scalar> operator+(const SIMDVector<float,simd_abi::scalar> &a, const SIMDVector<float,simd_abi::scalar> &b) {
    SIMDVector<float,simd_abi::scalar> out;
    out.value = a.value+b.value;
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::scalar> operator+(const SIMDVector<float,simd_abi::scalar> &a, float b) {
    SIMDVector<float,simd_abi::scalar> out;
    out.value = a.value+b;
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::scalar> operator+(float a, const SIMDVector<float,simd_abi::scalar> &b) {
    SIMDVector<float,simd_abi::scalar> out;
    out.value = a+b.value;
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::scalar> operator+(const SIMDVector<float,simd_abi::scalar> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<float,simd_abi::scalar> operator-(const SIMDVector<float,simd_abi::scalar> &a, const SIMDVector<float,simd_abi::scalar> &b) {
    SIMDVector<float,simd_abi::scalar> out;
    out.value = a.value-b.value;
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::scalar> operator-(const SIMDVector<float,simd_abi::scalar> &a, float b) {
    SIMDVector<float,simd_abi::scalar> out;
    out.value = a.value-b;
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::scalar> operator-(float a, const SIMDVector<float,simd_abi::scalar> &b) {
    SIMDVector<float,simd_abi::scalar> out;
    out.value = a-b.value;
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::scalar> operator-(const SIMDVector<float,simd_abi::scalar> &b) {
    SIMDVector<float,simd_abi::scalar> out;
    out.value = -b.value;
    return out;
}

FASTOR_INLINE SIMDVector<float,simd_abi::scalar> operator*(const SIMDVector<float,simd_abi::scalar> &a, const SIMDVector<float,simd_abi::scalar> &b) {
    SIMDVector<float,simd_abi::scalar> out;
    out.value = a.value*b.value;
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::scalar> operator*(const SIMDVector<float,simd_abi::scalar> &a, float b) {
    SIMDVector<float,simd_abi::scalar> out;
    out.value = a.value*b;
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::scalar> operator*(float a, const SIMDVector<float,simd_abi::scalar> &b) {
    SIMDVector<float,simd_abi::scalar> out;
    out.value = a*b.value;
    return out;
}

FASTOR_INLINE SIMDVector<float,simd_abi::scalar> operator/(const SIMDVector<float,simd_abi::scalar> &a, const SIMDVector<float,simd_abi::scalar> &b) {
    SIMDVector<float,simd_abi::scalar> out;
    out.value = a.value/b.value;
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::scalar> operator/(const SIMDVector<float,simd_abi::scalar> &a, float b) {
    SIMDVector<float,simd_abi::scalar> out;
    out.value = a.value/b;
    return out;
}
FASTOR_INLINE SIMDVector<float,simd_abi::scalar> operator/(float a, const SIMDVector<float,simd_abi::scalar> &b) {
    SIMDVector<float,simd_abi::scalar> out;
    out.value = a/b.value;
    return out;
}

FASTOR_INLINE SIMDVector<float,simd_abi::scalar> rcp(const SIMDVector<float,simd_abi::scalar> &a) {
    return 1.f / a.value;
}

FASTOR_INLINE SIMDVector<float,simd_abi::scalar> sqrt(const SIMDVector<float,simd_abi::scalar> &a) {
    return std::sqrt(a.value);
}

FASTOR_INLINE SIMDVector<float,simd_abi::scalar> rsqrt(const SIMDVector<float,simd_abi::scalar> &a) {
    return 1.f/std::sqrt(a.value);
}

FASTOR_INLINE SIMDVector<float,simd_abi::scalar> abs(const SIMDVector<float,simd_abi::scalar> &a) {
    return std::abs(a.value);
}

}




#endif // SIMD_VECTOR_FLOAT_H

