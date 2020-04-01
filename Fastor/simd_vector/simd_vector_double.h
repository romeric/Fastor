#ifndef SIMD_VECTOR_DOUBLE_H
#define SIMD_VECTOR_DOUBLE_H

#include "simd_vector_base.h"

namespace Fastor {

// AVX VERSION
//--------------------------------------------------------------------------------------------------
#ifdef FASTOR_AVX_IMPL

template <>
struct SIMDVector<double, simd_abi::avx> {
    using value_type = __m256d;
    using scalar_value_type = double;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<double,simd_abi::avx>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<double,simd_abi::avx>>::value;}
    static constexpr int unroll_size(FASTOR_INDEX size) {return (static_cast<int>(size) - static_cast<int>(Size));}

    FASTOR_INLINE SIMDVector() : value(_mm256_setzero_pd()) {}
    FASTOR_INLINE SIMDVector(double num) : value(_mm256_set1_pd(num)) {}
    FASTOR_INLINE SIMDVector(__m256d regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const SIMDVector<double,simd_abi::avx> &a) : value(a.value) {}
    FASTOR_INLINE SIMDVector(const double *data, bool Aligned=true) {
        if (Aligned)
            value =_mm256_load_pd(data);
        else
            value = _mm256_loadu_pd(data);
    }
    FASTOR_INLINE SIMDVector(double *data, bool Aligned=true) {
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
    FASTOR_INLINE SIMDVector<double,simd_abi::avx> operator=(const SIMDVector<double,simd_abi::avx> &a) {
        value = a.value;
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

#ifdef FASTOR_SSE4_2_IMPL

template <>
struct SIMDVector<double, simd_abi::sse> {
    using value_type = __m128d;
    using scalar_value_type = double;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<double,simd_abi::sse>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<double,simd_abi::sse>>::value;}
    static constexpr int unroll_size(FASTOR_INDEX size) {return (static_cast<int>(size) - static_cast<int>(Size));}

    FASTOR_INLINE SIMDVector() : value(_mm_setzero_pd()) {}
    FASTOR_INLINE SIMDVector(double num) : value(_mm_set1_pd(num)) {}
    FASTOR_INLINE SIMDVector(__m128d regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const SIMDVector<double, simd_abi::sse> &a) : value(a.value) {}
    FASTOR_INLINE SIMDVector(const double *data, bool Aligned=true) {
        if (Aligned)
            value =_mm_load_pd(data);
        else
            value = _mm_loadu_pd(data);
    }
    FASTOR_INLINE SIMDVector(double *data, bool Aligned=true) {
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
    FASTOR_INLINE SIMDVector<double,simd_abi::sse> operator=(const SIMDVector<double,simd_abi::sse> &a) {
        value = a.value;
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
#ifdef __SSE4_1__
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



// SCALAR VERSION
//------------------------------------------------------------------------------------------------------------
template <>
struct SIMDVector<double, simd_abi::scalar> {
    using value_type = double;
    using scalar_value_type = double;
    static constexpr FASTOR_INDEX Size = 1;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return 1;}
    static constexpr int unroll_size(FASTOR_INDEX size) {return (static_cast<int>(size) - 1);}

    FASTOR_INLINE SIMDVector() : value(0) {}
    FASTOR_INLINE SIMDVector(double num) : value(num) {}
    FASTOR_INLINE SIMDVector(const SIMDVector<double,simd_abi::scalar> &a) : value(a.value) {}
    FASTOR_INLINE SIMDVector(const double *data, bool Aligned=true) : value(*data) {}
    FASTOR_INLINE SIMDVector(double *data, bool Aligned=true) : value(*data) {}

    FASTOR_INLINE SIMDVector<double,simd_abi::scalar> operator=(double num) {
        value = num;
        return *this;
    }
    FASTOR_INLINE SIMDVector<double,simd_abi::scalar> operator=(const SIMDVector<double,simd_abi::scalar> &a) {
        value = a.value;
        return *this;
    }

    FASTOR_INLINE void load(const double *data, bool ) {
        value = *data;
    }
    FASTOR_INLINE void store(double *data, bool ) const {
        data[0] = value;
    }

    FASTOR_INLINE void load(const double *data) {
        value = *data;
    }
    FASTOR_INLINE void store(double *data) const {
        data[0] = value;
    }

    FASTOR_INLINE void aligned_load(const double *data) {
        value = *data;
    }
    FASTOR_INLINE void aligned_store(double *data) const {
        data[0] = value;
    }

    FASTOR_INLINE double operator[](FASTOR_INDEX) const {return value;}
    FASTOR_INLINE double operator()(FASTOR_INDEX) const {return value;}

    FASTOR_INLINE void set(double num) {
        value = num;
    }

    FASTOR_INLINE void set_sequential(double num) {
        value = num;
    }

    FASTOR_INLINE void broadcast(const double *data) {
        value = *data;
    }

    // In-place operators
    FASTOR_INLINE void operator+=(double num) {
        value += num;
    }
    FASTOR_INLINE void operator+=(const SIMDVector<double,simd_abi::scalar> &a) {
        value += a.value;
    }

    FASTOR_INLINE void operator-=(double num) {
        value -= num;
    }
    FASTOR_INLINE void operator-=(const SIMDVector<double,simd_abi::scalar> &a) {
        value -= a.value;
    }

    FASTOR_INLINE void operator*=(double num) {
        value *= num;
    }
    FASTOR_INLINE void operator*=(const SIMDVector<double,simd_abi::scalar> &a) {
        value *= a.value;
    }

    FASTOR_INLINE void operator/=(double num) {
        value /= num;
    }
    FASTOR_INLINE void operator/=(const SIMDVector<double,simd_abi::scalar> &a) {
        value /= a.value;
    }
    // end of in-place operators

    FASTOR_INLINE SIMDVector<double,simd_abi::scalar> shift(FASTOR_INDEX) {
        return *this;
    }
    FASTOR_INLINE double sum() {return value;}
    FASTOR_INLINE double product() {return value;}
    FASTOR_INLINE SIMDVector<double,simd_abi::scalar> reverse() {
        return *this;
    }
    FASTOR_INLINE double minimum() {return value;}
    FASTOR_INLINE double maximum() {return value;}

    FASTOR_INLINE double dot(const SIMDVector<double,simd_abi::scalar> &other) {
        return value*other.value;
    }

    double value;
};


FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<double,simd_abi::scalar> a) {
    os << "[" << a.value << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<double,simd_abi::scalar> operator+(const SIMDVector<double,simd_abi::scalar> &a, const SIMDVector<double,simd_abi::scalar> &b) {
    SIMDVector<double,simd_abi::scalar> out;
    out.value = a.value+b.value;
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::scalar> operator+(const SIMDVector<double,simd_abi::scalar> &a, double b) {
    SIMDVector<double,simd_abi::scalar> out;
    out.value = a.value+b;
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::scalar> operator+(double a, const SIMDVector<double,simd_abi::scalar> &b) {
    SIMDVector<double,simd_abi::scalar> out;
    out.value = a+b.value;
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::scalar> operator+(const SIMDVector<double,simd_abi::scalar> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<double,simd_abi::scalar> operator-(const SIMDVector<double,simd_abi::scalar> &a, const SIMDVector<double,simd_abi::scalar> &b) {
    SIMDVector<double,simd_abi::scalar> out;
    out.value = a.value-b.value;
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::scalar> operator-(const SIMDVector<double,simd_abi::scalar> &a, double b) {
    SIMDVector<double,simd_abi::scalar> out;
    out.value = a.value-b;
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::scalar> operator-(double a, const SIMDVector<double,simd_abi::scalar> &b) {
    SIMDVector<double,simd_abi::scalar> out;
    out.value = a-b.value;
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::scalar> operator-(const SIMDVector<double,simd_abi::scalar> &b) {
    SIMDVector<double,simd_abi::scalar> out;
    out.value = -b.value;
    return out;
}

FASTOR_INLINE SIMDVector<double,simd_abi::scalar> operator*(const SIMDVector<double,simd_abi::scalar> &a, const SIMDVector<double,simd_abi::scalar> &b) {
    SIMDVector<double,simd_abi::scalar> out;
    out.value = a.value*b.value;
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::scalar> operator*(const SIMDVector<double,simd_abi::scalar> &a, double b) {
    SIMDVector<double,simd_abi::scalar> out;
    out.value = a.value*b;
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::scalar> operator*(double a, const SIMDVector<double,simd_abi::scalar> &b) {
    SIMDVector<double,simd_abi::scalar> out;
    out.value = a*b.value;
    return out;
}

FASTOR_INLINE SIMDVector<double,simd_abi::scalar> operator/(const SIMDVector<double,simd_abi::scalar> &a, const SIMDVector<double,simd_abi::scalar> &b) {
    SIMDVector<double,simd_abi::scalar> out;
    out.value = a.value/b.value;
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::scalar> operator/(const SIMDVector<double,simd_abi::scalar> &a, double b) {
    SIMDVector<double,simd_abi::scalar> out;
    out.value = a.value/b;
    return out;
}
FASTOR_INLINE SIMDVector<double,simd_abi::scalar> operator/(double a, const SIMDVector<double,simd_abi::scalar> &b) {
    SIMDVector<double,simd_abi::scalar> out;
    out.value = a/b.value;
    return out;
}

FASTOR_INLINE SIMDVector<double,simd_abi::scalar> rcp(const SIMDVector<double,simd_abi::scalar> &a) {
    return 1.0/a.value;
}

FASTOR_INLINE SIMDVector<double,simd_abi::scalar> sqrt(const SIMDVector<double,simd_abi::scalar> &a) {
    return std::sqrt(a.value);
}

FASTOR_INLINE SIMDVector<double,simd_abi::scalar> rsqrt(const SIMDVector<double,simd_abi::scalar> &a) {
    return 1.0/std::sqrt(a.value);
}

FASTOR_INLINE SIMDVector<double,simd_abi::scalar> abs(const SIMDVector<double,simd_abi::scalar> &a) {
    return std::abs(a.value);
}


}
#endif
