#ifndef SIMD_VECTOR_FLOAT_H
#define SIMD_VECTOR_FLOAT_H

#include "simd_vector_base.h"

namespace Fastor {


// AVX VERSION
//--------------------------------------------------------------------------------------------------

#ifdef __AVX__

template <>
struct SIMDVector<float,256> {
    static constexpr FASTOR_INDEX Size = get_vector_size<float>::size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return get_vector_size<float>::size;}
    static constexpr int unroll_size(FASTOR_INDEX size) {return (static_cast<int>(size) - static_cast<int>(Size));}

    FASTOR_INLINE SIMDVector() : value(_mm256_setzero_ps()) {}
    FASTOR_INLINE SIMDVector(float num) : value(_mm256_set1_ps(num)) {}
    FASTOR_INLINE SIMDVector(__m256 regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const SIMDVector<float> &a) : value(a.value) {}
    FASTOR_INLINE SIMDVector(const float *data) : value(_mm256_load_ps(data)) {}
    FASTOR_INLINE SIMDVector(float *data) : value(_mm256_load_ps(data)) {}

    FASTOR_INLINE SIMDVector<float> operator=(float num) {
        value = _mm256_set1_ps(num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<float> operator=(__m256 regi) {
        value = regi;
        return *this;
    }
    FASTOR_INLINE SIMDVector<float> operator=(const SIMDVector<float> &a) {
        value = a.value;
        return *this;
    }

    FASTOR_INLINE void load(const float *data, bool Aligned=true) {
        if (Aligned)
            value =_mm256_load_ps(data);
        else
            value = _mm256_loadu_ps(data);
    }
    FASTOR_INLINE void store(float *data, bool Aligned=true) {
        if (Aligned)
            _mm256_store_ps(data,value);
        else
            _mm256_storeu_ps(data,value);
    }

    FASTOR_INLINE float operator[](FASTOR_INDEX i) const {return value[i];}
    FASTOR_INLINE float operator()(FASTOR_INDEX i) const {return value[i];}

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
    FASTOR_INLINE void operator+=(const SIMDVector<float> &a) {
        value = _mm256_add_ps(value,a.value);
    }

    FASTOR_INLINE void operator-=(float num) {
        value = _mm256_sub_ps(value,_mm256_set1_ps(num));
    }
    FASTOR_INLINE void operator-=(__m256 regi) {
        value = _mm256_sub_ps(value,regi);
    }
    FASTOR_INLINE void operator-=(const SIMDVector<float> &a) {
        value = _mm256_sub_ps(value,a.value);
    }

    FASTOR_INLINE void operator*=(float num) {
        value = _mm256_mul_ps(value,_mm256_set1_ps(num));
    }
    FASTOR_INLINE void operator*=(__m256 regi) {
        value = _mm256_mul_ps(value,regi);
    }
    FASTOR_INLINE void operator*=(const SIMDVector<float> &a) {
        value = _mm256_mul_ps(value,a.value);
    }

    FASTOR_INLINE void operator/=(float num) {
        value = _mm256_div_ps(value,_mm256_set1_ps(num));
    }
    FASTOR_INLINE void operator/=(__m256 regi) {
        value = _mm256_div_ps(value,regi);
    }
    FASTOR_INLINE void operator/=(const SIMDVector<float> &a) {
        value = _mm256_div_ps(value,a.value);
    }
    // end of in-place operators

    FASTOR_INLINE SIMDVector<float> shift(FASTOR_INDEX i) {
        SIMDVector<float> out;
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
    FASTOR_INLINE SIMDVector<float> reverse() {
        SIMDVector<float> out;
        out.value = _mm256_reverse_ps(value);
        return out;
    }
    FASTOR_INLINE float minimum() {return _mm256_hmin_ps(value);}
    FASTOR_INLINE float maximum() {return _mm256_hmax_ps(value);}

    FASTOR_INLINE float dot(const SIMDVector<float> &other) {
        __m256 tmp = _mm256_dp_ps(value,other.value,0xff);
        return _mm256_get0_ps(tmp)+_mm256_get4_ps(tmp);
    }

    __m256 value;
};


std::ostream& operator<<(std::ostream &os, SIMDVector<float> a) {
    // ICC crashes without a copy
    const __m256 value = a.value;
    os << "[" << value[0] <<  " " << value[1] << " "
       << value[2] << " " << value[3] << " "
       << value[4] << " " << value[5] << " "
       << value[6] << " " << value[7] << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<float> operator+(const SIMDVector<float> &a, const SIMDVector<float> &b) {
    SIMDVector<float> out;
    out.value = _mm256_add_ps(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float> operator+(const SIMDVector<float> &a, float b) {
    SIMDVector<float> out;
    out.value = _mm256_add_ps(a.value,_mm256_set1_ps(b));
    return out;
}
FASTOR_INLINE SIMDVector<float> operator+(float a, const SIMDVector<float> &b) {
    SIMDVector<float> out;
    out.value = _mm256_add_ps(_mm256_set1_ps(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float> operator+(const SIMDVector<float> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<float> operator-(const SIMDVector<float> &a, const SIMDVector<float> &b) {
    SIMDVector<float> out;
    out.value = _mm256_sub_ps(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float> operator-(const SIMDVector<float> &a, float b) {
    SIMDVector<float> out;
    out.value = _mm256_sub_ps(a.value,_mm256_set1_ps(b));
    return out;
}
FASTOR_INLINE SIMDVector<float> operator-(float a, const SIMDVector<float> &b) {
    SIMDVector<float> out;
    out.value = _mm256_sub_ps(_mm256_set1_ps(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float> operator-(const SIMDVector<float> &b) {
    SIMDVector<float> out;
    out.value = _mm256_neg_ps(b.value);
    return out;
}

FASTOR_INLINE SIMDVector<float> operator*(const SIMDVector<float> &a, const SIMDVector<float> &b) {
    SIMDVector<float> out;
    out.value = _mm256_mul_ps(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float> operator*(const SIMDVector<float> &a, float b) {
    SIMDVector<float> out;
    out.value = _mm256_mul_ps(a.value,_mm256_set1_ps(b));
    return out;
}
FASTOR_INLINE SIMDVector<float> operator*(float a, const SIMDVector<float> &b) {
    SIMDVector<float> out;
    out.value = _mm256_mul_ps(_mm256_set1_ps(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<float> operator/(const SIMDVector<float> &a, const SIMDVector<float> &b) {
    SIMDVector<float> out;
    out.value = _mm256_div_ps(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float> operator/(const SIMDVector<float> &a, float b) {
    SIMDVector<float> out;
    out.value = _mm256_div_ps(a.value,_mm256_set1_ps(b));
    return out;
}
FASTOR_INLINE SIMDVector<float> operator/(float a, const SIMDVector<float> &b) {
    SIMDVector<float> out;
    out.value = _mm256_div_ps(_mm256_set1_ps(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<float> rcp(const SIMDVector<float> &a) {
    SIMDVector<float> out;
    out.value = _mm256_rcp_ps(a.value);
    return out;
}

FASTOR_INLINE SIMDVector<float> sqrt(const SIMDVector<float> &a) {
    SIMDVector<float> out;
    out.value = _mm256_sqrt_ps(a.value);
    return out;
}

FASTOR_INLINE SIMDVector<float> rsqrt(const SIMDVector<float> &a) {
    SIMDVector<float> out;
    out.value = _mm256_rsqrt_ps(a.value);
    return out;
}

FASTOR_INLINE SIMDVector<float> abs(const SIMDVector<float> &a) {
    SIMDVector<float> out;
    out.value = _mm256_abs_ps(a.value);
    return out;
}


#endif






// SSE VERSION
//--------------------------------------------------------------------------------------------------

#ifdef __SSE4_2__

template <>
struct SIMDVector<float,128> {
    static constexpr FASTOR_INDEX Size = get_vector_size<float,128>::size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return get_vector_size<float,128>::size;}
    static constexpr int unroll_size(FASTOR_INDEX size) {return (static_cast<int>(size) - static_cast<int>(Size));}

    FASTOR_INLINE SIMDVector() : value(_mm_setzero_ps()) {}
    FASTOR_INLINE SIMDVector(float num) : value(_mm_set1_ps(num)) {}
    FASTOR_INLINE SIMDVector(__m128 regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const SIMDVector<float,128> &a) : value(a.value) {}
    FASTOR_INLINE SIMDVector(const float *data) : value(_mm_load_ps(data)) {}
    FASTOR_INLINE SIMDVector(float *data) : value(_mm_load_ps(data)) {}

    FASTOR_INLINE SIMDVector<float,128> operator=(float num) {
        value = _mm_set1_ps(num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<float,128> operator=(__m128 regi) {
        value = regi;
        return *this;
    }
    FASTOR_INLINE SIMDVector<float,128> operator=(const SIMDVector<float,128> &a) {
        value = a.value;
        return *this;
    }

    FASTOR_INLINE void load(const float *data, bool Aligned=true) {
        if (Aligned)
            value =_mm_load_ps(data);
        else
            value = _mm_loadu_ps(data);
    }
    FASTOR_INLINE void store(float *data, bool Aligned=true) {
        if (Aligned)
            _mm_store_ps(data,value);
        else
            _mm_storeu_ps(data,value);
    }

    FASTOR_INLINE float operator[](FASTOR_INDEX i) const {return value[i];}
    FASTOR_INLINE float operator()(FASTOR_INDEX i) const {return value[i];}

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
    FASTOR_INLINE void operator+=(const SIMDVector<float,128> &a) {
        value = _mm_add_ps(value,a.value);
    }

    FASTOR_INLINE void operator-=(float num) {
        value = _mm_sub_ps(value,_mm_set1_ps(num));
    }
    FASTOR_INLINE void operator-=(__m128 regi) {
        value = _mm_sub_ps(value,regi);
    }
    FASTOR_INLINE void operator-=(const SIMDVector<float,128> &a) {
        value = _mm_sub_ps(value,a.value);
    }

    FASTOR_INLINE void operator*=(float num) {
        value = _mm_mul_ps(value,_mm_set1_ps(num));
    }
    FASTOR_INLINE void operator*=(__m128 regi) {
        value = _mm_mul_ps(value,regi);
    }
    FASTOR_INLINE void operator*=(const SIMDVector<float,128> &a) {
        value = _mm_mul_ps(value,a.value);
    }

    FASTOR_INLINE void operator/=(float num) {
        value = _mm_div_ps(value,_mm_set1_ps(num));
    }
    FASTOR_INLINE void operator/=(__m128 regi) {
        value = _mm_div_ps(value,regi);
    }
    FASTOR_INLINE void operator/=(const SIMDVector<float,128> &a) {
        value = _mm_div_ps(value,a.value);
    }
    // end of in-place operators

    FASTOR_INLINE SIMDVector<float,128> shift(FASTOR_INDEX i) {
        SIMDVector<float,128> out;
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
    FASTOR_INLINE SIMDVector<float,128> reverse() {
        SIMDVector<float,128> out;
        out.value = _mm_reverse_ps(value);
        return out;
    }
    FASTOR_INLINE float minimum() {return _mm_hmin_ps(value);}
    FASTOR_INLINE float maximum() {return _mm_hmax_ps(value);}

    FASTOR_INLINE float dot(const SIMDVector<float,128> &other) {
        return _mm_cvtss_f32(_mm_dp_ps(value,other.value,0xff));
    }

    __m128 value;
};


std::ostream& operator<<(std::ostream &os, SIMDVector<float,128> a) {
    // ICC crashes without a copy
    const __m128 value = a.value;
    os << "[" << value[0] <<  " " << value[1] << " "
       << value[2] << " " << value[3] << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<float,128> operator+(const SIMDVector<float,128> &a, const SIMDVector<float,128> &b) {
    SIMDVector<float,128> out;
    out.value = _mm_add_ps(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float,128> operator+(const SIMDVector<float,128> &a, float b) {
    SIMDVector<float,128> out;
    out.value = _mm_add_ps(a.value,_mm_set1_ps(b));
    return out;
}
FASTOR_INLINE SIMDVector<float,128> operator+(float a, const SIMDVector<float,128> &b) {
    SIMDVector<float,128> out;
    out.value = _mm_add_ps(_mm_set1_ps(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float,128> operator+(const SIMDVector<float,128> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<float,128> operator-(const SIMDVector<float,128> &a, const SIMDVector<float,128> &b) {
    SIMDVector<float,128> out;
    out.value = _mm_sub_ps(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float,128> operator-(const SIMDVector<float,128> &a, float b) {
    SIMDVector<float,128> out;
    out.value = _mm_sub_ps(a.value,_mm_set1_ps(b));
    return out;
}
FASTOR_INLINE SIMDVector<float,128> operator-(float a, const SIMDVector<float,128> &b) {
    SIMDVector<float,128> out;
    out.value = _mm_sub_ps(_mm_set1_ps(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float,128> operator-(const SIMDVector<float,128> &b) {
    SIMDVector<float,128> out;
    out.value = _mm_neg_ps(b.value);
    return out;
}

FASTOR_INLINE SIMDVector<float,128> operator*(const SIMDVector<float,128> &a, const SIMDVector<float,128> &b) {
    SIMDVector<float,128> out;
    out.value = _mm_mul_ps(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float,128> operator*(const SIMDVector<float,128> &a, float b) {
    SIMDVector<float,128> out;
    out.value = _mm_mul_ps(a.value,_mm_set1_ps(b));
    return out;
}
FASTOR_INLINE SIMDVector<float,128> operator*(float a, const SIMDVector<float,128> &b) {
    SIMDVector<float,128> out;
    out.value = _mm_mul_ps(_mm_set1_ps(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<float,128> operator/(const SIMDVector<float,128> &a, const SIMDVector<float,128> &b) {
    SIMDVector<float,128> out;
    out.value = _mm_div_ps(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<float,128> operator/(const SIMDVector<float,128> &a, float b) {
    SIMDVector<float,128> out;
    out.value = _mm_div_ps(a.value,_mm_set1_ps(b));
    return out;
}
FASTOR_INLINE SIMDVector<float,128> operator/(float a, const SIMDVector<float,128> &b) {
    SIMDVector<float,128> out;
    out.value = _mm_div_ps(_mm_set1_ps(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<float,128> rcp(const SIMDVector<float,128> &a) {
    SIMDVector<float,128> out;
    out.value = _mm_rcp_ps(a.value);
    return out;
}

FASTOR_INLINE SIMDVector<float,128> sqrt(const SIMDVector<float,128> &a) {
    SIMDVector<float,128> out;
    out.value = _mm_sqrt_ps(a.value);
    return out;
}

FASTOR_INLINE SIMDVector<float,128> rsqrt(const SIMDVector<float,128> &a) {
    SIMDVector<float,128> out;
    out.value = _mm_rsqrt_ps(a.value);
    return out;
}

FASTOR_INLINE SIMDVector<float,128> abs(const SIMDVector<float,128> &a) {
    SIMDVector<float,128> out;
    out.value = _mm_abs_ps(a.value);
    return out;
}


#endif




// SCALAR VERSION
//------------------------------------------------------------------------------------------------------------
template <>
struct SIMDVector<float, 32> {
    static constexpr FASTOR_INDEX Size = 1;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return 1;}
    static constexpr int unroll_size(FASTOR_INDEX size) {return (static_cast<int>(size) - 1);}

    FASTOR_INLINE SIMDVector() : value(0) {}
    FASTOR_INLINE SIMDVector(float num) : value(num) {}
    FASTOR_INLINE SIMDVector(const SIMDVector<float,32> &a) : value(a.value) {}
    FASTOR_INLINE SIMDVector(const float *data) : value(*data) {}
    FASTOR_INLINE SIMDVector(float *data) : value(*data) {}

    FASTOR_INLINE SIMDVector<float,32> operator=(float num) {
        value = num;
        return *this;
    }
    FASTOR_INLINE SIMDVector<float,32> operator=(const SIMDVector<float,32> &a) {
        value = a.value;
        return *this;
    }

    FASTOR_INLINE void load(const float *data, bool ) {
        value = *data;
    }
    FASTOR_INLINE void store(float *data, bool ) {
        data[0] = value;
    }

    FASTOR_INLINE void load(const float *data) {
        value = *data;
    }
    FASTOR_INLINE void store(float *data) {
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
    FASTOR_INLINE void operator+=(const SIMDVector<float,32> &a) {
        value += a.value;
    }

    FASTOR_INLINE void operator-=(float num) {
        value -= num;
    }
    FASTOR_INLINE void operator-=(const SIMDVector<float,32> &a) {
        value -= a.value;
    }

    FASTOR_INLINE void operator*=(float num) {
        value *= num;
    }
    FASTOR_INLINE void operator*=(const SIMDVector<float,32> &a) {
        value *= a.value;
    }

    FASTOR_INLINE void operator/=(float num) {
        value /= num;
    }
    FASTOR_INLINE void operator/=(const SIMDVector<float,32> &a) {
        value /= a.value;
    }
    // end of in-place operators

    FASTOR_INLINE SIMDVector<float,32> shift(FASTOR_INDEX) {
        return *this;
    }
    FASTOR_INLINE float sum() {return value;}
    FASTOR_INLINE float product() {return value;}
    FASTOR_INLINE SIMDVector<float,32> reverse() {
        return *this;
    }
    FASTOR_INLINE float minimum() {return value;}
    FASTOR_INLINE float maximum() {return value;}

    FASTOR_INLINE float dot(const SIMDVector<float,32> &other) {
        return value*other.value;
    }

    float value;
};


std::ostream& operator<<(std::ostream &os, SIMDVector<float,32> a) {
    os << "[" << a.value << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<float,32> operator+(const SIMDVector<float,32> &a, const SIMDVector<float,32> &b) {
    SIMDVector<float,32> out;
    out.value = a.value+b.value;
    return out;
}
FASTOR_INLINE SIMDVector<float,32> operator+(const SIMDVector<float,32> &a, float b) {
    SIMDVector<float,32> out;
    out.value = a.value+b;
    return out;
}
FASTOR_INLINE SIMDVector<float,32> operator+(float a, const SIMDVector<float,32> &b) {
    SIMDVector<float,32> out;
    out.value = a+b.value;
    return out;
}
FASTOR_INLINE SIMDVector<float,32> operator+(const SIMDVector<float,32> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<float,32> operator-(const SIMDVector<float,32> &a, const SIMDVector<float,32> &b) {
    SIMDVector<float,32> out;
    out.value = a.value-b.value;
    return out;
}
FASTOR_INLINE SIMDVector<float,32> operator-(const SIMDVector<float,32> &a, float b) {
    SIMDVector<float,32> out;
    out.value = a.value-b;
    return out;
}
FASTOR_INLINE SIMDVector<float,32> operator-(float a, const SIMDVector<float,32> &b) {
    SIMDVector<float,32> out;
    out.value = a-b.value;
    return out;
}
FASTOR_INLINE SIMDVector<float,32> operator-(const SIMDVector<float,32> &b) {
    SIMDVector<float,32> out;
    out.value = -b.value;
    return out;
}

FASTOR_INLINE SIMDVector<float,32> operator*(const SIMDVector<float,32> &a, const SIMDVector<float,32> &b) {
    SIMDVector<float,32> out;
    out.value = a.value*b.value;
    return out;
}
FASTOR_INLINE SIMDVector<float,32> operator*(const SIMDVector<float,32> &a, float b) {
    SIMDVector<float,32> out;
    out.value = a.value*b;
    return out;
}
FASTOR_INLINE SIMDVector<float,32> operator*(float a, const SIMDVector<float,32> &b) {
    SIMDVector<float,32> out;
    out.value = a*b.value;
    return out;
}

FASTOR_INLINE SIMDVector<float,32> operator/(const SIMDVector<float,32> &a, const SIMDVector<float,32> &b) {
    SIMDVector<float,32> out;
    out.value = a.value/b.value;
    return out;
}
FASTOR_INLINE SIMDVector<float,32> operator/(const SIMDVector<float,32> &a, float b) {
    SIMDVector<float,32> out;
    out.value = a.value/b;
    return out;
}
FASTOR_INLINE SIMDVector<float,32> operator/(float a, const SIMDVector<float,32> &b) {
    SIMDVector<float,32> out;
    out.value = a/b.value;
    return out;
}

FASTOR_INLINE SIMDVector<float,32> rcp(const SIMDVector<float,32> &a) {
    return 1.f / a.value;
}

FASTOR_INLINE SIMDVector<float,32> sqrt(const SIMDVector<float,32> &a) {
    return std::sqrt(a.value);
}

FASTOR_INLINE SIMDVector<float,32> rsqrt(const SIMDVector<float,32> &a) {
    return 1.f/std::sqrt(a.value);
}

FASTOR_INLINE SIMDVector<float,32> abs(const SIMDVector<float,32> &a) {
    return std::abs(a.value);
}

}




#endif // SIMD_VECTOR_FLOAT_H

