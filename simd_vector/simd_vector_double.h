#ifndef SIMD_VECTOR_DOUBLE_H
#define SIMD_VECTOR_DOUBLE_H

#include "simd_vector_base.h"

namespace Fastor {

template <>
struct SIMDVector<double> {
    static constexpr FASTOR_INDEX Size = get_vector_size<double>::size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return get_vector_size<double>::size;}
    static constexpr int unroll_size(FASTOR_INDEX size) {return (static_cast<int>(size) - static_cast<int>(Size));}

    FASTOR_INLINE SIMDVector() : value(_mm256_setzero_pd()) {}
    FASTOR_INLINE SIMDVector(double num) : value(_mm256_set1_pd(num)) {}
    FASTOR_INLINE SIMDVector(__m256d regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const SIMDVector<double> &a) : value(a.value) {}
    FASTOR_INLINE SIMDVector(const double *data) : value(_mm256_load_pd(data)) {}
    FASTOR_INLINE SIMDVector(double *data) : value(_mm256_load_pd(data)) {}

    FASTOR_INLINE SIMDVector<double> operator=(double num) {
        value = _mm256_set1_pd(num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<double> operator=(__m256d regi) {
        value = regi;
        return *this;
    }
    FASTOR_INLINE SIMDVector<double> operator=(const SIMDVector<double> &a) {
        value = a.value;
        return *this;
    }

    FASTOR_INLINE void load(const double *data, bool Aligned=true) {
        if (Aligned)
            value =_mm256_load_pd(data);
        else
            value = _mm256_loadu_pd(data);
    }
    FASTOR_INLINE void store(double *data, bool Aligned=true) {
        if (Aligned)
            _mm256_store_pd(data,value);
        else
            _mm256_storeu_pd(data,value);
    }

    FASTOR_INLINE double operator[](FASTOR_INDEX i) {return value[i];}
    FASTOR_INLINE double operator()(FASTOR_INDEX i) {return value[i];}

    FASTOR_INLINE void set(double num0, double num1, double num2, double num3) {
        value = _mm256_set_pd(num0,num1,num2,num3);
    }
    FASTOR_INLINE void set_sequential(double num0) {
        value = _mm256_setr_pd(num0,num0+1.0,num0+2.0,num0+3.0);
    }

    // In-place operators
    FASTOR_INLINE void operator+=(double num) {
        value = _mm256_add_pd(value,_mm256_set1_pd(num));
    }
    FASTOR_INLINE void operator+=(__m256d regi) {
        value = _mm256_add_pd(value,regi);
    }
    FASTOR_INLINE void operator+=(const SIMDVector<double> &a) {
        value = _mm256_add_pd(value,a.value);
    }

    FASTOR_INLINE void operator-=(double num) {
        value = _mm256_sub_pd(value,_mm256_set1_pd(num));
    }
    FASTOR_INLINE void operator-=(__m256d regi) {
        value = _mm256_sub_pd(value,regi);
    }
    FASTOR_INLINE void operator-=(const SIMDVector<double> &a) {
        value = _mm256_sub_pd(value,a.value);
    }

    FASTOR_INLINE void operator*=(double num) {
        value = _mm256_mul_pd(value,_mm256_set1_pd(num));
    }
    FASTOR_INLINE void operator*=(__m256d regi) {
        value = _mm256_mul_pd(value,regi);
    }
    FASTOR_INLINE void operator*=(const SIMDVector<double> &a) {
        value = _mm256_mul_pd(value,a.value);
    }

    FASTOR_INLINE void operator/=(double num) {
        value = _mm256_div_pd(value,_mm256_set1_pd(num));
    }
    FASTOR_INLINE void operator/=(__m256d regi) {
        value = _mm256_div_pd(value,regi);
    }
    FASTOR_INLINE void operator/=(const SIMDVector<double> &a) {
        value = _mm256_div_pd(value,a.value);
    }
    // end of in-place operators

    FASTOR_INLINE SIMDVector<double> shift(FASTOR_INDEX i) {
        SIMDVector<double> out;
        if (i==1)
            out.value = _mm256_shift1_pd(value);
        else if (i==2)
            out.value = _mm256_shift2_pd(value);
        else if (i==3)
            out.value = _mm256_shift3_pd(value);
        return out;
    }
    FASTOR_INLINE double sum() {return _mm256_sum_pd(value);}
    FASTOR_INLINE SIMDVector<double> reverse() {
        SIMDVector<double> out;
        out.value = _mm256_reverse_pd(value);
        return out;
    }
    FASTOR_INLINE double minimum() {return _mm256_hmin_pd(value);}
    FASTOR_INLINE double maximum() {return _mm256_hmax_pd(value);}

    __m256d value;
};


std::ostream& operator<<(std::ostream &os, SIMDVector<double> a) {
    // ICC crashes without a copy
    const __m256d value = a.value;
    os << "[" << value[0] <<  " " << value[1] << " " << value[2] << " " << value[3] << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<double> operator+(const SIMDVector<double> &a, const SIMDVector<double> &b) {
    SIMDVector<double> out;
    out.value = _mm256_add_pd(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double> operator+(const SIMDVector<double> &a, double b) {
    SIMDVector<double> out;
    out.value = _mm256_add_pd(a.value,_mm256_set1_pd(b));
    return out;
}
FASTOR_INLINE SIMDVector<double> operator+(double a, const SIMDVector<double> &b) {
    SIMDVector<double> out;
    out.value = _mm256_add_pd(_mm256_set1_pd(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<double> operator-(const SIMDVector<double> &a, const SIMDVector<double> &b) {
    SIMDVector<double> out;
    out.value = _mm256_sub_pd(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double> operator-(const SIMDVector<double> &a, double b) {
    SIMDVector<double> out;
    out.value = _mm256_sub_pd(a.value,_mm256_set1_pd(b));
    return out;
}
FASTOR_INLINE SIMDVector<double> operator-(double a, const SIMDVector<double> &b) {
    SIMDVector<double> out;
    out.value = _mm256_sub_pd(_mm256_set1_pd(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<double> operator*(const SIMDVector<double> &a, const SIMDVector<double> &b) {
    SIMDVector<double> out;
    out.value = _mm256_mul_pd(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double> operator*(const SIMDVector<double> &a, double b) {
    SIMDVector<double> out;
    out.value = _mm256_mul_pd(a.value,_mm256_set1_pd(b));
    return out;
}
FASTOR_INLINE SIMDVector<double> operator*(double a, const SIMDVector<double> &b) {
    SIMDVector<double> out;
    out.value = _mm256_mul_pd(_mm256_set1_pd(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<double> operator/(const SIMDVector<double> &a, const SIMDVector<double> &b) {
    SIMDVector<double> out;
    out.value = _mm256_div_pd(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<double> operator/(const SIMDVector<double> &a, double b) {
    SIMDVector<double> out;
    out.value = _mm256_div_pd(a.value,_mm256_set1_pd(b));
    return out;
}
FASTOR_INLINE SIMDVector<double> operator/(double a, const SIMDVector<double> &b) {
    SIMDVector<double> out;
    out.value = _mm256_div_pd(_mm256_set1_pd(a),b.value);
    return out;
}

FASTOR_INLINE SIMDVector<double> sqrt(const SIMDVector<double> &a) {
    SIMDVector<double> out;
    out.value = _mm256_sqrt_pd(a.value);
    return out;
}

}
#endif
