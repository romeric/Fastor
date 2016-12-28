#ifndef SIMD_VECTOR_INT_H
#define SIMD_VECTOR_INT_H

#include "simd_vector_base.h"

namespace Fastor {


// AVX VERSION
//-----------------------------------------------------------------------------------------------

#ifdef __AVX__

template<>
struct SIMDVector<int,256> {

    static constexpr FASTOR_INDEX Size = get_vector_size<int,256>::size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return get_vector_size<int,256>::size;}
    static constexpr int unroll_size(FASTOR_INDEX size) {return (static_cast<int>(size) - static_cast<int>(Size));}

    FASTOR_INLINE SIMDVector() : value(_mm256_setzero_si256()) {}
    FASTOR_INLINE SIMDVector(int num) : value(_mm256_set1_epi32(num)) {}
    FASTOR_INLINE SIMDVector(__m256i regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const SIMDVector<int> &a) : value(a.value) {}
    FASTOR_INLINE SIMDVector(const int *data) : value(_mm256_load_si256((__m256i*)data)) {}
    FASTOR_INLINE SIMDVector(int *data) : value(_mm256_load_si256((__m256i*)data)) {}

    FASTOR_INLINE SIMDVector<int> operator=(int num) {
        value = _mm256_set1_epi32(num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<int> operator=(__m256i regi) {
        value = regi;
        return *this;
    }
    FASTOR_INLINE SIMDVector<int> operator=(const SIMDVector<int> &a) {
        value = a.value;
        return *this;
    }

    FASTOR_INLINE void load(const int *data, bool Aligned=true) {
        if (Aligned)
            value =_mm256_load_si256((__m256i*)data);
        else
            value = _mm256_loadu_si256((__m256i*)data);
    }
    FASTOR_INLINE void store(int *data, bool Aligned=true) {
        if (Aligned)
            _mm256_store_si256((__m256i*)data,value);
        else
            _mm256_storeu_si256((__m256i*)data,value);
    }

    FASTOR_INLINE int operator[](FASTOR_INDEX i) {return value[i];}
    FASTOR_INLINE int operator()(FASTOR_INDEX i) {return value[i];}

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
    FASTOR_INLINE void operator+=(const SIMDVector<int> &a) {
        value = _mm256_add_epi32x(value,a.value);
    }

    FASTOR_INLINE void operator-=(int num) {
        value = _mm256_sub_epi32x(value,_mm256_set1_epi32(num));
    }
    FASTOR_INLINE void operator-=(__m256i regi) {
        value = _mm256_sub_epi32x(value,regi);
    }
    FASTOR_INLINE void operator-=(const SIMDVector<int> &a) {
        value = _mm256_sub_epi32x(value,a.value);
    }

    FASTOR_INLINE void operator*=(int num) {
        value = _mm256_mul_epi32x(value,_mm256_set1_epi32(num));
    }
    FASTOR_INLINE void operator*=(__m256i regi) {
        value = _mm256_mul_epi32x(value,regi);
    }
    FASTOR_INLINE void operator*=(const SIMDVector<int> &a) {
        value = _mm256_mul_epi32x(value,a.value);
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

    FASTOR_INLINE int sum() {
        int *vals = (int*)&value;
        int quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            quan += vals[i];
        return quan;
    }

    FASTOR_INLINE int dot(const SIMDVector<int> &other) {
        int *vals0 = (int*)&value;
        int *vals1 = (int*)&other.value;
        int quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            quan += vals0[i]*vals1[i];
        return quan;
    }

    __m256i value;
};

std::ostream& operator<<(std::ostream &os, SIMDVector<int> a) {
    const int *value = (int*) &a.value;
    os << "[" << value[0] <<  " " << value[1] << " " << value[2] << " " << value[3]
       << " " << value[4] <<  " " << value[5] << " " << value[6] << " " << value[7] << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<int> operator+(const SIMDVector<int> &a, const SIMDVector<int> &b) {
    SIMDVector<int> out;
    out.value = _mm256_add_epi32x(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int> operator+(const SIMDVector<int> &a, int b) {
    SIMDVector<int> out;
    out.value = _mm256_add_epi32x(a.value,_mm256_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int> operator+(int a, const SIMDVector<int> &b) {
    SIMDVector<int> out;
    out.value = _mm256_add_epi32x(_mm256_set1_epi32(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int> operator+(const SIMDVector<int> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<int> operator-(const SIMDVector<int> &a, const SIMDVector<int> &b) {
    SIMDVector<int> out;
    out.value = _mm256_sub_epi32x(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int> operator-(const SIMDVector<int> &a, int b) {
    SIMDVector<int> out;
    out.value = _mm256_sub_epi32x(a.value,_mm256_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int> operator-(int a, const SIMDVector<int> &b) {
    SIMDVector<int> out;
    out.value = _mm256_sub_epi32x(_mm256_set1_epi32(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int> operator-(const SIMDVector<int> &b) {
    SIMDVector<int> out;
    out.value = _mm256_castps_si256(_mm256_neg_ps(_mm256_castsi256_ps(b.value)));
    return out;
}

FASTOR_INLINE SIMDVector<int> operator*(const SIMDVector<int> &a, const SIMDVector<int> &b) {
    SIMDVector<int> out;
    out.value = _mm256_mul_epi32x(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int> operator*(const SIMDVector<int> &a, int b) {
    SIMDVector<int> out;
    out.value = _mm256_mul_epi32x(a.value,_mm256_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int> operator*(int a, const SIMDVector<int> &b) {
    SIMDVector<int> out;
    out.value = _mm256_mul_epi32x(_mm256_set1_epi32(a),b.value);
    return out;
}


#endif


// SSE VERSION
//-----------------------------------------------------------------------------------------------

#ifdef __SSE4_2__

template<>
struct SIMDVector<int,128> {

    static constexpr FASTOR_INDEX Size = get_vector_size<int,128>::size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return get_vector_size<int,128>::size;}
    static constexpr int unroll_size(FASTOR_INDEX size) {return (static_cast<int>(size) - static_cast<int>(Size));}

    FASTOR_INLINE SIMDVector() : value(_mm_setzero_si128()) {}
    FASTOR_INLINE SIMDVector(int num) : value(_mm_set1_epi32(num)) {}
    FASTOR_INLINE SIMDVector(__m128i regi) : value(regi) {}
    FASTOR_INLINE SIMDVector(const SIMDVector<int,128> &a) : value(a.value) {}
    FASTOR_INLINE SIMDVector(const int *data) : value(_mm_load_si128((__m128i*)data)) {}
    FASTOR_INLINE SIMDVector(int *data) : value(_mm_load_si128((__m128i*)data)) {}

    FASTOR_INLINE SIMDVector<int,128> operator=(int num) {
        value = _mm_set1_epi32(num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<int,128> operator=(__m128i regi) {
        value = regi;
        return *this;
    }
    FASTOR_INLINE SIMDVector<int,128> operator=(const SIMDVector<int,128> &a) {
        value = a.value;
        return *this;
    }

    FASTOR_INLINE void load(const int *data, bool Aligned=true) {
        if (Aligned)
            value =_mm_load_si128((__m128i*)data);
        else
            value = _mm_loadu_si128((__m128i*)data);
    }
    FASTOR_INLINE void store(int *data, bool Aligned=true) {
        if (Aligned)
            _mm_store_si128((__m128i*)data,value);
        else
            _mm_storeu_si128((__m128i*)data,value);
    }

    FASTOR_INLINE int operator[](FASTOR_INDEX i) {return value[i];}
    FASTOR_INLINE int operator()(FASTOR_INDEX i) {return value[i];}

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
    FASTOR_INLINE void operator+=(const SIMDVector<int,128> &a) {
        value = _mm_add_epi32(value,a.value);
    }

    FASTOR_INLINE void operator-=(int num) {
        value = _mm_sub_epi32(value,_mm_set1_epi32(num));
    }
    FASTOR_INLINE void operator-=(__m128i regi) {
        value = _mm_sub_epi32(value,regi);
    }
    FASTOR_INLINE void operator-=(const SIMDVector<int,128> &a) {
        value = _mm_sub_epi32(value,a.value);
    }

    FASTOR_INLINE void operator*=(int num) {
        value = _mm_mul_epi32(value,_mm_set1_epi32(num));
    }
    FASTOR_INLINE void operator*=(__m128i regi) {
        value = _mm_mul_epi32(value,regi);
    }
    FASTOR_INLINE void operator*=(const SIMDVector<int,128> &a) {
        value = _mm_mul_epi32(value,a.value);
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

    FASTOR_INLINE int sum() {
        int *vals = (int*)&value;
        int quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            quan += vals[i];
        return quan;
    }

    FASTOR_INLINE int dot(const SIMDVector<int,128> &other) {
        int *vals0 = (int*)&value;
        int *vals1 = (int*)&other.value;
        int quan = 0;
        for (FASTOR_INDEX i=0; i<Size; ++i)
            quan += vals0[i]*vals1[i];
        return quan;
    }

    __m128i value;
};

std::ostream& operator<<(std::ostream &os, SIMDVector<int,128> a) {
    const int *value = (int*) &a.value;
    os << "[" << value[0] <<  " " << value[1] << " " << value[2] << " " << value[3] << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<int,128> operator+(const SIMDVector<int,128> &a, const SIMDVector<int,128> &b) {
    SIMDVector<int,128> out;
    out.value = _mm_add_epi32(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int,128> operator+(const SIMDVector<int,128> &a, int b) {
    SIMDVector<int,128> out;
    out.value = _mm_add_epi32(a.value,_mm_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int,128> operator+(int a, const SIMDVector<int,128> &b) {
    SIMDVector<int,128> out;
    out.value = _mm_add_epi32(_mm_set1_epi32(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int,128> operator+(const SIMDVector<int,128> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<int,128> operator-(const SIMDVector<int,128> &a, const SIMDVector<int,128> &b) {
    SIMDVector<int,128> out;
    out.value = _mm_sub_epi32(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int,128> operator-(const SIMDVector<int,128> &a, int b) {
    SIMDVector<int,128> out;
    out.value = _mm_sub_epi32(a.value,_mm_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int,128> operator-(int a, const SIMDVector<int,128> &b) {
    SIMDVector<int,128> out;
    out.value = _mm_sub_epi32(_mm_set1_epi32(a),b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int,128> operator-(const SIMDVector<int,128> &b) {
    SIMDVector<int,128> out;
    out.value = _mm_castps_si128(_mm_neg_ps(_mm_castsi128_ps(b.value)));
    return out;
}

FASTOR_INLINE SIMDVector<int,128> operator*(const SIMDVector<int,128> &a, const SIMDVector<int,128> &b) {
    SIMDVector<int,128> out;
    out.value = _mm_mul_epi32x(a.value,b.value);
    return out;
}
FASTOR_INLINE SIMDVector<int,128> operator*(const SIMDVector<int,128> &a, int b) {
    SIMDVector<int,128> out;
    out.value = _mm_mul_epi32x(a.value,_mm_set1_epi32(b));
    return out;
}
FASTOR_INLINE SIMDVector<int,128> operator*(int a, const SIMDVector<int,128> &b) {
    SIMDVector<int,128> out;
    out.value = _mm_mul_epi32x(_mm_set1_epi32(a),b.value);
    return out;
}



#endif


// SCALAR VERSION
//------------------------------------------------------------------------------------------------------------
template <>
struct SIMDVector<int, 32> {
    static constexpr FASTOR_INDEX Size = 1;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return 1;}
    static constexpr int unroll_size(FASTOR_INDEX size) {return (static_cast<int>(size) - 1);}

    FASTOR_INLINE SIMDVector() : value(0) {}
    FASTOR_INLINE SIMDVector(int num) : value(num) {}
    FASTOR_INLINE SIMDVector(const SIMDVector<int,32> &a) : value(a.value) {}
    FASTOR_INLINE SIMDVector(const int *data) : value(*data) {}
    FASTOR_INLINE SIMDVector(int *data) : value(*data) {}

    FASTOR_INLINE SIMDVector<int,32> operator=(int num) {
        value = num;
        return *this;
    }
    FASTOR_INLINE SIMDVector<int,32> operator=(const SIMDVector<int,32> &a) {
        value = a.value;
        return *this;
    }

    FASTOR_INLINE void load(const int *data, bool ) {
        value = *data;
    }
    FASTOR_INLINE void store(int *data, bool ) {
        data[0] = value;
    }

    FASTOR_INLINE void load(const int *data) {
        value = *data;
    }
    FASTOR_INLINE void store(int *data) {
        data[0] = value;
    }

    FASTOR_INLINE int operator[](FASTOR_INDEX) {return value;}
    FASTOR_INLINE int operator()(FASTOR_INDEX) {return value;}

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
    FASTOR_INLINE void operator+=(const SIMDVector<int,32> &a) {
        value += a.value;
    }

    FASTOR_INLINE void operator-=(int num) {
        value -= num;
    }
    FASTOR_INLINE void operator-=(const SIMDVector<int,32> &a) {
        value -= a.value;
    }

    FASTOR_INLINE void operator*=(int num) {
        value *= num;
    }
    FASTOR_INLINE void operator*=(const SIMDVector<int,32> &a) {
        value *= a.value;
    }

    FASTOR_INLINE void operator/=(int num) {
        value /= num;
    }
    FASTOR_INLINE void operator/=(const SIMDVector<int,32> &a) {
        value /= a.value;
    }
    // end of in-place operators

    FASTOR_INLINE SIMDVector<int,32> shift(FASTOR_INDEX) {
        return *this;
    }
    FASTOR_INLINE int sum() {return value;}
    FASTOR_INLINE SIMDVector<int,32> reverse() {
        return *this;
    }
    FASTOR_INLINE int minimum() {return value;}
    FASTOR_INLINE int maximum() {return value;}

    FASTOR_INLINE int dot(const SIMDVector<int,32> &other) {
        return value*other.value;
    }

    int value;
};


std::ostream& operator<<(std::ostream &os, SIMDVector<int,32> a) {
    os << "[" << a.value << "]\n";
    return os;
}

FASTOR_INLINE SIMDVector<int,32> operator+(const SIMDVector<int,32> &a, const SIMDVector<int,32> &b) {
    SIMDVector<int,32> out;
    out.value = a.value+b.value;
    return out;
}
FASTOR_INLINE SIMDVector<int,32> operator+(const SIMDVector<int,32> &a, int b) {
    SIMDVector<int,32> out;
    out.value = a.value+b;
    return out;
}
FASTOR_INLINE SIMDVector<int,32> operator+(int a, const SIMDVector<int,32> &b) {
    SIMDVector<int,32> out;
    out.value = a+b.value;
    return out;
}
FASTOR_INLINE SIMDVector<int,32> operator+(const SIMDVector<int,32> &b) {
    return b;
}

FASTOR_INLINE SIMDVector<int,32> operator-(const SIMDVector<int,32> &a, const SIMDVector<int,32> &b) {
    SIMDVector<int,32> out;
    out.value = a.value-b.value;
    return out;
}
FASTOR_INLINE SIMDVector<int,32> operator-(const SIMDVector<int,32> &a, int b) {
    SIMDVector<int,32> out;
    out.value = a.value-b;
    return out;
}
FASTOR_INLINE SIMDVector<int,32> operator-(int a, const SIMDVector<int,32> &b) {
    SIMDVector<int,32> out;
    out.value = a-b.value;
    return out;
}
FASTOR_INLINE SIMDVector<int,32> operator-(const SIMDVector<int,32> &b) {
    SIMDVector<int,32> out;
    out.value = -b.value;
    return out;
}

FASTOR_INLINE SIMDVector<int,32> operator*(const SIMDVector<int,32> &a, const SIMDVector<int,32> &b) {
    SIMDVector<int,32> out;
    out.value = a.value*b.value;
    return out;
}
FASTOR_INLINE SIMDVector<int,32> operator*(const SIMDVector<int,32> &a, int b) {
    SIMDVector<int,32> out;
    out.value = a.value*b;
    return out;
}
FASTOR_INLINE SIMDVector<int,32> operator*(int a, const SIMDVector<int,32> &b) {
    SIMDVector<int,32> out;
    out.value = a*b.value;
    return out;
}

FASTOR_INLINE SIMDVector<int,32> operator/(const SIMDVector<int,32> &a, const SIMDVector<int,32> &b) {
    SIMDVector<int,32> out;
    out.value = a.value/b.value;
    return out;
}
FASTOR_INLINE SIMDVector<int,32> operator/(const SIMDVector<int,32> &a, int b) {
    SIMDVector<int,32> out;
    out.value = a.value/b;
    return out;
}
FASTOR_INLINE SIMDVector<int,32> operator/(int a, const SIMDVector<int,32> &b) {
    SIMDVector<int,32> out;
    out.value = a/b.value;
    return out;
}

FASTOR_INLINE SIMDVector<int,32> sqrt(const SIMDVector<int,32> &a) {
    return std::sqrt(a.value);
}


}


#endif // SIMD_VECTOR_INT_H
