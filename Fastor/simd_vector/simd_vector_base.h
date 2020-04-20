#ifndef SIMD_VECTOR_BASE_H
#define SIMD_VECTOR_BASE_H

#include "Fastor/commons/commons.h"
#include "Fastor/simd_vector/simd_vector_abi.h"
#include "Fastor/extended_intrinsics/extintrin.h"
#include "Fastor/math/internal_math.h"



namespace Fastor {

// The default SIMDVector class that falls back to scalar implementation
// if SIMD types are not available or if vectorisation is disallowed
//--------------------------------------------------------------------------------------------------------------------//
template <typename T, typename ABI = simd_abi::native>
struct SIMDVector {
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<T,ABI>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<T,ABI>>::value;}
    static constexpr int unroll_size(FASTOR_INDEX size) {return (static_cast<int>(size) - static_cast<int>(Size));}
    using value_type = T[Size];
    using scalar_value_type = T;
    using abi_type = ABI;

    FASTOR_INLINE SIMDVector() {
        std::fill(value, value+Size, 0.);
    }
    FASTOR_INLINE SIMDVector(T num) {
        std::fill(value, value+Size, num);
    }
    FASTOR_INLINE SIMDVector(const SIMDVector<T,ABI> &a) {
        std::copy(a.value,a.value+a.Size,value);
    }
    FASTOR_INLINE SIMDVector(const T *data, bool Aligned=true) {
        std::copy(data,data+Size,value);
    }
    FASTOR_INLINE SIMDVector(T *data, bool Aligned=true) {
        std::copy(data,data+Size,value);
    }

    FASTOR_INLINE SIMDVector<T,ABI> operator=(T num) {
        std::fill(value, value+Size, num);
        return *this;
    }
    FASTOR_INLINE SIMDVector<T,ABI> operator=(const SIMDVector<T,ABI> &a) {
        std::copy(a.value,a.value+a.Size,value);
        return *this;
    }

    FASTOR_INLINE void load(const T *data, bool Aligned=true) {
        std::copy(data,data+Size,value);
        unused(Aligned);
    }
    FASTOR_INLINE void store(T *data, bool Aligned=true) const {
        std::copy(value,value+Size,data);
        unused(Aligned);
    }

    FASTOR_INLINE void aligned_load(const T *data) {
        std::copy(data,data+Size,value);
    }
    FASTOR_INLINE void aligned_store(T *data) const {
        std::copy(value,value+Size,data);
    }

    FASTOR_INLINE void mask_load(const scalar_value_type *a, uint8_t mask, bool ) {
        // perhaps very inefficient but they never get used
        int maska[Size];
        mask_to_array(mask,maska);
        std::fill(value, value+Size, 0.);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                ((scalar_value_type*)&value)[Size - i - 1] = a[Size - i - 1];
            }
        }
    }
    FASTOR_INLINE void mask_store(scalar_value_type *a, uint8_t mask, bool ) const {
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
    }

    FASTOR_INLINE T operator[](FASTOR_INDEX i) const {return value[i];}
    FASTOR_INLINE T operator()(FASTOR_INDEX i) const {return value[i];}

    FASTOR_INLINE void set(T num) {
        for (FASTOR_INDEX i=0; i<Size;++i)
            value[i] = num;
    }
    template<typename U, typename ... Args>
    FASTOR_INLINE void set(U first, Args ... args) {
        T arr[Size] = {first,args...};
        std::reverse_copy(arr, arr+Size, value);
        // Relax this restriction
        static_assert(sizeof...(args)==Size,"CANNOT SET VECTOR WITH SPECIFIED NUMBER OF VALUES DUE TO ABI CONSIDERATION");
    }
    FASTOR_INLINE void set_sequential(T num0) {
        for (FASTOR_INDEX i=0; i<Size;++i)
            value[i] = num0+(T)i;
    }

    // In-place operators
    FASTOR_INLINE void operator+=(T num) {
        for (FASTOR_INDEX i=0; i<Size;++i)
            value[i] += num;
    }
    FASTOR_INLINE void operator+=(const SIMDVector<T,ABI> &a) {
        for (FASTOR_INDEX i=0; i<Size;++i)
            value[i] += a.value[i];
    }

    FASTOR_INLINE void operator-=(T num) {
        for (FASTOR_INDEX i=0; i<Size;++i)
            value[i] -= num;
    }
    FASTOR_INLINE void operator-=(const SIMDVector<T,ABI> &a) {
        for (FASTOR_INDEX i=0; i<Size;++i)
            value[i] -= a.value[i];
    }

    FASTOR_INLINE void operator*=(T num) {
        for (FASTOR_INDEX i=0; i<Size;++i)
            value[i] *= num;
    }
    FASTOR_INLINE void operator*=(const SIMDVector<T,ABI> &a) {
        for (FASTOR_INDEX i=0; i<Size;++i)
            value[i] *= a.value[i];
    }
    FASTOR_INLINE void operator/=(T num) {
        for (FASTOR_INDEX i=0; i<Size;++i)
            value[i] /= num;
    }
    FASTOR_INLINE void operator/=(const SIMDVector<T,ABI> &a) {
        for (FASTOR_INDEX i=0; i<Size;++i)
            value[i] /= a.value[i];
    }
    // end of in-place operators

    FASTOR_INLINE SIMDVector<T,ABI> shift(FASTOR_INDEX i) {
        SIMDVector<T,ABI> out;
        std::fill(out.value,out.value+out.Size,static_cast<T>(0));
        std::copy(value,value+Size, out.value+i);
        return out;
    }
    FASTOR_INLINE T sum() {
        T quan = 0;
        for (FASTOR_INDEX i=0; i<Size;++i)
            quan += value[i];
        return quan;
    }
    FASTOR_INLINE T product() {
        //! Don't use prod as that is the name of a meta-function
        T quan = 1;
        for (FASTOR_INDEX i=0; i<Size;++i)
            quan *= value[i];
        return quan;
    }
    FASTOR_INLINE SIMDVector<T,ABI> reverse() {
        SIMDVector<T,ABI> out;
        std::copy(value,value+Size,out.value);
        std::reverse(out.value,out.value+Size);
        return out;
    }
    FASTOR_INLINE T minimum() {
        T quan = 0;
        for (FASTOR_INDEX i=0; i<Size;++i)
            if (value[i]<quan)
                quan = value[i];
        return quan;
    }
    FASTOR_INLINE T maximum() {
        T quan = 0;
        for (FASTOR_INDEX i=0; i<Size;++i)
            if (value[i]>quan)
                quan = value[i];
        return quan;
    }

    FASTOR_INLINE T dot(const SIMDVector<T,ABI> &other) {
        T quan = 0;
        for (FASTOR_INDEX i=0; i<Size;++i)
            quan += value[i]*other.value[i];
        return quan;
    }

    template<typename U>
    FASTOR_INLINE SIMDVector<U,ABI> cast() {
        SIMDVector<U,ABI> out;
        for (FASTOR_INDEX i=0; i<Size;++i) {
            out.value[i] = static_cast<U>(value[i]);
        }
        return out;
    }

    T FASTOR_ALIGN value[Size];
};

template <typename T, typename ABI>
constexpr FASTOR_INDEX SIMDVector<T,ABI>::Size;

template<typename T, typename ABI>
FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<T,ABI> a) {
    os << "[";
    for (FASTOR_INDEX i=0; i<a.size(); ++i)
        os << a.value[i] << ' ';
    os << "]";
    return os;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator+(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a.value[i] + b.value[i];
    return out;
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator+(const SIMDVector<T,ABI> &a, T b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a.value[i] + b;
    return out;
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator+(T a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a + b.value[i];
    return out;
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator+(const SIMDVector<T,ABI> &b) {
    return b;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator-(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a.value[i] - b.value[i];
    return out;
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator-(const SIMDVector<T,ABI> &a, T b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a.value[i] - b;
    return out;
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator-(T a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a - b.value[i];
    return out;
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator-(const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = -b.value[i];
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator*(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a.value[i] * b.value[i];
    return out;
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator*(const SIMDVector<T,ABI> &a, T b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a.value[i] * b;
    return out;
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator*(T a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a * b.value[i];
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator/(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a.value[i] / b.value[i];
    return out;
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator/(const SIMDVector<T,ABI> &a, T b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a.value[i] / b;
    return out;
}
template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator/(T a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a / b.value[i];
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> rcp(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<a.Size; ++i)
        out.value[i] = T(1.)/a.value[i];
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> sqrt(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<a.Size; ++i)
        out.value[i] = std::sqrt(a.value[i]);
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> rsqrt(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<a.Size; ++i)
        out.value[i] = T(1.)/std::sqrt(a.value[i]);
    return out;
}

template<typename T, typename ABI>
FASTOR_INLINE SIMDVector<T,ABI> abs(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<a.Size; ++i)
        out.value[i] = std::abs(a.value[i]);
    return out;
}




// Binary comparison ops
//----------------------------------------------------------------------------------------------------------------//
#define FASTOR_MAKE_BINARY_CMP_SIMDVECTORS_OPS_(OP) \
template<typename T, typename ABI> \
FASTOR_INLINE SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> operator OP(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b) { \
    constexpr FASTOR_INDEX Size = SIMDVector<T,ABI>::Size;\
    T FASTOR_ALIGN val_a[Size];\
    a.store(val_a);\
    T FASTOR_ALIGN val_b[Size];\
    b.store(val_b);\
    SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> out;\
    bool FASTOR_ALIGN val_out[Size];\
    out.store(val_out);\
    for (FASTOR_INDEX i=0; i<Size; ++i) {\
        val_out[i] = val_a[i] OP val_b[i];\
    }\
    out.load(val_out);\
    return out;\
}\

FASTOR_MAKE_BINARY_CMP_SIMDVECTORS_OPS_(==)
FASTOR_MAKE_BINARY_CMP_SIMDVECTORS_OPS_(!=)
FASTOR_MAKE_BINARY_CMP_SIMDVECTORS_OPS_(>)
FASTOR_MAKE_BINARY_CMP_SIMDVECTORS_OPS_(<)
FASTOR_MAKE_BINARY_CMP_SIMDVECTORS_OPS_(>=)
FASTOR_MAKE_BINARY_CMP_SIMDVECTORS_OPS_(<=)
FASTOR_MAKE_BINARY_CMP_SIMDVECTORS_OPS_(&&)
FASTOR_MAKE_BINARY_CMP_SIMDVECTORS_OPS_(||)


#define FASTOR_MAKE_BINARY_CMP_SIMDVECTOR_SCALAR_OPS_(OP) \
template<typename T, typename U, typename ABI> \
FASTOR_INLINE SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> operator OP(const SIMDVector<T,ABI> &a, U b) { \
    constexpr FASTOR_INDEX Size = SIMDVector<T,ABI>::Size;\
    T FASTOR_ALIGN val_a[Size];\
    a.store(val_a);\
    SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> out;\
    bool FASTOR_ALIGN val_out[Size];\
    out.store(val_out);\
    for (FASTOR_INDEX i=0; i<Size; ++i) {\
        val_out[i] = val_a[i] OP T(b);\
    }\
    out.load(val_out);\
    return out;\
}\

FASTOR_MAKE_BINARY_CMP_SIMDVECTOR_SCALAR_OPS_(==)
FASTOR_MAKE_BINARY_CMP_SIMDVECTOR_SCALAR_OPS_(!=)
FASTOR_MAKE_BINARY_CMP_SIMDVECTOR_SCALAR_OPS_(>)
FASTOR_MAKE_BINARY_CMP_SIMDVECTOR_SCALAR_OPS_(<)
FASTOR_MAKE_BINARY_CMP_SIMDVECTOR_SCALAR_OPS_(>=)
FASTOR_MAKE_BINARY_CMP_SIMDVECTOR_SCALAR_OPS_(<=)
FASTOR_MAKE_BINARY_CMP_SIMDVECTOR_SCALAR_OPS_(&&)
FASTOR_MAKE_BINARY_CMP_SIMDVECTOR_SCALAR_OPS_(||)


#define FASTOR_MAKE_BINARY_CMP_SCALAR_SIMDVECTOR_OPS_(OP) \
template<typename T, typename U, typename ABI> \
FASTOR_INLINE SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> operator OP(U a, const SIMDVector<T,ABI> &b) { \
    constexpr FASTOR_INDEX Size = SIMDVector<T,ABI>::Size;\
    T FASTOR_ALIGN val_b[Size];\
    b.store(val_b);\
    SIMDVector<bool,simd_abi::fixed_size<SIMDVector<T,ABI>::Size>> out;\
    bool FASTOR_ALIGN val_out[Size];\
    out.store(val_out);\
    for (FASTOR_INDEX i=0; i<Size; ++i) {\
        val_out[i] = T(a) OP val_b[i];\
    }\
    out.load(val_out);\
    return out;\
}\

FASTOR_MAKE_BINARY_CMP_SCALAR_SIMDVECTOR_OPS_(==)
FASTOR_MAKE_BINARY_CMP_SCALAR_SIMDVECTOR_OPS_(!=)
FASTOR_MAKE_BINARY_CMP_SCALAR_SIMDVECTOR_OPS_(>)
FASTOR_MAKE_BINARY_CMP_SCALAR_SIMDVECTOR_OPS_(<)
FASTOR_MAKE_BINARY_CMP_SCALAR_SIMDVECTOR_OPS_(>=)
FASTOR_MAKE_BINARY_CMP_SCALAR_SIMDVECTOR_OPS_(<=)
FASTOR_MAKE_BINARY_CMP_SCALAR_SIMDVECTOR_OPS_(&&)
FASTOR_MAKE_BINARY_CMP_SCALAR_SIMDVECTOR_OPS_(||)
//----------------------------------------------------------------------------------------------------------------//

}

#endif // SIMD_VECTOR_H

