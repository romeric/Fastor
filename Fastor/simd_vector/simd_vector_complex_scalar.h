#ifndef SIMD_VECTOR_COMPLEX_SCALAR
#define SIMD_VECTOR_COMPLEX_SCALAR

#include "Fastor/util/extended_algorithms.h"
#include "Fastor/simd_vector/extintrin.h"
#include "Fastor/simd_vector/simd_vector_base.h"
#include "Fastor/simd_vector/simd_vector_double.h"
#include <cmath>
#include <complex>


namespace Fastor {


// SCALAR IMPLEMENTATION OF SIMDVECTOR FOR COMPLEX<T>
//------------------------------------------------------------------------------------------------------------

template <typename T>
struct SIMDVector<std::complex<T>, simd_abi::scalar> {
    using vector_type = SIMDVector<std::complex<T>, simd_abi::scalar>;
    using value_type = T;
    using scalar_value_type = std::complex<T>;
    using abi_type = simd_abi::scalar;
    static constexpr FASTOR_INDEX Size = 1;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return 1;}

    FASTOR_INLINE SIMDVector() : value_r(0), value_i(0) {}
    FASTOR_INLINE SIMDVector(scalar_value_type num) {
        value_r = num.real();
        value_i = num.imag();
    }
    FASTOR_INLINE SIMDVector(value_type reg0, value_type reg1) : value_r(reg0), value_i(reg1) {}
    FASTOR_INLINE SIMDVector(const scalar_value_type *data, bool Aligned=true) {
        value_r = (*data).real();
        value_i = (*data).imag();
    }

    FASTOR_INLINE SIMDVector<T,simd_abi::scalar> operator=(std::complex<T> num) {
        value_r = num.real();
        value_i = num.imag();
        return *this;
    }

    FASTOR_INLINE void load(const scalar_value_type *data, bool ) {
        value_r = (*data).real();
        value_i = (*data).imag();
    }
    FASTOR_INLINE void store(scalar_value_type *data, bool ) const {
        data[0] = scalar_value_type(value_r,value_i);
    }

    FASTOR_INLINE void aligned_load(const T *data) {
        value_r = (*data).real();
        value_i = (*data).imag();
    }
    FASTOR_INLINE void aligned_store(T *data) const {
        data[0] = scalar_value_type(value_r,value_i);
    }

    FASTOR_INLINE void mask_load(const scalar_value_type *data, uint8_t mask, bool ) {
        if (mask != 0x0) {
            value_r = (*data).real();
            value_i = (*data).imag();
        }
    }
    FASTOR_INLINE void mask_store(scalar_value_type *data, uint8_t mask, bool) const {
        if (mask != 0x0) {
            data[0] = scalar_value_type(value_r,value_i);
        }
    }

    FASTOR_INLINE T operator[](FASTOR_INDEX) const {return scalar_value_type(value_r,value_i);}

    FASTOR_INLINE SIMDVector<T,simd_abi::scalar> real() const {
        return value_r;
    }
    FASTOR_INLINE SIMDVector<T,simd_abi::scalar> imag() const {
        return value_i;
    }

    FASTOR_INLINE void set(scalar_value_type num) {
        value_r = num.real();
        value_i = num.imag();
    }

    FASTOR_INLINE void set_sequential(scalar_value_type num) {
        value_r = num.real();
        value_i = num.imag();
    }

    // In-place operators
    template<typename U=T, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator+=(U num) {
        value_r += num;
    }
    FASTOR_INLINE void operator+=(scalar_value_type num) {
        value_r += num.real();
        value_i += num.imag();
    }
    FASTOR_INLINE void operator+=(const vector_type &a) {
        value_r += a.value_r;
        value_i += a.value_i;
    }

    template<typename U=T, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator-=(U num) {
        value_r -= num;
    }
    FASTOR_INLINE void operator-=(scalar_value_type num) {
        value_r -= num.real();
        value_i -= num.imag();
    }
    FASTOR_INLINE void operator-=(const vector_type &a) {
        value_r -= a.value_r;
        value_i -= a.value_i;
    }

    template<typename U=T, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator*=(U num) {
        value_r *= num;
        value_i *= num;
    }
    FASTOR_INLINE void operator*=(scalar_value_type num) {
        scalar_value_type tmp(value_r, value_i);
        tmp *= num;
        value_r = tmp.real();
        value_i = tmp.imag();
    }
    FASTOR_INLINE void operator*=(const vector_type &a) {
        scalar_value_type tmp0(value_r, value_i);
        scalar_value_type tmp1(a.value_r, a.value_i);
        tmp0 *= tmp1;
        value_r = tmp0.real();
        value_i = tmp0.imag();
    }

    template<typename U=T, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
    FASTOR_INLINE void operator/=(U num) {
        value_r /= num;
        value_i /= num;
    }
    FASTOR_INLINE void operator/=(scalar_value_type num) {
        scalar_value_type tmp(value_r, value_i);
        tmp /= num;
        value_r = tmp.real();
        value_i = tmp.imag();
    }
    FASTOR_INLINE void operator/=(const vector_type &a) {
        scalar_value_type tmp0(value_r, value_i);
        scalar_value_type tmp1(a.value_r, a.value_i);
        tmp0 /= tmp1;
        value_r = tmp0.real();
        value_i = tmp0.imag();
    }
    // end of in-place operators

    FASTOR_INLINE scalar_value_type sum() const {return scalar_value_type(value_r, value_i);}
    FASTOR_INLINE scalar_value_type product() const {return scalar_value_type(value_r, value_i);}
    FASTOR_INLINE vector_type reverse() const { return scalar_value_type(value_r, value_i); }
    FASTOR_INLINE SIMDVector<T,simd_abi::scalar> magnitude() const { return std::abs(scalar_value_type(value_r, value_i));}
    FASTOR_INLINE SIMDVector<T,simd_abi::scalar> norm() const { return std::norm(scalar_value_type(value_r, value_i));}
    FASTOR_INLINE scalar_value_type minimum() const {return scalar_value_type(value_r, value_i);}
    FASTOR_INLINE scalar_value_type maximum() const {return scalar_value_type(value_r, value_i);}
    FASTOR_INLINE scalar_value_type dot(const vector_type &other) const {
        return vector_type(value_r, value_i)*vector_type(other.value_r, other.value_i);
    }

    value_type value_r;
    value_type value_i;
};

template <typename T>
FASTOR_HINT_INLINE std::ostream& operator<<(std::ostream &os, SIMDVector<std::complex<T>,simd_abi::scalar> a) {
    os << "[" << a.value_r <<  signum_string(a.value_i) << std::abs(a.value_i) << "j]\n";
    return os;
}

template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator+(const SIMDVector<std::complex<T>,simd_abi::scalar> &a, const SIMDVector<std::complex<T>,simd_abi::scalar> &b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(a);
    out += b;
    return out;
}
template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator+(const SIMDVector<std::complex<T>,simd_abi::scalar> &a, std::complex<T> b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(a);
    out += b;
    return out;
}
template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator+(std::complex<T> a, const SIMDVector<std::complex<T>,simd_abi::scalar> &b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(a);
    out += b;
    return out;
}
template <typename T, typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator+(const SIMDVector<std::complex<T>,simd_abi::scalar> &a, U b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(a);
    out += b;
    return out;
}
template <typename T, typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator+(U a, const SIMDVector<std::complex<T>,simd_abi::scalar> &b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(std::complex<T>(a,0));
    out += b;
    return out;
}
template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator+(const SIMDVector<std::complex<T>,simd_abi::scalar> &b) {
    return b;
}

template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator-(const SIMDVector<std::complex<T>,simd_abi::scalar> &a, const SIMDVector<std::complex<T>,simd_abi::scalar> &b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(a);
    out -= b;
    return out;
}
template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator-(const SIMDVector<std::complex<T>,simd_abi::scalar> &a, std::complex<T> b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(a);
    out -= b;
    return out;
}
template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator-(std::complex<T> a, const SIMDVector<std::complex<T>,simd_abi::scalar> &b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(a);
    out -= b;
    return out;
}
template <typename T, typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator-(const SIMDVector<std::complex<T>,simd_abi::scalar> &a, U b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(a);
    out -= b;
    return out;
}
template <typename T, typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator-(U a, const SIMDVector<std::complex<T>,simd_abi::scalar> &b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(std::complex<T>(a,0));
    out -= b;
    return out;
}
template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator-(const SIMDVector<std::complex<T>,simd_abi::scalar> &b) {
    return SIMDVector<std::complex<T>,simd_abi::scalar>(0,0) - b;
}

template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator*(const SIMDVector<std::complex<T>,simd_abi::scalar> &a, const SIMDVector<std::complex<T>,simd_abi::scalar> &b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(a);
    out *= b;
    return out;
}
template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator*(const SIMDVector<std::complex<T>,simd_abi::scalar> &a, std::complex<T> b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(a);
    out *= b;
    return out;
}
template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator*(std::complex<T> a, const SIMDVector<std::complex<T>,simd_abi::scalar> &b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(a);
    out *= b;
    return out;
}
template <typename T, typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator*(const SIMDVector<std::complex<T>,simd_abi::scalar> &a, U b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(a);
    out *= b;
    return out;
}
template <typename T, typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator*(U a, const SIMDVector<std::complex<T>,simd_abi::scalar> &b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(std::complex<T>(a,0));
    out *= b;
    return out;
}

template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator/(const SIMDVector<std::complex<T>,simd_abi::scalar> &a, const SIMDVector<std::complex<T>,simd_abi::scalar> &b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(a);
    out /= b;
    return out;
}
template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator/(const SIMDVector<std::complex<T>,simd_abi::scalar> &a, std::complex<T> b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(a);
    out /= b;
    return out;
}
template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator/(std::complex<T> a, const SIMDVector<std::complex<T>,simd_abi::scalar> &b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(a);
    out /= b;
    return out;
}
template <typename T, typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator/(const SIMDVector<std::complex<T>,simd_abi::scalar> &a, U b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(a);
    out /= b;
    return out;
}
template <typename T, typename U, enable_if_t_<is_arithmetic_v_<U>,bool> = false>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
operator/(U a, const SIMDVector<std::complex<T>,simd_abi::scalar> &b) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(std::complex<T>(a,0));
    out /= b;
    return out;
}

template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
rcp(const SIMDVector<std::complex<T>,simd_abi::scalar> &a) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out(std::complex<T>(1,0));
    out /= a;
    return out;
}

template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
sqrt(const SIMDVector<std::complex<T>,simd_abi::scalar> &a) {
    std::complex<T> out(a.value_r,a.value_i);
    return std::sqrt(out);
}

template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
rsqrt(const SIMDVector<std::complex<T>,simd_abi::scalar> &a) {
    return rcp(sqrt(a));
}

template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
abs(const SIMDVector<std::complex<T>,simd_abi::scalar> &a) {
    SIMDVector<std::complex<T>,simd_abi::scalar> out;
    out.value_r = a.magnitude().value;
    return out;
}

template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
conj(const SIMDVector<std::complex<T>,simd_abi::scalar> &a) {
    std::complex<T> out(a.value_r,a.value_i);
    return std::conj(out);
}

template <typename T>
FASTOR_INLINE SIMDVector<std::complex<T>,simd_abi::scalar>
arg(const SIMDVector<std::complex<T>,simd_abi::scalar> &a) {
    std::complex<T> out(a.value_r,a.value_i);
    return SIMDVector<std::complex<T>,simd_abi::scalar>(std::arg(out),0);
}
//------------------------------------------------------------------------------------------------------------

} // end of namespace Fastor

#endif // SIMD_VECTOR_COMPLEX_SCALAR
