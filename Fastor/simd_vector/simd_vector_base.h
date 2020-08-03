#ifndef SIMD_VECTOR_BASE_H
#define SIMD_VECTOR_BASE_H

#include "Fastor/meta/meta.h"
#include "Fastor/config/config.h"
#include "Fastor/simd_vector/extintrin.h"
#include "Fastor/simd_vector/simd_vector_abi.h"

#include<cmath>
#include<complex>


namespace Fastor {

/* The default SIMDVector class that falls back to scalar implementation
* if SIMD types are not available or if vectorisation is disallowed
*/
//--------------------------------------------------------------------------------------------------------------------//
template <typename CVT, typename ABI = simd_abi::native>
struct SIMDVector {
    using T = remove_cv_ref_t<CVT>;
    static constexpr FASTOR_INDEX Size = internal::get_simd_vector_size<SIMDVector<T,ABI>>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return internal::get_simd_vector_size<SIMDVector<T,ABI>>::value;}
    using vector_type = SIMDVector<T,ABI>;
    using value_type = T[Size];
    using scalar_value_type = T;
    using abi_type = ABI;

    FASTOR_INLINE SIMDVector() : value{} {}
    FASTOR_INLINE SIMDVector(T num) { std::fill(value, value+Size, num); }
    FASTOR_INLINE SIMDVector(const SIMDVector<T,ABI> &a) { std::copy(a.value,a.value+a.Size,value); }
    FASTOR_INLINE SIMDVector(const T *data, bool Aligned=true) { std::copy(data,data+Size,value); unused(Aligned); }

    FASTOR_INLINE SIMDVector<T,ABI> operator=(T num) { std::fill(value, value+Size, num); return *this;}
    FASTOR_INLINE SIMDVector<T,ABI> operator=(const SIMDVector<T,ABI> &a) { std::copy(a.value,a.value+a.Size,value); return *this; };

    FASTOR_INLINE void load(const T *data, bool Aligned=true )  { std::copy(data,data+Size,value);  unused(Aligned);}
    FASTOR_INLINE void store(T *data, bool Aligned=true ) const { std::copy(value,value+Size,data); unused(Aligned);}

    FASTOR_INLINE void aligned_load(const T *data)  { std::copy(data,data+Size,value); }
    FASTOR_INLINE void aligned_store(T *data) const { std::copy(value,value+Size,data);}

    FASTOR_INLINE void mask_load(const scalar_value_type *a, uint8_t mask, bool Aligned=false) {
        // perhaps very inefficient but they never get used
        int maska[Size];
        mask_to_array(mask,maska);
        std::fill(value, value+Size, 0);
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            if (maska[i] == -1) {
                ((scalar_value_type*)&value)[Size - i - 1] = a[Size - i - 1];
            }
        }
        unused(Aligned);
    }
    FASTOR_INLINE void mask_store(scalar_value_type *a, uint8_t mask, bool Aligned=false) const {
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
    }

    FASTOR_INLINE T operator[](FASTOR_INDEX i) const {return value[i];}
    FASTOR_INLINE T operator()(FASTOR_INDEX i) const {return value[i];}

    // For compatibility with complex simd vector
    template<typename U=T, enable_if_t_<is_complex_v_<U>,bool> = false>
    FASTOR_INLINE SIMDVector<simd_cmplx_value_t<vector_type>,ABI> real() const {
        simd_cmplx_value_t<vector_type> arr[Size];
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            arr[i] = value[i].real();
        }
        SIMDVector<simd_cmplx_value_t<vector_type>,ABI> out(arr,false);
        return out;
    }
    template<typename U=T, enable_if_t_<is_complex_v_<U>,bool> = false>
    FASTOR_INLINE SIMDVector<simd_cmplx_value_t<vector_type>,ABI> imag() const {
        simd_cmplx_value_t<vector_type> arr[Size];
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            arr[i] = value[i].imag();
        }
        SIMDVector<simd_cmplx_value_t<vector_type>,ABI> out(arr,false);
        return out;
    }

    FASTOR_INLINE void set(T num) {
        for (FASTOR_INDEX i=0; i<Size;++i)
            value[i] = num;
    }
    template<typename U, typename ... Args>
    FASTOR_INLINE void set(U first, Args ... args) {
        static_assert(sizeof...(args)+1==Size,"CANNOT SET VECTOR WITH SPECIFIED NUMBER OF VALUES DUE TO ABI CONSIDERATION");
        T arr[Size] = {first,args...};
        std::reverse_copy(arr, arr+Size, value);
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
    // This is for comatibility with complex simd vectors
    template<typename U=T, enable_if_t_<is_complex_v_<U>,bool> = false>
    FASTOR_INLINE SIMDVector<simd_cmplx_value_t<vector_type>,ABI> magnitude() {
        simd_cmplx_value_t<vector_type> arr[Size];
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            arr[i] = std::abs(value[i]);
        }
        SIMDVector<simd_cmplx_value_t<vector_type>,ABI> out(arr,false);
        return out;
    }
    template<typename U=T, enable_if_t_<is_complex_v_<U>,bool> = false>
    FASTOR_INLINE SIMDVector<simd_cmplx_value_t<vector_type>,ABI> norm() {
        simd_cmplx_value_t<vector_type> arr[Size];
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            arr[i] = std::norm(value[i]);
        }
        SIMDVector<simd_cmplx_value_t<vector_type>,ABI> out(arr,false);
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

#ifdef FASTOR_ZERO_INITIALISE
    FASTOR_ARCH_ALIGN T value[Size] = {};
#else
    FASTOR_ARCH_ALIGN T value[Size];
#endif
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

// For compatibility with complex simd vectors
template<typename T, typename ABI, enable_if_t_<is_complex_v_<T>,bool> = false>
FASTOR_INLINE SIMDVector<T,ABI> conj(const SIMDVector<T,ABI> &a) {
    T arr[SIMDVector<T,ABI>::Size];
    for (FASTOR_INDEX i=0UL; i<SIMDVector<T,ABI>::Size; ++i) {
       arr[i] = std::conj(a[i]);
    }
    return SIMDVector<T,ABI>(arr,false);
}
template<typename T, typename ABI, enable_if_t_<is_complex_v_<T>,bool> = false>
FASTOR_INLINE SIMDVector<T,ABI> arg(const SIMDVector<T,ABI> &a) {
    T arr[SIMDVector<T,ABI>::Size];
    for (FASTOR_INDEX i=0UL; i<SIMDVector<T,ABI>::Size; ++i) {
       arr[i] = std::arg(a[i]);
    }
    return SIMDVector<T,ABI>(arr,false);
}

} // end of namespace Fastor

#endif // SIMD_VECTOR_H

