#ifndef SIMD_VECTOR_BASE_H
#define SIMD_VECTOR_BASE_H

#include "commons/commons.h"
#include "extended_intrinsics/extintrin.h"
#include "math/internal_math.h"


namespace Fastor {

template<typename T,int ABI=256>
struct get_vector_size {
    static const FASTOR_INDEX size = ABI/sizeof(T)/8;
};

template<>
struct get_vector_size<double,256> {
    static const FASTOR_INDEX size = 4;
};
template<>
struct get_vector_size<float,256> {
    static const FASTOR_INDEX size = 8;
};
template<>
struct get_vector_size<int,256> {
    // Note that 256bit integer arithmatics were introduced under AVX2
    static const FASTOR_INDEX size = 8;
};
template<>
struct get_vector_size<double,128> {
    static const FASTOR_INDEX size = 2;
};
template<>
struct get_vector_size<float,128> {
    static const FASTOR_INDEX size = 4;
};
template<>
struct get_vector_size<int,128> {
    static const FASTOR_INDEX size = 4;
};



// THE DEFAULT SIMDVector TAKES CARE FALLING BACK TO SCALAR CODE
// WHEREEVER SIMD IS NOT AVAILABLE
template <typename T, int ABI=256>
struct SIMDVector {
    static constexpr FASTOR_INDEX Size = get_vector_size<T,ABI>::size;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return get_vector_size<T,ABI>::size;}
    static constexpr int unroll_size(FASTOR_INDEX size) {return (static_cast<int>(size) - static_cast<int>(Size));}

    FASTOR_INLINE SIMDVector() {}
    FASTOR_INLINE SIMDVector(T num) {
        for (FASTOR_INDEX i=0; i<Size;++i)
            value[i] = num;
    }
    FASTOR_INLINE SIMDVector(const SIMDVector<T,ABI> &a) {
        std::copy(a.value,a.value+a.Size,value);
    }
    FASTOR_INLINE SIMDVector(const T *data) {
        std::copy(data,data+Size,value);
    }
    FASTOR_INLINE SIMDVector(T *data) {
        std::copy(data,data+Size,value);
    }

    FASTOR_INLINE SIMDVector<T,ABI> operator=(T num) {
        for (FASTOR_INDEX i=0; i<Size;++i)
            value[i] = num;
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
    FASTOR_INLINE void store(T *data, bool Aligned=true) {
        std::copy(value,value+Size,data);
        unused(Aligned);
    }

    FASTOR_INLINE T operator[](FASTOR_INDEX i) {return value[i];}
    FASTOR_INLINE T operator()(FASTOR_INDEX i) {return value[i];}

    FASTOR_INLINE void set(T num) {
        for (FASTOR_INDEX i=0; i<Size;++i)
            value[i] = num;
    }
    template<typename U, typename ... Args>
    FASTOR_INLINE void set(U first, Args ... args) {
        static_assert(sizeof...(args)==1,"CANNOT SET VECTOR WITH VALUES DUE TO ABI CONSIDERATION");
    }
    FASTOR_INLINE void set_sequential(T num0) {
        for (FASTOR_INDEX i=0; i<Size;++i)
            value[i] = num0+i;
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
        T FASTOR_ALIGN tmp[Size];
        std::copy(value,value+Size,tmp);
        for (FASTOR_INDEX i=0; i<Size;++i)
            value[i] = tmp[Size-1-i];
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

    T FASTOR_ALIGN value[Size];
};

template<typename T, int ABI>
std::ostream& operator<<(std::ostream &os, SIMDVector<T,ABI> a) {
    os << "[";
    for (FASTOR_INDEX i=0; i<a.size(); ++i)
        os << a.value[i] << " ";
    os << "]";
    return os;
}

template<typename T, int ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator+(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a.value[i] + b.value[i];
    return out;
}
template<typename U, typename T, int ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator+(const SIMDVector<T,ABI> &a, U b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a.value[i] + static_cast<T>(b);
    return out;
}
template<typename U, typename T, int ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator+(U a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = static_cast<T>(a) + b.value[i];
    return out;
}
template<typename T, int ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator+(const SIMDVector<T,ABI> &b) {
    return b;
}

template<typename T, int ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator-(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a.value[i] - b.value[i];
    return out;
}
template<typename U, typename T, int ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator-(const SIMDVector<T,ABI> &a, U b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a.value[i] - static_cast<T>(b);
    return out;
}
template<typename U, typename T, int ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator-(U a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = static_cast<T>(a) - b.value[i];
    return out;
}
template<typename T, int ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator-(const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = -b.value[i];
    return out;
}

template<typename T, int ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator*(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a.value[i] * b.value[i];
    return out;
}
template<typename U, typename T, int ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator*(const SIMDVector<T,ABI> &a, U b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a.value[i] * static_cast<T>(b);
    return out;
}
template<typename U, typename T, int ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator*(U a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = static_cast<T>(a) * b.value[i];
    return out;
}

template<typename T, int ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator/(const SIMDVector<T,ABI> &a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a.value[i] / b.value[i];
    return out;
}
template<typename U, typename T, int ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator/(const SIMDVector<T,ABI> &a, U b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = a.value[i] / static_cast<T>(b);
    return out;
}
template<typename U, typename T, int ABI>
FASTOR_INLINE SIMDVector<T,ABI> operator/(U a, const SIMDVector<T,ABI> &b) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<SIMDVector<T,ABI>::Size; ++i)
        out.value[i] = static_cast<T>(a) / b.value[i];
    return out;
}

template<typename T, int ABI>
FASTOR_INLINE SIMDVector<T,ABI> sqrt(const SIMDVector<T,ABI> &a) {
    SIMDVector<T,ABI> out;
    for (FASTOR_INDEX i=0; i<a.Size; ++i)
        out.value[i] = std::sqrt(a.value[i]);
    return out;
}

}

#endif // SIMD_VECTOR_H

