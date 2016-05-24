#ifndef SIMDVECTOR_H
#define SIMDVECTOR_H

#include "simd_vector_base.h"
#include "simd_vector_float.h"
#include "simd_vector_double.h"


// Generic overloads
namespace Fastor {

template<typename T>
SIMDVector<T> exp(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_exp(a.value);
    return out;
}

template<typename T>
SIMDVector<T> log(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_log(a.value);
    return out;
}

template<typename T, typename U>
SIMDVector<T> pow(const SIMDVector<T> &a, const SIMDVector<U> &b) {
    SIMDVector<T> out;
    out.value = internal_pow(a.value, b.value);
    return out;
}

template<typename T, typename U>
SIMDVector<T> pow(const SIMDVector<T> &a, U bb) {
    SIMDVector<T> out;
    SIMDVector<T> b = static_cast<T>(bb);
    out.value = internal_pow(a.value, b.value);
    return out;
}

template<typename T>
SIMDVector<T> sin(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_sin(a.value);
    return out;
}

template<typename T>
SIMDVector<T> cos(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_cos(a.value);
    return out;
}

template<typename T>
SIMDVector<T> tan(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_tan(a.value);
    return out;
}

template<typename T>
SIMDVector<T> asin(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_asin(a.value);
    return out;
}

template<typename T>
SIMDVector<T> acos(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_acos(a.value);
    return out;
}

template<typename T>
SIMDVector<T> atan(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_atan(a.value);
    return out;
}

template<typename T>
SIMDVector<T> sinh(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_sinh(a.value);
    return out;
}

template<typename T>
SIMDVector<T> cosh(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_cosh(a.value);
    return out;
}

template<typename T>
SIMDVector<T> tanh(const SIMDVector<T> &a) {
    SIMDVector<T> out;
    out.value = internal_tanh(a.value);
    return out;
}

}

#endif // SIMDVECTOR_H

