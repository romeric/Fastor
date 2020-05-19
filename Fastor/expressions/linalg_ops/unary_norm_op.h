#ifndef UNARY_NORM_OP_H
#define UNARY_NORM_OP_H

#include "Fastor/meta/meta.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/backend/norm.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/expressions/expression_traits.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"

namespace Fastor {

// For tensors
template<typename T, enable_if_t_<is_arithmetic_v_<T>,bool> = false>
FASTOR_INLINE T norm(const T &a) {
    return std::abs(a);
}
template<typename T, size_t ... Rest>
FASTOR_INLINE T norm(const Tensor<T,Rest...> &a) {
    if (sizeof...(Rest) == 0)
        return *a.data();
    return _norm<T,pack_prod<Rest...>::value>(a.data());
}

// For generic expressions
template<class Derived, size_t DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::scalar_type norm(const AbstractTensor<Derived,DIMS> &_src) {

    const Derived &src = _src.self();
    using T = typename Derived::scalar_type;
    using V = choose_best_simd_vector_t<T>;
    T _scal=0;
#ifdef FASTOR_AVX512_IMPL
    V omm0, omm1, omm2, omm3, omm4, omm5, omm6, omm7;
#else
    V omm0, omm1, omm2, omm3;
#endif
    FASTOR_INDEX i = 0;

    // With AVX utilises all the 16 registers but hurts the performance
    // due to spill if eval has created temporary registers so only
    // activated for AVX512
#ifdef FASTOR_AVX512_IMPL
    for (; i < ROUND_DOWN(src.size(),8*V::Size); i+=8*V::Size) {
        const auto smm0 = src.template eval<T>(i);
        const auto smm1 = src.template eval<T>(i+V::Size);
        const auto smm2 = src.template eval<T>(i+2*V::Size);
        const auto smm3 = src.template eval<T>(i+3*V::Size);
        const auto smm4 = src.template eval<T>(i+4*V::Size);
        const auto smm5 = src.template eval<T>(i+5*V::Size);
        const auto smm6 = src.template eval<T>(i+6*V::Size);
        const auto smm7 = src.template eval<T>(i+7*V::Size);
        omm0 = fmadd(smm0,smm0,omm0);
        omm1 = fmadd(smm1,smm1,omm1);
        omm2 = fmadd(smm2,smm2,omm2);
        omm3 = fmadd(smm3,smm3,omm3);
        omm4 = fmadd(smm4,smm4,omm4);
        omm5 = fmadd(smm5,smm5,omm5);
        omm6 = fmadd(smm6,smm6,omm6);
        omm7 = fmadd(smm7,smm7,omm7);
    }
#endif
    for (; i < ROUND_DOWN(src.size(),4*V::Size); i+=4*V::Size) {
        const auto smm0 = src.template eval<T>(i);
        const auto smm1 = src.template eval<T>(i+V::Size);
        const auto smm2 = src.template eval<T>(i+2*V::Size);
        const auto smm3 = src.template eval<T>(i+3*V::Size);
        omm0 = fmadd(smm0,smm0,omm0);
        omm1 = fmadd(smm1,smm1,omm1);
        omm2 = fmadd(smm2,smm2,omm2);
        omm3 = fmadd(smm3,smm3,omm3);
    }
    for (; i < ROUND_DOWN(src.size(),2*V::Size); i+=2*V::Size) {
        const auto smm0 = src.template eval<T>(i);
        const auto smm1 = src.template eval<T>(i+V::Size);
        omm0 = fmadd(smm0,smm0,omm0);
        omm1 = fmadd(smm1,smm1,omm1);
    }
    for (; i < ROUND_DOWN(src.size(),V::Size); i+=V::Size) {
        const auto smm0 = src.template eval<T>(i);
        omm0 = fmadd(smm0,smm0,omm0);
    }
    for (; i < src.size(); ++i) {
        const auto smm0 = src.template eval_s<T>(i);
        _scal += smm0*smm0;
    }
#ifdef FASTOR_AVX512_IMPL
    return sqrts( (omm0 + omm1 + omm2 + omm3 + omm4 + omm5 + omm6 + omm7).sum() + _scal);
#else
    return sqrts( (omm0 + omm1 + omm2 + omm3).sum() + _scal);
#endif
}


template<class Derived, size_t DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::scalar_type norm(const AbstractTensor<Derived,DIMS> &_src) {
    const Derived &src = _src.self();
    using result_type = typename Derived::result_type;
    const result_type out(src);
    return norm(out);
}

} // end of namespace Fastor

#endif // UNARY_NORM_OP_H
