#ifndef ABSTRACT_TENSOR_FUNCTIONS_H
#define ABSTRACT_TENSOR_FUNCTIONS_H

#include "tensor/Tensor.h"

namespace Fastor {


// These are the set of functions work on any expression without themselves being a
// Fastor expression. Note that all the mathematical functions (sin, cos etc) are
// Fastor expressions

template<class Derived, size_t DIMS>
FASTOR_INLINE typename Derived::scalar_type norm(const AbstractTensor<Derived,DIMS> &_src) {
    using T = typename Derived::scalar_type;
    using V = SIMDVector<T,DEFAULT_ABI>;
    const Derived &src = _src.self();
    FASTOR_INDEX i;
    T _scal=0; V _vec(_scal);
    for (i = 0; i < ROUND_DOWN(src.size(),V::Size); i+=V::Size) {
        // Evaluate the expression once
        auto eval_vec = src.template eval<T>(i);
#ifdef __FMA__
        _vec = fmadd(eval_vec,eval_vec,_vec);
#else
        _vec += eval_vec*eval_vec;
#endif
    }
    for (; i < src.size(); ++i) {
        // Evaluate the expression once
        auto eval_scal = src.template eval_s<T>(i);
        _scal += eval_scal*eval_scal;
    }
    return sqrts(_vec.sum() + _scal);
}



template<class Derived0, typename Derived1, size_t DIMS,
    typename std::enable_if<std::is_same<typename Derived0::scalar_type, typename Derived1::scalar_type>::value,bool>::type=0>
FASTOR_INLINE typename Derived0::scalar_type inner(const AbstractTensor<Derived0,DIMS> &_a, const AbstractTensor<Derived1,DIMS> &_b) {
    using T = typename Derived0::scalar_type;
    using V = SIMDVector<T,DEFAULT_ABI>;
    const Derived0 &srca = _a.self();
    const Derived1 &srcb = _b.self();
#ifdef NDEBUG
    FASTOR_ASSERT(srca.size()==srcb.size(), "EXPRESSION SIZE MISMATCH");
#endif
    FASTOR_INDEX i;
    T _scal=0; V _vec(_scal);
    for (i = 0; i < ROUND_DOWN(srca.size(),V::Size); i+=V::Size) {
#ifdef __FMA__
        _vec = fmadd(srca.template eval<T>(i),srcb.template eval<T>(i),_vec);
#else
        _vec += srca.template eval<T>(i)*srcb.template eval<T>(i);
#endif
    }
    for (; i < srca.size(); ++i) {
        // Evaluate the expression once
        _scal += srca.template eval_s<T>(i)*srcb.template eval_s<T>(i);
    }
    return _vec.sum() + _scal;
}


}


#endif // #ifndef TENSOR_FUNCTIONS_H