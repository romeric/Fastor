#ifndef ABSTRACT_TENSOR_FUNCTIONS_H
#define ABSTRACT_TENSOR_FUNCTIONS_H

#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/TensorTraits.h"

namespace Fastor {

template<typename T, size_t ... Rest>
FASTOR_INLINE const Tensor<T,Rest...>& evaluate(const Tensor<T,Rest...> &src) {
    return src;
}
template<class Derived, size_t DIMS>
FASTOR_INLINE typename Derived::result_type evaluate(const AbstractTensor<Derived,DIMS> &src) {
    typename Derived::result_type out(src);
    return out;
}


// These are the set of functions work on any expression without themselves being a
// Fastor expression. Note that all the mathematical functions (sin, cos etc) are
// Fastor expressions


template<class Derived, size_t DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::scalar_type sum(const AbstractTensor<Derived,DIMS> &_src) {

    const Derived &src = _src.self();
    using T = typename Derived::scalar_type;
    using V = SIMDVector<T,DEFAULT_ABI>;
    FASTOR_INDEX i;
    T _scal=0; V _vec(_scal);
    for (i = 0; i < ROUND_DOWN(src.size(),V::Size); i+=V::Size) {
        _vec += src.template eval<T>(i);
    }
    for (; i < src.size(); ++i) {
        _scal += src.template eval_s<T>(i);
    }
    return _vec.sum() + _scal;
}

template<class Derived, size_t DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::scalar_type sum(const AbstractTensor<Derived,DIMS> &_src) {
    const Derived &src = _src.self();
    using result_type = typename Derived::result_type;
    const result_type out(src);
    return out.sum();
}


template<class Derived, size_t DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::scalar_type product(const AbstractTensor<Derived,DIMS> &_src) {

    const Derived &src = _src.self();
    using T = typename Derived::scalar_type;
    using V = SIMDVector<T,DEFAULT_ABI>;
    FASTOR_INDEX i;
    T _scal=0; V _vec(_scal);
    for (i = 0; i < ROUND_DOWN(src.size(),V::Size); i+=V::Size) {
        _vec *= src.template eval<T>(i);
    }
    for (; i < src.size(); ++i) {
        _scal *= src.template eval_s<T>(i);
    }
    return _vec.product() * _scal;
}

template<class Derived, size_t DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::scalar_type product(const AbstractTensor<Derived,DIMS> &_src) {
    const Derived &src = _src.self();
    using result_type = typename Derived::result_type;
    const result_type out(src);
    return out.product();
}


template<class Derived, size_t DIMS>
FASTOR_INLINE bool all_of(const AbstractTensor<Derived,DIMS> &_src) {
    static_assert(is_binary_cmp_op_v<Derived>,"INPUT SHOULD BE A BOOLEAN EXPRESSION");
    const Derived &src = _src.self();
    bool val = true;
    for (FASTOR_INDEX i = 0; i < src.size(); ++i) {
        if (src.template eval_s<bool>(i) == false) {
            val = false;
            break;
        }
    }
    return val;
}

template<class Derived, size_t DIMS>
FASTOR_INLINE bool any_of(const AbstractTensor<Derived,DIMS> &_src) {
    static_assert(is_binary_cmp_op_v<Derived>,"INPUT SHOULD BE A BOOLEAN EXPRESSION");
    const Derived &src = _src.self();
    bool val = false;
    for (FASTOR_INDEX i = 0; i < src.size(); ++i) {
        if (src.template eval_s<bool>(i) == true) {
            val = true;
            break;
        }
    }
    return val;
}

template<class Derived, size_t DIMS>
FASTOR_INLINE bool none_of(const AbstractTensor<Derived,DIMS> &_src) {
    static_assert(is_binary_cmp_op_v<Derived>,"INPUT SHOULD BE A BOOLEAN EXPRESSION");
    const Derived &src = _src.self();
    bool val = false;
    for (FASTOR_INDEX i = 0; i < src.size(); ++i) {
        if (src.template eval_s<bool>(i) == true) {
            val = true;
            break;
        }
    }
    return val;
}

}


#endif // #ifndef TENSOR_FUNCTIONS_H
