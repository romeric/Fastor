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


/* These are the set of functions that work on any expression that evaluate immediately
*/

/* Add all the elements of the tensor in a flattened sense
*/
template<class Derived, size_t DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::scalar_type sum(const AbstractTensor<Derived,DIMS> &_src) {
    const Derived &src = _src.self();
    using result_type = typename Derived::result_type;
    const result_type out(src);
    return out.sum();
}
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

/* Multiply all the elements of the tensor in a flattened sense
*/
template<class Derived, size_t DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::scalar_type product(const AbstractTensor<Derived,DIMS> &_src) {
    const Derived &src = _src.self();
    using result_type = typename Derived::result_type;
    const result_type out(src);
    return out.product();
}
template<class Derived, size_t DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::scalar_type product(const AbstractTensor<Derived,DIMS> &_src) {

    const Derived &src = _src.self();
    using T = typename Derived::scalar_type;
    using V = SIMDVector<T,DEFAULT_ABI>;
    FASTOR_INDEX i;
    T _scal=1; V _vec(_scal);
    for (i = 0; i < ROUND_DOWN(src.size(),V::Size); i+=V::Size) {
        _vec *= src.template eval<T>(i);
    }
    for (; i < src.size(); ++i) {
        _scal *= src.template eval_s<T>(i);
    }
    return _vec.product() * _scal;
}

/* Get the lower triangular matrix from a 2D expression
*/
template<class Derived, size_t DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::scalar_type tril(const AbstractTensor<Derived,DIMS> &_src, int k = 0) {
    static_assert(DIMS==2,"TENSOR HAS TO BE 2D FOR TRIL");
    const Derived &src = _src.self();
    using result_type = typename Derived::result_type;
    const result_type out(src);
    return tril(out);
}
template<class Derived, size_t DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::result_type tril(const AbstractTensor<Derived,DIMS> &_src, int k = 0) {
    static_assert(DIMS==2,"TENSOR HAS TO BE 2D FOR TRIL");
    const Derived &src = _src.self();
    using T = typename Derived::scalar_type;
    typename Derived::result_type out(0);

    int M = int(src.dimension(0));
    int N = int(src.dimension(1));
    for (int i = 0; i < M; ++i) {
        int jcount =  k + i < N ? k + i : N - 1;
        for (int j = 0; j <= jcount; ++j) {
            out(i,j) = src.template eval_s<T>(i,j);
        }
    }
    return out;
}

/* Get the upper triangular matrix from a 2D expression
*/
template<class Derived, size_t DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::scalar_type triu(const AbstractTensor<Derived,DIMS> &_src, int k = 0) {
    static_assert(DIMS==2,"TENSOR HAS TO BE 2D FOR TRIU");
    const Derived &src = _src.self();
    using result_type = typename Derived::result_type;
    const result_type out(src);
    return triu(out);
}
template<class Derived, size_t DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::result_type triu(const AbstractTensor<Derived,DIMS> &_src, int k = 0) {
    static_assert(DIMS==2,"TENSOR HAS TO BE 2D FOR TRIU");
    const Derived &src = _src.self();
    using T = typename Derived::scalar_type;
    typename Derived::result_type out(0);

    int M = int(src.dimension(0));
    int N = int(src.dimension(1));
    for (int i = 0; i < M; ++i) {
        int jcount =  k + i < 0 ? 0 : k + i;
        for (int j = jcount; j < N; ++j) {
            out(i,j) = src.template eval_s<T>(i,j);
        }
    }
    return out;
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
