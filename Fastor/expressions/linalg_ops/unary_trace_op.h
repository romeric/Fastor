#ifndef UNARY_TRACE_OP_H
#define UNARY_TRACE_OP_H

#include "Fastor/meta/meta.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/expressions/expression_traits.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"

namespace Fastor {

// For tensors
template<typename T, size_t I>
FASTOR_INLINE T trace(const Tensor<T,I,I> &a) {
    return _trace<T,I,I>(static_cast<const T *>(a.data()));
}

// For high order tensors
template<typename T, size_t ... Rest, typename std::enable_if<sizeof...(Rest)>=3,bool>::type=0>
FASTOR_INLINE
typename last_matrix_extracter<Tensor<T,Rest...>, typename std_ext::make_index_sequence<sizeof...(Rest)-2>::type>::type
trace(const Tensor<T,Rest...> &a) {

    using OutTensor = typename last_matrix_extracter<Tensor<T,Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Rest)-2>::type>::type;
    constexpr size_t remaining_product = last_matrix_extracter<Tensor<T,Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Rest)-2>::type>::remaining_product;

    constexpr size_t I = get_value<sizeof...(Rest)-1,Rest...>::value;
    constexpr size_t J = get_value<sizeof...(Rest),Rest...>::value;
    static_assert(I==J,"THE LAST TWO DIMENSIONS OF TENSOR MUST BE THE SAME");

    OutTensor out;
    T *a_data = a.data();
    T *out_data = out.data();

    for (size_t i=0; i<remaining_product; ++i) {
        out_data[i] = _trace<T,J,J>(static_cast<const T *>(a_data+i*J*J));
    }

    return out;
}

// Trace for generic expressions
template<class Derived, size_t DIMS, enable_if_t_<!requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::scalar_type trace(const AbstractTensor<Derived,DIMS> &_src) {

    const Derived& src = _src.self();
    using T = typename Derived::scalar_type;
    using tensor_type = typename Derived::result_type;
    constexpr size_t M = get_tensor_dimensions<tensor_type>::dims[0];
    constexpr size_t N = get_tensor_dimensions<tensor_type>::dims[1];
    static_assert(DIMS==2,"TENSOR EXPRESSION SHOULD BE UNIFORM (SQUARE)");
    static_assert(M==N,"TENSOR EXPRESSION SHOULD BE TWO DIMENSIONAL");

    FASTOR_INDEX i;
    T _scal=0;
    for (i = 0; i < M; ++i) {
        _scal += src.template eval_s<T>(i*(N+1));
    }
    return _scal;
}

template<class Derived, size_t DIMS, enable_if_t_<requires_evaluation_v<Derived>,bool> = false>
FASTOR_INLINE typename Derived::scalar_type trace(const AbstractTensor<Derived,DIMS> &_src) {

    using result_type = typename Derived::result_type;
    const result_type out(_src.self());
    return trace(out);
}

} // end of namespace Fastor

#endif // UNARY_TRACE_OP_H
