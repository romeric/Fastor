#ifndef UNARY_TRACE_OP_H
#define UNARY_TRACE_OP_H

#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/backend/determinant.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/tensor/TensorFunctions.h"
#include "Fastor/expressions/expression_traits.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"

namespace Fastor {


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