#ifndef UNARY_DET_OP_H
#define UNARY_DET_OP_H

#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/backend/determinant.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/tensor/TensorFunctions.h"
#include "Fastor/expressions/expression_traits.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"

namespace Fastor {

// Just like matmul the verbose versions should not work on expressions?
template<class Derived, size_t DIMS>
FASTOR_INLINE typename Derived::scalar_type determinant(const AbstractTensor<Derived,DIMS> &_src) {
    const Derived &src = _src.self();
    using result_type = typename Derived::result_type;
    const result_type out = evaluate(src);
    return determinant(out);
}

template<class Derived, size_t DIMS>
FASTOR_INLINE typename Derived::scalar_type det(const AbstractTensor<Derived,DIMS> &_src) {
    const Derived &src = _src.self();
    using result_type = typename Derived::result_type;
    const result_type out = evaluate(src);
    return determinant(out);
}


} // end of namespace Fastor

#endif // UNARY_DET_OP_H