#ifndef TENSOR_REF_H
#define TENSOR_REF_H

#include "commons/commons.h"
#include "backend/backend.h"
#include "simd_vector/SIMDVector.h"
#include "AbstractTensor.h"
#include "ranges.h"
#include "ForwardDeclare.h"
#include "expressions/smart_ops/smart_ops.h"


namespace Fastor {

template<typename T, size_t ... Rest>
class TensorRef: public AbstractTensor<TensorRef<T, Rest...>,sizeof...(Rest)> {
public:
    using scalar_type = T;
    using Dimension_t = std::integral_constant<FASTOR_INDEX, sizeof...(Rest)>;
    static constexpr FASTOR_INDEX Dimension = sizeof...(Rest);
    static constexpr FASTOR_INDEX Size = prod<Rest...>::value;
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
    static constexpr FASTOR_INDEX Remainder = prod<Rest...>::value % sizeof(T);
    static constexpr FASTOR_INDEX rank() {return sizeof...(Rest);}
    FASTOR_INLINE FASTOR_INDEX size() const {return prod<Rest...>::value;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX dim) const {
#ifndef NDEBUG
        FASTOR_ASSERT(dim>=0 && dim < sizeof...(Rest), "TENSOR SHAPE MISMATCH");
#endif
        const FASTOR_INDEX DimensionHolder[sizeof...(Rest)] = {Rest...};
        return DimensionHolder[dim];
    }

    // constexpr TensorRef(const scalar_type* data) : _data(data) {}
    constexpr TensorRef(scalar_type* data) : _data(data) {}


    // Expression templates evaluators
    //----------------------------------------------------------------------------------------------------------//
#undef TENSOR_EVALUATOR_H
    #include "tensor/TensorEvaluator.h"
#define TENSOR_EVALUATOR_H
    //----------------------------------------------------------------------------------------------------------//
#undef SCALAR_INDEXING_NONCONST_H
#undef SCALAR_INDEXING_CONST_H
    #include "tensor/ScalarIndexing.h"
#define SCALAR_INDEXING_NONCONST_H
#define SCALAR_INDEXING_CONST_H
    // #include "BlockIndexingRef.h"


    // No constructor should be added not even CRTP constructors



private:
    scalar_type* _data;
    // const scalar_type* _data;
};


}

#endif // TENSOR_REF_H