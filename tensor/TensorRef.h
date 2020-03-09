#ifndef TENSOR_REF_H
#define TENSOR_REF_H

#include "commons/commons.h"
#include "backend/backend.h"
#include "simd_vector/SIMDVector.h"
#include "AbstractTensor.h"
#include "Ranges.h"
#include "ForwardDeclare.h"
#include "expressions/smart_ops/smart_ops.h"
#include <tensor/TensorIO.h>


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

    // Provide generic AbstractTensors copy constructor though
    template<typename Derived, size_t DIMS>
    FASTOR_INLINE void operator=(const AbstractTensor<Derived,DIMS>& src_) {
        const Derived &src = src_.self();
#ifndef NDEBUG
        FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
            src.template eval<T>(i).store(_data+i, IS_ALIGNED);
        }
        for (; i < src.size(); ++i) {
            _data[i] = src.template eval_s<T>(i);
        }
    }

    // AbstractTensor and scalar in-place operators
    //----------------------------------------------------------------------------------------------------------//
#undef TENSOR_INPLACE_OPERATORS_H
    #include "TensorInplaceOperators.h"
#define TENSOR_INPLACE_OPERATORS_H
    //----------------------------------------------------------------------------------------------------------//




private:
    scalar_type* _data;
    // const scalar_type* _data;
};

OS_STREAM_TENSOR0(TensorRef)
OS_STREAM_TENSOR1(TensorRef)
OS_STREAM_TENSOR2(TensorRef)
OS_STREAM_TENSORn(TensorRef)


}

#endif // TENSOR_REF_H