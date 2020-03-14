#ifndef TENSOR_MAP_H
#define TENSOR_MAP_H

#include "Fastor/commons/commons.h"
#include "Fastor/backend/backend.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/Ranges.h"
#include "Fastor/tensor/ForwardDeclare.h"
#include "Fastor/expressions/smart_ops/smart_ops.h"
#include "Fastor/tensor/TensorIO.h"


namespace Fastor {

template<typename T, size_t ... Rest>
class TensorMap: public AbstractTensor<TensorMap<T, Rest...>,sizeof...(Rest)> {
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

    // Constructors
    //----------------------------------------------------------------------------------------------------------//
    constexpr TensorMap(scalar_type* data) : _data(data) {}
    template<size_t ... RestOther> constexpr TensorMap(Tensor<T,RestOther...> &a) : _data(a.data()) {}
    //----------------------------------------------------------------------------------------------------------//

    // Scalar indexing
    //----------------------------------------------------------------------------------------------------------//
#undef SCALAR_INDEXING_NONCONST_H
#undef SCALAR_INDEXING_CONST_H
#undef INDEX_RETRIEVER_H
    #include "Fastor/tensor/IndexRetriever.h"
    #include "Fastor/tensor/ScalarIndexing.h"
#define INDEX_RETRIEVER_H
#define SCALAR_INDEXING_NONCONST_H
#define SCALAR_INDEXING_CONST_H
    // Expression templates evaluators
    //----------------------------------------------------------------------------------------------------------//
#undef TENSOR_EVALUATOR_H
    #include "Fastor/tensor/TensorEvaluator.h"
#define TENSOR_EVALUATOR_H
    //----------------------------------------------------------------------------------------------------------//

    // No constructor should be added
    // Provide generic AbstractTensors copy constructor though
    //----------------------------------------------------------------------------------------------------------//
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
    #include "Fastor/tensor/TensorInplaceOperators.h"
#define TENSOR_INPLACE_OPERATORS_H
    //----------------------------------------------------------------------------------------------------------//

    //----------------------------------------------------------------------------------------------------------//
#undef TENSOR_METHODS_CONST_H
#undef TENSOR_METHODS_NONCONST_H
    #include "Fastor/tensor/TensorMethods.h"
#define TENSOR_METHODS_CONST_H
#define TENSOR_METHODS_NONCONST_H
    //----------------------------------------------------------------------------------------------------------//

    // Converters
    //----------------------------------------------------------------------------------------------------------//
#undef PODCONVERTERS_H
    #include "Fastor/tensor/PODConverters.h"
#define PODCONVERTERS_H
    //----------------------------------------------------------------------------------------------------------//

    // Cast method
    //----------------------------------------------------------------------------------------------------------//
    template<typename U>
    FASTOR_INLINE Tensor<U,Rest...> cast() const {
        Tensor<U,Rest...> out;
        U *out_data = out.data();
        for (FASTOR_INDEX i=0; i<Size; ++i) {
            out_data[get_mem_index(i)] = static_cast<U>(_data[i]);
        }
        return out;
    }
    //----------------------------------------------------------------------------------------------------------//


private:
    scalar_type* _data;
};

FASTOR_MAKE_OS_STREAM_TENSOR0(TensorMap)
FASTOR_MAKE_OS_STREAM_TENSOR1(TensorMap)
FASTOR_MAKE_OS_STREAM_TENSOR2(TensorMap)
FASTOR_MAKE_OS_STREAM_TENSORn(TensorMap)


}

#endif // TENSOR_MAP_H