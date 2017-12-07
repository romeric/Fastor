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

    // TensorRef(const scalar_type* data) : _data(data) {}
    TensorRef(scalar_type* data) : _data(data) {}


    // Expression templates evaluators
    //----------------------------------------------------------------------------------------------------------//
    template<typename U=T>
    FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
#ifdef BOUNDSCHECK
        // This is a generic evaluator and not for 1D cases only
        FASTOR_ASSERT((i>=0 && i<Size), "INDEX OUT OF BOUNDS");
#endif
        SIMDVector<T,DEFAULT_ABI> out;
        out.load(_data+i,false);
        return out;
    }
    template<typename U=T>
    FASTOR_INLINE T eval_s(FASTOR_INDEX i) const {
#ifdef BOUNDSCHECK
        // This is a generic evaluator and not for 1D cases only
        FASTOR_ASSERT((i>=0 && i<Size), "INDEX OUT OF BOUNDS");
#endif
        return _data[i];
    }
    template<typename U=T>
    FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        static_assert(Dimension==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
        constexpr int N = get_value<2,Rest...>::value;
#ifdef BOUNDSCHECK
        constexpr int M = get_value<1,Rest...>::value;
        FASTOR_ASSERT((i>=0 && i<M && j>=0 && j<N), "INDEX OUT OF BOUNDS");
#endif
        // return SIMDVector<T,DEFAULT_ABI>(&_data[i*N+j]); // Careful, causes segfaults
        SIMDVector<T,DEFAULT_ABI> _vec; _vec.load(&_data[i*N+j],false);
        return _vec;
    }
    template<typename U=T>
    FASTOR_INLINE T eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        static_assert(Dimension==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
#ifdef BOUNDSCHECK
        constexpr int M = get_value<1,Rest...>::value;
        constexpr int N = get_value<2,Rest...>::value;
        FASTOR_ASSERT((i>=0 && i<M && j>=0 && j<N), "INDEX OUT OF BOUNDS");
#endif
        return _data[i*get_value<2,Rest...>::value+j];
    }

    constexpr FASTOR_INLINE T eval(T i, T j) const {
        return _data[static_cast<FASTOR_INDEX>(i)*get_value<2,Rest...>::value+static_cast<FASTOR_INDEX>(j)];
    }
    //----------------------------------------------------------------------------------------------------------//
    #undef SCALAR_INDEXING_H
    #include "tensor/ScalarIndexing.h"
    #define SCALAR_INDEXING_H
    // #include "BlockIndexingRef.h"


    // No constructor should be added not even CRTP constructors



private:
    scalar_type* _data;
    // const scalar_type* _data;
};


}

#endif // TENSOR_REF_H