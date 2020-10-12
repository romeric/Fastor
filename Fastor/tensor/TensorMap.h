#ifndef TENSOR_MAP_H
#define TENSOR_MAP_H

#include "Fastor/config/config.h"
#include "Fastor/backend/backend.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/Ranges.h"
#include "Fastor/tensor/ForwardDeclare.h"
#include "Fastor/expressions/linalg_ops/linalg_ops.h"
#include "Fastor/tensor/TensorIO.h"


namespace Fastor {

template<typename T, size_t ... Rest>
class TensorMap: public AbstractTensor<TensorMap<T, Rest...>,sizeof...(Rest)> {
public:
    using scalar_type      = T;
    using simd_vector_type = choose_best_simd_vector_t<T>;
    using simd_abi_type    = typename simd_vector_type::abi_type;
    using result_type      = Tensor<remove_all_t<T>,Rest...>;
    using dimension_t      = std::integral_constant<FASTOR_INDEX, sizeof...(Rest)>;
    static constexpr FASTOR_INLINE FASTOR_INDEX rank() {return sizeof...(Rest);}
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return pack_prod<Rest...>::value;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX dim) const {
#if FASTOR_SHAPE_CHECK
        FASTOR_ASSERT(dim>=0 && dim < sizeof...(Rest), "TENSOR SHAPE MISMATCH");
#endif
        const FASTOR_INDEX DimensionHolder[sizeof...(Rest)] = {Rest...};
        return DimensionHolder[dim];
    }
    FASTOR_INLINE Tensor<T,Rest...>& noalias() {return *this;}

    // Constructors
    //----------------------------------------------------------------------------------------------------------//
    constexpr TensorMap(scalar_type* data) : _data(data) {}
    template<size_t ... RestOther> constexpr TensorMap(Tensor<T,RestOther...> &a) : _data(a.data()) {}
    //----------------------------------------------------------------------------------------------------------//

    // Raw pointer providers
    //----------------------------------------------------------------------------------------------------------//
    FASTOR_INLINE T* data() const { return const_cast<T*>(this->_data);}
    FASTOR_INLINE T* data() {return this->_data;}
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

    // Block indexing (all variants excluding iseq)
    //----------------------------------------------------------------------------------------------------------//
    template<typename ... Seq, enable_if_t_<!is_arithmetic_pack_v<Seq...> && !is_fixed_sequence_pack_v<Seq...>,bool> = false>
    FASTOR_INLINE TensorViewExpr<TensorMap<T,Rest...>,sizeof...(Seq)> operator()(Seq ... _seqs) {
        static_assert(dimension_t::value==sizeof...(Seq),"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
        return TensorViewExpr<TensorMap<T,Rest...>,sizeof...(Seq)>(*this, {_seqs...});
    }

    template<typename ...Fseq, enable_if_t_<is_fixed_sequence_pack_v<Fseq...>,bool> = false>
    FASTOR_INLINE TensorFixedViewExprnD<TensorMap<T,Rest...>,Fseq...> operator()(Fseq... ) {
        static_assert(dimension_t::value==sizeof...(Fseq),"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
        return TensorFixedViewExprnD<TensorMap<T,Rest...>,Fseq...>(*this);
    }

    FASTOR_INLINE TensorFilterViewExpr<TensorMap<T,Rest...>,Tensor<bool,Rest...>,sizeof...(Rest)>
    operator()(const Tensor<bool,Rest...> &_fl) {
        return TensorFilterViewExpr<TensorMap<T,Rest...>,Tensor<bool,Rest...>,sizeof...(Rest)>(*this,_fl);
    }
    FASTOR_INLINE TensorFilterViewExpr<TensorMap<T,Rest...>,TensorMap<bool,Rest...>,sizeof...(Rest)>
    operator()(const TensorMap<bool,Rest...> &_fl) {
        return TensorFilterViewExpr<TensorMap<T,Rest...>,TensorMap<bool,Rest...>,sizeof...(Rest)>(*this,_fl);
    }

    //----------------------------------------------------------------------------------------------------------//

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
    FASTOR_INLINE void operator=(const AbstractTensor<Derived,DIMS>& src) {
        FASTOR_ASSERT(src.self().size()==size(), "TENSOR SIZE MISMATCH");
        assign(*this, src.self());
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
        for (FASTOR_INDEX i=0; i<size(); ++i) {
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


template<typename Derived, size_t DIM, typename T, size_t ...Rest>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, const TensorMap<T,Rest...> &src) {
    if (dst.self().data()==src.data()) return;
    trivial_assign(dst.self(),src);
}
template<typename Derived, size_t DIM, typename T, size_t ...Rest>
FASTOR_INLINE void assign_add(const AbstractTensor<Derived,DIM> &dst, const TensorMap<T,Rest...> &src) {
    trivial_assign_add(dst.self(),src);
}
template<typename Derived, size_t DIM, typename T, size_t ...Rest>
FASTOR_INLINE void assign_sub(const AbstractTensor<Derived,DIM> &dst, const TensorMap<T,Rest...> &src) {
    trivial_assign_sub(dst.self(),src);
}
template<typename Derived, size_t DIM, typename T, size_t ...Rest>
FASTOR_INLINE void assign_mul(const AbstractTensor<Derived,DIM> &dst, const TensorMap<T,Rest...> &src) {
    trivial_assign_mul(dst.self(),src);
}
template<typename Derived, size_t DIM, typename T, size_t ...Rest>
FASTOR_INLINE void assign_div(const AbstractTensor<Derived,DIM> &dst, const TensorMap<T,Rest...> &src) {
    trivial_assign_div(dst.self(),src);
}


}

#endif // TENSOR_MAP_H
