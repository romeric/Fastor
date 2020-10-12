#ifndef TENSOR_H
#define TENSOR_H

#include "Fastor/config/config.h"
#include "Fastor/util/util.h"
#include "Fastor/backend/backend.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/Ranges.h"
#include "Fastor/tensor/ForwardDeclare.h"
#include "Fastor/expressions/linalg_ops/linalg_ops.h"

#include <array>
#include <vector>

namespace Fastor {

template<typename T, size_t ... Rest>
class Tensor: public AbstractTensor<Tensor<T,Rest...>,sizeof...(Rest)> {
public:
    using scalar_type      = T;
    using simd_vector_type = choose_best_simd_vector_t<T>;
    using simd_abi_type    = typename simd_vector_type::abi_type;
    using result_type      = Tensor<T,Rest...>;
    using dimension_t      = std::integral_constant<FASTOR_INDEX, sizeof...(Rest)>;
    static constexpr FASTOR_INLINE FASTOR_INDEX rank() {return sizeof...(Rest);}
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return pack_prod<Rest...>::value;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX dim) const {
#if FASTOR_SHAPE_CHECK
        FASTOR_ASSERT(dim>=0 && dim < sizeof...(Rest), "TENSOR SHAPE MISMATCH");
#endif
        constexpr FASTOR_INDEX DimensionHolder[sizeof...(Rest)] = {Rest...};
        return DimensionHolder[dim];
    }
    FASTOR_INLINE Tensor<T,Rest...>& noalias() {return *this;}

    // Classic constructors
    //----------------------------------------------------------------------------------------------------------//
    // Default constructor
    constexpr FASTOR_INLINE Tensor() = default;

    // Copy constructor
    FASTOR_INLINE Tensor(const Tensor<T,Rest...> &other) {
        // This constructor cannot be default
        if (_data == other.data()) return;
        // fast memcopy
        std::copy(other.data(),other.data()+size(),_data);
    };

    // Constructor from a scalar
    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false>
    FASTOR_INLINE Tensor(U num) {
        assign(*this, num);
    }

    // Initialiser list constructors
    //----------------------------------------------------------------------------------------------------------//
    #include "Fastor/tensor/InitializerListConstructors.h"
    //----------------------------------------------------------------------------------------------------------//

    // Classic array wrappers
    //----------------------------------------------------------------------------------------------------------//
    FASTOR_INLINE Tensor(const T *arr, int layout=RowMajor) {
        std::copy(arr,arr+size(),_data);
        if (layout == RowMajor)
            return;
        else
            *this = tocolumnmajor(*this);
    }
    FASTOR_INLINE Tensor(const std::array<T,pack_prod<Rest...>::value> &arr, int layout=RowMajor) {
        std::copy(arr.data(),arr.data()+pack_prod<Rest...>::value,_data);
        if (layout == RowMajor)
            return;
        else
            *this = tocolumnmajor(*this);
    }
    FASTOR_INLINE Tensor(const std::vector<T> &arr, int layout=RowMajor) {
        std::copy(arr.data(),arr.data()+pack_prod<Rest...>::value,_data);
        if (layout == RowMajor)
            return;
        else
            *this = tocolumnmajor(*this);
    }
    //----------------------------------------------------------------------------------------------------------//

    // CRTP constructors
    //----------------------------------------------------------------------------------------------------------//
    //----------------------------------------------------------------------------------------------------------//
    // Generic AbstractTensors
#ifndef FASTOR_DISABLE_SPECIALISED_CTR
    template<typename Derived, size_t DIMS,
        enable_if_t_<(!has_tensor_view_v<Derived> && !has_tensor_fixed_view_2d_v<Derived> &&
            !has_tensor_fixed_view_nd_v<Derived>) || DIMS!=sizeof...(Rest),bool> = false>
#else
    template<typename Derived, size_t DIMS>
#endif
    FASTOR_INLINE Tensor(const AbstractTensor<Derived,DIMS>& src) {
        FASTOR_ASSERT(src.self().size()==size(), "TENSOR SIZE MISMATCH");
        assign(*this, src.self());
    }
    //----------------------------------------------------------------------------------------------------------//

    // Specialised constructors
    //----------------------------------------------------------------------------------------------------------//
    #include "Fastor/tensor/SpecialisedConstructors.h"
    //----------------------------------------------------------------------------------------------------------//

    // AbstractTensor and scalar in-place operators
    //----------------------------------------------------------------------------------------------------------//
    #include "Fastor/tensor/TensorInplaceOperators.h"
    //----------------------------------------------------------------------------------------------------------//

    // Raw pointer providers
    //----------------------------------------------------------------------------------------------------------//
#ifdef FASTOR_ZERO_INITIALISE
    constexpr FASTOR_INLINE T* data() const { return const_cast<T*>(this->_data);}
#else
    FASTOR_INLINE T* data() const { return const_cast<T*>(this->_data);}
#endif

    FASTOR_INLINE T* data() {return this->_data;}
    //----------------------------------------------------------------------------------------------------------//

    // Scalar & block indexing
    //----------------------------------------------------------------------------------------------------------//
    #include "Fastor/tensor/IndexRetriever.h"
    #include "Fastor/tensor/ScalarIndexing.h"
    #include "Fastor/tensor/BlockIndexing.h"
    //----------------------------------------------------------------------------------------------------------//

    // Expression templates evaluators
    //----------------------------------------------------------------------------------------------------------//
    #include "Fastor/tensor/TensorEvaluator.h"
    //----------------------------------------------------------------------------------------------------------//

    // Tensor methods
    //----------------------------------------------------------------------------------------------------------//
    #include "Fastor/tensor/TensorMethods.h"
    //----------------------------------------------------------------------------------------------------------//

    // Converters
    //----------------------------------------------------------------------------------------------------------//
    #include "Fastor/tensor/PODConverters.h"
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


    //----------------------------------------------------------------------------------------------------------//
protected:
    template<typename Derived, size_t DIMS>
    FASTOR_INLINE void verify_dimensions(const AbstractTensor<Derived,DIMS>& src_) const {
        static_assert(DIMS==dimension_t::value, "TENSOR RANK MISMATCH");
#ifndef NDEBUG
        const Derived &src = src_.self();
        FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<dimension_t::value; ++i) {
            FASTOR_ASSERT(src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
    }
    //----------------------------------------------------------------------------------------------------------//

    //----------------------------------------------------------------------------------------------------------//
private:
#ifdef FASTOR_ZERO_INITIALISE
    FASTOR_ALIGN T _data[pack_prod<Rest...>::value] = {};
#else
    FASTOR_ALIGN T _data[pack_prod<Rest...>::value];
#endif
    //----------------------------------------------------------------------------------------------------------//
};


} // end of namespace Fastor


#include "Fastor/tensor/TensorAssignment.h"


#endif // TENSOR_H

