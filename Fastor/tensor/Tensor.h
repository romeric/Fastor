#ifndef TENSOR_H
#define TENSOR_H

#include "Fastor/commons/commons.h"
#include "Fastor/backend/backend.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/Ranges.h"
#include "Fastor/tensor/ForwardDeclare.h"
#include "Fastor/expressions/smart_ops/smart_ops.h"
#include "Fastor/meta/tensor_pre_meta.h"

namespace Fastor {

template<typename T, size_t ... Rest>
class Tensor: public AbstractTensor<Tensor<T,Rest...>,sizeof...(Rest)> {
private:
#ifdef FASTOR_ZERO_INITIALISE
    T FASTOR_ALIGN _data[prod<Rest...>::value] = {};
#else
    T FASTOR_ALIGN _data[prod<Rest...>::value];
#endif
public:
    using scalar_type = T;
    using Dimension_t = std::integral_constant<FASTOR_INDEX, sizeof...(Rest)>;
    static constexpr FASTOR_INDEX Dimension = sizeof...(Rest);
    static constexpr FASTOR_INDEX Size = prod<Rest...>::value;
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
    static constexpr FASTOR_INDEX Remainder = prod<Rest...>::value % sizeof(T);
    static constexpr FASTOR_INLINE FASTOR_INDEX rank() {return Dimension;}
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {return Size;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX dim) const {
#ifndef NDEBUG
        FASTOR_ASSERT(dim>=0 && dim < sizeof...(Rest), "TENSOR SHAPE MISMATCH");
#endif
        const FASTOR_INDEX DimensionHolder[sizeof...(Rest)] = {Rest...};
        return DimensionHolder[dim];
    }


    // Classic constructors
    //----------------------------------------------------------------------------------------------------------//
    constexpr FASTOR_INLINE Tensor(){}

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    FASTOR_INLINE Tensor(U num) {
        SIMDVector<T,DEFAULT_ABI> reg(static_cast<T>(num));
        FASTOR_INDEX i;
        for (i = 0; i< ROUND_DOWN(Size,Stride); i+=Stride) {
            reg.store(&_data[i]);
        }
        for (; i<Size; ++i) {
            _data[i] = (T)num;
        }
    }

    FASTOR_INLINE Tensor(const Tensor<T,Rest...> &other) {
        // This constructor cannot be default
        // Note that all other data members are static constexpr

        std::copy(other.data(),other.data()+Size,_data);
        // using V = SIMDVector<T,DEFAULT_ABI>;
        // const T* other_data = other.data();
        // FASTOR_INDEX i=0;
        // for (; i<ROUND_DOWN(Size,V::Size); i+=V::Size) {
        //     V(&other_data[i]).store(&_data[i]);
        // }
        // for (; i<Size; ++i) {
        //     _data[i] = other_data[i];
        // }
    };

    // List initialisers
    FASTOR_INLINE Tensor(const std::initializer_list<T> &lst) {
        static_assert(sizeof...(Rest)==1,"TENSOR RANK MISMATCH WITH LIST-INITIALISER");
#ifdef BOUNDSCHECK
        FASTOR_ASSERT(prod<Rest...>::value==lst.size(), "TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
#endif
        auto counter = 0;
        for (auto &i: lst) {_data[counter] = i; counter++;}
    }

    FASTOR_INLINE Tensor(const std::initializer_list<std::initializer_list<T>> &lst2d) {
        static_assert(sizeof...(Rest)==2,"TENSOR RANK MISMATCH WITH LIST-INITIALISER");
#ifndef NDEBUG
        constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
        constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
        auto size_ = 0;
        FASTOR_ASSERT(M==lst2d.size(),"TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
        for (auto &lst: lst2d) {
            auto curr_size = lst.size();
            FASTOR_ASSERT(N==lst.size(),"TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
            size_ += curr_size;
        }
        FASTOR_ASSERT(prod<Rest...>::value==size_, "TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
#endif
        auto counter = 0;
        for (auto &lst1d: lst2d) {for (auto &i: lst1d) {_data[counter] = i; counter++;}}
    }

    FASTOR_INLINE Tensor(const std::initializer_list<std::initializer_list<std::initializer_list<T>>> &lst3d) {
        static_assert(sizeof...(Rest)==3,"TENSOR RANK MISMATCH WITH LIST-INITIALISER");
#ifndef NDEBUG
        constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
        constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
        constexpr FASTOR_INDEX P = get_value<3,Rest...>::value;
        auto size_ = 0;
        FASTOR_ASSERT(M==lst3d.size(),"TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
        for (auto &lst2d: lst3d) {
            FASTOR_ASSERT(N==lst2d.size(),"TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
            for (auto &lst: lst2d) {
                auto curr_size = lst.size();
                FASTOR_ASSERT(P==lst.size(),"TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
                size_ += curr_size;
            }
        }
        FASTOR_ASSERT(prod<Rest...>::value==size_, "TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
#endif
        auto counter = 0;
        for (auto &lst2d: lst3d) {for (auto &lst1d: lst2d) {for (auto &i: lst1d) {_data[counter] = i; counter++;}}}
    }

    FASTOR_INLINE Tensor(const std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>> &lst4d) {
        static_assert(sizeof...(Rest)==4,"TENSOR RANK MISMATCH WITH LIST-INITIALISER");
#ifndef NDEBUG
        constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
        constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
        constexpr FASTOR_INDEX P = get_value<3,Rest...>::value;
        constexpr FASTOR_INDEX Q = get_value<4,Rest...>::value;
        auto size_ = 0;
        FASTOR_ASSERT(M==lst4d.size(),"TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
        for (auto &lst3d: lst4d) {
            FASTOR_ASSERT(N==lst3d.size(),"TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
            for (auto &lst2d: lst3d) {
                FASTOR_ASSERT(P==lst2d.size(),"TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
                for (auto &lst: lst2d) {
                    auto curr_size = lst.size();
                    FASTOR_ASSERT(Q==curr_size,"TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
                    size_ += curr_size;
                }
            }
        }
        FASTOR_ASSERT(prod<Rest...>::value==size_, "TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
#endif
        auto counter = 0;
        for (auto &lst3d: lst4d) {for (auto &lst2d: lst3d) {for (auto &lst1d: lst2d) {for (auto &i: lst1d) {_data[counter] = i; counter++;}}}}
    }

    // Classic array wrappers
    FASTOR_INLINE Tensor(const T *arr, int layout=RowMajor) {
        std::copy(arr,arr+Size,_data);
        if (layout == RowMajor)
            return;
        else
            *this = tocolumnmajor(*this);
    }
    FASTOR_INLINE Tensor(const std::array<T,sizeof...(Rest)> &arr) {std::copy(arr,arr+prod<Rest...>::value,_data);}
    //----------------------------------------------------------------------------------------------------------//

    // CRTP constructors
    //----------------------------------------------------------------------------------------------------------//
    //----------------------------------------------------------------------------------------------------------//
    // Generic AbstractTensors
    template<typename Derived, size_t DIMS,
        typename std::enable_if<(!has_tensor_view<Derived>::value &&
        !has_tensor_fixed_view_2d<Derived>::value) || DIMS!=sizeof...(Rest),bool>::type=0>
    FASTOR_INLINE Tensor(const AbstractTensor<Derived,DIMS>& src_) {
        using scalar_type_ = typename scalar_type_finder<Derived>::type;
        constexpr FASTOR_INDEX Stride_ = stride_finder<scalar_type_>::value;
        const Derived &src = src_.self();
#ifndef NDEBUG
        FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(src.size(),Stride_); i+=Stride_) {
            src.template eval<T>(i).store(&_data[i], IS_ALIGNED);
        }
        for (; i < src.size(); ++i) {
            _data[i] = src.template eval_s<T>(i);
        }
    }

    template<typename Derived, size_t DIMS,
        typename std::enable_if<(!has_tensor_view<Derived>::value &&
        !has_tensor_fixed_view_2d<Derived>::value) || DIMS!=sizeof...(Rest),bool>::type=0>
    FASTOR_INLINE Tensor<T,Rest...>& operator=(const AbstractTensor<Derived,DIMS>& src_) {
        using scalar_type_ = typename scalar_type_finder<Derived>::type;
        constexpr FASTOR_INDEX Stride_ = stride_finder<scalar_type_>::value;
        const Derived &src = src_.self();
#ifndef NDEBUG
        FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(src.size(),Stride_); i+=Stride_) {
            src.template eval<T>(i).store(&_data[i], IS_ALIGNED);
        }
        for (; i < src.size(); ++i) {
            _data[i] = src.template eval_s<T>(i);
        }
        return *this;
    }
    //----------------------------------------------------------------------------------------------------------//

    // Specialised constructors
    //----------------------------------------------------------------------------------------------------------//
    #include "SpecialisedConstructors.h"
    //----------------------------------------------------------------------------------------------------------//

    // In-place operators
    //----------------------------------------------------------------------------------------------------------//

    FASTOR_INLINE void operator +=(const Tensor<T,Rest...> &a) {
        using V = SIMDVector<T,DEFAULT_ABI>;
        T* a_data = a.data();
        V _vec;
        FASTOR_INDEX i=0;
        for (; i<ROUND_DOWN(Size,V::Size); i+=V::Size) {
            _vec = V(&_data[i]) + V(&a_data[i]);
            _vec.store(&_data[i]);
        }
        for (; i<Size; ++i) {
            _data[i] += a_data[i];
        }
    }

    FASTOR_INLINE void operator -=(const Tensor<T,Rest...> &a) {
        using V = SIMDVector<T,DEFAULT_ABI>;
        T* a_data = a.data();
        V _vec;
        FASTOR_INDEX i=0;
        for (; i<ROUND_DOWN(Size,V::Size); i+=V::Size) {
            _vec = V(&_data[i]) - V(&a_data[i]);
            _vec.store(&_data[i]);
        }
        for (; i<Size; ++i) {
            _data[i] -= a_data[i];
        }
    }

    FASTOR_INLINE void operator *=(const Tensor<T,Rest...> &a) {
        using V = SIMDVector<T,DEFAULT_ABI>;
        T* a_data = a.data();
        V _vec;
        FASTOR_INDEX i=0;
        for (; i<ROUND_DOWN(Size,V::Size); i+=V::Size) {
            _vec = V(&_data[i]) * V(&a_data[i]);
            _vec.store(&_data[i]);
        }
        for (; i<Size; ++i) {
            _data[i] *= a_data[i];
        }
    }

    FASTOR_INLINE void operator /=(const Tensor<T,Rest...> &a) {
        using V = SIMDVector<T,DEFAULT_ABI>;
        T* a_data = a.data();
        V _vec;
        FASTOR_INDEX i=0;
        for (; i<ROUND_DOWN(Size,V::Size); i+=V::Size) {
            _vec = V(&_data[i]) / V(&a_data[i]);
            _vec.store(&_data[i]);
        }
        for (; i<Size; ++i) {
            _data[i] /= a_data[i];
        }
    }

    // AbstractTensor and scalar in-place operators
    //----------------------------------------------------------------------------------------------------------//
    #include "TensorInplaceOperators.h"
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
    #include "IndexRetriever.h"
    #include "ScalarIndexing.h"
    #include "BlockIndexing.h"
    //----------------------------------------------------------------------------------------------------------//

    // Expression templates evaluators
    //----------------------------------------------------------------------------------------------------------//
    #include "TensorEvaluator.h"
    //----------------------------------------------------------------------------------------------------------//

    // FMA overloads
    //----------------------------------------------------------------------------------------------------------//
    // Disable this as these are not treated as specialisations and
    // hence lead to compilation errors concerning ambiguoity.
    // Ultimately -ffp-contract=fast should achieve the same thing
// #ifdef __FMA__
//     #include "FMAPlugin.h"
// #endif
    //----------------------------------------------------------------------------------------------------------//
    // Disable this as these are not treated as specialisations and
    // hence lead to compilation errors concerning ambiguoity.
    // Ultimately -ffp-contract=fast should achieve the same thing
    // #include "AuxiliaryPlugin.h"
    //----------------------------------------------------------------------------------------------------------//
    #include "SmartExpressionsPlugin.h"
    //----------------------------------------------------------------------------------------------------------//

    // Tensor methods
    //----------------------------------------------------------------------------------------------------------//
    #include "TensorMethods.h"
    //----------------------------------------------------------------------------------------------------------//

    // Converters
    //----------------------------------------------------------------------------------------------------------//
    #include "PODConverters.h"
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

    // Boolean functions
    //----------------------------------------------------------------------------------------------------------//
    constexpr FASTOR_INLINE bool is_uniform() const {
        //! A tensor is uniform if it spans equally in all dimensions,
        //! i.e. generalisation of square matrix to n dimension
        return no_of_unique<Rest...>::value==1 ? true : false;
    }

    template<typename U, size_t ... RestOther>
    FASTOR_INLINE bool is_equal(const Tensor<U,RestOther...> &other, const double Tol=PRECI_TOL) const {
        //! Two tensors are equal if they have the same type, rank, size and elements
        if(!std::is_same<T,U>::value) return false;
        if(sizeof...(Rest)!=sizeof...(RestOther)) return false;
        if(prod<Rest...>::value!=prod<RestOther...>::value) return false;
        else {
            bool out = true;
            const T *other_data = other.data();
            for (size_t i=0; i<Size; ++i) {
                if (std::fabs(_data[i]-other_data[i])>Tol) {
                    out = false;
                    break;
                }
            }
            return out;
        }
    }

    template<typename U, size_t ... RestOther>
    FASTOR_INLINE bool operator ==(const Tensor<U,RestOther...> &other) const {
        //! Two tensors are equal if they have the same type, rank, size and elements
            return is_equal(other);
    }

    template<typename U, size_t ... RestOther>
    FASTOR_INLINE bool operator !=(const Tensor<U,RestOther...> &other) const {
        //! Two tensors are equal if they have the same type, rank, size and elements
            return !is_equal(other);
    }

    FASTOR_INLINE bool is_orthogonal() const {
        //! A second order tensor A is orthogonal if A*A'= I
        if (!is_uniform())
            return false;
        else {
            static_assert(sizeof...(Rest)==2,"ORTHOGONALITY OF MATRIX WITH RANK!=2 CANNOT BE DETERMINED");
            Tensor<T,Rest...> out;
            out = matmul(transpose(*this),*this);
            Tensor<T,Rest...> ey; ey.eye();
            return is_equal(ey);
        }
    }

    FASTOR_INLINE bool does_belong_to_so3(const double Tol=PRECI_TOL) const {
        //! A second order tensor belongs to special orthogonal 3D group if
        //! it is orthogonal and its determinant is +1
        if (is_orthogonal()) {
            // Check if we are in 3D space
            if (prod<Rest...>::value!=9) {
                return false;
            }
            T out = _det<T,Rest...>(_data);
            if (std::fabs(out-1)>Tol) {
                return false;
            }
            return true;
        }
        else {
            return false;
        }
    }

    FASTOR_INLINE bool does_belong_to_sl3(const double Tol=PRECI_TOL) const {
        //! A second order tensor belongs to special linear 3D group if
        //! its determinant is +1
        T out = _det<T,Rest...>(_data);
        if (std::fabs(out-1.)>Tol) {
            return false;
        }
        return true;
    }

    FASTOR_INLINE bool is_symmetric(const double Tol=PRECI_TOL) const {
        if (is_uniform()) {
            bool bb = true;
            size_t M = dimension(0);
            size_t N = dimension(1);
            for (size_t i=0; i<M; ++i)
                for (size_t j=0; j<N; ++j)
                    if (std::fabs(_data[i*N+j] - _data[j*N+i])>Tol) {
                        bb = false;
                    }
            return bb;
        }
        else {
            return false;
        }
    }
    template<typename ... Args, typename std::enable_if<sizeof...(Args)==2,bool>::type=0>
    FASTOR_INLINE bool is_symmetric(Args ...) const {
        return true;
    }

    FASTOR_INLINE bool is_deviatoric(const double Tol=PRECI_TOL) const {
        if (std::fabs(trace(*this))<Tol)
            return true;
        else
            return false;
    }
    //----------------------------------------------------------------------------------------------------------//

protected:
    template<typename Derived, size_t DIMS>
    FASTOR_INLINE void verify_dimensions(const AbstractTensor<Derived,DIMS>& src_) const {
        static_assert(DIMS==Dimension, "TENSOR RANK MISMATCH");
#ifndef NDEBUG
        const Derived &src = src_.self();
        FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
    }
    //----------------------------------------------------------------------------------------------------------//

};


}


#endif // TENSOR_H

