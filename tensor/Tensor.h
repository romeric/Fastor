#ifndef TENSOR_H
#define TENSOR_H

#include "commons/commons.h"
#include "backend/backend.h"
#include "simd_vector/SIMDVector.h"
#include "AbstractTensor.h"
#include "ranges.h"
#include "ForwardDeclare.h"
#include "expressions/smart_ops/smart_ops.h"

namespace Fastor {


template<typename T, size_t ... Rest>
class Tensor: public AbstractTensor<Tensor<T,Rest...>,sizeof...(Rest)> {
private:
    T FASTOR_ALIGN _data[prod<Rest...>::value];
public:
    typedef T scalar_type;
    static constexpr FASTOR_INDEX Dimension = sizeof...(Rest);
    static constexpr FASTOR_INDEX Size = prod<Rest...>::value;
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
    static constexpr FASTOR_INDEX Remainder = prod<Rest...>::value % sizeof(T);
    static constexpr FASTOR_INLINE FASTOR_INDEX rank() {return Dimension;}
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
            reg.store(_data+i);
        }
        for (; i<Size; ++i) {
            _data[i] = (T)num;
        }
    }

    FASTOR_INLINE Tensor(const Tensor<T,Rest...> &other) {
        // This constructor cannot be default
        // Note that all other data members are static constexpr 
        std::copy(other.data(),other.data()+Size,_data);
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
        constexpr FASTOR_INDEX Q = get_value<3,Rest...>::value;
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
    FASTOR_INLINE Tensor(const T *arr) {std::copy(arr,arr+prod<Rest...>::value,_data);}
    FASTOR_INLINE Tensor(const std::array<T,sizeof...(Rest)> &arr) {std::copy(arr,arr+prod<Rest...>::value,_data);}
    //----------------------------------------------------------------------------------------------------------//

    // CRTP constructors
    //----------------------------------------------------------------------------------------------------------//
    //----------------------------------------------------------------------------------------------------------//
    // Generate both copy constructor and copy assignment operator, for validity and safety reasons
    // that expressions are evaluated directly into this
    template<typename Derived>
    FASTOR_INLINE Tensor(const AbstractTensor<Derived,sizeof...(Rest)>& src_) {
        verify_dimensions(src_);
        const Derived &src = src_.self();
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
            src.template eval<T>(i).store(_data+i, IS_ALIGNED);
        }
        for (; i < src.size(); ++i) {
            _data[i] = src.template eval_s<T>(i);
        }
    }

    template<typename Derived>
    FASTOR_INLINE Tensor<T,Rest...>& operator=(const AbstractTensor<Derived,sizeof...(Rest)>& src_) {
        verify_dimensions(src_);
        const Derived &src = src_.self();
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
            src.template eval<T>(i).store(_data+i, IS_ALIGNED);
        }
        for (; i < src.size(); ++i) {
            _data[i] = src.template eval_s<T>(i);
        }
        return *this;
    }

    // Generic AbstractTensors
    template<typename Derived, size_t DIMS, typename std::enable_if<DIMS!=sizeof...(Rest),bool>::type=0>
    FASTOR_INLINE Tensor(const AbstractTensor<Derived,DIMS>& src_) {
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

    template<typename Derived, size_t DIMS, typename std::enable_if<DIMS!=sizeof...(Rest),bool>::type=0>
    FASTOR_INLINE Tensor<T,Rest...>& operator=(const AbstractTensor<Derived,DIMS>& src_) {
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
        return *this;
    }

    // In-place operators
    //----------------------------------------------------------------------------------------------------------//

    FASTOR_INLINE void operator +=(const Tensor<T,Rest...> &a) {
        using V = SIMDVector<T,DEFAULT_ABI>;
        T* a_data = a.data();
        V _vec;
        FASTOR_INDEX i=0;
        for (; i<ROUND_DOWN(Size,V::Size); i+=V::Size) {
            _vec = V(_data+i) + V(a_data+i);
            _vec.store(_data+i);
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
            _vec = V(_data+i) - V(a_data+i);
            _vec.store(_data+i);
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
            _vec = V(_data+i) * V(a_data+i);
            _vec.store(_data+i);
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
            _vec = V(_data+i) / V(a_data+i);
            _vec.store(_data+i);
        }
        for (; i<Size; ++i) {
            _data[i] /= a_data[i];
        }
    }

    // CRTP Overloads for nth rank Tensors
    //---------------------------------------------------------------------------------------------//
    template<typename Derived, size_t DIMS>
    FASTOR_INLINE void operator +=(const AbstractTensor<Derived,DIMS>& src_) {
        const Derived &src = src_.self();
#ifdef NDEBUG
        FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
        using V = SIMDVector<T,DEFAULT_ABI>;
        V _vec;
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(Size,V::Size); i+=V::Size) {
            _vec = V(_data+i) + src.template eval<T>(i);
            _vec.store(_data+i);
        }
        for (; i < Size; ++i) {
            _data[i] += src.template eval_s<T>(i);
        }
    }

    template<typename Derived, size_t DIMS>
    FASTOR_INLINE void operator -=(const AbstractTensor<Derived,DIMS>& src_) {
        const Derived &src = src_.self();
#ifdef NDEBUG
        FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
        using V = SIMDVector<T,DEFAULT_ABI>;
        V _vec;
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(Size,V::Size); i+=V::Size) {
            _vec = V(_data+i) - src.template eval<T>(i);
            _vec.store(_data+i);
        }
        for (; i < Size; ++i) {
            _data[i] -= src.template eval_s<T>(i);
        }
    }

    template<typename Derived, size_t DIMS>
    FASTOR_INLINE void operator *=(const AbstractTensor<Derived,DIMS>& src_) {
        const Derived &src = src_.self();
#ifdef NDEBUG
        FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
        using V = SIMDVector<T,DEFAULT_ABI>;
        V _vec;
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(Size,V::Size); i+=V::Size) {
            _vec = V(_data+i) * src.template eval<T>(i);
            _vec.store(_data+i);
        }
        for (; i < Size; ++i) {
            _data[i] *= src.template eval_s<T>(i);
        }
    }

    template<typename Derived, size_t DIMS>
    FASTOR_INLINE void operator /=(const AbstractTensor<Derived,DIMS>& src_) {
        const Derived &src = src_.self();
#ifdef NDEBUG
        FASTOR_ASSERT(src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
        using V = SIMDVector<T,DEFAULT_ABI>;
        V _vec;
        FASTOR_INDEX i;
        for (i = 0; i <ROUND_DOWN(Size,V::Size); i+=V::Size) {
            _vec = V(_data+i) / src.template eval<T>(i);
            _vec.store(_data+i);
        }
        for (; i < Size; ++i) {
            _data[i] /= src.template eval_s<T>(i);
        }
    }
    //---------------------------------------------------------------------------------------------//

    // Scalar overloads for in-place operators
    //---------------------------------------------------------------------------------------------//
    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    FASTOR_INLINE void operator +=(U num) {
        using V = SIMDVector<T,DEFAULT_ABI>;
        V _vec, _vec_a((T)num);
        FASTOR_INDEX i=0;
        for (; i<ROUND_DOWN(Size,V::Size); i+=V::Size) {
            _vec = V(_data+i) + _vec_a;
            _vec.store(_data+i);
        }
        for (; i<Size; ++i) {
            _data[i] += (U)(num);
        }
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    FASTOR_INLINE void operator -=(U num) {
        using V = SIMDVector<T,DEFAULT_ABI>;
        V _vec, _vec_a((T)num);
        FASTOR_INDEX i=0;
        for (; i<ROUND_DOWN(Size,V::Size); i+=V::Size) {
            _vec = V(_data+i) - _vec_a;
            _vec.store(_data+i);
        }
        for (; i<Size; ++i) {
            _data[i] -= (U)(num);
        }
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    FASTOR_INLINE void operator *=(U num) {
        using V = SIMDVector<T,DEFAULT_ABI>;
        V _vec, _vec_a((T)num);
        FASTOR_INDEX i=0;
        for (; i<ROUND_DOWN(Size,V::Size); i+=V::Size) {
            _vec = V(_data+i) * _vec_a;
            _vec.store(_data+i);
        }
        for (; i<Size; ++i) {
            _data[i] *= (U)(num);
        }
    }

    template<typename U=T, typename std::enable_if<std::is_arithmetic<U>::value,bool>::type=0>
    FASTOR_INLINE void operator /=(U num) {
        using V = SIMDVector<T,DEFAULT_ABI>;
        T inum = T(1)/T(num);
        V _vec, _vec_a(inum);
        FASTOR_INDEX i=0;
        for (; i<ROUND_DOWN(Size,V::Size); i+=V::Size) {
            _vec = V(_data+i) * _vec_a;
            _vec.store(_data+i);
        }
        for (; i<Size; ++i) {
            _data[i] *= inum;
        }
    }
    //----------------------------------------------------------------------------------------------------------//


    // Expression templates evaluators
    //----------------------------------------------------------------------------------------------------------//
    template<typename U=T>
    FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
#ifdef BOUNDSCHECK
        // This is a generic evaluator and not for 1D cases only
        FASTOR_ASSERT((i>=0 && i<Size), "INDEX OUT OF BOUNDS");
#endif
        SIMDVector<T,DEFAULT_ABI> out;
        out.load(_data+i);
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

    constexpr FASTOR_INLINE const Tensor<T,Rest...>& evaluate() const {
        return *this;
    }
    //----------------------------------------------------------------------------------------------------------//

    // Raw pointer providers
    //----------------------------------------------------------------------------------------------------------//
    FASTOR_INLINE T* data() const { return const_cast<T*>(this->_data);}
    FASTOR_INLINE T* data() {return this->_data;}
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
    // Scalar & block indexing
    //----------------------------------------------------------------------------------------------------------//
    #include "ScalarIndexing.h"
    #include "BlockIndexing.h"
    //----------------------------------------------------------------------------------------------------------//

    // Further member functions
    //----------------------------------------------------------------------------------------------------------//
    template<typename U=T>
    FASTOR_INLINE void fill(U num0) {
        T num = static_cast<T>(num0);
        FASTOR_INDEX i=0;
        for (; i<ROUND_DOWN(Size,Stride); i+=Stride) {
            SIMDVector<T,DEFAULT_ABI> _vec = num;
            _vec.store(_data+i);
        }
        for (; i<Size; ++i) _data[i] = num0;
    }

    template<typename U=T>
    FASTOR_INLINE void iota(U num0=static_cast<U>(0)) {
        T num = static_cast<T>(num0);
        SIMDVector<T,DEFAULT_ABI> _vec;
        FASTOR_INDEX i=0;
        for (; i<ROUND_DOWN(Size,Stride); i+=Stride) {
            _vec.set_sequential(i+num);
            _vec.store(_data+i);
        }
        for (; i<Size; ++i) _data[i] = i+num0;
    }

    template<typename U=T>
    FASTOR_INLINE void arange(U num0=0) {iota(num0);}

    FASTOR_INLINE void zeros() {
        SIMDVector<T,DEFAULT_ABI> _zeros;
        FASTOR_INDEX i=0;
        for (; i<ROUND_DOWN(Size,Stride); i+=Stride) {
            _zeros.store(_data+i);
        }
        for (; i<Size; ++i) _data[i] = 0;
    }

    FASTOR_INLINE void ones() {
        this->fill(static_cast<T>(1));
    }

    FASTOR_INLINE void eye() {
        static_assert(sizeof...(Rest)>=2, "CANNOT BUILD AN IDENTITY TENSOR");
        static_assert(no_of_unique<Rest...>::value==1, "TENSOR MUST BE UNIFORM");
        zeros();

        constexpr int ndim = sizeof...(Rest);
        constexpr std::array<int,ndim> maxes_a = {Rest...};
        std::array<int,ndim> products;
        std::fill(products.begin(),products.end(),0);

        for (int j=ndim-1; j>0; --j) {
            int num = maxes_a[ndim-1];
            for (int k=0; k<j-1; ++k) {
                num *= maxes_a[ndim-1-k-1];
            }
            products[j] = num;
        }
        std::reverse(products.begin(),products.end());

        for (FASTOR_INDEX i=0; i<dimension(0); ++i) {
            int index_a = i;
            for(int it = 0; it< ndim; it++) {
                index_a += products[it]*i;
            }
            _data[index_a] = static_cast<T>(1);
        }
    }

    FASTOR_INLINE void random() {
        //! Populate tensor with random FP numbers
        for (FASTOR_INDEX i=0; i<this->Size; ++i) {
            _data[i] = (T)rand()/RAND_MAX;
        }
    }

    FASTOR_INLINE void randint() {
        //! Populate tensor with random integer numbers
        for (FASTOR_INDEX i=0; i<this->Size; ++i) {
            _data[i] = (T)rand();
        }
    }

    FASTOR_INLINE T sum() const {

        if ((Size==0) || (Size==1)) return _data[0];

        using V = SIMDVector<T,DEFAULT_ABI>;
        constexpr int unroll_upto = V::unroll_size(Size);
        constexpr int stride = V::Size;
        int i = 0;

        V vec =static_cast<T>(0);
        for (; i< unroll_upto; i+=stride) {
            vec += V(_data+i);
        }
        T scalar = static_cast<T>(0);
        for (; i< Size; ++i) {
            scalar += _data[i];
        }
        return vec.sum() + scalar;
    }

    FASTOR_INLINE T product() const {

        if ((Size==0) || (Size==1)) return _data[0];

        using V = SIMDVector<T,DEFAULT_ABI>;
        constexpr int unroll_upto = V::unroll_size(Size);
        constexpr int stride = V::Size;
        int i = 0;

        V vec =static_cast<T>(1);
        for (; i< unroll_upto; i+=stride) {
            vec *= V(_data+i);
        }
        T scalar = static_cast<T>(0);
        for (; i< Size; ++i) {
            scalar *= _data[i];
        }
        return vec.product()*scalar;
    }

    FASTOR_INLINE void reverse() {
        // in-place reverse
        if ((Size==0) || (Size==1)) return;
        // std::reverse(_data,_data+Size); return;

        // This requires copying the data to avoid aliasing
        // Despite that this method seems to be faster than
        // std::reverse for big _data both on GCC and Clang
        T FASTOR_ALIGN tmp[Size];
        std::copy(_data,_data+Size,tmp);

        // Although SSE register reversing is faster
        // The AVX one outperforms it
        using V = SIMDVector<T,DEFAULT_ABI>;
        // using V = SIMDVector<T,SSE>; 
        constexpr int unroll_upto = V::unroll_size(Size);
        constexpr int stride = V::Size;
        int i = 0;

        V vec;
        for (; i< unroll_upto; i+=stride) {
            vec.load(&tmp[Size - i - stride]);
            vec.reverse().store(_data+i);
        }
        for (; i< Size; ++i) {
            _data[i] = tmp[Size-i-1];
        }
    }


    // Converters
    FASTOR_INLINE T toscalar() const {
        //! Returns a scalar
        static_assert(Size==1,"ONLY TENSORS OF SIZE 1 CAN BE CONVERTED TO SCALAR");
        return _data[0];
    }

    FASTOR_INLINE std::array<T,Size> toarray() const {
        //! Returns std::array
        std::array<T,Size> out;
        std::copy(_data,_data+Size,out.begin());
        return out;
    }

    FASTOR_INLINE std::vector<T> tovector() const {
        //! Returns std::vector
        std::vector<T> out(Size);
        std::copy(_data,_data+Size,out.begin());
        return out;
    }


    // Boolean functions
    constexpr FASTOR_INLINE bool is_uniform() const {
        //! A tensor is uniform if it spans equally in all dimensions,
        //! i.e. generalisation of square matrix to n dimension
        return no_of_unique<Rest...>::value==1 ? true : false;
    }

    template<typename U, size_t ... RestOther>
    FASTOR_INLINE bool is_equal(const Tensor<U,RestOther...> &other) const {
        //! Two tensors are equal if they have the same type, rank, size and elements
        if(!std::is_same<T,U>::value) return false;
        if(sizeof...(Rest)!=sizeof...(RestOther)) return false;
        if(prod<Rest...>::value!=prod<RestOther...>::value) return false;
        else {
            bool out = true;
            T *other_data = other.data();
            for (size_t i=0; i<Size; ++i) {
                if (std::fabs(_data[i]-other_data[i])>PRECI_TOL) {
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

    FASTOR_INLINE bool does_belong_to_so3() const {
        //! A second order tensor belongs to special orthogonal 3D group if
        //! it is orthogonal and its determinant is +1
        if (is_orthogonal()) {
            // Check if we are in 3D space
            if (prod<Rest...>::value!=9) {
                return false;
            }
            T out = _det<T,Rest...>(_data);
            if (std::fabs(out-1)>PRECI_TOL) {
                return false;
            }
            return true;
        }
        else {
            return false;
        }
    }

    FASTOR_INLINE bool does_belong_to_sl3() const {
        //! A second order tensor belongs to special linear 3D group if
        //! its determinant is +1
        T out = _det<T,Rest...>(_data);
        if (std::fabs(out-1.)>PRECI_TOL) {
            return false;
        }
        return true;
    }

    FASTOR_INLINE bool is_symmetric() {
        if (is_uniform()) {
            bool bb = true;
            size_t M = dimension(0);
            size_t N = dimension(1);
            for (size_t i=0; i<M; ++i)
                for (size_t j=0; j<N; ++j)
                    if (std::fabs(_data[i*N+j] - _data[j*N+i])<PRECI_TOL) {
                        bb = false;
                    }
            return bb;
        }
        else {
            return false;
        }
    }
    template<typename ... Args, typename std::enable_if<sizeof...(Args)==2,bool>::type=0>
    FASTOR_INLINE bool is_symmetric(Args ...) {
        return true;
    }

    FASTOR_INLINE bool is_deviatoric() {
        if (std::fabs(trace(*this))<PRECI_TOL)
            return true;
        else
            return false;
    }
    //----------------------------------------------------------------------------------------------------------//

protected:
    template<typename Derived, size_t DIMS>
    FASTOR_INLINE void verify_dimensions(const AbstractTensor<Derived,DIMS>& src_) {
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

