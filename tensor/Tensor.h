#ifndef TENSOR_H
#define TENSOR_H

#include "commons/commons.h"
#include "backend/backend.h"
#include "simd_vector/SIMDVector.h"
#include "AbstractTensor.h"
#include "Range.h"
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


    // Classic constructors
    //----------------------------------------------------------------------------------------------------------//
    FASTOR_INLINE Tensor(){}

    FASTOR_INLINE Tensor(T i) {
        SIMDVector<T> reg = i;
        for (FASTOR_INDEX i=0; i<Size; i+=Stride) {
            reg.store(_data+i);
        }
    }

    FASTOR_INLINE Tensor(const Tensor<T,Rest...> &other) {
        FASTOR_ASSERT(other.Dimension==Dimension, "TENSOR RANK MISMATCH");
        FASTOR_ASSERT(other.size()==Size, "TENSOR SIZE MISMATCH");
        std::copy(other.data(),other.data()+other.size(),_data);
    }
    FASTOR_INLINE Tensor(Tensor<T,Rest...> &&other) {
        FASTOR_ASSERT(other.Dimension==Dimension, "TENSOR RANK MISMATCH");
        FASTOR_ASSERT(other.size()==Size, "TENSOR SIZE MISMATCH");
        std::copy(other.data(),other.data()+other.size(),_data);
    }
    FASTOR_INLINE Tensor<T,Rest...> operator=(const Tensor<T,Rest...> &other) {
        FASTOR_ASSERT(other.Dimension==Dimension, "TENSOR RANK MISMATCH");
        FASTOR_ASSERT(other.size()==Size, "TENSOR SIZE MISMATCH");
        std::copy(other.data(),other.data()+other.Size,_data);
        return *this;
    }
    //----------------------------------------------------------------------------------------------------------//

    // CRTP constructors
    //----------------------------------------------------------------------------------------------------------//
    template<typename Derived, size_t DIMS>
    FASTOR_INLINE Tensor(const AbstractTensor<Derived,DIMS>& src_) {
        const Derived &src = src_.self();
        static_assert(DIMS==Dimension, "TENSOR RANK MISMATCH");
        FASTOR_ASSERT(src.size()==Size, "TENSOR SIZE MISMATCH");
#ifdef SHAPE_CHECK
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        for (FASTOR_INDEX i = 0; i < Size; i+=Stride) {
            src.eval(static_cast<T>(i)).store(_data+i);
        }
    }

    template<typename Derived, size_t DIMS>
    FASTOR_INLINE Tensor<T,Rest...>& operator=(const AbstractTensor<Derived,DIMS>& src_) {
        const Derived &src = src_.self();
        static_assert(DIMS==Dimension, "TENSOR RANK MISMATCH");
        FASTOR_ASSERT(src.size()==Size, "TENSOR SIZE MISMATCH");
#ifdef SHAPE_CHECK
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        for (FASTOR_INDEX i = 0; i < Size; i+=Stride) {
            src.eval(static_cast<T>(i)).store(_data+i);
        }
        return *this;
    }

    template<typename Derived, size_t DIMS>
    FASTOR_INLINE Tensor<T,Rest...>& operator=(AbstractTensor<Derived,DIMS>&& src_) {
        const Derived &src = src_.self();
        static_assert(DIMS==Dimension, "TENSOR RANK MISMATCH");
        FASTOR_ASSERT(src.size()==Size, "TENSOR SIZE MISMATCH");
#ifdef SHAPE_CHECK
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif
        for (FASTOR_INDEX i = 0; i < Size; i+=Stride) {
            src.eval(static_cast<T>(i)).store(_data+i);
        }
        return *this;
    }
    //----------------------------------------------------------------------------------------------------------//

    // Expression templates evaluators
    //----------------------------------------------------------------------------------------------------------//
    FASTOR_INLINE SIMDVector<T> eval(T i) const {
        SIMDVector<T> out;
        out.load(_data+static_cast<FASTOR_INDEX>(i));
        return out;
    }


    FASTOR_INLINE T eval(T i, T j) const {
        return _data[static_cast<FASTOR_INDEX>(i)*get_value<2,Rest...>::value+static_cast<FASTOR_INDEX>(j)];
    }

    template<typename Derived, size_t DIMS>
    static FASTOR_INLINE Tensor<T,Rest...> evaluate(const AbstractTensor<Derived,DIMS>& src_) {
        Tensor<T,Rest...> out;
        const Derived &src = src_.self();
        static_assert(DIMS==Tensor<T,Rest...>::Dimension, "TENSOR RANK MISMATCH");
        FASTOR_ASSERT(src.size()==Tensor<T,Rest...>::Size, "TENSOR SIZE MISMATCH");
        for (FASTOR_INDEX i=0; i<Tensor<T,Rest...>::Dimension; ++i) {
            FASTOR_ASSERT(src.dimension(i)==out.dimension(i), "TENSOR SHAPE MISMATCH");
        }

        for (FASTOR_INDEX i = 0; i <Tensor<T,Rest...>::Size; i+=Tensor<T,Rest...>::Stride) {
            src.eval(static_cast<T>(i)).store(out.data()+i);
        }
        return out;
    }
    //----------------------------------------------------------------------------------------------------------//


    // Smart binders
    //----------------------------------------------------------------------------------------------------------//
    template<size_t I, size_t J, size_t K>
    FASTOR_INLINE Tensor(const BinaryMatMulOp<Tensor<T,I,J>,Tensor<T,J,K>>& src_) {
        constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
        for (FASTOR_INDEX i = 0; i < dimension(0); i++) {
            for (FASTOR_INDEX j = 0; j < dimension(1); j++) {
                _data[i*N+j] = src_.eval(static_cast<T>(i),static_cast<T>(j));
            }
        }
    }
    template<size_t I, size_t J, size_t K>
    FASTOR_INLINE Tensor(BinaryMatMulOp<Tensor<T,I,J>,Tensor<T,J,K>> &&src_) {
        constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
        for (FASTOR_INDEX i = 0; i < dimension(0); i++) {
            for (FASTOR_INDEX j = 0; j < dimension(1); j++) {
                _data[i*N+j] = src_.eval(static_cast<T>(i),static_cast<T>(j));
            }
        }
    }

    template<size_t I, size_t J, size_t K>
    FASTOR_INLINE Tensor(const BinaryMatMulOp<Tensor<T,I,J>,BinaryMatMulOp<Tensor<T,J,K>,Tensor<T,K>>>& src_) {
        T FASTOR_ALIGN tmp[Size];
        _matmul<T,J,K,K>(src_.rhs.lhs.data(),src_.rhs.rhs.data(),tmp);
        _matmul<T,J,K,K>(src_.lhs.lhs.data(),tmp,_data);
    }

    template<class Derived0, class Derived1, class Derived2>
    FASTOR_INLINE Tensor(const BinaryMatMulOp<BinaryMatMulOp<AbstractTensor<Derived0,Derived0::Dimension>,
                         AbstractTensor<Derived1,Derived1::Dimension>>,
                         AbstractTensor<Derived2,Derived2::Dimension>>& src_) {
        // The generic version of reducing matrix-matrix to matrix-vector multiplications, for instance A*B*(a+b)
        T FASTOR_ALIGN tmp[Size];
        unused(src_);
        FASTOR_ASSERT(false,"NOT IMPLEMENTED YET");
    }

    template<size_t I,size_t J>
    FASTOR_INLINE Tensor(const UnaryTransposeOp<Tensor<T,I,J>>& src_) {
        constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
        constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
        static_assert((J==M && I==N), "DIMENSIONS OF OUTPUT TENSOR DO NOT MATCH WITH ITS TRANSPOSE");
        for (FASTOR_INDEX i = 0; i < dimension(0); i++) {
            for (FASTOR_INDEX j = 0; j < dimension(1); j++) {
                _data[i*N+j] = src_.eval(static_cast<T>(i),static_cast<T>(j));
            }
        }
    }

    template<size_t I, size_t J>
    FASTOR_INLINE Tensor(const UnaryTraceOp<Tensor<T,I,J>>& src_) {
        static_assert(sizeof...(Rest)==0, "TRACE OPERATOR WORKS ON SECOND ORDER TENSORS AND RETURNS A SCALAR");
        _data[0] = src_.eval(static_cast<T>(0));
    }

    template<size_t I>
    FASTOR_INLINE Tensor(const UnaryTraceOp<UnaryTransposeOp<Tensor<T,I,I>>> &a) {
        static_assert(sizeof...(Rest)==0, "TRACE OPERATOR WORKS ON SECOND ORDER TENSORS AND RETURNS A SCALAR");
        _data[0] = _trace<T,I,I>(a.expr.expr.data());
    }

    template<size_t I>
    FASTOR_INLINE Tensor(const UnaryDetOp<UnaryTransposeOp<Tensor<T,I,I>>> &a) {
        static_assert(sizeof...(Rest)==0, "DETERMINANT OPERATOR WORKS ON SECOND ORDER TENSORS AND RETURNS A SCALAR");
        _data[0] = _det<T,I,I>(a.expr.expr.data());
    }

    template<size_t I, size_t J, size_t K>
    FASTOR_INLINE Tensor(const UnaryTraceOp<BinaryMatMulOp<UnaryTransposeOp<Tensor<T,I,J>>,Tensor<T,J,K>>> &a) {
        static_assert(I==K, "SECOND ORDER TENSOR MUST BE SQUARE");
        static_assert(sizeof...(Rest)==0, "TRACE OPERATOR WORKS ON SECOND ORDER TENSORS AND RETURNS A SCALAR");
        if (I!=J) { _data[0] = _doublecontract_transpose<T,I,J>(a.expr.lhs.expr.data(),a.expr.rhs.data()); }
        else { _data[0] = _doublecontract<T,I,K>(a.expr.lhs.expr.data(),a.expr.rhs.data());}
    }

    template<size_t I, size_t J, size_t K>
    FASTOR_INLINE Tensor(const UnaryTraceOp<BinaryMatMulOp<Tensor<T,I,J>,UnaryTransposeOp<Tensor<T,J,K>>>> &a) {
        static_assert(I==K, "SECOND ORDER TENSOR MUST BE SQUARE");
        static_assert(sizeof...(Rest)==0, "TRACE OPERATOR WORKS ON SECOND ORDER TENSORS AND RETURNS A SCALAR");
        if (I!=J) { _data[0] = _doublecontract_transpose<T,I,J>(a.expr.lhs.data(),a.expr.rhs.expr.data()); }
        else { _data[0] = _doublecontract<T,I,K>(a.expr.lhs.data(),a.expr.rhs.expr.data());}
    }


    template<size_t I>
    FASTOR_INLINE Tensor(const UnaryDetOp<Tensor<T,I,I>> &src_) {
        // This is essentially immediate evaluation as UnaryDetOp does not bind to other expressions
        static_assert(sizeof...(Rest)==0, "DETERMINANT OPERATOR WORKS ON SECOND ORDER TENSORS AND RETURNS A SCALAR");
        _data[0] = src_.eval(static_cast<T>(0)); // Passing a zero is just a hack to make the type known to eval
    }
    template<size_t I>
    FASTOR_INLINE Tensor(UnaryDetOp<Tensor<T,I,I>> &&src_) {
             static_assert(sizeof...(Rest)==0, "DETERMINANT OPERATOR WORKS ON SECOND ORDER TENSORS AND RETURNS A SCALAR");
        _data[0] = src_.eval(static_cast<T>(0));
    }

    template<size_t I>
    FASTOR_INLINE Tensor(const UnaryAdjOp<Tensor<T,I,I>> &src_) {
        static_assert(I==get_value<1,Rest...>::value && I==get_value<2,Rest...>::value, "DIMENSION MISMATCH");
        src_.eval(_data);
    }
    template<size_t I>
    FASTOR_INLINE Tensor(UnaryAdjOp<Tensor<T,I,I>> &&src_) {
        static_assert(I==get_value<1,Rest...>::value && I==get_value<2,Rest...>::value, "DIMENSION MISMATCH");
        src_.eval(_data);
    }

    template<size_t I>
    FASTOR_INLINE Tensor(const UnaryCofOp<Tensor<T,I,I>> &src_) {
        static_assert(I==get_value<1,Rest...>::value && I==get_value<2,Rest...>::value, "DIMENSION MISMATCH");
        src_.eval(_data);
    }
    template<size_t I>
    FASTOR_INLINE Tensor(UnaryCofOp<Tensor<T,I,I>> &&src_) {
        static_assert(I==get_value<1,Rest...>::value && I==get_value<2,Rest...>::value, "DIMENSION MISMATCH");
        src_.eval(_data);
    }

    template<size_t I>
    FASTOR_INLINE Tensor(const UnaryInvOp<Tensor<T,I,I>> &src_) {
        static_assert(I==get_value<1,Rest...>::value && I==get_value<2,Rest...>::value, "DIMENSION MISMATCH");
        T det_data = src_.eval(_data);
        FASTOR_WARN(std::abs(det_data)>PRECI_TOL, "WARNING: TENSOR IS NEARLY SINGULAR");
        *this = *this/det_data;
    }
    template<size_t I>
    FASTOR_INLINE Tensor(UnaryInvOp<Tensor<T,I,I>> &&src_) {
        static_assert(I==get_value<1,Rest...>::value && I==get_value<2,Rest...>::value, "DIMENSION MISMATCH");
        T det_data = src_.eval(_data);
        FASTOR_WARN(std::abs(det_data)>PRECI_TOL, "WARNING: TENSOR IS NEARLY SINGULAR");
        *this = *this/det_data;
    }

    template<size_t I>
    FASTOR_INLINE Tensor(const UnaryDetOp<UnaryInvOp<Tensor<T,I,I>>> &src_) {
        static_assert(sizeof...(Rest)==0, "DETERMINANT OPERATOR WORKS ON SECOND ORDER TENSORS AND RETURNS A SCALAR");
        _data[0] = static_cast<T>(1)/_det<T,I,I>(src_.expr.expr.data());
    }

    template<size_t I>
    FASTOR_INLINE Tensor(const BinaryMatMulOp<UnaryInvOp<Tensor<T,I,I>>,Tensor<T,I,I>> &src_) {
        if (src_.lhs.expr==src_.rhs) {
            this->eye();
        }
        else {
            using V = SIMDVector<T>; V vec;
            Tensor<T,I,I> inverser;
            T *inv_data = inverser.data();
            _adjoint<T,I,I>(src_.lhs.expr.data(),inv_data);
            T det = _det<T,I,I>(src_.lhs.expr.data());
            for (FASTOR_INDEX i=0; i<I*I; ++i) {
                vec.load(&inv_data[i]);
                vec /= det;
                vec.store(&inv_data[i]);
            }
            _matmul<T,I,I,I>(inv_data,src_.rhs.data(),_data);
        }
    }

    template<size_t I>
    FASTOR_INLINE Tensor(const BinaryMatMulOp<Tensor<T,I,I>,UnaryInvOp<Tensor<T,I,I>>> &src_) {
        if (src_.lhs==src_.rhs.expr) {
            this->eye();
        }
        else {
            using V = SIMDVector<T>; V vec;
            Tensor<T,I,I> inverser;
            T *inv_data = inverser.data();
            _adjoint<T,I,I>(src_.rhs.expr.data(),inv_data);
            T det = _det<T,I,I>(src_.rhs.expr.data());
            for (FASTOR_INDEX i=0; i<I*I; i+=V::Size) {
                vec.load(&inv_data[i]);
                vec /= det;
                vec.store(&inv_data[i]);
            }
            _matmul<T,I,I,I>(src_.lhs.data(),inv_data,_data);
        }
    }

    template<size_t I>
    FASTOR_INLINE Tensor(const UnaryTransposeOp<UnaryAdjOp<Tensor<T,I,I>>> &src_) {
        static_assert(I==get_value<1,Rest...>::value && I==get_value<2,Rest...>::value, "DIMENSION MISMATCH");
        _cofactor<T,I,I>(src_.expr.expr.data(),_data);
    }

    template<size_t I>
    FASTOR_INLINE Tensor(const UnaryTransposeOp<UnaryCofOp<Tensor<T,I,I>>> &src_) {
        static_assert(I==get_value<1,Rest...>::value && I==get_value<2,Rest...>::value, "DIMENSION MISMATCH");
        _adjoint<T,I,I>(src_.expr.expr.data(),_data);
    }

    //----------------------------------------------------------------------------------------------------------//
    template<size_t ndim, size_t nodeperelem>
    FASTOR_INLINE Tensor(const BinaryMatMulOp<BinaryMatMulOp<UnaryInvOp<BinaryMatMulOp<Tensor<T, ndim, nodeperelem>,
                                   Tensor<T, nodeperelem, ndim> > >,
                                   Tensor<T, ndim, nodeperelem> >, Tensor<T, nodeperelem, ndim> > &src) {
        //! Domain-aware expression for chaining multiple operators [used for calculating the
        //! deformation gradient F, for instance]

        static_assert(Size==ndim*ndim,"RESULTING TENSOR MUST BE SQUARE");
        this->zeros();

#ifndef IDEAL_IMPL

        const T FASTOR_ALIGN *x = src.rhs.data();
#ifdef FASTOR_INTEL
        T FASTOR_ALIGN *X = src.lhs.lhs.expr.rhs.data();
#else
        T FASTOR_ALIGN *X = src.rhs.data();
#endif
        const T FASTOR_ALIGN *Jm = src.lhs.rhs.data();

        T FASTOR_ALIGN PG[ndim*ndim] = {static_cast<T>(0)};
        _matmul<T,ndim,nodeperelem,ndim>(Jm,X,PG);
        T FASTOR_ALIGN invPG[ndim*ndim];
        _inverse<T,ndim>(PG,invPG);
        T FASTOR_ALIGN MG[ndim*nodeperelem] = {static_cast<T>(0)};
        _matmul<T,ndim,ndim,nodeperelem>(invPG,Jm,MG);
#ifdef FASTOR_GCC
        _matmul<T,ndim,nodeperelem,ndim>(MG,x,_data);
#endif
#ifdef FASTOR_INTEL
        unused(_data);
#endif
#ifdef FASTOR_CLANG
        T FASTOR_ALIGN xx[ndim*ndim];
        _matmul<T,ndim,nodeperelem,ndim>(MG,x,xx);
        std::copy(xx,xx+ndim*ndim,_data);
#endif

#else
        const T *x = src.rhs.data();
        const T *Jm = src.lhs.rhs.data();
        const T *X = src.lhs.lhs.expr.rhs.data();

        T FASTOR_ALIGN PG[ndim*ndim] = {static_cast<T>(0.)};
        _matmul<T,ndim,nodeperelem,ndim>(Jm,X,PG);
        T FASTOR_ALIGN invPG[ndim*ndim];
        _inverse<T,ndim>(PG,invPG);
        T FASTOR_ALIGN MG[ndim*nodeperelem] = {static_cast<T>(0.)};
        _matmul<T,ndim,ndim,nodeperelem>(invPG,Jm,MG);
        _matmul<T,ndim,nodeperelem,ndim>(MG,x,_data);

#endif
    }

    //----------------------------------------------------------------------------------------------------------//

    // Raw pointer providers
    //----------------------------------------------------------------------------------------------------------//
    FASTOR_INLINE T* data() const {
        return const_cast<T*>(this->_data);
    }
    FASTOR_INLINE T* data() {
        return this->_data;
    }
    //----------------------------------------------------------------------------------------------------------//

    // Scalar indexing
    //----------------------------------------------------------------------------------------------------------//
    template<typename... Args, typename std::enable_if<sizeof...(Args)==1
                            && sizeof...(Args)==Dimension && is_arithmetic_pack<Args...>::value,bool>::type =0>
    FASTOR_INLINE T& operator()(Args ... args) {
        constexpr FASTOR_INDEX indices = sizeof...(Args);
        static_assert(indices==Dimension, "INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
        constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
        const FASTOR_INDEX i = get_index<0>(args...);
#ifdef BOUNDSCHECK
        assert( ( (i>=0 && i<M)) && "INDEX OUT OF BOUNDS");
#endif
        return _data[i];
    }

    template<typename... Args, typename std::enable_if<sizeof...(Args)==1
                            && sizeof...(Args)==Dimension && is_arithmetic_pack<Args...>::value,bool>::type =0>
    FASTOR_INLINE const T& operator()(Args ... args) const {
        constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
        const FASTOR_INDEX i = get_index<0>(args...);
#ifdef BOUNDSCHECK
        assert( ( (i>=0 && i<M)) && "INDEX OUT OF BOUNDS");
#endif
        return _data[i];
    }

    template<typename... Args, typename std::enable_if<sizeof...(Args)==2
                            && sizeof...(Args)==Dimension && is_arithmetic_pack<Args...>::value,bool>::type =0>
    FASTOR_INLINE T& operator()(Args ... args) {
        constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
        constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
        const FASTOR_INDEX i = get_index<0>(args...);
        const FASTOR_INDEX j = get_index<1>(args...);
#ifdef BOUNDSCHECK
        assert( ( (i>=0 && i<M) && (j>=0 && j<N)) && "INDEX OUT OF BOUNDS");
#endif
        return _data[i*N+j];
    }

    template<typename... Args, typename std::enable_if<sizeof...(Args)==2
                            && sizeof...(Args)==Dimension && is_arithmetic_pack<Args...>::value,bool>::type =0>
    FASTOR_INLINE const T& operator()(Args ... args) const {
        constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
        constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
        const FASTOR_INDEX i = get_index<0>(args...);
        const FASTOR_INDEX j = get_index<1>(args...);
#ifdef BOUNDSCHECK
        assert( ( (i>=0 && i<M) && (j>=0 && j<N)) && "INDEX OUT OF BOUNDS");
#endif
        return _data[i*N+j];
    }

    template<typename... Args, typename std::enable_if<sizeof...(Args)==3
                            && sizeof...(Args)==Dimension && is_arithmetic_pack<Args...>::value,bool>::type =0>
    FASTOR_INLINE T& operator()(Args ... args) {
        constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
        constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
        constexpr FASTOR_INDEX P = get_value<3,Rest...>::value;
        const FASTOR_INDEX i = get_index<0>(args...);
        const FASTOR_INDEX j = get_index<1>(args...);
        const FASTOR_INDEX k = get_index<2>(args...);
#ifdef BOUNDSCHECK
        assert( ( (i>=0 && i<M) && (j>=0 && j<N) && (k>=0 && k<P)) && "INDEX OUT OF BOUNDS");
#endif
        return _data[i*N*P+j*P+k];
    }

    template<typename... Args, typename std::enable_if<sizeof...(Args)==3
                            && sizeof...(Args)==Dimension && is_arithmetic_pack<Args...>::value,bool>::type =0>
    FASTOR_INLINE const T&  operator()(Args ... args) const {
        constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
        constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
        constexpr FASTOR_INDEX P = get_value<3,Rest...>::value;
        const FASTOR_INDEX i = get_index<0>(args...);
        const FASTOR_INDEX j = get_index<1>(args...);
        const FASTOR_INDEX k = get_index<2>(args...);
#ifdef BOUNDSCHECK
        assert( ( (i>=0 && i<M) && (j>=0 && j<N) && (k>=0 && k<P)) && "INDEX OUT OF BOUNDS");
#endif
        return _data[i*N*P+j*P+k];
    }

    template<typename... Args, typename std::enable_if<sizeof...(Args)==4
                            && sizeof...(Args)==Dimension && is_arithmetic_pack<Args...>::value,bool>::type =0>
    FASTOR_INLINE T& operator()(Args ... args) {
        constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
        constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
        constexpr FASTOR_INDEX P = get_value<3,Rest...>::value;
        constexpr FASTOR_INDEX Q = get_value<4,Rest...>::value;
        const FASTOR_INDEX i = get_index<0>(args...);
        const FASTOR_INDEX j = get_index<1>(args...);
        const FASTOR_INDEX l = get_index<3>(args...);
        const FASTOR_INDEX k = get_index<2>(args...);
#ifdef BOUNDSCHECK
        assert( ( (i>=0 && i<M) && (j>=0 && j<N)
                  && (k>=0 && k<P) && (l>=0 && l<Q)) && "INDEX OUT OF BOUNDS");
#endif
        return _data[i*N*P*Q+j*P*Q+k*Q+l];
    }

    template<typename... Args, typename std::enable_if<sizeof...(Args)==4
                            && sizeof...(Args)==Dimension && is_arithmetic_pack<Args...>::value,bool>::type =0>
    FASTOR_INLINE const T& operator()(Args ... args) const {
        constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
        constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
        constexpr FASTOR_INDEX P = get_value<3,Rest...>::value;
        constexpr FASTOR_INDEX Q = get_value<4,Rest...>::value;
        const FASTOR_INDEX i = get_index<0>(args...);
        const FASTOR_INDEX j = get_index<1>(args...);
        const FASTOR_INDEX l = get_index<3>(args...);
        const FASTOR_INDEX k = get_index<2>(args...);
#ifdef BOUNDSCHECK
        assert( ( (i>=0 && i<M) && (j>=0 && j<N)
                  && (k>=0 && k<P) && (l>=0 && l<Q)) && "INDEX OUT OF BOUNDS");
#endif
        return _data[i*N*P*Q+j*P*Q+k*Q+l];
    }

    template<typename... Args, typename std::enable_if<sizeof...(Args)>=5
                            && sizeof...(Args)==Dimension && is_arithmetic_pack<Args...>::value,bool>::type =0>
    FASTOR_INLINE T& operator()(Args ... args) {
        const int largs[sizeof...(Args)] = {args...};
        constexpr int DimensionHolder[Dimension] = {Rest...};
#ifdef BOUNDSCHECK
        for (int i=0; i<Dimension; ++i) {
            assert( (largs[i]>=0 && largs[i]<DimensionHolder[i]) && "INDEX OUT OF BOUNDS");
        }
#endif
        std::array<int,Dimension> products;
        for (int i=Dimension-1; i>0; --i) {
            int num = DimensionHolder[Dimension-1];
            for (int j=0; j<i-1; ++j) {
                num *= DimensionHolder[Dimension-1-j-1];
            }
            products[i] = num;
        }

        int index = largs[Dimension-1];
        for (int i=Dimension-1; i>0; --i) {
            index += products[i]*largs[Dimension-i-1];
        }
        return _data[index];
    }

    template<typename... Args, typename std::enable_if<sizeof...(Args)>=5
                            && sizeof...(Args)==Dimension && is_arithmetic_pack<Args...>::value,bool>::type =0>
    FASTOR_INLINE const T& operator()(Args ... args) const {
        const int largs[sizeof...(Args)] = {args...};
        constexpr int DimensionHolder[Dimension] = {Rest...};
#ifdef BOUNDSCHECK
        for (int i=0; i<Dimension; ++i) {
            assert( (largs[i]>=0 && largs[i]<DimensionHolder[i]) && "INDEX OUT OF BOUNDS");
        }
#endif
        std::array<int,Dimension> products;
        for (int i=Dimension-1; i>0; --i) {
            int num = DimensionHolder[Dimension-1];
            for (int j=0; j<i-1; ++j) {
                num *= DimensionHolder[Dimension-1-j-1];
            }
            products[i] = num;
        }

        int index = largs[Dimension-1];
        for (int i=Dimension-1; i>0; --i) {
            index += products[i]*largs[Dimension-i-1];
        }
        return _data[index];
    }
    //----------------------------------------------------------------------------------------------------------//

    // Block indexing
    //----------------------------------------------------------------------------------------------------------//
    // Calls scalar indexing so they are fully bounds checked.
    template<size_t F, size_t L, size_t S>
    FASTOR_INLINE Tensor<T,range_detector<F,L,S>::value> operator()(const range<F,L,S>& idx) {

        static_assert(1==Dimension, "INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
        unused(idx);
        Tensor<T,range_detector<F,L,S>::value> out;
        FASTOR_INDEX counter = 0;
        for (FASTOR_INDEX i=F; i<L; i+=S) {
            out(counter) = this->operator()(i);
            counter++;
        }
        return out;
    }

    template<size_t F0, size_t L0, size_t S0, size_t F1, size_t L1, size_t S1>
    FASTOR_INLINE Tensor<T,range_detector<F0,L0,S0>::value,range_detector<F1,L1,S1>::value>
            operator()(const range<F0,L0,S0>& idx0, const range<F1,L1,S1>& idx1) {

        static_assert(2==Dimension, "INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
        unused(idx0); unused(idx1);
        Tensor<T,range_detector<F0,L0,S0>::value,range_detector<F1,L1,S1>::value> out;
        FASTOR_INDEX counter_i = 0;
        for (FASTOR_INDEX i=F0; i<L0; i+=S0) {
            FASTOR_INDEX counter_j = 0;
            for (FASTOR_INDEX j=F1; j<L1; j+=S1) {
                out(counter_i,counter_j) = this->operator()(i,j);
                counter_j++;
            }
            counter_i++;
        }
        return out;
    }

    template<size_t F0, size_t L0, size_t S0, size_t F1, size_t L1, size_t S1, size_t F2, size_t L2, size_t S2>
    FASTOR_INLINE Tensor<T,range_detector<F0,L0,S0>::value,range_detector<F1,L1,S1>::value,range_detector<F2,L2,S2>::value>
            operator()(const range<F0,L0,S0>& idx0, const range<F1,L1,S1>& idx1,
                                                        const range<F2,L2,S2>& idx2) const {
        static_assert(3==Dimension, "INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
        unused(idx0); unused(idx1); unused(idx2);
        Tensor<T,range_detector<F0,L0,S0>::value,
                range_detector<F1,L1,S1>::value,
                range_detector<F2,L2,S2>::value> out;
        FASTOR_INDEX counter_i = 0;
        for (FASTOR_INDEX i=F0; i<L0; i+=S0) {
            FASTOR_INDEX counter_j = 0;
            for (FASTOR_INDEX j=F1; j<L1; j+=S1) {
                FASTOR_INDEX counter_k = 0;
                for (FASTOR_INDEX k=F2; k<L2; k+=S2) {
                    out(counter_i,counter_j,counter_k) = this->operator()(i,j,k);
                    counter_k++;
                }
                counter_j++;
            }
            counter_i++;
        }
        return out;
    }

    template<size_t F0, size_t L0, size_t S0,
             size_t F1, size_t L1, size_t S1,
             size_t F2, size_t L2, size_t S2,
             size_t F3, size_t L3, size_t S3>
    FASTOR_INLINE Tensor<T,range_detector<F0,L0,S0>::value,
            range_detector<F1,L1,S1>::value,
            range_detector<F2,L2,S2>::value,
            range_detector<F3,L3,S3>::value>
            operator ()(const range<F0,L0,S0>& idx0, const range<F1,L1,S1>& idx1,
                  const range<F2,L2,S2>& idx2, const range<F3,L3,S3>& idx3) {

        static_assert(4==Dimension, "INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
        unused(idx0); unused(idx1); unused(idx2); unused(idx3);
        Tensor<T,range_detector<F0,L0,S0>::value,
                    range_detector<F1,L1,S1>::value,
                    range_detector<F2,L2,S2>::value,
                    range_detector<F3,L3,S3>::value> out;
        FASTOR_INDEX counter_i = 0;
        for (FASTOR_INDEX i=F0; i<L0; i+=S0) {
            FASTOR_INDEX counter_j = 0;
            for (FASTOR_INDEX j=F1; j<L1; j+=S1) {
                FASTOR_INDEX counter_k = 0;
                for (FASTOR_INDEX k=F2; k<L2; k+=S2) {
                    FASTOR_INDEX counter_l = 0;
                    for (FASTOR_INDEX l=F3; l<L3; l+=S3) {
                        out(counter_i,counter_j,counter_k) = this->operator()(i,j,k,l);
                        counter_l++;
                    }
                    counter_k++;
                }
                counter_j++;
            }
            counter_i++;
        }
        return out;
    }
    //----------------------------------------------------------------------------------------------------------//


    // In-place operators
    //----------------------------------------------------------------------------------------------------------//

    FASTOR_INLINE void operator +=(const Tensor<T,Rest...> &a) {
        using V = SIMDVector<T>;
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
        using V = SIMDVector<T>;
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
        using V = SIMDVector<T>;
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
        using V = SIMDVector<T>;
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
    //----------------------------------------------------------------------------------------------------------//

    // Further member functions
    //----------------------------------------------------------------------------------------------------------//
    static constexpr FASTOR_INLINE FASTOR_INDEX rank() {return Dimension;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX dim) const {
//        constexpr FASTOR_INDEX DimensionHolder[Dimension] = {Rest...}; // c++14
        FASTOR_INDEX DimensionHolder[Dimension] = {Rest...};
        return DimensionHolder[dim];
    }

    template<typename U=T>
    FASTOR_INLINE void fill(U num0) {
        T num = static_cast<T>(num0);
        FASTOR_INDEX i=0;
        for (; i<ROUND_DOWN(Size,Stride); i+=Stride) {
            SIMDVector<T> _vec = num;
            _vec.store(_data+i);
        }
        for (; i<Size; ++i) _data[i] = num0;
    }

    template<typename U=T>
    FASTOR_INLINE void iota(U num0=static_cast<U>(0)) {
        T num = static_cast<T>(num0);
        SIMDVector<T> _vec;
        FASTOR_INDEX i=0;
        for (; i<ROUND_DOWN(Size,Stride); i+=Stride) {
            _vec.set_sequential(i+num);
            _vec.store(_data+i);
        }
        for (; i<Size; ++i) _data[i] = i;
    }

    template<typename U=T>
    FASTOR_INLINE void arange(U num0=static_cast<U>(0)) {
        iota(num0);
    }

    FASTOR_INLINE void zeros() {
        SIMDVector<T> _zeros;
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

        using V = SIMDVector<T>;
        constexpr int unroll_upto = V::unroll_size(Size);
        constexpr int stride = V::Size;
        int i = 0;

        V vec =static_cast<T>(0);
        for (; i< unroll_upto; i+=stride) {
            vec += V(_data+i);
        }
        T scalar = static_cast<T>(0);
        for (int j=i; j< Size; j++) {
            scalar += _data[j];
        }
        return vec.sum() + scalar;
    }

    FASTOR_INLINE T product() const {

        if ((Size==0) || (Size==1)) return _data[0];

        using V = SIMDVector<T>;
        constexpr int unroll_upto = V::unroll_size(Size);
        constexpr int stride = V::Size;
        int i = 0;

        V vec =static_cast<T>(1);
        for (; i< unroll_upto; i+=stride) {
            vec *= V(_data+i);
        }
        T scalar = static_cast<T>(0);
        for (int j=i; j< Size; j++) {
            scalar *= _data[j];
        }
        return vec.product()*scalar;
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

};




// Indexing specific to vectors and matrices
//----------------------------------------------------------------------------------------------------------//
template<typename T, size_t I, size_t J>
FASTOR_INLINE Tensor<T,I> column(const Tensor<T,I,J> &arr, int n) {
    Tensor<T,I> out;
    FASTOR_INDEX counter = 0;
    for (FASTOR_INDEX i=0; i<I; ++i) {
        out(counter) = arr(i,n);
        counter++;
    }
    return out;
}

template<typename T, size_t I, size_t J>
FASTOR_INLINE Tensor<T,J> row(const Tensor<T,I,J> &arr, int n) {
    Tensor<T,J> out;
    FASTOR_INDEX counter = 0;
    for (FASTOR_INDEX i=0; i<J; ++i) {
        out(counter) = arr(n,i);
        counter++;
    }
    return out;
}


//----------------------------------------------------------------------------------------------------------//



}


#endif // TENSOR_H

