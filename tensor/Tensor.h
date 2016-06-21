#ifndef TENSOR_H
#define TENSOR_H

#include "commons/commons.h"
#include "backend/backend.h"
#include "simd_vector/SIMDVector.h"
#include "AbstractTensor.h"

namespace Fastor {

template<typename TLhs, typename TRhs>
struct BinaryMatMulOp;

template<typename Expr>
struct UnaryTransposeOp;

template<typename Expr>
struct UnaryTraceOp;

template<typename Expr>
struct UnaryDetOp;

template<typename Expr>
struct UnaryAdjOp;

template<typename Expr>
struct UnaryCofOp;

template<typename Expr>
struct UnaryInvOp;


template<typename T, size_t ... Rest>
class Tensor: public AbstractTensor<Tensor<T,Rest...>,sizeof...(Rest)> {
private:
    T FASTOR_ALIGN _data[prod<Rest...>::value];
public:
    typedef T scalar_type;
    static constexpr FASTOR_INDEX Dimension = sizeof...(Rest);
    static constexpr FASTOR_INDEX Size = prod<Rest...>::value;
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::Stride;
    static constexpr FASTOR_INDEX Remainder = prod<Rest...>::value % sizeof(T);


    FASTOR_INLINE Tensor(){}

    FASTOR_INLINE Tensor(T i) {
        SIMDVector<T> reg = i;
        for (FASTOR_INDEX i=0; i<Size; i+=Stride) {
            reg.store(_data+i);
        }
    }

//    FASTOR_INLINE T* operator ()() {return data();}

//    template<typename Derived, size_t DIMS>
//    FASTOR_INLINE Tensor(const AbstractTensor<Derived,DIMS>& src_) {
//        const Derived &src = src_.self();
//        for (FASTOR_INDEX i = 0; i < Size; ++i) {
//          _data[i] = src.eval(i);
//        }
//      }

//    template<typename EType, size_t DIMS>
//    FASTOR_INLINE Tensor<T,M,N,Rest...>& operator=(const AbstractTensor<EType,DIMS>& src_) {
//        const EType &src = src_.self();
//        for (FASTOR_INDEX i = 0; i < Size; ++i) {
//          _data[i] = src.eval(i);
//        }
//        return *this;
//    }

    // evaluation function, evaluate this expression at position i
//      FASTOR_INLINE T eval(FASTOR_INDEX i) const {
//        return _data[i];
//      }

    //-----------------------------------------------------------------------------
    //-----------------------------------------------------------------------------
//    template<typename Derived, size_t DIMS>
//    FASTOR_INLINE Tensor(const AbstractTensor<Derived,DIMS>& src_) {
//        const Derived &src = src_.self();
//        for (FASTOR_INDEX i = 0; i < Size; i+=Stride) {
//            store(_data+i,src.eval(static_cast<T>(i)));
////            store(_data+i,src.eval<Derived,T>(static_cast<T>(i)));
//        }
//    }

//    template<typename Derived, size_t DIMS>
//    FASTOR_INLINE Tensor<T,M,N,Rest...>& operator=(const AbstractTensor<Derived,DIMS>& src_) {
//        const Derived &src = src_.self();
//        for (FASTOR_INDEX i = 0; i < Size; i+=Stride) {
//            store(_data+i,src.eval(static_cast<T>(i)));
////            store(_data+i,src.eval<Derived,T>(static_cast<T>(i)));
//        }
//        return *this;
//    }
    //-----------------------------------------------------------------------------
    FASTOR_INLINE Tensor(const Tensor<T,Rest...> &other) {
        FASTOR_ASSERT(other.Dimension==Dimension, "TENSOR RANK MISMATCH");
        FASTOR_ASSERT(other.size()==Size, "TENSOR SIZE MISMATCH");
        std::copy(other.data(),other.data()+other.size(),_data);
    }
    FASTOR_INLINE Tensor(Tensor<T,Rest...> &&other) {
        FASTOR_ASSERT(other.Dimension==Dimension, "TENSOR RANK MISMATCH");
        FASTOR_ASSERT(other.size()==Size, "TENSOR SIZE MISMATCH");
//        _data = other.data();
        std::copy(other.data(),other.data()+other.size(),_data);
    }
    FASTOR_INLINE Tensor<T,Rest...> operator=(const Tensor<T,Rest...> &other) {
        FASTOR_ASSERT(other.Dimension==Dimension, "TENSOR RANK MISMATCH");
        FASTOR_ASSERT(other.size()==Size, "TENSOR SIZE MISMATCH");
        std::copy(other.data(),other.data()+other.Size,_data);
        return *this;
    }

    template<typename Derived, size_t DIMS>
    FASTOR_INLINE Tensor(const AbstractTensor<Derived,DIMS>& src_) {
        const Derived &src = src_.self();
        static_assert(DIMS==Dimension, "TENSOR RANK MISMATCH");
        FASTOR_ASSERT(src.size()==Size, "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }

        for (FASTOR_INDEX i = 0; i < Size; i+=Stride) {
            src.eval(static_cast<T>(i)).store(_data+i);
        }
    }

    template<typename Derived, size_t DIMS>
    FASTOR_INLINE Tensor<T,Rest...>& operator=(const AbstractTensor<Derived,DIMS>& src_) {
        const Derived &src = src_.self();
        static_assert(DIMS==Dimension, "TENSOR RANK MISMATCH");
        FASTOR_ASSERT(src.size()==Size, "TENSOR SIZE MISMATCH");
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }

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
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(src.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }

        for (FASTOR_INDEX i = 0; i < Size; i+=Stride) {
            src.eval(static_cast<T>(i)).store(_data+i);
        }
        return *this;
    }

    FASTOR_INLINE SIMDVector<T> eval(T i) const {
        SIMDVector<T> out;
        out.load(_data+static_cast<FASTOR_INDEX>(i));
        return out;
    }


    FASTOR_INLINE T eval(T i, T j) const {
        return _data[static_cast<FASTOR_INDEX>(i)*get_value<2,Rest...>::value+static_cast<FASTOR_INDEX>(j)];
    }

    template<size_t I, size_t J, size_t K>
    FASTOR_INLINE Tensor(const BinaryMatMulOp<Tensor<T,I,J>,Tensor<T,J,K>>& src_) {
        constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
        for (FASTOR_INDEX i = 0; i < dimension(0); i++) {
            for (FASTOR_INDEX j = 0; j < dimension(1); j++) {
                _data[i*N+j] = src_.eval(static_cast<T>(i),static_cast<T>(j));
            }
        }
    }

//    template<typename DerivedL, typename DerivedR>
//    FASTOR_INLINE Tensor(const BinaryMatMulOp<DerivedL,DerivedR>& src_) {
////        FASTOR_ASSERT(src_.lhs.dimension(1)==src_.rhs.dimension(0), "DIMENSION MISMATCH IN MATMUL");
////        FASTOR_ASSERT(src_.lhs.dimension(0)==dimension(0) && src_.rhs.dimension(1)==dimension(1), "DIMENSION MISMATCH IN MATMUL");
////        print(src_.lhs.dimension(0),src_.lhs.dimension(1),src_.rhs.dimension(0),src_.rhs.dimension(1));
//        constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
//        for (FASTOR_INDEX i = 0; i < dimension(0); i++) {
//            for (FASTOR_INDEX j = 0; j < dimension(1); j++) {
//                _data[i*N+j] = src_.eval(static_cast<T>(i),static_cast<T>(j));
//            }
//        }
//    }

    template<size_t I, size_t J, size_t K>
    FASTOR_INLINE Tensor(const BinaryMatMulOp<Tensor<T,I,J>,BinaryMatMulOp<Tensor<T,J,K>,Tensor<T,K>>>& src_) {
        T tmp[Size];
        _matmul<T,J,K,K>(src_.rhs.lhs.data(),src_.rhs.rhs.data(),tmp);
        _matmul<T,J,K,K>(src_.lhs.lhs.data(),tmp,_data);
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

//    template<typename Derived>
//    FASTOR_INLINE Tensor(const UnaryTransposeOp<Derived>& src_) {
//        constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
//        for (FASTOR_INDEX i = 0; i < dimension(0); i++) {
//            for (FASTOR_INDEX j = 0; j < dimension(1); j++) {
//                _data[i*N+j] = src_.eval(static_cast<T>(i),static_cast<T>(j));
//            }
//        }
//    }

    template<size_t I, size_t J>
    FASTOR_INLINE Tensor(const UnaryTraceOp<Tensor<T,I,J>>& src_) {
        static_assert(sizeof...(Rest)==0, "TRACE OPERATOR WORKS ON SECOND ORDER TENSORS AND RETURNS A SCALAR");
        _data[0] = src_.eval(static_cast<T>(0));
    }
//    template<size_t I, size_t J>
//    FASTOR_INLINE Tensor<T,Rest...> operator=(const UnaryTraceOp<Tensor<T,I,J>>& src_) {
//        static_assert(sizeof...(Rest)==0, "TRACE OPERATOR WORKS ON SECOND ORDER TENSORS AND RETURNS A SCALAR");
//        _data[0] = src_.eval(static_cast<T>(0));
//        return *this;
//    }

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
        if (I!=J) { _data[0] = _doublecontract_transpose<T,I,J>(a.expr.lhs.expr.data(),a.expr.rhs.data()); }
        else { _data[0] = _doublecontract<T,I,K>(a.expr.lhs.expr.data(),a.expr.rhs.data());}
    }


    template<size_t I>
    FASTOR_INLINE Tensor(const UnaryDetOp<Tensor<T,I,I>> &src_) {
        // This is essentially immediate evaluation as UnaryDetOp does not bind to other expressions
        static_assert(sizeof...(Rest)==0, "DETERMINANT OPERATOR WORKS ON SECOND ORDER TENSORS AND RETURNS A SCALAR");
        _data[0] = src_.eval(static_cast<T>(0)); // Passing a zero is just a hack to make the type known to eval
    }

    template<size_t I>
    FASTOR_INLINE Tensor(const UnaryAdjOp<Tensor<T,I,I>> &src_) {
        static_assert(I==get_value<1,Rest...>::value && I==get_value<2,Rest...>::value, "DIMENSION MISMATCH");
        src_.eval(_data);
    }

    template<size_t I>
    FASTOR_INLINE Tensor(const UnaryCofOp<Tensor<T,I,I>> &src_) {
        static_assert(I==get_value<1,Rest...>::value && I==get_value<2,Rest...>::value, "DIMENSION MISMATCH");
        src_.eval(_data);
    }

    template<size_t I>
    FASTOR_INLINE Tensor(const UnaryTransposeOp<UnaryCofOp<Tensor<T,I,I>>> &src_) {
        static_assert(I==get_value<1,Rest...>::value && I==get_value<2,Rest...>::value, "DIMENSION MISMATCH");
        _adjoint<T,I,I>(src_.expr.expr.data(),_data);
    }

    template<size_t I>
    FASTOR_INLINE Tensor(const UnaryInvOp<Tensor<T,I,I>> &src_) {
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
    // compiler typically builds copy assignment operators
//    template<size_t I>
//    FASTOR_INLINE Tensor<T,I,I> operator=(const UnaryTransposeOp<UnaryCofOp<Tensor<T,I,I>>> &src_) {
//        Tensor<T,I,I> out;
//        static_assert(I==get_value<1,Rest...>::value, "DIMENSION MISMATCH");
//        _adjoint<T,I,I>(src_.expr.expr.data(),out.data());
//        return out;
//    }


    template<size_t I>
    FASTOR_INLINE Tensor(const UnaryTransposeOp<UnaryAdjOp<Tensor<T,I,I>>> &src_) {
        static_assert(I==get_value<1,Rest...>::value, "DIMENSION MISMATCH");
        _cofactor<T,I,I>(src_.expr.expr.data(),_data);
    }



//    FASTOR_INLINE T& eval_el(T i) const {
//        return _data[i];
//    }
    ////////////////////////////
//    template<typename U=T, typename std::enable_if<std::is_same<U,float>::value && std::is_same<T,float>::value,bool>::type=0>
//      FASTOR_INLINE __m256 eval(U i) const {
//          return load(_data+static_cast<int>(i));
//    }

//    template<typename U=T, typename std::enable_if<std::is_same<U,double>::value && std::is_same<T,double>::value,bool>::type=0>
//    FASTOR_INLINE __m256d eval(U i) const {
//        return load(_data+static_cast<int>(i));
//    }
    //////////////////////////
    //-----------------------------------------------------------------------------

      // evaluation function, evaluate this expression at position i
  //    template<typename std::enable_if<std::is_same<T,float>::value,bool>::type=0>
  //      FASTOR_INLINE __m256 eval(FASTOR_INDEX i) const {
  //          return load(_data+i);
  //      }

//      FASTOR_INLINE __m256d eval(FASTOR_INDEX i) const {
////          return _mm256_load_pd(_data+i);
//          return load(_data+i);
//      }


    FASTOR_INLINE T* data() const {
        return const_cast<T*>(this->_data);
    }
    FASTOR_INLINE T* data() {
        return this->_data;
    }

    template<typename... Args, typename std::enable_if<sizeof...(Args)==1 && sizeof...(Args)==Dimension,bool>::type =0>
    FASTOR_INLINE T& operator()(Args ... args) {
        constexpr FASTOR_INDEX indices = sizeof...(Args);
        static_assert(indices==Dimension, "INDEXING TENSOR WITH WRONG NUMBER OF ARGUMENTS");
        constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
        const FASTOR_INDEX i = get_index<0>(args...);
#ifdef BOUNDSCHECK
        assert( ( (i>=0 && i<M)) && "INDEX OUT OF BOUNDS");
#endif
        return _data[i];
    }

    template<typename... Args, typename std::enable_if<sizeof...(Args)==1 && sizeof...(Args)==Dimension,bool>::type =0>
    FASTOR_INLINE const T& operator()(Args ... args) const {
        constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
        const FASTOR_INDEX i = get_index<0>(args...);
#ifdef BOUNDSCHECK
        assert( ( (i>=0 && i<M)) && "INDEX OUT OF BOUNDS");
#endif
        return _data[i];
    }

    template<typename... Args, typename std::enable_if<sizeof...(Args)==2 && sizeof...(Args)==Dimension,bool>::type =0>
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

    template<typename... Args, typename std::enable_if<sizeof...(Args)==2 && sizeof...(Args)==Dimension,bool>::type =0>
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

    template<typename... Args, typename std::enable_if<sizeof...(Args)==3 && sizeof...(Args)==Dimension,bool>::type =0>
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

    template<typename... Args, typename std::enable_if<sizeof...(Args)==3 && sizeof...(Args)==Dimension,bool>::type =0>
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

    template<typename... Args, typename std::enable_if<sizeof...(Args)==4 && sizeof...(Args)==Dimension,bool>::type =0>
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

    template<typename... Args, typename std::enable_if<sizeof...(Args)==4 && sizeof...(Args)==Dimension,bool>::type =0>
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

    template<typename... Args, typename std::enable_if<sizeof...(Args)>=5 && sizeof...(Args)==Dimension,bool>::type =0>
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

    template<typename... Args, typename std::enable_if<sizeof...(Args)>=5 && sizeof...(Args)==Dimension,bool>::type =0>
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

    FASTOR_INLINE FASTOR_INDEX rank() {return Dimension;}

    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX dim) const {
//        constexpr FASTOR_INDEX DimensionHolder[Dimension] = {Rest...}; // c++14
        FASTOR_INDEX DimensionHolder[Dimension] = {Rest...};
        return DimensionHolder[dim];
    }

    template<typename U=T>
    FASTOR_INLINE void fill(U num0) {
        T num = static_cast<T>(num0);
        for (FASTOR_INDEX i=0; i<Size; i+=Stride) {
            SIMDVector<T> _vec = num;
            _vec.store(_data+i);
        }
    }

    template<typename U=T>
    FASTOR_INLINE void iota(U num0) {
        T num = static_cast<T>(num0);
        SIMDVector<T> _vec;
        for (FASTOR_INDEX i=0; i<Size; i+=Stride) {
            _vec.set_sequential(i+num);
            _vec.store(_data+i);
        }
    }

    template<typename U=T>
    FASTOR_INLINE void arange(U num0) {
        iota(num0);
    }

    FASTOR_INLINE void zeros() {
        SIMDVector<T> _zeros;
        for (FASTOR_INDEX i=0; i< Size; i+=Stride) {
            _zeros.store(_data+i);
        }
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

        for (int i=0; i<dimension(0); ++i) {
            int index_a = i;
            for(int it = 0; it< ndim; it++) {
                index_a += products[it]*i;
            }
            _data[index_a] = static_cast<T>(1);
        }
    }

    FASTOR_INLINE void random() {
        for (FASTOR_INDEX i=0; i<this->Size; ++i) {
            _data[i] = (T)rand()/RAND_MAX;
        }
    }

    FASTOR_INLINE T sum() const {
        T summ = static_cast<T>(0);
        for (FASTOR_INDEX i=0; i< Size; i++) {
            summ += _data[i];
        }
        return summ;
    }

    // Special function
    constexpr FASTOR_INLINE bool is_uniform() const {
        //! A tensor is uniform if it spans equally in all dimensions,
        //! i.e. generalisation square matrix to n dimension
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

    FASTOR_INLINE bool is_orthogonal() const {
        // A second order tensor A is orthogonal if A*A'=I
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
    FASTOR_INLINE bool is_symmetric(Args ... args) {
        return true;
    }

    FASTOR_INLINE bool is_deviatoric() {
        if (std::fabs(trace(*this))<PRECI_TOL)
            return true;
        else
            return false;
    }
};


template<typename T>
std::ostream& operator<<(std::ostream &os, const Tensor<T> &a) {
    os.precision(9);
    os << *a.data();
    return os;
}

template<typename T, size_t M>
std::ostream& operator<<(std::ostream &os, const Tensor<T,M> &a) {

    os.precision(9);
    auto &&w = std::setw(7);
    os << "⎡" << w << a(0) << " ⎤\n";
    for (size_t i = 1; i + 1 < M; ++i) {
        os << "⎢" << w << a(i) << " ⎥\n";
    }
    os << "⎣" << w << a(M - 1);

    return os << " ⎦\n";
}

template<typename T, size_t M, size_t N>
std::ostream& operator<<(std::ostream &os, const Tensor<T,M,N> &a) {

    os.precision(9);
    auto &&w = std::setw(7);
    if (M>1) {
        os << "⎡" << w << a(0,0);
        for (size_t j = 1; j < N; ++j) {
            os << ", " << w << a(0,j);
        }
        os << " ⎤\n";
        for (size_t i = 1; i + 1 < M; ++i) {
            os << "⎢" << w << a(i,0);
            for (size_t j = 1; j < N; ++j) {
                os << ", " << w << a(i,j);
            }
            os << " ⎥\n";
        }
        os << "⎣" << w << a(M - 1,0);
        for (size_t j = 1; j < N; ++j) {
            os << ", " << w << a(M - 1,j);
        }
    }
    else {
        os << "⎡" << w << a(0,0);
        for (size_t j = 1; j < N; ++j) {
            os << ", " << w << a(0,j);
        }
    }

    return os << " ⎦\n";
}


template<typename T, size_t P, size_t M, size_t ... Rest,
         typename std::enable_if<sizeof...(Rest)==1,bool>::type=0>
std::ostream& operator<<(std::ostream &os, const Tensor<T,P,M,Rest...> &a) {
    constexpr size_t N = get_value<3,P,M,Rest...>::value;
    os.precision(9);
    auto &&w = std::setw(7);
    for (size_t k=0; k<P; ++k) {
        os << "["<< k << ",:,:]\n";
        if (M>1) {
            os << "⎡" << w << a(k,0,0);
            for (size_t j = 1; j < N; ++j) {
                os << ", " << w << a(k,0,j);
            }
            os << " ⎤\n";
            for (size_t i = 1; i + 1 < M; ++i) {
                os << "⎢" << w << a(k,i,0);
                for (size_t j = 1; j < N; ++j) {
                    os << ", " << w << a(k,i,j);
                }
                os << " ⎥\n";
            }
            os << "⎣" << w << a(k,M - 1,0);
            for (size_t j = 1; j < N; ++j) {
                os << ", " << w << a(k,M - 1,j);
            }
        }
        else {
            os << "⎡" << w << a(k,0,0);
            for (size_t j = 1; j < N; ++j) {
                os << ", " << w << a(k,0,j);
            }
        }
        os << " ⎦\n";
    }

    return os;
}

template<typename T, size_t P, size_t Q, size_t ... Rest,
         typename std::enable_if<sizeof...(Rest)==2,bool>::type=0>
std::ostream& operator<<(std::ostream &os, const Tensor<T,Q,P,Rest...> &a) {
    constexpr size_t M = get_value<3,P,Q,Rest...>::value;
    constexpr size_t N = get_value<4,P,Q,Rest...>::value;
    os.precision(9);
    auto &&w = std::setw(7);
    for (size_t l=0; l<Q; ++l) {
        for (size_t k=0; k<P; ++k) {
            os << "["<< l << "," << k << ",:,:]\n";
            os << "⎡" << w << a(l,k,0,0);
            for (size_t j = 1; j < N; ++j) {
                os << ", " << w << a(l,k,0,j);
            }
            os << " ⎤\n";
            for (size_t i = 1; i + 1 < M; ++i) {
                os << "⎢" << w << a(l,k,i,0);
                for (size_t j = 1; j < N; ++j) {
                    os << ", " << w << a(l,k,i,j);
                }
                os << " ⎥\n";
            }
            os << "⎣" << w << a(l,k,M - 1,0);
            for (size_t j = 1; j < N; ++j) {
                os << ", " << w << a(l,k,M - 1,j);
            }
            os << " ⎦\n";
        }
    }
    return os;
}



template<typename T>
using std_matrix = typename std::vector<std::vector<T>>::type;

// Generate combinations
template<size_t M, size_t N, size_t ... Rest>
FASTOR_INLINE std::vector<std::vector<int>> index_generator() {
    // Do NOT change int to size_t, comparison overflows
//    Tensor<float,prod<M,N,Rest...>::value,sizeof...(Rest)+2> idx;
//    std::vector<std::vector<int>> idx(prod<M,N,Rest...>::value);
    std::vector<std::vector<int>> idx; idx.resize(prod<M,N,Rest...>::value);
    std::array<int,sizeof...(Rest)+2> maxes = {M,N,Rest...};
    std::array<int,sizeof...(Rest)+2> a;
    int i,j;
    std::fill(a.begin(),a.end(),0);

    auto counter=0;
    while(1)
    {
        std::vector<int> current_idx; //current_idx.reserve(sizeof...(Rest)+2);
        for(i = 0; i< sizeof...(Rest)+2; i++) {
            current_idx.push_back(a[i]);
        }
        idx[counter] = current_idx;
        counter++;
        for(j = sizeof...(Rest)+2-1 ; j>=0 ; j--)
        {
            if(++a[j]<maxes[j])
                break;
            else
                a[j]=0;
        }
        if(j<0)
            break;
    }
    return idx;
}


template<typename T, size_t M, size_t N, size_t ... Rest,
         typename std::enable_if<sizeof...(Rest)>=3,bool>::type=0>
std::ostream& operator<<(std::ostream &os, const Tensor<T,M,N,Rest...> &a) {

    T *a_data = a.data();

    constexpr int DimensionHolder[sizeof...(Rest)+2] = {M,N,Rest...};
    int prods = 1;
    for (int i=0; i<a.Dimension-2; ++i) {
        prods *= DimensionHolder[i];
    }
    int lastrowcol = 1;
    for (int i=a.Dimension-2; i<a.Dimension; ++i) {
        lastrowcol *= DimensionHolder[i];
    }

    std::vector<std::vector<int>> combs = index_generator<M,N,Rest...>();
    os.precision(9);
    auto &&w = std::setw(7);
    size_t dims_2d = DimensionHolder[a.Dimension-2]*DimensionHolder[a.Dimension-1];
    for (int dims=0; dims<prods; ++dims) {
        os << "[";
        for (size_t kk=0; kk<sizeof...(Rest); ++kk) {
            os << combs[dims_2d*dims][kk] << ",";
        }
        os << ":,:]\n";
        if (DimensionHolder[a.Dimension-2] > 1) {
            os << "⎡" << w << a_data[lastrowcol*dims];
            for (size_t j = 1; j < DimensionHolder[a.Dimension-1]; ++j) {
                os << ", " << w << a_data[j+lastrowcol*dims];
            }
            os << " ⎤\n";
            for (size_t i = 1; i + 1 < DimensionHolder[a.Dimension-2]; ++i) {
                os << "⎢" << w << a_data[i*DimensionHolder[a.Dimension-1]+lastrowcol*dims];
                for (size_t j = 1; j < DimensionHolder[a.Dimension-1]; ++j) {
                    os << ", " << w << a_data[i*DimensionHolder[a.Dimension-1]+j+lastrowcol*dims];
                }
                os << " ⎥\n";
            }
            os << "⎣" << w << a_data[(DimensionHolder[a.Dimension-2]-1)*DimensionHolder[a.Dimension-1]+lastrowcol*dims];
            for (size_t j = 1; j < DimensionHolder[a.Dimension-1]; ++j) {
                os << ", " << w << a_data[(DimensionHolder[a.Dimension-2]-1)*DimensionHolder[a.Dimension-1]+j+lastrowcol*dims];
            }
        }
        else {
            os << "⎡" << w << a_data[lastrowcol*dims];
            for (size_t j = 1; j < DimensionHolder[a.Dimension-1]; ++j) {
                os << ", " << w << a_data[j+lastrowcol*dims];
            }
        }
        os << " ⎦\n";
    }

    return os;
}


template<typename T, size_t I, size_t J>
FASTOR_INLINE Tensor<T,J,I> transpose(const Tensor<T,I,J> &a) {
    Tensor<T,J,I> out;
    _transpose<T,I,J>(static_cast<const T *>(a.data()),out.data());
    return out;
}

template<typename T, size_t I>
FASTOR_INLINE T trace(const Tensor<T,I,I> &a) {
    return _trace<T,I,I>(static_cast<const T *>(a.data()));
}

template<typename T, size_t I>
FASTOR_INLINE T determinant(const Tensor<T,I,I> &a) {
    return _det<T,I,I>(static_cast<const T *>(a.data()));
}

template<typename T, size_t I>
FASTOR_INLINE Tensor<T,I,I> cofactor(const Tensor<T,I,I> &a) {
    Tensor<T,I,I> out;
    _cofactor<T,I,I>(a.data(),out.data());
    return out;
}

template<typename T, size_t I>
FASTOR_INLINE Tensor<T,I,I> adjoint(const Tensor<T,I,I> &a) {
    Tensor<T,I,I> out;
    _adjoint<T,I,I>(a.data(),out.data());
    return out;
}

template<typename T, size_t I>
FASTOR_INLINE Tensor<T,I,I> inverse(const Tensor<T,I,I> &a) {
    return adjoint(a)/determinant(a);
}

template<typename T, size_t ... Rest>
FASTOR_INLINE T norm(const Tensor<T,Rest...> &a) {
    if (sizeof...(Rest) == 0)
        return *a.data();
    return _norm<T,prod<Rest...>::value>(a.data());
}

// matmul - matvec overloads
template<typename T, size_t I, size_t J, size_t K>
FASTOR_INLINE Tensor<T,I,K> matmul(const Tensor<T,I,J> &a, const Tensor<T,J,K> &b) {
    Tensor<T,I,K> out; out.zeros();
    _matmul<T,I,J,K>(a.data(),b.data(),out.data());
    return out;
}

template<typename T, size_t I, size_t J>
FASTOR_INLINE Tensor<T,J> matmul(const Tensor<T,I,J> &a, const Tensor<T,J> &b) {
    Tensor<T,J> out;
    _matmul<T,I,J,1>(a.data(),b.data(),out.data());
    return out;
}

template<typename T, size_t I, size_t J>
FASTOR_INLINE Tensor<T,I> matmul(const Tensor<T,J> &b, const Tensor<T,J,I> &a) {
    Tensor<T,I> out;
    _matmul<T,J,I,1>(a.data(),b.data(),out.data());
    return out;
}

// Tensor cross product of two 2nd order tensors
template<typename T, size_t I, size_t J, typename std::enable_if<I==3 && J==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J> cross(const Tensor<T,I,J> &b, const Tensor<T,I,J> &a) {
    Tensor<T,I,J> out;
    _crossproduct<T,I,I,J>(a.data(),b.data(),out.data());
    return out;
}

template<typename T, size_t I, size_t J, typename std::enable_if<I==2 && J==2,bool>::type=0>
FASTOR_INLINE Tensor<T,I+1,J+1> cross(const Tensor<T,I,J> &b, const Tensor<T,I,J> &a) {
    Tensor<T,I+1,J+1> out;
    _crossproduct<T,I,I,J>(a.data(),b.data(),out.data());
    return out;
}

// Tensor cross product of a vector with 2nd order tensor
template<typename T, size_t I, size_t J, typename std::enable_if<I==3 && J==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J> cross(const Tensor<T,I> &b, const Tensor<T,I,J> &a) {
    Tensor<T,I,J> out;
    _crossproduct<T,I,1,J>(a.data(),b.data(),out.data());
    return out;
}

template<typename T, size_t I, size_t J, typename std::enable_if<I==2 && J==2,bool>::type=0>
FASTOR_INLINE Tensor<T,I+1,J+1> cross(const Tensor<T,I> &b, const Tensor<T,I,J> &a) {
    Tensor<T,I+1,J+1> out;
    _crossproduct<T,I,1,J>(a.data(),b.data(),out.data());
    return out;
}

// Tensor cross product of a 2nd order tensor with a vector
template<typename T, size_t I, size_t J, typename std::enable_if<I==3 && J==3,bool>::type=0>
FASTOR_INLINE Tensor<T,I,J> cross(const Tensor<T,I,J> &b, const Tensor<T,J> &a) {
    Tensor<T,I,J> out;
    _crossproduct<T,I,J,1>(a.data(),b.data(),out.data());
    return out;
}

template<typename T, size_t I, size_t J, typename std::enable_if<I==2 && J==2,bool>::type=0>
FASTOR_INLINE Tensor<T,I+1,J+1> cross(const Tensor<T,I,J> &b, const Tensor<T,J> &a) {
    Tensor<T,I+1,J+1> out;
    _crossproduct<T,I,J,1>(a.data(),b.data(),out.data());
    return out;
}


// Constant tensors
static FASTOR_INLINE
Tensor<float,3,3,3> levi_civita_ps() {
    Tensor<float,3,3,3> LeCi_ps;
    LeCi_ps(0,1,2) = 1.f;
    LeCi_ps(1,2,0) = 1.f;
    LeCi_ps(2,0,1) = 1.f;
    LeCi_ps(1,0,2) = -1.f;
    LeCi_ps(2,1,0) = -1.f;
    LeCi_ps(0,2,1) = -1.f;

    return LeCi_ps;
}

static FASTOR_INLINE
Tensor<double,3,3,3> levi_civita_pd() {
    Tensor<double,3,3,3> LeCi_pd;
    LeCi_pd(0,1,2) = 1.;
    LeCi_pd(1,2,0) = 1.;
    LeCi_pd(2,0,1) = 1.;
    LeCi_pd(1,0,2) = -1.;
    LeCi_pd(2,1,0) = -1.;
    LeCi_pd(0,2,1) = -1.;

    return LeCi_pd;
}

//template<typename T>
//Tensor<T,3,3,3> levi_civita() {
//    Tensor<T,3,3,3> out; out.zeros();
//}

template<typename T, size_t ... Rest>
static FASTOR_INLINE
Tensor<T,Rest...> kronecker_delta() {
    Tensor<T,Rest...> out; out.eye();
    return out;
}


}


#endif // TENSOR_H

