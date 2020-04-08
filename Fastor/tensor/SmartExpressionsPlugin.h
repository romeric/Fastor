#ifndef SMART_EXPRESSION_OP_H
#define SMART_EXPRESSION_OP_H

// Plugin for all smart expressions - smart binders
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
        using V = SIMDVector<T,DEFAULT_ABI>; V vec;
        Tensor<T,I,I> inverser;
        T *inv_data = inverser.data();
        _adjoint<T,I,I>(src_.lhs.expr.data(),inv_data);
        T det = _det<T,I,I>(src_.lhs.expr.data());
        for (FASTOR_INDEX i=0; i<I*I; i+=V::Size) {
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
        using V = SIMDVector<T,DEFAULT_ABI>; V vec;
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

#endif // SMART_EXPRESSION_OP_H
