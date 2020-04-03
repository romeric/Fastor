
// Plugin for all smart expressions

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
