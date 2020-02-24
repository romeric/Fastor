#ifndef TENSOR_INPLACE_OPERATORS_H
#define TENSOR_INPLACE_OPERATORS_H

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

#endif // TENSOR_INPLACE_OPERATORS_H