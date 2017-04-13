
// Further auxiliary plugings


// Change DivOp to MulOp if operand is std::arithmetic
//-------------------------------------------------------------------------------------------//
template<typename Derived0, typename Derived1,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            std::is_arithmetic<Derived1>::value ,bool>::type=0>
FASTOR_INLINE Tensor(const BinaryDivOp<Derived0,Derived1,sizeof...(Rest)> &src) {
    verify_dimensions(src);
    T num = T(1.0) / src.rhs;
    SIMDVector<T,DEFAULT_ABI> _vec, _vnum(num);
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = src.lhs.template eval<T>(i) * _vnum;
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.lhs.template eval_s<T>(i) * num;
    }
}
template<typename Derived0, typename Derived1,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            std::is_arithmetic<Derived1>::value ,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest...>& operator=(const BinaryDivOp<Derived0,Derived1,sizeof...(Rest)> &src) {
    verify_dimensions(src);
    T num = T(1.0) / src.rhs;
    SIMDVector<T,DEFAULT_ABI> _vec, _vnum(num);
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = src.lhs.template eval<T>(i) * _vnum;
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.lhs.template eval_s<T>(i) * num;
    }
    return *this;
}
template<typename Derived0, typename Derived1,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            std::is_arithmetic<Derived1>::value ,bool>::type=0>
FASTOR_INLINE void operator+=(const BinaryDivOp<Derived0,Derived1,sizeof...(Rest)> &src) {
    verify_dimensions(src);
    T num = T(1.0) / src.rhs;
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec, _vnum(num);
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) + src.lhs.template eval<T>(i) * _vnum;
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] += src.lhs.template eval_s<T>(i) * num;
    }
}
template<typename Derived0, typename Derived1,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            std::is_arithmetic<Derived1>::value ,bool>::type=0>
FASTOR_INLINE void operator-=(const BinaryDivOp<Derived0,Derived1,sizeof...(Rest)> &src) {
    verify_dimensions(src);
    T num = T(1.0) / src.rhs;
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec, _vnum(num);
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) - src.lhs.template eval<T>(i) * _vnum;
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] -= src.lhs.template eval_s<T>(i) * num;
    }
}
template<typename Derived0, typename Derived1,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            std::is_arithmetic<Derived1>::value ,bool>::type=0>
FASTOR_INLINE void operator*=(const BinaryDivOp<Derived0,Derived1,sizeof...(Rest)> &src) {
    verify_dimensions(src);
    T num = T(1.0) / src.rhs;
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec, _vnum(num);
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) * src.lhs.template eval<T>(i) * _vnum;
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] *= src.lhs.template eval_s<T>(i) * num;
    }
}
template<typename Derived0, typename Derived1,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            std::is_arithmetic<Derived1>::value ,bool>::type=0>
FASTOR_INLINE void operator/=(const BinaryDivOp<Derived0,Derived1,sizeof...(Rest)> &src) {
    verify_dimensions(src);
    T num = T(1.0) / src.rhs;
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec, _vnum(T(1.0) / src.rhs);
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) / (src.lhs.template eval<T>(i) * _vnum);
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] /= src.lhs.template eval_s<T>(i) * num;
    }
}
//-------------------------------------------------------------------------------------------//