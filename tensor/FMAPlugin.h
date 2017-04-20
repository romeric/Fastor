
// FMA Overloads for tensors
// Note that with ffp-contract=fast both GCC and Clang fuse mul+add to fma
// So some of these methods are unncessary, however in that case the compiler
// has the option to spill the code it wants. Clang 3.9 generates much more
// more compact code if the following methods are defined 



//----------------------------------------------------------------------------------------------------------------------------------//
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.rhs.lhs.template eval<T>(i), src.rhs.rhs.template eval<T>(i), src.lhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs.template eval_s<T>(i) + src.lhs.template eval_s<T>(i);
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.rhs.lhs.template eval<T>(i), V(src.rhs.rhs), src.lhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs + src.lhs.template eval_s<T>(i);
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.rhs.rhs.template eval<T>(i), V(src.rhs.lhs), src.lhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.rhs.rhs.template eval_s<T>(i) * src.rhs.lhs + src.lhs.template eval_s<T>(i);
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.rhs.lhs.template eval<T>(i), src.rhs.rhs.template eval<T>(i), V(src.lhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs.template eval_s<T>(i) + src.lhs;
    }
}

template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.lhs.lhs.template eval<T>(i), src.lhs.rhs.template eval<T>(i), src.rhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs.template eval_s<T>(i) + src.rhs.template eval_s<T>(i) ;
    }
}

template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.lhs.lhs.template eval<T>(i), src.lhs.rhs.template eval<T>(i), V(src.rhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs.template eval_s<T>(i) + src.rhs;
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.lhs.lhs.template eval<T>(i), V(src.lhs.rhs), src.rhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs + src.rhs.template eval_s<T>(i) ;
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.lhs.rhs.template eval<T>(i), V(src.lhs.lhs), src.rhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.lhs.rhs.template eval_s<T>(i) * src.lhs.lhs + src.rhs.template eval_s<T>(i) ;
    }
}

// Change SubOp to AddOp and use FMA if operand is std::arithmetic
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor(const BinarySubOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.rhs.lhs.template eval<T>(i), src.rhs.rhs.template eval<T>(i), V(-src.lhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs.template eval_s<T>(i) - src.lhs;
    }
}
// Change SubOp to AddOp and use FMA if operand is std::arithmetic
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor(const BinarySubOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.lhs.lhs.template eval<T>(i), src.lhs.rhs.template eval<T>(i), V(-src.rhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs.template eval_s<T>(i) - src.rhs;
    }
}
// //----------------------------------------------------------------------------------------------------------------------------------//






//----------------------------------------------------------------------------------------------------------------------------------//
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest...>& operator=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.rhs.lhs.template eval<T>(i), src.rhs.rhs.template eval<T>(i), src.lhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs.template eval_s<T>(i) + src.lhs.template eval_s<T>(i);
    }
    return *this;
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest...>& operator=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.rhs.lhs.template eval<T>(i), V(src.rhs.rhs), src.lhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs + src.lhs.template eval_s<T>(i);
    }
    return *this;
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest...>& operator=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.rhs.rhs.template eval<T>(i), V(src.rhs.lhs), src.lhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.rhs.rhs.template eval_s<T>(i) * src.rhs.lhs + src.lhs.template eval_s<T>(i);
    }
    return *this;
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest...>& operator=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.rhs.lhs.template eval<T>(i), src.rhs.rhs.template eval<T>(i), V(src.lhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs.template eval_s<T>(i) + src.lhs;
    }
    return *this;
}

template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest...>& operator=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.lhs.lhs.template eval<T>(i), src.lhs.rhs.template eval<T>(i), src.rhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs.template eval_s<T>(i) + src.rhs.template eval_s<T>(i) ;
    }
    return *this;
}

template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest...>& operator=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.lhs.lhs.template eval<T>(i), src.lhs.rhs.template eval<T>(i), V(src.rhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs.template eval_s<T>(i) + src.rhs;
    }
    return *this;
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest...>& operator=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.lhs.lhs.template eval<T>(i), V(src.lhs.rhs), src.rhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs + src.rhs.template eval_s<T>(i) ;
    }
    return *this;
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest...>& operator=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.lhs.rhs.template eval<T>(i), V(src.lhs.lhs), src.rhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.lhs.rhs.template eval_s<T>(i) * src.lhs.lhs + src.rhs.template eval_s<T>(i) ;
    }
    return *this;
}

// Change SubOp to AddOp and use FMA if operand is std::arithmetic
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest...>& operator=(const BinarySubOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.rhs.lhs.template eval<T>(i), src.rhs.rhs.template eval<T>(i), V(-src.lhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs.template eval_s<T>(i) - src.lhs;
    }
    return *this;
}
// Change SubOp to AddOp and use FMA if operand is std::arithmetic
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest...>& operator=(const BinarySubOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = fmadd(src.lhs.lhs.template eval<T>(i), src.lhs.rhs.template eval<T>(i), V(-src.rhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] = src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs.template eval_s<T>(i) - src.rhs;
    }
    return *this;
}
//----------------------------------------------------------------------------------------------------------------------------------//





//----------------------------------------------------------------------------------------------------------------------------------//
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator+=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) + fmadd(src.rhs.lhs.template eval<T>(i), src.rhs.rhs.template eval<T>(i), src.lhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] += src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs.template eval_s<T>(i) + src.lhs.template eval_s<T>(i);
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator+=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) + fmadd(src.rhs.lhs.template eval<T>(i), V(src.rhs.rhs), src.lhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] += src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs + src.lhs.template eval_s<T>(i);
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator+=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) + fmadd(src.rhs.rhs.template eval<T>(i), V(src.rhs.lhs), src.lhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] += src.rhs.rhs.template eval_s<T>(i) * src.rhs.lhs + src.lhs.template eval_s<T>(i);
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator+=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) + fmadd(src.rhs.lhs.template eval<T>(i), src.rhs.rhs.template eval<T>(i), V(src.lhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] += src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs.template eval_s<T>(i) + src.lhs;
    }
}

template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator+=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) + fmadd(src.lhs.lhs.template eval<T>(i), src.lhs.rhs.template eval<T>(i), src.rhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] += src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs.template eval_s<T>(i) + src.rhs.template eval_s<T>(i) ;
    }
}

template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator+=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) + fmadd(src.lhs.lhs.template eval<T>(i), src.lhs.rhs.template eval<T>(i), V(src.rhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] += src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs.template eval_s<T>(i) + src.rhs;
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator+=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) + fmadd(src.lhs.lhs.template eval<T>(i), V(src.lhs.rhs), src.rhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] += src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs + src.rhs.template eval_s<T>(i) ;
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator+=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) + fmadd(src.lhs.rhs.template eval<T>(i), V(src.lhs.lhs), src.rhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] += src.lhs.rhs.template eval_s<T>(i) * src.lhs.lhs + src.rhs.template eval_s<T>(i) ;
    }
}

// Change SubOp to AddOp and use FMA if operand is std::arithmetic
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator+=(const BinarySubOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) + fmadd(src.rhs.lhs.template eval<T>(i), src.rhs.rhs.template eval<T>(i), V(-src.lhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] += src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs.template eval_s<T>(i) - src.lhs;
    }
}
// Change SubOp to AddOp and use FMA if operand is std::arithmetic
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator+=(const BinarySubOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) + fmadd(src.lhs.lhs.template eval<T>(i), src.lhs.rhs.template eval<T>(i), V(-src.rhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] += src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs.template eval_s<T>(i) - src.rhs;
    }
}
//----------------------------------------------------------------------------------------------------------------------------------//




//----------------------------------------------------------------------------------------------------------------------------------//
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator-=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) - fmadd(src.rhs.lhs.template eval<T>(i), src.rhs.rhs.template eval<T>(i), src.lhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] -= src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs.template eval_s<T>(i) + src.lhs.template eval_s<T>(i);
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator-=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) - fmadd(src.rhs.lhs.template eval<T>(i), V(src.rhs.rhs), src.lhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] -= src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs + src.lhs.template eval_s<T>(i);
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator-=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) - fmadd(src.rhs.rhs.template eval<T>(i), V(src.rhs.lhs), src.lhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] -= src.rhs.rhs.template eval_s<T>(i) * src.rhs.lhs + src.lhs.template eval_s<T>(i);
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator-=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) - fmadd(src.rhs.lhs.template eval<T>(i), src.rhs.rhs.template eval<T>(i), V(src.lhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] -= src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs.template eval_s<T>(i) + src.lhs;
    }
}

template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator-=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) - fmadd(src.lhs.lhs.template eval<T>(i), src.lhs.rhs.template eval<T>(i), src.rhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] -= src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs.template eval_s<T>(i) + src.rhs.template eval_s<T>(i) ;
    }
}

template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator-=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) - fmadd(src.lhs.lhs.template eval<T>(i), src.lhs.rhs.template eval<T>(i), V(src.rhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] -= src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs.template eval_s<T>(i) + src.rhs;
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator-=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) - fmadd(src.lhs.lhs.template eval<T>(i), V(src.lhs.rhs), src.rhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] -= src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs + src.rhs.template eval_s<T>(i) ;
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator-=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) - fmadd(src.lhs.rhs.template eval<T>(i), V(src.lhs.lhs), src.rhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] -= src.lhs.rhs.template eval_s<T>(i) * src.lhs.lhs + src.rhs.template eval_s<T>(i) ;
    }
}

// Change SubOp to AddOp and use FMA if operand is std::arithmetic
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator-=(const BinarySubOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) - fmadd(src.rhs.lhs.template eval<T>(i), src.rhs.rhs.template eval<T>(i), V(-src.lhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] -= src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs.template eval_s<T>(i) - src.lhs;
    }
}
// Change SubOp to AddOp and use FMA if operand is std::arithmetic
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator-=(const BinarySubOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) - fmadd(src.lhs.lhs.template eval<T>(i), src.lhs.rhs.template eval<T>(i), V(-src.rhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] -= src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs.template eval_s<T>(i) - src.rhs;
    }
}
//----------------------------------------------------------------------------------------------------------------------------------//








//----------------------------------------------------------------------------------------------------------------------------------//
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator*=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) * fmadd(src.rhs.lhs.template eval<T>(i), src.rhs.rhs.template eval<T>(i), src.lhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] *= src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs.template eval_s<T>(i) + src.lhs.template eval_s<T>(i);
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator*=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) * fmadd(src.rhs.lhs.template eval<T>(i), V(src.rhs.rhs), src.lhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] *= src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs + src.lhs.template eval_s<T>(i);
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator*=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) * fmadd(src.rhs.rhs.template eval<T>(i), V(src.rhs.lhs), src.lhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] *= src.rhs.rhs.template eval_s<T>(i) * src.rhs.lhs + src.lhs.template eval_s<T>(i);
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator*=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) * fmadd(src.rhs.lhs.template eval<T>(i), src.rhs.rhs.template eval<T>(i), V(src.lhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] *= src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs.template eval_s<T>(i) + src.lhs;
    }
}

template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator*=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) * fmadd(src.lhs.lhs.template eval<T>(i), src.lhs.rhs.template eval<T>(i), src.rhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] *= src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs.template eval_s<T>(i) + src.rhs.template eval_s<T>(i) ;
    }
}

template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator*=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) * fmadd(src.lhs.lhs.template eval<T>(i), src.lhs.rhs.template eval<T>(i), V(src.rhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] *= src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs.template eval_s<T>(i) + src.rhs;
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator*=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) * fmadd(src.lhs.lhs.template eval<T>(i), V(src.lhs.rhs), src.rhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] *= src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs + src.rhs.template eval_s<T>(i) ;
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator*=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) * fmadd(src.lhs.rhs.template eval<T>(i), V(src.lhs.lhs), src.rhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] *= src.lhs.rhs.template eval_s<T>(i) * src.lhs.lhs + src.rhs.template eval_s<T>(i) ;
    }
}

// Change SubOp to AddOp and use FMA if operand is std::arithmetic
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator*=(const BinarySubOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) * fmadd(src.rhs.lhs.template eval<T>(i), src.rhs.rhs.template eval<T>(i), V(-src.lhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] *= src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs.template eval_s<T>(i) - src.lhs;
    }
}
// Change SubOp to AddOp and use FMA if operand is std::arithmetic
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator*=(const BinarySubOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) * fmadd(src.lhs.lhs.template eval<T>(i), src.lhs.rhs.template eval<T>(i), V(-src.rhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] *= src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs.template eval_s<T>(i) - src.rhs;
    }
}
//----------------------------------------------------------------------------------------------------------------------------------//




//----------------------------------------------------------------------------------------------------------------------------------//
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator/=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) / fmadd(src.rhs.lhs.template eval<T>(i), src.rhs.rhs.template eval<T>(i), src.lhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] /= src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs.template eval_s<T>(i) + src.lhs.template eval_s<T>(i);
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator/=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) / fmadd(src.rhs.lhs.template eval<T>(i), V(src.rhs.rhs), src.lhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] /= src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs + src.lhs.template eval_s<T>(i);
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator/=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) / fmadd(src.rhs.rhs.template eval<T>(i), V(src.rhs.lhs), src.lhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] /= src.rhs.rhs.template eval_s<T>(i) * src.rhs.lhs + src.lhs.template eval_s<T>(i);
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator/=(const BinaryAddOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) / fmadd(src.rhs.lhs.template eval<T>(i), src.rhs.rhs.template eval<T>(i), V(src.lhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] /= src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs.template eval_s<T>(i) + src.lhs;
    }
}

template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator/=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) / fmadd(src.lhs.lhs.template eval<T>(i), src.lhs.rhs.template eval<T>(i), src.rhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] /= src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs.template eval_s<T>(i) + src.rhs.template eval_s<T>(i) ;
    }
}

template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator/=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) / fmadd(src.lhs.lhs.template eval<T>(i), src.lhs.rhs.template eval<T>(i), V(src.rhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] /= src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs.template eval_s<T>(i) + src.rhs;
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator/=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) / fmadd(src.lhs.lhs.template eval<T>(i), V(src.lhs.rhs), src.rhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] /= src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs + src.rhs.template eval_s<T>(i) ;
    }
}
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator/=(const BinaryAddOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) / fmadd(src.lhs.rhs.template eval<T>(i), V(src.lhs.lhs), src.rhs.template eval<T>(i) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] /= src.lhs.rhs.template eval_s<T>(i) * src.lhs.lhs + src.rhs.template eval_s<T>(i) ;
    }
}

// Change SubOp to AddOp and use FMA if operand is std::arithmetic
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            !std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator/=(const BinarySubOp<Derived0,BinaryMulOp<Derived1,Derived2,sizeof...(Rest)>, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) / fmadd(src.rhs.lhs.template eval<T>(i), src.rhs.rhs.template eval<T>(i), V(-src.lhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] /= src.rhs.lhs.template eval_s<T>(i) * src.rhs.rhs.template eval_s<T>(i) - src.lhs;
    }
}
// Change SubOp to AddOp and use FMA if operand is std::arithmetic
template<typename Derived0, typename Derived1, typename Derived2,
    typename std::enable_if<!std::is_arithmetic<Derived0>::value &&
                            !std::is_arithmetic<Derived1>::value &&
                            std::is_arithmetic<Derived2>::value,bool>::type=0>
FASTOR_INLINE void operator/=(const BinarySubOp<BinaryMulOp<Derived0,Derived1,sizeof...(Rest)>, Derived2, sizeof...(Rest) > &src) {
    verify_dimensions(src);
    using V = SIMDVector<T,DEFAULT_ABI>;
    V _vec;
    FASTOR_INDEX i;
    for (i = 0; i <ROUND_DOWN(src.size(),Stride); i+=Stride) {
        _vec = V(_data+i) / fmadd(src.lhs.lhs.template eval<T>(i), src.lhs.rhs.template eval<T>(i), V(-src.rhs) );
        _vec.store(_data+i);
    }
    for (; i <src.size(); ++i) {
        _data[i] /= src.lhs.lhs.template eval_s<T>(i) * src.lhs.rhs.template eval_s<T>(i) - src.rhs;
    }
}
//----------------------------------------------------------------------------------------------------------------------------------//