#ifndef TENSOR_EVALUATOR_H
#define TENSOR_EVALUATOR_H

// Expression templates evaluators
//----------------------------------------------------------------------------------------------------------//
template<typename U=T>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
    SIMDVector<T,DEFAULT_ABI> _vec;
    _vec.load(&_data[get_mem_index(i)],false);
    return _vec;
}
template<typename U=T>
FASTOR_INLINE T eval_s(FASTOR_INDEX i) const {
    return _data[get_mem_index(i)];
}
template<typename U=T>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
    SIMDVector<T,DEFAULT_ABI> _vec;
    _vec.load(&_data[get_flat_index(i,j)],false);
    return _vec;
}
template<typename U=T>
FASTOR_INLINE T eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
    return _data[get_flat_index(i,j)];
}

template<typename U=T>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> teval(const std::array<int, Dimension> &as) const {
    SIMDVector<T,DEFAULT_ABI> _vec;
    _vec.load(&_data[get_flat_index(as)],false);
    return _vec;
}
template<typename U=T>
FASTOR_INLINE T teval_s(const std::array<int, Dimension> &as) const {
    return _data[get_flat_index(as)];
}

// This is purely for smart ops
constexpr FASTOR_INLINE T eval(T i, T j) const {
    return _data[get_flat_index((FASTOR_INDEX)i,(FASTOR_INDEX)j)];
}
//----------------------------------------------------------------------------------------------------------//

#endif // end of TENSOR_EVALUATOR_H