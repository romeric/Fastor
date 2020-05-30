#ifndef TENSOR_EVALUATOR_H
#define TENSOR_EVALUATOR_H

// Expression templates evaluators
//----------------------------------------------------------------------------------------------------------//
template<typename U=T>
FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i) const {
    SIMDVector<U,simd_abi_type> _vec;
    _vec.load(&_data[get_mem_index(i)],false);
    return _vec;
}
template<typename U=T>
FASTOR_INLINE T eval_s(FASTOR_INDEX i) const {
    return _data[get_mem_index(i)];
}
template<typename U=T>
FASTOR_INLINE SIMDVector<U,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
    SIMDVector<U,simd_abi_type> _vec;
    _vec.load(&_data[get_flat_index(i,j)],false);
    return _vec;
}
template<typename U=T>
FASTOR_INLINE T eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
    return _data[get_flat_index(i,j)];
}

template<typename U=T>
FASTOR_INLINE SIMDVector<U,simd_abi_type> teval(const std::array<int, dimension_t::value> &as) const {
    SIMDVector<U,simd_abi_type> _vec;
    _vec.load(&_data[get_flat_index(as)],false);
    return _vec;
}
template<typename U=T>
FASTOR_INLINE T teval_s(const std::array<int, dimension_t::value> &as) const {
    return _data[get_flat_index(as)];
}
//----------------------------------------------------------------------------------------------------------//

#endif // end of TENSOR_EVALUATOR_H
