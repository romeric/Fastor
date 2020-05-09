#ifndef SCALAR_INDEXING_NONCONST_H
#define SCALAR_INDEXING_NONCONST_H


// Scalar indexing non-const
//----------------------------------------------------------------------------------------------------------//
template<typename... Args, typename std::enable_if<sizeof...(Args)==dimension_t::value &&
                                is_arithmetic_pack<Args...>::value,bool>::type =0>
FASTOR_INLINE T& operator()(Args ... args) {
    return _data[get_flat_index(args...)];
}
//----------------------------------------------------------------------------------------------------------//

#endif // SCALAR_INDEXING_NONCONST_H


#ifndef SCALAR_INDEXING_CONST_H
#define SCALAR_INDEXING_CONST_H

// Scalar indexing const
//----------------------------------------------------------------------------------------------------------//
template<typename... Args, typename std::enable_if<sizeof...(Args)==dimension_t::value &&
                                is_arithmetic_pack<Args...>::value,bool>::type =0>
constexpr FASTOR_INLINE const T& operator()(Args ... args) const {
    return _data[get_flat_index(args...)];
}

//----------------------------------------------------------------------------------------------------------//

#endif // SCALAR_INDEXING_CONST_H
