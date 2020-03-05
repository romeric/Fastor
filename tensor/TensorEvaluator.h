#ifndef TENSOR_EVALUATOR_H
#define TENSOR_EVALUATOR_H

// Expression templates evaluators
//----------------------------------------------------------------------------------------------------------//
template<typename U=T>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
#ifdef BOUNDSCHECK
    // This is a generic evaluator and not for 1D cases only
    FASTOR_ASSERT((i>=0 && i<Size), "INDEX OUT OF BOUNDS");
#endif
    SIMDVector<T,DEFAULT_ABI> out;
    out.load(&_data[i]);
    return out;
}
template<typename U=T>
FASTOR_INLINE T eval_s(FASTOR_INDEX i) const {
#ifdef BOUNDSCHECK
    // This is a generic evaluator and not for 1D cases only
    FASTOR_ASSERT((i>=0 && i<Size), "INDEX OUT OF BOUNDS");
#endif
    return _data[i];
}
template<typename U=T>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
    static_assert(Dimension==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    constexpr int N = get_value<2,Rest...>::value;
#ifdef BOUNDSCHECK
    constexpr int M = get_value<1,Rest...>::value;
    FASTOR_ASSERT((i>=0 && i<M && j>=0 && j<N), "INDEX OUT OF BOUNDS");
#endif
    // return SIMDVector<T,DEFAULT_ABI>(&_data[i*N+j]); // Careful, causes segfaults
    SIMDVector<T,DEFAULT_ABI> _vec; _vec.load(&_data[i*N+j],false);
    return _vec;
}
template<typename U=T>
FASTOR_INLINE T eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
    static_assert(Dimension==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
#ifdef BOUNDSCHECK
    constexpr int M = get_value<1,Rest...>::value;
    constexpr int N = get_value<2,Rest...>::value;
    FASTOR_ASSERT((i>=0 && i<M && j>=0 && j<N), "INDEX OUT OF BOUNDS");
#endif
    return _data[i*get_value<2,Rest...>::value+j];
}

// template<typename... Args, typename std::enable_if<sizeof...(Args)==Dimension_t::value
//                     && is_arithmetic_pack<Args...>::value,bool>::type =0>
// FASTOR_INLINE const T& eval(Args ... args) const {
//     return operator()(args...);
// }
// template<typename... Args, typename std::enable_if<sizeof...(Args)==Dimension_t::value
//                     && is_arithmetic_pack<Args...>::value,bool>::type =0>
// FASTOR_INLINE const T& eval_s(Args ... args) const {
//     return operator()(args...);
// }

// This is purely for smart ops
constexpr FASTOR_INLINE T eval(T i, T j) const {
    return _data[static_cast<FASTOR_INDEX>(i)*get_value<2,Rest...>::value+static_cast<FASTOR_INDEX>(j)];
}


template<typename U=T>
FASTOR_INLINE SIMDVector<T,DEFAULT_ABI> teval(const std::array<int, Dimension> &as) const {

    constexpr std::array<size_t,Dimension> products_ = nprods_views<Index<Rest...>,
        typename std_ext::make_index_sequence<Dimension>::type>::values;

    int index = 0;
    for (int i=0; i<Dimension; ++i) {
        index += products_[i]*as[i];
    }

#ifdef BOUNDSCHECK
    FASTOR_ASSERT((index>=0 && index<Size), "INDEX OUT OF BOUNDS");
#endif

    SIMDVector<T,DEFAULT_ABI> _vec; _vec.load(&_data[index],false);
    return _vec;
}
template<typename U=T>
FASTOR_INLINE T teval_s(const std::array<int, Dimension> &as) const {

    constexpr std::array<size_t,Dimension> products_ = nprods_views<Index<Rest...>,
        typename std_ext::make_index_sequence<Dimension>::type>::values;

    int index = 0;
    for (int i=0; i<Dimension; ++i) {
        index += products_[i]*as[i];
    }

#ifdef BOUNDSCHECK
    FASTOR_ASSERT((index>=0 && index<Size), "INDEX OUT OF BOUNDS");
#endif

    return _data[index];
}

//----------------------------------------------------------------------------------------------------------//

#endif // end of TENSOR_EVALUATOR_H
