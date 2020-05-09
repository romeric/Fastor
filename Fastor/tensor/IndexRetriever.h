#ifndef INDEX_RETRIEVER_H
#define INDEX_RETRIEVER_H

// Retrieving index
// Given a flat index get the flat index in to the tensor - note that for tensor class the incoming index is
// the out-going index but it may not be the same for special tensors for instance for SingleValueTensor and
// views and such the index may be different
//----------------------------------------------------------------------------------------------------------//
template<typename U>
FASTOR_INLINE U get_mem_index(U index) const {
#if FASTOR_BOUNDS_CHECK
    FASTOR_ASSERT((index>=0 && index < size()), "INDEX OUT OF BOUNDS");
#endif
    return index;
}

// Retrieving index for nD tensors
// Given a multi-dimensional index get the flat index in to the tensor
//----------------------------------------------------------------------------------------------------------//
template<typename... Args, typename std::enable_if<sizeof...(Args)==1
                        && sizeof...(Args)==dimension_t::value && is_arithmetic_pack<Args...>::value,bool>::type =0>
FASTOR_INLINE size_t get_flat_index(Args ... args) const {
    constexpr size_t M = get_value<1,Rest...>::value;
    const size_t i = get_index<0>(args...) < 0 ? M + get_index<0>(args...) : get_index<0>(args...);
#if FASTOR_BOUNDS_CHECK
    FASTOR_ASSERT( ( (i>=0 && i<M)), "INDEX OUT OF BOUNDS");
#endif
    return i;
}

template<typename... Args, typename std::enable_if<sizeof...(Args)==2
                        && sizeof...(Args)==dimension_t::value && is_arithmetic_pack<Args...>::value,bool>::type =0>
FASTOR_INLINE size_t get_flat_index(Args ... args) const {
    constexpr size_t M = get_value<1,Rest...>::value;
    constexpr size_t N = get_value<2,Rest...>::value;
    const size_t i = get_index<0>(args...) < 0 ? M + get_index<0>(args...) : get_index<0>(args...);
    const size_t j = get_index<1>(args...) < 0 ? N + get_index<1>(args...) : get_index<1>(args...);
#if FASTOR_BOUNDS_CHECK
    FASTOR_ASSERT( ( (i>=0 && i<M) && (j>=0 && j<N)), "INDEX OUT OF BOUNDS");
#endif
    return i*N+j;
}

template<typename... Args, typename std::enable_if<sizeof...(Args)==3
                        && sizeof...(Args)==dimension_t::value && is_arithmetic_pack<Args...>::value,bool>::type =0>
FASTOR_INLINE size_t get_flat_index(Args ... args) const {
    constexpr size_t M = get_value<1,Rest...>::value;
    constexpr size_t N = get_value<2,Rest...>::value;
    constexpr size_t P = get_value<3,Rest...>::value;
    const size_t i = get_index<0>(args...) < 0 ? M + get_index<0>(args...) : get_index<0>(args...);
    const size_t j = get_index<1>(args...) < 0 ? N + get_index<1>(args...) : get_index<1>(args...);
    const size_t k = get_index<2>(args...) < 0 ? P + get_index<2>(args...) : get_index<2>(args...);
#if FASTOR_BOUNDS_CHECK
    FASTOR_ASSERT( ( (i>=0 && i<M) && (j>=0 && j<N) && (k>=0 && k<P)), "INDEX OUT OF BOUNDS");
#endif
    return i*N*P+j*P+k;
}

template<typename... Args, typename std::enable_if<sizeof...(Args)==4
                        && sizeof...(Args)==dimension_t::value && is_arithmetic_pack<Args...>::value,bool>::type =0>
FASTOR_INLINE size_t get_flat_index(Args ... args) const {
    constexpr size_t M = get_value<1,Rest...>::value;
    constexpr size_t N = get_value<2,Rest...>::value;
    constexpr size_t P = get_value<3,Rest...>::value;
    constexpr size_t Q = get_value<4,Rest...>::value;
    const size_t i = get_index<0>(args...) < 0 ? M + get_index<0>(args...) : get_index<0>(args...);
    const size_t j = get_index<1>(args...) < 0 ? N + get_index<1>(args...) : get_index<1>(args...);
    const size_t k = get_index<2>(args...) < 0 ? P + get_index<2>(args...) : get_index<2>(args...);
    const size_t l = get_index<3>(args...) < 0 ? Q + get_index<3>(args...) : get_index<3>(args...);
#if FASTOR_BOUNDS_CHECK
    FASTOR_ASSERT( ( (i>=0 && i<M) && (j>=0 && j<N)
              && (k>=0 && k<P) && (l>=0 && l<Q)), "INDEX OUT OF BOUNDS");
#endif
    return i*N*P*Q+j*P*Q+k*Q+l;
}

template<typename... Args, typename std::enable_if<sizeof...(Args)>=5
                        && sizeof...(Args)==dimension_t::value && is_arithmetic_pack<Args...>::value,bool>::type =0>
FASTOR_INLINE size_t get_flat_index(Args ... args) const {
    /* The type of largs needs to be the type of incoming pack i.e.
        whatever the tensor is indexed with
    */
    get_nth_type<0,Args...> largs[sizeof...(Args)] = {args...};

    constexpr std::array<size_t,dimension_t::value> products_ = nprods_views<Index<Rest...>,
        typename std_ext::make_index_sequence<dimension_t::value>::type>::values;
    constexpr size_t DimensionHolder[dimension_t::value] = {Rest...};

    for (size_t i=0; i<dimension_t::value; ++i) {
        if ( largs[i] < 0 ) largs[i] += DimensionHolder[i];
#if FASTOR_BOUNDS_CHECK
        FASTOR_ASSERT( (largs[i]>=0 && largs[i]<DimensionHolder[i]), "INDEX OUT OF BOUNDS");
#endif
    }
    size_t index = 0;
    for (size_t i=0; i<dimension_t::value; ++i) {
        index += products_[i]*largs[i];
    }
    return index;
}
//----------------------------------------------------------------------------------------------------------//


FASTOR_INLINE int get_flat_index(const std::array<int, dimension_t::value> &as) const {
    constexpr std::array<size_t,dimension_t::value> products_ = nprods_views<Index<Rest...>,
        typename std_ext::make_index_sequence<dimension_t::value>::type>::values;
    size_t index = 0;
    for (size_t i=0; i<dimension_t::value; ++i) {
        index += products_[i]*as[i];
    }
#if FASTOR_BOUNDS_CHECK
    FASTOR_ASSERT((index>=0 && index<size()), "INDEX OUT OF BOUNDS");
#endif
    return index;
}


//----------------------------------------------------------------------------------------------------------//

#endif // INDEX_RETRIEVER_H
