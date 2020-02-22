#ifndef SCALAR_INDEXING_NONCONST_H
#define SCALAR_INDEXING_NONCONST_H


// Scalar indexing non-const
//----------------------------------------------------------------------------------------------------------//
template<typename... Args, typename std::enable_if<sizeof...(Args)==1
                        && sizeof...(Args)==Dimension_t::value && is_arithmetic_pack<Args...>::value,bool>::type =0>
FASTOR_INLINE T& operator()(Args ... args) {
    constexpr int M = get_value<1,Rest...>::value;
    const int i = get_index<0>(args...) < 0 ? M + get_index<0>(args...) : get_index<0>(args...);
#ifdef BOUNDSCHECK
    assert( ( (i>=0 && i<M)) && "INDEX OUT OF BOUNDS");
#endif
    return _data[i];
}

template<typename... Args, typename std::enable_if<sizeof...(Args)==2
                        && sizeof...(Args)==Dimension_t::value && is_arithmetic_pack<Args...>::value,bool>::type =0>
FASTOR_INLINE T& operator()(Args ... args) {
    constexpr int M = get_value<1,Rest...>::value;
    constexpr int N = get_value<2,Rest...>::value;
    const int i = get_index<0>(args...) < 0 ? M + get_index<0>(args...) : get_index<0>(args...);
    const int j = get_index<1>(args...) < 0 ? N + get_index<1>(args...) : get_index<1>(args...);
#ifdef BOUNDSCHECK
    assert( ( (i>=0 && i<M) && (j>=0 && j<N)) && "INDEX OUT OF BOUNDS");
#endif
    return _data[i*N+j];
}

template<typename... Args, typename std::enable_if<sizeof...(Args)==3
                        && sizeof...(Args)==Dimension_t::value && is_arithmetic_pack<Args...>::value,bool>::type =0>
FASTOR_INLINE T& operator()(Args ... args) {
    constexpr int M = get_value<1,Rest...>::value;
    constexpr int N = get_value<2,Rest...>::value;
    constexpr int P = get_value<3,Rest...>::value;
    const int i = get_index<0>(args...) < 0 ? M + get_index<0>(args...) : get_index<0>(args...);
    const int j = get_index<1>(args...) < 0 ? N + get_index<1>(args...) : get_index<1>(args...);
    const int k = get_index<2>(args...) < 0 ? P + get_index<2>(args...) : get_index<2>(args...);
#ifdef BOUNDSCHECK
    assert( ( (i>=0 && i<M) && (j>=0 && j<N) && (k>=0 && k<P)) && "INDEX OUT OF BOUNDS");
#endif
    return _data[i*N*P+j*P+k];
}

template<typename... Args, typename std::enable_if<sizeof...(Args)==4
                        && sizeof...(Args)==Dimension_t::value && is_arithmetic_pack<Args...>::value,bool>::type =0>
FASTOR_INLINE T& operator()(Args ... args) {
    constexpr int M = get_value<1,Rest...>::value;
    constexpr int N = get_value<2,Rest...>::value;
    constexpr int P = get_value<3,Rest...>::value;
    constexpr int Q = get_value<4,Rest...>::value;
    const int i = get_index<0>(args...) < 0 ? M + get_index<0>(args...) : get_index<0>(args...);
    const int j = get_index<1>(args...) < 0 ? N + get_index<1>(args...) : get_index<1>(args...);
    const int k = get_index<2>(args...) < 0 ? P + get_index<2>(args...) : get_index<2>(args...);
    const int l = get_index<3>(args...) < 0 ? Q + get_index<3>(args...) : get_index<3>(args...);
#ifdef BOUNDSCHECK
    assert( ( (i>=0 && i<M) && (j>=0 && j<N)
              && (k>=0 && k<P) && (l>=0 && l<Q)) && "INDEX OUT OF BOUNDS");
#endif
    return _data[i*N*P*Q+j*P*Q+k*Q+l];
}

template<typename... Args, typename std::enable_if<sizeof...(Args)>=5
                        && sizeof...(Args)==Dimension_t::value && is_arithmetic_pack<Args...>::value,bool>::type =0>
FASTOR_INLINE T& operator()(Args ... args) {
    int largs[sizeof...(Args)] = {args...};
    constexpr int DimensionHolder[Dimension] = {Rest...};
    for (int i=0; i<Dimension; ++i) {
        if (largs[i] < 0) largs[i] += DimensionHolder[i];
#ifdef BOUNDSCHECK
        assert( (largs[i]>=0 && largs[i]<DimensionHolder[i]) && "INDEX OUT OF BOUNDS");
#endif
    }
    std::array<int,Dimension> products;
    for (int i=Dimension-1; i>0; --i) {
        int num = DimensionHolder[Dimension-1];
        for (int j=0; j<i-1; ++j) {
            num *= DimensionHolder[Dimension-1-j-1];
        }
        products[i] = num;
    }

    int index = largs[Dimension-1];
    for (int i=Dimension-1; i>0; --i) {
        index += products[i]*largs[Dimension-i-1];
    }
    return _data[index];
}
//----------------------------------------------------------------------------------------------------------//

#endif // SCALAR_INDEXING_NONCONST_H




#ifndef SCALAR_INDEXING_CONST_H
#define SCALAR_INDEXING_CONST_H

// Scalar indexing const
//----------------------------------------------------------------------------------------------------------//
template<typename... Args, typename std::enable_if<sizeof...(Args)==1
                        && sizeof...(Args)==Dimension_t::value && is_arithmetic_pack<Args...>::value,bool>::type =0>
FASTOR_INLINE const T& operator()(Args ... args) const {
    constexpr int M = get_value<1,Rest...>::value;
    const int i = get_index<0>(args...) < 0 ? M + get_index<0>(args...) : get_index<0>(args...);
#ifdef BOUNDSCHECK
    assert( ( (i>=0 && i<M)) && "INDEX OUT OF BOUNDS");
#endif
    return _data[i];
}

template<typename... Args, typename std::enable_if<sizeof...(Args)==2
                        && sizeof...(Args)==Dimension_t::value && is_arithmetic_pack<Args...>::value,bool>::type =0>
FASTOR_INLINE const T& operator()(Args ... args) const {
    constexpr int M = get_value<1,Rest...>::value;
    constexpr int N = get_value<2,Rest...>::value;
    const int i = get_index<0>(args...) < 0 ? M + get_index<0>(args...) : get_index<0>(args...);
    const int j = get_index<1>(args...) < 0 ? N + get_index<1>(args...) : get_index<1>(args...);
#ifdef BOUNDSCHECK
    assert( ( (i>=0 && i<M) && (j>=0 && j<N)) && "INDEX OUT OF BOUNDS");
#endif
    return _data[i*N+j];
}

template<typename... Args, typename std::enable_if<sizeof...(Args)==3
                        && sizeof...(Args)==Dimension_t::value && is_arithmetic_pack<Args...>::value,bool>::type =0>
FASTOR_INLINE const T&  operator()(Args ... args) const {
    constexpr int M = get_value<1,Rest...>::value;
    constexpr int N = get_value<2,Rest...>::value;
    constexpr int P = get_value<3,Rest...>::value;
    const int i = get_index<0>(args...) < 0 ? M + get_index<0>(args...) : get_index<0>(args...);
    const int j = get_index<1>(args...) < 0 ? N + get_index<1>(args...) : get_index<1>(args...);
    const int k = get_index<2>(args...) < 0 ? P + get_index<2>(args...) : get_index<2>(args...);
#ifdef BOUNDSCHECK
    assert( ( (i>=0 && i<M) && (j>=0 && j<N) && (k>=0 && k<P)) && "INDEX OUT OF BOUNDS");
#endif
    return _data[i*N*P+j*P+k];
}

template<typename... Args, typename std::enable_if<sizeof...(Args)==4
                        && sizeof...(Args)==Dimension_t::value && is_arithmetic_pack<Args...>::value,bool>::type =0>
FASTOR_INLINE const T& operator()(Args ... args) const {
    constexpr int M = get_value<1,Rest...>::value;
    constexpr int N = get_value<2,Rest...>::value;
    constexpr int P = get_value<3,Rest...>::value;
    constexpr int Q = get_value<4,Rest...>::value;
    const int i = get_index<0>(args...) < 0 ? M + get_index<0>(args...) : get_index<0>(args...);
    const int j = get_index<1>(args...) < 0 ? N + get_index<1>(args...) : get_index<1>(args...);
    const int k = get_index<2>(args...) < 0 ? P + get_index<2>(args...) : get_index<2>(args...);
    const int l = get_index<3>(args...) < 0 ? Q + get_index<3>(args...) : get_index<3>(args...);
#ifdef BOUNDSCHECK
    assert( ( (i>=0 && i<M) && (j>=0 && j<N)
              && (k>=0 && k<P) && (l>=0 && l<Q)) && "INDEX OUT OF BOUNDS");
#endif
    return _data[i*N*P*Q+j*P*Q+k*Q+l];
}

template<typename... Args, typename std::enable_if<sizeof...(Args)>=5
                        && sizeof...(Args)==Dimension_t::value && is_arithmetic_pack<Args...>::value,bool>::type =0>
FASTOR_INLINE const T& operator()(Args ... args) const {
    int largs[sizeof...(Args)] = {args...};
    constexpr int DimensionHolder[Dimension] = {Rest...};
    for (int i=0; i<Dimension; ++i) {
        if (largs[i]==-1) largs[i] += DimensionHolder[i];
#ifdef BOUNDSCHECK
        assert( (largs[i]>=0 && largs[i]<DimensionHolder[i]) && "INDEX OUT OF BOUNDS");
#endif
    }
    std::array<int,Dimension> products;
    for (int i=Dimension-1; i>0; --i) {
        int num = DimensionHolder[Dimension-1];
        for (int j=0; j<i-1; ++j) {
            num *= DimensionHolder[Dimension-1-j-1];
        }
        products[i] = num;
    }

    int index = largs[Dimension-1];
    for (int i=Dimension-1; i>0; --i) {
        index += products[i]*largs[Dimension-i-1];
    }
    return _data[index];
}
//----------------------------------------------------------------------------------------------------------//

#endif // SCALAR_INDEXING_CONST_H