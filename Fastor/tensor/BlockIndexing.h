#ifndef BLOCK_INDEXING_H
#define BLOCK_INDEXING_H


//----------------------------------------------------------------------------------------------------------//
// Block indexing
//----------------------------------------------------------------------------------------------------------//
// Calls scalar indexing so they are fully bounds checked.
template<size_t F, size_t L, size_t S>
FASTOR_INLINE Tensor<T,range_detector<F,L,S>::value> operator()(const iseq<F,L,S>& idx) {

    static_assert(1==dimension_t::value, "INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    Tensor<T,range_detector<F,L,S>::value> out;
    FASTOR_INDEX counter = 0;
    for (FASTOR_INDEX i=F; i<L; i+=S) {
        out(counter) = this->operator()(i);
        counter++;
    }
    return out;
}

template<size_t F0, size_t L0, size_t S0, size_t F1, size_t L1, size_t S1>
FASTOR_INLINE Tensor<T,range_detector<F0,L0,S0>::value,range_detector<F1,L1,S1>::value>
        operator()(iseq<F0,L0,S0>, iseq<F1,L1,S1>)  {

    static_assert(2==dimension_t::value, "INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");

    Tensor<T,range_detector<F0,L0,S0>::value,range_detector<F1,L1,S1>::value> out;
    FASTOR_INDEX counter_i = 0;
    for (FASTOR_INDEX i=F0; i<L0; i+=S0) {
        FASTOR_INDEX counter_j = 0;
        for (FASTOR_INDEX j=F1; j<L1; j+=S1) {
            out(counter_i,counter_j) = this->operator()(i,j);
            counter_j++;
        }
        counter_i++;
    }

    return out;
}

template<size_t F0, size_t L0, size_t S0, size_t F1, size_t L1, size_t S1, size_t F2, size_t L2, size_t S2>
FASTOR_INLINE Tensor<T,range_detector<F0,L0,S0>::value,range_detector<F1,L1,S1>::value,range_detector<F2,L2,S2>::value>
        operator()(iseq<F0,L0,S0>, iseq<F1,L1,S1>, iseq<F2,L2,S2>) const {
    static_assert(3==dimension_t::value, "INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    Tensor<T,range_detector<F0,L0,S0>::value,
            range_detector<F1,L1,S1>::value,
            range_detector<F2,L2,S2>::value> out;
    FASTOR_INDEX counter_i = 0;
    for (FASTOR_INDEX i=F0; i<L0; i+=S0) {
        FASTOR_INDEX counter_j = 0;
        for (FASTOR_INDEX j=F1; j<L1; j+=S1) {
            FASTOR_INDEX counter_k = 0;
            for (FASTOR_INDEX k=F2; k<L2; k+=S2) {
                out(counter_i,counter_j,counter_k) = this->operator()(i,j,k);
                counter_k++;
            }
            counter_j++;
        }
        counter_i++;
    }
    return out;
}

template<size_t F0, size_t L0, size_t S0,
         size_t F1, size_t L1, size_t S1,
         size_t F2, size_t L2, size_t S2,
         size_t F3, size_t L3, size_t S3>
FASTOR_INLINE Tensor<T,range_detector<F0,L0,S0>::value,
        range_detector<F1,L1,S1>::value,
        range_detector<F2,L2,S2>::value,
        range_detector<F3,L3,S3>::value>
        operator ()(iseq<F0,L0,S0>, iseq<F1,L1,S1>,
                    iseq<F2,L2,S2>, iseq<F3,L3,S3>) {

    static_assert(4==dimension_t::value, "INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    Tensor<T,range_detector<F0,L0,S0>::value,
                range_detector<F1,L1,S1>::value,
                range_detector<F2,L2,S2>::value,
                range_detector<F3,L3,S3>::value> out;
    FASTOR_INDEX counter_i = 0;
    for (FASTOR_INDEX i=F0; i<L0; i+=S0) {
        FASTOR_INDEX counter_j = 0;
        for (FASTOR_INDEX j=F1; j<L1; j+=S1) {
            FASTOR_INDEX counter_k = 0;
            for (FASTOR_INDEX k=F2; k<L2; k+=S2) {
                FASTOR_INDEX counter_l = 0;
                for (FASTOR_INDEX l=F3; l<L3; l+=S3) {
                    out(counter_i,counter_j,counter_k,counter_l) = this->operator()(i,j,k,l);
                    counter_l++;
                }
                counter_k++;
            }
            counter_j++;
        }
        counter_i++;
    }
    return out;
}

//----------------------------------------------------------------------------------------------------------//
//----------------------------------------------------------------------------------------------------------//
FASTOR_INLINE TensorViewExpr<Tensor<T,Rest...>,1> operator()(seq _s) {
    static_assert(dimension_t::value==1,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorViewExpr<Tensor<T,Rest...>,1>(*this,_s);
}

FASTOR_INLINE TensorViewExpr<Tensor<T,Rest...>,2> operator()(seq _s0, seq _s1) {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorViewExpr<Tensor<T,Rest...>,2>(*this,_s0,_s1);
}
template<int F0, int L0, int S0>
FASTOR_INLINE TensorViewExpr<Tensor<T,Rest...>,2>
operator()(fseq<F0,L0,S0> _s0, seq _s1) {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorViewExpr<Tensor<T,Rest...>,2>(*this,_s0,_s1);
}
template<int F0, int L0, int S0>
FASTOR_INLINE TensorViewExpr<Tensor<T,Rest...>,2>
operator()(seq _s0, fseq<F0,L0,S0> _s1) {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorViewExpr<Tensor<T,Rest...>,2>(*this,_s0,_s1);
}
template<typename Int, typename std::enable_if<std::is_integral<Int>::value,bool>::type=0>
FASTOR_INLINE TensorViewExpr<Tensor<T,Rest...>,2> operator()(seq _s0, Int num) {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorViewExpr<Tensor<T,Rest...>,2>(*this,_s0,seq(num));
}
template<typename Int, typename std::enable_if<std::is_integral<Int>::value,bool>::type=0>
FASTOR_INLINE TensorViewExpr<Tensor<T,Rest...>,2> operator()(Int num, seq _s1) {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorViewExpr<Tensor<T,Rest...>,2>(*this,seq(num),_s1);
}

template<typename ... Seq, enable_if_t_<!is_arithmetic_pack_v<Seq...> && !is_fixed_sequence_pack_v<Seq...>,bool> = false>
FASTOR_INLINE TensorViewExpr<Tensor<T,Rest...>,sizeof...(Seq)> operator()(Seq ... _seqs) {
    static_assert(dimension_t::value==sizeof...(Seq),"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorViewExpr<Tensor<T,Rest...>,sizeof...(Seq)>(*this, {_seqs...});
}

template<typename ...Fseq, enable_if_t_<is_fixed_sequence_pack_v<Fseq...>,bool> = false>
FASTOR_INLINE TensorFixedViewExprnD<Tensor<T,Rest...>,Fseq...> operator()(Fseq... ) {
    static_assert(dimension_t::value==sizeof...(Fseq),"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorFixedViewExprnD<Tensor<T,Rest...>,Fseq...>(*this);
}

// if fseq == fall - then just return a reference to the tensor
template<int F0, int L0, int S0,
    typename std::enable_if<
    internal::fseq_range_detector<typename to_positive<fseq<F0,L0,S0>,
    get_value<1,Rest...>::value>::type>::value == get_value<1,Rest...>::value,
    bool>::type =0>
FASTOR_INLINE Tensor<T,Rest...>&
operator()(fseq<F0,L0,S0>) {
    static_assert(dimension_t::value==1,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return (*this);
}
// if fseq != fall - return a view
template<int F0, int L0, int S0,
    typename std::enable_if<
    internal::fseq_range_detector<typename to_positive<fseq<F0,L0,S0>,
    get_value<1,Rest...>::value>::type>::value != get_value<1,Rest...>::value,
    bool>::type =0>
FASTOR_INLINE TensorFixedViewExpr1D<Tensor<T,Rest...>,
        typename to_positive<fseq<F0,L0,S0>,pack_prod<Rest...>::value>::type,1>
operator()(fseq<F0,L0,S0>) {
    static_assert(dimension_t::value==1,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorFixedViewExpr1D<Tensor<T,Rest...>,
        typename to_positive<fseq<F0,L0,S0>,pack_prod<Rest...>::value>::type,1>(*this);
}

// if fseq == fall - then just return a reference to the tensor
template<int F0, int L0, int S0, int F1, int L1, int S1,
    typename std::enable_if<
    internal::fseq_range_detector<typename to_positive<fseq<F0,L0,S0>,get_value<1,Rest...>::value>::type>::value == get_value<1,Rest...>::value &&
    internal::fseq_range_detector<typename to_positive<fseq<F1,L1,S1>,get_value<2,Rest...>::value>::type>::value == get_value<2,Rest...>::value,
    bool>::type =0>
FASTOR_INLINE Tensor<T,Rest...>&
operator()(fseq<F0,L0,S0>, fseq<F1,L1,S1>) {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return (*this);
}
// if fseq != fall - return a view
template<int F0, int L0, int S0, int F1, int L1, int S1,
    typename std::enable_if<
    internal::fseq_range_detector<typename to_positive<fseq<F0,L0,S0>,get_value<1,Rest...>::value>::type>::value != get_value<1,Rest...>::value ||
    internal::fseq_range_detector<typename to_positive<fseq<F1,L1,S1>,get_value<2,Rest...>::value>::type>::value != get_value<2,Rest...>::value,
    bool>::type =0>
FASTOR_INLINE TensorFixedViewExpr2D<Tensor<T,Rest...>,
        typename to_positive<fseq<F0,L0,S0>,get_value<1,Rest...>::value>::type,
        typename to_positive<fseq<F1,L1,S1>,get_value<2,Rest...>::value>::type,2>
operator()(fseq<F0,L0,S0>, fseq<F1,L1,S1>) {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorFixedViewExpr2D<Tensor<T,Rest...>,
        typename to_positive<fseq<F0,L0,S0>,get_value<1,Rest...>::value>::type,
        typename to_positive<fseq<F1,L1,S1>,get_value<2,Rest...>::value>::type,2>(*this);
}

template<int F0, int L0, int S0, typename Int, typename std::enable_if<std::is_integral<Int>::value,bool>::type=0>
FASTOR_INLINE TensorViewExpr<Tensor<T,Rest...>,2> operator()(fseq<F0,L0,S0> _s, Int num) {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorViewExpr<Tensor<T,Rest...>,2>(*this,seq(_s),seq(num));
}
template<int F0, int L0, int S0, typename Int,
    typename std::enable_if<std::is_integral<Int>::value,bool>::type=0>
FASTOR_INLINE TensorViewExpr<Tensor<T,Rest...>,2> operator()(Int num, fseq<F0,L0,S0> _s) {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorViewExpr<Tensor<T,Rest...>,2>(*this,seq(num),seq(_s));
}

// Selecting a row and returning a TensorMap - does not seem to speed up the code
//----------------------------------------------------------------------------------------------------------//
// template<int F0, int L0, int S0, typename Int,
//     typename std::enable_if<std::is_integral<Int>::value &&
//     internal::fseq_range_detector<typename to_positive<fseq<F0,L0,S0>,
//     get_value<2,Rest...>::value>::type>::value != get_value<2,Rest...>::value,bool>::type=0>
// FASTOR_INLINE TensorViewExpr<Tensor<T,Rest...>,2> operator()(Int num, fseq<F0,L0,S0> _s) {
//     static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
//     return TensorViewExpr<Tensor<T,Rest...>,2>(*this,seq(num),seq(_s));
// }
// // Selecting a row from a 2D tensor returns a TensorMap
// template<typename Int, int F0, int L0, int S0,
//     typename std::enable_if<std::is_integral<Int>::value &&
//     internal::fseq_range_detector<typename to_positive<fseq<F0,L0,S0>,
//     get_value<2,Rest...>::value>::type>::value == get_value<2,Rest...>::value,bool>::type=0>
// FASTOR_INLINE TensorMap<T,get_value<2,Rest...>::value> operator()(Int num, fseq<F0,L0,S0> _s) {
//     static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
//     constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
//     return TensorMap<T,N>(&_data[num*N]);
// }
//----------------------------------------------------------------------------------------------------------//

template<typename Int, size_t N, typename std::enable_if<std::is_integral<Int>::value,bool>::type=0>
FASTOR_INLINE TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,N>,1> operator()(const Tensor<Int,N> &_it) {
    static_assert(dimension_t::value==1,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,N>,1>(*this,_it);
}

template<typename Int, size_t ... IterSizes, typename std::enable_if<std::is_integral<Int>::value,bool>::type=0>
FASTOR_INLINE TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,sizeof...(Rest)>
operator()(const Tensor<Int,IterSizes...> &_it) {
    static_assert(dimension_t::value==sizeof...(IterSizes),"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,sizeof...(Rest)>(*this,_it);
}

template<typename Int0, typename Int1, size_t M, size_t N,
    typename std::enable_if<std::is_integral<Int0>::value && std::is_integral<Int1>::value,bool>::type=0>
FASTOR_INLINE TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int0,M,N>,2>
operator()(const Tensor<Int0,M> &_it0, const Tensor<Int1,N> &_it1) {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    Tensor<Int0,M,N> tmp_it;
    constexpr int NCols = get_value<2,Rest...>::value;
    for (FASTOR_INDEX i = 0; i<M; ++i) {
        for (FASTOR_INDEX j=0; j<N; ++j) {
            tmp_it(i,j) = _it0(i)*NCols + _it1(j);
        }
    }
    return TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int0,M,N>,2>(*this,tmp_it);
}

template<typename Int0, typename Int1, size_t M,
    typename std::enable_if<std::is_integral<Int0>::value && std::is_integral<Int1>::value,bool>::type=0>
FASTOR_INLINE TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int0,M,1>,2>
operator()(const Tensor<Int0,M> &_it0, Int1 num) {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    Tensor<Int0,M,1> tmp_it;
    constexpr int NCols = get_value<2,Rest...>::value;
    for (FASTOR_INDEX i = 0; i<M; ++i) {
        tmp_it(i,0) = _it0(i)*NCols + num;
    }
    return TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int0,M,1>,2>(*this,tmp_it);
}

template<typename Int0, typename Int1, size_t M,
    typename std::enable_if<std::is_integral<Int0>::value && std::is_integral<Int1>::value,bool>::type=0>
FASTOR_INLINE TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int0,M,1>,2>
operator()(Int1 num, const Tensor<Int0,M> &_it0) {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    Tensor<Int0,M,1> tmp_it;
    constexpr int NCols = get_value<2,Rest...>::value;
    for (FASTOR_INDEX i = 0; i<M; ++i) {
        tmp_it(i,0) = num*NCols + _it0(i);
    }
    return TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int0,M,1>,2>(*this,tmp_it);
}

template<typename Int, size_t M, int F, int L, int S,
    typename std::enable_if<std::is_integral<Int>::value,bool>::type=0>
FASTOR_INLINE TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,M,
    to_positive<fseq<F,L,S>,get_value<2,Rest...>::value>::type::Size>,2>
operator()(const Tensor<Int,M> &_it0, fseq<F,L,S>) {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    constexpr int NCols = get_value<2,Rest...>::value;
    using _seq = typename to_positive<fseq<F,L,S>,NCols>::type;
    constexpr int ColSize = _seq::Size;
    Tensor<Int,M,ColSize> tmp_it;
    for (FASTOR_INDEX i = 0; i<M; ++i) {
        for (FASTOR_INDEX j=0; j<ColSize; ++j) {
            tmp_it(i,j) = _it0(i)*NCols + _seq::_step*j + _seq::_first;
        }
    }
    return TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,M,
        to_positive<fseq<F,L,S>,get_value<2,Rest...>::value>::type::Size>,2> (*this,tmp_it);
}

template<typename Int, size_t N, int F, int L, int S,
    typename std::enable_if<std::is_integral<Int>::value,bool>::type=0>
FASTOR_INLINE TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,
    to_positive<fseq<F,L,S>,get_value<1,Rest...>::value>::type::Size,N>,2>
operator()(fseq<F,L,S>, const Tensor<Int,N> &_it0) {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    constexpr int NRows = get_value<1,Rest...>::value;
    constexpr int NCols = get_value<2,Rest...>::value;
    using _seq = typename to_positive<fseq<F,L,S>,NRows>::type;
    constexpr int RowSize = _seq::Size;
    Tensor<Int,RowSize,N> tmp_it;
    for (FASTOR_INDEX i = 0; i<RowSize; ++i) {
        for (FASTOR_INDEX j=0; j<N; ++j) {
            tmp_it(i,j) = (_seq::_step*i + _seq::_first)*NCols + _it0(j);
        }
    }
    return TensorRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,
        to_positive<fseq<F,L,S>,get_value<1,Rest...>::value>::type::Size,N>,2> (*this,tmp_it);
}



//----------------------------------------------------------------------------------------------------------//
// Filter views
FASTOR_INLINE TensorFilterViewExpr<Tensor<T,Rest...>,Tensor<bool,Rest...>,sizeof...(Rest)>
operator()(const Tensor<bool,Rest...> &_fl) {
    return TensorFilterViewExpr<Tensor<T,Rest...>,Tensor<bool,Rest...>,sizeof...(Rest)>(*this,_fl);
}
FASTOR_INLINE TensorFilterViewExpr<Tensor<T,Rest...>,TensorMap<bool,Rest...>,sizeof...(Rest)>
operator()(const TensorMap<bool,Rest...> &_fl) {
    return TensorFilterViewExpr<Tensor<T,Rest...>,TensorMap<bool,Rest...>,sizeof...(Rest)>(*this,_fl);
}
//----------------------------------------------------------------------------------------------------------//

FASTOR_INLINE TensorConstViewExpr<Tensor<T,Rest...>,1> operator()(seq _s) const {
    static_assert(dimension_t::value==1,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorConstViewExpr<Tensor<T,Rest...>,1>(*this,_s);
}

FASTOR_INLINE TensorConstViewExpr<Tensor<T,Rest...>,2> operator()(seq _s0, seq _s1) const {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorConstViewExpr<Tensor<T,Rest...>,2>(*this,_s0,_s1);
}
template<typename Int, typename std::enable_if<std::is_integral<Int>::value,bool>::type=0>
FASTOR_INLINE TensorConstViewExpr<Tensor<T,Rest...>,2> operator()(seq _s0, Int num) const {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorConstViewExpr<Tensor<T,Rest...>,2>(*this,_s0,seq(num));
}
template<typename Int, typename std::enable_if<std::is_integral<Int>::value,bool>::type=0>
FASTOR_INLINE TensorConstViewExpr<Tensor<T,Rest...>,2> operator()(Int num, seq _s1) const {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorConstViewExpr<Tensor<T,Rest...>,2>(*this,seq(num),_s1);
}

template<typename ... Seq, enable_if_t_<!is_arithmetic_pack_v<Seq...> && !is_fixed_sequence_pack_v<Seq...>,bool> = false>
FASTOR_INLINE TensorConstViewExpr<Tensor<T,Rest...>,sizeof...(Seq)> operator()(Seq ... _seqs) const {
    static_assert(dimension_t::value==sizeof...(Seq),"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorConstViewExpr<Tensor<T,Rest...>,sizeof...(Seq)>(*this, {_seqs...});
}

template<typename ...Fseq, enable_if_t_<is_fixed_sequence_pack_v<Fseq...>,bool> = false>
FASTOR_INLINE TensorConstFixedViewExprnD<Tensor<T,Rest...>,Fseq...> operator()(Fseq... ) const {
    static_assert(dimension_t::value==sizeof...(Fseq),"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorConstFixedViewExprnD<Tensor<T,Rest...>,Fseq...>(*this);
}

template<int F0, int L0, int S0>
FASTOR_INLINE TensorConstFixedViewExpr1D<Tensor<T,Rest...>,
        typename to_positive<fseq<F0,L0,S0>,pack_prod<Rest...>::value>::type,1> operator()(fseq<F0,L0,S0>) const {
    static_assert(dimension_t::value==1,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorConstFixedViewExpr1D<Tensor<T,Rest...>,
        typename to_positive<fseq<F0,L0,S0>,pack_prod<Rest...>::value>::type,1>(*this);
}

template<int F0, int L0, int S0, int F1, int L1, int S1>
FASTOR_INLINE TensorConstFixedViewExpr2D<Tensor<T,Rest...>,
        typename to_positive<fseq<F0,L0,S0>,get_value<1,Rest...>::value>::type,
        typename to_positive<fseq<F1,L1,S1>,get_value<2,Rest...>::value>::type,2>
operator()(fseq<F0,L0,S0>, fseq<F1,L1,S1>) const {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorConstFixedViewExpr2D<Tensor<T,Rest...>,
        typename to_positive<fseq<F0,L0,S0>,get_value<1,Rest...>::value>::type,
        typename to_positive<fseq<F1,L1,S1>,get_value<2,Rest...>::value>::type,2>(*this);
}

template<int F0, int L0, int S0, typename Int, typename std::enable_if<std::is_integral<Int>::value,bool>::type=0>
FASTOR_INLINE TensorConstViewExpr<Tensor<T,Rest...>,2> operator()(fseq<F0,L0,S0> _s, Int num) const {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorConstViewExpr<Tensor<T,Rest...>,2>(*this,seq(_s),seq(num));
}

template<int F0, int L0, int S0, typename Int, typename std::enable_if<std::is_integral<Int>::value,bool>::type=0>
FASTOR_INLINE TensorConstViewExpr<Tensor<T,Rest...>,2> operator()(Int num, fseq<F0,L0,S0> _s) const {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorConstViewExpr<Tensor<T,Rest...>,2>(*this,seq(num),seq(_s));
}

template<typename Int, size_t N, typename std::enable_if<std::is_integral<Int>::value,bool>::type=0>
FASTOR_INLINE TensorConstRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,N>,1>
operator()(const Tensor<Int,N> &_it) const {
    static_assert(dimension_t::value==1,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorConstRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,N>,1>(*this,_it);
}

template<typename Int, size_t ... IterSizes, typename std::enable_if<std::is_integral<Int>::value,bool>::type=0>
FASTOR_INLINE TensorConstRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,sizeof...(Rest)>
operator()(const Tensor<Int,IterSizes...> &_it) const {
    static_assert(dimension_t::value==sizeof...(IterSizes),"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    return TensorConstRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,IterSizes...>,sizeof...(Rest)>(*this,_it);
}

template<typename Int0, typename Int1, size_t M, size_t N,
    typename std::enable_if<std::is_integral<Int0>::value && std::is_integral<Int1>::value,bool>::type=0>
FASTOR_INLINE TensorConstRandomViewExpr<Tensor<T,Rest...>,Tensor<Int0,M,N>,2>
operator()(const Tensor<Int0,M> &_it0, const Tensor<Int1,N> &_it1) const {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    Tensor<Int0,M,N> tmp_it;
    constexpr int NCols = get_value<2,Rest...>::value;
    for (FASTOR_INDEX i = 0; i<M; ++i) {
        for (FASTOR_INDEX j=0; j<N; ++j) {
            tmp_it(i,j) = _it0(i)*NCols + _it1(j);
        }
    }
    return TensorConstRandomViewExpr<Tensor<T,Rest...>,Tensor<Int0,M,N>,2>(*this,tmp_it);
}

template<typename Int0, typename Int1, size_t M,
    typename std::enable_if<std::is_integral<Int0>::value && std::is_integral<Int1>::value,bool>::type=0>
FASTOR_INLINE TensorConstRandomViewExpr<Tensor<T,Rest...>,Tensor<Int0,M,1>,2>
operator()(const Tensor<Int0,M> &_it0, Int1 num) const {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    Tensor<Int0,M,1> tmp_it;
    constexpr int NCols = get_value<2,Rest...>::value;
    for (FASTOR_INDEX i = 0; i<M; ++i) {
        tmp_it(i,0) = _it0(i)*NCols + num;
    }
    return TensorConstRandomViewExpr<Tensor<T,Rest...>,Tensor<Int0,M,1>,2>(*this,tmp_it);
}

template<typename Int0, typename Int1, size_t M,
    typename std::enable_if<std::is_integral<Int0>::value && std::is_integral<Int1>::value,bool>::type=0>
FASTOR_INLINE TensorConstRandomViewExpr<Tensor<T,Rest...>,Tensor<Int0,M,1>,2>
operator()(Int1 num, const Tensor<Int0,M> &_it0) const {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    Tensor<Int0,M,1> tmp_it;
    constexpr int NCols = get_value<2,Rest...>::value;
    for (FASTOR_INDEX i = 0; i<M; ++i) {
        tmp_it(i,0) = num*NCols + _it0(i);
    }
    return TensorConstRandomViewExpr<Tensor<T,Rest...>,Tensor<Int0,M,1>,2>(*this,tmp_it);
}

template<typename Int, size_t M, int F, int L, int S,
    typename std::enable_if<std::is_integral<Int>::value,bool>::type=0>
FASTOR_INLINE TensorConstRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,M,
    to_positive<fseq<F,L,S>,get_value<2,Rest...>::value>::type::Size>,2>
operator()(const Tensor<Int,M> &_it0, fseq<F,L,S>) const {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    constexpr int NCols = get_value<2,Rest...>::value;
    using _seq = typename to_positive<fseq<F,L,S>,NCols>::type;
    constexpr int ColSize = _seq::Size;
    Tensor<Int,M,ColSize> tmp_it;
    for (FASTOR_INDEX i = 0; i<M; ++i) {
        for (FASTOR_INDEX j=0; j<ColSize; ++j) {
            tmp_it(i,j) = _it0(i)*NCols + _seq::_step*j + _seq::_first;
        }
    }
    return TensorConstRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,M,
        to_positive<fseq<F,L,S>,get_value<2,Rest...>::value>::type::Size>,2> (*this,tmp_it);
}

template<typename Int, size_t N, int F, int L, int S,
    typename std::enable_if<std::is_integral<Int>::value,bool>::type=0>
FASTOR_INLINE TensorConstRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,
    to_positive<fseq<F,L,S>,get_value<1,Rest...>::value>::type::Size,N>,2>
operator()(fseq<F,L,S>, const Tensor<Int,N> &_it0) const {
    static_assert(dimension_t::value==2,"INDEXING TENSOR WITH INCORRECT NUMBER OF ARGUMENTS");
    constexpr int NRows = get_value<1,Rest...>::value;
    constexpr int NCols = get_value<2,Rest...>::value;
    using _seq = typename to_positive<fseq<F,L,S>,NRows>::type;
    constexpr int RowSize = _seq::Size;
    Tensor<Int,RowSize,N> tmp_it;
    for (FASTOR_INDEX i = 0; i<RowSize; ++i) {
        for (FASTOR_INDEX j=0; j<N; ++j) {
            tmp_it(i,j) = (_seq::_step*i + _seq::_first)*NCols + _it0(j);
        }
    }
    return TensorConstRandomViewExpr<Tensor<T,Rest...>,Tensor<Int,
        to_positive<fseq<F,L,S>,get_value<1,Rest...>::value>::type::Size,N>,2> (*this,tmp_it);
}
//----------------------------------------------------------------------------------------------------------//
//----------------------------------------------------------------------------------------------------------//




#endif // BLOCK_INDEXING_H
