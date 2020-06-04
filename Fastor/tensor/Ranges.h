#ifndef RANGES_H
#define RANGES_H

#include "Fastor/config/config.h"
#include "Fastor/meta/meta.h"
#include <initializer_list>

namespace Fastor {

// range detector for fseq
//----------------------------------------------------------------------------------------------------------//
template<int first, int last, int step>
struct range_detector {
    static constexpr int range = last - first;
    static constexpr int value = range % step==0 ? range/step : range/step+1;
};

namespace internal {
template<class Seq>
struct fseq_range_detector;

template<template<int,int,int> class Seq, int first, int last, int step>
struct fseq_range_detector<Seq<first,last,step>> {
    static constexpr int range = last - first;
    static constexpr int value = range % step==0 ? range/step : range/step+1;
};
} // internal
//----------------------------------------------------------------------------------------------------------//




// Immediate sequence
//----------------------------------------------------------------------------------------------------------//
template<size_t F, size_t L, size_t S=1>
struct iseq {
    static constexpr size_t _first = F;
    static constexpr size_t _last= L;
    static constexpr size_t _step = S;
};
//----------------------------------------------------------------------------------------------------------//


// Fixed sequence
//----------------------------------------------------------------------------------------------------------//
template<int F, int L, int S=1>
struct fseq {
    static constexpr int _first = F;
    static constexpr int _last= L;
    static constexpr int _step = S;

    static constexpr int Size = range_detector<F,L,S>::value;
    constexpr FASTOR_INLINE int size() const {return range_detector<F,L,S>::value;}
};

static constexpr fseq<0,-1,1> fall;

template<int F>
static constexpr fseq<F,F+1,1> fix{};

static constexpr fseq<0 ,1 ,1> ffirst;
static constexpr fseq<-1,-1,1> flast;
//----------------------------------------------------------------------------------------------------------//


// Dynamic sequence
//----------------------------------------------------------------------------------------------------------//
struct seq {

    int _first;
    int _last;
    int _step = 1;

    constexpr FASTOR_INLINE seq(int _f, int _l, int _s=1) : _first(_f), _last(_l), _step(_s) {}
    constexpr FASTOR_INLINE seq(int num) : _first(num), _last(num+1), _step(1) {}

    template<int F, int L, int S=1>
    constexpr FASTOR_INLINE seq(fseq<F,L,S>) : _first(F), _last(L), _step(S) {}

    // Do not allow construction of seq using std::initializer_list, as it happens
    // implicitly. Overloading operator() with std::initializer_list should imply
    // TensorRandomView, not TensorView
    template<typename T>
    constexpr FASTOR_INLINE seq(std::initializer_list<T> _s1) = delete;

    // Do not provide this overload as it is meaningless [iseq stands for immediate evaluation]
    // template<size_t F, size_t L, size_t S=1>
    // constexpr FASTOR_INLINE seq(iseq<F,L,S>) : _first(F), _last(L), _step(S) {}

    FASTOR_INLINE int size() const {
        int range = _last - _first;
        return range % _step==0 ? range/_step : range/_step+1;
    }

    constexpr FASTOR_INLINE bool operator==(seq other) const {
        return (_first==other._first && _last==other._last && _step==other._step) ? true : false;
    }
    constexpr FASTOR_INLINE bool operator!=(seq other) const {
        return (_first!=other._first || _last!=other._last || _step!=other._step) ? true : false;
    }

};

static constexpr int first = 0;
static constexpr int last = -1;

// static constexpr seq all = seq(0,-1,1);
// why not this?
static constexpr fseq<0,-1,1> all;
//----------------------------------------------------------------------------------------------------------//


// traits
//----------------------------------------------------------------------------------------------------------//
template<typename T>
struct is_fixed_sequence {
    static constexpr bool value = false;
};
template<int F, int L, int S>
struct is_fixed_sequence<fseq<F,L,S>> {
    static constexpr bool value = true;
};
template<int F, int L>
struct is_fixed_sequence<fseq<F,L,1>> {
    static constexpr bool value = true;
};

template<typename T>
static constexpr bool is_fixed_sequence_v = is_fixed_sequence<T>::value;

template<typename ... T>
struct is_fixed_sequence_pack;
template<typename T, typename ... Ts>
struct is_fixed_sequence_pack<T,Ts...> {
    static constexpr bool value = is_fixed_sequence<T>::value && is_fixed_sequence_pack<Ts...>::value;
};
template<typename T>
struct is_fixed_sequence_pack<T> {
    static constexpr bool value = is_fixed_sequence<T>::value;
};

template<typename ... Ts>
static constexpr bool is_fixed_sequence_pack_v = is_fixed_sequence_pack<Ts...>::value;
//----------------------------------------------------------------------------------------------------------//



// Transform sequence with negative indices to positive indices;
//----------------------------------------------------------------------------------------------------------//
template<class Seq, int N>
struct to_positive;

template<int F, int L, int S, int N>
struct to_positive<fseq<F,L,S>,N> {
    // Same logic as seq used in the constructor of dynamic tensor views
    static constexpr int _first = (L==0 && F==-1) ? N-1 : ( L < 0 && F < 0 ? F + N + 1 : F);
    static constexpr int _last  = (L < 0 && F >=0) ? L + N + 1 : ( (L==0 && F==-1) ? N : ( L < 0 && F < 0 ? L + N + 1 : L) );
    using type = fseq<_first,_last,S>;
};

template<int F, int L, int S, int N>
struct to_positive<iseq<F,L,S>,N> {
    // Same logic as seq used in the constructor of dynamic tensor views
    static constexpr int _first = (L==0 && F==-1) ? N-1 : ( L < 0 && F < 0 ? F + N + 1 : F);
    static constexpr int _last  = (L < 0 && F >=0) ? L + N + 1 : ( (L==0 && F==-1) ? N : ( L < 0 && F < 0 ? L + N + 1 : L) );
    using type = iseq<_first,_last,S>;
};

template<class Seq, int N>
using to_positive_t = typename to_positive<Seq,N>::type;
//----------------------------------------------------------------------------------------------------------//


//----------------------------------------------------------------------------------------------------------//
template<typename Derived, class Seq, typename ... Fseqs>
struct get_fixed_sequence_pack_dimensions;

template<template<typename,size_t...> class Derived, typename T, size_t ...Rest, size_t... ss, typename ... Fseqs>
struct get_fixed_sequence_pack_dimensions<Derived<T, Rest...>, std_ext::index_sequence<ss...>, Fseqs...>{
    static constexpr std::array<int,sizeof...(Fseqs)> dims = { internal::fseq_range_detector<to_positive_t<Fseqs,Rest>>::value... };
    // using type = Derived<T,internal::fseq_range_detector<Fseqs>::value...>;
    using type = Derived<T,internal::fseq_range_detector<to_positive_t<Fseqs,Rest>>::value...>;
    static constexpr size_t Size = pack_prod<dims[ss]...>::value;
};

template<template<typename,size_t...> class Derived, typename T, size_t ...Rest, size_t... ss, typename ... Fseqs>
constexpr std::array<int,sizeof...(Fseqs)> get_fixed_sequence_pack_dimensions<Derived<T, Rest...>, std_ext::index_sequence<ss...>, Fseqs...>::dims;
//----------------------------------------------------------------------------------------------------------//



//----------------------------------------------------------------------------------------------------------//
template<size_t F0, size_t L0, size_t S0, size_t F1, size_t L1, size_t S1, size_t Ncol, class Y>
struct ravel_2d_indices;

template<size_t F0, size_t L0, size_t S0, size_t F1, size_t L1, size_t S1, size_t Ncol, size_t ... ss>
struct ravel_2d_indices<F0,L0,S0,F1,L1,S1,Ncol,std_ext::index_sequence<ss...>> {
    static constexpr size_t size_1 = range_detector<F1,L1,S1>::value;
    static constexpr std::array<size_t,sizeof...(ss)> idx = {(S0*(ss/size_1)*Ncol + S1*(ss%size_1) + F0*Ncol + F1)...};
};
template<size_t F0, size_t L0, size_t S0, size_t F1, size_t L1, size_t S1, size_t Ncol, size_t ... ss>
constexpr std::array<size_t,sizeof...(ss)>
ravel_2d_indices<F0,L0,S0,F1,L1,S1,Ncol,std_ext::index_sequence<ss...>>::idx;
//----------------------------------------------------------------------------------------------------------//


}

#endif // RANGES_H

