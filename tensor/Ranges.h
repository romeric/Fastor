#ifndef RANGES_H
#define RANGES_H

#include "commons/commons.h"
#include <initializer_list>

namespace Fastor {

template<size_t F, size_t L, size_t S=1>
struct iseq {
    static constexpr size_t _first = F;
    static constexpr size_t _last= L;
    static constexpr size_t _step = S;
};


template<int F, int L, int S=1>
struct fseq {
    static constexpr int _first = F;
    static constexpr int _last= L;
    static constexpr int _step = S;

    static constexpr int Size = range_detector<F,L,S>::value;
    constexpr FASTOR_INLINE int size() const {return range_detector<F,L,S>::value;}
};

constexpr fseq<0,-1,1> fall;


struct seq {

    int _first;
    int _last;
    int _step;

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

constexpr int first = 0;
constexpr int last = -1;

static constexpr seq all = seq(0,-1,1);



// Transform sequence with negative indices to positive indices;
template<class Seq, size_t N>
struct to_positive;

template<int F, int L, int S, size_t N>
struct to_positive<fseq<F,L,S>,N> {
    static constexpr int first = (F < 0) ? F + N + 1 : F;
    static constexpr int last = (L < 0) ? L + N + 1 : L;
    using type = fseq<first,last,S>;
};

template<int F, int L, int S, size_t N>
struct to_positive<iseq<F,L,S>,N> {
    static constexpr int first = (F < 0) ? F + N + 1 : F;
    static constexpr int last = (L < 0) ? L + N + 1 : L;
    using type = iseq<first,last,S>;
};


}

#endif // RANGES_H

