#ifndef INDICIAL_H
#define INDICIAL_H

#include "commons/commons.h"

namespace Fastor {


enum {
    Ind_I,
    Ind_J,
    Ind_K,
    Ind_L,
    Ind_M,
    Ind_N,
    Ind_O,
    Ind_P,
    Ind_Q,
    Ind_R,
    Ind_S,
    Int_T,
    Ind_U,
    Ind_V,
    Ind_W,
    Ind_X,
    Ind_Y,
    Ind_Z
};


//template <size_t ... All>
//struct Index {
//    static const size_t NoIndices = sizeof...(All);
//    static constexpr size_t _IndexHolder[sizeof...(All)] = {All...};
//};

//template<size_t ... All>
//constexpr size_t Index<All...>::_IndexHolder[sizeof...(All)];


template <FASTOR_INDEX ... All>
struct Index {
    static const FASTOR_INDEX NoIndices = sizeof...(All);
    static constexpr FASTOR_INDEX _IndexHolder[sizeof...(All)] = {All...};
};

template<FASTOR_INDEX ... All>
constexpr FASTOR_INDEX Index<All...>::_IndexHolder[sizeof...(All)];

//template <int ... All>
//struct Index {
//    static const int NoIndices = sizeof...(All);
//    static constexpr int _IndexHolder[sizeof...(All)] = {All...};
//};

//template<int ... All>
//constexpr int Index<All...>::_IndexHolder[sizeof...(All)];

}

#endif // INDICIAL_H

