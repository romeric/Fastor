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


template <FASTOR_INDEX ... All>
struct Index {
    static const FASTOR_INDEX NoIndices = sizeof...(All);
    static constexpr FASTOR_INDEX _IndexHolder[sizeof...(All)] = {All...};
};

template<FASTOR_INDEX ... All>
constexpr FASTOR_INDEX Index<All...>::_IndexHolder[sizeof...(All)];

}

#endif // INDICIAL_H

