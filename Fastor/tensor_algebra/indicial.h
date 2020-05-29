#ifndef INDICIAL_H
#define INDICIAL_H

#include "Fastor/commons/commons.h"
#include "Fastor/meta/einsum_meta.h"
#include <array>

namespace Fastor {

template <FASTOR_INDEX ... All>
struct Index {
    static constexpr FASTOR_INDEX Size = sizeof...(All);
    static constexpr std::array<FASTOR_INDEX,sizeof...(All)> values = {All...};
};

template<FASTOR_INDEX ... All>
constexpr FASTOR_INDEX Index<All...>::Size;

template<FASTOR_INDEX ... All>
constexpr std::array<FASTOR_INDEX,sizeof...(All)> Index<All...>::values;


template<FASTOR_INDEX N>
struct makeIndex {
    using type = typename concat_<typename makeIndex<N-1>::type,Index<N-1>>::type;
};
template<>
struct makeIndex<0> {
    using type = Index<0>;
};


template<FASTOR_INDEX ... All>
struct OIndex : public Index<All...> {
    static constexpr FASTOR_INDEX Size = sizeof...(All);
    using parent_type = Index<All...>;
};

template<FASTOR_INDEX ... All>
constexpr FASTOR_INDEX OIndex<All...>::Size;

} // end of namespace Fastor

#endif // INDICIAL_H

