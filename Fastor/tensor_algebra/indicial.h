#ifndef INDICIAL_H
#define INDICIAL_H

#include "Fastor/config/config.h"
#include "Fastor/meta/einsum_meta.h"
#include <array>

namespace Fastor {

//-----------------------------------------------------------------------------------------------------------//
template <FASTOR_INDEX ... All>
struct Index {
    static constexpr FASTOR_INDEX Size = sizeof...(All);
    static constexpr std::array<FASTOR_INDEX,sizeof...(All)> values = {All...};
};

template<FASTOR_INDEX ... All>
constexpr FASTOR_INDEX Index<All...>::Size;

template<FASTOR_INDEX ... All>
constexpr std::array<FASTOR_INDEX,sizeof...(All)> Index<All...>::values;

namespace internal {
template<FASTOR_INDEX N, class seq>
struct make_index_impl;
template<FASTOR_INDEX N, size_t ... ss>
struct make_index_impl<N,std_ext::index_sequence<ss...>> {
    using type = Index<ss...>;
};
} // internal

template<FASTOR_INDEX N>
struct make_index {
    using type = typename internal::make_index_impl<N,typename std_ext::make_index_sequence<N>::type>::type;
};

template<FASTOR_INDEX N>
using make_index_t = typename make_index<N>::type;


template<FASTOR_INDEX ... All>
struct OIndex : public Index<All...> {
    static constexpr FASTOR_INDEX Size = sizeof...(All);
    using parent_type = Index<All...>;
};

template<FASTOR_INDEX ... All>
constexpr FASTOR_INDEX OIndex<All...>::Size;

template<class Idx>
struct to_oindex;
template<FASTOR_INDEX ... All>
struct to_oindex<Index<All...>>
{
    using type = OIndex<All...>;
};

template<class Idx>
using to_oindex_t = typename to_oindex<Idx>::type;

//-----------------------------------------------------------------------------------------------------------//

} // end of namespace Fastor

#endif // INDICIAL_H

