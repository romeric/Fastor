#ifndef TENSOR_META_H
#define TENSOR_META_H

#include "Fastor/config/config.h"
#include "Fastor/meta/meta.h"

namespace Fastor {

//----------------------------------------------------------------------------------------------------------//
// UpLoType
namespace UpLoType {
struct General {};
struct Lower {};
struct UniLower {};
struct StrictlyLower {};
struct Upper {};
struct UniUpper {};
struct StrictlyUpper {};
struct Diagonal {};
struct BiDiagonal {};
struct TriDiagonal {};
struct BlockDiagonal {};
struct Symmetric {};
struct SymmetricPositiveDefinite {};
}
//----------------------------------------------------------------------------------------------------------//


//----------------------------------------------------------------------------------------------------------//
template <size_t...T>
struct is_unique : std::integral_constant<bool, true> {};

template <size_t T, size_t U, size_t... VV>
struct is_unique<T, U, VV...> : std::integral_constant<bool, T != U && is_unique<T, VV...>::value> {};

template <size_t...T>
struct no_of_unique : std::integral_constant<size_t, 0> {};

template <size_t T, size_t... UU>
struct no_of_unique<T, UU...> : std::integral_constant<size_t, is_unique<T, UU...>::value + no_of_unique<UU...>::value> {};



//----------------------------------------------------------------------------------------------------------//
// Note that Intel's ICC 2017 does not support make_index_sequence and the following
// version also seems faster than c++14's built-in (for clang) on Linux systems
namespace std_ext  // back port to c++11
{
    template <size_t... Ints>
    struct index_sequence
    {
        using type = index_sequence;
        using value_type = size_t;
        static constexpr std::size_t size() { return sizeof...(Ints); }
    };

    // --------------------------------------------------------------

    template <class Sequence1, class Sequence2>
    struct _merge_and_renumber;

    template <size_t... I1, size_t... I2>
    struct _merge_and_renumber<index_sequence<I1...>, index_sequence<I2...>>
      : index_sequence<I1..., (sizeof...(I1)+I2)...>
    { };

    // --------------------------------------------------------------

    template <size_t N>
    struct make_index_sequence
      : _merge_and_renumber<typename make_index_sequence<N/2>::type,
                            typename make_index_sequence<N - N/2>::type>
    { };

    template<> struct make_index_sequence<0> : index_sequence<> { };
    template<> struct make_index_sequence<1> : index_sequence<0> { };
}
//----------------------------------------------------------------------------------------------------------//




//----------------------------------------------------------------------------------------------------------//
template <class... > struct typelist { };
template <class T, T ... Vals>
using typelist_c = typelist<std::integral_constant<T, Vals>...>;

template <class... > struct concat;
template <> struct concat<> { using type = typelist<>; };
template <class... Ts> struct concat<typelist<Ts...>> { using type = typelist<Ts...>; };
template <class... Ts, class... Us, class... Args>
struct concat<typelist<Ts...>, typelist<Us...>, Args...> : concat<typelist<Ts..., Us...>, Args...> { };

template <class T, class TL> struct filter_out;
template <class T, class... Ts> struct filter_out<T, typelist<Ts...>>
    : concat< conditional_t_<is_same_v_<T, Ts>, typelist<>, typelist<Ts>>...> { };

template <class T, class TL>
using filter_out_t = typename filter_out<T, TL>::type;



template <class >
struct uniq;

template <class TL>
using uniq_t = typename uniq<TL>::type;

template <>
struct uniq<typelist<>> {
    using type = typelist<>;
};

template <class T, class... Ts>
struct uniq<typelist<T, Ts...>>
    : concat<typelist<T>, uniq_t<filter_out_t<T, typelist<Ts...>>>>
{ };

template <size_t N>
using size_t_ = std::integral_constant<size_t, N>;

template <class > struct length;
template <class T> using length_t = typename length<T>::type;
template <class... Ts>
struct length<typelist<Ts...>>
: size_t_<sizeof...(Ts)>
{ };

template <size_t... Ns>
using no_of_uniques = length_t<uniq_t<typelist<size_t_<Ns>...>>>;
//----------------------------------------------------------------------------------------------------------//


////////////////
template <class T, template <T...> class Z>
struct quote_c {
    template <class... Ts>
    using apply = Z<Ts::value...>;
};

template <class MFC, class TL>
struct apply_typelist;

template <class MFC, class TL>
using apply_typelist_t = typename apply_typelist<MFC, TL>::type;


template <class MFC, class... Ts>
struct apply_typelist<MFC, typelist<Ts...>> {
    using type = typename MFC::template apply<Ts...>;
};
//----------------------------------------------------------------------------------------------------------//



// Check if indices appear more than twice [for Einstein summation]
//----------------------------------------------------------------------------------------------------------//
namespace useless {

template <typename T>
constexpr const T& ct_max(T const& t1, T const& t2) {
    return t1 < t2 ? t2 : t1;
}


template <size_t S, size_t... Sizes>
struct count__;

template <size_t S>
struct count__<S> {
    static constexpr size_t value = std::integral_constant<size_t, 0>::value;
};

template <size_t S1, size_t... Sizes>
struct count__<S1, S1, Sizes...> {
    static constexpr size_t value = std::integral_constant<size_t, 1 + count__<S1, Sizes...>::value>::value;
};

template <size_t S1, size_t S2, size_t... Sizes>
struct count__<S1, S2, Sizes...> {
   static constexpr size_t value = count__<S1, Sizes...>::value;
};

template <size_t...all>
struct max_count;

template <>
struct max_count<> {
    static constexpr size_t value = std::integral_constant<size_t, 0>::value;
};

template <size_t S, size_t... Sizes>
struct max_count<S, Sizes...> {
    static constexpr size_t value = std::integral_constant<size_t, ct_max(1 + count__<S, Sizes...>::value,
                                            max_count<Sizes...>::value)>::value;
};

} // useless

template <size_t... Sizes>
struct no_more_than_two {
    static constexpr size_t value = std::integral_constant<bool, useless::max_count<Sizes...>::value <= 2>::value;
};
//----------------------------------------------------------------------------------------------------------//


}

#endif // TENSOR_META_H
