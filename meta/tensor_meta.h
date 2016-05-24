#ifndef TENSOR_META_H
#define TENSOR_META_H

namespace Fastor {

template<typename> struct stride_finder;
template<> struct stride_finder<double> {
    static const size_t Stride = 4;
};
template<> struct stride_finder<float> {
    static const size_t Stride = 8;
};

template<size_t Idx, size_t ... Rest>
struct get_value;
template<size_t Idx, size_t First, size_t ... Rest>
struct get_value<Idx, First, Rest...> {
    static const size_t value = get_value<Idx-1, Rest...>::value;
};
template<size_t First, size_t ... Rest>
struct get_value<1,First,Rest...> {
    static const size_t value = First;
};

template <size_t N, typename... Args>
auto get_index(Args&&... as) -> decltype(std::get<N>(std::forward_as_tuple(std::forward<Args>(as)...)))
{
    return std::get<N>(std::forward_as_tuple(std::forward<Args>(as)...));
}

template<size_t...> struct add;
template<size_t Head, size_t ...Rest>
struct add<Head, Rest...> {
    static const size_t value = Head+add<Rest...>::value;
};
template<>
struct add<> {
    static const size_t value = 0;
};

template<size_t...> struct prod;
template<size_t Head, size_t ...Rest>
struct prod<Head, Rest...> {
    static const size_t value = Head*prod<Rest...>::value;
};
template<>
struct prod<> {
    static const size_t value = 1;
};

template<size_t Idx, size_t ... Rest>
struct prod_nel;
template<size_t Idx, size_t First, size_t ... Rest>
struct prod_nel<Idx, First, Rest...> {
    static const size_t value = prod_nel<Idx-1, Rest...>::value;
};
template<size_t First, size_t ... Rest>
struct prod_nel<1, First, Rest...> {
    static const size_t value = prod<Rest...>::value;
};

template<size_t Idx, size_t ... Rest>
struct get_all {
    static const size_t indices[sizeof...(Rest)];
};

//---------------
//square root of integers at compile time, use like meta_sqrt<36>::ret
template<int Y,
         int InfX = 0,
         int SupX = ((Y==1) ? 1 : Y/2),
         bool Done = ((SupX-InfX)<=1 ? true : ((SupX*SupX <= Y) && ((SupX+1)*(SupX+1) > Y))) >
                                // use ?: instead of || just to shut up a stupid gcc 4.3 warning
class meta_sqrt
{
    enum {
  MidX = (InfX+SupX)/2,
  TakeInf = MidX*MidX > Y ? 1 : 0,
  NewInf = int(TakeInf) ? InfX : int(MidX),
  NewSup = int(TakeInf) ? int(MidX) : SupX
};
  public:
    enum { ret = meta_sqrt<Y,NewInf,NewSup>::ret };
};

template<int Y, int InfX, int SupX>
class meta_sqrt<Y, InfX, SupX, true>
{
    public:  enum { ret = (SupX*SupX <= Y) ? SupX : InfX };
};
//---------------

//
template <size_t...>
struct is_unique : std::integral_constant<bool, true> {};

template <size_t T, size_t U, size_t... VV>
struct is_unique<T, U, VV...> : std::integral_constant<bool, T != U && is_unique<T, VV...>::value> {};

template <size_t...>
struct no_of_unique : std::integral_constant<size_t, 0> {};

template <size_t T, size_t... UU>
struct no_of_unique<T, UU...> : std::integral_constant<size_t, is_unique<T, UU...>::value + no_of_unique<UU...>::value> {};



//---------------------------------------------------------------------------------
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
////////////////////////////








///////////////////////////////////////////////////////////////

// simple backport to c++11, as conditional_t is part of c++14
template< bool B, class T, class F >
using conditional_t_ = typename std::conditional<B,T,F>::type;

template <class... > struct typelist { };
template <class T, T ... Vals>
using typelist_c = typelist<std::integral_constant<T, Vals>...>;

template <class... > struct concat;
template <> struct concat<> { using type = typelist<>; };

template <class... Ts>
struct concat<typelist<Ts...>> {
    using type = typelist<Ts...>;
};

template <class... Ts, class... Us, class... Args>
struct concat<typelist<Ts...>, typelist<Us...>, Args...>
: concat<typelist<Ts..., Us...>, Args...>
{ };

template <class T, class TL>
struct filter_out;

template <class T, class TL>
using filter_out_t = typename filter_out<T, TL>::type;

template <class T, class... Ts>
struct filter_out<T, typelist<Ts...>>
    : concat<
        conditional_t_<std::is_same<T, Ts>::value, typelist<>, typelist<Ts>>...
        // c++14 version
        //std::conditional_t<std::is_same<T, Ts>::value, typelist<>, typelist<Ts>>...
        >
{ };

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
using no_of_uniques2 = length_t<uniq_t<typelist<size_t_<Ns>...>>>;


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


//template<size_t ...Rest>
//struct my_struct {
////    static const size_t value = no_of_uniques2<Rest...>::value;
//    static const size_t value = sizeof...(Rest);
//};


//template <class T, T ... Vals>
//using typelist_c = typelist<std::integral_constant<T, Vals>...>;

//template <size_t N,
//          size_t ... Rest,
//          class R = apply_typelist_t<quote_c<size_t, my_struct>,
//                        uniq_t<typelist_c<size_t, N, Rest...>>>
//          >
//R foos(const my_struct<N,Rest...> &a) {
//    // ...
//}

///////////////////////////////////////////////////////////////

}

#endif // TENSOR_META_H

