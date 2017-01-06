#ifndef TENSOR_META_H
#define TENSOR_META_H

#include "commons/commons.h"

namespace Fastor {

template<typename T> struct stride_finder {
    static const size_t value = 32  / sizeof(T);
};

template<> struct stride_finder<double> {
    static const size_t value = 4;
};
template<> struct stride_finder<float> {
    static const size_t value = 8;
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

template<size_t...Rest> struct add;
template<size_t Head, size_t ...Rest>
struct add<Head, Rest...> {
    static const size_t value = Head+add<Rest...>::value;
};
template<>
struct add<> {
    static const size_t value = 0;
};

template<size_t...Rest> struct prod;
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


//-----------
template<size_t first, size_t last, size_t step>
struct range_detector {
    static constexpr size_t range = last - first;
    static constexpr size_t value = range % step==0 ? range/step : range/step+1;
};
//-----------


//-------
template <int ... rest>
struct meta_min;

template <int m, int n, int ... rest>
struct meta_min<m,n,rest...> {
    static constexpr int pval = meta_min<m,n>::value;
    static const int value = (pval <= meta_min<pval,rest...>::value) ?
                pval : meta_min<pval,rest...>::value;
};

template <int m, int n>
struct meta_min<m,n> {
    static const int value = (m<=n) ? m : n;
};


template <int ... rest>
struct meta_max;

template <int m, int n, int ... rest>
struct meta_max<m,n,rest...> {
    static constexpr int pval = meta_max<m,n>::value;
    static const int value = (pval >= meta_max<pval,rest...>::value) ?
                pval : meta_max<pval,rest...>::value;
};

template <int m, int n>
struct meta_max<m,n> {
    static const int value = (m>=n) ? m : n;
};
//-------
template<int ... rest>
struct meta_argmin;

template<int m, int n, int ... rest>
struct meta_argmin<m,n,rest...> {
    static constexpr int pval = meta_min<m,n>::value;
    static const int value = (pval <= meta_min<pval,rest...>::value) ?
                meta_argmin<m,n>::value : meta_argmin<pval,rest...>::value+1;
};

//template<int m, int n, int p, int q, int r>
//struct meta_argmin<m,n,p,q,r> {
//    static constexpr int pval = meta_min<m,n,p,q>::value;
//    static constexpr int value = ( pval <= meta_min<pval,r>::value) ? meta_argmin<m,n,p,q>::value : 4;
//};

//template<int m, int n, int p, int q>
//struct meta_argmin<m,n,p,q> {
//    static constexpr int pval = meta_min<m,n,p>::value;
//    static constexpr int value = ( pval <= meta_min<pval,q>::value) ? meta_argmin<m,n,p>::value : 3;
//};

//template<int m, int n, int p>
//struct meta_argmin<m,n,p> {
//    static constexpr int pval = meta_min<m,n>::value;
//    static constexpr int value = ( pval <= meta_min<pval,p>::value) ? meta_argmin<m,n>::value : 2;
//};

template<int m, int n>
struct meta_argmin<m,n> {
    static const int value = (m<n) ? 0 : 1;
};

//-------

//-------
template<int ... rest>
struct meta_argmax;

template<int m, int n, int ... rest>
struct meta_argmax<m,n,rest...> {
    static constexpr int pval = meta_max<m,n>::value;
    static const int value = (pval >= meta_max<pval,rest...>::value) ?
                meta_argmax<m,n>::value : meta_argmax<pval,rest...>::value+1;
};

template<int m, int n>
struct meta_argmax<m,n> {
    static const int value = (m>n) ? 0 : 1;
};

//-------

//---------------
//square root of integers at compile time, use like meta_sqrt<36>::ret
template<int Y,
         int InfX = 0,
         int SupX = ((Y==1) ? 1 : Y/2),
         bool Done = ((SupX-InfX)<=1 ? true : ((SupX*SupX <= Y) && ((SupX+1)*(SupX+1) > Y))) >
                                // use ?: instead of || just to shut up a gcc 4.3 warning
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


// Checks if all parameters in a variadic parameter pack are arithmetic
//-------------------------------------------------------------------------------------------------//
template<class ... T>
struct is_arithmetic_pack;

template<class T, class ... U>
struct is_arithmetic_pack<T,U...> {
    static constexpr bool value = std::is_arithmetic<T>::value && is_arithmetic_pack<U...>::value;
};

template<class T>
struct is_arithmetic_pack<T> {
    static constexpr bool value = std::is_arithmetic<T>::value;
};
//-------------------------------------------------------------------------------------------------//

//
template <size_t...T>
struct is_unique : std::integral_constant<bool, true> {};

template <size_t T, size_t U, size_t... VV>
struct is_unique<T, U, VV...> : std::integral_constant<bool, T != U && is_unique<T, VV...>::value> {};

template <size_t...T>
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
using no_of_uniques = length_t<uniq_t<typelist<size_t_<Ns>...>>>;


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

////////////////////////


//template<size_t ...Rest>
//struct my_struct {
////    static const size_t value = no_of_uniques2<Rest...>::value;
////    static const size_t value = sizeof...(Rest);
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




// Check if indices appear more than twice [for Einstein summation]
//---------------------------------------------------------------------------------------------------------------
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

template <size_t... Sizes>
struct no_more_than_two {
    static constexpr size_t value = std::integral_constant<bool, max_count<Sizes...>::value <= 2>::value;
};

//template <size_t... Sizes>
//struct no_more_than_two {
//    static constexpr size_t value = std::integral_constant<bool, max_count<Sizes...>::value <= 2>::value;
//};



//template <size_t S, size_t... Sizes>
//struct count__;

//template <size_t S>
//struct count__<S>: std::integral_constant<size_t, 0> {};

//template <size_t S1, size_t... Sizes>
//struct count__<S1, S1, Sizes...>:
//    std::integral_constant<size_t, 1 + count__<S1, Sizes...>{}> {};

//template <size_t S1, size_t S2, size_t... Sizes>
//struct count__<S1, S2, Sizes...>:
//    count__<S1, Sizes...> {};

//template <size_t...>
//struct max_count;

//template <>
//struct max_count<>: std::integral_constant<size_t, 0> { };

//template <size_t S, size_t... Sizes>
//struct max_count<S, Sizes...>:
//    std::integral_constant<size_t, ct_max(1 + count__<S, Sizes...>{},
//                                            max_count<Sizes...>::value)> { };

////template <size_t... Sizes>
////struct no_more_than_two: std::integral_constant<bool, max_count<Sizes...>{} <= 2> { };

//template <size_t... Sizes>
//struct no_more_than_two std::integral_constant<bool, max_count<Sizes...>{} <= 2> { };

//---------------------------------------------------------------------------------------------------------------


}

#endif // TENSOR_META_H

