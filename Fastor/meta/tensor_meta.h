#ifndef TENSOR_META_H
#define TENSOR_META_H

#include "Fastor/commons/commons.h"

namespace Fastor {


template<typename T> struct stride_finder {
    static constexpr size_t value = DEFAULT_ABI /  8 / sizeof(T);
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
template<size_t Idx>
// Work around to avoid compiler errors
struct get_value<Idx> {
    static const size_t value = 0;
};

template <size_t N, typename... Args>
constexpr inline auto get_index(Args&&... as)
-> decltype(std::get<N>(std::forward_as_tuple(std::forward<Args>(as)...))) {
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

//! Partial product of a sequence up to an index up_to
template<class T, T N>
constexpr
inline T partial_prod(const T (&ind)[N], T up_to, T num=0) {
    return num == up_to ? ind[num] : ind[num]*partial_prod(ind, up_to, num+1);
}

//! Partial product of a sequence up to an index num from the end
template<class T, T N>
constexpr
inline T partial_prod_reverse(const T (&ind)[N], T up_to, T num=N-1) {
    return num == up_to ? ind[num] : ind[num]*partial_prod_reverse(ind, up_to, num-1);
}

template<size_t Idx, size_t ... Rest>
struct get_all {
    static const size_t indices[sizeof...(Rest)];
};


template<typename T>
constexpr T size_proder_(T one){
    return one.size();
}
template<typename T>
constexpr T size_proder_(T one, T two){
    return one.size()*two.size();
}
template<typename T, typename ... Ts>
constexpr T size_proder_(T one, T two, Ts ... ts) {
    return _proder_(_proder_(one,two),ts...);
}


//-----------
// template<size_t first, size_t last, size_t step>
// struct range_detector {
//     static constexpr size_t range = last - first;
//     static constexpr size_t value = range % step==0 ? range/step : range/step+1;
// };
template<int first, int last, int step>
struct range_detector {
    static constexpr int range = last - first;
    static constexpr int value = range % step==0 ? range/step : range/step+1;
};
//-----------


// Expression binding type for binary operators
//-----------
template<class T>
struct ExprBinderType {
#ifndef FASTOR_COPY_EXPR
    using type = typename std::conditional<std::is_arithmetic<T>::value, T, const T&>::type;
#else
    using type = T;
#endif
};
//-----------


// Use std::less and std::greater instead?
//-----------
template<size_t I, size_t J>
struct is_less {
    static constexpr bool value = I < J;
};

//-----------
template<size_t I, size_t J>
struct is_less_equal {
    static constexpr bool value = I <= J;
};

template<size_t I, size_t J>
struct is_greater {
    static constexpr bool value = I > J;
};

template<size_t I, size_t J>
struct is_greater_equal {
    static constexpr bool value = I >= J;
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
////////////////////////////





///////////
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
/////////////









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

