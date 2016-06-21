#ifndef EINSUM_META_H
#define EINSUM_META_H

#include "tensor/Tensor.h"

namespace Fastor {



template <FASTOR_INDEX ... All>
struct Index;

// Finding contraction indices
/////////////////////////////////////////////////////////////////
#include <type_traits>

// Count number of 'i' in 'rest...', base case
template <std::size_t i, std::size_t... rest>
struct Count : std::integral_constant<std::size_t, 0>
{};

// Count number of 'i' in 'rest...', inductive case
template <std::size_t i, std::size_t j, std::size_t... rest>
struct Count<i, j, rest...> :
    std::integral_constant<std::size_t,
                           Count<i, rest...>::value + ((i == j) ? 1 : 0)>
{};

// Is 'i' contained in 'rest...'?
template <std::size_t i, std::size_t... rest>
struct Contains :
    std::integral_constant<bool, (Count<i, rest...>::value > 0)>
{};


// Accumulation of counts of indices in all, base case
template <typename All, typename Remainder,
          typename AccIdx, typename AccCount>
struct Counts {
    using indices = AccIdx;
    using counts = AccCount;
};

// Accumulation of counts of indices in all, inductive case
template <std::size_t... all, std::size_t i, std::size_t... rest,
          std::size_t... indices, std::size_t... counts>
struct Counts<Index<all...>, Index<i, rest...>,
              Index<indices...>, Index<counts...>>
    : std::conditional<Contains<i, indices...>::value,
                       Counts<Index<all...>, Index<rest...>,
                              Index<indices...>,
                              Index<counts...>>,
                       Counts<Index<all...>, Index<rest...>,
                              Index<indices..., i>,
                              Index<counts...,
                                      Count<i, all...>::value>>>::type
{};

// Get value in From that matched the first value of Idx that matched idx
template <std::size_t idx, typename Idx, typename From>
struct First : std::integral_constant<std::size_t, 0>
{};
template <std::size_t i, std::size_t j, std::size_t k,
          std::size_t... indices, std::size_t... values>
struct First<i, Index<j, indices...>, Index<k, values...>>
    : std::conditional<i == j,
                       std::integral_constant<std::size_t, k>,
                       First<i, Index<indices...>,
                             Index<values...>>>::type
{};

// Return whether all values in From that match Idx being idx are tgt
template <std::size_t idx, std::size_t tgt, typename Idx, typename From>
struct AllMatchTarget : std::true_type
{};
template <std::size_t idx, std::size_t tgt,
          std::size_t i, std::size_t j,
          std::size_t... indices, std::size_t... values>
struct AllMatchTarget<idx, tgt,
                      Index<i, indices...>, Index<j, values...>>
    : std::conditional<i == idx && j != tgt, std::false_type,
                       AllMatchTarget<idx, tgt, Index<indices...>,
                                      Index<values...>>>::type
{};

/* Generate the dimensions, given the counts, indices, and values */
template <typename Counts, typename Indices,
          typename AllIndices, typename Values, typename Accum>
struct GenDims;

template <typename A, typename V, typename R>
struct GenDims<Index<>, Index<>, A, V, R> {
    using type = R;
};
template <typename T, std::size_t i, std::size_t c,
          std::size_t... counts, std::size_t... indices,
          std::size_t... dims, typename AllIndices, typename Values>
struct GenDims<Index<c, counts...>, Index<i, indices...>,
               AllIndices, Values, Tensor<T, dims...>>
{
    static constexpr auto value = First<i, AllIndices, Values>::value;
    static_assert(AllMatchTarget<i, value, AllIndices, Values>::value,
                  "CONTRACTION NOT POSSIBLE AS INDICES DO NOT CORRESPOND TO MATCHING DIMENSIONS");
    using type = typename GenDims<
        Index<counts...>, Index<indices...>,
        AllIndices, Values,
        typename std::conditional<c == 1,
                                  Tensor<T, dims..., value>,
                                  Tensor<T, dims...>>::type>::type;
};

/* Put it all together */
template <typename I, typename A>
struct ContractionType;

template <typename T, std::size_t... indices, std::size_t... values>
struct ContractionType<Index<indices...>, Tensor<T, values...>> {
    static_assert(sizeof...(indices) == sizeof...(values),
                   "NUMBER OF INDICES AND DIMENSIONS DO NOT MATCH");
    using counts = Counts<Index<indices...>,
                          Index<indices...>,
                          Index<>, Index<>>;
    using type = typename GenDims<typename counts::counts,
                                  typename counts::indices,
                                  Index<indices...>, Index<values...>,
                                  Tensor<T>>::type;
};
/////////////////////////////////////////////////////////////////








////
// products generator
template<int N>
constexpr int products(const size_t (&seq)[N], int i = N-1) {
    return i == (N-1) ? seq[N-1] : products(seq, i+1)*seq[i];
}

template<int N>
constexpr int shifter(const size_t (&seq)[N], int i) {
    return i < N-1 ? seq[i+1] : shifter(seq, i-1);
}

template<int N>
constexpr int zeroer(const size_t (&seq)[N], int i) {
    return i == N-1 ? 0 : seq[i];
}


template<class Idx, class Seq>
struct nprods;

template<size_t ... Rest, size_t ... ss>
struct nprods<Index<Rest...>,std_ext::index_sequence<ss...>> {
//    using type = Index<products(vals,ss)...>;
    constexpr static size_t vals[] = {Rest...};
    static constexpr size_t pvals[sizeof...(Rest)] = {products(vals,ss)...};
    static constexpr size_t svals[sizeof...(Rest)] = {shifter(pvals,ss)...};
//    static constexpr size_t values[sizeof...(Rest)] = {zeroer(svals,ss)...};
    static constexpr std::array<size_t,sizeof...(Rest)> values = {zeroer(svals,ss)...};

//    static void generate() {
////        constexpr size_t values[sizeof...(Rest)] = {products(vals,ss)...};
//        constexpr size_t values[sizeof...(Rest)] = {products(vals,ss)...};
//        constexpr size_t pvalues[sizeof...(Rest)] = {shifter(values,ss)...};
//        constexpr size_t zvalues[sizeof...(Rest)] = {zeroer(pvalues,ss)...};
////        print<size_t,sizeof...(Rest)>(values);
////        print<size_t,sizeof...(Rest)>(pvalues);
//        print<size_t,sizeof...(Rest)>(zvalues);
//    }
};

//template<size_t ... Rest, size_t ... ss>
//constexpr size_t nprods<Index<Rest...>,std_ext::index_sequence<ss...>>::values[sizeof...(Rest)];

template<size_t ... Rest, size_t ... ss>
constexpr std::array<size_t,sizeof...(Rest)> nprods<Index<Rest...>,std_ext::index_sequence<ss...>>::values;


}

#endif // EINSUM_META_H

