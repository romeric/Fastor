#ifndef EINSUM_META_H
#define EINSUM_META_H

#include "Fastor/meta/meta.h"
#include "Fastor/config/config.h"
#include "Fastor/tensor/Tensor.h"
#include <array>

namespace Fastor {

//namespace detail {


template <FASTOR_INDEX ... All>
struct Index;


// Find return type of tensor contraction
//------------------------------------------------------------------------------------------------------------//
// is ind[i] unique in ind?
template<size_t N>
constexpr bool is_uniq(const int (&ind)[N], size_t i, size_t cur = 0){
    return cur == N ? true :
           (cur == i || ind[cur] != ind[i]) ? is_uniq(ind, i, cur + 1) : false;
}

// For every i where ind[i] == index, is dim[i] == dimension?
template<size_t N>
constexpr bool check_all_eq(int index, int dimension,
                            const int (&ind)[N], const int (&dim)[N], size_t cur = 0) {
    return cur == N ? true :
           (ind[cur] != index || dim[cur] == dimension) ?
                check_all_eq(index, dimension, ind, dim, cur + 1) : false;
}

// if position i should be contracted away, return 1001001, otherwise return dim[i].
// triggers a compile-time error on mismatch.
template<size_t N>
constexpr int calc(size_t i, const int (&ind)[N], const int (&dim)[N]){
    return is_uniq(ind, i) ? dim[i] :
           check_all_eq(ind[i], dim[i], ind, dim) ? 1001001 : throw "dimension mismatch";
}

// if position i should be contracted away, return 1001001, otherwise return ind[i].
// triggers a compile-time error on mismatch.
template<size_t N>
constexpr int calc_idx(size_t i, const int (&ind)[N], const int (&dim)[N]){
    return is_uniq(ind, i) ? ind[i] :
           check_all_eq(ind[i], dim[i], ind, dim) ? 1001001 : throw "dimension mismatch";
}

//Now we need a way to get rid of the 1001001s:
template<class Ind, class... Inds>
struct concat_ { using type = Ind; };
template<size_t... I1, size_t... I2, class... Inds>
struct concat_<Index<I1...>, Index<I2...>, Inds...>
    :  concat_<Index<I1..., I2...>, Inds...> {};

template<int I, int... Is>
struct filter_
    :  concat_<typename std::conditional<Is == I, Index<>, Index<Is>>::type...> {};

template <int I, int ...Is>
using filter_t = typename filter_<I, Is...>::type;


//Use them:
template<class Ind, class Arr, class Seq>
struct contraction_impl;

template<template<typename, size_t ...> class Derived, typename T, size_t... Ind, size_t... Dim, size_t... Seq>
struct contraction_impl<Index<Ind...>, Derived<T, Dim...>, std_ext::index_sequence<Seq...>>{
    static constexpr int ind[sizeof...(Ind)] = { Ind... };
    static constexpr int dim[sizeof...(Dim)] = { Dim... };
    static constexpr int result[sizeof...(Seq)] = {calc(Seq, ind, dim)...};

    template<size_t... Dims>
    static auto unpack_helper(Index<Dims...>) -> Derived<T, Dims...>;

    using type = decltype(unpack_helper(typename filter_<1001001,  result[Seq]...>::type{}));

    // Get the indices instead of values
    static constexpr int result2[sizeof...(Seq)] = {calc_idx(Seq, ind, dim)...};
    template<size_t... Dims>
    static auto unpack_helper2(Index<Dims...>) -> Index<Dims...>;
    using indices = decltype(unpack_helper2(typename filter_<1001001,  result2[Seq]...>::type{}));
};
//------------------------------------------------------------------------------------------------------------//




// Generate product of dimensions e.g. products_a, products_b and products_out
//------------------------------------------------------------------------------------------------------------//
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
    constexpr static size_t vals[sizeof...(Rest)] = {Rest...};
    static constexpr size_t pvals[sizeof...(Rest)] = {products(vals,ss)...};
    static constexpr size_t svals[sizeof...(Rest)] = {shifter(pvals,ss)...};
    static constexpr std::array<size_t,sizeof...(Rest)> values = {zeroer(svals,ss)...};
};

template<size_t ... Rest, size_t ... ss>
constexpr std::array<size_t,sizeof...(Rest)> nprods<Index<Rest...>,std_ext::index_sequence<ss...>>::values;
//------------------------------------------------------------------------------------------------------------//


// For views only
//------------------------------------------------------------------------------------------------------------//
template<int N>
constexpr int _oner(const size_t (&seq)[N], int i) {
    return i == N-1 ? 1 : seq[i];
}

template<class Idx, class Seq>
struct nprods_views;

template<size_t ... Rest, size_t ... ss>
struct nprods_views<Index<Rest...>,std_ext::index_sequence<ss...>> {
    constexpr static size_t vals[sizeof...(Rest)] = {Rest...};
    static constexpr size_t pvals[sizeof...(Rest)] = {products(vals,ss)...};
    static constexpr size_t svals[sizeof...(Rest)] = {shifter(pvals,ss)...};
    static constexpr std::array<size_t,sizeof...(Rest)> values = {_oner(svals,ss)...};
};

template<size_t ... Rest, size_t ... ss>
constexpr std::array<size_t,sizeof...(Rest)> nprods_views<Index<Rest...>,std_ext::index_sequence<ss...>>::values;
//------------------------------------------------------------------------------------------------------------//






// Auxilary meta functions for tensor contraction
//------------------------------------------------------------------------------------------------------------//
// this is a meta-function equivalent to numpy's "where"
template<size_t N>
constexpr int find_index(const size_t (&ind)[N], int num, size_t i=0){
    return (i==N) ? N : (static_cast<int>(ind[i])==num ? i : find_index(ind,num,i+1));
}
template<size_t N>
constexpr int find_index(const std::array<size_t,N> &ind, int num, size_t i=0){
    return (i==N) ? N : (static_cast<int>(ind[i])==num ? i : find_index(ind,num,i+1));
}

// check if a given value is ind1 and not ind0
template<size_t M, size_t N>
constexpr bool check_diff(const size_t (&ind0)[M], const size_t (&ind1)[N], int num){
    return (find_index(ind0,num) == static_cast<int>(M)) & (find_index(ind1,num) < static_cast<int>(N));
}

// this is a meta-function somewhat equivalent to numpy's "setdiff1d"
// if a given value is in ind1 and not in ind0, then it returns the index in to the array ind1 such that
// ind1[index] = value (num)
template<size_t M, size_t N>
constexpr int find_index_diff(const size_t (&ind0)[M], const size_t (&ind1)[N], int num){
    return check_diff(ind0,ind1,num) ? find_index(ind1,num) : N;
}

// based on index from find_index_diff retrieve the actual value
template<size_t M, size_t N>
constexpr int retrieve_value(const size_t (&ind0)[M], const size_t (&ind1)[N], const int (&nums1)[N], size_t i=0){
    return find_index_diff(ind0,ind1,ind1[i]) == static_cast<int>(N)
            ? 1 : nums1[find_index(ind1,ind1[i])];
}


// does an array ind contain a number num (bool equivalent of find_index)
template<size_t N>
constexpr bool contains(const size_t (&ind)[N], int num){
    return find_index(ind,num)!=N;
}


template<class Dims>
struct put_dims_in_Index;

template<template<typename,size_t...> class Derived, size_t ... Rest, typename T>
struct put_dims_in_Index<Derived<T, Rest...>> {
    using type = Index<Rest...>;
};
//------------------------------------------------------------------------------------------------------------//


//! Checks vectorisability and returns a stride and a type
//------------------------------------------------------------------------------------------------------------//
template<class Idx0, class Idx1, class Tens>
struct is_vectorisable;

template<typename T, size_t ...Idx0, size_t ...Idx1, size_t...Rest>
struct is_vectorisable<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest...>> {
    template<typename ABI> using _vec_size = internal::get_simd_vector_size<SIMDVector<T,ABI>>;
    static constexpr size_t fastest_changing_index = get_value<sizeof...(Rest),Rest...>::value;
    static constexpr size_t idx[sizeof...(Idx0)] = {Idx0...};
    static constexpr bool does_2nd_tensor_disappear = ((int)no_of_unique<Idx0...,Idx1...>::value == (int)sizeof...(Idx0) - (int)sizeof...(Idx1));
    static constexpr bool last_index_contracted = contains(idx,get_value<sizeof...(Idx1),Idx1...>::value);
    static constexpr bool is_reducible = does_2nd_tensor_disappear && last_index_contracted;
    static constexpr bool value = (!last_index_contracted) && (fastest_changing_index % _vec_size<simd_abi::sse>::value==0);
    static constexpr bool sse_vectorisability = (!last_index_contracted) &&
            (fastest_changing_index % _vec_size<simd_abi::sse>::value==0 && fastest_changing_index % _vec_size<simd_abi::avx>::value!=0);
    static constexpr bool avx_vectorisability = (!last_index_contracted) &&
            (fastest_changing_index % _vec_size<simd_abi::sse>::value==0 && fastest_changing_index % _vec_size<simd_abi::avx>::value==0);
    static constexpr int stride = (avx_vectorisability ? _vec_size<simd_abi::avx>::value : (sse_vectorisability ? _vec_size<simd_abi::sse>::value : 1));

    using type = typename std::conditional<avx_vectorisability,SIMDVector<T,simd_abi::avx>,
        typename std::conditional<sse_vectorisability,SIMDVector<T,simd_abi::sse>,SIMDVector<T,simd_abi::scalar>>::type>::type;
};


template<size_t ...Idx0, size_t ...Idx1, size_t...Rest>
struct is_vectorisable<Index<Idx0...>,Index<Idx1...>,Tensor<float,Rest...>> {
    static constexpr size_t fastest_changing_index = get_value<sizeof...(Rest),Rest...>::value;
    static constexpr size_t idx[sizeof...(Idx0)] = {Idx0...};
    static constexpr bool does_2nd_tensor_disappear = ((int)no_of_unique<Idx0...,Idx1...>::value == (int)sizeof...(Idx0) - (int)sizeof...(Idx1));
    static constexpr bool last_index_contracted = contains(idx,get_value<sizeof...(Idx1),Idx1...>::value);
    static constexpr bool is_reducible = does_2nd_tensor_disappear && last_index_contracted;
    static constexpr bool value = (!last_index_contracted) && (fastest_changing_index % 4==0);
    static constexpr bool sse_vectorisability = (!last_index_contracted) && (fastest_changing_index % 4==0 && fastest_changing_index % 8!=0);
    static constexpr bool avx_vectorisability = (!last_index_contracted) && (fastest_changing_index % 4==0 && fastest_changing_index % 8==0);
    static constexpr int stride = (avx_vectorisability ? 8 : (sse_vectorisability ? 4 : 1));

    using type = typename std::conditional<avx_vectorisability,SIMDVector<float,simd_abi::avx>,
        typename std::conditional<sse_vectorisability,SIMDVector<float,simd_abi::sse>,SIMDVector<float,simd_abi::scalar>>::type>::type;
};

template<size_t ...Idx0, size_t ...Idx1, size_t...Rest>
struct is_vectorisable<Index<Idx0...>,Index<Idx1...>,Tensor<double,Rest...>> {
    static constexpr size_t fastest_changing_index = get_value<sizeof...(Rest),Rest...>::value;
    static constexpr size_t idx[sizeof...(Idx0)] = {Idx0...};
    static constexpr bool does_2nd_tensor_disappear = ((int)no_of_unique<Idx0...,Idx1...>::value == (int)sizeof...(Idx0) - (int)sizeof...(Idx1));
    static constexpr bool last_index_contracted = contains(idx,get_value<sizeof...(Idx1),Idx1...>::value);
    static constexpr bool is_reducible = does_2nd_tensor_disappear && last_index_contracted;
    static constexpr bool value = (!last_index_contracted) && (fastest_changing_index % 2==0);
    static constexpr bool sse_vectorisability = (!last_index_contracted) && (fastest_changing_index % 2==0 && fastest_changing_index % 4!=0);
    static constexpr bool avx_vectorisability = (!last_index_contracted) && (fastest_changing_index % 2==0 && fastest_changing_index % 4==0);
    static constexpr int stride = (avx_vectorisability ? 4 : (sse_vectorisability ? 2 : 1));

    using type = typename std::conditional<avx_vectorisability,SIMDVector<double,simd_abi::avx>,
        typename std::conditional<sse_vectorisability,SIMDVector<double,simd_abi::sse>,SIMDVector<double,simd_abi::scalar>>::type>::type;
};

//------------------------------------------------------------------------------------------------------------//


//! Checks reducible vectorisability and returns a stride and a type (use this for working on general strides)
//------------------------------------------------------------------------------------------------------------//
template<class Idx, class Tens>
struct is_reducibly_vectorisable;

template<typename T, size_t ...Idx, size_t...Rest>
struct is_reducibly_vectorisable<Index<Idx...>,Tensor<T,Rest...>> {
    template<typename ABI> using _vec_size = internal::get_simd_vector_size<SIMDVector<T,ABI>>;
    static constexpr size_t fastest_changing_index = get_value<sizeof...(Rest),Rest...>::value;
    static constexpr bool value = (fastest_changing_index % _vec_size<simd_abi::sse>::value==0);
    static constexpr bool sse_vectorisability = (fastest_changing_index % _vec_size<simd_abi::sse>::value==0 &&
                                                 fastest_changing_index % _vec_size<simd_abi::avx>::value!=0);
    static constexpr bool avx_vectorisability = (fastest_changing_index % _vec_size<simd_abi::sse>::value==0 &&
                                                 fastest_changing_index % _vec_size<simd_abi::avx>::value==0);
    static constexpr int stride = (avx_vectorisability ? _vec_size<simd_abi::avx>::value :
                                   (sse_vectorisability ? _vec_size<simd_abi::sse>::value : 1));

    using type = typename std::conditional<avx_vectorisability,SIMDVector<T,simd_abi::avx>,
        typename std::conditional<sse_vectorisability,SIMDVector<T,simd_abi::sse>,SIMDVector<T,simd_abi::scalar>>::type>::type;
};
//------------------------------------------------------------------------------------------------------------//




//! Check if indices in Einstein summation appear more than twice
//------------------------------------------------------------------------------------------------------------//
template<class ... Ind>
struct einsum_index_checker;

template<size_t ... Idx>
struct einsum_index_checker<Index<Idx...>> {
    static constexpr bool value = no_more_than_two<Idx...>::value;
};

//template<size_t ... Idx0, size_t ... Idx1>
//struct einsum_index_checker<Index<Idx0...>,Index<Idx1...>> {
//    static constexpr bool value = no_more_than_two<Idx0...,Idx1...>::value;
//};
//------------------------------------------------------------------------------------------------------------//









//------------------------------------------------------------------------------------------------------------//
template<class T, class U, class V, class W>
struct get_resuling_tensor;

template<size_t ... Idx0, size_t ... Idx1, size_t ... Rest0, size_t ... Rest1, typename T>
struct get_resuling_tensor<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>> {
    using type = typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
                              typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type;
};


template<class T, class U, class V, class W>
struct get_resuling_index;

template<size_t ... Idx0, size_t ... Idx1, size_t ... Rest0, size_t ... Rest1, typename T>
struct get_resuling_index<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>> {
    using type = typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
                              typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::indices;
};
//------------------------------------------------------------------------------------------------------------//



// How many nested loops should be set up
//------------------------------------------------------------------------------------------------------------//
template<class T, class U, class V, class W, class Seq>
struct no_of_loops_to_set;

template<size_t ... Idx0, size_t ... Idx1, size_t ... Rest0, size_t ... Rest1, typename T, size_t ... ss>
struct no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>> {

    using index_temp = apply_typelist_t<quote_c<size_t, Index>,
                    uniq_t<typelist_c<size_t, Idx0...,Idx1...>>>;

    static constexpr size_t concat_idx[sizeof...(Idx0)+sizeof...(Idx1)] = {Idx0...,Idx1...};
    static constexpr size_t concat_nums[sizeof...(Idx0)+sizeof...(Idx1)] = {Rest0...,Rest1...};
    static constexpr std::array<size_t,sizeof...(ss)> idx_in_concat = {find_index(concat_idx,index_temp::values[ss])...};
    static constexpr std::array<size_t,sizeof...(ss)> dims = {concat_nums[idx_in_concat[ss]]...};
    static constexpr int value = pack_prod<dims[ss]...>::value;

    using type = Tensor<T,dims[ss]...>;
    using indices = Index<index_temp::values[ss]...>;

    using dims_type = Index<dims[ss]...>;
};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Rest0, size_t ... Rest1, typename T, size_t ... ss>
constexpr std::array<size_t,sizeof...(ss)>
no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>>::dims;
//------------------------------------------------------------------------------------------------------------//






// Find indices from the number of loops to set up e.g. idx_a, idx_b and idx_out
//------------------------------------------------------------------------------------------------------------//
template<class Ind0, class Ind1, class Tensor0, class Tensor1, class Seq>
struct IndexFirstTensor;

template<class T, size_t... Idx0, size_t... Idx1, size_t ...Rest0, size_t ...Rest1, size_t ... ss>
struct IndexFirstTensor<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>> {

    using index_temp = typename no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
            typename std_ext::make_index_sequence<no_of_unique<Idx0...,Idx1...>::value>::type>::indices;

    static constexpr size_t idx[sizeof...(Idx0)] = {Idx0...};
    static constexpr std::array<size_t,sizeof...(Idx0)>
    indices = {find_index(index_temp::values, idx[ss])...};

    using type = Tensor<T,indices[ss]...>;

};

template<class T, size_t... Idx0, size_t... Idx1, size_t ...Rest0, size_t ...Rest1, size_t ... ss>
constexpr std::array<size_t,sizeof...(Idx0)>
IndexFirstTensor<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>>::indices;


template<class Ind0, class Ind1, class Tensor0, class Tensor1, class Seq>
struct IndexSecondTensor;

template<class T, size_t... Idx0, size_t... Idx1, size_t ...Rest0, size_t ...Rest1, size_t ... ss>
struct IndexSecondTensor<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>> {

    using index_temp = typename no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
            typename std_ext::make_index_sequence<no_of_unique<Idx0...,Idx1...>::value>::type>::indices;

    static constexpr size_t idx[sizeof...(Idx1)] = {Idx1...};
    static constexpr std::array<size_t,sizeof...(Idx1)>
    indices = {find_index(index_temp::values, idx[ss])...};

    using type = Tensor<T,indices[ss]...>;

};

template<class T, size_t... Idx0, size_t... Idx1, size_t ...Rest0, size_t ...Rest1, size_t ... ss>
constexpr std::array<size_t,sizeof...(Idx1)>
IndexSecondTensor<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>>::indices;



template<class Ind0, class Ind1, class Tensor0, class Tensor1, class Seq>
struct IndexResultingTensor;

template<class T, size_t... Idx0, size_t... Idx1, size_t ...Rest0, size_t ...Rest1, size_t ... ss>
struct IndexResultingTensor<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>> {

    using index_temp = typename no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
            typename std_ext::make_index_sequence<no_of_unique<Idx0...,Idx1...>::value>::type>::indices;

    using resulting_index_0 =  typename get_resuling_index<Index<Idx0...>,Index<Idx1...>,
                                    Tensor<T,Rest0...>,Tensor<T,Rest1...>>::type;

    static constexpr std::array<size_t,sizeof...(ss)>
    indices = {find_index(index_temp::values, resulting_index_0::values[ss])...};

    using type = Tensor<T,indices[ss]...>;
};

template<class T, size_t... Idx0, size_t... Idx1, size_t ...Rest0, size_t ...Rest1, size_t ... ss>
constexpr std::array<size_t,sizeof...(ss)>
IndexResultingTensor<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>>::indices;
//------------------------------------------------------------------------------------------------------------//



//------------------------------------------------------------------------------------------------------------//
// This is for pure inner reduction and not permuted reduction
template<class Idx0, class Idx1>
struct is_pair_reduction;

template<size_t ... Idx0, size_t ... Idx1>
struct is_pair_reduction<Index<Idx0...>,Index<Idx1...>> {
    static constexpr bool value = is_same_v_<Index<Idx0...>,Index<Idx1...>>;
};

// helper
template<class Idx0, class Idx1>
constexpr bool is_pair_reduction_v = is_pair_reduction<Idx0,Idx1>::value;


// Reduction for a single tensor
template<class Idx, class Tens>
struct is_single_reduction;

template<size_t ... Idx0, typename T, size_t ... Rest0>
struct is_single_reduction<Index<Idx0...>, Tensor<T,Rest0...>> {
    using resulting_tensor = typename contraction_impl<Index<Idx0...>, Tensor<T,Rest0...>,
      typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::type;
    static constexpr bool value = resulting_tensor::dimension_t::value == 0;
};

// helper
template<class Idx, class Tens>
constexpr bool is_single_reduction_v = is_single_reduction<Idx,Tens>::value;
//------------------------------------------------------------------------------------------------------------//



//--------------------------------------------------------------------------------------------------------------//
template<size_t N>
constexpr int contain_prod(const size_t (&ind)[N], const size_t (&sseq)[N], size_t num){
    return (ind[num]==0) ? static_cast<int>(sseq[num]) : -1;
}

template<size_t N>
constexpr int last_indices_prod(const int (&sseq)[N], int num){
    return  num > 0 ? ( (sseq[num-1]!=-1) ? sseq[num-1]*last_indices_prod(sseq,num-1) : 1) : 1;
}


template<class Idx0, class Idx1, class Tens0, class Tens1, class SSeq>
struct general_stride_finder;

template<typename T, size_t ...Idx0, size_t ...Idx1, size_t...Rest0, size_t...Rest1, size_t ... ss>
struct general_stride_finder<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>, std_ext::index_sequence<ss...>> {

    using OutIndices = typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
      typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::indices;

    static constexpr size_t b_idx[sizeof...(Idx1)] = {Idx1...};
    static constexpr size_t b_dim[sizeof...(Rest1)] = {Rest1...};
//    static constexpr std::array<size_t,sizeof...(ss)> container_idx = {contains(OutIndices::values,b_idx[ss])...};
    static constexpr size_t container_idxx[sizeof...(Rest1)] = {contains(OutIndices::values,b_idx[ss])...};
//    static constexpr std::array<int,sizeof...(ss)> container_dim = {contain_prod(container_idxx,b_dim,ss)...};
    static constexpr int container_dim[sizeof...(Rest1)] = {contain_prod(container_idxx,b_dim,ss)...};
    static constexpr int value = last_indices_prod(container_dim,sizeof...(Rest1));
};

//template<typename T, size_t ... Idx0, size_t ... Idx1, size_t ... Rest0, size_t ... Rest1, size_t ... ss>
//constexpr std::array<size_t,sizeof...(ss)>
//general_stride_finder<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>>::container_idx;
//template<typename T, size_t ... Idx0, size_t ... Idx1, size_t ... Rest0, size_t ... Rest1, size_t ... ss>
//constexpr std::array<int,sizeof...(ss)>
//general_stride_finder<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>>::container_dim;
//--------------------------------------------------------------------------------------------------------------------//






namespace internal {
//--------------------------------------------------------------------------------------------------------------------//
//! Given the einsum indices of two tensors, matches their indices to see
//! if it is a genearalised matrix-vector multiplication. If either the indices
//! of the first tensor or the second tensor disappear while matching indices
//! from the end, then it is a genearalised matrix-vector multiplication.
//! Note that this function matches indices from the end in that it can detect
//! generalised matrix-vector product of the forms <ijk,jk> or <jk,ijk> (it can detect swapping)
//! but it cannot detect <ij,ijk> or <ijk,ij> which is vector-matrix product
template<class T, T N0, T N1>
constexpr
inline bool match_indices_from_end(const T (&ind0)[N0],
    const T (&ind1)[N1], T num0=N0-1, T num1=N1-1) {
    return ind1[num1] == ind0[num0] ? ( num1 == 0 ? ind1[num1] == ind0[num0] :
        (num0 == 0 ? ind1[num1] == ind0[num0] : match_indices_from_end(ind0, ind1, num0 - 1, num1 - 1))) :
        false;
}

//! Same as above but gives the index up to which the higher order tensor (generalised matrix)
//! matches the lower order tensor (generalised vector) from the end. The index is for higher
//! higher order tensor counting from the start including the index itself for instance
//! <ij,j> will return 0 (i.e. index 0 does not match) and <ijk,k> will return 1 (i.e. indices 0 and 1 do no match)
//! The function in essence gives the remainder indices that don't match the vector
//!
//! This function only works (gives the correct index) if the accompanying boolean
//! function (match_indices_from_end) is true i.e. it only works if we have a true
//! generalised matrix-vector product otherwise gives an incorrect index
template<class T, T N0, T N1>
constexpr
inline T match_indices_from_end_index(const T (&ind0)[N0],
    const T (&ind1)[N1], T num0=N0-1, T num1=N1-1) {
    return ind1[num1] == ind0[num0] ? ( num1 == 0 ? (ind1[num1] == ind0[num0] ? num0-1 : num0) :
        num0 == 0 ? (ind1[num1] == ind0[num0] ? num1-1 : num1) :
        match_indices_from_end_index(ind0, ind1, num0 - 1, num1 - 1) ) : num0-1;
}


//! Given the einsum indices of two tensors, matches their indices to see
//! if it is a genearalised vector-matrix multiplication. If either the indices
//! of the first tensor or the second tensor disappear while matching indices
//! from the beggining, then it is a genearalised vector-matrix multiplication.
//! Note that this function matches indices from the beggingin in that it can detect
//! generalised vector-matrix product of the forms <ijk,ij> or <ij,ijk> (it can detect swapping)
//! but it cannot detect <jk,ijk> or <ijk,jk> which is matrix-vector product
template<class T, T N0, T N1>
constexpr
inline bool match_indices_from_start(const T (&ind0)[N0],
    const T (&ind1)[N1], T num0=0, T num1=0) {
    return ind1[num1] == ind0[num0] ? ( num1 == N1-1 ? ind1[num1] == ind0[num0] :
        (num0 == N0-1 ? ind1[num1] == ind0[num0] :  match_indices_from_start(ind0, ind1, num0 + 1, num1 + 1))) :
        false;
}

//! Same as above but gives the index up to which the higher order tensor (generalised matrix)
//! matches the lower order tensor (generalised vector) from the start. The index is for higher
//! higher order tensor counting from the end excluding the index itself for instance
//! <ij,i> will return 1 (only 0 matches) and <ijk,ij> will return 2 (i.e. indices 0 and 1 match)
//! The function in essence gives the remainder indices that don't match the vector
//!
//! function (match_indices_from_start) is true i.e. it only works if we have a true
//! generalised vector-matrix product otherwise gives an incorrect index
template<class T, T N0, T N1>
constexpr
inline T match_indices_from_start_index(const T (&ind0)[N0],
    const T (&ind1)[N1], T num0=0, T num1=0) {
    return ind1[num1] == ind0[num0] ? ( num1 == N1-1  ?  (ind1[num1] == ind0[num0] ? num0+1 : num0) :
        num0 == N0-1 ? (ind1[num1] == ind0[num0] ? num1+1 : num1) :
        match_indices_from_start_index(ind0, ind1, num0 + 1, num1 + 1) ) : num0;
}


//! Given the einsum indices of two tensors, matches their indices to see
//! if it is a standard genearalised matrix-matrix multiplication.
//! By standard we mean <ijk,jkl>. This function cannot detect transposed cases of
//! gemm such as <ijk,ljk> and so on.
//! Note that this function matches indices from the two ends but it also detects
//! <ijk,ijk> as gemm. So it should be used in conjunction with is_vector_matrix/is_matrix_vec
//! and is_inner. Look at the corresponding struct that uses it
template<class T, T N0, T N1>
constexpr
inline bool match_indices_from_two_ends(const T (&ind0)[N0],
    const T (&ind1)[N1], int ncontracted, T num0=N0-1, T num1=0) {
    return ncontracted != 0 ?
        (ncontracted == 1 ? (ind1[num1] == ind0[num0 - ncontracted + 1] ? true : false)
        : (ind1[num1] == ind0[num0 - ncontracted + 1] ?
        match_indices_from_two_ends(ind0, ind1, ncontracted-1, num0, num1 + 1) :
        false)) :
        false;
}




template<class Idx0, class Idx1>
struct is_generalised_matrix_vector;

template<size_t ... Idx0, size_t ... Idx1>
struct is_generalised_matrix_vector<Index<Idx0...>,Index<Idx1...> > {
    static constexpr size_t which_one_is_vector = sizeof...(Idx0) > sizeof...(Idx1) ? 1 : 0;
    static constexpr size_t idx0[sizeof...(Idx0)] = {Idx0...};
    static constexpr size_t idx1[sizeof...(Idx1)] = {Idx1...};
    static constexpr bool value = match_indices_from_end(idx0, idx1) && sizeof...(Idx0) != sizeof...(Idx1);
    static constexpr size_t matches_up_to = match_indices_from_end_index(idx0, idx1);
};


template<class Idx0, class Idx1>
struct is_generalised_vector_matrix;

template<size_t ... Idx0, size_t ... Idx1>
struct is_generalised_vector_matrix<Index<Idx0...>,Index<Idx1...> > {
    static constexpr size_t which_one_is_vector = sizeof...(Idx0) > sizeof...(Idx1) ? 1 : 0;
    static constexpr size_t idx0[sizeof...(Idx0)] = {Idx0...};
    static constexpr size_t idx1[sizeof...(Idx1)] = {Idx1...};
    static constexpr bool value = match_indices_from_start(idx0, idx1) && sizeof...(Idx0) != sizeof...(Idx1);
    static constexpr size_t matches_up_to = match_indices_from_start_index(idx0, idx1);
};


template<class Idx0, class Idx1>
struct is_generalised_matrix_matrix;

template<size_t ... Idx0, size_t ... Idx1>
struct is_generalised_matrix_matrix<Index<Idx0...>,Index<Idx1...> > {
    static constexpr bool is_mat_vec = is_generalised_matrix_vector<Index<Idx0...>,Index<Idx1...>>::value;
    static constexpr bool is_vec_mat = is_generalised_vector_matrix<Index<Idx0...>,Index<Idx1...>>::value;
    static constexpr int ncontracted = sizeof...(Idx0) + sizeof...(Idx1) - no_of_unique<Idx0...,Idx1...>::value;
    static constexpr bool is_inner = sizeof...(Idx0) == sizeof...(Idx1) && no_of_unique<Idx0...,Idx1...>::value == sizeof...(Idx1);
    static constexpr size_t idx0[sizeof...(Idx0)] = {Idx0...};
    static constexpr size_t idx1[sizeof...(Idx1)] = {Idx1...};
    static constexpr bool value = !is_mat_vec && !is_vec_mat && !is_inner && match_indices_from_two_ends(idx0, idx1, ncontracted);
};
//--------------------------------------------------------------------------------------------------------------------//
} // namespace internal







// A complete tensor contraction meta-engine
//--------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------//
#if FASTOR_CXX_VERSION >= 2017
template<size_t N>
inline constexpr std::array<int,N> find_remaining(const std::array<size_t,N> &maxes_out, int total) {
    std::array<int,N> remainings = {};
    remainings[0] = total / maxes_out[0];
    for (int i=1; i<N; ++i) {
        remainings[i] = remainings[i-1]/maxes_out[i];
    }
    return remainings;

}

template<size_t N, int total>
inline constexpr std::array<std::array<int,N>,(size_t)total> cartesian_product_2(const std::array<size_t,N> &maxes_out) {
    std::array<std::array<int,N>,(size_t)total> as_all = {};
    for (int i=0; i<total; i+=1) {
        int remaining = total;
        for (int n=0; n<N; ++n) {
            remaining /= maxes_out[n];
            as_all[i][n] = ( i / remaining ) % maxes_out[n];
        }
    }
    return as_all;
}
#endif

template<int N>
constexpr int find_remaining(const int (&maxes_out)[N], int remaining, int i) {
    return i==0 ? remaining/maxes_out[0] : find_remaining(maxes_out,remaining,i-1) / maxes_out[i];
}

template<int N>
constexpr int cartesian_product_single(const int (&maxes_out)[N], int remaining, int i, int n=0) {
    return (i/(find_remaining(maxes_out,remaining,n))) % maxes_out[n];
}

template<int Idx, class Tens, class Seq>
struct gen_single_cartesian_product;

template<int I, size_t ... Rest, size_t ... ss, typename T>
struct gen_single_cartesian_product<I,Tensor<T,Rest...>,std_ext::index_sequence<ss...>> {
    static constexpr int vals[sizeof...(Rest)] = {Rest...};
    static constexpr std::array<int,sizeof...(ss)> values = {cartesian_product_single(vals,pack_prod<Rest...>::value,I,ss)...};
};

template<int I, size_t ... Rest, size_t ... ss, typename T>
constexpr std::array<int,sizeof...(ss)> gen_single_cartesian_product<I,Tensor<T,Rest...>,std_ext::index_sequence<ss...>>::values;


template<typename T, int i, size_t ... Rest>
constexpr std::array<int,sizeof...(Rest)> all_cartesian_product() {
    return gen_single_cartesian_product<i,Tensor<T,Rest...>,typename std_ext::make_index_sequence<sizeof...(Rest)>::type>::values;
}


template<class Tens, class Seq>
struct cartesian_product;

template<size_t ... Rest, size_t ... ss, typename T>
struct cartesian_product<Tensor<T,Rest...>,std_ext::index_sequence<ss...>> {
    static constexpr std::array<std::array<int,sizeof...(Rest)>,sizeof...(ss)> values = {all_cartesian_product<T,ss,Rest...>()...};

};

template<size_t ... Rest, size_t ... ss, typename T>
constexpr std::array<std::array<int,sizeof...(Rest)>,sizeof...(ss)> cartesian_product<Tensor<T,Rest...>,std_ext::index_sequence<ss...>>::values;


template<size_t N, size_t O,size_t All>
constexpr int get_indices(const std::array<size_t,N> &products,
                          const std::array<size_t,N>& idx,
                          const std::array<std::array<int,O>,All> &as_all,
                          int i,
                          int it) {
    return it==0 ? as_all[i][idx[static_cast<int>(N)-1]] + products[0]*as_all[i][idx[0]] :
        products[it]*as_all[i][idx[it]]+get_indices(products,idx,as_all,i,it-1);
}

// Blowing compilation time and memory usage 101
//--------------------------------------------------------------------------------------------------------------//
template<class Idx0, class Idx1, class Tens0, class Tens1, class Seq>
struct contract_meta_engine;

template<size_t ... Idx0, size_t ... Idx1, size_t ... Rest0, size_t ... Rest1, size_t ... ss, typename T>
struct contract_meta_engine<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>> {

    using OutTensor = typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
                              typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type;
    using OutIndices = typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
                              typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::indices;

    static constexpr int a_dim   = sizeof...(Rest0);
    static constexpr int b_dim   = sizeof...(Rest1);
    static constexpr int out_dim = OutTensor::Dimension;
    static constexpr int total   = sizeof...(ss);

    static constexpr auto& idx_a = IndexFirstTensor<Index<Idx0...>,Index<Idx1...>, Tensor<T,Rest0...>,Tensor<T,Rest1...>,
                              typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::indices;
    static constexpr auto& idx_b = IndexSecondTensor<Index<Idx0...>,Index<Idx1...>, Tensor<T,Rest0...>,Tensor<T,Rest1...>,
                              typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::indices;
    static constexpr auto& idx_out = IndexResultingTensor<Index<Idx0...>,Index<Idx1...>, Tensor<T,Rest0...>,Tensor<T,Rest1...>,
                              typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::indices;

    static constexpr int uniques = no_of_unique<Idx0...,Idx1...>::value;
    using uniques_type = typename std_ext::make_index_sequence<uniques>::type;
    static constexpr auto& maxes_out = no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
            uniques_type>::dims;

    using maxes_out_type = typename no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
            uniques_type>::type;

    static constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
    static constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
    using Index_with_dims = typename put_dims_in_Index<OutTensor>::type;
    static constexpr std::array<size_t,Index_with_dims::Size> products_out = nprods<Index_with_dims,
            typename std_ext::make_index_sequence<Index_with_dims::Size>::type>::values;

    // Generate the cartesian product
    static constexpr auto& as_all = cartesian_product<maxes_out_type,typename std_ext::make_index_sequence<total>::type>::values;
    // Alternatively you can pass the ss... directly into cartesian_product but that does not change anything in terms of
    // memory usage or compilation time
    //using maxes_out_indices = typename no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
    //        uniques_type>::indices;
    //static constexpr std::array<std::array<int,maxes_out_indices::NoIndices>,total> as_all = {all_cartesian_product<ss,2,3,4,2>()...};

    static constexpr std::array<int,sizeof...(ss)> index_a   = {get_indices(products_a,idx_a,as_all,ss,a_dim-1)...};
    static constexpr std::array<int,sizeof...(ss)> index_b   = {get_indices(products_b,idx_b,as_all,ss,b_dim-1)...};
    static constexpr std::array<int,sizeof...(ss)> index_out = {get_indices(products_out,idx_out,as_all,ss,out_dim-1)...};
};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Rest0, size_t ... Rest1, size_t ... ss, typename T>
constexpr std::array<int,sizeof...(ss)>
contract_meta_engine<Index<Idx0...>,Index<Idx1...>,
Tensor<T,Rest0...>,Tensor<T,Rest1...>,
std_ext::index_sequence<ss...>>::index_a;

template<size_t ... Idx0, size_t ... Idx1, size_t ... Rest0, size_t ... Rest1, size_t ... ss, typename T>
constexpr std::array<int,sizeof...(ss)>
contract_meta_engine<Index<Idx0...>,Index<Idx1...>,
Tensor<T,Rest0...>,Tensor<T,Rest1...>,
std_ext::index_sequence<ss...>>::index_b;

template<size_t ... Idx0, size_t ... Idx1, size_t ... Rest0, size_t ... Rest1, size_t ... ss, typename T>
constexpr std::array<int,sizeof...(ss)>
contract_meta_engine<Index<Idx0...>,Index<Idx1...>,
Tensor<T,Rest0...>,Tensor<T,Rest1...>,
std_ext::index_sequence<ss...>>::index_out;

//--------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------//








// The followings are generic implemenations that work for any type of complex tensor network
//------------------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------------------//


// Find how many loops needs to be set. Works for any complex tensor network
//------------------------------------------------------------------------------------------------------------//
// This is equivalent to no_of_loops_to_set_up but more generic (works for arbitrary tensor networks)
template<class TotalIdx, class TotalDims, class Seq>
struct loop_setter;

template<size_t ... Idx, size_t ... Rest, size_t ... ss, typename T>
struct loop_setter<Index<Idx...>,Tensor<T,Rest...>,std_ext::index_sequence<ss...>> {

    using index_temp = apply_typelist_t<quote_c<size_t, Index>,
                    uniq_t<typelist_c<size_t, Idx...>>>;

    static constexpr size_t concat_idx[sizeof...(Idx)] = {Idx...};
    static constexpr size_t concat_nums[sizeof...(Rest)] = {Rest...};
    static constexpr std::array<size_t,sizeof...(ss)> idx_in_concat = {find_index(concat_idx,index_temp::values[ss])...};
    static constexpr std::array<size_t,sizeof...(ss)> dims = {concat_nums[idx_in_concat[ss]]...};
    static constexpr int value = pack_prod<dims[ss]...>::value;

    using type = Tensor<T,dims[ss]...>;
    using indices = Index<index_temp::values[ss]...>;

    using dims_type = Index<dims[ss]...>;
};

template<size_t ... Idx, size_t ... Rest, size_t ... ss, typename T>
constexpr std::array<size_t,sizeof...(ss)>
loop_setter<Index<Idx...>,Tensor<T,Rest...>,std_ext::index_sequence<ss...>>::dims;
//------------------------------------------------------------------------------------------------------------//


// Get indices of every individual tensor in the network. Works for any complex tensor network
//------------------------------------------------------------------------------------------------------------//
template<class Ind, class Tens, class Ind_t, class Tens_t, class Seq>
struct IndexTensors;

template<template<typename,size_t...> class Derived0,
    template<typename,size_t...> class Derived1,
    typename T, size_t... Idx, size_t... Idx_t, size_t ...Rest, size_t ...Rest_t, size_t ... ss>
struct IndexTensors<Index<Idx...>,Derived0<T,Rest...>,Index<Idx_t...>,Derived1<T,Rest_t...>,std_ext::index_sequence<ss...>> {

    using index_temp = typename loop_setter<Index<Idx...>,Tensor<T,Rest...>,
            typename std_ext::make_index_sequence<no_of_unique<Idx...>::value>::type>::indices;

    static constexpr size_t idx[sizeof...(Idx_t)] = {Idx_t...};
    static constexpr std::array<size_t,sizeof...(Idx_t)>
    indices = {find_index(index_temp::values, idx[ss])...};

    using type = Tensor<T,indices[ss]...>;

};

template<template<typename,size_t...> class Derived0,
    template<typename,size_t...> class Derived1,
    typename T, size_t... Idx, size_t... Idx_t, size_t ...Rest, size_t ...Rest_t, size_t ... ss>
constexpr std::array<size_t,sizeof...(Idx_t)>
IndexTensors<Index<Idx...>,Derived0<T,Rest...>,Index<Idx_t...>,Derived1<T,Rest_t...>,std_ext::index_sequence<ss...>>::indices;
//------------------------------------------------------------------------------------------------------------//



namespace internal {
//------------------------------------------------------------------------------------------------------------//
template<class arg>
struct meta_argmin_wrapper;
template<size_t ...rest>
struct meta_argmin_wrapper<Index<rest...>> {
    static constexpr int value = meta_argmin<rest...>::value;
};
template<size_t idx>
struct meta_argmin_wrapper<Index<idx>> {
    static constexpr int value = idx;
};


// Complete compile-time sorting algorithm: Does not work if there a duplicate entries in a pack
template<class arg>
struct meta_sort;
template<size_t ...rest>
struct meta_sort<Index<rest...>> {
    static constexpr int least_value_idx = meta_argmin_wrapper<Index<rest...>>::value;
    static constexpr int least_value = get_value<least_value_idx+1,rest...>::value;
    using reduced_seq = typename filter_<least_value,rest...>::type;
    using new_seq = typename concat_<Index<least_value>,typename meta_sort<reduced_seq>::new_seq>::type;
};
template<size_t value>
struct meta_sort<Index<value>> {
    using reduced_seq = Index<value>;
    using new_seq = Index<value>;
};


// Complete compile-time arg-sorting algorithm: Does not work if there a duplicate entries in a pack
template<class arg, class seq>
struct meta_argsort;
template<size_t ...rest, size_t ...ss>
struct meta_argsort<Index<rest...>,Index<ss...>> {
    static constexpr int least_value_idx = meta_argmin_wrapper<Index<rest...>>::value;
    static constexpr int least_value = get_value<least_value_idx+1,rest...>::value;
    using reduced_seq = typename filter_<least_value,rest...>::type;
    using new_seq = typename concat_<Index<least_value>,typename meta_sort<reduced_seq>::new_seq>::type;

    static constexpr int least_index = get_value<least_value_idx+1,ss...>::value;

    using reduced_argseq = typename filter_<least_index,ss...>::type;
    using new_argseq = typename concat_<Index<least_index>,typename meta_argsort<reduced_seq,reduced_argseq>::new_argseq>::type;
};
template<size_t value, size_t ss>
struct meta_argsort<Index<value>,Index<ss>> {
    using reduced_seq = Index<value>;
    using new_seq = Index<value>;

    using reduced_argseq = Index<ss>;
    using new_argseq = Index<ss>;
};
//------------------------------------------------------------------------------------------------------------//


// Permutation functions
//------------------------------------------------------------------------------------------------------------//
template<size_t N>
constexpr size_t count_less(const size_t (&seq)[N], size_t i, size_t cur = 0) {
    return cur == N ? 0 : (count_less(seq, i, cur + 1) + (seq[cur] < i ? 1 : 0));
}

/* Check if a compile time array is sequential */
template<size_t N>
constexpr bool is_sequential(const size_t (&seq)[N], size_t i=0) {
    return i+1 == N ? true : ( seq[i] + 1 == seq[i+1] ? is_sequential(seq, i+1) : false ) ;
}
template<size_t N>
constexpr bool is_sequential(const std::array<size_t,N> &seq, size_t i=0) {
    return i+1 == N ? true : ( seq[i] + 1 == seq[i+1] ? is_sequential(seq, i+1) : false ) ;
}

// permutation helper class
template<class Idx, class Tens, class Seq>
struct permute_impl;

template<typename T, size_t ... ls, size_t ... fs, size_t... ss>
struct permute_impl<Index<ls...>, Tensor<T, fs...>, std_ext::index_sequence<ss...>> {
    constexpr static size_t lst[sizeof...(ls)] = { ls... };
    constexpr static size_t fvals[sizeof...(ls)] = {fs...};
    using resulting_tensor = Tensor<T,fvals[count_less(lst, lst[ss])]...>;
    using resulting_index  = typename meta_argsort<Index<ls...>,Index<ss...>>::new_argseq;
    using maxes_out_type   = Index<fvals[meta_argsort<Index<ls...>,Index<ss...>>::new_argseq::values[ss]]...>;
    static constexpr bool requires_permutation = !(is_same_v_<resulting_tensor,Tensor<T, fs...>> &&
                                                    is_sequential(resulting_index::values));
};

// permute helper class
template<class Idx, class Tens, class Seq>
struct new_permute_impl;

template<typename T, size_t ... ls, size_t ... fs, size_t... ss>
struct new_permute_impl<Index<ls...>, Tensor<T, fs...>, std_ext::index_sequence<ss...>> {
    constexpr static size_t lst[sizeof...(ls)] = { ls... };
    constexpr static size_t fvals[sizeof...(ls)] = {fs...};
    using resulting_tensor = Tensor<T,fvals[count_less(lst, lst[ss])]...>;
    constexpr static size_t aranger[sizeof...(ss)] = { ss... };
    using resulting_index = Index<aranger[count_less(lst, lst[ss])]...>;
    static constexpr bool requires_permutation = !(is_same_v_<resulting_tensor,Tensor<T, fs...>> &&
                                                    is_sequential(resulting_index::values));
};
//------------------------------------------------------------------------------------------------------------//


//------------------------------------------------------------------------------------------------------------//
#if FASTOR_CXX_VERSION >= 2017
template<size_t N>
constexpr std::array<size_t, N> get_floor_map(const std::array<size_t, N> &idx) {
    std::array<size_t, N> out = {};
    for (size_t i=0; i<N; ++i) {
        out[idx[i]] = i;
    }
    return out;
}

/* Find how many swaps/permutations are necessary going from the resulting einsum index to the output index
*/
template<size_t N>
constexpr std::array<size_t, N> find_permuation(const std::array<size_t, N> &idx0, const std::array<size_t, N> &idx1) {
    std::array<size_t, N> out = {};
    for (size_t i=0; i<N; ++i) {
        // int idx = find_index(idx1, idx0[i]);
        int idx = find_index(idx0, idx1[i]);
        out[i] = idx;
    }
    return out;
}

/* This meta function is used for explicit einsum when the user explicitly sets the type of the output.
    In such cases the einsum is followed by a permutation for instance [einsum<Index<l,i,k,l,j>,OIndex<i,k,j>>(a)].
    In the above example the resulting index from einsum is already Index<i,k,j> but the user decides to force this
    nevertheless using OIndex<i,k,j>. Now if permuation is simply performed using OIndex<i,k,j> the final output would
    be incorrect since permuation only works with what indices it is given and does not have the knowledge of the context.
    This meta function is responsible for creating a mapping between einsum and permutation. It takes the resulting index
    of einsum and the permutation index [the output index OIndex] and figures out how the tensor should be permuted.
*/
template<typename Ind0, typename Ind1, typename seq>
struct permute_mapped_index_impl;

// // specialisation for when the input and output indices are the same - the no permutation case
// template<size_t ... Idx0, size_t ... ss>
// struct permute_mapped_index_impl<Index<Idx0...>,Index<Idx0...>,std_ext::index_sequence<ss...>> {
//     using resulting_index = Index<ss...>;
// };

template<size_t ... Idx0, size_t ... Idx1, size_t ... ss>
struct permute_mapped_index_impl<Index<Idx0...>,Index<Idx1...>,std_ext::index_sequence<ss...>> {
    constexpr static size_t einsum_idx[sizeof...(Idx0)]         = { Idx0... };
    constexpr static size_t to_be_permuted_idx[sizeof...(Idx1)] = { Idx1... };
    /* Given that the resulting index from einsum can be discontinuous like Index<5,8,2> we need to first create a continuous
        Index starting from zero. This is done through [get_floor_map] function. We floor both resulting einsum and output
        indices and we then call the [find_permuation] function to know how many swaps/permutations are necessary going from
        the resulting einsum index to the output index
    */
    static constexpr std::array<size_t, sizeof...(Idx0)> argsort_idx0 = meta_argsort<Index<Idx0...>,Index<ss...>>::new_argseq::values;
    static constexpr std::array<size_t, sizeof...(Idx1)> argsort_idx1 = meta_argsort<Index<Idx1...>,Index<ss...>>::new_argseq::values;
    static constexpr std::array<size_t, sizeof...(Idx0)> mapped_idx0 = get_floor_map(argsort_idx0);
    static constexpr std::array<size_t, sizeof...(Idx1)> mapped_idx1 = get_floor_map(argsort_idx1);
    static constexpr std::array<size_t, sizeof...(Idx1)> mapped_idx = find_permuation(mapped_idx0,mapped_idx1);
    using resulting_index = Index<mapped_idx[ss]...>;
};

/* Provided for debugging */
template<size_t ... Idx0, size_t ... Idx1, size_t ... ss>
constexpr std::array<size_t, sizeof...(Idx0)>
permute_mapped_index_impl<Index<Idx0...>,Index<Idx1...>,std_ext::index_sequence<ss...>>::argsort_idx0;
template<size_t ... Idx0, size_t ... Idx1, size_t ... ss>
constexpr std::array<size_t, sizeof...(Idx1)>
permute_mapped_index_impl<Index<Idx0...>,Index<Idx1...>,std_ext::index_sequence<ss...>>::argsort_idx1;
template<size_t ... Idx0, size_t ... Idx1, size_t ... ss>
constexpr std::array<size_t, sizeof...(Idx0)>
permute_mapped_index_impl<Index<Idx0...>,Index<Idx1...>,std_ext::index_sequence<ss...>>::mapped_idx0;
template<size_t ... Idx0, size_t ... Idx1, size_t ... ss>
constexpr std::array<size_t, sizeof...(Idx1)>
permute_mapped_index_impl<Index<Idx0...>,Index<Idx1...>,std_ext::index_sequence<ss...>>::mapped_idx1;
template<size_t ... Idx0, size_t ... Idx1, size_t ... ss>
constexpr std::array<size_t, sizeof...(Idx1)>
permute_mapped_index_impl<Index<Idx0...>,Index<Idx1...>,std_ext::index_sequence<ss...>>::mapped_idx;


template<typename Ind0, typename Ind1>
struct permute_mapped_index {
    using resulting_index = typename permute_mapped_index_impl<Ind0,Ind1,
        typename std_ext::make_index_sequence<Ind0::Size>::type>::resulting_index;
};

template<typename Ind0, typename Ind1>
using permute_mapped_index_t = typename permute_mapped_index<Ind0,Ind1>::resulting_index;

#endif // CXX 2017
//------------------------------------------------------------------------------------------------------------//

} // internal


template<class Idx, class Tens>
struct requires_permutation;
template<typename T, size_t ... Idx, size_t ... Rest>
struct requires_permutation<Index<Idx...>, Tensor<T, Rest...>> {
    using _permute_impl = internal::permute_impl<Index<Idx...>, Tensor<T,Rest...>,
            typename std_ext::make_index_sequence<sizeof...(Idx)>::type>;
    static constexpr bool value = _permute_impl::requires_permutation;
};

// helper
template<class Idx, class Tens>
constexpr bool requires_permutation_v = requires_permutation<Idx,Tens>::value;


template<class Idx, class Tens>
struct requires_permute;
template<typename T, size_t ... Idx, size_t ... Rest>
struct requires_permute<Index<Idx...>, Tensor<T, Rest...>> {
    using _permute_impl = internal::new_permute_impl<Index<Idx...>, Tensor<T,Rest...>,
            typename std_ext::make_index_sequence<sizeof...(Idx)>::type>;
    static constexpr bool value = _permute_impl::requires_permutation;
};

// helper
template<class Idx, class Tens>
constexpr bool requires_permute_v = requires_permute<Idx,Tens>::value;
//------------------------------------------------------------------------------------------------------------//


// einsum helper to extract the resulting index and the resulting tensor
//------------------------------------------------------------------------------------------------------------//
template<typename ...Ts>
struct permute_helper;

template<class Index_I,
         typename T, size_t ... Rest0>
struct permute_helper<Index_I,Tensor<T,Rest0...>> {
    using resulting_index  = typename internal::new_permute_impl<Index_I, Tensor<T,Rest0...>,
        typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::resulting_index;
    using resulting_tensor = typename internal::new_permute_impl<Index_I, Tensor<T,Rest0...>,
        typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::resulting_tensor;
};
//------------------------------------------------------------------------------------------------------------//



}

#endif // EINSUM_META_H

