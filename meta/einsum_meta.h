#ifndef EINSUM_META_H
#define EINSUM_META_H

#include "tensor/Tensor.h"

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
// triggers a compile-time error when used in a constant expression on mismatch.
template<size_t N>
constexpr int calc(size_t i, const int (&ind)[N], const int (&dim)[N]){
    return is_uniq(ind, i) ? dim[i] :
           check_all_eq(ind[i], dim[i], ind, dim) ? 1001001 : throw "dimension mismatch";
}

// if position i should be contracted away, return 1001001, otherwise return ind[i].
// triggers a compile-time error when used in a constant expression on mismatch.
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

// filter out all instances of I from Is...,
// return the rest as an Indices
template<int I, int... Is>
struct filter_
    :  concat_<typename std::conditional<Is == I, Index<>, Index<Is>>::type...> {};
//Use them:
template<class Ind, class Arr, class Seq>
struct contraction_impl;

template<class T, size_t... Ind, size_t... Dim, size_t... Seq>
struct contraction_impl<Index<Ind...>, Tensor<T, Dim...>, std_ext::index_sequence<Seq...>>{
    static constexpr int ind[sizeof...(Ind)] = { Ind... };
    static constexpr int dim[sizeof...(Dim)] = { Dim... };
    static constexpr int result[sizeof...(Seq)] = {calc(Seq, ind, dim)...};

    template<size_t... Dims>
    static auto unpack_helper(Index<Dims...>) -> Tensor<T, Dims...>;

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






// Auxilary meta functions for tensor contraction
//------------------------------------------------------------------------------------------------------------//
// this is a meta-function equivalent to numpy's "where"
template<size_t N>
constexpr int find_index(const size_t (&ind)[N], int num, size_t i=0){
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

template<size_t ... Rest, typename T>
struct put_dims_in_Index<Tensor<T, Rest...>> {
    using type = Index<Rest...>;
};
//------------------------------------------------------------------------------------------------------------//


//! Checks vectorisability and returns a stride and a type
//------------------------------------------------------------------------------------------------------------//
template<class Idx0, class Idx1, class Tens>
struct is_vectorisable;

template<size_t ...Idx0, size_t ...Idx1, size_t...Rest>
struct is_vectorisable<Index<Idx0...>,Index<Idx1...>,Tensor<float,Rest...>> {
    static constexpr size_t fastest_changing_index = get_value<sizeof...(Rest),Rest...>::value;
    static constexpr size_t idx[sizeof...(Idx0)] = {Idx0...};
    static constexpr bool last_index_contracted = contains(idx,get_value<sizeof...(Idx1),Idx1...>::value);
    static constexpr bool value = (!last_index_contracted) && (fastest_changing_index % 4==0);
    static constexpr bool sse_vectorisability = (!last_index_contracted) && (fastest_changing_index % 4==0 && fastest_changing_index % 8!=0);
    static constexpr bool avx_vectorisability = (!last_index_contracted) && (fastest_changing_index % 4==0 && fastest_changing_index % 8==0);
    static constexpr int stride = (avx_vectorisability ? 8 : (sse_vectorisability ? 4 : 1));

    using type = typename std::conditional<avx_vectorisability,SIMDVector<float,256>,
        typename std::conditional<sse_vectorisability,SIMDVector<float,128>,SIMDVector<float,32>>::type>::type;
};

template<size_t ...Idx0, size_t ...Idx1, size_t...Rest>
struct is_vectorisable<Index<Idx0...>,Index<Idx1...>,Tensor<double,Rest...>> {
    static constexpr size_t fastest_changing_index = get_value<sizeof...(Rest),Rest...>::value;
    static constexpr size_t idx[sizeof...(Idx0)] = {Idx0...};
    static constexpr bool last_index_contracted = contains(idx,get_value<sizeof...(Idx1),Idx1...>::value);
    static constexpr bool value = (!last_index_contracted) && (fastest_changing_index % 2==0);
    static constexpr bool sse_vectorisability = (!last_index_contracted) && (fastest_changing_index % 2==0 && fastest_changing_index % 4!=0);
    static constexpr bool avx_vectorisability = (!last_index_contracted) && (fastest_changing_index % 2==0 && fastest_changing_index % 4==0);
    static constexpr int stride = (avx_vectorisability ? 4 : (sse_vectorisability ? 2 : 1));

    using type = typename std::conditional<avx_vectorisability,SIMDVector<double,256>,
        typename std::conditional<sse_vectorisability,SIMDVector<double,128>,SIMDVector<double,64>>::type>::type;
};
//------------------------------------------------------------------------------------------------------------//




// Cost model for by-pair tensor contraction
//------------------------------------------------------------------------------------------------------------//
template<class Ind0, class Ind1, class Tensor0, class Tensor1, class Seq>
struct pair_flop_cost;

template<class T, size_t... Idx0, size_t... Idx1, size_t ...Rest0, size_t ...Rest1, size_t... ss>
struct pair_flop_cost<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>> {

    static constexpr size_t ind0[sizeof...(Idx0)] = {Idx0... };
    static constexpr size_t ind1[sizeof...(Idx1)] = {Idx1... };
    static constexpr int nums0[sizeof...(Rest0)] = {Rest0... };
    static constexpr int nums1[sizeof...(Rest1)] = {Rest1... };
    static constexpr int cost_tensor0 = prod<Rest0...>::value;
    static constexpr int remaining_cost = prod<retrieve_value(ind0,ind1,nums1,ss)...>::value;
    static constexpr int value = cost_tensor0*remaining_cost;
};
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



// How many loops should be set up
//------------------------------------------------------------------------------------------------------------//
template<class T, class U, class V, class W, class Seq>
struct no_of_loops_to_set;

template<size_t ... Idx0, size_t ... Idx1, size_t ... Rest0, size_t ... Rest1, typename T, size_t ... ss>
struct no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>> {

    using index_temp = apply_typelist_t<quote_c<size_t, Index>,
                    uniq_t<typelist_c<size_t, Idx0...,Idx1...>>>;

    static constexpr size_t concat_idx[sizeof...(Idx0)+sizeof...(Idx1)] = {Idx0...,Idx1...};
    static constexpr size_t concat_nums[sizeof...(Idx0)+sizeof...(Idx1)] = {Rest0...,Rest1...};
    static constexpr std::array<size_t,sizeof...(ss)> idx_in_concat = {find_index(concat_idx,index_temp::_IndexHolder[ss])...};
    static constexpr std::array<size_t,sizeof...(ss)> dims = {concat_nums[idx_in_concat[ss]]...};
    static constexpr int value = prod<dims[ss]...>::value;

    using type = Tensor<T,dims[ss]...>;
    using indices = Index<index_temp::_IndexHolder[ss]...>;

    using dims_type = Index<dims[ss]...>;
};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Rest0, size_t ... Rest1, typename T, size_t ... ss>
constexpr std::array<size_t,sizeof...(ss)>
no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>>::dims;
//------------------------------------------------------------------------------------------------------------//




// Cost model for triplet tensor contraction
//------------------------------------------------------------------------------------------------------------//
template<class Ind0, class Ind1, class Ind2, class Tensor0, class Tensor1, class Tensor2>
struct triplet_flop_cost;

template<class T, size_t... Idx0, size_t... Idx1, size_t... Idx2, size_t ...Rest0, size_t ...Rest1, size_t ...Rest2>
struct triplet_flop_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,
        Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>> {

    using concat_tensor_01 = typename no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
            typename std_ext::make_index_sequence<no_of_unique<Idx0...,Idx1...>::value>::type>::type;

    using concat_index_01 = typename no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
            typename std_ext::make_index_sequence<no_of_unique<Idx0...,Idx1...>::value>::type>::indices;

    static constexpr int value = pair_flop_cost<concat_index_01,Index<Idx2...>,concat_tensor_01,Tensor<T,Rest2...>,
                    typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::value;
};
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
    indices = {find_index(index_temp::_IndexHolder, idx[ss])...};

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
    indices = {find_index(index_temp::_IndexHolder, idx[ss])...};

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
    indices = {find_index(index_temp::_IndexHolder, resulting_index_0::_IndexHolder[ss])...};

    using type = Tensor<T,indices[ss]...>;
};

template<class T, size_t... Idx0, size_t... Idx1, size_t ...Rest0, size_t ...Rest1, size_t ... ss>
constexpr std::array<size_t,sizeof...(ss)>
IndexResultingTensor<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>>::indices;
//------------------------------------------------------------------------------------------------------------//



//------------------------------------------------------------------------------------------------------------//
template<class Idx0, class Idx1>
struct is_reduction;

template<size_t ... Idx0, size_t ... Idx1>
struct is_reduction<Index<Idx0...>,Index<Idx1...>> {
    static constexpr bool size_check = sizeof...(Idx0)==sizeof...(Idx1);
    static constexpr bool index_check = no_of_unique<Idx0...,Idx1...>::value==no_of_unique<Idx1...>::value;
    static constexpr bool value = size_check && index_check;
};

















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
    static constexpr std::array<size_t,sizeof...(ss)> idx_in_concat = {find_index(concat_idx,index_temp::_IndexHolder[ss])...};
    static constexpr std::array<size_t,sizeof...(ss)> dims = {concat_nums[idx_in_concat[ss]]...};
    static constexpr int value = prod<dims[ss]...>::value;

    using type = Tensor<T,dims[ss]...>;
    using indices = Index<index_temp::_IndexHolder[ss]...>;

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

template<class T, size_t... Idx, size_t... Idx_t, size_t ...Rest, size_t ...Rest_t, size_t ... ss>
struct IndexTensors<Index<Idx...>,Tensor<T,Rest...>,Index<Idx_t...>,Tensor<T,Rest_t...>,std_ext::index_sequence<ss...>> {

    using index_temp = typename loop_setter<Index<Idx...>,Tensor<T,Rest...>,
            typename std_ext::make_index_sequence<no_of_unique<Idx...>::value>::type>::indices;

    static constexpr size_t idx[sizeof...(Idx_t)] = {Idx_t...};
    static constexpr std::array<size_t,sizeof...(Idx_t)>
    indices = {find_index(index_temp::_IndexHolder, idx[ss])...};

    using type = Tensor<T,indices[ss]...>;

};

template<class T, size_t... Idx, size_t... Idx_t, size_t ...Rest, size_t ...Rest_t, size_t ... ss>
constexpr std::array<size_t,sizeof...(Idx_t)>
IndexTensors<Index<Idx...>,Tensor<T,Rest...>,Index<Idx_t...>,Tensor<T,Rest_t...>,std_ext::index_sequence<ss...>>::indices;
//------------------------------------------------------------------------------------------------------------//


//}

}

#endif // EINSUM_META_H

