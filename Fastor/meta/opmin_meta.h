#ifndef OPMIN_META_H
#define OPMIN_META_H

#ifndef FASTOR_DONT_PERFORM_OP_MIN

#include "tensor_meta.h"
#include "einsum_meta.h"

namespace Fastor {


// Cost model for by-pair tensor contraction
//------------------------------------------------------------------------------------------------------------//
template<class Ind0, class Ind1, class Tensor0, class Tensor1, class Seq>
struct pair_flop_cost;

template<class T, size_t... Idx0, size_t... Idx1, size_t ...Rest0, size_t ...Rest1, size_t... ss>
struct pair_flop_cost<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>> {

    static constexpr size_t ind0[sizeof...(Idx0)] = {Idx0... };
    static constexpr size_t ind1[sizeof...(Idx1)] = {Idx1... };
    static constexpr int nums1[sizeof...(Rest1)] = {Rest1... };
    static constexpr size_t cost_tensor0 = pack_prod<Rest0...>::value;
    static constexpr size_t remaining_cost = pack_prod<retrieve_value(ind0,ind1,nums1,ss)...>::value;
    static constexpr size_t value = cost_tensor0*remaining_cost;

    using resulting_tensor =  typename get_resuling_tensor<Index<Idx0...>,Index<Idx1...>,
                                    Tensor<T,Rest0...>,Tensor<T,Rest1...>>::type;
    using resulting_index =  typename get_resuling_index<Index<Idx0...>,Index<Idx1...>,
                                    Tensor<T,Rest0...>,Tensor<T,Rest1...>>::type;
};


template<class T, size_t... Idx0, size_t ...Rest0, size_t... ss>
struct pair_flop_cost<Index<Idx0...>,Index<>,Tensor<T,Rest0...>,Tensor<T>,std_ext::index_sequence<ss...>> {
    static constexpr size_t value = pack_prod<Rest0...>::value;
};

template<class T, size_t... Idx1, size_t ...Rest1, size_t... ss>
struct pair_flop_cost<Index<>,Index<Idx1...>,Tensor<T>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>> {
    static constexpr size_t value = pack_prod<Rest1...>::value;
};

template<class T, size_t... ss>
struct pair_flop_cost<Index<>,Index<>,Tensor<T>,Tensor<T>,std_ext::index_sequence<ss...>> {
    static constexpr size_t value = 1;
};
//------------------------------------------------------------------------------------------------------------//



// Cost of triplet tensor contraction (single evaluation)
//------------------------------------------------------------------------------------------------------------//
template<class Ind0, class Ind1, class Ind2, class Tensor0, class Tensor1, class Tensor2>
struct single_evaluation_triplet_cost;

template<class T, size_t... Idx0, size_t... Idx1, size_t... Idx2, size_t ...Rest0, size_t ...Rest1, size_t ...Rest2>
struct single_evaluation_triplet_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,
        Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>> {

    using concat_tensor_01 = typename no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
            typename std_ext::make_index_sequence<no_of_unique<Idx0...,Idx1...>::value>::type>::type;

    using concat_index_01 = typename no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
            typename std_ext::make_index_sequence<no_of_unique<Idx0...,Idx1...>::value>::type>::indices;

    static constexpr size_t value = pair_flop_cost<concat_index_01,Index<Idx2...>,concat_tensor_01,Tensor<T,Rest2...>,
                    typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::value;
};
//------------------------------------------------------------------------------------------------------------//




// Cost model for triplet tensor network contraction
//------------------------------------------------------------------------------------------------------------//
template<class Ind0, class Ind1, class Ind2, class Tensor0, class Tensor1, class Tensor2>
struct triplet_flop_cost;

template<class T, size_t... Idx0, size_t... Idx1, size_t... Idx2, size_t ...Rest0, size_t ...Rest1, size_t ...Rest2>
struct triplet_flop_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,
        Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>> {

    // Perform depth-first search
    //---------------------------------------------------------------------
    // first two tensors contracted first
    using resulting_tensor_0 =  typename get_resuling_tensor<Index<Idx0...>,Index<Idx1...>,
                                    Tensor<T,Rest0...>,Tensor<T,Rest1...>>::type;
    using resulting_index_0 =  typename get_resuling_index<Index<Idx0...>,Index<Idx1...>,
                                    Tensor<T,Rest0...>,Tensor<T,Rest1...>>::type;

    static constexpr size_t flop_count_01_0 = pair_flop_cost<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
            typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::value;

    static constexpr size_t flop_count_01_1 = pair_flop_cost<resulting_index_0,Index<Idx2...>,resulting_tensor_0,Tensor<T,Rest2...>,
            typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::value;

    static constexpr size_t flop_count_01 = flop_count_01_0 + flop_count_01_1;


    // first and last tensors contracted first
    using resulting_tensor_1 =  typename get_resuling_tensor<Index<Idx0...>,Index<Idx2...>,
                                    Tensor<T,Rest0...>,Tensor<T,Rest2...>>::type;
    using resulting_index_1 =  typename get_resuling_index<Index<Idx0...>,Index<Idx2...>,
                                    Tensor<T,Rest0...>,Tensor<T,Rest2...>>::type;

    static constexpr size_t flop_count_02_0 = pair_flop_cost<Index<Idx0...>,Index<Idx2...>,Tensor<T,Rest0...>,Tensor<T,Rest2...>,
            typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::value;

    static constexpr size_t flop_count_02_1 = pair_flop_cost<resulting_index_1,Index<Idx1...>,resulting_tensor_1,Tensor<T,Rest1...>,
            typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::value;

    static constexpr size_t flop_count_02 = flop_count_02_0 + flop_count_02_1;


    // second and last tensors contracted first
    using resulting_tensor_2 =  typename get_resuling_tensor<Index<Idx1...>,Index<Idx2...>,
                                    Tensor<T,Rest1...>,Tensor<T,Rest2...>>::type;
    using resulting_index_2 =  typename get_resuling_index<Index<Idx1...>,Index<Idx2...>,
                                    Tensor<T,Rest1...>,Tensor<T,Rest2...>>::type;

    static constexpr size_t flop_count_12_0 = pair_flop_cost<Index<Idx1...>,Index<Idx2...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,
            typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::value;

    static constexpr size_t flop_count_12_1 = pair_flop_cost<resulting_index_2,Index<Idx0...>,resulting_tensor_2,Tensor<T,Rest0...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::value;

    static constexpr size_t flop_count_12 = flop_count_12_0 + flop_count_12_1;

    static constexpr size_t flop_count_012 = single_evaluation_triplet_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,
            Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>>::value;

    static constexpr size_t min_cost = meta_min<flop_count_01,flop_count_02,flop_count_12,flop_count_012>::value;
    static constexpr int which_variant = meta_argmin<flop_count_01,flop_count_02,flop_count_12,flop_count_012>::value;

    // this is the overall resulting tensor and index from overall triplet contraction
    using resulting_tensor = typename std::conditional<
        which_variant==0,
        typename get_resuling_tensor<resulting_index_0,Index<Idx2...>,resulting_tensor_0,Tensor<T,Rest2...>>::type,
        typename std::conditional<
            which_variant==1,
            typename get_resuling_tensor<Index<Idx1...>,resulting_index_1,Tensor<T,Rest1...>,resulting_tensor_1>::type,
            typename get_resuling_tensor<Index<Idx0...>,resulting_index_2,Tensor<T,Rest0...>,resulting_tensor_2>::type
            >::type
        >::type;

    using resulting_index = typename std::conditional<
        which_variant==0,
        typename get_resuling_index<resulting_index_0,Index<Idx2...>,resulting_tensor_0,Tensor<T,Rest2...>>::type,
        typename std::conditional<
            which_variant==1,
            typename get_resuling_index<Index<Idx1...>,resulting_index_1,Tensor<T,Rest1...>,resulting_tensor_1>::type,
            typename get_resuling_index<Index<Idx0...>,resulting_index_2,Tensor<T,Rest0...>,resulting_tensor_2>::type
            >::type
        >::type;

};
//------------------------------------------------------------------------------------------------------------//





// Cost model for quartet tensor network contraction
//------------------------------------------------------------------------------------------------------------//
template<class Ind0, class Ind1, class Ind2, class Ind3, class Tensor0, class Tensor1, class Tensor2, class Tensor3>
struct quartet_flop_cost;

template<class T, size_t... Idx0, size_t... Idx1, size_t... Idx2, size_t... Idx3,
         size_t ...Rest0, size_t ...Rest1, size_t ...Rest2, size_t ...Rest3>
struct quartet_flop_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,Index<Idx3...>,
        Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>> {

    // Perform depth-first search
    //---------------------------------------------------------------------
    // first three tensors contracted first
    using triplet_cost_012 = triplet_flop_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,
            Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>>;

    using resulting_tensor_0 = typename triplet_cost_012::resulting_tensor;
    using resulting_index_0  = typename triplet_cost_012::resulting_index;

    static constexpr size_t flop_count_012 = triplet_cost_012::min_cost;
    static constexpr size_t flop_count_012_3 = pair_flop_cost<resulting_index_0,Index<Idx3...>,resulting_tensor_0,Tensor<T,Rest3...>,
            typename std_ext::make_index_sequence<sizeof...(Rest3)>::type>::value;

    static constexpr size_t flop_count_0 = flop_count_012 + flop_count_012_3;


    // first, second and last tensors contracted first
    using triplet_cost_013 = triplet_flop_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx3...>,
            Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest3...>>;

    using resulting_tensor_1 = typename triplet_cost_013::resulting_tensor;
    using resulting_index_1  = typename triplet_cost_013::resulting_index;

    static constexpr size_t flop_count_013 = triplet_cost_013::min_cost;
    static constexpr size_t flop_count_013_2 = pair_flop_cost<resulting_index_1,Index<Idx2...>,resulting_tensor_1,Tensor<T,Rest2...>,
            typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::value;

    static constexpr size_t flop_count_1 = flop_count_013 + flop_count_013_2;


    // first, third and last tensors contracted first
    using triplet_cost_023 = triplet_flop_cost<Index<Idx0...>,Index<Idx2...>,Index<Idx3...>,
            Tensor<T,Rest0...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>>;

    using resulting_tensor_2 = typename triplet_cost_023::resulting_tensor;
    using resulting_index_2  = typename triplet_cost_023::resulting_index;

    static constexpr size_t flop_count_023 = triplet_cost_023::min_cost;
    static constexpr size_t flop_count_023_1 = pair_flop_cost<resulting_index_2,Index<Idx1...>,resulting_tensor_2,Tensor<T,Rest1...>,
            typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::value;

    static constexpr size_t flop_count_2 = flop_count_023 + flop_count_023_1;


    // last three tensors contracted first
    using triplet_cost_123 = triplet_flop_cost<Index<Idx1...>,Index<Idx2...>,Index<Idx3...>,
            Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>>;

    using resulting_tensor_3 = typename triplet_cost_123::resulting_tensor;
    using resulting_index_3  = typename triplet_cost_123::resulting_index;

    static constexpr size_t flop_count_123 = triplet_cost_123::min_cost;
    static constexpr size_t flop_count_123_0 = pair_flop_cost<resulting_index_3,Index<Idx0...>,resulting_tensor_3,Tensor<T,Rest0...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::value;

    static constexpr size_t flop_count_3 = flop_count_123 + flop_count_123_0;

    static constexpr size_t min_cost = meta_min<flop_count_0,flop_count_1,flop_count_2,flop_count_3>::value;
    static constexpr int which_variant = meta_argmin<flop_count_0,flop_count_1,flop_count_2,flop_count_3>::value;


    // this is the overall resulting tensor and index from overall quartet contraction
    using resulting_tensor = typename std::conditional<
        which_variant==0,
        typename get_resuling_tensor<resulting_index_0,Index<Idx3...>,resulting_tensor_0,Tensor<T,Rest3...>>::type,
        typename std::conditional<
            which_variant==1,
            typename get_resuling_tensor<Index<Idx2...>,resulting_index_1,Tensor<T,Rest2...>,resulting_tensor_1>::type,
            typename std::conditional<
                which_variant==2,
                typename get_resuling_tensor<Index<Idx1...>,resulting_index_2,Tensor<T,Rest1...>,resulting_tensor_2>::type,
                typename get_resuling_tensor<Index<Idx0...>,resulting_index_3,Tensor<T,Rest0...>,resulting_tensor_3>::type
                >::type
            >::type
        >::type;

    using resulting_index = typename std::conditional<
        which_variant==0,
        typename get_resuling_index<resulting_index_0,Index<Idx3...>,resulting_tensor_0,Tensor<T,Rest3...>>::type,
        typename std::conditional<
            which_variant==1,
            typename get_resuling_index<Index<Idx2...>,resulting_index_1,Tensor<T,Rest2...>,resulting_tensor_1>::type,
            typename std::conditional<
                which_variant==2,
                typename get_resuling_index<Index<Idx1...>,resulting_index_2,Tensor<T,Rest1...>,resulting_tensor_2>::type,
                typename get_resuling_index<Index<Idx0...>,resulting_index_3,Tensor<T,Rest0...>,resulting_tensor_3>::type
                >::type
            >::type
        >::type;

};
//------------------------------------------------------------------------------------------------------------//






// Cost model for quintet tensor contraction
//------------------------------------------------------------------------------------------------------------//
template<class Ind0, class Ind1, class Ind2, class Ind3, class Ind4,
         class Tensor0, class Tensor1, class Tensor2, class Tensor3, class Tensor4>
struct quintet_flop_cost;

template<class T, size_t... Idx0, size_t... Idx1, size_t... Idx2, size_t... Idx3, size_t... Idx4,
         size_t ...Rest0, size_t ...Rest1, size_t ...Rest2, size_t ...Rest3, size_t ...Rest4>
struct quintet_flop_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,Index<Idx3...>, Index<Idx4...>,
        Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>,Tensor<T,Rest4...>> {

    // Perform depth-first search
    //---------------------------------------------------------------------
    // first four tensors contracted first
    using quartet_cost_0123 = quartet_flop_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,Index<Idx3...>,
            Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>>;

    using resulting_tensor_0 = typename quartet_cost_0123::resulting_tensor;
    using resulting_index_0  = typename quartet_cost_0123::resulting_index;

    static constexpr size_t flop_count_0123 = quartet_cost_0123::min_cost;
    static constexpr size_t flop_count_0123_4 = pair_flop_cost<resulting_index_0,Index<Idx4...>,resulting_tensor_0,Tensor<T,Rest4...>,
            typename std_ext::make_index_sequence<sizeof...(Rest4)>::type>::value;

    static constexpr size_t flop_count_0 = flop_count_0123 + flop_count_0123_4;


    // 1st, 2nd, 3rd, 5th tensors contracted first
    using quartet_cost_0124 = quartet_flop_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,Index<Idx4...>,
            Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest4...>>;

    using resulting_tensor_1 = typename quartet_cost_0124::resulting_tensor;
    using resulting_index_1  = typename quartet_cost_0124::resulting_index;

    static constexpr size_t flop_count_0124 = quartet_cost_0124::min_cost;
    static constexpr size_t flop_count_0124_3 = pair_flop_cost<resulting_index_1,Index<Idx3...>,resulting_tensor_1,Tensor<T,Rest3...>,
            typename std_ext::make_index_sequence<sizeof...(Rest3)>::type>::value;

    static constexpr size_t flop_count_1 = flop_count_0124 + flop_count_0124_3;


    // 1st, 2nd, 4th, 5th tensors contracted first
    using quartet_cost_0134 = quartet_flop_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx3...>,Index<Idx4...>,
            Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest3...>,Tensor<T,Rest4...>>;

    using resulting_tensor_2 = typename quartet_cost_0134::resulting_tensor;
    using resulting_index_2  = typename quartet_cost_0134::resulting_index;

    static constexpr size_t flop_count_0134 = quartet_cost_0134::min_cost;
    static constexpr size_t flop_count_0134_2 = pair_flop_cost<resulting_index_2,Index<Idx2...>,resulting_tensor_2,Tensor<T,Rest2...>,
            typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::value;

    static constexpr size_t flop_count_2 = flop_count_0134 + flop_count_0134_2;


    // 1st, 3rd, 4th, 5th tensors contracted first
    using quartet_cost_0234 = quartet_flop_cost<Index<Idx0...>,Index<Idx2...>,Index<Idx3...>,Index<Idx4...>,
            Tensor<T,Rest0...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>,Tensor<T,Rest4...>>;

    using resulting_tensor_3 = typename quartet_cost_0234::resulting_tensor;
    using resulting_index_3  = typename quartet_cost_0234::resulting_index;

    static constexpr size_t flop_count_0234 = quartet_cost_0234::min_cost;
    static constexpr size_t flop_count_0234_1 = pair_flop_cost<resulting_index_3,Index<Idx1...>,resulting_tensor_3,Tensor<T,Rest1...>,
            typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::value;

    static constexpr size_t flop_count_3 = flop_count_0234 + flop_count_0234_1;


    // last four tensors contracted first
    using quartet_cost_1234 = quartet_flop_cost<Index<Idx1...>,Index<Idx2...>,Index<Idx3...>,Index<Idx4...>,
            Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>,Tensor<T,Rest4...>>;

    using resulting_tensor_4 = typename quartet_cost_1234::resulting_tensor;
    using resulting_index_4  = typename quartet_cost_1234::resulting_index;

    static constexpr size_t flop_count_1234 = quartet_cost_1234::min_cost;
    static constexpr size_t flop_count_1234_0 = pair_flop_cost<resulting_index_4,Index<Idx0...>,resulting_tensor_4,Tensor<T,Rest0...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::value;

    static constexpr size_t flop_count_4 = flop_count_1234 + flop_count_1234_0;


    static constexpr size_t min_cost = meta_min<flop_count_0,flop_count_1,flop_count_2,flop_count_3,flop_count_4>::value;
    static constexpr int which_variant = meta_argmin<flop_count_0,flop_count_1,flop_count_2,flop_count_3,flop_count_4>::value;


    // this is the overall resulting tensor and index from overall quintet contraction
    using resulting_tensor = typename std::conditional<
        which_variant==0,
        typename get_resuling_tensor<resulting_index_0,Index<Idx4...>,resulting_tensor_0,Tensor<T,Rest4...>>::type,
        typename std::conditional<
            which_variant==1,
            typename get_resuling_tensor<Index<Idx3...>,resulting_index_1,Tensor<T,Rest3...>,resulting_tensor_1>::type,
            typename std::conditional<
                which_variant==2,
                typename get_resuling_tensor<Index<Idx2...>,resulting_index_2,Tensor<T,Rest2...>,resulting_tensor_2>::type,
                typename std::conditional<
                    which_variant==3,
                    typename get_resuling_tensor<Index<Idx1...>,resulting_index_3,Tensor<T,Rest1...>,resulting_tensor_3>::type,
                    typename get_resuling_tensor<Index<Idx0...>,resulting_index_4,Tensor<T,Rest0...>,resulting_tensor_4>::type
                    >::type
                >::type
            >::type
        >::type;

    using resulting_index = typename std::conditional<
        which_variant==0,
        typename get_resuling_index<resulting_index_0,Index<Idx4...>,resulting_tensor_0,Tensor<T,Rest4...>>::type,
        typename std::conditional<
            which_variant==1,
            typename get_resuling_index<Index<Idx3...>,resulting_index_1,Tensor<T,Rest3...>,resulting_tensor_1>::type,
            typename std::conditional<
                which_variant==2,
                typename get_resuling_index<Index<Idx2...>,resulting_index_2,Tensor<T,Rest2...>,resulting_tensor_2>::type,
                typename std::conditional<
                    which_variant==3,
                    typename get_resuling_index<Index<Idx1...>,resulting_index_3,Tensor<T,Rest1...>,resulting_tensor_3>::type,
                    typename get_resuling_index<Index<Idx0...>,resulting_index_4,Tensor<T,Rest0...>,resulting_tensor_4>::type
                    >::type
                >::type
            >::type
        >::type;

};
//------------------------------------------------------------------------------------------------------------//




// Cost model for sixtet tensor contraction
//------------------------------------------------------------------------------------------------------------//
template<class Ind0, class Ind1, class Ind2, class Ind3, class Ind4, class Ind5,
         class Tensor0, class Tensor1, class Tensor2, class Tensor3, class Tensor4, class Tensor5>
struct sixtet_flop_cost;

template<class T, size_t... Idx0, size_t... Idx1, size_t... Idx2, size_t... Idx3, size_t... Idx4, size_t... Idx5,
         size_t ...Rest0, size_t ...Rest1, size_t ...Rest2, size_t ...Rest3, size_t ...Rest4, size_t ...Rest5>
struct sixtet_flop_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,Index<Idx3...>, Index<Idx4...>, Index<Idx5...>,
        Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>,Tensor<T,Rest4...>,Tensor<T,Rest5...>> {

    // Perform depth-first search
    //---------------------------------------------------------------------
    // first 5 tensors contracted first
    using quintet_cost_01234 = quintet_flop_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,Index<Idx3...>,Index<Idx4...>,
            Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>,Tensor<T,Rest4...>>;

    using resulting_tensor_0 = typename quintet_cost_01234::resulting_tensor;
    using resulting_index_0  = typename quintet_cost_01234::resulting_index;

    static constexpr size_t flop_count_01234 = quintet_cost_01234::min_cost;
    static constexpr size_t flop_count_01234_5 = pair_flop_cost<resulting_index_0,Index<Idx5...>,resulting_tensor_0,Tensor<T,Rest5...>,
            typename std_ext::make_index_sequence<sizeof...(Rest5)>::type>::value;

    static constexpr size_t flop_count_0 = flop_count_01234 + flop_count_01234_5;


    // 1st, 2nd, 3rd, 4th, 6th tensors contracted first
    using quintet_cost_01235 = quintet_flop_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,Index<Idx3...>,Index<Idx5...>,
            Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>,Tensor<T,Rest5...>>;

    using resulting_tensor_1 = typename quintet_cost_01235::resulting_tensor;
    using resulting_index_1  = typename quintet_cost_01235::resulting_index;

    static constexpr size_t flop_count_01235 = quintet_cost_01235::min_cost;
    static constexpr size_t flop_count_01235_4 = pair_flop_cost<resulting_index_1,Index<Idx4...>,resulting_tensor_1,Tensor<T,Rest4...>,
            typename std_ext::make_index_sequence<sizeof...(Rest4)>::type>::value;

    static constexpr size_t flop_count_1 = flop_count_01235 + flop_count_01235_4;


    // 1st, 2nd, 3rd, 5th, 6th tensors contracted first
    using quintet_cost_01245 = quintet_flop_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,Index<Idx4...>,Index<Idx5...>,
            Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest4...>,Tensor<T,Rest5...>>;

    using resulting_tensor_2 = typename quintet_cost_01245::resulting_tensor;
    using resulting_index_2  = typename quintet_cost_01245::resulting_index;

    static constexpr size_t flop_count_01245 = quintet_cost_01235::min_cost;
    static constexpr size_t flop_count_01245_3 = pair_flop_cost<resulting_index_2,Index<Idx3...>,resulting_tensor_2,Tensor<T,Rest3...>,
            typename std_ext::make_index_sequence<sizeof...(Rest3)>::type>::value;

    static constexpr size_t flop_count_2 = flop_count_01245 + flop_count_01245_3;


    // 1st, 2nd, 4th, 5th, 6th tensors contracted first
    using quintet_cost_01345 = quintet_flop_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx3...>,Index<Idx4...>,Index<Idx5...>,
            Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest3...>,Tensor<T,Rest4...>,Tensor<T,Rest5...>>;

    using resulting_tensor_3 = typename quintet_cost_01345::resulting_tensor;
    using resulting_index_3  = typename quintet_cost_01345::resulting_index;

    static constexpr size_t flop_count_01345 = quintet_cost_01235::min_cost;
    static constexpr size_t flop_count_01345_2 = pair_flop_cost<resulting_index_3,Index<Idx2...>,resulting_tensor_3,Tensor<T,Rest2...>,
            typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::value;

    static constexpr size_t flop_count_3 = flop_count_01345 + flop_count_01345_2;


    // 1st, 3rd, 4th, 5th, 6th tensors contracted first
    using quintet_cost_02345 = quintet_flop_cost<Index<Idx0...>,Index<Idx2...>,Index<Idx3...>,Index<Idx4...>,Index<Idx5...>,
            Tensor<T,Rest0...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>,Tensor<T,Rest4...>,Tensor<T,Rest5...>>;

    using resulting_tensor_4 = typename quintet_cost_02345::resulting_tensor;
    using resulting_index_4  = typename quintet_cost_02345::resulting_index;

    static constexpr size_t flop_count_02345 = quintet_cost_01235::min_cost;
    static constexpr size_t flop_count_02345_1 = pair_flop_cost<resulting_index_4,Index<Idx1...>,resulting_tensor_4,Tensor<T,Rest1...>,
            typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::value;

    static constexpr size_t flop_count_4 = flop_count_02345 + flop_count_02345_1;


    // last 5 tensors contracted first
    using quintet_cost_12345 = quintet_flop_cost<Index<Idx1...>,Index<Idx2...>,Index<Idx3...>,Index<Idx4...>,Index<Idx5...>,
            Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>,Tensor<T,Rest4...>,Tensor<T,Rest5...>>;

    using resulting_tensor_5 = typename quintet_cost_12345::resulting_tensor;
    using resulting_index_5  = typename quintet_cost_12345::resulting_index;

    static constexpr size_t flop_count_12345 = quintet_cost_01235::min_cost;
    static constexpr size_t flop_count_12345_0 = pair_flop_cost<resulting_index_5,Index<Idx0...>,resulting_tensor_5,Tensor<T,Rest0...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::value;

    static constexpr size_t flop_count_5 = flop_count_12345 + flop_count_12345_0;


    static constexpr size_t min_cost = meta_min<flop_count_0,flop_count_1,flop_count_2,flop_count_3,flop_count_4,flop_count_5>::value;
    static constexpr int which_variant = meta_argmin<flop_count_0,flop_count_1,flop_count_2,flop_count_3,flop_count_4,flop_count_5>::value;


    // this is the overall resulting tensor and index from overall sextet contraction
    using resulting_tensor = typename std::conditional<
        which_variant==0,
        typename get_resuling_tensor<resulting_index_0,Index<Idx5...>,resulting_tensor_0,Tensor<T,Rest5...>>::type,
        typename std::conditional<
            which_variant==1,
            typename get_resuling_tensor<Index<Idx4...>,resulting_index_1,Tensor<T,Rest4...>,resulting_tensor_1>::type,
            typename std::conditional<
                which_variant==2,
                typename get_resuling_tensor<Index<Idx3...>,resulting_index_2,Tensor<T,Rest3...>,resulting_tensor_2>::type,
                typename std::conditional<
                    which_variant==3,
                    typename get_resuling_tensor<Index<Idx2...>,resulting_index_3,Tensor<T,Rest2...>,resulting_tensor_3>::type,
                    typename std::conditional<
                        which_variant==4,
                        typename get_resuling_tensor<Index<Idx1...>,resulting_index_4,Tensor<T,Rest1...>,resulting_tensor_4>::type,
                        typename get_resuling_tensor<Index<Idx0...>,resulting_index_5,Tensor<T,Rest0...>,resulting_tensor_5>::type
                        >::type
                    >::type
                >::type
            >::type
        >::type;

    using resulting_index = typename std::conditional<
        which_variant==0,
        typename get_resuling_index<resulting_index_0,Index<Idx5...>,resulting_tensor_0,Tensor<T,Rest5...>>::type,
        typename std::conditional<
            which_variant==1,
            typename get_resuling_index<Index<Idx4...>,resulting_index_1,Tensor<T,Rest4...>,resulting_tensor_1>::type,
            typename std::conditional<
                which_variant==2,
                typename get_resuling_index<Index<Idx3...>,resulting_index_2,Tensor<T,Rest3...>,resulting_tensor_2>::type,
                typename std::conditional<
                    which_variant==3,
                    typename get_resuling_index<Index<Idx2...>,resulting_index_3,Tensor<T,Rest2...>,resulting_tensor_3>::type,
                    typename std::conditional<
                        which_variant==4,
                        typename get_resuling_index<Index<Idx1...>,resulting_index_4,Tensor<T,Rest1...>,resulting_tensor_4>::type,
                        typename get_resuling_index<Index<Idx0...>,resulting_index_5,Tensor<T,Rest0...>,resulting_tensor_5>::type
                        >::type
                    >::type
                >::type
            >::type
        >::type;

};
//------------------------------------------------------------------------------------------------------------//





// einsum helper to extract the resulting index and the resulting tensor
//------------------------------------------------------------------------------------------------------------//
template<typename ...Ts>
struct einsum_helper;

template<class Ind0,
         typename T, size_t ... Rest0>
struct einsum_helper<Ind0,Tensor<T,Rest0...>> {
    using resulting_index = typename contraction_impl<Ind0, Tensor<T,Rest0...>,
        typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::indices;
    using resulting_tensor = typename contraction_impl<Ind0, Tensor<T,Rest0...>,
      typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::type;
};

template<class Ind0, class Ind1,
         class Tensor0, class Tensor1>
struct einsum_helper<Ind0,Ind1,Tensor0,Tensor1> {
    using resulting_index  = typename get_resuling_index<Ind0,Ind1,Tensor0,Tensor1>::type;
    using resulting_tensor = typename get_resuling_tensor<Ind0,Ind1,Tensor0,Tensor1>::type;
};

template<class Ind0, class Ind1, class Ind2,
         class Tensor0, class Tensor1, class Tensor2>
struct einsum_helper<Ind0,Ind1,Ind2,Tensor0,Tensor1,Tensor2> {
    using cost_model = triplet_flop_cost<Ind0,Ind1,Ind2,
                                Tensor0,Tensor1,Tensor2>;
    using resulting_index  = typename cost_model::resulting_index;
    using resulting_tensor = typename cost_model::resulting_tensor;
};

template<class Ind0, class Ind1, class Ind2, class Ind3,
         class Tensor0, class Tensor1, class Tensor2, class Tensor3>
struct einsum_helper<Ind0,Ind1,Ind2,Ind3,Tensor0,Tensor1,Tensor2,Tensor3> {
    using cost_model = quartet_flop_cost<Ind0,Ind1,Ind2,Ind3,
                                Tensor0,Tensor1,Tensor2,Tensor3>;
    using resulting_index  = typename cost_model::resulting_index;
    using resulting_tensor = typename cost_model::resulting_tensor;
};

template<class Ind0, class Ind1, class Ind2, class Ind3, class Ind4,
         class Tensor0, class Tensor1, class Tensor2, class Tensor3, class Tensor4>
struct einsum_helper<Ind0,Ind1,Ind2,Ind3,Ind4,Tensor0,Tensor1,Tensor2,Tensor3,Tensor4> {
    using cost_model = quintet_flop_cost<Ind0,Ind1,Ind2,Ind3,Ind4,
                                Tensor0,Tensor1,Tensor2,Tensor3,Tensor4>;
    using resulting_index  = typename cost_model::resulting_index;
    using resulting_tensor = typename cost_model::resulting_tensor;
};

template<class Ind0, class Ind1, class Ind2, class Ind3, class Ind4, class Ind5,
         class Tensor0, class Tensor1, class Tensor2, class Tensor3, class Tensor4, class Tensor5>
struct einsum_helper<Ind0,Ind1,Ind2,Ind3,Ind4,Ind5,Tensor0,Tensor1,Tensor2,Tensor3,Tensor4,Tensor5> {
    using cost_model = sixtet_flop_cost<Ind0,Ind1,Ind2,Ind3,Ind4,Ind5,
                                Tensor0,Tensor1,Tensor2,Tensor3,Tensor4,Tensor5>;
    using resulting_index  = typename cost_model::resulting_index;
    using resulting_tensor = typename cost_model::resulting_tensor;
};
//------------------------------------------------------------------------------------------------------------//


} // end of namespace Fastor


#endif // FASTOR_DONT_PERFORM_OP_MIN


#endif // OPMIN_META_H
