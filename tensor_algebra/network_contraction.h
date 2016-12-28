#ifndef NETWORK_CONTRACTION_H
#define NETWORK_CONTRACTION_H

#include "contraction.h"
#include "strided_contraction.h"

namespace Fastor {


// Three tensor network
//---------------------------------------------------------------------------------------------------------------------//
template<class T, class U, class V>
struct extractor_contract {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2>
struct extractor_contract<Index<Idx0...>, Index<Idx1...>, Index<Idx2...> > {
  template<typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2>
    static
    typename contraction_impl<Index<Idx0...,Idx1...,Idx2...>, Tensor<T,Rest0...,Rest1...,Rest2...>,
                              typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+sizeof...(Rest2)>::type>::type
    contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c) {

        // Perform depth-first search
        //---------------------------------------------------------------------
        // first two tensors contracted first
        using resulting_tensor_0 =  typename get_resuling_tensor<Index<Idx0...>,Index<Idx1...>,
                                        Tensor<T,Rest0...>,Tensor<T,Rest1...>>::type;
        using resulting_index_0 =  typename get_resuling_index<Index<Idx0...>,Index<Idx1...>,
                                        Tensor<T,Rest0...>,Tensor<T,Rest1...>>::type;

        constexpr int flop_count_01_0 = pair_flop_cost<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
                typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::value;

        constexpr int flop_count_01_1 = pair_flop_cost<resulting_index_0,Index<Idx2...>,resulting_tensor_0,Tensor<T,Rest2...>,
                typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::value;

        constexpr int flop_count_01 = flop_count_01_0 + flop_count_01_1;


        // first and last tensors contracted first
        using resulting_tensor_1 =  typename get_resuling_tensor<Index<Idx0...>,Index<Idx2...>,
                                        Tensor<T,Rest0...>,Tensor<T,Rest2...>>::type;
        using resulting_index_1 =  typename get_resuling_index<Index<Idx0...>,Index<Idx2...>,
                                        Tensor<T,Rest0...>,Tensor<T,Rest2...>>::type;

        constexpr int flop_count_02_0 = pair_flop_cost<Index<Idx0...>,Index<Idx2...>,Tensor<T,Rest0...>,Tensor<T,Rest2...>,
                typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::value;

        constexpr int flop_count_02_1 = pair_flop_cost<resulting_index_1,Index<Idx1...>,resulting_tensor_1,Tensor<T,Rest1...>,
                typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::value;

        constexpr int flop_count_02 = flop_count_02_0 + flop_count_02_1;


        // second and last tensors contracted first
        using resulting_tensor_2 =  typename get_resuling_tensor<Index<Idx1...>,Index<Idx2...>,
                                        Tensor<T,Rest1...>,Tensor<T,Rest2...>>::type;
        using resulting_index_2 =  typename get_resuling_index<Index<Idx1...>,Index<Idx2...>,
                                        Tensor<T,Rest1...>,Tensor<T,Rest2...>>::type;

        constexpr int flop_count_12_0 = pair_flop_cost<Index<Idx1...>,Index<Idx2...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,
                typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::value;

        constexpr int flop_count_12_1 = pair_flop_cost<resulting_index_2,Index<Idx0...>,resulting_tensor_2,Tensor<T,Rest0...>,
                typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::value;

        constexpr int flop_count_12 = flop_count_12_0 + flop_count_12_1;

        constexpr int flop_count_012 = triplet_flop_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,
                Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>>::value;

        constexpr int which_variant = meta_argmin<flop_count_01,flop_count_02,flop_count_12,flop_count_012>::value;

        if (which_variant == 0) {
            auto tmp = contraction<Index<Idx0...>,Index<Idx1...>>(a,b);
            return contraction<resulting_index_0,Index<Idx2...>>(tmp,c);
        }
        else if (which_variant == 1) {
            auto tmp = contraction<Index<Idx0...>,Index<Idx2...>>(a,c);
            return contraction<resulting_index_1,Index<Idx1...>>(tmp,b);
        }
        else if (which_variant == 2) {
            auto tmp = contraction<Index<Idx1...>,Index<Idx2...>>(b,c);
            return contraction<Index<Idx0...>,resulting_index_2>(a,tmp);
        }
        else {
            // actual implementation goes here
            auto tmp = contraction<Index<Idx0...>,Index<Idx1...>>(a,b);
            return contraction<resulting_index_0,Index<Idx2...>>(tmp,c);
        }

//        // for benchmarks
//        unused(which_variant);
//        auto tmp = contraction<Index<Idx1...>,Index<Idx2...>>(b,c);
//        return contraction<Index<Idx0...>,resulting_index_2>(a,tmp);

    }

};




template<class Index_I, class Index_J, class Index_K,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2>
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c)
-> decltype(extractor_contract<Index_I,Index_J,Index_K>::contract_impl(a,b,c)) {
    return extractor_contract<Index_I,Index_J,Index_K>::contract_impl(a,b,c);
}






// Four tensor network
//---------------------------------------------------------------------------------------------------------------------//
template<class T, class U, class V, class W>
struct extractor_contract_4 {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2, size_t ... Idx3>
struct extractor_contract_4<Index<Idx0...>, Index<Idx1...>, Index<Idx2...>, Index<Idx3...> > {
  template<typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3>
    static
    typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...>, Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...>,
                              typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+sizeof...(Rest2)+sizeof...(Rest3)>::type>::type
    contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d) {


        using resulting_tensor_0 =  typename get_resuling_tensor<Index<Idx0...>,Index<Idx1...>,
                                        Tensor<T,Rest0...>,Tensor<T,Rest1...>>::type;
        using resulting_index_0 =  typename get_resuling_index<Index<Idx0...>,Index<Idx1...>,
                                        Tensor<T,Rest0...>,Tensor<T,Rest1...>>::type;
        using resulting_tensor_1 =  typename get_resuling_tensor<resulting_index_0,Index<Idx2...>,
                                        resulting_tensor_0,Tensor<T,Rest2...>>::type;
        using resulting_index_1 =  typename get_resuling_index<resulting_index_0,Index<Idx2...>,
                                        resulting_tensor_0,Tensor<T,Rest2...>>::type;

        resulting_tensor_0 tmp0 = contraction<Index<Idx0...>,Index<Idx1...>>(a,b);
        resulting_tensor_1 tmp1 = contraction<resulting_index_0,Index<Idx2...>>(tmp0,c);
        auto res = contraction<resulting_index_1,Index<Idx3...>>(tmp1,d);

        return res;


//        // for benchmarks
//        using resulting_tensor_0 =  typename get_resuling_tensor<Index<Idx0...>,Index<Idx1...>,
//                                        Tensor<T,Rest0...>,Tensor<T,Rest1...>>::type;
//        using resulting_index_0 =  typename get_resuling_index<Index<Idx0...>,Index<Idx1...>,
//                                        Tensor<T,Rest0...>,Tensor<T,Rest1...>>::type;

//        using resulting_tensor_1 =  typename get_resuling_tensor<Index<Idx2...>,Index<Idx3...>,
//                                        Tensor<T,Rest2...>,Tensor<T,Rest3...>>::type;
//        using resulting_index_1 =  typename get_resuling_index<Index<Idx2...>,Index<Idx3...>,
//                                        Tensor<T,Rest2...>,Tensor<T,Rest3...>>::type;

//        resulting_tensor_0 tmp0 = contraction<Index<Idx0...>,Index<Idx1...>>(a,b);
//        resulting_tensor_1 tmp1 = contraction<Index<Idx2...>,Index<Idx3...>>(c,d);
//        auto res = contraction<resulting_index_0,resulting_index_1>(tmp0,tmp1);

        return res;

    }
};




template<class Index_I, class Index_J, class Index_K, class Index_L,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3>
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d)
-> decltype(extractor_contract_4<Index_I,Index_J,Index_K,Index_L>::contract_impl(a,b,c,d)) {
    return extractor_contract_4<Index_I,Index_J,Index_K,Index_L>::contract_impl(a,b,c,d);
}



// Five tensor network
//---------------------------------------------------------------------------------------------------------------------//
template<class T, class U, class V, class W, class X>
struct extractor_contract_5 {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2, size_t ... Idx3, size_t ... Idx4>
struct extractor_contract_5<Index<Idx0...>, Index<Idx1...>, Index<Idx2...>, Index<Idx3...>, Index<Idx4...> > {
  template<typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4>
    static
    typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...>, Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...>,
                              typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                                sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)>::type>::type
    contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                  const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                  const Tensor<T,Rest4...> &e) {


        using resulting_tensor_0 =  typename get_resuling_tensor<Index<Idx0...>,Index<Idx1...>,
                                        Tensor<T,Rest0...>,Tensor<T,Rest1...>>::type;
        using resulting_index_0 =  typename get_resuling_index<Index<Idx0...>,Index<Idx1...>,
                                        Tensor<T,Rest0...>,Tensor<T,Rest1...>>::type;
        using resulting_tensor_1 =  typename get_resuling_tensor<resulting_index_0,Index<Idx2...>,
                                        resulting_tensor_0,Tensor<T,Rest2...>>::type;
        using resulting_index_1 =  typename get_resuling_index<resulting_index_0,Index<Idx2...>,
                                        resulting_tensor_0,Tensor<T,Rest2...>>::type;
        using resulting_tensor_2 =  typename get_resuling_tensor<resulting_index_1,Index<Idx3...>,
                                        resulting_tensor_1,Tensor<T,Rest3...>>::type;
        using resulting_index_2 =  typename get_resuling_index<resulting_index_1,Index<Idx3...>,
                                        resulting_tensor_1,Tensor<T,Rest3...>>::type;


        resulting_tensor_0 tmp0 = contraction<Index<Idx0...>,Index<Idx1...>>(a,b);
        resulting_tensor_1 tmp1 = contraction<resulting_index_0,Index<Idx2...>>(tmp0,c);
        resulting_tensor_2 tmp2 = contraction<resulting_index_1,Index<Idx3...>>(tmp1,d);
        auto tmp3 = contraction<resulting_index_2,Index<Idx4...>>(tmp2,e);

        return tmp3;

    }
};




template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4>
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                 const Tensor<T,Rest4...> &e)
-> decltype(extractor_contract_5<Index_I,Index_J,Index_K,Index_L,Index_M>::contract_impl(a,b,c,d,e)) {
    return extractor_contract_5<Index_I,Index_J,Index_K,Index_L,Index_M>::contract_impl(a,b,c,d,e);
}



// Six tensor network
//---------------------------------------------------------------------------------------------------------------------//
template<class T, class U, class V, class W, class X, class Y>
struct extractor_contract_6 {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2, size_t ... Idx3, size_t ... Idx4, size_t ... Idx5>
struct extractor_contract_6<Index<Idx0...>, Index<Idx1...>, Index<Idx2...>, Index<Idx3...>, Index<Idx4...>, Index<Idx5...> > {
  template<typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4, size_t ... Rest5>
    static
    typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...>, Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...>,
                              typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                                sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)>::type>::type
    contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                  const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                  const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f) {


        using resulting_tensor_0 =  typename get_resuling_tensor<Index<Idx0...>,Index<Idx1...>,
                                        Tensor<T,Rest0...>,Tensor<T,Rest1...>>::type;
        using resulting_index_0 =  typename get_resuling_index<Index<Idx0...>,Index<Idx1...>,
                                        Tensor<T,Rest0...>,Tensor<T,Rest1...>>::type;
        using resulting_tensor_1 =  typename get_resuling_tensor<resulting_index_0,Index<Idx2...>,
                                        resulting_tensor_0,Tensor<T,Rest2...>>::type;
        using resulting_index_1 =  typename get_resuling_index<resulting_index_0,Index<Idx2...>,
                                        resulting_tensor_0,Tensor<T,Rest2...>>::type;
        using resulting_tensor_2 =  typename get_resuling_tensor<resulting_index_1,Index<Idx3...>,
                                        resulting_tensor_1,Tensor<T,Rest3...>>::type;
        using resulting_index_2 =  typename get_resuling_index<resulting_index_1,Index<Idx3...>,
                                        resulting_tensor_1,Tensor<T,Rest3...>>::type;
        using resulting_tensor_3 =  typename get_resuling_tensor<resulting_index_2,Index<Idx4...>,
                                        resulting_tensor_2,Tensor<T,Rest4...>>::type;
        using resulting_index_3 =  typename get_resuling_index<resulting_index_2,Index<Idx4...>,
                                        resulting_tensor_2,Tensor<T,Rest4...>>::type;


        resulting_tensor_0 tmp0 = contraction<Index<Idx0...>,Index<Idx1...>>(a,b);
        resulting_tensor_1 tmp1 = contraction<resulting_index_0,Index<Idx2...>>(tmp0,c);
        resulting_tensor_2 tmp2 = contraction<resulting_index_1,Index<Idx3...>>(tmp1,d);
        resulting_tensor_3 tmp3 = contraction<resulting_index_2,Index<Idx4...>>(tmp2,e);
        auto tmp4 = contraction<resulting_index_3,Index<Idx5...>>(tmp3,f);

        return tmp4;

    }
};




template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, class Index_N,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4, size_t ... Rest5>
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                 const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f)
-> decltype(extractor_contract_6<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>::contract_impl(a,b,c,d,e,f)) {
    return extractor_contract_6<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>::contract_impl(a,b,c,d,e,f);
}

}

#endif // NETWORK_CONTRACTION_H
