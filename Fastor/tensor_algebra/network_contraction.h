#ifndef NETWORK_CONTRACTION_H
#define NETWORK_CONTRACTION_H

#ifndef FASTOR_DONT_PERFORM_OP_MIN

#include "Fastor/tensor_algebra/einsum.h"
#include "Fastor/meta/opmin_meta.h"

namespace Fastor {


// Three tensor network
//---------------------------------------------------------------------------------------------------------------------//
template<class T, class U, class V>
struct extractor_contract_3 {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2>
struct extractor_contract_3<Index<Idx0...>, Index<Idx1...>, Index<Idx2...> > {
  template<typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2>
    static
    typename contraction_impl<
        Index<Idx0...,Idx1...,Idx2...>, Tensor<T,Rest0...,Rest1...,Rest2...>,
        typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+sizeof...(Rest2)>::type>::type
    FASTOR_INLINE
    contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c) {

        using cost_model = triplet_flop_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,
            Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>>;

        using resulting_index_0 = typename cost_model::resulting_index_0;
        using resulting_index_1 = typename cost_model::resulting_index_1;
        using resulting_index_2 = typename cost_model::resulting_index_2;

#ifndef FASTOR_KEEP_DP_FIXED

        constexpr int which_variant = cost_model::which_variant;

        FASTOR_IF_CONSTEXPR (which_variant == 0) {
            auto tmp = einsum<Index<Idx0...>,Index<Idx1...>>(a,b);
            return einsum<resulting_index_0,Index<Idx2...>>(tmp,c);
        }
        else FASTOR_IF_CONSTEXPR (which_variant == 1) {
            auto tmp = einsum<Index<Idx0...>,Index<Idx2...>>(a,c);
            return einsum<Index<Idx1...>,resulting_index_1>(b,tmp);
        }
        else {
            auto tmp = einsum<Index<Idx1...>,Index<Idx2...>>(b,c);
            return einsum<Index<Idx0...>,resulting_index_2>(a,tmp);
        }

#else
        // for benchmarks
        auto tmp = einsum<Index<Idx1...>,Index<Idx2...>>(b,c);
        return einsum<Index<Idx0...>,resulting_index_2>(a,tmp);
#endif

    }

};


template<class Index_I, class Index_J, class Index_K,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2>
FASTOR_INLINE
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c)
-> decltype(extractor_contract_3<Index_I,Index_J,Index_K>::contract_impl(a,b,c)) {
    return extractor_contract_3<Index_I,Index_J,Index_K>::contract_impl(a,b,c);
}
//---------------------------------------------------------------------------------------------------------------------//





// Four tensor network
//---------------------------------------------------------------------------------------------------------------------//
template<class T, class U, class V, class W>
struct extractor_contract_4 {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2, size_t ... Idx3>
struct extractor_contract_4<Index<Idx0...>, Index<Idx1...>, Index<Idx2...>, Index<Idx3...> > {
  template<typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3>
    static
    typename contraction_impl<
        Index<Idx0...,Idx1...,Idx2...,Idx3...>, Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...>,
        typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+sizeof...(Rest2)+sizeof...(Rest3)>::type>::type
    FASTOR_INLINE
    contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d) {

#ifndef FASTOR_KEEP_DP_FIXED

        using cost_model = quartet_flop_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,Index<Idx3...>,
            Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>>;

        using resulting_index_0 = typename cost_model::resulting_index_0;
        using resulting_index_1 = typename cost_model::resulting_index_1;
        using resulting_index_2 = typename cost_model::resulting_index_2;
        using resulting_index_3 = typename cost_model::resulting_index_3;

        constexpr int which_variant = cost_model::which_variant;

        FASTOR_IF_CONSTEXPR (which_variant==0) {
            auto tmp = einsum<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>>(a,b,c);
            return einsum<resulting_index_0,Index<Idx3...>>(tmp,d);
        }
        else FASTOR_IF_CONSTEXPR (which_variant==1) {
            auto tmp = einsum<Index<Idx0...>,Index<Idx1...>,Index<Idx3...>>(a,b,d);
            return einsum<Index<Idx2...>,resulting_index_1>(c,tmp);
        }
        else FASTOR_IF_CONSTEXPR (which_variant==2) {
            auto tmp = einsum<Index<Idx0...>,Index<Idx2...>,Index<Idx3...>>(a,c,d);
            return einsum<Index<Idx1...>,resulting_index_2>(b,tmp);
        }
        else {
            auto tmp = einsum<Index<Idx1...>,Index<Idx2...>,Index<Idx3...>>(b,c,d);
            return einsum<Index<Idx0...>,resulting_index_3>(a,tmp);
        }

#else
        // for benchmarks
        using resulting_tensor_0 =  typename get_resuling_tensor<Index<Idx0...>,Index<Idx1...>,
                                        Tensor<T,Rest0...>,Tensor<T,Rest1...>>::type;
        using resulting_index_0 =  typename get_resuling_index<Index<Idx0...>,Index<Idx1...>,
                                        Tensor<T,Rest0...>,Tensor<T,Rest1...>>::type;

        using resulting_tensor_1 =  typename get_resuling_tensor<Index<Idx2...>,Index<Idx3...>,
                                        Tensor<T,Rest2...>,Tensor<T,Rest3...>>::type;
        using resulting_index_1 =  typename get_resuling_index<Index<Idx2...>,Index<Idx3...>,
                                        Tensor<T,Rest2...>,Tensor<T,Rest3...>>::type;

        resulting_tensor_0 tmp0 = einsum<Index<Idx0...>,Index<Idx1...>>(a,b);
        resulting_tensor_1 tmp1 = einsum<Index<Idx2...>,Index<Idx3...>>(c,d);
        auto res = einsum<resulting_index_0,resulting_index_1>(tmp0,tmp1);

        return res;
#endif

    }
};


template<class Index_I, class Index_J, class Index_K, class Index_L,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3>
FASTOR_INLINE
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d)
-> decltype(extractor_contract_4<Index_I,Index_J,Index_K,Index_L>::contract_impl(a,b,c,d)) {
    return extractor_contract_4<Index_I,Index_J,Index_K,Index_L>::contract_impl(a,b,c,d);
}
//---------------------------------------------------------------------------------------------------------------------//



// Five tensor network
//---------------------------------------------------------------------------------------------------------------------//
template<class T, class U, class V, class W, class X>
struct extractor_contract_5 {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2, size_t ... Idx3, size_t ... Idx4>
struct extractor_contract_5<Index<Idx0...>, Index<Idx1...>, Index<Idx2...>, Index<Idx3...>, Index<Idx4...> > {
  template<typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4>
    static
    typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...>, Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...>,
                              typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+
                                sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)>::type>::type
    FASTOR_INLINE
    contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                  const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                  const Tensor<T,Rest4...> &e) {

        using cost_model = quintet_flop_cost<
            Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,Index<Idx3...>,Index<Idx4...>,
            Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>,Tensor<T,Rest4...>>;

        using resulting_index_0 = typename cost_model::resulting_index_0;
        using resulting_index_1 = typename cost_model::resulting_index_1;
        using resulting_index_2 = typename cost_model::resulting_index_2;
        using resulting_index_3 = typename cost_model::resulting_index_3;
        using resulting_index_4 = typename cost_model::resulting_index_4;

        constexpr int which_variant = cost_model::which_variant;

        FASTOR_IF_CONSTEXPR (which_variant==0) {
            auto tmp = einsum<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,Index<Idx3...>>(a,b,c,d);
            return einsum<resulting_index_0,Index<Idx4...>>(tmp,e);
        }
        else FASTOR_IF_CONSTEXPR (which_variant==1) {
            auto tmp = einsum<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,Index<Idx4...>>(a,b,c,e);
            return einsum<Index<Idx3...>,resulting_index_1>(d,tmp);
        }
        else FASTOR_IF_CONSTEXPR (which_variant==2) {
            auto tmp = einsum<Index<Idx0...>,Index<Idx1...>,Index<Idx3...>,Index<Idx4...>>(a,b,d,e);
            return einsum<Index<Idx2...>,resulting_index_2>(c,tmp);
        }
        else FASTOR_IF_CONSTEXPR (which_variant==3) {
            auto tmp = einsum<Index<Idx0...>,Index<Idx2...>,Index<Idx3...>,Index<Idx4...>>(a,c,d,e);
            return einsum<Index<Idx1...>,resulting_index_3>(b,tmp);
        }
        else {
            auto tmp = einsum<Index<Idx1...>,Index<Idx2...>,Index<Idx3...>,Index<Idx4...>>(b,c,d,e);
            return einsum<Index<Idx0...>,resulting_index_4>(a,tmp);
        }

    }
};


template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4>
FASTOR_INLINE
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                 const Tensor<T,Rest4...> &e)
-> decltype(extractor_contract_5<Index_I,Index_J,Index_K,Index_L,Index_M>::contract_impl(a,b,c,d,e)) {
    return extractor_contract_5<Index_I,Index_J,Index_K,Index_L,Index_M>::contract_impl(a,b,c,d,e);
}
//---------------------------------------------------------------------------------------------------------------------//



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
    FASTOR_INLINE
    contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                  const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                  const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f) {


        using cost_model = sixtet_flop_cost<Index<Idx0...>,
            Index<Idx1...>,Index<Idx2...>,Index<Idx3...>,Index<Idx4...>,Index<Idx5...>,
            Tensor<T,Rest0...>,Tensor<T,Rest1...>,Tensor<T,Rest2...>,Tensor<T,Rest3...>,Tensor<T,Rest4...>,Tensor<T,Rest5...>>;

        using resulting_index_0 = typename cost_model::resulting_index_0;
        using resulting_index_1 = typename cost_model::resulting_index_1;
        using resulting_index_2 = typename cost_model::resulting_index_2;
        using resulting_index_3 = typename cost_model::resulting_index_3;
        using resulting_index_4 = typename cost_model::resulting_index_4;
        using resulting_index_5 = typename cost_model::resulting_index_5;

        constexpr int which_variant = cost_model::which_variant;

        FASTOR_IF_CONSTEXPR (which_variant==0) {
            auto tmp = einsum<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,Index<Idx3...>,Index<Idx4...>>(a,b,c,d,e);
            return einsum<resulting_index_0,Index<Idx5...>>(tmp,f);
        }
        else FASTOR_IF_CONSTEXPR (which_variant==1) {
            auto tmp = einsum<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,Index<Idx3...>,Index<Idx5...>>(a,b,c,d,f);
            return einsum<Index<Idx4...>,resulting_index_1>(e,tmp);
        }
        else FASTOR_IF_CONSTEXPR (which_variant==2) {
            auto tmp = einsum<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,Index<Idx4...>,Index<Idx5...>>(a,b,c,e,f);
            return einsum<Index<Idx3...>,resulting_index_2>(d,tmp);
        }
        else FASTOR_IF_CONSTEXPR (which_variant==3) {
            auto tmp = einsum<Index<Idx0...>,Index<Idx1...>,Index<Idx3...>,Index<Idx4...>,Index<Idx5...>>(a,b,d,e,f);
            return einsum<Index<Idx2...>,resulting_index_3>(c,tmp);
        }
        else FASTOR_IF_CONSTEXPR (which_variant==4) {
            auto tmp = einsum<Index<Idx0...>,Index<Idx2...>,Index<Idx3...>,Index<Idx4...>,Index<Idx5...>>(a,c,d,e,f);
            return einsum<Index<Idx1...>,resulting_index_4>(b,tmp);
        }
        else {
            auto tmp = einsum<Index<Idx1...>,Index<Idx2...>,Index<Idx3...>,Index<Idx4...>,Index<Idx5...>>(b,c,d,e,f);
            return einsum<Index<Idx0...>,resulting_index_5>(a,tmp);
        }

    }

};


template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, class Index_N,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4, size_t ... Rest5>
FASTOR_INLINE
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                 const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f)
-> decltype(extractor_contract_6<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>::contract_impl(a,b,c,d,e,f)) {
    return extractor_contract_6<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>::contract_impl(a,b,c,d,e,f);
}
//---------------------------------------------------------------------------------------------------------------------//



// Seven tensor network
//---------------------------------------------------------------------------------------------------------------------//
template<class T, class U, class V, class W, class X, class Y, class Z>
struct extractor_contract_7 {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2, size_t ... Idx3,
         size_t ... Idx4, size_t ... Idx5, size_t ... Idx6>
struct extractor_contract_7<Index<Idx0...>, Index<Idx1...>, Index<Idx2...>,
        Index<Idx3...>, Index<Idx4...>, Index<Idx5...>, Index<Idx6...> > {
  template<typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2,
           size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6>
    static
    typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...>,
        Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...>,
                              typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                                sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)+sizeof...(Rest6)>::type>::type
    FASTOR_INLINE
    contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                  const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                  const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
                  const Tensor<T,Rest6...> &g) {

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
        using resulting_tensor_4 =  typename get_resuling_tensor<resulting_index_3,Index<Idx5...>,
                                        resulting_tensor_3,Tensor<T,Rest5...>>::type;
        using resulting_index_4 =  typename get_resuling_index<resulting_index_3,Index<Idx5...>,
                                        resulting_tensor_3,Tensor<T,Rest5...>>::type;


        resulting_tensor_0 tmp0 = einsum<Index<Idx0...>,Index<Idx1...>>(a,b);
        resulting_tensor_1 tmp1 = einsum<resulting_index_0,Index<Idx2...>>(tmp0,c);
        resulting_tensor_2 tmp2 = einsum<resulting_index_1,Index<Idx3...>>(tmp1,d);
        resulting_tensor_3 tmp3 = einsum<resulting_index_2,Index<Idx4...>>(tmp2,e);
        resulting_tensor_4 tmp4 = einsum<resulting_index_3,Index<Idx5...>>(tmp3,f);
        auto tmp5 = einsum<resulting_index_4,Index<Idx6...>>(tmp4,g);

        return tmp5;

    }
};


template<class Index_I, class Index_J, class Index_K, class Index_L,
         class Index_M, class Index_N, class Index_O,
         typename T, size_t ... Rest0, size_t ... Rest1,
         size_t ... Rest2, size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6>
FASTOR_INLINE
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                 const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
                 const Tensor<T,Rest6...> &g)
-> decltype(extractor_contract_7<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O>::contract_impl(a,b,c,d,e,f,g)) {
    return extractor_contract_7<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O>::contract_impl(a,b,c,d,e,f,g);
}
//---------------------------------------------------------------------------------------------------------------------//



// Eight tensor network
//---------------------------------------------------------------------------------------------------------------------//
template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
struct extractor_contract_8 {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2, size_t ... Idx3,
         size_t ... Idx4, size_t ... Idx5, size_t ... Idx6, size_t ... Idx7>
struct extractor_contract_8<Index<Idx0...>, Index<Idx1...>, Index<Idx2...>,
        Index<Idx3...>, Index<Idx4...>, Index<Idx5...>, Index<Idx6...>, Index<Idx7...> > {
  template<typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2,
           size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6, size_t ... Rest7>
    static
    typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...>,
        Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...>,
                              typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                                sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)+\
                                sizeof...(Rest6)+sizeof...(Rest7)>::type>::type
    contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                  const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                  const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
                  const Tensor<T,Rest6...> &g, const Tensor<T,Rest7...> &h) {


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
        using resulting_tensor_4 =  typename get_resuling_tensor<resulting_index_3,Index<Idx5...>,
                                        resulting_tensor_3,Tensor<T,Rest5...>>::type;
        using resulting_index_4 =  typename get_resuling_index<resulting_index_3,Index<Idx5...>,
                                        resulting_tensor_3,Tensor<T,Rest5...>>::type;
        using resulting_tensor_5 =  typename get_resuling_tensor<resulting_index_4,Index<Idx6...>,
                                        resulting_tensor_4,Tensor<T,Rest6...>>::type;
        using resulting_index_5 =  typename get_resuling_index<resulting_index_4,Index<Idx6...>,
                                        resulting_tensor_4,Tensor<T,Rest6...>>::type;


        resulting_tensor_0 tmp0 = einsum<Index<Idx0...>,Index<Idx1...>>(a,b);
        resulting_tensor_1 tmp1 = einsum<resulting_index_0,Index<Idx2...>>(tmp0,c);
        resulting_tensor_2 tmp2 = einsum<resulting_index_1,Index<Idx3...>>(tmp1,d);
        resulting_tensor_3 tmp3 = einsum<resulting_index_2,Index<Idx4...>>(tmp2,e);
        resulting_tensor_4 tmp4 = einsum<resulting_index_3,Index<Idx5...>>(tmp3,f);
        resulting_tensor_5 tmp5 = einsum<resulting_index_4,Index<Idx6...>>(tmp4,g);
        auto tmp6 = einsum<resulting_index_5,Index<Idx7...>>(tmp5,h);

        return tmp6;

    }
};


template<class Index_I, class Index_J, class Index_K, class Index_L,
         class Index_M, class Index_N, class Index_O, class Index_P,
         typename T, size_t ... Rest0, size_t ... Rest1,
         size_t ... Rest2, size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6, size_t ... Rest7>
FASTOR_INLINE
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                 const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
                 const Tensor<T,Rest6...> &g, const Tensor<T,Rest7...> &h)
-> decltype(extractor_contract_8<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O,Index_P>::contract_impl(a,b,c,d,e,f,g,h)) {
    return extractor_contract_8<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O,Index_P>::contract_impl(a,b,c,d,e,f,g,h);
}
//---------------------------------------------------------------------------------------------------------------------//

} // end of namespace Fastor

#endif // FASTOR_DONT_PERFORM_OP_MIN

#endif // NETWORK_CONTRACTION_H
