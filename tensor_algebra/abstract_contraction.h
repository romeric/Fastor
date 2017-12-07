#ifndef ABSTRACT_CONTRACTION_H
#define ABSTRACT_CONTRACTION_H

#include "tensor/Tensor.h"
#include "indicial.h"
#include "meta/tensor_post_meta.h"


namespace Fastor {


template<class T, class U>
struct extractor_abstract_contract {};

template<size_t ... Idx0, size_t ... Idx1>
struct extractor_abstract_contract<Index<Idx0...>, Index<Idx1...>> {


    template<typename Derived0, typename Derived1, size_t DIM0, size_t DIM1>
      static
      typename contraction_impl<Index<Idx0...,Idx1...>,
                typename concat_tensor<typename tensor_type_finder<Derived0>::type, typename tensor_type_finder<Derived1>::type>::type,
                typename std_ext::make_index_sequence<DIM0+DIM1>::type>::type
      contract_impl(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b) {

        static_assert(!is_reduction<Index<Idx0...>,Index<Idx1...>>::value,"REDUCTION TO SCALAR REQUESTED. USE REDUCTION FUNCTION INSTEAD");

        using tensor_type_0 = typename tensor_type_finder<Derived0>::type;
        using tensor_type_1 = typename tensor_type_finder<Derived1>::type;
        using concatenated_tensor = typename concat_tensor<tensor_type_0, tensor_type_1>::type;

        using ContractionImpl = contraction_impl<Index<Idx0...,Idx1...>, concatenated_tensor,
          typename std_ext::make_index_sequence<DIM0+DIM1>::type>;

        using OutTensor = typename ContractionImpl::type;
        using OutIndice = typename ContractionImpl::indices;

        using T = typename scalar_type_finder<Derived0>::type;

        OutTensor out;
        out.zeros();
        T *out_data = out.data();
        const Derived0 &a_src = a.self();
        const Derived1 &b_src = b.self();


        constexpr int a_dim = DIM0;
        constexpr int b_dim = DIM1;
        constexpr int out_dim =  no_of_unique<Idx0...,Idx1...>::value;

        constexpr auto& idx_a = IndexTensors<
                Index<Idx0..., Idx1...>,
                concatenated_tensor,
                Index<Idx0...>,tensor_type_0,
                typename std_ext::make_index_sequence<DIM0>::type>::indices;

        constexpr auto& idx_b = IndexTensors<
                Index<Idx0..., Idx1...>,
                concatenated_tensor,
                Index<Idx1...>,tensor_type_1,
                typename std_ext::make_index_sequence<DIM1>::type>::indices;

        constexpr auto& idx_out = IndexTensors<
                Index<Idx0..., Idx1...>,
                concatenated_tensor,
                OutIndice,OutTensor,
                typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::indices;

        using nloops = loop_setter<
                  Index<Idx0...,Idx1...>,
                  concatenated_tensor,
                  typename std_ext::make_index_sequence<out_dim>::type>;
        constexpr auto& maxes_out = nloops::dims;
        constexpr int total = nloops::value;

        constexpr std::array<size_t,a_dim> products_a = nprods<typename put_dims_in_Index<tensor_type_0>::type,
            typename std_ext::make_index_sequence<a_dim>::type>::values;
        constexpr std::array<size_t,b_dim> products_b = nprods<typename put_dims_in_Index<tensor_type_1>::type,
            typename std_ext::make_index_sequence<b_dim>::type>::values;
        constexpr std::array<size_t,OutTensor::Dimension> products_out = nprods<typename put_dims_in_Index<OutTensor>::type,
            typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::values;

        // // Check for reducible vectorisability
        // constexpr int general_stride = general_stride_finder<Index<Idx0...>,Index<Idx1...>,
        //         tensor_type_0,tensor_type_1, typename std_ext::make_index_sequence<b_dim>::type>::value;

// #ifndef FASTOR_DONT_VECTORISE
//         using vectorisability = is_reducibly_vectorisable<OutIndice,OutTensor>;
//         constexpr int stride = vectorisability::stride;
//         using V = typename vectorisability::type;
// #else
//         constexpr int stride = 1;
//         using V = SIMDVector<T,sizeof(T)*8>;
// #endif

// #ifndef FASTOR_DONT_VECTORISE
//               using vectorisability = is_vectorisable<Index<Idx0...>,Index<Idx1...>,tensor_type_1>;
//               constexpr int stride = vectorisability::stride;
//               using V = typename vectorisability::type;
// #else
//               constexpr int stride = 1;
//               using V = SIMDVector<T,sizeof(T)*8>;
//               // using V = SIMDVector<T,DEFAULT_ABI>;
// #endif

        int as[out_dim];
        std::fill(as,as+out_dim,0);
        int it;

        int i = 0;
        for (; i < total; ++i) {
            int remaining = total;
            for (int n = 0; n < out_dim; ++n) {
                remaining /= maxes_out[n];
                as[n] = ( i / remaining ) % maxes_out[n];
            }

            int index_a = as[idx_a[a_dim-1]];
            for(it = 0; it< a_dim; it++) {
                index_a += products_a[it]*as[idx_a[it]];
            }
            int index_b = as[idx_b[b_dim-1]];
            for(it = 0; it< b_dim; it++) {
                index_b += products_b[it]*as[idx_b[it]];
            }
            int index_out = as[idx_out[OutTensor::Dimension-1]];
            for(it = 0; it< static_cast<int>(OutTensor::Dimension); it++) {
                index_out += products_out[it]*as[idx_out[it]];
            }

            // V _vec_out = a_src.template eval<T>((FASTOR_INDEX)index_a)*b_src.template eval<T>((FASTOR_INDEX)index_b) +  V(out_data+index_out);
            // _vec_out.store(out_data+index_out);

            // Note that vectorisation of this would require V and eval to be the same
            // which cannot the same under all compilation flags
            out_data[index_out] += a_src.template eval_s<T>((FASTOR_INDEX)index_a)*b_src.template eval_s<T>((FASTOR_INDEX)index_b);
        }

        // int i = 0;
        // for (; i < ROUND_DOWN(total,stride); i+=stride) {
        //     int remaining = total;
        //     for (int n = 0; n < out_dim; ++n) {
        //         remaining /= maxes_out[n];
        //         as[n] = ( i / remaining ) % maxes_out[n];
        //     }

        //     int index_a = as[idx_a[a_dim-1]];
        //     for(it = 0; it< a_dim; it++) {
        //         index_a += products_a[it]*as[idx_a[it]];
        //     }
        //     int index_b = as[idx_b[b_dim-1]];
        //     for(it = 0; it< b_dim; it++) {
        //         index_b += products_b[it]*as[idx_b[it]];
        //     }
        //     int index_out = as[idx_out[OutTensor::Dimension-1]];
        //     for(it = 0; it< static_cast<int>(OutTensor::Dimension); it++) {
        //         index_out += products_out[it]*as[idx_out[it]];
        //     }

        //     SIMDVector<T,DEFAULT_ABI> _vec_out = a_src.template eval<T>((FASTOR_INDEX)index_a)*b_src.template eval<T>((FASTOR_INDEX)index_b) +  SIMDVector<T,DEFAULT_ABI>(out_data+index_out);
        //     _vec_out.store(out_data+index_out);
        //     print(type_name<SIMDVector<T,DEFAULT_ABI>>());
        // }


        // for (; i < total; ++i) {
        //     int remaining = total;
        //     for (int n = 0; n < out_dim; ++n) {
        //         remaining /= maxes_out[n];
        //         as[n] = ( i / remaining ) % maxes_out[n];
        //     }

        //     int index_a = as[idx_a[a_dim-1]];
        //     for(it = 0; it< a_dim; it++) {
        //         index_a += products_a[it]*as[idx_a[it]];
        //     }
        //     int index_b = as[idx_b[b_dim-1]];
        //     for(it = 0; it< b_dim; it++) {
        //         index_b += products_b[it]*as[idx_b[it]];
        //     }
        //     int index_out = as[idx_out[OutTensor::Dimension-1]];
        //     for(it = 0; it< static_cast<int>(OutTensor::Dimension); it++) {
        //         index_out += products_out[it]*as[idx_out[it]];
        //     }

        //     out_data[index_out] += a_src.template eval_s<T>((FASTOR_INDEX)index_a)*b_src.template eval_s<T>((FASTOR_INDEX)index_b);
        // }

        return out;

      }
};


template<class Index_I, class Index_J,
        typename Derived0, typename Derived1, size_t DIM0, size_t DIM1>
auto einsum(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b) //{
-> decltype(extractor_abstract_contract<Index_I,Index_J>::contract_impl(a,b)) {
    return extractor_abstract_contract<Index_I,Index_J>::contract_impl(a,b);
}


#ifndef FASTOR_DONT_PERFORM_OP_MIN

// Network
template<class T, class U, class V>
struct extractor_abstract_contract_3 {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2>
struct extractor_abstract_contract_3<Index<Idx0...>, Index<Idx1...>, Index<Idx2...> > {
  template<typename Derived0, typename Derived1, typename Derived2, size_t DIM0, size_t DIM1, size_t DIM2>
    static
    typename contraction_impl<Index<Idx0...,Idx1...,Idx2...>,
            typename concat_tensor<
                typename tensor_type_finder<Derived0>::type,
                typename tensor_type_finder<Derived1>::type,
                typename tensor_type_finder<Derived2>::type>::type,
                              typename std_ext::make_index_sequence<DIM0+DIM1+DIM2>::type>::type
    contract_impl(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b, const AbstractTensor<Derived2,DIM2> &c) {

        using tensor_type_0 = typename tensor_type_finder<Derived0>::type;
        using tensor_type_1 = typename tensor_type_finder<Derived1>::type;
        using tensor_type_2 = typename tensor_type_finder<Derived2>::type;

        using cost_model = triplet_flop_cost<Index<Idx0...>,Index<Idx1...>,Index<Idx2...>,
            tensor_type_0,tensor_type_1,tensor_type_2>;

        using resulting_index_0 = typename cost_model::resulting_index_0;
        // using resulting_index_1 = typename cost_model::resulting_index_1;
        using resulting_index_2 = typename cost_model::resulting_index_2;

        constexpr int which_variant = cost_model::which_variant;

#ifdef FASTOR_PRINT_COST
        constexpr int flop_cost = cost_model::min_cost;
        print(flop_cost);
#endif

        if (which_variant == 0) {
            auto tmp = einsum<Index<Idx0...>,Index<Idx1...>>(a,b);
            return einsum<resulting_index_0,Index<Idx2...>>(tmp,c);
        }
        // leads to incorrect results
        // else if (which_variant == 1) {
        //     auto tmp = einsum<Index<Idx0...>,Index<Idx2...>>(a,c);
        //     return einsum<Index<Idx1...>,resulting_index_1>(b,tmp);
        // }
        else if (which_variant == 2) {
            auto tmp = einsum<Index<Idx1...>,Index<Idx2...>>(b,c);
            return einsum<Index<Idx0...>,resulting_index_2>(a,tmp);
        }
        else {
            // actual implementation goes here
            auto tmp = einsum<Index<Idx0...>,Index<Idx1...>>(a,b);
            return einsum<resulting_index_0,Index<Idx2...>>(tmp,c);
        }
    }

};




template<class Index_I, class Index_J, class Index_K,
         typename Derived0, typename Derived1, typename Derived2,
         size_t DIM0, size_t DIM1, size_t DIM2>
auto einsum(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b, const AbstractTensor<Derived2,DIM2> &c)
-> decltype(extractor_abstract_contract_3<Index_I,Index_J,Index_K>::contract_impl(a,b,c)) {
    return extractor_abstract_contract_3<Index_I,Index_J,Index_K>::contract_impl(a,b,c);
}

#endif

}


#endif