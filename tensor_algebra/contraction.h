#ifndef CONTRACTION_H
#define CONTRACTION_H

#include "tensor/Tensor.h"
#include "indicial.h"


namespace Fastor {

//using namespace details;
//namespace details {


// Define this if faster (complete meta-engine based) tensor contraction is required.
// Note that it blows up memory consumption and compilation time

//#define Opt

// A complete tensor contraction meta-engine
//--------------------------------------------------------------------------------------------------------------//
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
    static constexpr std::array<int,sizeof...(ss)> values = {cartesian_product_single(vals,prod<Rest...>::value,I,ss)...};
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
template<class Idx0, class Idx1, class Tens0, class Tens1, class Seq>
struct contract_meta_engine;

template<size_t ... Idx0, size_t ... Idx1, size_t ... Rest0, size_t ... Rest1, size_t ... ss, typename T>
struct contract_meta_engine<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>> {

    using OutTensor = typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
                              typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type;

    static constexpr int a_dim = sizeof...(Rest0);
    static constexpr int b_dim = sizeof...(Rest1);
    static constexpr int out_dim = OutTensor::Dimension;
    static constexpr int total = sizeof...(ss);

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
    static constexpr std::array<size_t,Index_with_dims::NoIndices> products_out = nprods<Index_with_dims,
            typename std_ext::make_index_sequence<Index_with_dims::NoIndices>::type>::values;

    // Generate the cartesian product
    static constexpr auto& as_all = cartesian_product<maxes_out_type,typename std_ext::make_index_sequence<total>::type>::values;
    // Alternatively you can pass the ss... directly into cartesian_product but that does not change anything in terms of
    // memory usage or compilation time
    //using maxes_out_indices = typename no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
    //        uniques_type>::indices;
    //static constexpr std::array<std::array<int,maxes_out_indices::NoIndices>,total> as_all = {all_cartesian_product<ss,2,3,4,2>()...};

    static constexpr std::array<int,sizeof...(ss)> index_a = {get_indices(products_a,idx_a,as_all,ss,a_dim-1)...};
    static constexpr std::array<int,sizeof...(ss)> index_b = {get_indices(products_b,idx_b,as_all,ss,b_dim-1)...};
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




template<class T, class U>
struct extractor_contract_2 {};

template<size_t ... Idx0, size_t ... Idx1>
struct extractor_contract_2<Index<Idx0...>, Index<Idx1...>> {

#ifdef Opt

    template<typename T, size_t ... Rest0, size_t ... Rest1>
      static
      typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
               typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type
      contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) {

          static_assert(!is_reduction<Index<Idx0...>,Index<Idx1...>>::value,"REDUCTION TO SCALAR REQUESTED. USE REDUCTION FUNCTION INSTEAD");

          constexpr int total = no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
                  typename std_ext::make_index_sequence<no_of_unique<Idx0...,Idx1...>::value>::type>::value;

          using index_generator = contract_meta_engine<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
              typename std_ext::make_index_sequence<total>::type>;

          using OutTensor = typename index_generator::OutTensor;

          constexpr auto& index_a = index_generator::index_a;
          constexpr auto& index_b = index_generator::index_b;
          constexpr auto& index_out = index_generator::index_out;

          OutTensor out;
          out.zeros();
          const T *a_data = a.data();
          const T *b_data = b.data();
          T *out_data = out.data();

          using vectorisability = is_vectorisable<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest1...>>;
          constexpr int stride = vectorisability::stride;
          using V = typename vectorisability::type;

          V _vec_a;
          for (int i = 0; i < total; i+=stride) {
              // out_data[index_out[i]] += a_data[index_a[i]]*b_data[index_b[i]];
              _vec_a.set(*(a_data+index_a[i]));
              V _vec_out = _vec_a*V(b_data+index_b[i]) +  V(out_data+index_out[i]);
              _vec_out.store(out_data+index_out[i]);
          }

        return out;
    }
};

#else

      template<typename T, size_t ... Rest0, size_t ... Rest1>
        static
        typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
                 typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type
        contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) {

          static_assert(!is_reduction<Index<Idx0...>,Index<Idx1...>>::value,"REDUCTION TO SCALAR REQUESTED. USE REDUCTION FUNCTION INSTEAD");

          using OutTensor = typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type;

          OutTensor out;
          out.zeros();
          const T *a_data = a.data();
          const T *b_data = b.data();
          T *out_data = out.data();

          constexpr int a_dim = sizeof...(Rest0);
          constexpr int b_dim = sizeof...(Rest1);

          constexpr auto& idx_a = IndexFirstTensor<Index<Idx0...>,Index<Idx1...>, Tensor<T,Rest0...>,Tensor<T,Rest1...>,
                                    typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::indices;
          constexpr auto& idx_b = IndexSecondTensor<Index<Idx0...>,Index<Idx1...>, Tensor<T,Rest0...>,Tensor<T,Rest1...>,
                                    typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::indices;
          constexpr auto& idx_out = IndexResultingTensor<Index<Idx0...>,Index<Idx1...>, Tensor<T,Rest0...>,Tensor<T,Rest1...>,
                                    typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::indices;

//          constexpr auto& maxes_out = no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
//                  typename std_ext::make_index_sequence<no_of_unique<Idx0...,Idx1...>::value>::type>::dims;

          constexpr int total = no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
                  typename std_ext::make_index_sequence<no_of_unique<Idx0...,Idx1...>::value>::type>::value;

          using maxes_out_type = typename no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
                  typename std_ext::make_index_sequence<no_of_unique<Idx0...,Idx1...>::value>::type>::type;

          constexpr auto& as_all = cartesian_product<maxes_out_type,typename std_ext::make_index_sequence<total>::type>::values;

          constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
          constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
          using Index_with_dims = typename put_dims_in_Index<OutTensor>::type;
          constexpr std::array<size_t,Index_with_dims::NoIndices> products_out = nprods<Index_with_dims,
                  typename std_ext::make_index_sequence<Index_with_dims::NoIndices>::type>::values;


          using vectorisability = is_vectorisable<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest1...>>;
          constexpr int stride = vectorisability::stride;
          using V = typename vectorisability::type;

          int it;
          V _vec_a;
          for (int i = 0; i < total; i+=stride) {
              int index_a = as_all[i][idx_a[a_dim-1]];
              for(it = 0; it< a_dim; it++) {
                  index_a += products_a[it]*as_all[i][idx_a[it]];
              }

              int index_b = as_all[i][idx_b[b_dim-1]];
              for(it = 0; it< b_dim; it++) {
                  index_b += products_b[it]*as_all[i][idx_b[it]];
              }
              int index_out = as_all[i][idx_out[idx_out.size()-1]];
              for(it = 0; it< idx_out.size(); it++) {
                  index_out += products_out[it]*as_all[i][idx_out[it]];
              }

              _vec_a.set(*(a_data+index_a));
              V _vec_out = _vec_a*V(b_data+index_b) +  V(out_data+index_out);
              _vec_out.store(out_data+index_out);
          }


//          constexpr int stride = 1;
//          for (int i = 0; i < total; i+=stride) {

//              int index_a = as_all[i][idx_a[a_dim-1]];
//              for(it = 0; it< a_dim; it++) {
//                  index_a += products_a[it]*as_all[i][idx_a[it]];
//              }

//              int index_b = as_all[i][idx_b[b_dim-1]];
//              for(it = 0; it< b_dim; it++) {
//                  index_b += products_b[it]*as_all[i][idx_b[it]];
//              }
//              int index_out = as_all[i][idx_out[idx_out.size()-1]];
//              for(it = 0; it< idx_out.size(); it++) {
//                  index_out += products_out[it]*as_all[i][idx_out[it]];
//              }

//              // std::cout << index_a << " " << index_b << " " << index_out << "\n";
//              out_data[index_out] += a_data[index_a]*b_data[index_b];
//          }

          return out;
      }
};


#endif




template<class Index_I, class Index_J,
         typename T, size_t ... Rest0, size_t ... Rest1>
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b)
-> decltype(extractor_contract_2<Index_I,Index_J>::contract_impl(a,b)) {
    return extractor_contract_2<Index_I,Index_J>::contract_impl(a,b);
}
//---------------------------------------------------------------------------------------------------------------------//













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
//        using resulting_tensor_2 =  typename get_resuling_tensor<resulting_index_1,Index<Idx3...>,
//                                        resulting_tensor_1,Tensor<T,Rest3...>>::type;
//        using resulting_index_2 =  typename get_resuling_index<resulting_index_1,Index<Idx3...>,
//                                        resulting_tensor_1,Tensor<T,Rest3...>>::type;


        resulting_tensor_0 tmp0 = contraction<Index<Idx0...>,Index<Idx1...>>(a,b);
        resulting_tensor_1 tmp1 = contraction<resulting_index_0,Index<Idx2...>>(tmp0,c);
        auto tmp2 = contraction<resulting_index_1,Index<Idx3...>>(tmp1,d);

        return tmp2;

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

#endif // CONTRACTION_H

