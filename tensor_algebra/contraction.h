#ifndef CONTRACTION_H
#define CONTRACTION_H

#include "tensor/Tensor.h"
#include "indicial.h"


namespace Fastor {

/////////

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
//template<size_t I, size_t... Is>
//struct filter_
//    :  concat_<typename std::conditional<Is == I, Index<>, Index<Is>>::type...> {};
template<int I, int... Is>
struct filter_
    :  concat_<typename std::conditional<Is == I, Index<>, Index<Is>>::type...> {};
//template<int I, int... Is>
//struct filter_ {
//    using type = concat_<typename std::conditional<Is == I, Index<>, Index<Is>>::type...>;
//};
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
////////////////////



template<class T>
struct prod_index;

template<size_t ... rest>
struct prod_index<Index<rest...>> {
    const static size_t value = prod<rest...>::value;
};

template<class T, class U>
struct prod_index2;

template<size_t ... rest0, size_t ... rest1, typename T>
struct prod_index2<Index<rest0...>,Tensor<T,rest1...>> {
    const static size_t value = prod<rest0...,rest1...>::value;
};

template<class T, class U>
struct find_loop_type;

template<size_t ... rest0, size_t ... rest1, typename T>
struct find_loop_type<Index<rest0...>,Tensor<T,rest1...>> {
    using type = apply_typelist_t<quote_c<size_t, Index>,
                        uniq_t<typelist_c<size_t, rest0...,rest1...>>>;
};


template<class T, size_t ... Rest>
struct contract_impl_prod;

template<typename T, size_t ... Rest>
struct contract_impl_prod<Tensor<T,Rest...>> {
    const static size_t value = prod<Rest...>::value;
};


////////////////


template<class T, size_t...Rest>
struct contract_vec_impl;

template<class T, size_t...Rest>
struct contract_vec_impl<Tensor<T,Rest...>> {
    static const size_t value = get_value<sizeof...(Rest),Rest...>::value;
};









///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////
//template<size_t N>
//constexpr int find_idx(const size_t (&ind)[N], int num, size_t i=0){
//    return ind[i]==num ? i : find_idx(ind,num,i+1);
//}

//template<size_t N>
//constexpr int find_idx_corrector(const size_t (&ind)[N], int num){
//    return ind[static_cast<int>(N)-1]==num ? static_cast<int>(N)-1 : N;
//}

//template<size_t N>
//constexpr int find_index(const size_t (&ind)[N], int num){
//    return (find_idx(ind,num)+1==static_cast<int>(N) &&
//            find_idx(ind,num)+1==find_idx_corrector(ind,num)) ? find_idx_corrector(ind,num) : find_idx(ind,num);
//}
///////////////////////////////

// this is a meta-function equivalent to numpy's "where"
template<size_t N>
constexpr int find_index(const size_t (&ind)[N], int num, size_t i=0){
    return (i==N) ? N : (ind[i]==num ? i : find_index(ind,num,i+1));
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


template<class Ind0, class Ind1, class Tensor0, class Tensor1, class Seq>
struct pair_flop_cost;

template<class T, size_t... Idx0, size_t... Idx1, size_t ...Rest0, size_t ...Rest1, size_t... ss>
struct pair_flop_cost<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>> {

    static constexpr size_t ind0[sizeof...(Idx0)] = {Idx0... };
    static constexpr size_t ind1[sizeof...(Idx1)] = {Idx1... };
    static constexpr int nums0[sizeof...(Rest0)] = {Rest0... };
    static constexpr int nums1[sizeof...(Rest1)] = {Rest1... };

//    static constexpr std::array<int,sizeof...(Idx1)> cost = {cost_model(ind0,ind1,nums1,ss)...};
    static constexpr int cost_tensor0 = prod<Rest0...>::value;
    static constexpr int remaining_cost = prod<retrieve_value(ind0,ind1,nums1,ss)...>::value;
    static constexpr int value = cost_tensor0*remaining_cost;
//    static constexpr int cost1 = find_idx(resulting_type::_IndexHolder,3);

};

//template<class T, size_t... Idx0, size_t... Idx1, size_t ...Rest0, size_t ...Rest1, size_t... ss>
//constexpr std::array<int,sizeof...(Idx1)> costs<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>>::cost;



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

template<class T, class U, class V, class W, class Seq>
struct no_of_loops_to_set;

template<size_t ... Idx0, size_t ... Idx1, size_t ... Rest0, size_t ... Rest1, typename T, size_t ... ss>
struct no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,std_ext::index_sequence<ss...>> {

    using index_temp = apply_typelist_t<quote_c<size_t, Index>,
                    uniq_t<typelist_c<size_t, Idx0...,Idx1...>>>;

    static constexpr size_t concat_idx[sizeof...(Idx0)+sizeof...(Idx1)] = {Idx0...,Idx1...};
    static constexpr size_t concat_nums[sizeof...(Idx0)+sizeof...(Idx1)] = {Rest0...,Rest1...};
//    static constexpr std::array<size_t,no_of_unique<Idx0...,Idx1...>::value> idx_in_concat = {find_index(concat_idx,index_temp::_IndexHolder[ss])...};
//    static constexpr std::array<size_t,no_of_unique<Idx0...,Idx1...>::value> idx = {concat_nums[idx_in_concat[ss]]...};

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
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


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


/////////
template<class Dims>
struct put_dims_in_Index;

template<size_t ... Rest, typename T>
struct put_dims_in_Index<Tensor<T, Rest...>> {
    using type = Index<Rest...>;
};
/////////



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

//    static void foo() {
//        constexpr int as2[sizeof...(Rest)] = {Rest...};
//        constexpr std::array<int,sizeof...(ss)> values2 = {cartesian_product_single(as2,prod<Rest...>::value,I,ss)...};
////        constexpr std::array<int,sizeof...(ss)> values2 = {find_remaining(as2,480,ss)...};
//        print(values2);
//    }
};

template<int I, size_t ... Rest, size_t ... ss, typename T>
constexpr std::array<int,sizeof...(ss)> gen_single_cartesian_product<I,Tensor<T,Rest...>,std_ext::index_sequence<ss...>>::values;


template<int i, size_t ... Rest>
constexpr std::array<int,sizeof...(Rest)> all_cartesian_product() {
    return gen_single_cartesian_product<i,Tensor<double,Rest...>,typename std_ext::make_index_sequence<sizeof...(Rest)>::type>::values;
}


template<class Tens, class Seq>
struct cartesian_product;

template<size_t ... Rest, size_t ... ss, typename T>
struct cartesian_product<Tensor<T,Rest...>,std_ext::index_sequence<ss...>> {
    static constexpr std::array<std::array<int,sizeof...(Rest)>,sizeof...(ss)> values = {all_cartesian_product<ss,Rest...>()...};

//    static void foo() {
////        constexpr int as2[sizeof...(ss)] = {ss...};
////        constexpr std::array<int,sizeof...(ss)> values2 = {get_all2<ss,Rest...>()...};
//        constexpr std::array<std::array<int,sizeof...(Rest)>,sizeof...(ss)> values2 = {get_all2<ss,Rest...>()...};
////        constexpr std::array<int,sizeof...(ss)> values2 = {find_remaining(as2,480,ss)...};
//        print(values2);
//    }
};

template<size_t ... Rest, size_t ... ss, typename T>
constexpr std::array<std::array<int,sizeof...(Rest)>,sizeof...(ss)> cartesian_product<Tensor<T,Rest...>,std_ext::index_sequence<ss...>>::values;


//////////



// std::enable_if<contract_vec_impl<typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
// typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type>::value % 2=0,bool>::type=0

template<class T, class U>
struct extractor_contract_2 {};

template<size_t ... Idx0, size_t ... Idx1>
struct extractor_contract_2<Index<Idx0...>, Index<Idx1...>> {

    // float
    template<typename T, size_t ... Rest0, size_t ... Rest1, typename std::enable_if<std::is_same<T,float>::value,bool>::type=0>
      static
      typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
               typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type
      contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) {


          using OutTensor = typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
                                    typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type;

          OutTensor out;
          out.zeros();
          T *a_data = a.data();
          T *b_data = b.data();
          T *out_data = out.data();

          constexpr int a_dim = sizeof...(Rest0);
          constexpr int b_dim = sizeof...(Rest1);
          constexpr std::array<int,b_dim> maxes_b = {Rest1...};
          constexpr std::array<int,a_dim> ma = {Idx0...};
          constexpr std::array<int,b_dim> mb = {Idx1...};

          // Note that maxes_out is not the same length/size as the resulting
          // tensor (out), but it rather stores the info about number of loops
          std::vector<int> maxes_out = {Rest0...};
          std::vector<int> idx_a(a_dim), idx_b;
          std::iota(idx_a.begin(),idx_a.end(),0);


          std::vector<int> mout = {Idx0...};

          for (int i=0; i<b_dim; ++i) {
              auto itt = std::find(ma.begin(), ma.end(), mb[i]);
              if (itt == ma.end()) {
                  maxes_out.push_back(maxes_b[i]);
                  mout.push_back(mb[i]);
              }
          }

          // find where idx_out is
          std::vector<int> idx_out;
          for (int i=0; i<a_dim; ++i) {
              auto itt = std::find(mb.begin(), mb.end(), ma[i]);
              if (itt == mb.end()) {
                  idx_out.push_back(i);
              }
          }
          int dim_so_far = idx_out.size();
          for (int i=0; i<b_dim; ++i) {
              auto itt = std::find(ma.begin(), ma.end(), mb[i]);
              if (itt == ma.end()) {
                  idx_out.push_back(i+dim_so_far);
              }
          }

          for (int i=0; i<b_dim; ++i) {
              auto itt = std::find(mout.begin(), mout.end(), mb[i]);
              if (itt != mout.end()) {
                  idx_b.push_back(itt-mout.begin());
              }
          }

          std::array<int,OutTensor::Dimension> products_out; products_out[0]=0;
          for (int j=idx_out.size()-1; j>0; --j) {
              int num = maxes_out[idx_out[idx_out.size()-1]];
              for (int k=0; k<j-1; ++k) {
                  num *= maxes_out[idx_out[idx_out.size()-1-k-1]];
              }
              products_out[j] = num;
          }
          std::reverse(products_out.begin(),products_out.end());


          constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
          constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;

          constexpr int out_dim =  no_of_unique<Idx0...,Idx1...>::value;
          int as[out_dim];
          std::fill(as,as+out_dim,0);
          int it;


          int total = 1;
          {
              for (int i=0; i<maxes_out.size(); ++i) {
                  total *=maxes_out[i];
              }
          }

          constexpr size_t unroller = contract_vec_impl<OutTensor>::value;

          if (unroller % 4 == 0 && unroller % 8 != 0) {
              using V = SIMDVector<T,128>;
              V _vec_a;
              constexpr int stride = 4;
              for (int i = 0; i < total; i+=stride) {
                  int remaining = total;
                  for (int n = 0; n < out_dim; ++n) {
                      remaining /= maxes_out[n];
                      as[n] = ( i / remaining ) % maxes_out[n];
                  }

                  int index_a = as[a_dim-1];
                  for(it = 0; it< a_dim; it++) {
                      index_a += products_a[it]*as[idx_a[it]];
                  }
                  int index_b = as[out_dim-1];
                  for(it = 0; it< b_dim; it++) {
                      index_b += products_b[it]*as[idx_b[it]];
                  }
                  int index_out = as[out_dim-1];
                  for(it = 0; it< idx_out.size(); it++) {
                      index_out += products_out[it]*as[idx_out[it]];
                  }

                  _vec_a.set(*(a_data+index_a));
                  V _vec_out = _vec_a*V(b_data+index_b) +  V(out_data+index_out);
                  _vec_out.store(out_data+index_out);
              }
          }
          else if (unroller % 4 == 0 && unroller % 8 == 0) {
              using V = SIMDVector<T,256>;
              V _vec_a;

              constexpr int stride = 8;
              for (int i = 0; i < total; i+=stride) {
                  int remaining = total;
                  for (int n = 0; n < out_dim; ++n) {
                      remaining /= maxes_out[n];
                      as[n] = ( i / remaining ) % maxes_out[n];
                  }

                  int index_a = as[a_dim-1];
                  for(it = 0; it< a_dim; it++) {
                      index_a += products_a[it]*as[idx_a[it]];
                  }
                  int index_b = as[out_dim-1];
                  for(it = 0; it< b_dim; it++) {
                      index_b += products_b[it]*as[idx_b[it]];
                  }
                  int index_out = as[out_dim-1];
                  for(it = 0; it< idx_out.size(); it++) {
                      index_out += products_out[it]*as[idx_out[it]];
                  }

                  _vec_a.set(*(a_data+index_a));
                  V _vec_out = _vec_a*V(b_data+index_b) +  V(out_data+index_out);
                  _vec_out.store(out_data+index_out);

              }
          }
          else {
              constexpr int stride = 1;
              for (int i = 0; i < total; i+=stride) {
                  int remaining = total;
                  for (int n = 0; n < out_dim; ++n) {
                      remaining /= maxes_out[n];
                      as[n] = ( i / remaining ) % maxes_out[n];
                  }

                  int index_a = as[a_dim-1];
                  for(it = 0; it< a_dim; it++) {
                      index_a += products_a[it]*as[idx_a[it]];
                  }
                  int index_b = as[out_dim-1];
                  for(it = 0; it< b_dim; it++) {
                      index_b += products_b[it]*as[idx_b[it]];
                  }
                  int index_out = as[out_dim-1];
                  for(it = 0; it< idx_out.size(); it++) {
                      index_out += products_out[it]*as[idx_out[it]];
                  }
    //              std::cout << index_a << " " << index_b << " " << index_out << "\n";
                    out_data[index_out] += a_data[index_a]*b_data[index_b];
              }
          }

          return out;
    }

      // double
      template<typename T, size_t ... Rest0, size_t ... Rest1, typename std::enable_if<std::is_same<T,double>::value,bool>::type=0>
        static
        typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
                 typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type
        contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) {


          using OutTensor = typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
                                    typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type;

          OutTensor out;
          out.zeros();
          const T *a_data = a.data();
          const T *b_data = b.data();
          T *out_data = out.data();

          constexpr int a_dim = sizeof...(Rest0);
          constexpr int b_dim = sizeof...(Rest1);
//          constexpr std::array<int,b_dim> maxes_b = {Rest1...};
//          constexpr std::array<int,a_dim> ma = {Idx0...};
//          constexpr std::array<int,b_dim> mb = {Idx1...};

//          std::vector<int> maxes_out = {Rest0...};
//          std::vector<int> idx_a(a_dim), idx_b;
//          std::iota(idx_a.begin(),idx_a.end(),0);
//          std::vector<int> mout = {Idx0...};

//          for (int i=0; i<b_dim; ++i) {
//              auto itt = std::find(ma.begin(), ma.end(), mb[i]);
//              if (itt == ma.end()) {
//                  maxes_out.push_back(maxes_b[i]);
//                  mout.push_back(mb[i]);
//              }
//          }

//          // find where idx_out is
//          std::vector<int> idx_out;
//          for (int i=0; i<a_dim; ++i) {
//              auto itt = std::find(mb.begin(), mb.end(), ma[i]);
//              if (itt == mb.end()) {
//                  idx_out.push_back(i);
////                  print(itt-mb.begin());
//              }
//          }

//          int dim_so_far = idx_out.size();
//          for (int i=0; i<b_dim; ++i) {
//              auto itt = std::find(ma.begin(), ma.end(), mb[i]);
//              if (itt == ma.end()) {
//                  idx_out.push_back(i+dim_so_far);
//              }
//          }


//          for (int i=0; i<b_dim; ++i) {
//              auto itt = std::find(mout.begin(), mout.end(), mb[i]);
//              if (itt != mout.end()) {
//                  idx_b.push_back(itt-mout.begin());
//              }
//          }

          constexpr auto& idx_a = IndexFirstTensor<Index<Idx0...>,Index<Idx1...>, Tensor<T,Rest0...>,Tensor<T,Rest1...>,
                                    typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::indices;
          constexpr auto& idx_b = IndexSecondTensor<Index<Idx0...>,Index<Idx1...>, Tensor<T,Rest0...>,Tensor<T,Rest1...>,
                                    typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::indices;
          constexpr auto& idx_out = IndexResultingTensor<Index<Idx0...>,Index<Idx1...>, Tensor<T,Rest0...>,Tensor<T,Rest1...>,
                                    typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::indices;

          constexpr auto& maxes_out = no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
                  typename std_ext::make_index_sequence<no_of_unique<Idx0...,Idx1...>::value>::type>::dims;

          constexpr int total = no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
                  typename std_ext::make_index_sequence<no_of_unique<Idx0...,Idx1...>::value>::type>::value;

//          std::array<int,OutTensor::Dimension> products_out; products_out[0]=0;
//          for (int j=idx_out.size()-1; j>0; --j) {
//              int num = maxes_out[idx_out[idx_out.size()-1]];
//              for (int k=0; k<j-1; ++k) {
//                  num *= maxes_out[idx_out[idx_out.size()-1-k-1]];
//              }
//              products_out[j] = num;
//          }
//          std::reverse(products_out.begin(),products_out.end());


          constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
          constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
          using Index_with_dims = typename put_dims_in_Index<OutTensor>::type;
          constexpr std::array<size_t,Index_with_dims::NoIndices> products_out = nprods<Index_with_dims,
                  typename std_ext::make_index_sequence<Index_with_dims::NoIndices>::type>::values;
//          print(type_name<maxxx>());
//          print(products_outt);
//        print(products_out,products_outt);

//          print(idx_aa,idx_bb,idx_outt);
//          print("\n\n");
//          print(idx_a,idx_b,idx_out);


//          int out_dim = maxes_out.size();
          constexpr int out_dim =  no_of_unique<Idx0...,Idx1...>::value;

          int as[out_dim];
          std::fill(as,as+out_dim,0);
          int it;


//          int total = 1;
//          {
//              for (int i=0; i<maxes_out.size(); ++i) {
//                  total *=maxes_out[i];
//              }
//          }
          print(maxes_out);

          constexpr int stride = 1;
          for (int i = 0; i < total; i+=stride) {
              int remaining = total;
              for (int n = 0; n < out_dim; ++n) {
                  remaining /= maxes_out[n];
                  as[n] = ( i / remaining ) % maxes_out[n];
//                      std::cout << as[n] << " ";
//                  print(remaining);
              }
//                  print();

              int index_a = as[a_dim-1];
              for(it = 0; it< a_dim; it++) {
                  index_a += products_a[it]*as[idx_a[it]];
              }
//              int index_b = as[out_dim-1];
//              int index_b = as[out_dim-2];
              int index_b = as[idx_b[b_dim-1]];
              for(it = 0; it< b_dim; it++) {
                  index_b += products_b[it]*as[idx_b[it]];
              }
//              int index_out = as[out_dim-1];
              int index_out = as[idx_out[idx_out.size()-1]];
              for(it = 0; it< idx_out.size(); it++) {
                  index_out += products_out[it]*as[idx_out[it]];
              }
//              std::cout << as[idx_b[0]] << " " << as[idx_b[1]] << " " <<  as[idx_b[2]] << "\n";
//              std::cout << as[idx_b[0]]*12 << " " << as[idx_b[1]]*4 << " " <<  as[idx_b[2]] << "\n";
//              std::cout << products_b[0] << " " << products_b[1] << " " <<  products_b[2] << "\n";

//              std::cout << index_a << " " << index_b << " " << index_out << "\n";
                out_data[index_out] += a_data[index_a]*b_data[index_b];
          }


//          constexpr size_t unroller = contract_vec_impl<OutTensor>::value;

//          print(type_name<OutTensor>());
//          print(idx_a,idx_b,idx_out);

//          if (unroller % 2 != 0 && unroller % 4 != 0) {
//              using V = SIMDVector<T,128>;
//              V _vec_a;
//              constexpr int stride = 2;
//              for (int i = 0; i < total; i+=stride) {
//                  int remaining = total;
//                  for (int n = 0; n < out_dim; ++n) {
//                      remaining /= maxes_out[n];
//                      as[n] = ( i / remaining ) % maxes_out[n];
//                  }

//                  int index_a = as[a_dim-1];
//                  for(it = 0; it< a_dim; it++) {
//                      index_a += products_a[it]*as[idx_a[it]];
//                  }
//                  int index_b = as[out_dim-1];
//                  for(it = 0; it< b_dim; it++) {
//                      index_b += products_b[it]*as[idx_b[it]];
//                  }
//                  int index_out = as[out_dim-1];
//                  for(it = 0; it< idx_out.size(); it++) {
//                      index_out += products_out[it]*as[idx_out[it]];
//                  }

//                  _vec_a.set(*(a_data+index_a));
//                  V _vec_out = _vec_a*V(b_data+index_b) +  V(out_data+index_out);
//                  _vec_out.store(out_data+index_out);
//              }
//          }
//          else if (unroller % 2 != 0 && unroller % 4 == 0) {
//              using V = SIMDVector<T,256>;
//              V _vec_a;

//              constexpr int stride = 4;
//              for (int i = 0; i < total; i+=stride) {
//                  int remaining = total;
//                  for (int n = 0; n < out_dim; ++n) {
//                      remaining /= maxes_out[n];
//                      as[n] = ( i / remaining ) % maxes_out[n];
//                  }

//                  int index_a = as[a_dim-1];
//                  for(it = 0; it< a_dim; it++) {
//                      index_a += products_a[it]*as[idx_a[it]];
//                  }
//                  int index_b = as[out_dim-1];
//                  for(it = 0; it< b_dim; it++) {
//                      index_b += products_b[it]*as[idx_b[it]];
//                  }
//                  int index_out = as[out_dim-1];
//                  for(it = 0; it< idx_out.size(); it++) {
//                      index_out += products_out[it]*as[idx_out[it]];
//                  }

//                  _vec_a.set(*(a_data+index_a));
//                  V _vec_out = _vec_a*V(b_data+index_b) +  V(out_data+index_out);
//                  _vec_out.store(out_data+index_out);

//              }
//          }
//          else {
//              constexpr int stride = 1;
//              for (int i = 0; i < total; i+=stride) {
//                  int remaining = total;
//                  for (int n = 0; n < out_dim; ++n) {
//                      remaining /= maxes_out[n];
//                      as[n] = ( i / remaining ) % maxes_out[n];
////                      std::cout << as[n] << " ";
//                  }
////                  print();

//                  int index_a = as[a_dim-1];
//                  for(it = 0; it< a_dim; it++) {
//                      index_a += products_a[it]*as[idx_a[it]];
//                  }
//                  int index_b = as[out_dim-1];
//                  for(it = 0; it< b_dim; it++) {
//                      index_b += products_b[it]*as[idx_b[it]];
//                  }
//                  int index_out = as[out_dim-1];
//                  for(it = 0; it< idx_out.size(); it++) {
//                      index_out += products_out[it]*as[idx_out[it]];
//                  }
//                  std::cout << index_a << " " << index_b << " " << index_out << "\n";
//                    out_data[index_out] += a_data[index_a]*b_data[index_b];
//              }
//          }

          return out;
      }
};

template<class Index_I, class Index_J,
         typename T, size_t ... Rest0, size_t ... Rest1>
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b)
-> decltype(extractor_contract_2<Index_I,Index_J>::contract_impl(a,b)) {
    return extractor_contract_2<Index_I,Index_J>::contract_impl(a,b);
}



/////////////////////////

























template<int N, int... Rest>
struct iota_array {
    static constexpr auto& value = iota_array<N - 1, N, Rest...>::value;
};
//template<int... Rest>
//struct iota_array<0, Rest...> {
//    static constexpr std::array<int,sizeof...(Rest)+1> value = { 0, Rest... };
//};
//template<int... Rest>
//constexpr std::array<int,sizeof...(Rest)+1> iota_array<0, Rest...>::value;

template<int... Rest>
struct iota_array<0, Rest...> {
    static constexpr int value[] = { 0, Rest... };
};
template<int... Rest>
constexpr int iota_array<0, Rest...>::value[];














//template<class T, T ... Rest>
//struct extractor_contract {};
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
        //----------------------------------------------
        // first two tensors contracted first
//        using loop_type_0 = apply_typelist_t<quote_c<size_t, Index>,
//                    uniq_t<typelist_c<size_t, Rest0...,Rest1...>>>;
//        using contract_0 = typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
//                                  typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type;
//        using loop_type_1 = typename find_loop_type<Index<Rest2...>,contract_0>::type;
//        constexpr size_t flop_count_0 = prod_index<loop_type_0>::value + prod_index<loop_type_1>::value;
//        // second and third tensors contracted first
//        using loop_type_2 = apply_typelist_t<quote_c<size_t, Index>,
//                    uniq_t<typelist_c<size_t, Rest1...,Rest2...>>>;
//        using contract_1 = typename contraction_impl<Index<Idx1...,Idx2...>, Tensor<T,Rest1...,Rest2...>,
//                                  typename std_ext::make_index_sequence<sizeof...(Rest1)+sizeof...(Rest2)>::type>::type;
//        using loop_type_3 = typename find_loop_type<Index<Rest0...>,contract_1>::type;
//        constexpr size_t flop_count_1 = prod_index<loop_type_2>::value + prod_index<loop_type_3>::value;
//        // first and third tensors contracted first
//        using loop_type_4 = apply_typelist_t<quote_c<size_t, Index>,
//                    uniq_t<typelist_c<size_t, Rest0...,Rest2...>>>;
//        using contract_2 = typename contraction_impl<Index<Idx0...,Idx2...>, Tensor<T,Rest0...,Rest2...>,
//                                  typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest2)>::type>::type;
//        using loop_type_5 = typename find_loop_type<Index<Rest1...>,contract_2>::type;
//        constexpr size_t flop_count_2 = prod_index<loop_type_4>::value + prod_index<loop_type_5>::value;
//        // all tensors contracted at once
//        using loop_type_6 = apply_typelist_t<quote_c<size_t, Index>,
//                    uniq_t<typelist_c<size_t, Rest1...,Rest1...,Rest2...>>>;
//        constexpr size_t flop_count_3 = prod_index<loop_type_6>::value;
        //----------------------------------------------



        // Perform depth-first search
        //////////////
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

//        constexpr int which_variant_pairs = meta_argmin<flop_count_01,flop_count_02,flop_count_12>::value;
//        print(which_variant);
//        print(flop_count_01,flop_count_02,flop_count_12,flop_count_012);
//        print(meta_argmin<2,3,1,5>::value);


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


}

#endif // CONTRACTION_H

