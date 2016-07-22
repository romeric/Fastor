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
};
////////////////////



template<class T>
struct prod_index;
//template<class T, class U>
//struct prod_index;

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

template<int a_dim, int out_dim>
std::vector<int> which_where(const size_t * __restrict__ a, const size_t * __restrict__ out) {

    std::vector<int> vec;
    for (auto i=0; i<a_dim; ++i) {
        for (auto j=0; j<out_dim; ++j) {
            if (a[i]==out[j]) {
                vec.push_back(j);
            }
        }
    }
    return vec;
}

template<class T, size_t...Rest>
struct contract_vec_impl;

template<class T, size_t...Rest>
struct contract_vec_impl<Tensor<T,Rest...>> {
    static const size_t value = get_value<sizeof...(Rest),Rest...>::value;
};


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

//    // float
//    template<typename T, size_t ... Rest0, size_t ... Rest1, typename std::enable_if<std::is_same<T,float>::value,bool>::type=0>
//      static
//      typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
//               typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type
//      contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) {


//        using OutTensor = typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
//                                  typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type;

////        print(contract_vec_impl<OutTensor>::value);
////        print(type_name<OutTensor>());
//        OutTensor out;

//        out.zeros();
//        T *a_data = a.data();
//        T *b_data = b.data();
//        T *out_data = out.data();

//        constexpr int a_dim = sizeof...(Rest0);
//        constexpr int b_dim = sizeof...(Rest1);
////        int out_dim = no_of_unique<Rest0...,Rest1...>::value;
////        constexpr std::array<int,a_dim> maxes_a = {Rest0...};
////        constexpr std::array<int,b_dim> maxes_b = {Rest1...};
//        constexpr size_t maxes_a[a_dim] = {Rest0...};
//        constexpr size_t maxes_b[b_dim] = {Rest1...};
////        constexpr size_t maxes_out[a_dim+b_dim] = {Rest0...,Rest1...};
////        constexpr std::array<size_t,a_dim+b_dim> maxes_out = {Rest0...,Rest1...};

//        constexpr std::array<int,a_dim> ma = {Idx0...};
//        constexpr std::array<int,b_dim> mb = {Idx1...};

//        std::vector<int> maxes_out;
//        std::vector<int> idx_a(a_dim), idx_b, idx_tmp, idx_out(OutTensor::Dimension);
//        std::iota(idx_a.begin(),idx_a.end(),0);
//        {
//            for (int i=0; i<a_dim; ++i) {
//                maxes_out.push_back(maxes_a[i]);
//                idx_tmp.push_back(i);
//            }
//            int min_dim = a_dim;
//            if (min_dim>static_cast<int>(b_dim)) min_dim = b_dim;
//            int counter = idx_tmp.size();
//            for (int i=0; i<min_dim; ++i) {
//                if (ma[i]!=mb[i]) {
//                    maxes_out.push_back(maxes_b[i]);
//                    idx_tmp.push_back(counter);
//                    counter++;
//                }
//            }

//            // find where idx_b is in maxes_out
//            counter = a_dim;
//            for (int i=0; i<b_dim; ++i) {
//                if (ma[i]!=mb[i]) {
//                    idx_b.push_back(idx_tmp[counter]);
//                    counter++;
//                }
//                else
//                    idx_b.push_back(idx_tmp[i]);
//            }

//            std::vector<int> inn;
//            for (int i=0; i<a_dim; ++i) {
//                if (ma[i]==mb[i]) {
//                    inn.push_back(i);
//                }
//            }
//            std::vector<int>::iterator itr;
//            itr = std::set_difference(idx_tmp.begin(),idx_tmp.end(),inn.begin(),inn.end(),idx_out.begin());
//        }
//        int out_dim = maxes_out.size();


//        std::array<int,a_dim> products_a; products_a[0]=0;
//        for (int j=a_dim-1; j>0; --j) {
//            int num = maxes_a[a_dim-1];
//            for (int k=0; k<j-1; ++k) {
//                num *= maxes_a[a_dim-1-k-1];
//            }
//            products_a[j] = num;
//        }
//        std::array<int,b_dim> products_b; products_b[0]=0;
//        for (int j=b_dim-1; j>0; --j) {
//            int num = maxes_b[b_dim-1];
//            for (int k=0; k<j-1; ++k) {
//                num *= maxes_b[b_dim-1-k-1];
//            }
//            products_b[j] = num;
//        }

//        std::vector<int> products_out(idx_out.size()); products_out[0]=0;
//        for (int j=idx_out.size()-1; j>0; --j) {
//            int num = maxes_out[idx_out[idx_out.size()-1]];
//            for (int k=0; k<j-1; ++k) {
//                num *= maxes_out[idx_out[idx_out.size()-1-k-1]];
//            }
//            products_out[j] = num;
//        }

//        std::reverse(products_a.begin(),products_a.end());
//        std::reverse(products_b.begin(),products_b.end());
//        std::reverse(products_out.begin(),products_out.end());
////        print(products_a,products_b,products_out);

//        int as[out_dim];
//        std::fill(as,as+out_dim,0);
//        int it;

//        int total = 1;
//        {
//            for (int i=0; i<maxes_out.size(); ++i) {
//                total *=maxes_out[i];
//            }
//        }

//        constexpr size_t unroller = contract_vec_impl<OutTensor>::value;

//        if (unroller % 4 == 0 && unroller % 8 != 0) {
//            using V = SIMDVector<T,128>;
//            V _vec_a;

//            constexpr int stride = 4;
//            for (int i = 0; i < total; i+=stride) {
//                int remaining = total;
//                for (int n = 0; n < out_dim; ++n) {
//                    remaining /= maxes_out[n];
//                    as[n] = ( i / remaining ) % maxes_out[n];
//                }

//                int index_a = as[a_dim-1];
//                for(it = 0; it< a_dim; it++) {
//                    index_a += products_a[it]*as[idx_a[it]];
//                }
//                int index_b = as[out_dim-1];
//                for(it = 0; it< b_dim; it++) {
//                    index_b += products_b[it]*as[idx_b[it]];
//                }
//                int index_out = as[out_dim-1];
//                for(it = 0; it< idx_out.size(); it++) {
//                    index_out += products_out[it]*as[idx_out[it]];
//                }

////                std::cout << index_a << " " << index_b << " " << index_out << "\n";
////                out_data[index_out] += a_data[index_a]*b_data[index_b];

//                _vec_a.set(*(a_data+index_a));
//                V _vec_out = _vec_a*V(b_data+index_b) +  V(out_data+index_out);
//                _vec_out.store(out_data+index_out);


//            }
//        }
//        else if (unroller % 4 == 0 && unroller % 8 == 0) {
//            using V = SIMDVector<T,256>;
//            V _vec_a;

//            constexpr int stride = 8;
//            for (int i = 0; i < total; i+=stride) {
//                int remaining = total;
//                for (int n = 0; n < out_dim; ++n) {
//                    remaining /= maxes_out[n];
//                    as[n] = ( i / remaining ) % maxes_out[n];
//                }

//                int index_a = as[a_dim-1];
//                for(it = 0; it< a_dim; it++) {
//                    index_a += products_a[it]*as[idx_a[it]];
//                }
//                int index_b = as[out_dim-1];
//                for(it = 0; it< b_dim; it++) {
//                    index_b += products_b[it]*as[idx_b[it]];
//                }
//                int index_out = as[out_dim-1];
//                for(it = 0; it< idx_out.size(); it++) {
//                    index_out += products_out[it]*as[idx_out[it]];
//                }

////                std::cout << index_a << " " << index_b << " " << index_out << "\n";
////                out_data[index_out] += a_data[index_a]*b_data[index_b];

//                _vec_a.set(*(a_data+index_a));
//                V _vec_out = _vec_a*V(b_data+index_b) +  V(out_data+index_out);
//                _vec_out.store(out_data+index_out);

//            }
//        }
//        else {

//            constexpr int stride = 1;
//            for (int i = 0; i < total; i+=stride) {
//                int remaining = total;
//                for (int n = 0; n < out_dim; ++n) {
//                    remaining /= maxes_out[n];
//                    as[n] = ( i / remaining ) % maxes_out[n];
//                }

//                int index_a = as[a_dim-1];
//                for(it = 0; it< a_dim; it++) {
//                    index_a += products_a[it]*as[idx_a[it]];
//                }
//                int index_b = as[out_dim-1];
//                for(it = 0; it< b_dim; it++) {
//                    index_b += products_b[it]*as[idx_b[it]];
//                }
//                int index_out = as[out_dim-1];
//                for(it = 0; it< idx_out.size(); it++) {
//                    index_out += products_out[it]*as[idx_out[it]];
//                }

////                std::cout << index_a << " " << index_b << " " << index_out << "\n";
//                out_data[index_out] += a_data[index_a]*b_data[index_b];

//            }
//        }

////        print(out);
//        return out;
//    }


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
          T *a_data = a.data();
          T *b_data = b.data();
          T *out_data = out.data();

          constexpr int a_dim = sizeof...(Rest0);
          constexpr int b_dim = sizeof...(Rest1);
//          constexpr std::array<int,a_dim> maxes_a = {Rest0...};
          constexpr std::array<int,b_dim> maxes_b = {Rest1...};

          constexpr std::array<int,a_dim> ma = {Idx0...};
          constexpr std::array<int,b_dim> mb = {Idx1...};
//          print(no_of_uniques2<0,1,1,2>::value);
//          print(ma,mb);

          // Note that maxes_out is not the same length/size as the resulting
          // tensor (out), but it rather stores the info about number of loops
          std::vector<int> maxes_out = {Rest0...};
          std::vector<int> idx_a(a_dim), idx_b;
//          std::copy(maxes_a.begin(),maxes_a.end(),maxes_out.begin());
          std::iota(idx_a.begin(),idx_a.end(),0);

//          std::vector<int> mout(a_dim);
//          std::copy(ma.begin(),ma.end(),mout.begin());

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


//          std::array<int,a_dim> products_a; products_a[0]=0;
//          for (int j=a_dim-1; j>0; --j) {
//              int num = maxes_a[a_dim-1];
//              for (int k=0; k<j-1; ++k) {
//                  num *= maxes_a[a_dim-1-k-1];
//              }
//              products_a[j] = num;
//          }

//          std::array<int,b_dim> products_b; products_b[0]=0;
//          for (int j=b_dim-1; j>0; --j) {
//              int num = maxes_b[b_dim-1];
//              for (int k=0; k<j-1; ++k) {
//                  num *= maxes_b[b_dim-1-k-1];
//              }
//              products_b[j] = num;
//          }

//          std::vector<int> products_out(idx_out.size()); products_out[0]=0;
//          for (int j=idx_out.size()-1; j>0; --j) {
//              int num = maxes_out[idx_out[idx_out.size()-1]];
//              for (int k=0; k<j-1; ++k) {
//                  num *= maxes_out[idx_out[idx_out.size()-1-k-1]];
//              }
//              products_out[j] = num;
//          }

          std::array<int,OutTensor::Dimension> products_out; products_out[0]=0;
          for (int j=idx_out.size()-1; j>0; --j) {
              int num = maxes_out[idx_out[idx_out.size()-1]];
              for (int k=0; k<j-1; ++k) {
                  num *= maxes_out[idx_out[idx_out.size()-1-k-1]];
              }
              products_out[j] = num;
          }

//          std::reverse(products_a.begin(),products_a.end());
//          std::reverse(products_b.begin(),products_b.end());
          std::reverse(products_out.begin(),products_out.end());
//          print(products_a,products_b,products_out);


          constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
          constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
//          constexpr std::array<size_t,out_dim> products_outt = nprods<Index<Rest0...,Rest1...>,typename std_ext::make_index_sequence<out_dim>::type>::values;
//          print(products_out,products_outt);
//          print(products_out);

//          int out_dim = maxes_out.size();
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

          if (unroller % 2 == 0 && unroller % 4 != 0) {
              using V = SIMDVector<T,128>;
              V _vec_a;
              constexpr int stride = 2;
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
          else if (unroller % 2 == 0 && unroller % 4 == 0) {
              using V = SIMDVector<T,256>;
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
          else {
              constexpr int stride = 1;
              for (int i = 0; i < total; i+=stride) {
                  int remaining = total;
                  for (int n = 0; n < out_dim; ++n) {
                      remaining /= maxes_out[n];
                      as[n] = ( i / remaining ) % maxes_out[n];
    //                  print(as[n]);
                  }
    //              print();

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



//      // double
//      template<typename T, size_t ... Rest0, size_t ... Rest1, typename std::enable_if<std::is_same<T,double>::value,bool>::type=0>
//        static
//        typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
//                 typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type
//        contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) {


//          using OutTensor = typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
//                                    typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type;

//          OutTensor out;
//          out.zeros();
//          T *a_data = a.data();
//          T *b_data = b.data();
//          T *out_data = out.data();

//          constexpr int a_dim = sizeof...(Rest0);
//          constexpr int b_dim = sizeof...(Rest1);
//          constexpr size_t maxes_a[a_dim] = {Rest0...};
//          constexpr size_t maxes_b[b_dim] = {Rest1...};

//          constexpr std::array<int,a_dim> ma = {Idx0...};
//          constexpr std::array<int,b_dim> mb = {Idx1...};

//          std::vector<int> maxes_out;
//          std::vector<int> idx_a(a_dim), idx_b, idx_tmp, idx_out(OutTensor::Dimension);
//          std::iota(idx_a.begin(),idx_a.end(),0);
//          {
//              for (int i=0; i<a_dim; ++i) {
//                  maxes_out.push_back(maxes_a[i]);
//                  idx_tmp.push_back(i);
//              }
//              int min_dim = a_dim;
//              if (min_dim>static_cast<int>(b_dim)) min_dim = b_dim;
//              int counter = idx_tmp.size();
//              for (int i=0; i<min_dim; ++i) {
//                  if (ma[i]!=mb[i]) {
//                      maxes_out.push_back(maxes_b[i]);
//                      idx_tmp.push_back(counter);
//                      counter++;
//                  }
//              }

//              // find where idx_b is in maxes_out
//              counter = a_dim;
//              for (int i=0; i<b_dim; ++i) {
//                  if (ma[i]!=mb[i]) {
//                      idx_b.push_back(idx_tmp[counter]);
//                      counter++;
//                  }
//                  else
//                      idx_b.push_back(idx_tmp[i]);
//              }

//              std::vector<int> inn;
//              for (int i=0; i<a_dim; ++i) {
//                  if (ma[i]==mb[i]) {
//                      inn.push_back(i);
//                  }
//              }
//              std::vector<int>::iterator itr;
//              itr = std::set_difference(idx_tmp.begin(),idx_tmp.end(),inn.begin(),inn.end(),idx_out.begin());
//          }
//          int out_dim = maxes_out.size();

//          std::array<int,a_dim> products_a; products_a[0]=0;
//          for (int j=a_dim-1; j>0; --j) {
//              int num = maxes_a[a_dim-1];
//              for (int k=0; k<j-1; ++k) {
//                  num *= maxes_a[a_dim-1-k-1];
//              }
//              products_a[j] = num;
//          }
//          std::array<int,b_dim> products_b; products_b[0]=0;
//          for (int j=b_dim-1; j>0; --j) {
//              int num = maxes_b[b_dim-1];
//              for (int k=0; k<j-1; ++k) {
//                  num *= maxes_b[b_dim-1-k-1];
//              }
//              products_b[j] = num;
//          }

//          std::vector<int> products_out(idx_out.size()); products_out[0]=0;
//          for (int j=idx_out.size()-1; j>0; --j) {
//              int num = maxes_out[idx_out[idx_out.size()-1]];
//              for (int k=0; k<j-1; ++k) {
//                  num *= maxes_out[idx_out[idx_out.size()-1-k-1]];
//              }
//              products_out[j] = num;
//          }

//          std::reverse(products_a.begin(),products_a.end());
//          std::reverse(products_b.begin(),products_b.end());
//          std::reverse(products_out.begin(),products_out.end());

//          int as[out_dim];
//          std::fill(as,as+out_dim,0);
//          int it;


//          int total = 1;
//          {
//              for (int i=0; i<maxes_out.size(); ++i) {
//                  total *=maxes_out[i];
//              }
//          }

//          constexpr size_t unroller = contract_vec_impl<OutTensor>::value;

//          if (unroller % 2 == 0 && unroller % 4 != 0) {
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
//          else if (unroller % 2 == 0 && unroller % 4 == 0) {
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

//                    out_data[index_out] += a_data[index_a]*b_data[index_b];

//              }
//          }

//  //        print(out);
//          return out;
//      }

};

template<class Index_I, class Index_J,
         typename T, size_t ... Rest0, size_t ... Rest1>
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b)
-> decltype(extractor_contract_2<Index_I,Index_J>::contract_impl(a,b)) {
    return extractor_contract_2<Index_I,Index_J>::contract_impl(a,b);
}



/////////////////////////





//template<size_t ... rest>
//struct extract_tensor {

//};


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
        // first two tensors contracted first
        using loop_type_0 = apply_typelist_t<quote_c<size_t, Index>,
                    uniq_t<typelist_c<size_t, Rest0...,Rest1...>>>;
        using contract_0 = typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
                                  typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type;
        using loop_type_1 = typename find_loop_type<Index<Rest2...>,contract_0>::type;
        constexpr size_t flop_count_0 = prod_index<loop_type_0>::value + prod_index<loop_type_1>::value;
        // second and third tensors contracted first
        using loop_type_2 = apply_typelist_t<quote_c<size_t, Index>,
                    uniq_t<typelist_c<size_t, Rest1...,Rest2...>>>;
        using contract_1 = typename contraction_impl<Index<Idx1...,Idx2...>, Tensor<T,Rest1...,Rest2...>,
                                  typename std_ext::make_index_sequence<sizeof...(Rest1)+sizeof...(Rest2)>::type>::type;
        using loop_type_3 = typename find_loop_type<Index<Rest0...>,contract_1>::type;
        constexpr size_t flop_count_1 = prod_index<loop_type_2>::value + prod_index<loop_type_3>::value;
        // first and third tensors contracted first
        using loop_type_4 = apply_typelist_t<quote_c<size_t, Index>,
                    uniq_t<typelist_c<size_t, Rest0...,Rest2...>>>;
        using contract_2 = typename contraction_impl<Index<Idx0...,Idx2...>, Tensor<T,Rest0...,Rest2...>,
                                  typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest2)>::type>::type;
        using loop_type_5 = typename find_loop_type<Index<Rest1...>,contract_2>::type;
        constexpr size_t flop_count_2 = prod_index<loop_type_4>::value + prod_index<loop_type_5>::value;
        // all tensors contracted at once
        using loop_type_6 = apply_typelist_t<quote_c<size_t, Index>,
                    uniq_t<typelist_c<size_t, Rest1...,Rest1...,Rest2...>>>;
        constexpr size_t flop_count_3 = prod_index<loop_type_6>::value;

//        print(flop_count_0,flop_count_1,flop_count_2,flop_count_3);
//        print(meta_min<flop_count_0,flop_count_1,flop_count_2,flop_count_3>::value);
//        print(meta_argmin<flop_count_0,flop_count_1,flop_count_2,flop_count_3>::value);

//        print(meta_argmin<40,30,20,6>::value);

        constexpr int which_variant = meta_argmin<flop_count_0,flop_count_1,flop_count_2,flop_count_3>::value;
//        print(which_variant);

        if (which_variant == 0) {
//            return contraction<Index<>>(contraction(a,b),c);
//            return
            contraction<Index<Idx0...>,Index<Idx1...>>(a,b);
        }
        else if (which_variant == 1) {
//            return contraction(a,contraction(b,c));
        }
        else if (which_variant == 2) {
//            return contraction(contraction(a,c),b);
        }
        else {
            // actual implementation goes here
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

