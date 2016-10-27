#ifndef CONTRACTION_NO_OPT_H
#define CONTRACTION_NO_OPT_H

#include "tensor/Tensor.h"
#include "indicial.h"
#include "contraction.h"


namespace Fastor {


// Three tensor singleton
//------------------------------------------------------------------------------------------------------------//
template<class T, class U, class V, int Optimise>
struct extractor_contract_3_no_opt {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2>
struct extractor_contract_3_no_opt<Index<Idx0...>, Index<Idx1...>, Index<Idx2...>,NoDepthFirst> {


      template<typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2>
        static
        typename contraction_impl<Index<Idx0...,Idx1...,Idx2...>, Tensor<T,Rest0...,Rest1...,Rest2...>,
                 typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+sizeof...(Rest2)>::type>::type
        contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c) {

          using OutTensor = typename contraction_impl<Index<Idx0...,Idx1...,Idx2...>, Tensor<T,Rest0...,Rest1...,Rest2...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+sizeof...(Rest2)>::type>::type;
          using OutIndice = typename contraction_impl<Index<Idx0...,Idx1...,Idx2...>, Tensor<T,Rest0...,Rest1...,Rest2...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+sizeof...(Rest2)>::type>::indices;

          OutTensor out;
          out.zeros();
          const T *a_data = a.data();
          const T *b_data = b.data();
          const T *c_data = c.data();
          T *out_data = out.data();

          constexpr int a_dim = sizeof...(Rest0);
          constexpr int b_dim = sizeof...(Rest1);
          constexpr int c_dim = sizeof...(Rest2);
          constexpr int out_dim =  no_of_unique<Idx0...,Idx1...,Idx2...>::value;

          constexpr auto& idx_a = IndexTensors<
                  Index<Idx0..., Idx1...,Idx2...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...>,
                  Index<Idx0...>,Tensor<T,Rest0...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::indices;

          constexpr auto& idx_b = IndexTensors<
                  Index<Idx0..., Idx1...,Idx2...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...>,
                  Index<Idx1...>,Tensor<T,Rest1...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::indices;

          constexpr auto& idx_c = IndexTensors<
                  Index<Idx0..., Idx1...,Idx2...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...>,
                  Index<Idx2...>,Tensor<T,Rest2...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::indices;

          constexpr auto& idx_out = IndexTensors<
                  Index<Idx0..., Idx1...,Idx2...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...>,
                  OutIndice,OutTensor,
                  typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::indices;

//          print(idx_a,idx_b,idx_c,idx_out);

          using nloops = loop_setter<
                    Index<Idx0...,Idx1...,Idx2...>,
                    Tensor<T,Rest0...,Rest1...,Rest2...>,
                    typename std_ext::make_index_sequence<out_dim>::type>;
          constexpr auto& maxes_out = nloops::dims;
          constexpr int total = nloops::value;
//            print(maxes_out);


          constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
          constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
          constexpr std::array<size_t,c_dim> products_c = nprods<Index<Rest2...>,typename std_ext::make_index_sequence<c_dim>::type>::values;

          using Index_with_dims = typename put_dims_in_Index<OutTensor>::type;
          constexpr std::array<size_t,OutTensor::Dimension> products_out = \
                  nprods<Index_with_dims,typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::values;

//          print(type_name<OutTensor>());
//          print(type_name<OutIndice>());
//          print(products_b,products_out);

          using vectorisability = is_vectorisable<
                    Index<Idx0...,Idx1...>,
                    Index<Idx2...>,Tensor<T,Rest2...>>;
          constexpr int stride = vectorisability::stride;
          using V = typename vectorisability::type;

          int as[out_dim];
          std::fill(as,as+out_dim,0);

          int it;
          V _vec_a, _vec_b;

          for (int i = 0; i < total; i+=stride) {
              int remaining = total;
              for (int n = 0; n < out_dim; ++n) {
                  remaining /= maxes_out[n];
                  as[n] = ( i / remaining ) % maxes_out[n];
//                  std::cout << as[n] << " ";
              }
//              print();

              int index_a = as[idx_a[a_dim-1]];
              for(it = 0; it< a_dim; it++) {
                  index_a += products_a[it]*as[idx_a[it]];
              }
              int index_b = as[idx_b[b_dim-1]];
              for(it = 0; it< b_dim; it++) {
                  index_b += products_b[it]*as[idx_b[it]];
              }
              int index_c = as[idx_c[c_dim-1]];
              for(it = 0; it< c_dim; it++) {
                  index_c += products_c[it]*as[idx_c[it]];
              }
              int index_out = as[idx_out[OutTensor::Dimension-1]];
              for(it = 0; it< OutTensor::Dimension; it++) {
                  index_out += products_out[it]*as[idx_out[it]];
              }
//                std::cout << index_a << " " << index_b << " " << index_c << " " << index_out << "\n";
              _vec_a.set(*(a_data+index_a));
              _vec_b.set(*(b_data+index_b));
              V _vec_out = _vec_a*_vec_b*V(c_data+index_c) +  V(out_data+index_out);
              _vec_out.store(out_data+index_out);
          }

          return out;
      }
};




template<class Index_I, class Index_J, class Index_K, int Optimise=NoDepthFirst,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2>
auto contraction_(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c)
-> decltype(extractor_contract_3_no_opt<Index_I,Index_J,Index_K,NoDepthFirst>::contract_impl(a,b,c)) {
    return extractor_contract_3_no_opt<Index_I,Index_J,Index_K,NoDepthFirst>::contract_impl(a,b,c);
}
//---------------------------------------------------------------------------------------------------------------------//



// Four tensor singleton
//------------------------------------------------------------------------------------------------------------//
template<class T, class U, class V, class W, int Optimise>
struct extractor_contract_4_no_opt {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2, size_t ... Idx3>
struct extractor_contract_4_no_opt<Index<Idx0...>, Index<Idx1...>, Index<Idx2...>, Index<Idx3...>,NoDepthFirst> {


      template<typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3>
        static
        typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...>, Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...>,
                 typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+sizeof...(Rest2)+sizeof...(Rest3)>::type>::type
        contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d) {


          using OutTensor = typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...>, Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+sizeof...(Rest2)+sizeof...(Rest3)>::type>::type;
          using OutIndice = typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...>, Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+sizeof...(Rest2)+sizeof...(Rest3)>::type>::indices;

          OutTensor out;
          out.zeros();
          const T *a_data = a.data();
          const T *b_data = b.data();
          const T *c_data = c.data();
          const T *d_data = d.data();
          T *out_data = out.data();

          constexpr int a_dim = sizeof...(Rest0);
          constexpr int b_dim = sizeof...(Rest1);
          constexpr int c_dim = sizeof...(Rest2);
          constexpr int d_dim = sizeof...(Rest3);
          constexpr int out_dim =  no_of_unique<Idx0...,Idx1...,Idx2...,Idx3...>::value;

          constexpr auto& idx_a = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...>,
                  Index<Idx0...>,Tensor<T,Rest0...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::indices;

          constexpr auto& idx_b = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...>,
                  Index<Idx1...>,Tensor<T,Rest1...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::indices;

          constexpr auto& idx_c = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...>,
                  Index<Idx2...>,Tensor<T,Rest2...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::indices;

          constexpr auto& idx_d = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...>,
                  Index<Idx3...>,Tensor<T,Rest3...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest3)>::type>::indices;

          constexpr auto& idx_out = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...>,
                  OutIndice,OutTensor,
                  typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::indices;

          using nloops = loop_setter<
                    Index<Idx0...,Idx1...,Idx2...,Idx3...>,
                    Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...>,
                    typename std_ext::make_index_sequence<out_dim>::type>;
          constexpr auto& maxes_out = nloops::dims;
          constexpr int total = nloops::value;


          constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
          constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
          constexpr std::array<size_t,c_dim> products_c = nprods<Index<Rest2...>,typename std_ext::make_index_sequence<c_dim>::type>::values;
          constexpr std::array<size_t,d_dim> products_d = nprods<Index<Rest3...>,typename std_ext::make_index_sequence<d_dim>::type>::values;

          using Index_with_dims = typename put_dims_in_Index<OutTensor>::type;
          constexpr std::array<size_t,OutTensor::Dimension> products_out = \
                  nprods<Index_with_dims,typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::values;

          using vectorisability = is_vectorisable<
                    Index<Idx0...,Idx1...,Idx2...>,
                    Index<Idx3...>,Tensor<T,Rest3...>>;
          constexpr int stride = vectorisability::stride;
          using V = typename vectorisability::type;

          int as[out_dim];
          std::fill(as,as+out_dim,0);

          int it;
          V _vec_a, _vec_b, _vec_c;

          for (int i = 0; i < total; i+=stride) {
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
              int index_c = as[idx_c[c_dim-1]];
              for(it = 0; it< c_dim; it++) {
                  index_c += products_c[it]*as[idx_c[it]];
              }
              int index_d = as[idx_d[d_dim-1]];
              for(it = 0; it< d_dim; it++) {
                  index_d += products_d[it]*as[idx_d[it]];
              }
              int index_out = as[idx_out[OutTensor::Dimension-1]];
              for(it = 0; it< OutTensor::Dimension; it++) {
                  index_out += products_out[it]*as[idx_out[it]];
              }
              _vec_a.set(*(a_data+index_a));
              _vec_b.set(*(b_data+index_b));
              _vec_c.set(*(c_data+index_c));
              V _vec_out = _vec_a*_vec_b*_vec_c*V(d_data+index_d) +  V(out_data+index_out);
              _vec_out.store(out_data+index_out);
          }

          return out;
      }
};




template<class Index_I, class Index_J, class Index_K, class Index_L, int Optimise=NoDepthFirst,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3>
auto contraction_(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d)
-> decltype(extractor_contract_4_no_opt<Index_I,Index_J,Index_K,Index_L,NoDepthFirst>::contract_impl(a,b,c,d)) {
    return extractor_contract_4_no_opt<Index_I,Index_J,Index_K,Index_L,NoDepthFirst>::contract_impl(a,b,c,d);
}
//---------------------------------------------------------------------------------------------------------------------//











// Five tensor singleton
//------------------------------------------------------------------------------------------------------------//
template<class T, class U, class V, class W, class X, int Optimise>
struct extractor_contract_5_no_opt {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2, size_t ... Idx3, size_t ... Idx4>
struct extractor_contract_5_no_opt<Index<Idx0...>, Index<Idx1...>, Index<Idx2...>, Index<Idx3...>, Index<Idx4...>,NoDepthFirst> {


      template<typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4>
        static
        typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...>, Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...>,
                 typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)>::type>::type
        contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                      const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                      const Tensor<T,Rest4...> &e) {


          using OutTensor = typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)>::type>::type;
          using OutIndice = typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...>, Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)>::type>::indices;

          OutTensor out;
          out.zeros();
          const T *a_data = a.data();
          const T *b_data = b.data();
          const T *c_data = c.data();
          const T *d_data = d.data();
          const T *e_data = e.data();
          T *out_data = out.data();

          constexpr int a_dim = sizeof...(Rest0);
          constexpr int b_dim = sizeof...(Rest1);
          constexpr int c_dim = sizeof...(Rest2);
          constexpr int d_dim = sizeof...(Rest3);
          constexpr int e_dim = sizeof...(Rest4);
          constexpr int out_dim =  no_of_unique<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...>::value;

          constexpr auto& idx_a = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...>,
                  Index<Idx0...>,Tensor<T,Rest0...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::indices;

          constexpr auto& idx_b = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...>,
                  Index<Idx1...>,Tensor<T,Rest1...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::indices;

          constexpr auto& idx_c = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...>,
                  Index<Idx2...>,Tensor<T,Rest2...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::indices;

          constexpr auto& idx_d = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...>,
                  Index<Idx3...>,Tensor<T,Rest3...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest3)>::type>::indices;

          constexpr auto& idx_e = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...>,
                  Index<Idx4...>,Tensor<T,Rest4...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest4)>::type>::indices;

          constexpr auto& idx_out = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...>,
                  OutIndice,OutTensor,
                  typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::indices;

          using nloops = loop_setter<
                    Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...>,
                    Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...>,
                    typename std_ext::make_index_sequence<out_dim>::type>;
          constexpr auto& maxes_out = nloops::dims;
          constexpr int total = nloops::value;


          constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
          constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
          constexpr std::array<size_t,c_dim> products_c = nprods<Index<Rest2...>,typename std_ext::make_index_sequence<c_dim>::type>::values;
          constexpr std::array<size_t,d_dim> products_d = nprods<Index<Rest3...>,typename std_ext::make_index_sequence<d_dim>::type>::values;
          constexpr std::array<size_t,e_dim> products_e = nprods<Index<Rest4...>,typename std_ext::make_index_sequence<e_dim>::type>::values;

          using Index_with_dims = typename put_dims_in_Index<OutTensor>::type;
          constexpr std::array<size_t,OutTensor::Dimension> products_out = \
                  nprods<Index_with_dims,typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::values;

          using vectorisability = is_vectorisable<
                    Index<Idx0...,Idx1...,Idx2...,Idx3...>,
                    Index<Idx4...>,Tensor<T,Rest4...>>;
          constexpr int stride = vectorisability::stride;
          using V = typename vectorisability::type;

//          // for benchmarks
//          constexpr int stride = 1;
//          using V = SIMDVector<T,64>;

          int as[out_dim];
          std::fill(as,as+out_dim,0);

          int it;
          V _vec_a, _vec_b, _vec_c, _vec_d;

          for (int i = 0; i < total; i+=stride) {
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
              int index_c = as[idx_c[c_dim-1]];
              for(it = 0; it< c_dim; it++) {
                  index_c += products_c[it]*as[idx_c[it]];
              }
              int index_d = as[idx_d[d_dim-1]];
              for(it = 0; it< d_dim; it++) {
                  index_d += products_d[it]*as[idx_d[it]];
              }
              int index_e = as[idx_e[e_dim-1]];
              for(it = 0; it< e_dim; it++) {
                  index_d += products_e[it]*as[idx_e[it]];
              }
              int index_out = as[idx_out[OutTensor::Dimension-1]];
              for(it = 0; it< OutTensor::Dimension; it++) {
                  index_out += products_out[it]*as[idx_out[it]];
              }
              _vec_a.set(*(a_data+index_a));
              _vec_b.set(*(b_data+index_b));
              _vec_c.set(*(c_data+index_c));
              _vec_d.set(*(d_data+index_d));
              V _vec_out = _vec_a*_vec_b*_vec_c*_vec_d*V(e_data+index_e) +  V(out_data+index_out);
              _vec_out.store(out_data+index_out);
          }

          return out;
      }
};




template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, int Optimise=NoDepthFirst,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4>
auto contraction_(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d, const Tensor<T,Rest4...> &e)
-> decltype(extractor_contract_5_no_opt<Index_I,Index_J,Index_K,Index_L,Index_M,NoDepthFirst>::contract_impl(a,b,c,d,e)) {
    return extractor_contract_5_no_opt<Index_I,Index_J,Index_K,Index_L,Index_M,NoDepthFirst>::contract_impl(a,b,c,d,e);
}
//---------------------------------------------------------------------------------------------------------------------//



}


#endif // CONTRACTION_NO_OPT_H

