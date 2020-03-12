#ifndef CONTRACTION_NO_OPT_H
#define CONTRACTION_NO_OPT_H


#ifdef FASTOR_DONT_PERFORM_OP_MIN

#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor_algebra/indicial.h"
#include "Fastor/tensor_algebra/contraction.h"


namespace Fastor {


// Three tensor singleton
//------------------------------------------------------------------------------------------------------------//
template<class T, class U, class V>
struct extractor_contract_3_no_opt {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2>
struct extractor_contract_3_no_opt<Index<Idx0...>, Index<Idx1...>, Index<Idx2...>> {


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


          using nloops = loop_setter<
                    Index<Idx0...,Idx1...,Idx2...>,
                    Tensor<T,Rest0...,Rest1...,Rest2...>,
                    typename std_ext::make_index_sequence<out_dim>::type>;
          constexpr auto& maxes_out = nloops::dims;
          constexpr int total = nloops::value;

#ifdef FASTOR_PRINT_COST
        print(total);
#endif

          constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
          constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
          constexpr std::array<size_t,c_dim> products_c = nprods<Index<Rest2...>,typename std_ext::make_index_sequence<c_dim>::type>::values;

          using Index_with_dims = typename put_dims_in_Index<OutTensor>::type;
          constexpr std::array<size_t,OutTensor::Dimension> products_out = \
                  nprods<Index_with_dims,typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::values;

#ifndef FASTOR_DONT_VECTORISE
          using vectorisability = is_vectorisable<
                    Index<Idx0...,Idx1...>,
                    Index<Idx2...>,Tensor<T,Rest2...>>;
          constexpr int stride = vectorisability::stride;
          using V = typename vectorisability::type;
#else
          constexpr int stride = 1;
          using V = SIMDVector<T,sizeof(T)*8>;
#endif

          int as[out_dim];
          std::fill(as,as+out_dim,0);

          int it;
          V _vec_a, _vec_b;

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
              int index_out = as[idx_out[OutTensor::Dimension-1]];
              for(it = 0; it< OutTensor::Dimension; it++) {
                  index_out += products_out[it]*as[idx_out[it]];
              }

              _vec_a.set(*(a_data+index_a));
              _vec_b.set(*(b_data+index_b));
              V _vec_out = _vec_a*_vec_b*V(c_data+index_c) +  V(out_data+index_out);
              _vec_out.store(out_data+index_out);
          }

          return out;
      }
};



template<class Index_I, class Index_J, class Index_K,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2>
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c)
-> decltype(extractor_contract_3_no_opt<Index_I,Index_J,Index_K>::contract_impl(a,b,c)) {
    return extractor_contract_3_no_opt<Index_I,Index_J,Index_K>::contract_impl(a,b,c);
}
//---------------------------------------------------------------------------------------------------------------------//



// Four tensor singleton
//------------------------------------------------------------------------------------------------------------//
template<class T, class U, class V, class W>
struct extractor_contract_4_no_opt {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2, size_t ... Idx3>
struct extractor_contract_4_no_opt<Index<Idx0...>, Index<Idx1...>, Index<Idx2...>, Index<Idx3...>> {


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

#ifdef FASTOR_PRINT_COST
        print(total);
#endif

          constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
          constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
          constexpr std::array<size_t,c_dim> products_c = nprods<Index<Rest2...>,typename std_ext::make_index_sequence<c_dim>::type>::values;
          constexpr std::array<size_t,d_dim> products_d = nprods<Index<Rest3...>,typename std_ext::make_index_sequence<d_dim>::type>::values;

          using Index_with_dims = typename put_dims_in_Index<OutTensor>::type;
          constexpr std::array<size_t,OutTensor::Dimension> products_out = \
                  nprods<Index_with_dims,typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::values;

#ifndef FASTOR_DONT_VECTORISE
          using vectorisability = is_vectorisable<
                    Index<Idx0...,Idx1...,Idx2...>,
                    Index<Idx3...>,Tensor<T,Rest3...>>;
          constexpr int stride = vectorisability::stride;
          using V = typename vectorisability::type;
#else
          constexpr int stride = 1;
          using V = SIMDVector<T,sizeof(T)*8>;
#endif

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




template<class Index_I, class Index_J, class Index_K, class Index_L,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3>
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d)
-> decltype(extractor_contract_4_no_opt<Index_I,Index_J,Index_K,Index_L>::contract_impl(a,b,c,d)) {
    return extractor_contract_4_no_opt<Index_I,Index_J,Index_K,Index_L>::contract_impl(a,b,c,d);
}
//---------------------------------------------------------------------------------------------------------------------//











// Five tensor singleton
//------------------------------------------------------------------------------------------------------------//
template<class T, class U, class V, class W, class X>
struct extractor_contract_5_no_opt {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2, size_t ... Idx3, size_t ... Idx4>
struct extractor_contract_5_no_opt<Index<Idx0...>, Index<Idx1...>, Index<Idx2...>, Index<Idx3...>, Index<Idx4...>> {


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

#ifdef FASTOR_PRINT_COST
        print(total);
#endif

          constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
          constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
          constexpr std::array<size_t,c_dim> products_c = nprods<Index<Rest2...>,typename std_ext::make_index_sequence<c_dim>::type>::values;
          constexpr std::array<size_t,d_dim> products_d = nprods<Index<Rest3...>,typename std_ext::make_index_sequence<d_dim>::type>::values;
          constexpr std::array<size_t,e_dim> products_e = nprods<Index<Rest4...>,typename std_ext::make_index_sequence<e_dim>::type>::values;

          using Index_with_dims = typename put_dims_in_Index<OutTensor>::type;
          constexpr std::array<size_t,OutTensor::Dimension> products_out = \
                  nprods<Index_with_dims,typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::values;

#ifndef FASTOR_DONT_VECTORISE
          using vectorisability = is_vectorisable<
                    Index<Idx0...,Idx1...,Idx2...,Idx3...>,
                    Index<Idx4...>,Tensor<T,Rest4...>>;
          constexpr int stride = vectorisability::stride;
          using V = typename vectorisability::type;
#else
          constexpr int stride = 1;
          using V = SIMDVector<T,sizeof(T)*8>;
#endif

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
                  index_e += products_e[it]*as[idx_e[it]];
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




template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4>
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d, const Tensor<T,Rest4...> &e)
-> decltype(extractor_contract_5_no_opt<Index_I,Index_J,Index_K,Index_L,Index_M>::contract_impl(a,b,c,d,e)) {
    return extractor_contract_5_no_opt<Index_I,Index_J,Index_K,Index_L,Index_M>::contract_impl(a,b,c,d,e);
}
//---------------------------------------------------------------------------------------------------------------------//







// Six tensor singleton
//------------------------------------------------------------------------------------------------------------//
template<class T, class U, class V, class W, class X, class Y>
struct extractor_contract_6_no_opt {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2, size_t ... Idx3, size_t ... Idx4, size_t ... Idx5>
struct extractor_contract_6_no_opt<Index<Idx0...>, Index<Idx1...>,
        Index<Idx2...>, Index<Idx3...>, Index<Idx4...>, Index<Idx5...>> {


      template<typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4, size_t ... Rest5>
        static
        typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...>,
                 typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                    sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)>::type>::type
        contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                      const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                      const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f) {


          using OutTensor = typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)>::type>::type;
          using OutIndice = typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)>::type>::indices;

          OutTensor out;
          out.zeros();
          const T *a_data = a.data();
          const T *b_data = b.data();
          const T *c_data = c.data();
          const T *d_data = d.data();
          const T *e_data = e.data();
          const T *f_data = f.data();
          T *out_data = out.data();

          constexpr int a_dim = sizeof...(Rest0);
          constexpr int b_dim = sizeof...(Rest1);
          constexpr int c_dim = sizeof...(Rest2);
          constexpr int d_dim = sizeof...(Rest3);
          constexpr int e_dim = sizeof...(Rest4);
          constexpr int f_dim = sizeof...(Rest5);
          constexpr int out_dim =  no_of_unique<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...>::value;

          constexpr auto& idx_a = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...>,
                  Index<Idx0...>,Tensor<T,Rest0...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::indices;

          constexpr auto& idx_b = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...>,
                  Index<Idx1...>,Tensor<T,Rest1...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::indices;

          constexpr auto& idx_c = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...>,
                  Index<Idx2...>,Tensor<T,Rest2...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::indices;

          constexpr auto& idx_d = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...>,
                  Index<Idx3...>,Tensor<T,Rest3...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest3)>::type>::indices;

          constexpr auto& idx_e = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...>,
                  Index<Idx4...>,Tensor<T,Rest4...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest4)>::type>::indices;

          constexpr auto& idx_f = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...>,
                  Index<Idx5...>,Tensor<T,Rest5...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest5)>::type>::indices;

          constexpr auto& idx_out = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...>,
                  OutIndice,OutTensor,
                  typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::indices;

          using nloops = loop_setter<
              Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...>,
              Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...>,
                    typename std_ext::make_index_sequence<out_dim>::type>;
          constexpr auto& maxes_out = nloops::dims;
          constexpr int total = nloops::value;

#ifdef FASTOR_PRINT_COST
        print(total);
#endif

          constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
          constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
          constexpr std::array<size_t,c_dim> products_c = nprods<Index<Rest2...>,typename std_ext::make_index_sequence<c_dim>::type>::values;
          constexpr std::array<size_t,d_dim> products_d = nprods<Index<Rest3...>,typename std_ext::make_index_sequence<d_dim>::type>::values;
          constexpr std::array<size_t,e_dim> products_e = nprods<Index<Rest4...>,typename std_ext::make_index_sequence<e_dim>::type>::values;
          constexpr std::array<size_t,f_dim> products_f = nprods<Index<Rest5...>,typename std_ext::make_index_sequence<f_dim>::type>::values;

          using Index_with_dims = typename put_dims_in_Index<OutTensor>::type;
          constexpr std::array<size_t,OutTensor::Dimension> products_out = \
                  nprods<Index_with_dims,typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::values;

#ifndef FASTOR_DONT_VECTORISE
          using vectorisability = is_vectorisable<
                    Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...>,
                    Index<Idx5...>,Tensor<T,Rest5...>>;
          constexpr int stride = vectorisability::stride;
          using V = typename vectorisability::type;
#else
          constexpr int stride = 1;
          using V = SIMDVector<T,sizeof(T)*8>;
#endif

          int as[out_dim];
          std::fill(as,as+out_dim,0);

          int it;
          V _vec_a, _vec_b, _vec_c, _vec_d, _vec_e;

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
                  index_e += products_e[it]*as[idx_e[it]];
              }
              int index_f = as[idx_f[f_dim-1]];
              for(it = 0; it< f_dim; it++) {
                  index_f += products_f[it]*as[idx_f[it]];
              }
              int index_out = as[idx_out[OutTensor::Dimension-1]];
              for(it = 0; it< OutTensor::Dimension; it++) {
                  index_out += products_out[it]*as[idx_out[it]];
              }
              _vec_a.set(*(a_data+index_a));
              _vec_b.set(*(b_data+index_b));
              _vec_c.set(*(c_data+index_c));
              _vec_d.set(*(d_data+index_d));
              _vec_e.set(*(e_data+index_e));
              V _vec_out = _vec_a*_vec_b*_vec_c*_vec_d*_vec_e*V(f_data+index_f) +  V(out_data+index_out);
              _vec_out.store(out_data+index_out);
          }

          return out;
      }
};




template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, class Index_N,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4, size_t ... Rest5>
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                 const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f)
-> decltype(extractor_contract_6_no_opt<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>::contract_impl(a,b,c,d,e,f)) {
    return extractor_contract_6_no_opt<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>::contract_impl(a,b,c,d,e,f);
}
//---------------------------------------------------------------------------------------------------------------------//







// Seven tensor singleton
//------------------------------------------------------------------------------------------------------------//
template<class T, class U, class V, class W, class X, class Y, class Z>
struct extractor_contract_7_no_opt {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2, size_t ... Idx3, size_t ... Idx4, size_t ... Idx5, size_t ... Idx6>
struct extractor_contract_7_no_opt<Index<Idx0...>, Index<Idx1...>,
        Index<Idx2...>, Index<Idx3...>, Index<Idx4...>, Index<Idx5...>, Index<Idx6...>> {


      template<typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2,
               size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6>
        static
        typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...>,
                 typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                    sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)+sizeof...(Rest6)>::type>::type
        contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                      const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                      const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
                      const Tensor<T,Rest6...> &g) {


          using OutTensor = typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)+sizeof...(Rest6)>::type>::type;
          using OutIndice = typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)+sizeof...(Rest6)>::type>::indices;

          OutTensor out;
          out.zeros();
          const T *a_data = a.data();
          const T *b_data = b.data();
          const T *c_data = c.data();
          const T *d_data = d.data();
          const T *e_data = e.data();
          const T *f_data = f.data();
          const T *g_data = g.data();
          T *out_data = out.data();

          constexpr int a_dim = sizeof...(Rest0);
          constexpr int b_dim = sizeof...(Rest1);
          constexpr int c_dim = sizeof...(Rest2);
          constexpr int d_dim = sizeof...(Rest3);
          constexpr int e_dim = sizeof...(Rest4);
          constexpr int f_dim = sizeof...(Rest5);
          constexpr int g_dim = sizeof...(Rest6);
          constexpr int out_dim =  no_of_unique<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...>::value;

          constexpr auto& idx_a = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...>,
                  Index<Idx0...>,Tensor<T,Rest0...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::indices;

          constexpr auto& idx_b = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...>,
                  Index<Idx1...>,Tensor<T,Rest1...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::indices;

          constexpr auto& idx_c = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...>,
                  Index<Idx2...>,Tensor<T,Rest2...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::indices;

          constexpr auto& idx_d = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...>,
                  Index<Idx3...>,Tensor<T,Rest3...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest3)>::type>::indices;

          constexpr auto& idx_e = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...>,
                  Index<Idx4...>,Tensor<T,Rest4...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest4)>::type>::indices;

          constexpr auto& idx_f = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...>,
                  Index<Idx5...>,Tensor<T,Rest5...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest5)>::type>::indices;

          constexpr auto& idx_g = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...>,
                  Index<Idx6...>,Tensor<T,Rest6...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest6)>::type>::indices;

          constexpr auto& idx_out = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...>,
                  OutIndice,OutTensor,
                  typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::indices;

          using nloops = loop_setter<
              Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...>,
              Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...>,
                    typename std_ext::make_index_sequence<out_dim>::type>;
          constexpr auto& maxes_out = nloops::dims;
          constexpr int total = nloops::value;

#ifdef FASTOR_PRINT_COST
        print(total);
#endif

          constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
          constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
          constexpr std::array<size_t,c_dim> products_c = nprods<Index<Rest2...>,typename std_ext::make_index_sequence<c_dim>::type>::values;
          constexpr std::array<size_t,d_dim> products_d = nprods<Index<Rest3...>,typename std_ext::make_index_sequence<d_dim>::type>::values;
          constexpr std::array<size_t,e_dim> products_e = nprods<Index<Rest4...>,typename std_ext::make_index_sequence<e_dim>::type>::values;
          constexpr std::array<size_t,f_dim> products_f = nprods<Index<Rest5...>,typename std_ext::make_index_sequence<f_dim>::type>::values;
          constexpr std::array<size_t,g_dim> products_g = nprods<Index<Rest6...>,typename std_ext::make_index_sequence<g_dim>::type>::values;

          using Index_with_dims = typename put_dims_in_Index<OutTensor>::type;
          constexpr std::array<size_t,OutTensor::Dimension> products_out = \
                  nprods<Index_with_dims,typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::values;

#ifndef FASTOR_DONT_VECTORISE
          using vectorisability = is_vectorisable<
                    Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...>,
                    Index<Idx6...>,Tensor<T,Rest6...>>;
          constexpr int stride = vectorisability::stride;
          using V = typename vectorisability::type;
#else
          constexpr int stride = 1;
          using V = SIMDVector<T,sizeof(T)*8>;
#endif

          int as[out_dim];
          std::fill(as,as+out_dim,0);

          int it;
          V _vec_a, _vec_b, _vec_c, _vec_d, _vec_e, _vec_f;

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
                  index_e += products_e[it]*as[idx_e[it]];
              }
              int index_f = as[idx_f[f_dim-1]];
              for(it = 0; it< f_dim; it++) {
                  index_f += products_f[it]*as[idx_f[it]];
              }
              int index_g = as[idx_g[g_dim-1]];
              for(it = 0; it< g_dim; it++) {
                  index_g += products_g[it]*as[idx_g[it]];
              }
              int index_out = as[idx_out[OutTensor::Dimension-1]];
              for(it = 0; it< OutTensor::Dimension; it++) {
                  index_out += products_out[it]*as[idx_out[it]];
              }
              _vec_a.set(*(a_data+index_a));
              _vec_b.set(*(b_data+index_b));
              _vec_c.set(*(c_data+index_c));
              _vec_d.set(*(d_data+index_d));
              _vec_e.set(*(e_data+index_e));
              _vec_f.set(*(f_data+index_f));
              V _vec_out = _vec_a*_vec_b*_vec_c*_vec_d*_vec_e*_vec_f*V(g_data+index_g) +  V(out_data+index_out);
              _vec_out.store(out_data+index_out);
          }

          return out;
      }
};




template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, class Index_N, class Index_O,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2,
         size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6>
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                 const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
                 const Tensor<T,Rest6...> &g)
-> decltype(extractor_contract_7_no_opt<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O>::contract_impl(a,b,c,d,e,f,g)) {
    return extractor_contract_7_no_opt<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O>::contract_impl(a,b,c,d,e,f,g);
}
//---------------------------------------------------------------------------------------------------------------------//










// Eight tensor singleton
//------------------------------------------------------------------------------------------------------------//
template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
struct extractor_contract_8_no_opt {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2,
         size_t ... Idx3, size_t ... Idx4, size_t ... Idx5, size_t ... Idx6,
         size_t ... Idx7>
struct extractor_contract_8_no_opt<Index<Idx0...>, Index<Idx1...>,
        Index<Idx2...>, Index<Idx3...>, Index<Idx4...>, Index<Idx5...>, Index<Idx6...>,
        Index<Idx7...>> {


      template<typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2,
               size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6,
               size_t ... Rest7>
        static
        typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...>,
                 typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                    sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)+\
                    sizeof...(Rest6)+sizeof...(Rest7)>::type>::type
        contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                      const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                      const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
                      const Tensor<T,Rest6...> &g, const Tensor<T,Rest7...> &h) {


          using OutTensor = typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,
                Idx5...,Idx6...,Idx7...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)+\
                sizeof...(Rest6)+sizeof...(Rest7)>::type>::type;
          using OutIndice = typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,
                Idx5...,Idx6...,Idx7...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)+\
                sizeof...(Rest6)+sizeof...(Rest7)>::type>::indices;

          OutTensor out;
          out.zeros();
          const T *a_data = a.data();
          const T *b_data = b.data();
          const T *c_data = c.data();
          const T *d_data = d.data();
          const T *e_data = e.data();
          const T *f_data = f.data();
          const T *g_data = g.data();
          const T *h_data = h.data();
          T *out_data = out.data();

          constexpr int a_dim = sizeof...(Rest0);
          constexpr int b_dim = sizeof...(Rest1);
          constexpr int c_dim = sizeof...(Rest2);
          constexpr int d_dim = sizeof...(Rest3);
          constexpr int e_dim = sizeof...(Rest4);
          constexpr int f_dim = sizeof...(Rest5);
          constexpr int g_dim = sizeof...(Rest6);
          constexpr int h_dim = sizeof...(Rest7);
          constexpr int out_dim =  no_of_unique<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,
                  Idx5...,Idx6...,Idx7...>::value;

          constexpr auto& idx_a = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...>,
                  Index<Idx0...>,Tensor<T,Rest0...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::indices;

          constexpr auto& idx_b = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...>,
                  Index<Idx1...>,Tensor<T,Rest1...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::indices;

          constexpr auto& idx_c = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...>,
                  Index<Idx2...>,Tensor<T,Rest2...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::indices;

          constexpr auto& idx_d = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...>,
                  Index<Idx3...>,Tensor<T,Rest3...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest3)>::type>::indices;

          constexpr auto& idx_e = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...>,
                  Index<Idx4...>,Tensor<T,Rest4...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest4)>::type>::indices;

          constexpr auto& idx_f = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...>,
                  Index<Idx5...>,Tensor<T,Rest5...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest5)>::type>::indices;

          constexpr auto& idx_g = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...>,
                  Index<Idx6...>,Tensor<T,Rest6...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest6)>::type>::indices;

          constexpr auto& idx_h = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...>,
                  Index<Idx7...>,Tensor<T,Rest7...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest7)>::type>::indices;

          constexpr auto& idx_out = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...>,
                  OutIndice,OutTensor,
                  typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::indices;

          using nloops = loop_setter<
            Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...>,
                    typename std_ext::make_index_sequence<out_dim>::type>;
          constexpr auto& maxes_out = nloops::dims;
          constexpr int total = nloops::value;

#ifdef FASTOR_PRINT_COST
        print(total);
#endif

          constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
          constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
          constexpr std::array<size_t,c_dim> products_c = nprods<Index<Rest2...>,typename std_ext::make_index_sequence<c_dim>::type>::values;
          constexpr std::array<size_t,d_dim> products_d = nprods<Index<Rest3...>,typename std_ext::make_index_sequence<d_dim>::type>::values;
          constexpr std::array<size_t,e_dim> products_e = nprods<Index<Rest4...>,typename std_ext::make_index_sequence<e_dim>::type>::values;
          constexpr std::array<size_t,f_dim> products_f = nprods<Index<Rest5...>,typename std_ext::make_index_sequence<f_dim>::type>::values;
          constexpr std::array<size_t,g_dim> products_g = nprods<Index<Rest6...>,typename std_ext::make_index_sequence<g_dim>::type>::values;
          constexpr std::array<size_t,h_dim> products_h = nprods<Index<Rest7...>,typename std_ext::make_index_sequence<h_dim>::type>::values;

          using Index_with_dims = typename put_dims_in_Index<OutTensor>::type;
          constexpr std::array<size_t,OutTensor::Dimension> products_out = \
                  nprods<Index_with_dims,typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::values;

#ifndef FASTOR_DONT_VECTORISE
          using vectorisability = is_vectorisable<
                    Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...>,
                    Index<Idx7...>,Tensor<T,Rest7...>>;
          constexpr int stride = vectorisability::stride;
          using V = typename vectorisability::type;
#else
          constexpr int stride = 1;
          using V = SIMDVector<T,sizeof(T)*8>;
#endif

          int as[out_dim];
          std::fill(as,as+out_dim,0);

          int it;
          V _vec_a, _vec_b, _vec_c, _vec_d, _vec_e, _vec_f, _vec_g;

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
                  index_e += products_e[it]*as[idx_e[it]];
              }
              int index_f = as[idx_f[f_dim-1]];
              for(it = 0; it< f_dim; it++) {
                  index_f += products_f[it]*as[idx_f[it]];
              }
              int index_g = as[idx_g[g_dim-1]];
              for(it = 0; it< g_dim; it++) {
                  index_g += products_g[it]*as[idx_g[it]];
              }
              int index_h = as[idx_h[h_dim-1]];
              for(it = 0; it< h_dim; it++) {
                  index_h += products_h[it]*as[idx_h[it]];
              }
              int index_out = as[idx_out[OutTensor::Dimension-1]];
              for(it = 0; it< OutTensor::Dimension; it++) {
                  index_out += products_out[it]*as[idx_out[it]];
              }
              _vec_a.set(*(a_data+index_a));
              _vec_b.set(*(b_data+index_b));
              _vec_c.set(*(c_data+index_c));
              _vec_d.set(*(d_data+index_d));
              _vec_e.set(*(e_data+index_e));
              _vec_f.set(*(f_data+index_f));
              _vec_g.set(*(g_data+index_g));
              V _vec_out = _vec_a*_vec_b*_vec_c*_vec_d*_vec_e*_vec_f*_vec_g*V(h_data+index_h) +  V(out_data+index_out);
              _vec_out.store(out_data+index_out);
          }

          return out;
      }
};




template<class Index_0, class Index_1, class Index_2,
         class Index_3, class Index_4, class Index_5, class Index_6,
         class Index_7,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2,
         size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6,
         size_t ... Rest7>
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                 const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
                 const Tensor<T,Rest6...> &g, const Tensor<T,Rest7...> &h)
-> decltype(extractor_contract_8_no_opt<Index_0,Index_1,Index_2,Index_3,
            Index_4,Index_5,Index_6,Index_7>::contract_impl(a,b,c,d,e,f,g,h)) {
    return extractor_contract_8_no_opt<Index_0,Index_1,Index_2,Index_3,
            Index_4,Index_5,Index_6,Index_7>::contract_impl(a,b,c,d,e,f,g,h);
}
//---------------------------------------------------------------------------------------------------------------------//












// Nine tensor singleton
//------------------------------------------------------------------------------------------------------------//
template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
struct extractor_contract_9_no_opt {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2,
         size_t ... Idx3, size_t ... Idx4, size_t ... Idx5, size_t ... Idx6,
         size_t ... Idx7, size_t ... Idx8>
struct extractor_contract_9_no_opt<Index<Idx0...>, Index<Idx1...>,
        Index<Idx2...>, Index<Idx3...>, Index<Idx4...>, Index<Idx5...>, Index<Idx6...>,
        Index<Idx7...>, Index<Idx8...>> {


      template<typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2,
               size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6,
               size_t ... Rest7, size_t ... Rest8>
        static
        typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...>,
                 typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                    sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)+\
                    sizeof...(Rest6)+sizeof...(Rest7)+sizeof...(Rest8)>::type>::type
        contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                      const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                      const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
                      const Tensor<T,Rest6...> &g, const Tensor<T,Rest7...> &h,
                      const Tensor<T,Rest8...> &h1) {


          using OutTensor = typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,
                Idx5...,Idx6...,Idx7...,Idx8...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)+\
                sizeof...(Rest6)+sizeof...(Rest7)+sizeof...(Rest8)>::type>::type;
          using OutIndice = typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,
                Idx5...,Idx6...,Idx7...,Idx8...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)+\
                sizeof...(Rest6)+sizeof...(Rest7)+sizeof...(Rest8)>::type>::indices;

          OutTensor out;
          out.zeros();
          const T *a_data = a.data();
          const T *b_data = b.data();
          const T *c_data = c.data();
          const T *d_data = d.data();
          const T *e_data = e.data();
          const T *f_data = f.data();
          const T *g_data = g.data();
          const T *h_data = h.data();
          const T *h1_data = h1.data();
          T *out_data = out.data();

          constexpr int a_dim = sizeof...(Rest0);
          constexpr int b_dim = sizeof...(Rest1);
          constexpr int c_dim = sizeof...(Rest2);
          constexpr int d_dim = sizeof...(Rest3);
          constexpr int e_dim = sizeof...(Rest4);
          constexpr int f_dim = sizeof...(Rest5);
          constexpr int g_dim = sizeof...(Rest6);
          constexpr int h_dim = sizeof...(Rest7);
          constexpr int h1_dim = sizeof...(Rest8);
          constexpr int out_dim =  no_of_unique<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,
                  Idx5...,Idx6...,Idx7...,Idx8...>::value;

          constexpr auto& idx_a = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...>,
                  Index<Idx0...>,Tensor<T,Rest0...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::indices;

          constexpr auto& idx_b = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...>,
                  Index<Idx1...>,Tensor<T,Rest1...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::indices;

          constexpr auto& idx_c = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...>,
                  Index<Idx2...>,Tensor<T,Rest2...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::indices;

          constexpr auto& idx_d = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...>,
                  Index<Idx3...>,Tensor<T,Rest3...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest3)>::type>::indices;

          constexpr auto& idx_e = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...>,
                  Index<Idx4...>,Tensor<T,Rest4...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest4)>::type>::indices;

          constexpr auto& idx_f = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...>,
                  Index<Idx5...>,Tensor<T,Rest5...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest5)>::type>::indices;

          constexpr auto& idx_g = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...>,
                  Index<Idx6...>,Tensor<T,Rest6...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest6)>::type>::indices;

          constexpr auto& idx_h = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...>,
                  Index<Idx7...>,Tensor<T,Rest7...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest7)>::type>::indices;

          constexpr auto& idx_h1 = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...>,
                  Index<Idx8...>,Tensor<T,Rest8...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest8)>::type>::indices;

          constexpr auto& idx_out = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...>,
                  OutIndice,OutTensor,
                  typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::indices;

          using nloops = loop_setter<
            Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...>,
                    typename std_ext::make_index_sequence<out_dim>::type>;
          constexpr auto& maxes_out = nloops::dims;
          constexpr int total = nloops::value;

#ifdef FASTOR_PRINT_COST
        print(total);
#endif

          constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
          constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
          constexpr std::array<size_t,c_dim> products_c = nprods<Index<Rest2...>,typename std_ext::make_index_sequence<c_dim>::type>::values;
          constexpr std::array<size_t,d_dim> products_d = nprods<Index<Rest3...>,typename std_ext::make_index_sequence<d_dim>::type>::values;
          constexpr std::array<size_t,e_dim> products_e = nprods<Index<Rest4...>,typename std_ext::make_index_sequence<e_dim>::type>::values;
          constexpr std::array<size_t,f_dim> products_f = nprods<Index<Rest5...>,typename std_ext::make_index_sequence<f_dim>::type>::values;
          constexpr std::array<size_t,g_dim> products_g = nprods<Index<Rest6...>,typename std_ext::make_index_sequence<g_dim>::type>::values;
          constexpr std::array<size_t,h_dim> products_h = nprods<Index<Rest7...>,typename std_ext::make_index_sequence<h_dim>::type>::values;
          constexpr std::array<size_t,h1_dim> products_h1 = nprods<Index<Rest8...>,typename std_ext::make_index_sequence<h1_dim>::type>::values;

          using Index_with_dims = typename put_dims_in_Index<OutTensor>::type;
          constexpr std::array<size_t,OutTensor::Dimension> products_out = \
                  nprods<Index_with_dims,typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::values;

#ifndef FASTOR_DONT_VECTORISE
          using vectorisability = is_vectorisable<
                    Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...>,
                    Index<Idx8...>,Tensor<T,Rest8...>>;
          constexpr int stride = vectorisability::stride;
          using V = typename vectorisability::type;
#else
          constexpr int stride = 1;
          using V = SIMDVector<T,sizeof(T)*8>;
#endif

          int as[out_dim];
          std::fill(as,as+out_dim,0);

          int it;
          V _vec_a, _vec_b, _vec_c, _vec_d, _vec_e, _vec_f, _vec_g, _vec_h;

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
                  index_e += products_e[it]*as[idx_e[it]];
              }
              int index_f = as[idx_f[f_dim-1]];
              for(it = 0; it< f_dim; it++) {
                  index_f += products_f[it]*as[idx_f[it]];
              }
              int index_g = as[idx_g[g_dim-1]];
              for(it = 0; it< g_dim; it++) {
                  index_g += products_g[it]*as[idx_g[it]];
              }
              int index_h = as[idx_h[h_dim-1]];
              for(it = 0; it< h_dim; it++) {
                  index_h += products_h[it]*as[idx_h[it]];
              }
              int index_h1 = as[idx_h1[h1_dim-1]];
              for(it = 0; it< h1_dim; it++) {
                  index_h1 += products_h1[it]*as[idx_h1[it]];
              }
              int index_out = as[idx_out[OutTensor::Dimension-1]];
              for(it = 0; it< OutTensor::Dimension; it++) {
                  index_out += products_out[it]*as[idx_out[it]];
              }
              _vec_a.set(*(a_data+index_a));
              _vec_b.set(*(b_data+index_b));
              _vec_c.set(*(c_data+index_c));
              _vec_d.set(*(d_data+index_d));
              _vec_e.set(*(e_data+index_e));
              _vec_f.set(*(f_data+index_f));
              _vec_g.set(*(g_data+index_g));
              _vec_h.set(*(h_data+index_h));
              V _vec_out = _vec_a*_vec_b*_vec_c*_vec_d*_vec_e*_vec_f*_vec_g*_vec_h*V(h1_data+index_h1) +  V(out_data+index_out);
              _vec_out.store(out_data+index_out);
          }

          return out;
      }
};




template<class Index_0, class Index_1, class Index_2,
         class Index_3, class Index_4, class Index_5, class Index_6,
         class Index_7, class Index_8,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2,
         size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6,
         size_t ... Rest7, size_t ... Rest8>
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                 const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
                 const Tensor<T,Rest6...> &g, const Tensor<T,Rest7...> &h,
                 const Tensor<T,Rest8...> &h1)
-> decltype(extractor_contract_9_no_opt<Index_0,Index_1,Index_2,Index_3,
            Index_4,Index_5,Index_6,Index_7,Index_8>::contract_impl(a,b,c,d,e,f,g,h,h1)) {
    return extractor_contract_9_no_opt<Index_0,Index_1,Index_2,Index_3,
            Index_4,Index_5,Index_6,Index_7,Index_8>::contract_impl(a,b,c,d,e,f,g,h,h1);
}
//---------------------------------------------------------------------------------------------------------------------//










// Ten tensor singleton
//------------------------------------------------------------------------------------------------------------//
template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
struct extractor_contract_10_no_opt {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2,
         size_t ... Idx3, size_t ... Idx4, size_t ... Idx5, size_t ... Idx6,
         size_t ... Idx7, size_t ... Idx8, size_t ... Idx9>
struct extractor_contract_10_no_opt<Index<Idx0...>, Index<Idx1...>,
        Index<Idx2...>, Index<Idx3...>, Index<Idx4...>, Index<Idx5...>, Index<Idx6...>,
        Index<Idx7...>, Index<Idx8...>, Index<Idx9...>> {


      template<typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2,
               size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6,
               size_t ... Rest7, size_t ... Rest8, size_t ... Rest9>
        static
        typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...>,
                 typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                    sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)+\
                    sizeof...(Rest6)+sizeof...(Rest7)+sizeof...(Rest8)+sizeof...(Rest9)>::type>::type
        contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                      const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                      const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
                      const Tensor<T,Rest6...> &g, const Tensor<T,Rest7...> &h,
                      const Tensor<T,Rest8...> &h1, const Tensor<T,Rest9...> &h2) {


          using OutTensor = typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,
                Idx5...,Idx6...,Idx7...,Idx8...,Idx9...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)+\
                sizeof...(Rest6)+sizeof...(Rest7)+sizeof...(Rest8)+sizeof...(Rest9)>::type>::type;
          using OutIndice = typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,
                Idx5...,Idx6...,Idx7...,Idx8...,Idx9...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)+\
                sizeof...(Rest6)+sizeof...(Rest7)+sizeof...(Rest8)+sizeof...(Rest9)>::type>::indices;

          OutTensor out;
          out.zeros();
          const T *a_data = a.data();
          const T *b_data = b.data();
          const T *c_data = c.data();
          const T *d_data = d.data();
          const T *e_data = e.data();
          const T *f_data = f.data();
          const T *g_data = g.data();
          const T *h_data = h.data();
          const T *h1_data = h1.data();
          const T *h2_data = h2.data();
          T *out_data = out.data();

          constexpr int a_dim = sizeof...(Rest0);
          constexpr int b_dim = sizeof...(Rest1);
          constexpr int c_dim = sizeof...(Rest2);
          constexpr int d_dim = sizeof...(Rest3);
          constexpr int e_dim = sizeof...(Rest4);
          constexpr int f_dim = sizeof...(Rest5);
          constexpr int g_dim = sizeof...(Rest6);
          constexpr int h_dim = sizeof...(Rest7);
          constexpr int h1_dim = sizeof...(Rest8);
          constexpr int h2_dim = sizeof...(Rest9);
          constexpr int out_dim =  no_of_unique<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,
                  Idx5...,Idx6...,Idx7...,Idx8...,Idx9...>::value;

          constexpr auto& idx_a = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...>,
                  Index<Idx0...>,Tensor<T,Rest0...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::indices;

          constexpr auto& idx_b = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...>,
                  Index<Idx1...>,Tensor<T,Rest1...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::indices;

          constexpr auto& idx_c = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...>,
                  Index<Idx2...>,Tensor<T,Rest2...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::indices;

          constexpr auto& idx_d = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...>,
                  Index<Idx3...>,Tensor<T,Rest3...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest3)>::type>::indices;

          constexpr auto& idx_e = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...>,
                  Index<Idx4...>,Tensor<T,Rest4...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest4)>::type>::indices;

          constexpr auto& idx_f = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...>,
                  Index<Idx5...>,Tensor<T,Rest5...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest5)>::type>::indices;

          constexpr auto& idx_g = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...>,
                  Index<Idx6...>,Tensor<T,Rest6...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest6)>::type>::indices;

          constexpr auto& idx_h = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...>,
                  Index<Idx7...>,Tensor<T,Rest7...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest7)>::type>::indices;

          constexpr auto& idx_h1 = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...>,
                  Index<Idx8...>,Tensor<T,Rest8...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest8)>::type>::indices;

          constexpr auto& idx_h2 = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...>,
                  Index<Idx9...>,Tensor<T,Rest9...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest9)>::type>::indices;

          constexpr auto& idx_out = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...>,
                  OutIndice,OutTensor,
                  typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::indices;

          using nloops = loop_setter<
            Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...>,
                    typename std_ext::make_index_sequence<out_dim>::type>;
          constexpr auto& maxes_out = nloops::dims;
          constexpr int total = nloops::value;

#ifdef FASTOR_PRINT_COST
        print(total);
#endif

          constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
          constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
          constexpr std::array<size_t,c_dim> products_c = nprods<Index<Rest2...>,typename std_ext::make_index_sequence<c_dim>::type>::values;
          constexpr std::array<size_t,d_dim> products_d = nprods<Index<Rest3...>,typename std_ext::make_index_sequence<d_dim>::type>::values;
          constexpr std::array<size_t,e_dim> products_e = nprods<Index<Rest4...>,typename std_ext::make_index_sequence<e_dim>::type>::values;
          constexpr std::array<size_t,f_dim> products_f = nprods<Index<Rest5...>,typename std_ext::make_index_sequence<f_dim>::type>::values;
          constexpr std::array<size_t,g_dim> products_g = nprods<Index<Rest6...>,typename std_ext::make_index_sequence<g_dim>::type>::values;
          constexpr std::array<size_t,h_dim> products_h = nprods<Index<Rest7...>,typename std_ext::make_index_sequence<h_dim>::type>::values;
          constexpr std::array<size_t,h1_dim> products_h1 = nprods<Index<Rest8...>,typename std_ext::make_index_sequence<h1_dim>::type>::values;
          constexpr std::array<size_t,h2_dim> products_h2 = nprods<Index<Rest9...>,typename std_ext::make_index_sequence<h2_dim>::type>::values;

          using Index_with_dims = typename put_dims_in_Index<OutTensor>::type;
          constexpr std::array<size_t,OutTensor::Dimension> products_out = \
                  nprods<Index_with_dims,typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::values;

#ifndef FASTOR_DONT_VECTORISE
          using vectorisability = is_vectorisable<
                    Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...>,
                    Index<Idx9...>,Tensor<T,Rest9...>>;
          constexpr int stride = vectorisability::stride;
          using V = typename vectorisability::type;
#else
          constexpr int stride = 1;
          using V = SIMDVector<T,sizeof(T)*8>;
#endif

          int as[out_dim];
          std::fill(as,as+out_dim,0);

          int it;
          V _vec_a, _vec_b, _vec_c, _vec_d, _vec_e, _vec_f, _vec_g, _vec_h, _vec_h1;

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
                  index_e += products_e[it]*as[idx_e[it]];
              }
              int index_f = as[idx_f[f_dim-1]];
              for(it = 0; it< f_dim; it++) {
                  index_f += products_f[it]*as[idx_f[it]];
              }
              int index_g = as[idx_g[g_dim-1]];
              for(it = 0; it< g_dim; it++) {
                  index_g += products_g[it]*as[idx_g[it]];
              }
              int index_h = as[idx_h[h_dim-1]];
              for(it = 0; it< h_dim; it++) {
                  index_h += products_h[it]*as[idx_h[it]];
              }
              int index_h1 = as[idx_h1[h1_dim-1]];
              for(it = 0; it< h1_dim; it++) {
                  index_h1 += products_h1[it]*as[idx_h1[it]];
              }
              int index_h2 = as[idx_h2[h2_dim-1]];
              for(it = 0; it< h2_dim; it++) {
                  index_h2 += products_h2[it]*as[idx_h2[it]];
              }
              int index_out = as[idx_out[OutTensor::Dimension-1]];
              for(it = 0; it< OutTensor::Dimension; it++) {
                  index_out += products_out[it]*as[idx_out[it]];
              }
              _vec_a.set(*(a_data+index_a));
              _vec_b.set(*(b_data+index_b));
              _vec_c.set(*(c_data+index_c));
              _vec_d.set(*(d_data+index_d));
              _vec_e.set(*(e_data+index_e));
              _vec_f.set(*(f_data+index_f));
              _vec_g.set(*(g_data+index_g));
              _vec_h.set(*(h_data+index_h));
              _vec_h1.set(*(h1_data+index_h1));
              V _vec_out = _vec_a*_vec_b*_vec_c*_vec_d*_vec_e*_vec_f*_vec_g*_vec_h*_vec_h1*V(h2_data+index_h2) +  V(out_data+index_out);
              _vec_out.store(out_data+index_out);
          }

          return out;
      }
};




template<class Index_0, class Index_1, class Index_2, class Index_3, class Index_4, class Index_5, class Index_6,
         class Index_7, class Index_8, class Index_9,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2,
         size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6,
         size_t ... Rest7, size_t ... Rest8, size_t ... Rest9>
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                 const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
                 const Tensor<T,Rest6...> &g, const Tensor<T,Rest7...> &h,
                 const Tensor<T,Rest8...> &h1, const Tensor<T,Rest9...> &h2)
-> decltype(extractor_contract_10_no_opt<Index_0,Index_1,Index_2,Index_3,
            Index_4,Index_5,Index_6,Index_7,Index_8,Index_9>::contract_impl(a,b,c,d,e,f,g,h,h1,h2)) {
    return extractor_contract_10_no_opt<Index_0,Index_1,Index_2,Index_3,
            Index_4,Index_5,Index_6,Index_7,Index_8,Index_9>::contract_impl(a,b,c,d,e,f,g,h,h1,h2);
}
//---------------------------------------------------------------------------------------------------------------------//














// Eleven tensor singleton
//------------------------------------------------------------------------------------------------------------//
template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10>
struct extractor_contract_11_no_opt {};

template<size_t ... Idx0, size_t ... Idx1, size_t ... Idx2,
         size_t ... Idx3, size_t ... Idx4, size_t ... Idx5, size_t ... Idx6,
         size_t ... Idx7, size_t ... Idx8, size_t ... Idx9, size_t ... Idx10>
struct extractor_contract_11_no_opt<Index<Idx0...>, Index<Idx1...>,
        Index<Idx2...>, Index<Idx3...>, Index<Idx4...>, Index<Idx5...>, Index<Idx6...>,
        Index<Idx7...>, Index<Idx8...>, Index<Idx9...>, Index<Idx10...>> {


      template<typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2,
               size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6,
               size_t ... Rest7, size_t ... Rest8, size_t ... Rest9, size_t ... Rest10>
        static
        typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...,Idx10...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...,Rest10...>,
                 typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                    sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)+\
                    sizeof...(Rest6)+sizeof...(Rest7)+sizeof...(Rest8)+sizeof...(Rest9)+sizeof...(Rest10)>::type>::type
        contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                      const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                      const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
                      const Tensor<T,Rest6...> &g, const Tensor<T,Rest7...> &h,
                      const Tensor<T,Rest8...> &h1, const Tensor<T,Rest9...> &h2,
                      const Tensor<T,Rest10...> &h3) {


          using OutTensor = typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,
                Idx5...,Idx6...,Idx7...,Idx8...,Idx9...,Idx10...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...,Rest10...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)+\
                sizeof...(Rest6)+sizeof...(Rest7)+sizeof...(Rest8)+sizeof...(Rest9)+sizeof...(Rest10)>::type>::type;
          using OutIndice = typename contraction_impl<Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,
                Idx5...,Idx6...,Idx7...,Idx8...,Idx9...,Idx10...>,
            Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...,Rest10...>,
            typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)+\
                sizeof...(Rest2)+sizeof...(Rest3)+sizeof...(Rest4)+sizeof...(Rest5)+\
                sizeof...(Rest6)+sizeof...(Rest7)+sizeof...(Rest8)+sizeof...(Rest9)+sizeof...(Rest10)>::type>::indices;

          OutTensor out;
          out.zeros();
          const T *a_data = a.data();
          const T *b_data = b.data();
          const T *c_data = c.data();
          const T *d_data = d.data();
          const T *e_data = e.data();
          const T *f_data = f.data();
          const T *g_data = g.data();
          const T *h_data = h.data();
          const T *h1_data = h1.data();
          const T *h2_data = h2.data();
          const T *h3_data = h3.data();
          T *out_data = out.data();

          constexpr int a_dim = sizeof...(Rest0);
          constexpr int b_dim = sizeof...(Rest1);
          constexpr int c_dim = sizeof...(Rest2);
          constexpr int d_dim = sizeof...(Rest3);
          constexpr int e_dim = sizeof...(Rest4);
          constexpr int f_dim = sizeof...(Rest5);
          constexpr int g_dim = sizeof...(Rest6);
          constexpr int h_dim = sizeof...(Rest7);
          constexpr int h1_dim = sizeof...(Rest8);
          constexpr int h2_dim = sizeof...(Rest9);
          constexpr int h3_dim = sizeof...(Rest10);
          constexpr int out_dim =  no_of_unique<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,
                  Idx5...,Idx6...,Idx7...,Idx8...,Idx9...,Idx10...>::value;

          constexpr auto& idx_a = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...,Idx10...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...,Rest10...>,
                  Index<Idx0...>,Tensor<T,Rest0...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::indices;

          constexpr auto& idx_b = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...,Idx10...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...,Rest10...>,
                  Index<Idx1...>,Tensor<T,Rest1...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::indices;

          constexpr auto& idx_c = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...,Idx10...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...,Rest10...>,
                  Index<Idx2...>,Tensor<T,Rest2...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest2)>::type>::indices;

          constexpr auto& idx_d = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...,Idx10...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...,Rest10...>,
                  Index<Idx3...>,Tensor<T,Rest3...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest3)>::type>::indices;

          constexpr auto& idx_e = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...,Idx10...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...,Rest10...>,
                  Index<Idx4...>,Tensor<T,Rest4...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest4)>::type>::indices;

          constexpr auto& idx_f = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...,Idx10...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...,Rest10...>,
                  Index<Idx5...>,Tensor<T,Rest5...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest5)>::type>::indices;

          constexpr auto& idx_g = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...,Idx10...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...,Rest10...>,
                  Index<Idx6...>,Tensor<T,Rest6...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest6)>::type>::indices;

          constexpr auto& idx_h = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...,Idx10...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...,Rest10...>,
                  Index<Idx7...>,Tensor<T,Rest7...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest7)>::type>::indices;

          constexpr auto& idx_h1 = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...,Idx10...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...,Rest10...>,
                  Index<Idx8...>,Tensor<T,Rest8...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest8)>::type>::indices;

          constexpr auto& idx_h2 = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...,Idx10...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...,Rest10...>,
                  Index<Idx9...>,Tensor<T,Rest9...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest9)>::type>::indices;

          constexpr auto& idx_h3 = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...,Idx10...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...,Rest10...>,
                  Index<Idx10...>,Tensor<T,Rest10...>,
                  typename std_ext::make_index_sequence<sizeof...(Rest10)>::type>::indices;

          constexpr auto& idx_out = IndexTensors<
                  Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...,Idx10...>,
                  Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...,Rest10...>,
                  OutIndice,OutTensor,
                  typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::indices;

          using nloops = loop_setter<
              Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...,Idx10...>,
              Tensor<T,Rest0...,Rest1...,Rest2...,Rest3...,Rest4...,Rest5...,Rest6...,Rest7...,Rest8...,Rest9...,Rest10...>,
                    typename std_ext::make_index_sequence<out_dim>::type>;
          constexpr auto& maxes_out = nloops::dims;
          constexpr int total = nloops::value;

#ifdef FASTOR_PRINT_COST
        print(total);
#endif

          constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
          constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
          constexpr std::array<size_t,c_dim> products_c = nprods<Index<Rest2...>,typename std_ext::make_index_sequence<c_dim>::type>::values;
          constexpr std::array<size_t,d_dim> products_d = nprods<Index<Rest3...>,typename std_ext::make_index_sequence<d_dim>::type>::values;
          constexpr std::array<size_t,e_dim> products_e = nprods<Index<Rest4...>,typename std_ext::make_index_sequence<e_dim>::type>::values;
          constexpr std::array<size_t,f_dim> products_f = nprods<Index<Rest5...>,typename std_ext::make_index_sequence<f_dim>::type>::values;
          constexpr std::array<size_t,g_dim> products_g = nprods<Index<Rest6...>,typename std_ext::make_index_sequence<g_dim>::type>::values;
          constexpr std::array<size_t,h_dim> products_h = nprods<Index<Rest7...>,typename std_ext::make_index_sequence<h_dim>::type>::values;
          constexpr std::array<size_t,h1_dim> products_h1 = nprods<Index<Rest8...>,typename std_ext::make_index_sequence<h1_dim>::type>::values;
          constexpr std::array<size_t,h2_dim> products_h2 = nprods<Index<Rest9...>,typename std_ext::make_index_sequence<h2_dim>::type>::values;
          constexpr std::array<size_t,h3_dim> products_h3 = nprods<Index<Rest10...>,typename std_ext::make_index_sequence<h3_dim>::type>::values;

          using Index_with_dims = typename put_dims_in_Index<OutTensor>::type;
          constexpr std::array<size_t,OutTensor::Dimension> products_out = \
                  nprods<Index_with_dims,typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::values;

#ifndef FASTOR_DONT_VECTORISE
          using vectorisability = is_vectorisable<
                    Index<Idx0...,Idx1...,Idx2...,Idx3...,Idx4...,Idx5...,Idx6...,Idx7...,Idx8...,Idx9...>,
                    Index<Idx10...>,Tensor<T,Rest10...>>;
          constexpr int stride = vectorisability::stride;
          using V = typename vectorisability::type;
#else
          constexpr int stride = 1;
          using V = SIMDVector<T,sizeof(T)*8>;
#endif

          int as[out_dim];
          std::fill(as,as+out_dim,0);

          int it;
          V _vec_a, _vec_b, _vec_c, _vec_d, _vec_e, _vec_f, _vec_g, _vec_h, _vec_h1, _vec_h2;

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
                  index_e += products_e[it]*as[idx_e[it]];
              }
              int index_f = as[idx_f[f_dim-1]];
              for(it = 0; it< f_dim; it++) {
                  index_f += products_f[it]*as[idx_f[it]];
              }
              int index_g = as[idx_g[g_dim-1]];
              for(it = 0; it< g_dim; it++) {
                  index_g += products_g[it]*as[idx_g[it]];
              }
              int index_h = as[idx_h[h_dim-1]];
              for(it = 0; it< h_dim; it++) {
                  index_h += products_h[it]*as[idx_h[it]];
              }
              int index_h1 = as[idx_h1[h1_dim-1]];
              for(it = 0; it< h1_dim; it++) {
                  index_h1 += products_h1[it]*as[idx_h1[it]];
              }
              int index_h2 = as[idx_h2[h2_dim-1]];
              for(it = 0; it< h2_dim; it++) {
                  index_h2 += products_h2[it]*as[idx_h2[it]];
              }
              int index_h3 = as[idx_h3[h3_dim-1]];
              for(it = 0; it< h3_dim; it++) {
                  index_h3 += products_h3[it]*as[idx_h3[it]];
              }
              int index_out = as[idx_out[OutTensor::Dimension-1]];
              for(it = 0; it< OutTensor::Dimension; it++) {
                  index_out += products_out[it]*as[idx_out[it]];
              }
              _vec_a.set(*(a_data+index_a));
              _vec_b.set(*(b_data+index_b));
              _vec_c.set(*(c_data+index_c));
              _vec_d.set(*(d_data+index_d));
              _vec_e.set(*(e_data+index_e));
              _vec_f.set(*(f_data+index_f));
              _vec_g.set(*(g_data+index_g));
              _vec_h.set(*(h_data+index_h));
              _vec_h1.set(*(h1_data+index_h1));
              _vec_h2.set(*(h2_data+index_h2));
              V _vec_out = _vec_a*_vec_b*_vec_c*_vec_d*_vec_e*_vec_f*_vec_g*_vec_h*_vec_h1*_vec_h2*V(h3_data+index_h3) +  V(out_data+index_out);
              _vec_out.store(out_data+index_out);
          }

          return out;
      }
};




template<class Index_0, class Index_1, class Index_2, class Index_3, class Index_4, class Index_5, class Index_6,
         class Index_7, class Index_8, class Index_9, class Index_10,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2,
         size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6,
         size_t ... Rest7, size_t ... Rest8, size_t ... Rest9, size_t ... Rest10>
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                 const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
                 const Tensor<T,Rest6...> &g, const Tensor<T,Rest7...> &h,
                 const Tensor<T,Rest8...> &h1, const Tensor<T,Rest9...> &h2,
                 const Tensor<T,Rest10...> &h3)
-> decltype(extractor_contract_11_no_opt<Index_0,Index_1,Index_2,Index_3,
            Index_4,Index_5,Index_6,Index_7,Index_8,Index_9,Index_10>::contract_impl(a,b,c,d,e,f,g,h,h1,h2,h3)) {
    return extractor_contract_11_no_opt<Index_0,Index_1,Index_2,Index_3,
            Index_4,Index_5,Index_6,Index_7,Index_8,Index_9,Index_10>::contract_impl(a,b,c,d,e,f,g,h,h1,h2,h3);
}
//---------------------------------------------------------------------------------------------------------------------//





}

#endif


#endif // CONTRACTION_NO_OPT_H

