#ifndef CONTRACTION_H
#define CONTRACTION_H

#include "Fastor/backend/dyadic.h"
#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor_algebra/indicial.h"


namespace Fastor {


//using namespace details;
//namespace details {


// Define this if fastest (complete meta-engine based) tensor contraction is required.
// Note that this blows up memory consumption and compilation time exponentially

//#define CONTRACT_OPT 2

// Define this if faster (partial meta-engine based) tensor contraction is required.
// Note that this blows up memory consumption and compilation time but not as much

//#define CONTRACT_OPT 1


template<class Idx0, class Idx1, class Tens0, class Tens1, size_t ... Args>
struct RecursiveCartesian;

template<typename T, size_t ...Idx0, size_t ...Idx1, size_t ...Rest0, size_t ...Rest1, size_t First, size_t ... Lasts>
struct RecursiveCartesian<Index<Idx0...>, Index<Idx1...>, Tensor<T,Rest0...>, Tensor<T,Rest1...>, First, Lasts...> {

    static constexpr int out_dim =  no_of_unique<Idx0...,Idx1...>::value;
    static
    FASTOR_INLINE
    void Do(const T *a_data, const T *b_data, T *out_data, std::array<int,out_dim> &as, std::array<int,out_dim> &idx) {
        for (size_t i=0; i<First; ++i) {
            idx[sizeof...(Lasts)] = i;
            RecursiveCartesian<Index<Idx0...>, Index<Idx1...>,
                Tensor<T,Rest0...>, Tensor<T,Rest1...>,Lasts...>::Do(a_data, b_data, out_data, as, idx);
        }
    }
};


template<typename T, size_t Last, size_t ...Idx0, size_t ...Idx1, size_t ...Rest0, size_t ...Rest1>
struct RecursiveCartesian<Index<Idx0...>, Index<Idx1...>, Tensor<T,Rest0...>, Tensor<T,Rest1...>,Last>
{
    using _contraction_impl = contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
    typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>;
    using resulting_tensor = typename _contraction_impl::type;
    using resulting_index  = typename _contraction_impl::indices;
    static constexpr bool _is_reduction = resulting_tensor::dimension_t::value == 0;

    static constexpr int a_dim = sizeof...(Rest0);
    static constexpr int b_dim = sizeof...(Rest1);
    static constexpr int out_dim =  no_of_unique<Idx0...,Idx1...>::value;

    static constexpr auto& idx_a = IndexTensors<
          Index<Idx0..., Idx1...>,
          Tensor<T,Rest0...,Rest1...>,
          Index<Idx0...>,Tensor<T,Rest0...>,
          typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::indices;

    static constexpr auto& idx_b = IndexTensors<
          Index<Idx0..., Idx1...>,
          Tensor<T,Rest0...,Rest1...>,
          Index<Idx1...>,Tensor<T,Rest1...>,
          typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::indices;

    static constexpr auto& idx_out = IndexTensors<
          Index<Idx0..., Idx1...>,
          Tensor<T,Rest0...,Rest1...>,
          resulting_index,resulting_tensor,
          typename std_ext::make_index_sequence<resulting_tensor::Dimension>::type>::indices;

    using nloops = loop_setter<
            Index<Idx0...,Idx1...>,
            Tensor<T,Rest0...,Rest1...>,
            typename std_ext::make_index_sequence<out_dim>::type>;
    static constexpr auto& maxes_out = nloops::dims;
    static constexpr int total = nloops::value;

    static constexpr std::array<size_t,sizeof...(Rest0)> products_a = nprods<Index<Rest0...>,
        typename std_ext::make_index_sequence<a_dim>::type>::values;
    static constexpr std::array<size_t,sizeof...(Rest1)> products_b = nprods<Index<Rest1...>,
        typename std_ext::make_index_sequence<b_dim>::type>::values;

#ifndef FASTOR_DONT_VECTORISE
    using vectorisability = is_vectorisable<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest1...>>;
    static constexpr int stride = vectorisability::stride;
    using V = typename vectorisability::type;
#else
    static constexpr int stride = 1;
    using V = SIMDVector<T,simd_abi::scalar>;
#endif



    static
    FASTOR_INLINE
    void Do(const T *a_data, const T *b_data, T *out_data, std::array<int,out_dim> &as, std::array<int,out_dim> &idx)
    {
        FASTOR_IF_CONSTEXPR(!_is_reduction) {

            using Index_with_dims = typename put_dims_in_Index<resulting_tensor>::type;
            constexpr std::array<size_t,resulting_tensor::Dimension> products_out = \
                nprods<Index_with_dims,typename std_ext::make_index_sequence<resulting_tensor::Dimension>::type>::values;

            V _vec_a;
            for (size_t i=0; i<Last; i+=stride) {
                idx[0] = i;
                std::reverse_copy(idx.begin(),idx.end(),as.begin());

                int index_a = as[idx_a[a_dim-1]];
                for(int it = 0; it< a_dim; it++) {
                  index_a += products_a[it]*as[idx_a[it]];
                }
                int index_b = as[idx_b[b_dim-1]];
                for(int it = 0; it< b_dim; it++) {
                  index_b += products_b[it]*as[idx_b[it]];
                }
                int index_out = as[idx_out[resulting_tensor::Dimension-1]];
                for(int it = 0; it< static_cast<int>(resulting_tensor::Dimension); it++) {
                  index_out += products_out[it]*as[idx_out[it]];
                }

                // out_data[index_out] += a_data[index_a]*b_data[index_b];
                // _vec_a.set(*(a_data+index_a));
                _vec_a.set(a_data[index_a]);
                // V _vec_out = _vec_a*V(b_data+index_b) +  V(out_data+index_out);
                V _vec_out = fmadd(_vec_a,V(&b_data[index_b]),  V(&out_data[index_out]));
                _vec_out.store(out_data+index_out,false);
                // _vec_out.aligned_store(&out_data[index_out]);
            }
        }

        else {

            using Index_with_dims = typename put_dims_in_Index<resulting_tensor>::type;
            constexpr std::array<size_t,resulting_tensor::Dimension> products_out = \
                nprods<Index_with_dims,typename std_ext::make_index_sequence<resulting_tensor::Dimension>::type>::values;

            V _vec_a;
            for (size_t i=0; i<Last; i+=stride) {
                idx[0] = i;
                std::reverse_copy(idx.begin(),idx.end(),as.begin());

                int index_a = as[idx_a[a_dim-1]];
                for(int it = 0; it< a_dim; it++) {
                  index_a += products_a[it]*as[idx_a[it]];
                }
                int index_b = as[idx_b[b_dim-1]];
                for(int it = 0; it< b_dim; it++) {
                  index_b += products_b[it]*as[idx_b[it]];
                }

                out_data[0] += a_data[index_a]*b_data[index_b];
            }
        }
        return;
    }
};


// template<typename T, size_t ...Idx0, size_t ...Idx1, size_t ...Rest0, size_t ...Rest1>
// struct RecursiveCartesian<Index<Idx0...>, Index<Idx1...>, Tensor<T,Rest0...>, Tensor<T,Rest1...>>
// {
//     using OutTensor = typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
//     typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type;
//     using OutIndices = typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
//     typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::indices;

//     static constexpr int a_dim = sizeof...(Rest0);
//     static constexpr int b_dim = sizeof...(Rest1);
//     static constexpr int out_dim =  no_of_unique<Idx0...,Idx1...>::value;

//     static constexpr auto& idx_a = IndexTensors<
//           Index<Idx0..., Idx1...>,
//           Tensor<T,Rest0...,Rest1...>,
//           Index<Idx0...>,Tensor<T,Rest0...>,
//           typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::indices;

//     static constexpr auto& idx_b = IndexTensors<
//           Index<Idx0..., Idx1...>,
//           Tensor<T,Rest0...,Rest1...>,
//           Index<Idx1...>,Tensor<T,Rest1...>,
//           typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::indices;

//     static constexpr auto& idx_out = IndexTensors<
//           Index<Idx0..., Idx1...>,
//           Tensor<T,Rest0...,Rest1...>,
//           OutIndices,OutTensor,
//           typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::indices;

//     using nloops = loop_setter<
//             Index<Idx0...,Idx1...>,
//             Tensor<T,Rest0...,Rest1...>,
//             typename std_ext::make_index_sequence<out_dim>::type>;
//     static constexpr auto& maxes_out = nloops::dims;
//     static constexpr int total = nloops::value;

//     static constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
//     static constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;

//     using Index_with_dims = typename put_dims_in_Index<OutTensor>::type;
//     static constexpr std::array<size_t,OutTensor::Dimension> products_out =
//           nprods<Index_with_dims,typename std_ext::make_index_sequence<OutTensor::Dimension>::type>::values;

// #ifndef FASTOR_DONT_VECTORISE
//               using vectorisability = is_vectorisable<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest1...>>;
//               static constexpr int stride = vectorisability::stride;
//               using V = typename vectorisability::type;
// #else
//               static constexpr int stride = 1;
//               using V = SIMDVector<T,sizeof(T)*8>;
// #endif


//     static void Do(const T *a_data, const T *b_data, T *out_data, std::array<int,out_dim> &as, std::array<int,out_dim> &idx)
//     {
//         std::reverse_copy(idx.begin(),idx.end(),as.begin());

//         int index_a = as[idx_a[a_dim-1]];
//         for(int it = 0; it< a_dim; it++) {
//           index_a += products_a[it]*as[idx_a[it]];
//         }
//         int index_b = as[idx_b[b_dim-1]];
//         for(int it = 0; it< b_dim; it++) {
//           index_b += products_b[it]*as[idx_b[it]];
//         }
//         int index_out = as[idx_out[OutTensor::Dimension-1]];
//         for(int it = 0; it< static_cast<int>(OutTensor::Dimension); it++) {
//           index_out += products_out[it]*as[idx_out[it]];
//         }

//         out_data[index_out] += a_data[index_a]*b_data[index_b];
//         return;
//     }
// };

template<typename T, size_t Last, size_t ...Idx0, size_t ...Idx1, size_t ...Rest0, size_t ...Rest1>
constexpr std::array<size_t,sizeof...(Rest0)> RecursiveCartesian<Index<Idx0...>, Index<Idx1...>,
  Tensor<T,Rest0...>, Tensor<T,Rest1...>,Last>::products_a;

template<typename T, size_t Last, size_t ...Idx0, size_t ...Idx1, size_t ...Rest0, size_t ...Rest1>
constexpr std::array<size_t,sizeof...(Rest1)> RecursiveCartesian<Index<Idx0...>, Index<Idx1...>,
  Tensor<T,Rest0...>, Tensor<T,Rest1...>,Last>::products_b;

// template<typename T, size_t Last, size_t ...Idx0, size_t ...Idx1, size_t ...Rest0, size_t ...Rest1>
// constexpr std::array<size_t,RecursiveCartesian<Index<Idx0...>, Index<Idx1...>,
//   Tensor<T,Rest0...>, Tensor<T,Rest1...>,Last>::OutTensor::Dimension> RecursiveCartesian<Index<Idx0...>, Index<Idx1...>,
//   Tensor<T,Rest0...>, Tensor<T,Rest1...>,Last>::products_out;





template<class Idx0, class Idx1, class Tens0, class Tens1, class Args>
struct RecursiveCartesianDispatcher;

template<typename T, size_t ...Idx0, size_t ...Idx1, size_t ...Rest0, size_t ...Rest1, size_t ... Args>
struct RecursiveCartesianDispatcher<Index<Idx0...>, Index<Idx1...>, Tensor<T,Rest0...>, Tensor<T,Rest1...>, Index<Args...> >
{
    static constexpr int out_dim =  no_of_unique<Idx0...,Idx1...>::value;

    static FASTOR_INLINE void Do(const T *a_data, const T *b_data, T *out_data,
      std::array<int,out_dim> &as, std::array<int,out_dim> &idx) {
      return RecursiveCartesian<Index<Idx0...>, Index<Idx1...>,
        Tensor<T,Rest0...>, Tensor<T,Rest1...>, Args...>::Do(a_data, b_data, out_data, as, idx);
    }
};



template<class T, class U, class enable=void>
struct extractor_contract_2 {};

template<size_t ... Idx0, size_t ... Idx1>
struct extractor_contract_2<Index<Idx0...>, Index<Idx1...>,
          typename std::enable_if<no_of_unique<Idx0...,Idx1...>::value!=sizeof...(Idx0)+sizeof...(Idx1)>::type> {

    template<typename T, size_t ... Rest0, size_t ... Rest1>
      static
      FASTOR_INLINE
      typename contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
               typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>::type
      contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) {

          using _contraction_impl = contraction_impl<Index<Idx0...,Idx1...>, Tensor<T,Rest0...,Rest1...>,
              typename std_ext::make_index_sequence<sizeof...(Rest0)+sizeof...(Rest1)>::type>;
          using resulting_tensor = typename _contraction_impl::type;
          // is reduction including permuted reduction
          constexpr bool _is_reduction = resulting_tensor::dimension_t::value == 0;

#if CONTRACT_OPT==-3

          static_assert(!_is_reduction,"THIS VARIANT OF EINSUM CANNOT DEAL WITH REDUCTION CASES");

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

#ifndef FASTOR_DONT_VECTORISE
          using vectorisability = is_vectorisable<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest1...>>;
          constexpr int stride = vectorisability::stride;
          using V = typename vectorisability::type;
#else
          constexpr int stride = 1;
          using V = SIMDVector<T,sizeof(T)*8>;
#endif

          V _vec_a;
          for (int i = 0; i < total; i+=stride) {
              // out_data[index_out[i]] += a_data[index_a[i]]*b_data[index_b[i]];
              _vec_a.set(*(a_data+index_a[i]));
              V _vec_out = _vec_a*V(b_data+index_b[i]) +  V(out_data+index_out[i]);
              // _vec_out.store(out_data+index_out[i]);
              _vec_out.aligned_store(out_data+index_out[i]);
          }

        return out;
    }

#elif CONTRACT_OPT==-2

          static_assert(!_is_reduction,"THIS VARIANT OF EINSUM CANNOT DEAL WITH REDUCTION CASES");

          resulting_tensor out;
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
                                    typename std_ext::make_index_sequence<resulting_tensor::Dimension>::type>::indices;

          constexpr int total = no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
                  typename std_ext::make_index_sequence<no_of_unique<Idx0...,Idx1...>::value>::type>::value;

          using maxes_out_type = typename no_of_loops_to_set<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest0...>,Tensor<T,Rest1...>,
                  typename std_ext::make_index_sequence<no_of_unique<Idx0...,Idx1...>::value>::type>::type;

          constexpr auto& as_all = cartesian_product<maxes_out_type,typename std_ext::make_index_sequence<total>::type>::values;
          // constexpr auto as_all = cartesian_product_2<maxes_out.size(),total>(maxes_out);

          constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
          constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
          using Index_with_dims = typename put_dims_in_Index<resulting_tensor>::type;
          constexpr std::array<size_t,Index_with_dims::NoIndices> products_out = nprods<Index_with_dims,
                  typename std_ext::make_index_sequence<Index_with_dims::NoIndices>::type>::values;

#ifndef FASTOR_DONT_VECTORISE
          using vectorisability = is_vectorisable<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest1...>>;
          constexpr int stride = vectorisability::stride;
          using V = typename vectorisability::type;
#else
          constexpr int stride = 1;
          using V = SIMDVector<T,simd_abi::scalar>;
#endif

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
              // _vec_out.store(out_data+index_out);
              _vec_out.aligned_store(out_data+index_out);
          }

          return out;
      }

#elif CONTRACT_OPT==-1

              using resulting_index  = typename _contraction_impl::indices;
              resulting_tensor out;
              out.zeros();
              const T *a_data = a.data();
              const T *b_data = b.data();
              T *out_data = out.data();

              constexpr int a_dim = sizeof...(Rest0);
              constexpr int b_dim = sizeof...(Rest1);
              constexpr int out_dim =  no_of_unique<Idx0...,Idx1...>::value;

              constexpr auto& idx_a = IndexTensors<
                      Index<Idx0..., Idx1...>,
                      Tensor<T,Rest0...,Rest1...>,
                      Index<Idx0...>,Tensor<T,Rest0...>,
                      typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::indices;

              constexpr auto& idx_b = IndexTensors<
                      Index<Idx0..., Idx1...>,
                      Tensor<T,Rest0...,Rest1...>,
                      Index<Idx1...>,Tensor<T,Rest1...>,
                      typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::indices;

              constexpr auto& idx_out = IndexTensors<
                      Index<Idx0..., Idx1...>,
                      Tensor<T,Rest0...,Rest1...>,
                      resulting_index,resulting_tensor,
                      typename std_ext::make_index_sequence<resulting_tensor::Dimension>::type>::indices;

              using nloops = loop_setter<
                        Index<Idx0...,Idx1...>,
                        Tensor<T,Rest0...,Rest1...>,
                        typename std_ext::make_index_sequence<out_dim>::type>;
              constexpr auto& maxes_out = nloops::dims;
              constexpr int total = nloops::value;

              constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
              constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;

#ifndef FASTOR_DONT_VECTORISE
              using vectorisability = is_vectorisable<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest1...>>;
              constexpr int stride = vectorisability::stride;
              using V = typename vectorisability::type;
#else
              constexpr int stride = 1;
              using V = SIMDVector<T,sizeof(T)*8>;
#endif
              std::array<int,out_dim> as = {};

              int it;
              V _vec_a;

#if FASTOR_CXX_VERSION >= 2017
              constexpr std::array<int,out_dim> remainings = find_remaining(maxes_out, total);
#endif

              FASTOR_IF_CONSTEXPR(!_is_reduction) {

                  using Index_with_dims = typename put_dims_in_Index<resulting_tensor>::type;
                  constexpr std::array<size_t,resulting_tensor::Dimension> products_out = \
                          nprods<Index_with_dims,typename std_ext::make_index_sequence<resulting_tensor::Dimension>::type>::values;

                  for (int i = 0; i < total; i+=stride) {

#if FASTOR_CXX_VERSION >= 2017
                      for (int n = 0; n < out_dim; ++n) {
                          as[n] = ( i / remainings[n] ) % (int)maxes_out[n];
                      }
#else
                      int remaining = total;
                      for (int n = 0; n < out_dim; ++n) {
                          remaining /= maxes_out[n];
                          as[n] = ( i / remaining ) % maxes_out[n];
                      }
#endif

                      int index_a = as[idx_a[a_dim-1]];
                      for(it = 0; it< a_dim; it++) {
                          index_a += products_a[it]*as[idx_a[it]];
                      }
                      int index_b = as[idx_b[b_dim-1]];
                      for(it = 0; it< b_dim; it++) {
                          index_b += products_b[it]*as[idx_b[it]];
                      }
                      int index_out = as[idx_out[resulting_tensor::Dimension-1]];
                      for(it = 0; it< static_cast<int>(resulting_tensor::Dimension); it++) {
                          index_out += products_out[it]*as[idx_out[it]];
                      }
    //                  println(index_out,index_a,index_b,"\n");
                      _vec_a.set(*(a_data+index_a));
    //                  _vec_a.broadcast(&a_data[index_a]);
                      V _vec_out = _vec_a*V(b_data+index_b) +  V(out_data+index_out);
                      // V _vec_out = fmadd(_vec_a,V(b_data+index_b),  V(out_data+index_out));
                      // _vec_out.store(out_data+index_out);
                      _vec_out.aligned_store(out_data+index_out);
                  }

                  // ACTUALLY MUCH SLOWER
                  // constexpr auto as_all = cartesian_product_2<out_dim,total>(maxes_out);

                  // for (int i = 0; i < total; i+=stride) {
                  //     int index_a = as_all[i][idx_a[a_dim-1]];
                  //     for(it = 0; it< a_dim; it++) {
                  //         index_a += products_a[it]*as_all[i][idx_a[it]];
                  //     }
                  //     int index_b = as_all[i][idx_b[b_dim-1]];
                  //     for(it = 0; it< b_dim; it++) {
                  //         index_b += products_b[it]*as_all[i][idx_b[it]];
                  //     }
                  //     int index_out = as_all[i][idx_out[idx_out.size()-1]];
                  //     for(it = 0; it< idx_out.size(); it++) {
                  //         index_out += products_out[it]*as_all[i][idx_out[it]];
                  //     }

                  //     _vec_a.set(*(a_data+index_a));
                  //     // V _vec_out = _vec_a*V(b_data+index_b) +  V(out_data+index_out);
                  //     V _vec_out = fmadd(_vec_a,V(b_data+index_b),  V(out_data+index_out));
                  //     _vec_out.aligned_store(out_data+index_out);
                  // }

              }

              else {
                  for (int i = 0; i < total; i+=stride) {

#if FASTOR_CXX_VERSION >= 2017
                      for (int n = 0; n < out_dim; ++n) {
                          as[n] = ( i / remainings[n] ) % (int)maxes_out[n];
                      }
#else
                      int remaining = total;
                      for (int n = 0; n < out_dim; ++n) {
                          remaining /= maxes_out[n];
                          as[n] = ( i / remaining ) % maxes_out[n];
                      }
#endif

                      int index_a = as[idx_a[a_dim-1]];
                      for(it = 0; it< a_dim; it++) {
                          index_a += products_a[it]*as[idx_a[it]];
                      }
                      int index_b = as[idx_b[b_dim-1]];
                      for(it = 0; it< b_dim; it++) {
                          index_b += products_b[it]*as[idx_b[it]];
                      }
                      out_data[0] += a_data[index_a]*b_data[index_b];
                  }
              }

              return out;
          }
#elif CONTRACT_OPT==1

              using resulting_index  = typename _contraction_impl::indices;
              resulting_tensor out;
              out.zeros();
              const T *a_data = a.data();
              const T *b_data = b.data();
              T *out_data = out.data();

              constexpr int a_dim = sizeof...(Rest0);
              constexpr int b_dim = sizeof...(Rest1);
              constexpr int out_dim =  no_of_unique<Idx0...,Idx1...>::value;

              constexpr auto& idx_a = IndexTensors<
                      Index<Idx0..., Idx1...>,
                      Tensor<T,Rest0...,Rest1...>,
                      Index<Idx0...>,Tensor<T,Rest0...>,
                      typename std_ext::make_index_sequence<sizeof...(Rest0)>::type>::indices;

              constexpr auto& idx_b = IndexTensors<
                      Index<Idx0..., Idx1...>,
                      Tensor<T,Rest0...,Rest1...>,
                      Index<Idx1...>,Tensor<T,Rest1...>,
                      typename std_ext::make_index_sequence<sizeof...(Rest1)>::type>::indices;

              constexpr auto& idx_out = IndexTensors<
                      Index<Idx0..., Idx1...>,
                      Tensor<T,Rest0...,Rest1...>,
                      resulting_index,resulting_tensor,
                      typename std_ext::make_index_sequence<resulting_tensor::Dimension>::type>::indices;

              using nloops = loop_setter<
                        Index<Idx0...,Idx1...>,
                        Tensor<T,Rest0...,Rest1...>,
                        typename std_ext::make_index_sequence<out_dim>::type>;
              constexpr auto& maxes_out = nloops::dims;
              constexpr int total = nloops::value;

              constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
              constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;

#ifndef FASTOR_DONT_VECTORISE
              using vectorisability = is_vectorisable<Index<Idx0...>,Index<Idx1...>,Tensor<T,Rest1...>>;
              constexpr int stride = vectorisability::stride;
              using V = typename vectorisability::type;
#else
              constexpr int stride = 1;
              using V = SIMDVector<T,sizeof(T)*8>;
#endif
              std::array<int,out_dim> as = {};

              int it, jt, counter = 0;
              V _vec_a;

              FASTOR_IF_CONSTEXPR(!_is_reduction) {
                  using Index_with_dims = typename put_dims_in_Index<resulting_tensor>::type;
                  constexpr std::array<size_t,resulting_tensor::Dimension> products_out = \
                          nprods<Index_with_dims,typename std_ext::make_index_sequence<resulting_tensor::Dimension>::type>::values;

                  while(true)
                  {
                      int index_a = as[idx_a[a_dim-1]];
                      for(it = 0; it< a_dim; it++) {
                         index_a += products_a[it]*as[idx_a[it]];
                      }
                      int index_b = as[idx_b[b_dim-1]];
                      for(it = 0; it< b_dim; it++) {
                         index_b += products_b[it]*as[idx_b[it]];
                      }

                      int index_out = as[idx_out[resulting_tensor::Dimension-1]];
                      for(it = 0; it< static_cast<int>(resulting_tensor::Dimension); it++) {
                         index_out += products_out[it]*as[idx_out[it]];
                      }

                      _vec_a.set(*(a_data+index_a));
                      // V _vec_out = _vec_a*V(b_data+index_b) +  V(out_data+index_out);
                      V _vec_out = fmadd(_vec_a,V(b_data+index_b),  V(out_data+index_out));
                      _vec_out.store(out_data+index_out,false);
                      // _vec_out.aligned_store(out_data+index_out);

                      counter++;
                      for(jt = maxes_out.size()-1; jt>=0 ; jt--)
                      {
                          if (jt == maxes_out.size()-1) as[jt]+=stride;
                          else as[jt] +=1;
                          if(as[jt]<maxes_out[jt])
                              break;
                          else
                              as[jt]=0;
                      }
                      if(jt<0)
                          break;
                  }
              }

              else {
                  while(true)
                  {
                      int index_a = as[idx_a[a_dim-1]];
                      for(it = 0; it< a_dim; it++) {
                         index_a += products_a[it]*as[idx_a[it]];
                      }
                      int index_b = as[idx_b[b_dim-1]];
                      for(it = 0; it< b_dim; it++) {
                         index_b += products_b[it]*as[idx_b[it]];
                      }

                      out_data[0] += a_data[index_a]*b_data[index_b];

                      counter++;
                      for(jt = maxes_out.size()-1; jt>=0 ; jt--)
                      {
                          if (jt == maxes_out.size()-1) as[jt]+=1;
                          else as[jt] +=1;
                          if(as[jt]<maxes_out[jt])
                              break;
                          else
                              as[jt]=0;
                      }
                      if(jt<0)
                          break;
                  }
              }

              return out;
          }
#else

              resulting_tensor out;
              out.zeros();
              const T *a_data = a.data();
              const T *b_data = b.data();
              T *out_data = out.data();

              constexpr int out_dim =  no_of_unique<Idx0...,Idx1...>::value;

              std::array<int,out_dim> as = {};
              std::array<int,out_dim> idx = {};

              using nloops = loop_setter<
                        Index<Idx0...,Idx1...>,
                        Tensor<T,Rest0...,Rest1...>,
                        typename std_ext::make_index_sequence<out_dim>::type>;
              using dims_type = typename nloops::dims_type;

              RecursiveCartesianDispatcher<Index<Idx0...>,Index<Idx1...>,
                Tensor<T,Rest0...>,Tensor<T,Rest1...>,dims_type>::Do(a_data,b_data,out_data,as,idx);

              return out;
          }
#endif

};


// Specialisation for outer product case
template<size_t ... Idx0, size_t ... Idx1>
struct extractor_contract_2<Index<Idx0...>, Index<Idx1...>,
          typename std::enable_if<no_of_unique<Idx0...,Idx1...>::value==sizeof...(Idx0)+sizeof...(Idx1)>::type> {

    template<typename T, size_t ... Rest0, size_t ... Rest1>
          static Tensor<T,Rest0...,Rest1...>
          FASTOR_INLINE contract_impl(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) {
              Tensor<T,Rest0...,Rest1...> out;
              _dyadic<T,pack_prod<Rest0...>::value, pack_prod<Rest1...>::value>(a.data(),b.data(),out.data());
              return out;
          }
};





template<class Index_I, class Index_J,
         typename T, size_t ... Rest0, size_t ... Rest1>
FASTOR_INLINE
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b)
-> decltype(extractor_contract_2<Index_I,Index_J>::contract_impl(a,b)) {
    return extractor_contract_2<Index_I,Index_J>::contract_impl(a,b);
}

}

#endif // CONTRACTION_H

