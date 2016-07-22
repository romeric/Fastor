#include <Fastor.h>

using namespace Fastor;

namespace scalar_version {

template<class T, size_t...Rest>
struct contract_vec_impl;

template<class T, size_t...Rest>
struct contract_vec_impl<Tensor<T,Rest...>> {
    static const size_t value = get_value<sizeof...(Rest),Rest...>::value;
};

template<class T, class U>
struct extractor_contract_2 {};

template<size_t ... Idx0, size_t ... Idx1>
struct extractor_contract_2<Index<Idx0...>, Index<Idx1...>> {

      template<typename T, size_t ... Rest0, size_t ... Rest1>
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
          constexpr size_t maxes_a[a_dim] = {Rest0...};
          constexpr size_t maxes_b[b_dim] = {Rest1...};

          constexpr std::array<int,a_dim> ma = {Idx0...};
          constexpr std::array<int,b_dim> mb = {Idx1...};

          std::vector<int> maxes_out;
          std::vector<int> idx_a(a_dim), idx_b, idx_tmp, idx_out(OutTensor::Dimension);
          std::iota(idx_a.begin(),idx_a.end(),0);
          {
              for (int i=0; i<a_dim; ++i) {
                  maxes_out.push_back(maxes_a[i]);
                  idx_tmp.push_back(i);
              }
              int min_dim = a_dim;
              if (min_dim>static_cast<int>(b_dim)) min_dim = b_dim;
              int counter = idx_tmp.size();
              for (int i=0; i<min_dim; ++i) {
                  if (ma[i]!=mb[i]) {
                      maxes_out.push_back(maxes_b[i]);
                      idx_tmp.push_back(counter);
                      counter++;
                  }
              }

              // find where idx_b is in maxes_out
              counter = a_dim;
              for (int i=0; i<b_dim; ++i) {
                  if (ma[i]!=mb[i]) {
                      idx_b.push_back(idx_tmp[counter]);
                      counter++;
                  }
                  else
                      idx_b.push_back(idx_tmp[i]);
              }

              std::vector<int> inn;
              for (int i=0; i<a_dim; ++i) {
                  if (ma[i]==mb[i]) {
                      inn.push_back(i);
                  }
              }
              std::vector<int>::iterator itr;
              itr = std::set_difference(idx_tmp.begin(),idx_tmp.end(),inn.begin(),inn.end(),idx_out.begin());
          }
          int out_dim = maxes_out.size();

          std::array<int,a_dim> products_a; products_a[0]=0;
          for (int j=a_dim-1; j>0; --j) {
              int num = maxes_a[a_dim-1];
              for (int k=0; k<j-1; ++k) {
                  num *= maxes_a[a_dim-1-k-1];
              }
              products_a[j] = num;
          }
          std::array<int,b_dim> products_b; products_b[0]=0;
          for (int j=b_dim-1; j>0; --j) {
              int num = maxes_b[b_dim-1];
              for (int k=0; k<j-1; ++k) {
                  num *= maxes_b[b_dim-1-k-1];
              }
              products_b[j] = num;
          }

          std::vector<int> products_out(idx_out.size()); products_out[0]=0;
          for (int j=idx_out.size()-1; j>0; --j) {
              int num = maxes_out[idx_out[idx_out.size()-1]];
              for (int k=0; k<j-1; ++k) {
                  num *= maxes_out[idx_out[idx_out.size()-1-k-1]];
              }
              products_out[j] = num;
          }

          std::reverse(products_a.begin(),products_a.end());
          std::reverse(products_b.begin(),products_b.end());
          std::reverse(products_out.begin(),products_out.end());

          int as[out_dim];
          std::fill(as,as+out_dim,0);
          int it;


          int total = 1;
          {
              for (int i=0; i<maxes_out.size(); ++i) {
                  total *=maxes_out[i];
              }
          }

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

                out_data[index_out] += a_data[index_a]*b_data[index_b];
          }
          return out;
      }

};

template<class Index_I, class Index_J,
         typename T, size_t ... Rest0, size_t ... Rest1>
auto contraction(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b)
-> decltype(extractor_contract_2<Index_I,Index_J>::contract_impl(a,b)) {
    return extractor_contract_2<Index_I,Index_J>::contract_impl(a,b);
}

}

template<size_t NITER, typename T, class Index_I, class Index_J, class Tensor0, class Tensor1>
void iterate_over_scalar(const Tensor0 &a, const Tensor1& b) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        auto out = scalar_version::contraction<Index_I,Index_J>(a,b);
        unused(a); unused(b); unused(out);
    }
}

template<size_t NITER, typename T, class Index_I, class Index_J, class Tensor0, class Tensor1>
void run(const Tensor0 &a, const Tensor1& b) {

    double elapsed_time; size_t cycles;
    std::tie(elapsed_time,cycles) = rtimeit(static_cast<void (*)(const Tensor0 &, 
        const Tensor1&)>(&iterate_over_scalar<NITER,T,Index_I,Index_J,Tensor0,Tensor1>),a,b);
    print(elapsed_time);

    // write
    const std::string filename = "Scalar_products_results";
    write(filename,elapsed_time);
}


int main() {

    print("Running SIMD benchmark\n");
    print("1 Index contraction: single precision SSE");
    Tensor<float,2,3,4,5,2> af0; Tensor<float,2,3,3,4> bf0; af0.arange(0); bf0.arange(1);
    run<100UL,float,Index<0,1,2,3,7>,Index<4,1,5,6>>(af0,bf0);
    print("1 Index contraction: double precision SSE");
    Tensor<double,2,3,4,5,2> ad0; Tensor<double,2,3,3,2> bd0; ad0.arange(0); bd0.arange(1);
    run<100UL,double,Index<0,1,2,3,7>,Index<4,1,5,6>>(ad0,bd0);

    print("1 Index contraction: single precision AVX");
    Tensor<float,2,3,4,5,2> af1; Tensor<float,2,3,2,8> bf1; af0.arange(0); bf1.arange(1);
    run<100UL,float,Index<0,1,2,3,7>,Index<4,1,5,6>>(af1,bf1);
    print("1 Index contraction: double precision AVX");
    Tensor<double,2,3,4,5,2> ad1; Tensor<double,2,3,2,8> bd1; ad0.arange(0); bd1.arange(1);
    run<100UL,double,Index<0,1,2,3,7>,Index<4,1,5,6>>(ad1,bd1);

    print("2 Index contraction: single precision SSE");
    Tensor<float,3,4,5,8> af2; Tensor<float,3,4,4> bf2;   af2.arange(0); bf2.arange(1);
    run<100UL,float,Index<0,1,2,3>,Index<0,1,5>>(af2,bf2);
    print("2 Index contraction: double precision SSE");
    Tensor<double,3,4,5,8> ad2; Tensor<double,3,4,2> bd2;   ad2.arange(0); bd2.arange(1);
    run<100UL,double,Index<0,1,2,3>,Index<0,1,5>>(ad2,bd2);

    print("2 Index contraction: single precision AVX");
    Tensor<float,3,4,5,8> af3; Tensor<float,3,4,8> bf3;   af3.arange(0); bf3.arange(1);
    run<100UL,float,Index<0,1,2,3>,Index<0,1,5>>(af3,bf3);
    print("2 Index contraction: double precision AVX");
    Tensor<double,3,4,5,8> ad3; Tensor<double,3,4,8> bd3;   ad3.arange(0); bd3.arange(1);
    run<100UL,double,Index<0,1,2,3>,Index<0,1,5>>(ad3,bd3);

    print("3 Index contraction: single precision SSE");
    Tensor<float,3,4,2,8> af4; Tensor<float,3,4,2,4> bf4;   af4.arange(0); bf4.arange(1);
    run<100UL,float,Index<0,1,2,3>,Index<0,1,2,4>>(af4,bf4);
    print("3 Index contraction: double precision SSE");
    Tensor<double,3,4,2,8> ad4; Tensor<double,3,4,2,2> bd4;   ad4.arange(0); bd4.arange(1);
    run<100UL,double,Index<0,1,2,3>,Index<0,1,2,4>>(ad4,bd4);

    print("3 Index contraction: single precision AVX");
    Tensor<float,3,4,2,8> af5; Tensor<float,3,4,2,8> bf5;   af5.arange(0); bf5.arange(1);
    run<100UL,float,Index<0,1,2,3>,Index<0,1,2,4>>(af5,bf5);
    print("3 Index contraction: double precision AVX");
    Tensor<double,3,4,2,8> ad5; Tensor<double,3,4,2,8> bd5;   ad5.arange(0); bd5.arange(1);
    run<100UL,double,Index<0,1,2,3>,Index<0,1,2,4>>(ad5,bd5);

    print("8 Index contraction: single precision SSE");
    Tensor<float,2,3,2,3,2,3,2,3,2> af6; Tensor<float,2,3,2,3,2,3,2,3,4> bf6;   af6.arange(0); bf6.arange(1);
    run<100UL,float,Index<0,1,2,3,4,5,6,7,8>,Index<0,1,2,3,4,5,6,7,9>>(af6,bf6);
    print("8 Index contraction: double precision SSE");
    Tensor<double,2,3,2,3,2,3,2,3,2> ad6; Tensor<double,2,3,2,3,2,3,2,3,2> bd6;   ad6.arange(0); bd6.arange(1);
    run<100UL,double,Index<0,1,2,3,4,5,6,7,8>,Index<0,1,2,3,4,5,6,7,9>>(ad6,bd6);

    print("8 Index contraction: single precision SSE");
    Tensor<float,2,3,2,3,2,3,2,3,2> af7; Tensor<float,2,3,2,3,2,3,2,3,8> bf7;   af7.arange(0); bf7.arange(1);
    run<100UL,float,Index<0,1,2,3,4,5,6,7,8>,Index<0,1,2,3,4,5,6,7,9>>(af7,bf7);
    print("8 Index contraction: double precision SSE");
    Tensor<double,2,3,2,3,2,3,2,3,2> ad7; Tensor<double,2,3,2,3,2,3,2,3,4> bd7;   ad7.arange(0); bd7.arange(1);
    run<100UL,double,Index<0,1,2,3,4,5,6,7,8>,Index<0,1,2,3,4,5,6,7,9>>(ad7,bd7);

    return 0;
}