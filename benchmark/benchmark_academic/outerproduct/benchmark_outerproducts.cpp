#include <Fastor.h>

using namespace Fastor;

// 2D
// #define NITER 100000UL
// 3D & 4D
// #define NITER 10000UL
// 5D
// #define NITER 1000UL
// 12D
// #define NITER 10UL





// namespace scalar_version {

// template<template<typename,size_t...Rest0> class Tensor0,
//          template<typename,size_t...Rest1> class Tensor1,
//          typename T, size_t ... Rest0, size_t ... Rest1>
// FASTOR_INLINE Tensor<T,Rest0...,Rest1...> outer(const Tensor0<T,Rest0...> &a, const Tensor1<T,Rest1...> &b) {
//     Tensor<T,Rest0...,Rest1...> out;
//     out.zeros();
//     T *a_data = a.data();
//     T *b_data = b.data();
//     T *out_data = out.data();

//     constexpr int a_dim = sizeof...(Rest0);
//     constexpr int b_dim = sizeof...(Rest1);
//     constexpr int out_dim = a_dim+b_dim;
//     constexpr std::array<int,out_dim> maxes_out = {Rest0...,Rest1...};

//     constexpr std::array<size_t,a_dim> products_a = nprods<Index<Rest0...>,typename std_ext::make_index_sequence<a_dim>::type>::values;
//     constexpr std::array<size_t,b_dim> products_b = nprods<Index<Rest1...>,typename std_ext::make_index_sequence<b_dim>::type>::values;
//     constexpr std::array<size_t,out_dim> products_out = nprods<Index<Rest0...,Rest1...>,typename std_ext::make_index_sequence<out_dim>::type>::values;


//     int as[out_dim];
//     std::fill(as,as+out_dim,0);
//     int it;

//     constexpr int stride = 1;
//     constexpr int total = prod<Rest0...,Rest1...>::value;
//     for (int i = 0; i < total; i+=stride) {
//         int remaining = total;
//         for (int n = 0; n < out_dim; ++n) {
//             remaining /= maxes_out[n];
//             as[n] = ( i / remaining ) % maxes_out[n];
//         }

//         int index_a = as[a_dim-1];
//         for(it = 0; it< a_dim; it++) {
//             index_a += products_a[it]*as[it];
//         }
//         int index_b = as[out_dim-1];
//         for(it = a_dim; it< out_dim; it++) {
//             index_b += products_b[it-a_dim]*as[it];
//         }
//         int index_out = as[out_dim-1];
//         for(it = 0; it< out_dim; it++) {
//             index_out += products_out[it]*as[it];
//         }

//        out_data[index_out] = a_data[index_a]*b_data[index_b];

//        // unused(as);
//     }

//     return out;
// }

// }

// template<typename T, size_t ... Rest>
// void iterate_over_scalar(const Tensor<T,Rest...> &a, const Tensor<T,Rest...>& b) {
//     size_t iter = 0;
//     for (; iter<NITER; ++iter) {
//         auto out = scalar_version::outer(a,b);
//         unused(a); unused(b); unused(out);
//     }
// }






template<size_t NITER, typename T, size_t ... Rest>
void iterate_over_fastor(const Tensor<T,Rest...> &a, const Tensor<T,Rest...>& b) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        auto out = Fastor::outer(a,b);
        unused(a); unused(b); unused(out);
        // further hack for gcc, seemingly  doesn't hurt performance of _crossproduct 
        // T *out_data = out.data();
        // out_data[1] += 1; 
    }
}

// template<typename T, size_t ... Rest>
// Tensor<T,Rest...,Rest...> iterate_over_fastor(const Tensor<T,Rest...> &a, const Tensor<T,Rest...>& b) {
//     size_t iter = 0;
//     Tensor<T,Rest...,Rest...> out;
//     for (; iter<NITER; ++iter) {
//         out = Fastor::outer(a,b);
//         unused(out);
//         // further hack for gcc, seemingly  doesn't hurt performance of _crossproduct 
//         T *out_data = out.data();
//         out_data[1] += 1; 
//     }
//     return out;
// }


template<size_t NITER, typename T, size_t ... Rest>
void run() {

    Tensor<T,Rest...> a, b;
    a.random(); b.random();
    // timeit(static_cast<void (*)(const Tensor<T,Rest...> &, const Tensor<T,Rest...>&)>(&iterate_over_fastor<T,Rest...>),a,b);
    // timeit(static_cast<Tensor<T,Rest...,Rest...> (*)(const Tensor<T,Rest...> &, const Tensor<T,Rest...>&)>(&iterate_over_fastor<T,Rest...>),a,b);

    // double elapsed_time0, elapsed_time1; size_t cycles0, cycles1;
    double elapsed_time; size_t cycles;
    std::tie(elapsed_time,cycles) = rtimeit(static_cast<void (*)(const Tensor<T,Rest...> &, 
        const Tensor<T,Rest...>&)>(&iterate_over_fastor<NITER,T,Rest...>),a,b);
    print(elapsed_time);
    // print(elapsed_time0/elapsed_time1);

    // write
    const std::string filename = "SIMD_products_results";
    std::array<size_t,sizeof...(Rest)> arr = {Rest...};
    write(filename,"Tensor<"+type_name<T>()+","+itoa(arr)+">"+" x "+"Tensor<"+type_name<T>()+","+itoa(arr)+">");
    write(filename,elapsed_time);
}


int main() {

    print("Running SIMD benchmark\n");
    print("2D outer products: single precision");
    run<100000UL,float,4,4>();
    run<100000UL,float,2,16>();
    print("2D outer products: double precision");
    run<100000UL,double,2,2>();
    run<100000UL,double,4,4>();

    print("3D outer products: single precision");
    run<10000UL,float,4,4,4>();
    run<10000UL,float,2,3,16>();
    print("3D outer products: double precision");
    run<10000UL,double,4,3,2>();
    run<10000UL,double,2,3,4>();

    print("4D outer products: single precision");
    run<1000UL,float,2,3,4,4>();
    run<1000UL,float,2,3,4,8>();
    print("4D outer products: double precision");
    run<1000UL,double,5,4,3,2>();
    run<1000UL,double,2,3,5,4>();

    print("5D outer products: single precision");
    run<1000UL,float,2,2,2,2,4>();
    run<1000UL,float,2,2,2,3,16>();
    print("8D outer products: double precision");
    run<1000UL,double,2,2,2,2,2>();
    run<1000UL,double,2,2,2,2,8>();

    print("6D outer products: single precision");
    run<100UL,float,2,2,2,2,2,4>();
    run<100UL,float,2,2,2,2,2,8>();
    print("6D outer products: double precision");
    run<100UL,double,2,2,2,2,2,2>();
    run<100UL,double,2,2,2,2,2,4>();

    print("8D outer products: single precision");
    run<10UL,float,2,2,2,2,2,2,2,4>();
    run<10UL,float,2,2,2,2,2,2,2,8>();
    print("8D outer products: double precision");
    run<10UL,double,2,2,2,2,2,2,2,2>();
    run<10UL,double,2,2,2,2,2,2,2,4>();

    return 0;
}