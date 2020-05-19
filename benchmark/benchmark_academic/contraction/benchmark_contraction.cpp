#include <Fastor/Fastor.h>
using namespace Fastor;

template<size_t NITER, typename T, class Index_I, class Index_J, class Tensor0, class Tensor1>
void iterate_over_scalar(const Tensor0 &a, const Tensor1& b) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        auto out = contraction<Index_I,Index_J>(a,b);
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
    const std::string filename = "SIMD_products_results";
    write(filename,elapsed_time);
}


int main() {

    print(FBLU(BOLD("Running SIMD benchmark for non-isomorphic tensor products\n")));
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
    run<100UL,float,Index<0,1,2,3>,Index<0,1,2,4>>(af4,bf4);
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
