
// Turn off vectorisation for Fastor to generate scalar code
#define FASTOR_DONT_VECTORISE

#include <Fastor/Fastor.h>
using namespace Fastor;


template<size_t NITER, typename T, size_t ... Rest>
void iterate_over_scalar(const Tensor<T,Rest...> &a, const Tensor<T,Rest...>& b) {
    size_t iter = 0;
    for (; iter<NITER; ++iter) {
        auto out = outer(a,b);
        unused(a); unused(b); unused(out);
    }
}


template<size_t NITER, typename T, size_t ... Rest>
void run() {

    Tensor<T,Rest...> a, b;
    a.random(); b.random();

    double elapsed_time; size_t cycles;
    std::tie(elapsed_time,cycles) = rtimeit(static_cast<void (*)(const Tensor<T,Rest...> &, 
        const Tensor<T,Rest...>&)>(&iterate_over_scalar<NITER,T,Rest...>),a,b);
    print(elapsed_time);

    // write
    const std::string filename = "Scalar_products_results";
    std::array<size_t,sizeof...(Rest)> arr = {Rest...};
    write(filename,"Tensor<"+type_name<T>()+","+itoa(arr)+">"+" x "+"Tensor<"+type_name<T>()+","+itoa(arr)+">");
    write(filename,elapsed_time);
}


int main() {

    print(FBLU(BOLD("Running scalar benchmark for isomorphic tensor products\n")));
    print("2D outer products: single precision");
    run<100000UL,float,4,4>();
    run<100000UL,float,2,16>();
    print("2D outer products: double precision");
    run<100000UL,double,3,2>();
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
