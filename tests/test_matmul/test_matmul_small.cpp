#include <Fastor/Fastor.h>
using namespace Fastor;

#define Tol 1e-09
#define BigTol 1e-5

template<typename T, size_t M, size_t K, size_t N>
Tensor<T,M,N> matmul_ref(const Tensor<T,M,K> &a, const Tensor<T,K,N> &b) {

    Tensor<T,M,N> out; out.zeros();
    for (size_t i=0; i<M; ++i) {
        for (size_t j=0; j<K; ++j) {
            for (size_t k=0; k<N; ++k) {
                out(i,k) += a(i,j)*b(j,k);
            }
        }
    }

    return out;
}


template<typename T, size_t M, size_t K, size_t N>
void test() {

    Tensor<T,M,K> a; a.iota(1);
    Tensor<T,K,N> b; b.iota(4);

    auto c1 = matmul_ref(a,b);
    Tensor<T,M,N> c3;
    internal::_matmul_base_masked<T,M,K,N>(a.data(),b.data(),c3.data());
    Tensor<T,M,N> c4;
    internal::_matmul_mk_smalln<T,M,K,N>(a.data(),b.data(),c4.data());

    // print(c1,"\n",c3);
    // print(std::abs(sum(c1-c3)));
    // print(std::abs(sum(c1-c4)));
    FASTOR_DOES_CHECK_PASS(std::abs(sum(c1-c3)) < BigTol);
    FASTOR_DOES_CHECK_PASS(std::abs(sum(c1-c4)) < BigTol);
}



template<typename T, size_t incr, size_t K, size_t N>
void run() {

    test<T,1+incr,K,N>();
    test<T,2+incr,K,N>();
    test<T,3+incr,K,N>();
    test<T,4+incr,K,N>();
    test<T,5+incr,K,N>();
    test<T,6+incr,K,N>();
    test<T,7+incr,K,N>();
    test<T,8+incr,K,N>();
    test<T,9+incr,K,N>();
    test<T,10+incr,K,N>();
}

template<typename T>
void runs() {

    // covers mk_smalln for avx doubles
    run<T,0,3,2>();
    run<T,0,3,3>();
    run<T,0,3,4>();
    run<T,0,3,7>();
    run<T,0,3,8>();
    run<T,0,2,11>();
    run<T,0,2,12>();
    run<T,0,2,15>();
    run<T,0,2,16>();
    run<T,0,2,19>();
    run<T,0,2,20>();

    // the rest
    run<T,0,2,23>();
    run<T,0,2,24>();
    run<T,0,2,40>();
    run<T,0,2,43>();

    print(FGRN(BOLD("All tests passed successfully")));
}





int main() {


    // print(FBLU(BOLD("Testing tensor matmul small: single precision")));
    // runs<float>();
    print(FBLU(BOLD("Testing tensor matmul small: double precision")));
    runs<double>();



    return 0;
}
