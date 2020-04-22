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
void SINGLE_TEST(const Tensor<T,M,K> &a, const Tensor<T,K,N> &b, T tol=Tol) {

    auto c1 = matmul_ref(a,b);
    Tensor<T,M,N> c2 = matmul(a,b);
    Tensor<T,M,N> c3;
    internal::_matmul_base<T,M,K,N>(a.data(),b.data(),c3.data());

    // print(c1,"\n",c2);
    // print(std::abs(sum(c1-c2)));
    // print(std::abs(sum(c1-c3)));
    FASTOR_EXIT_ASSERT(std::abs(sum(c1-c2)) < tol);
    FASTOR_EXIT_ASSERT(std::abs(sum(c1-c3)) < tol);
}


template<typename T, size_t K, size_t N>
void SINGLE_TEST(const Tensor<T,K> &a, const Tensor<T,K,N> &b) {

    Tensor<T,N> c1 = matmul_ref<T,1,K,N>(a,b);
    auto c2 = matmul(a,b);
    Tensor<T,N> c3;
    internal::_matmul_base<T,1,K,N>(a.data(),b.data(),c3.data());

    FASTOR_EXIT_ASSERT(std::abs(sum(c1-c2)) < Tol);
    FASTOR_EXIT_ASSERT(std::abs(sum(c1-c3)) < Tol);
}


template<typename T, size_t M, size_t K>
void SINGLE_TEST(const Tensor<T,M,K> &a, const Tensor<T,K> &b) {

    auto c1 = matmul_ref<T,M,K,1>(a,b);
    auto c2 = matmul(a,b);
    Tensor<T,M,1> c3;
    internal::_matmul_base<T,M,K,1>(a.data(),b.data(),c3.data());

    FASTOR_EXIT_ASSERT(std::abs(sum(c1-c2)) < Tol);
    FASTOR_EXIT_ASSERT(std::abs(sum(c1-c3)) < Tol);
}



template<typename T>
void run() {

    {
        Tensor<T,2,2> a; a.iota(99);
        Tensor<T,2,2> b; b.iota(21);
        SINGLE_TEST(a,b);
        a(1,1) = -2.5;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,3,3> a; a.iota(9);
        Tensor<T,3,3> b; b.iota(-4);
        SINGLE_TEST(a,b);
        a(1,1) = -2.5;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,4,4> a; a.iota(9);
        Tensor<T,4,4> b; b.iota(-4);
        SINGLE_TEST(a,b);
        a(1,1) = -2.5;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,5,5> a; a.iota(9);
        Tensor<T,5,5> b; b.iota(-4);
        SINGLE_TEST(a,b);
        a(1,1) = -2;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,8,8> a; a.iota(9);
        Tensor<T,8,8> b; b.iota(-4);
        SINGLE_TEST(a,b);
        a(1,1) = -2;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,16,16> a; a.iota(9);
        Tensor<T,16,16> b; b.iota(-4);
        SINGLE_TEST(a,b);
        a(1,1) = -2;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,32,32> a; a.iota(9);
        Tensor<T,32,32> b; b.iota(-4);
        SINGLE_TEST(a,b);
        a(1,1) = -2;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,8,3> a; a.iota(9);
        Tensor<T,3,12> b; b.iota(-4);
        SINGLE_TEST(a,b);
        a(1,1) = -2;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,12,3> a; a.iota(9);
        Tensor<T,3,8> b; b.iota(-4);
        SINGLE_TEST(a,b);
        a(1,1) = -2;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,12,3> a; a.iota(9);
        Tensor<T,3,20> b; b.iota(-4);
        SINGLE_TEST(a,b);
        a(1,1) = -2;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,20,3> a; a.iota(9);
        Tensor<T,3,12> b; b.iota(-4);
        SINGLE_TEST(a,b);
        a(1,1) = -2;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,17,17> a; a.iota(9);
        Tensor<T,17,19> b; b.iota(-4);
        SINGLE_TEST(a,b);
        a(1,1) = -2;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,5,2> a; a.iota(9);
        Tensor<T,2,3> b; b.iota(-4);
        SINGLE_TEST(a,b);
        a(1,1) = -2;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,2,4> a; a.iota(9);
        Tensor<T,4,8> b; b.iota(-4);
        SINGLE_TEST(a,b);
        a(1,1) = -2;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,2,20> a; a.iota(-5);
        Tensor<T,20,8> b; b.iota(0);
        SINGLE_TEST(a,b);
        a(1,1) = -2;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,2,18> a; a.iota(-5);
        Tensor<T,18,2> b; b.iota(0);
        SINGLE_TEST(a,b);
        a(1,1) = -2;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,3,20> a; a.iota(-5);
        Tensor<T,20,3> b; b.iota(0);
        SINGLE_TEST(a,b);
        a(1,1) = -2;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,4,20> a; a.iota(-5);
        Tensor<T,20,4> b; b.iota(0);
        SINGLE_TEST(a,b);
        a(1,10) = -2;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,3,20> a; a.iota(-5);
        Tensor<T,20,4> b; b.iota(0);
        SINGLE_TEST(a,b);
        a(1,1) = -2;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,8,51> a; a.iota(-5);
        Tensor<T,51,9> b; b.iota(0);
        SINGLE_TEST(a,b);
        a(1,1) = -2;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,2,3> a; a.iota(-5);
        Tensor<T,3,12> b; b.iota(0);
        SINGLE_TEST(a,b);
        a(1,1) = -2;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,2,3> a; a.iota(-5);
        Tensor<T,3,24> b; b.iota(0);
        SINGLE_TEST(a,b);
        a(1,1) = -2;
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,7,1> a; a.iota(-5);
        Tensor<T,1,7> b; b.iota(0);
        SINGLE_TEST(a,b);
        a(3,0) = -200;
        SINGLE_TEST(a,b);
    }

    // matrix-vector and vector-matrix
    {
        Tensor<T,1,2> a; a.iota(-5.1);
        Tensor<T,2,2> b; b.iota(2.0);
        SINGLE_TEST(a,b,(T)BigTol);
    }

    {
        Tensor<T,1,3> a; a.iota(-5.1);
        Tensor<T,3,3> b; b.iota(2.0);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,1,4> a; a.iota(-5.1);
        Tensor<T,4,4> b; b.iota(2.0);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,2,2> a; a.iota(-5.1);
        Tensor<T,2,1> b; b.iota(2.0);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,3,3> a; a.iota(-5.1);
        Tensor<T,3,1> b; b.iota(2.0);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,4,4> a; a.iota(-5);
        Tensor<T,4,1> b; b.iota(2.0);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,2,5> a; a.iota(-5);
        Tensor<T,5,1> b; b.iota(2.0);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,3,17> a; a.iota(-5);
        Tensor<T,17,1> b; b.iota(2.0);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,4,17> a; a.iota(-5);
        Tensor<T,17,1> b; b.iota(2.0);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,5,17> a; a.iota(-5);
        Tensor<T,17,1> b; b.iota(2.0);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,6,17> a; a.iota(-5);
        Tensor<T,17,1> b; b.iota(2.0);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,7,17> a; a.iota(-5);
        Tensor<T,17,1> b; b.iota(2.0);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,8,17> a; a.iota(-5);
        Tensor<T,17,1> b; b.iota(2.0);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,9,17> a; a.iota(-5);
        Tensor<T,17,1> b; b.iota(2.0);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,5,8> a; a.iota(-5);
        Tensor<T,8,1> b; b.iota(2.0);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,25,17> a; a.iota(-5);
        Tensor<T,17,1> b; b.iota(2.0);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,1,2> a; a.iota(-5);
        Tensor<T,2,4> b; b.iota(0);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,2> a; a.iota(-5);
        Tensor<T,2,4> b; b.iota(0);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,1,2> a; a.iota(-5);
        Tensor<T,2,1> b; b.iota(0);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,1,20> a; a.iota(-5);
        Tensor<T,20> b; b.iota(0);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,1,19> a; a.iota(-50);
        Tensor<T,19> b; b.iota(50);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,1,21> a; a.iota(-50);
        Tensor<T,21,1> b; b.iota(50);
        SINGLE_TEST(a,b);
    }

    {
        Tensor<T,1,63> a; a.iota(-50);
        Tensor<T,63,1> b; b.iota(50);
        SINGLE_TEST(a,b);
    }

    print(FGRN(BOLD("All tests passed successfully")));
}





int main() {


    print(FBLU(BOLD("Testing tensor contraction: single precision")));
    run<float>();
    print(FBLU(BOLD("Testing tensor contraction: double precision")));
    run<double>();



    return 0;
}
