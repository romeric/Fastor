#include <Fastor/Fastor.h>

using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5
#define HugeTol 1e-2

template<typename T, size_t M, size_t N>
Tensor<T,N,M> transpose_ref(const Tensor<T,M,N>& a) {
    Tensor<T,N,M> out;
    for (size_t i=0; i<M; ++i) {
        for (size_t j=0; j<N; ++j) {
            out(j,i) = a(i,j);
        }
    }
    return out;
}

template<typename T, size_t M, size_t N>
void TEST_TRANSPOSE(Tensor<T,M,N>& a) {

    Tensor<T,N,M> b1 = transpose_ref(a);
    Tensor<T,N,M> b2 = transpose(a);
    Tensor<T,N,M> b3 = trans(a);

    for (size_t i=0; i<N; ++i) {
        for (size_t j=0; j<M; ++j) {
            FASTOR_EXIT_ASSERT(std::abs(b1(i,j) - b2(i,j)) < Tol);
            FASTOR_EXIT_ASSERT(std::abs(b1(i,j) - b3(i,j)) < Tol);
        }
    }

    FASTOR_EXIT_ASSERT(std::abs(norm(transpose(a))-norm(a))< HugeTol);
    FASTOR_EXIT_ASSERT(std::abs(norm(trans(a))-norm(a))< HugeTol);
}


template<typename T>
void test_transpose() {

    {
        Tensor<T,2,2> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,3,3> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,4,4> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,5,5> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,7,7> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,8,8> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,9,9> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,10,10> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,12,12> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,16,16> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,17,17> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,20,20> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,24,24> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,40,40> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }

    // non-square
    {
        Tensor<T,2,3> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,3,4> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,4,5> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,5,6> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,6,7> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }
    {
        Tensor<T,17,29> t1; t1.iota(5);
        TEST_TRANSPOSE(t1);
    }

    // transpose expressions
    {
        Tensor<T,2,3> t1; t1.iota(5);
        FASTOR_EXIT_ASSERT(std::abs(sum(transpose(t1 + 0)) - sum(t1)) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(trans(t1 + 0))     - sum(t1)) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(sum(transpose(t1 + t1*2 - t1 - t1)) - sum(t1)) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(trans(t1 + t1*2 - t1 - t1))     - sum(t1)) < BigTol);
    }

    print(FGRN(BOLD("All tests passed successfully")));

}

int main() {

    print(FBLU(BOLD("Testing tensor transpose routine: single precision")));
    test_transpose<float>();
    print(FBLU(BOLD("Testing tensor transpose routine: double precision")));
    test_transpose<double>();

    return 0;
}

