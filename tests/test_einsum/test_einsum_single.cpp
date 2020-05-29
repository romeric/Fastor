#include <Fastor/Fastor.h>
using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5
#define HugeTol 1e-2


template<typename T>
void run() {

    using std::abs;
    enum {i,j,k,l,m,n};

    // check trace
    {
        Tensor<T,2,2> a; a.iota(3);

        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<i,i>>(a) - trace(a))) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<j,j>>(a) - trace(a))) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<k,k>>(a) - trace(a))) < Tol );

        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<0,0>>(a) - trace(a))) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<1,1>>(a) - trace(a))) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<2,2>>(a) - trace(a))) < Tol );

        FASTOR_EXIT_ASSERT(std::abs(sum(contraction<Index<2,2>>(a) - trace(a))) < Tol );
    }

    {
        Tensor<T,3,3> a; a.iota(3);

        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<i,i>>(a) - trace(a))) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<j,j>>(a) - trace(a))) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<k,k>>(a) - trace(a))) < Tol );

        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<0,0>>(a) - trace(a))) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<1,1>>(a) - trace(a))) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<2,2>>(a) - trace(a))) < Tol );

        FASTOR_EXIT_ASSERT(std::abs(sum(contraction<Index<2,2>>(a) - trace(a))) < Tol );
    }

    {
        Tensor<T,10,10> a; a.iota(3);

        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<i,i>>(a) - trace(a))) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<j,j>>(a) - trace(a))) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<k,k>>(a) - trace(a))) < Tol );

        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<0,0>>(a) - trace(a))) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<1,1>>(a) - trace(a))) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<2,2>>(a) - trace(a))) < Tol );

        FASTOR_EXIT_ASSERT(std::abs(sum(contraction<Index<2,2>>(a) - trace(a))) < Tol );
    }

    // expressions
    {
        Tensor<T,10,10> a; a.iota(3);

        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<i,i>>(a+0) - trace(a))) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<j,j>>(a+0) - trace(a))) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<k,k>>(a+0) - trace(a))) < Tol );

        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<0,0>>(a+0) - trace(a))) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<1,1>>(a+0) - trace(a))) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(sum(einsum<Index<2,2>>(a+0) - trace(a))) < Tol );

        FASTOR_EXIT_ASSERT(std::abs(sum(contraction<Index<2,2>>(a+1) - trace(a+1))) < Tol );
    }

    {
        Tensor<T,3,3,3,3> a; a.iota(4);
        auto b1 = einsum<Index<i,k,j,k>>(a);
        FASTOR_EXIT_ASSERT(std::abs(b1(0,0)  -   42.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b1(0,1)  -   51.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b1(1,2)  -  141.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b1(2,2)  -  222.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b1.sum() - 1188.0) < Tol );

        auto b2 = einsum<Index<i,i,j,j>>(a);
        FASTOR_EXIT_ASSERT(std::abs(b2.sum() - 396.0) < Tol );

        auto b3 = einsum<Index<i,j,i,j>>(a);
        FASTOR_EXIT_ASSERT(std::abs(b3.sum() - 396.0) < Tol );

        auto b4 = einsum<Index<i,i,j,k>>(a);
        FASTOR_EXIT_ASSERT(std::abs(b4(0,0)  -  120.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b4(0,1)  -  123.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b4(1,2)  -  135.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b4(2,2)  -  144.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b4.sum() - 1188.0) < Tol );

        auto b5 = einsum<Index<i,i,j,k>>(a+a);
        FASTOR_EXIT_ASSERT(std::abs(b5(0,0)  -  240.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b5(0,1)  -  246.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b5(1,2)  -  270.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b5(2,2)  -  288.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b5.sum() - 2376.0) < Tol );
    }

    // contraction
    {
        Tensor<T,3,3,3,3> a; a.iota(4);
        auto b1 = contraction<Index<i,k,j,k>>(a);
        FASTOR_EXIT_ASSERT(std::abs(b1(0,0)  -   42.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b1(0,1)  -   51.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b1(1,2)  -  141.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b1(2,2)  -  222.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b1.sum() - 1188.0) < Tol );

        auto b2 = contraction<Index<i,i,j,j>>(a);
        FASTOR_EXIT_ASSERT(std::abs(b2.sum() - 396.0) < Tol );

        auto b3 = contraction<Index<i,j,i,j>>(a);
        FASTOR_EXIT_ASSERT(std::abs(b3.sum() - 396.0) < Tol );

        auto b4 = contraction<Index<i,i,j,k>>(a);
        FASTOR_EXIT_ASSERT(std::abs(b4(0,0)  -  120.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b4(0,1)  -  123.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b4(1,2)  -  135.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b4(2,2)  -  144.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b4.sum() - 1188.0) < Tol );

        auto b5 = contraction<Index<i,i,j,k>>(a+a);
        FASTOR_EXIT_ASSERT(std::abs(b5(0,0)  -  240.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b5(0,1)  -  246.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b5(1,2)  -  270.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b5(2,2)  -  288.0) < Tol );
        FASTOR_EXIT_ASSERT(std::abs(b5.sum() - 2376.0) < Tol );
    }

    print(FGRN(BOLD("All tests passed successfully")));
}

int main() {

    print(FBLU(BOLD("Testing all single einsum: single precision")));
    run<float>();
    print(FBLU(BOLD("Testing all single einsum: double precision")));
    run<double>();

    return 0;
}
