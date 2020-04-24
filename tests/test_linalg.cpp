#include <Fastor/Fastor.h>

using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5
#define HugeTol 1e-2


template<typename T>
void test_linalg() {

    // 2D
    {
        Tensor<T,2,2> t1; t1.iota(5);
        FASTOR_EXIT_ASSERT(std::abs(determinant(t1)+2.0)< BigTol);
        FASTOR_EXIT_ASSERT(std::abs(trace(t1)-13)< Tol);
        FASTOR_EXIT_ASSERT(std::abs(norm(matmul(t1,t1)) - 173.4646938) < BigTol);
        // FASTOR_EXIT_ASSERT(std::abs(norm(matmul(t1-0,t1+t1-t1)) - 173.4646938) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(cofactor(t1)) - 13.1909059) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(transpose(adjoint(t1))) - 13.1909059) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(det(t1)+2.)< Tol);
        FASTOR_EXIT_ASSERT(std::abs(trace(t1) - 13)< Tol);
        // FASTOR_EXIT_ASSERT(std::abs(norm(static_cast<Tensor<T,2,2>>(lmatmul(t1,t1))) - 173.4646938)< BigTol);
        // FASTOR_EXIT_ASSERT(std::abs(norm(static_cast<Tensor<T,2,2>>(lcofactor(t1))) - 13.1909059)< BigTol);
        // FASTOR_EXIT_ASSERT(std::abs(norm(static_cast<Tensor<T,2,2>>(ltranspose(ladjoint(t1)))) - 13.1909059)< BigTol);

        Tensor<T,2> t2; t2.iota(102);
        FASTOR_EXIT_ASSERT(std::abs(norm(outer(t2,t2)) - 21013.0) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(inner(t2,t2) - 21013.0) < Tol);
    }

    {
        Tensor<T,2,2> t1; t1.iota(5);
        FASTOR_EXIT_ASSERT(std::abs(norm(transpose(t1))-norm(t1))< HugeTol);
    }
    {
        Tensor<T,3,3> t1; t1.iota(5);
        FASTOR_EXIT_ASSERT(std::abs(norm(transpose(t1))-norm(t1))< HugeTol);
    }
    {
        Tensor<T,4,4> t1; t1.iota(5);
        FASTOR_EXIT_ASSERT(std::abs(norm(transpose(t1))-norm(t1))< HugeTol);
    }
    {
        Tensor<T,5,5> t1; t1.iota(5);
        FASTOR_EXIT_ASSERT(std::abs(norm(transpose(t1))-norm(t1))< HugeTol);
    }
    {
        Tensor<T,7,7> t1; t1.iota(5);
        FASTOR_EXIT_ASSERT(std::abs(norm(transpose(t1))-norm(t1))< HugeTol);
    }
    {
        Tensor<T,8,8> t1; t1.iota(5);
        FASTOR_EXIT_ASSERT(std::abs(norm(transpose(t1))-norm(t1))< HugeTol);
    }
    {
        Tensor<T,9,9> t1; t1.iota(5);
        FASTOR_EXIT_ASSERT(std::abs(norm(transpose(t1))-norm(t1))< HugeTol);
    }
    {
        Tensor<T,10,10> t1; t1.iota(5);
        FASTOR_EXIT_ASSERT(std::abs(norm(transpose(t1))-norm(t1))< HugeTol);
    }
    {
        Tensor<T,12,12> t1; t1.iota(5);
        FASTOR_EXIT_ASSERT(std::abs(norm(transpose(t1))-norm(t1))< HugeTol);
    }
    {
        Tensor<T,16,16> t1; t1.iota(5);
        FASTOR_EXIT_ASSERT(std::abs(norm(transpose(t1))-norm(t1))< HugeTol);
    }
    {
        Tensor<T,17,17> t1; t1.iota(5);
        FASTOR_EXIT_ASSERT(std::abs(norm(transpose(t1))-norm(t1))< HugeTol);
    }
    {
        Tensor<T,40,40> t1; t1.iota(5);
        FASTOR_EXIT_ASSERT(std::abs(norm(transpose(t1))-norm(t1))< HugeTol);
    }


    // 3D
    {
        Tensor<T,3,3> t1; t1.iota(0);
        FASTOR_EXIT_ASSERT(std::abs(determinant(t1))< Tol);
        FASTOR_EXIT_ASSERT(std::abs(trace(t1)-12)< Tol);
        FASTOR_EXIT_ASSERT(std::abs(norm(matmul(t1,t1)) - 187.637949) < BigTol);
        // FASTOR_EXIT_ASSERT(std::abs(norm(matmul(2*t1-t1,t1/2*2)) - 187.637949) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(cofactor(t1)) - 18) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(transpose(adjoint(t1))) - 18) < BigTol);

        // FASTOR_EXIT_ASSERT(std::abs(norm(static_cast<Tensor<T>>(ldeterminant(t1))))< Tol);
        // FASTOR_EXIT_ASSERT(std::abs(norm(static_cast<Tensor<T>>(ltrace(t1))) - 12)< Tol);
        // FASTOR_EXIT_ASSERT(std::abs(norm(static_cast<Tensor<T,3,3>>(lmatmul(t1,t1))) - 187.637949)< BigTol);
        // FASTOR_EXIT_ASSERT(std::abs(norm(static_cast<Tensor<T,3,3>>(lcofactor(t1))) - 18)< BigTol);
        // FASTOR_EXIT_ASSERT(std::abs(norm(static_cast<Tensor<T,3,3>>(ltranspose(ladjoint(t1)))) - 18)< BigTol);

        Tensor<T,3> t2; t2.iota(102);
        FASTOR_EXIT_ASSERT(std::abs(norm(outer(t2,t2)) - 31829.0) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(inner(t2,t2) - 31829.0) < Tol);
    }

    // Misc
    {
        Tensor<T,2,2> a0; a0.iota(3);
        a0 = a0 + transpose(a0);
        FASTOR_EXIT_ASSERT(a0.is_symmetric(Tol));

        Tensor<T,3,3> a1; a1.iota(5);
        a1 = a1 + transpose(a1);
        FASTOR_EXIT_ASSERT(a1.is_symmetric(Tol));

        Tensor<T,4,4> a2; a2.iota(55);
        a2 = 0.5*(a2 + transpose(a2))+1;
        FASTOR_EXIT_ASSERT(a2.is_symmetric(Tol));

        FASTOR_EXIT_ASSERT(a0.is_equal(a0));
        FASTOR_EXIT_ASSERT(a1.is_equal(a1,Tol));
        FASTOR_EXIT_ASSERT(a2.is_equal(a2,BigTol));
    }

    print(FGRN(BOLD("All tests passed successfully")));

}

int main() {

    print(FBLU(BOLD("Testing tensor algebra routines: single precision")));
    test_linalg<float>();
    print(FBLU(BOLD("Testing tensor algebra routines: double precision")));
    test_linalg<double>();

    return 0;
}

