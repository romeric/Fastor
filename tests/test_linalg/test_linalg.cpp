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
        FASTOR_EXIT_ASSERT(std::abs(determinant(t1)+2.0) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(trace(t1)-13) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(norm(matmul(t1,t1)) - 173.4646938) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(matmul(t1-0,t1+t1-t1)) - 173.4646938) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(cofactor(t1)) - 13.1909059) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(transpose(adjoint(t1))) - 13.1909059) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(det(t1)+2.) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(trace(t1) - 13) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(norm(t1 % t1) - 173.4646938) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(cof(t1)) - 13.1909059)< BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(trans(adj(t1))) - 13.1909059)< BigTol);

        Tensor<T,2> t2; t2.iota(102);
        FASTOR_EXIT_ASSERT(std::abs(norm(outer(t2,t2)) - 21013.0) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(inner(t2,t2) - 21013.0) < Tol);
    }

    // 3D
    {
        Tensor<T,3,3> t1; t1.iota(0);
        FASTOR_EXIT_ASSERT(std::abs(determinant(t1)) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(trace(t1)-12) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(norm(matmul(t1,t1)) - 187.637949) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(matmul(2*t1-t1,t1/2*2)) - 187.637949) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(cofactor(t1)) - 18) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(transpose(adjoint(t1))) - 18) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(det(t1)) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(trace(t1) - 12) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(norm(t1 % t1) - 187.637949) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(cof(t1)) - 18)< BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(trans(adj(t1))) - 18) < BigTol);

        Tensor<T,3> t2; t2.iota(102);
        FASTOR_EXIT_ASSERT(std::abs(norm(outer(t2,t2)) - 31829.0) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(inner(t2,t2) - 31829.0) < Tol);
    }

    // cross product
    {
        // classic cross product
        Tensor<T,3> a = {1,2,3};
        Tensor<T,3> b = {4,5,17};
        Tensor<T,3> res = cross(a,b);
        FASTOR_EXIT_ASSERT(std::abs(res(0) - 19) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(res(1) + 5 ) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(res(2) + 3 ) < Tol);

        FASTOR_EXIT_ASSERT(std::abs(sum(cross(a  ,b+0)) - 11) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(sum(cross(a+0,b  )) - 11) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(sum(cross(a+0,b+0)) - 11) < Tol);
    }

    // tensor cross product
    {
        Tensor<T,3,3> a; a.iota(5);
        for (size_t i=0; i<3; ++i) a(i,i) = 10;

        FASTOR_EXIT_ASSERT(std::abs(sum(0.5*cross(a,a) - cofactor(a))) < Tol);
    }

    // cofactor and adjoint
    {
        Tensor<T,2,2> a0; a0.random();
        Tensor<T,3,3> a1; a1.random();
        Tensor<T,4,4> a2; a2.random();
        for (size_t i=0; i<2; ++i) a0(i,i) = 10;
        for (size_t i=0; i<3; ++i) a1(i,i) = 10;
        for (size_t i=0; i<4; ++i) a2(i,i) = 10;

        FASTOR_EXIT_ASSERT(std::abs(sum(transpose( adjoint(a0)) - cofactor(a0))) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(transpose(cofactor(a0)) -  adjoint(a0))) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(sum(transpose( adjoint(a1)) - cofactor(a1))) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(transpose(cofactor(a1)) -  adjoint(a1))) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(sum(transpose( adjoint(a2)) - cofactor(a2))) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(transpose(cofactor(a2)) -  adjoint(a2))) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(sum(trans(adj(a0)) - cof(a0))) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(trans(cof(a0)) - adj(a0))) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(sum(trans(adj(a1)) - cof(a1))) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(trans(cof(a1)) - adj(a1))) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(sum(trans(adj(a2)) - cof(a2))) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(trans(cof(a2)) - adj(a2))) < BigTol);

        // expressions
        FASTOR_EXIT_ASSERT(std::abs(sum(transpose( adjoint(a0-1)) - cofactor(a0-1))) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(transpose(cofactor(a0+2)) -  adjoint(a0+2))) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(sum(trans(adj(a1+0)) - cof(a1-a1+a1))) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(trans(cof(a1+0)) - adj(a1+a1-a1))) < BigTol);

        // if trans(cof/adj) dispatches to adj/cof then test
        FASTOR_EXIT_ASSERT(std::abs(sum(transpose(adj(a1+0)) - cof(a1-a1+a1))) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(transpose(cof(a1+0)) - adj(a1+a1-a1))) < BigTol);
    }

    // det by LU
    {
        Tensor<T,4,4> A; A.arange();
        for (size_t i=0; i<4; ++i) A(i,i) = T(10);
        FASTOR_EXIT_ASSERT(std::abs(determinant<DetCompType::LU>(A) - determinant(A)) < HugeTol);
        FASTOR_EXIT_ASSERT(std::abs(det<DetCompType::LU>(A)         - det(A)        ) < HugeTol);
    }

    print(FGRN(BOLD("All tests passed successfully")));

}

int main() {

    print(FBLU(BOLD("Testing linear algebra routines: single precision")));
    test_linalg<float>();
    print(FBLU(BOLD("Testing linear algebra routines: double precision")));
    test_linalg<double>();

    return 0;
}

