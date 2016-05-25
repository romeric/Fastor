#include <Fastor.h>

using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5


template<typename T>
void test_linalg() {

    Tensor<T,3,3> t1; t1.iota(0);
    assert(std::abs(determinant(t1))< Tol);
    assert(std::abs(trace(t1)-12)< Tol);
    assert(std::abs(norm(matmul(t1,t1)) - 187.637949) < BigTol);
    assert(std::abs(norm(cofactor(t1)) - 18) < BigTol);
    assert(std::abs(norm(transpose(adjoint(t1))) - 18) < BigTol);

    assert(std::abs(norm(static_cast<Tensor<T>>(ldeterminant(t1))))< Tol);
    assert(std::abs(norm(static_cast<Tensor<T>>(ltrace(t1))) - 12)< Tol);
    assert(std::abs(norm(static_cast<Tensor<T,3,3>>(lmatmul(t1,t1))) - 187.637949)< BigTol);
    assert(std::abs(norm(static_cast<Tensor<T,3,3>>(lcofactor(t1))) - 18)< BigTol);
    assert(std::abs(norm(static_cast<Tensor<T,3,3>>(ltranspose(ladjoint(t1)))) - 18)< BigTol);

    print(FGRN(BOLD("All tests passed successfully")));

}

int main() {

    print(FBLU(BOLD("Testing basic tensor construction routines with single precision")));
    test_linalg<float>();
    print(FBLU(BOLD("Testing basic tensor construction routines with double precision")));
    test_linalg<double>();

    return 0;
}

