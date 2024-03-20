#include <Fastor/Fastor.h>

using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5


template<typename T>
void test_numerics() {

    // The test assumes that TensorMap is already
    // tested elsewhere

    // reshape
    {
        Tensor<T,3,4> a; a.iota(1);

        FASTOR_DOES_CHECK_PASS(std::abs(a.sum() - reshape<4,3>(a).sum()) < Tol);
        FASTOR_DOES_CHECK_PASS(std::abs(a.sum() - reshape<2,2,3>(a).sum()) < Tol);
        FASTOR_DOES_CHECK_PASS(std::abs(a.sum() - reshape<1,2,2,3>(a).sum()) < Tol);
        FASTOR_DOES_CHECK_PASS(std::abs(a.sum() - reshape<2,3,2>(a).sum()) < Tol);
        FASTOR_DOES_CHECK_PASS(std::abs(a.sum() - reshape<3,2,2>(a).sum()) < Tol);

        FASTOR_DOES_CHECK_PASS(reshape<4,3>(a).dimension(0) - 4 < Tol);
        FASTOR_DOES_CHECK_PASS(reshape<4,3>(a).dimension(1) - 3 < Tol);

        FASTOR_DOES_CHECK_PASS(reshape<2,2,3>(a).dimension(0) - 2 < Tol);
        FASTOR_DOES_CHECK_PASS(reshape<2,2,3>(a).dimension(1) - 2 < Tol);
        FASTOR_DOES_CHECK_PASS(reshape<2,2,3>(a).dimension(2) - 3 < Tol);
    }

    // flatten
    {
        Tensor<T,3,4> a; a.iota(1);

        FASTOR_DOES_CHECK_PASS(std::abs(a.sum() - flatten(a).sum()) < Tol);

        FASTOR_DOES_CHECK_PASS(flatten(a).dimension(0) - 12 < Tol);
    }

    // squeeze
    {
        Tensor<T,1,3,4> a; a.iota(1);

        FASTOR_DOES_CHECK_PASS(std::abs(a.sum() - squeeze(a).sum()) < Tol);

        FASTOR_DOES_CHECK_PASS(squeeze(a).dimension(0) - 3 < Tol);
        FASTOR_DOES_CHECK_PASS(squeeze(a).dimension(1) - 4 < Tol);

        FASTOR_DOES_CHECK_PASS(squeeze(Tensor<T,1,2,1>{}).dimension(0) - 2 < Tol);
        FASTOR_DOES_CHECK_PASS(is_same_v_<decltype(squeeze(Tensor<T,1,1>{})),TensorMap<T>> == true);
        FASTOR_DOES_CHECK_PASS(is_same_v_<decltype(squeeze(Tensor<T,1,1,3>{})),TensorMap<T,3>> == true);
    }

    print(FGRN(BOLD("All tests passed successfully")));

}


int main() {

    print(FBLU(BOLD("Testing numerics with single precision")));
    test_numerics<float>();
    print(FBLU(BOLD("Testing numerics with double precision")));
    test_numerics<double>();

    return 0;
}

