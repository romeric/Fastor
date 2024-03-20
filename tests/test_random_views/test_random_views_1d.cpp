#include <Fastor/Fastor.h>
using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5

template<typename T>
void run() {

    using std::abs;
    {
        Tensor<T,67> a1; a1.iota(0);
        Tensor<int,45> it1; it1.iota(1);

        // Check construction from views
        Tensor<T,45> a2 = a1(it1);
        FASTOR_DOES_CHECK_PASS(abs(a2.sum() - 1035) < Tol);
        a2 += a1(it1);
        FASTOR_DOES_CHECK_PASS(abs(a2.sum() - 2*1035) < Tol);
        a2 -= a1(it1);
        FASTOR_DOES_CHECK_PASS(abs(a2.sum() - 1035) < Tol);
        a2 *= 2 + a1(it1);
        FASTOR_DOES_CHECK_PASS(abs(a2.sum() - 33465) < Tol);
        a2 /= a1(it1) + 5000;
        FASTOR_DOES_CHECK_PASS(abs(a2.sum() - 6.64796) < 1e-2);

        // Assigning to a view from numbers/tensors/views
        Tensor<T,13> a3; a3.iota(10);
        Tensor<int,7> it2; it2.iota(5);
        a3(it2) = 4;
        FASTOR_DOES_CHECK_PASS(abs(a3.sum() - 110) < Tol);
        a3(it2) += 3;
        FASTOR_DOES_CHECK_PASS(abs(a3.sum() - 131) < Tol);
        a3(it2) -= 3;
        FASTOR_DOES_CHECK_PASS(abs(a3.sum() - 110) < Tol);
        a3(it2) *= 3;
        FASTOR_DOES_CHECK_PASS(abs(a3.sum() - 166) < Tol);
        a3(it2) /= 3;
        FASTOR_DOES_CHECK_PASS(abs(a3.sum() - 110) < Tol);

        Tensor<T,7> a4; a4.iota(1);
        it2.iota(2);
        a3.fill(1);
        a3(it2) = a4;
        FASTOR_DOES_CHECK_PASS(abs(a3.sum() - 34) < Tol);
        a3(it2) += 2*a4;
        FASTOR_DOES_CHECK_PASS(abs(a3.sum() - 90) < Tol);
        a3(it2) -= a4+2;
        FASTOR_DOES_CHECK_PASS(abs(a3.sum() - 48) < Tol);
        a3(it2) *= -a4-100;
        FASTOR_DOES_CHECK_PASS(abs(a3.sum() + 4418) < Tol);
        a3(it2) /= 2+a4;
        FASTOR_DOES_CHECK_PASS(abs(a3.sum() + 626.5666) < 1e-2);

        it2.iota(0);
        a3.iota(10);
        a3(it2) = a4(it2);
        FASTOR_DOES_CHECK_PASS(abs(a3.sum() - 145) < Tol);
        a3(it2) += a4(it2);
        FASTOR_DOES_CHECK_PASS(abs(a3.sum() - 173) < Tol);
        a3(it2) -= a4(it2);
        FASTOR_DOES_CHECK_PASS(abs(a3.sum() - 145) < Tol);
        a3(it2) *= a4(it2);
        FASTOR_DOES_CHECK_PASS(abs(a3.sum() - 257) < Tol);
        a3(it2) /= a4(it2);
        FASTOR_DOES_CHECK_PASS(abs(a3.sum() - 145) < BigTol);

        Tensor<T,7> a5; a5.iota(1);
        a4(it2) = a5(it2);
        FASTOR_DOES_CHECK_PASS(abs(a4.sum() - 28) < Tol);
        a4(it2) += a5(it2);
        FASTOR_DOES_CHECK_PASS(abs(a4.sum() - 56) < Tol);
        a4(it2) -= a5(it2);
        FASTOR_DOES_CHECK_PASS(abs(a4.sum() - 28) < Tol);
        a4(it2) *= a5(it2);
        FASTOR_DOES_CHECK_PASS(abs(a4.sum() - 140) < Tol);
        a4(it2) /= a5(it2);
        FASTOR_DOES_CHECK_PASS(abs(a4.sum() - 28) < Tol);

        // Check overlap
        Tensor<int,7> it3; it3.iota();
        it2 = {1,2,3,4,5,6,0};
        a4(it2).noalias() = a4(it3);
        FASTOR_DOES_CHECK_PASS(abs(a4(0) - 7) < Tol && abs(a4(-1) - 6) < Tol);
        a4(it2).noalias() += a4(it3);
        FASTOR_DOES_CHECK_PASS(abs(a4(0) - 13) < Tol && abs(a4(-1) - 11) < Tol);
        a4(it2).noalias() -= a4(it3);
        FASTOR_DOES_CHECK_PASS(abs(a4(1) + 5) < Tol && abs(a4(-1) - 2) < Tol);
        a4(it2).noalias() *= a4(it3);
        FASTOR_DOES_CHECK_PASS(abs(a4(1) + 10) < Tol && abs(a4(-1) - 4) < Tol);
        a4(it2).noalias() /= a4(it3);
        FASTOR_DOES_CHECK_PASS(abs(a4(1) + 2.5) < BigTol && abs(a4(-1) - 1) < BigTol);
        
        Tensor<size_t,7> it4; it4.iota();
        a4.iota(1);
        a4(it2).noalias() = a4(it4);
        FASTOR_DOES_CHECK_PASS(abs(a4(0) - 7) < Tol && abs(a4(-1) - 6) < Tol);
        a4(it2).noalias() += a4(it4);
        FASTOR_DOES_CHECK_PASS(abs(a4(0) - 13) < Tol && abs(a4(-1) - 11) < Tol);
        a4(it2).noalias() -= a4(it4);
        FASTOR_DOES_CHECK_PASS(abs(a4(1) + 5) < Tol && abs(a4(-1) - 2) < Tol);
        a4(it2).noalias() *= a4(it4);
        FASTOR_DOES_CHECK_PASS(abs(a4(1) + 10) < Tol && abs(a4(-1) - 4) < Tol);
        a4(it2).noalias() /= a4(it4);
        FASTOR_DOES_CHECK_PASS(abs(a4(1) + 2.5) < BigTol && abs(a4(-1) - 1) < BigTol);

        print(FGRN(BOLD("All tests passed successfully")));
    }
}


int main() {

    print(FBLU(BOLD("Testing 1-dimensional random tensor views: single precision")));
    run<float>();
    print(FBLU(BOLD("Testing 1-dimensional random tensor views: double precision")));
    run<double>();

    return 0;
}
