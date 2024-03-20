#include <Fastor/Fastor.h>

using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5
#define HugeTol 1e-2


template<typename T>
void test_qr() {

    // QR
    {
        Tensor<T,4,4> A; A.arange();
        for (size_t i=0; i<4; ++i) A(i,i) = 10;

        {
            Tensor<T,4,4> Q, R;
            qr(A, Q, R);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(A - Q%R)) < BigTol);
        }
        {
            Tensor<T,4,4> Q, R;
            qr(A+0, Q, R);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(A - Q%R)) < BigTol);
        }
        {
            Tensor<T,4,4> Q, R;
            qr<QRCompType::MGSR>(A+0, Q, R);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(A - Q%R)) < BigTol);
        }
        {
            Tensor<T,4,4> Q, R;
            Tensor<size_t,4> P;
            qr<QRCompType::MGSRPiv>(A+0, Q, R, P);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(A - Q%R)) < BigTol);
        }
        {
            Tensor<T,4,4> Q, R, P;
            qr<QRCompType::MGSRPiv>(A+0, Q, R, P);
            FASTOR_DOES_CHECK_PASS(std::abs(sum(A - Q%R)) < BigTol);
        }
        // absdet by QR
        {
            FASTOR_DOES_CHECK_PASS(std::abs(absdet<DetCompType::QR>(A) - std::abs(determinant(A))) < HugeTol);
        }
        // logdet by QR
        {
            FASTOR_DOES_CHECK_PASS(std::abs(logdet<DetCompType::QR>(A) - std::log(std::abs(determinant(A)))) < HugeTol);
        }
    }

    print(FGRN(BOLD("All tests passed successfully")));

}

int main() {

    print(FBLU(BOLD("Testing linear algebra routines: single precision")));
    test_qr<float>();
    print(FBLU(BOLD("Testing linear algebra routines: double precision")));
    test_qr<double>();

    return 0;
}

