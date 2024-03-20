#include <Fastor/Fastor.h>
using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5
#define HugeTol 1e-2


template<typename T>
void run() {

    using std::abs;

    {
        Tensor<T,10> v; v.iota(100);
        Tensor<T,10,10> A; A.iota(1);

        // Assign vectors to matrices
        A(all,0) = v;
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 5635) < Tol);
        A(all,0) += v;
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 6680) < Tol);
        A(-1,all) -= v;
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 5635) < Tol);
        A(seq(0,-1,2),2) *= v(seq(0,5));
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 27550) < Tol);
        A(4,seq(0,last,1)) /= 100*v;
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 22585.484375) < HugeTol);
        A(last,last) = v(last);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 22703.484375) < HugeTol);

        // Assign vector views to matrices - dynamic
        A.iota(1);
        A(all,0) = v(all);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 5635) < Tol);
        A(all,0) += v(all);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 6680) < Tol);
        A(-1,all) -= v(all);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 5635) < Tol);
        A(seq(0,-1,2),2) *= v(seq(0,5));
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 27550) < Tol);
        A(4,seq(0,last,1)) /= 100*v(all);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 22585.484375) < HugeTol);

        // Assign vector views to matrices - static
        A.iota(1);
        A(fall,0) = v(fall);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 5635) < Tol);
        A(fall,0) += v(all);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 6680) < Tol);
        A(last,fall) -= v(all);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 5635) < Tol);
        A(fseq<0,-1,2>(),2) *= v(fseq<0,5>());
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 27550) < Tol);
        A(4,fseq<0,last,1>()) /= 100*v(fall);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 22585.484375) < HugeTol);

        // Assigning matrix views to vectors 
        A.iota(1);
        v = A(all,0);
        FASTOR_DOES_CHECK_PASS(abs(v.sum() - 460) < Tol);
        v = A(1,all);
        FASTOR_DOES_CHECK_PASS(abs(v.sum() - 155) < Tol);
        v += A(2,all);
        FASTOR_DOES_CHECK_PASS(abs(v.sum() - 410) < Tol);
        v -= A(all,3);
        FASTOR_DOES_CHECK_PASS(abs(v.sum() + 80) < Tol);
        v *= A(all,0);
        FASTOR_DOES_CHECK_PASS(abs(v.sum() + 10280) < BigTol);
        v /= -A(all,1);
        FASTOR_DOES_CHECK_PASS(abs(v.sum() - 94.38235) < HugeTol);

        // Assigning matrix views to vector views - dynamic
        A.iota(1);
        v(all) = A(all,0);
        FASTOR_DOES_CHECK_PASS(abs(v.sum() - 460) < Tol);
        v(all) = A(1,all);
        FASTOR_DOES_CHECK_PASS(abs(v.sum() - 155) < Tol);
        v(all) += A(2,all);
        FASTOR_DOES_CHECK_PASS(abs(v.sum() - 410) < Tol);
        v(all) -= A(all,3);
        FASTOR_DOES_CHECK_PASS(abs(v.sum() + 80) < Tol);
        v(all) *= A(all,0);
        FASTOR_DOES_CHECK_PASS(abs(v.sum() + 10280) < BigTol);
        v(all) /= -A(all,1);
        FASTOR_DOES_CHECK_PASS(abs(v.sum() - 94.38235) < HugeTol);

        // Assigning matrix views to vector views - static
        A.iota(1);
        v(fall) = A(fall,0);
        FASTOR_DOES_CHECK_PASS(abs(v.sum() - 460) < Tol);
        v(fall) = A(1,all);
        FASTOR_DOES_CHECK_PASS(abs(v.sum() - 155) < Tol);
        v(fall) += A(2,all);
        FASTOR_DOES_CHECK_PASS(abs(v.sum() - 410) < Tol);
        v(fall) -= A(all,3);
        FASTOR_DOES_CHECK_PASS(abs(v.sum() + 80) < Tol);
        v(fall) *= A(all,0);
        FASTOR_DOES_CHECK_PASS(abs(v.sum() + 10280) < BigTol);
        v(fall) /= -A(all,1);
        FASTOR_DOES_CHECK_PASS(abs(v.sum() - 94.38235) < HugeTol);
    }


    {
        Tensor<T,10,10> A; A.iota(100);
        Tensor<T,10,10,3> B; B.iota(1);
        Tensor<T,10,5,10,3> C; C.iota(1000);

        // Assign matrices to high order tensors matrices
        B(all,all,1) = A;
        FASTOR_DOES_CHECK_PASS(abs(B.sum() - 45050) < Tol);
        B(all,all,0) += A;
        FASTOR_DOES_CHECK_PASS(abs(B.sum() - 60000) < Tol);
        B(all,0,all) -= A(all,seq(0,3));
        FASTOR_DOES_CHECK_PASS(abs(B.sum() - 55620) < Tol);
        B(all,all,2) *= A(all,all) + 2;
        FASTOR_DOES_CHECK_PASS(abs(B.sum() - 2362800) < Tol);
        B(all,1,1) = 50;
        FASTOR_DOES_CHECK_PASS(abs(B.sum() - 2361840) < Tol);
        B(all,1,1) /= -200*A(1,all);
        FASTOR_DOES_CHECK_PASS(abs(B.sum() - 2361339.97815) < 0.5);

        // Assigning high order tensor views to matrices 
        B.iota(1);
        A = B(all,all,1);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 15050) < Tol);
        A += C(all,0,all,1);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 184000) < Tol);
        A -= C(all,0,all,1);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 15050) < Tol);
        A *= C(all,0,all,1) + B(all,all,0);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 32146800) < BigTol);
        A /= C(all,0,all,1) - B(all,all,0);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 18377.20703) < 0.5);


        // Assigning high order tensor views to matrix views - dynamic
        B.iota(1);
        A(all,all) = B(all,all,1);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 15050) < Tol);
        A(all,all) += C(all,0,all,1);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 184000) < Tol);
        A(all,all) -= C(all,0,all,1);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 15050) < Tol);
        A(all,all) *= C(all,0,all,1) + B(all,all,0);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 32146800) < BigTol);
        A(all,all) /= C(all,0,all,1) - B(all,all,0);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 18377.20703) < 0.5);


        // Assigning high order tensor views to matrix views - static
        B.iota(1);
        A(fall,fall) = B(fall,fall,fseq<1,2>());
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 15050) < Tol);
        A(fall,fall) += C(fall,fseq<0,1>(),all,fseq<1,2>());
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 184000) < Tol);
        A(fall,fall) -= C(fall,fseq<0,1>(),all,fseq<1,2>());
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 15050) < Tol);
        A(fall,fall) *= C(fall,fseq<0,1>(),all,fseq<1,2>()) + B(fall,fall,fseq<0,1>());
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 32146800) < BigTol);
        A(fall,fall) /= C(fall,fseq<0,1>(),all,fseq<1,2>()) - B(fall,fall,fseq<0,1>());
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 18377.20703) < 0.5);


        // Complex high order tensors - dynamic
        A.iota(); B.iota(); C.iota();
        C(all,all,1,2) = B(all,seq(0,5),1);
        FASTOR_DOES_CHECK_PASS(abs(C.sum() - 1094350) < HugeTol);
        C(seq(0,4,2),all,0,0) += A(seq(1,3),seq(first,5,1));
        FASTOR_DOES_CHECK_PASS(abs(C.sum() - 1094520) < HugeTol);
        C(seq(0,4,2),all,0,0) -= A(seq(1,3),seq(first,5,1));
        FASTOR_DOES_CHECK_PASS(abs(C.sum() - 1094350) < HugeTol);


        // Complex high order tensors - dynamic+static
        A.iota(); B.iota(); C.iota();
        C(fall,fall,1,2) = B(all,seq(0,5),1);
        FASTOR_DOES_CHECK_PASS(abs(C.sum() - 1094350) < HugeTol);
        C(seq(0,4,2),fall,0,0) += A(fseq<1,3>(),fseq<first,5,1>());
        FASTOR_DOES_CHECK_PASS(abs(C.sum() - 1094520) < HugeTol);
        C(seq(0,4,2),fall,0,0) -= A(fseq<1,3>(),fseq<first,5,1>());
        FASTOR_DOES_CHECK_PASS(abs(C.sum() - 1094350) < HugeTol);

        // Complex high order tensors - dynamic+static
        A.iota(); B.fill(10); C.iota();
        A = B(all,all,1) + C(all,1,all,0) - sqrt(B(all,all,0));
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 72533.8) < 0.5);
        A(fall,fall) = B(all,all,1) + C(all,1,all,0) - sqrt(B(all,all,0));
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 72533.8) < 0.5);
        A(all,all) = B(fall,fall,1) + C(fall,1,fall,0) - sqrt(B(fall,fall,0));
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 72533.8) < 0.5);
        A.iota(); C.iota(1);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 4950) < Tol);
        A = abs(A);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 4950) < Tol);
        A = abs(-A);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 4950) < Tol);
        A(all,all) += B(all,all,0) + C(all,0,all,0)/C(all,0,all,0) - 1 - B(all,all,0);
        FASTOR_DOES_CHECK_PASS(abs(A.sum() - 4950) < Tol);
        A.fill(0); B.fill(10); C.fill(20);
        A(all,all) -= log(B(all,all,0)) + abs(B(all,fall,1)) + sin(C(all,0,all,0)) - 1 - cos(B(all,all,0));
        FASTOR_DOES_CHECK_PASS(abs(A.sum() + 1305.46) < 2.);
    }


    print(FGRN(BOLD("All tests passed successfully")));
}

int main() {

    print(FBLU(BOLD("Testing complex tensor expressions: single precision")));
    run<float>();
    print(FBLU(BOLD("Testing complex tensor expressions: double precision")));
    run<double>();


    return 0;
}
