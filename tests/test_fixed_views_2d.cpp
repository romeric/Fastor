#include <Fastor.h>
using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5


template<typename T>
void run() {

    using std::abs;
    {
        // Check construction from views
        Tensor<T,15,18> a1; a1.iota(11);
        decltype(a1) a2 = a1(fall,fall);
        FASTOR_EXIT_ASSERT(abs(norm(a1)-norm(a2)) < Tol);
        decltype(a1) a3;
        a3 = a1(fseq<0,-1>{},fseq<first,last,1>());
        FASTOR_EXIT_ASSERT(abs(norm(a1)-norm(a3)) < Tol);
        Tensor<T,3,5> a4 = a1(fseq<3,last,4>{},fseq<4,18,3>{});
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 2205) < Tol);
        a4 += 5*a1(fseq<3,last,4>{},fseq<4,18,3>{}) / 4 ; // This covers eval
        FASTOR_EXIT_ASSERT(abs(norm(a4) - 1380.12329) < BigTol);

        // Assigning to a view from numbers/tensors/views
        a4(fall,fseq<first,2>()) = 2;
        FASTOR_EXIT_ASSERT(abs(norm(a4) - 1087.620165) < 1e-3);
        a4(fall,fseq<first,2>()) += 11;
        FASTOR_EXIT_ASSERT(abs(norm(a4) - 1088.0751) < 1e-3);
        a4(fall,fseq<first,2>()) -= 11;
        FASTOR_EXIT_ASSERT(abs(norm(a4) - 1087.620165) < 1e-3);
        a4(fseq<0,2,1>(),fseq<first,2>()) *= 3;
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 3065.5) < Tol);
        a4(fseq<0,2,1>{},fseq<first,2>()) /= 3;
        FASTOR_EXIT_ASSERT(abs(norm(a4) - 1087.620165) < 1e-3);

        a4(fall,fall) = a4;
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 3049.5) < Tol);
        a4(fall,fall) += 2*a4;
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 9148.5) < Tol);
        a4(fseq<first,last>(),fseq<0,-1>()) -= 2*a4;
        FASTOR_EXIT_ASSERT(abs(a4.sum() + 9148.5) < Tol);
        a4(fseq<0,2>(),fseq<1,5,2>()) *=10;
        FASTOR_EXIT_ASSERT(abs(a4.sum() + 23107.5) < Tol);
        a4(fseq<first,last>(),fseq<first,last>()) *=a4;
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 139586878.125) < 20); // SP gives a large difference
        a4(fall,fall) /=a4;
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 15.) < Tol);

        a4(fall,fall) = a1(fseq<3,last,4>(),fseq<4,18,3>());
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 2205) < Tol);
        decltype(a4) a5; a5.iota(12);
        a4.ones();
        a4(fall,fall) += a5(fall,fall);
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 300.) < Tol);
        a4(fall,fall) -= a5(fall,fall);
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 15.) < Tol);
        a4(fseq<0,2>(),fseq<0,2>()) *= a5(fseq<1,3>{},fseq<2,4>());
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 99.) < Tol);
        a4(fseq<0,2>{},fseq<0,2>()) /= a5(fseq<1,3>{},fseq<2,4>());
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 15.) < Tol);

        // Check overlap - fseq does not allow overlap
        a4(all,all) = a4(all,all);
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 15.) < Tol);
        a4(all,all).noalias() = a4(all,all);
        FASTOR_EXIT_ASSERT(abs(a4.sum() - 15.) < Tol);

        print(FGRN(BOLD("All tests passed successfully")));
    }
}

int main() {

    print(FBLU(BOLD("Testing 2-dimensional tensor views: single precision")));
    run<float>();
    print(FBLU(BOLD("Testing 2-dimensional tensor views: double precision")));
    run<double>();


    return 0;
}
