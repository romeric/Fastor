#include <Fastor.h>
using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5

template<typename T>
void run() {

    using std::abs;
    {
        Tensor<T,67> a1; a1.iota(0);

        // scalar indexing
        FASTOR_EXIT_ASSERT(abs(a1(23) - 23) < Tol);
        FASTOR_EXIT_ASSERT(abs(a1(-1) - 66) < Tol);
        FASTOR_EXIT_ASSERT(abs(a1(last) - 66) < Tol);


        // Check construction from views
        Tensor<T,56> a2 = a1(fseq<11,last>());
        FASTOR_EXIT_ASSERT(abs(a2.sum() - 2156) < Tol);
        decltype(a2) a3 = a2(fall);
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 2156) < Tol);
        a3 += a2(fseq<first,last>());
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 4312) < Tol);
        a3 -= a2(fseq<first,last>());
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 2156) < Tol);
        a3 *= 2+a2(fseq<first,last>());
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 101948) < Tol);
        a3 /= a2(fseq<first,last>{}) - 5;
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 2632.45) < 1e-2);
        a3 = a2(fall) + 2*a2(fseq<first,last>());
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 6468) < Tol);

        // Assigning to a view from numbers/tensors/views
        a3.iota(10);
        a3(fseq<5,10>()) = 4;
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 2035) < Tol);
        a3(fseq<25,last>()) = 3;
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 578) < Tol);
        a3(fseq<25,last>()) += 3;
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 671) < Tol);
        a3(fseq<25,last>{}) -= 3;
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 578) < Tol);
        a3(fseq<25,last-1>{}) *= 3;
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 758) < Tol);
        a3(fseq<25,last-1,3>{}) /= 3;
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 698) < Tol);

        a3(fall) = a2;
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 2156) < Tol);
        a3(fall) += 2*a2;
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 6468) < Tol);
        a3(fall) -= a2+2;
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 4200) < Tol);
        a3(fall) *= -a2;
        FASTOR_EXIT_ASSERT(abs(a3.sum() + 190960) < Tol);
        a3(fall) /= a2;
        FASTOR_EXIT_ASSERT(abs(a3.sum() + 4200) < Tol);

        a3(fall) = a2(fall);
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 2156) < Tol);
        a3(fall) = a2(fseq<first,last>());
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 2156) < Tol);
        a3(fseq<0,last,10>()) = 0;
        a3(fseq<0,last,10>()) += a2(fseq<0,last,10>{});
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 2156) < Tol);
        a3(fseq<1,last,9>()) -= a2(fseq<1,last,9>());
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 1883) < Tol);
        a3(fall) = a2(fall);
        a3(fseq<0,20,2>()) *= a2(fseq<22,42,2>());
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 10686) < Tol);
        a3(fseq<0,20,2>()) /= a2(fseq<22,42,2>{});
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 2156) < Tol);


        // Check overlap - fseq does not allow overlap - so only check perfect overlap
        a3.iota();
        a3(fseq<0,10>()) = a3(fseq<0,10>()); // perfect overlap -> fine
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 1540) < Tol);
        a3(fseq<0,10>()) = a3(fseq<1,11>()); // overlap but data written first and then read -> fine
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 1550) < Tol);
        a3(fseq<3,last,5>()) = a3(fseq<2,last-1,5>()); // no overlap -> fine
        FASTOR_EXIT_ASSERT(abs(a3.sum() - 1539) < Tol);


        // Check scanning
        Tensor<T,10> a4; a4.arange(10);
        Tensor<T,1> a5 = a4(fseq<9,10>());
        FASTOR_EXIT_ASSERT(abs(a5.toscalar() - 19) < Tol);
        a5 = a4(fseq<last-1,last>());
        FASTOR_EXIT_ASSERT(abs(a5.toscalar() - 19) < Tol);
        FASTOR_EXIT_ASSERT(abs(a4(last) - 19) < Tol);
        FASTOR_EXIT_ASSERT(abs(a4(last-1) - 18) < Tol);
        for (int i=0; i<10; ++i)
            FASTOR_EXIT_ASSERT(abs(a4(last-i) -  (19-i) ) < Tol);
        for (int i=0; i<10; ++i)
            FASTOR_EXIT_ASSERT(abs(a4(i) -  (10+i) ) < Tol);

        print(FGRN(BOLD("All tests passed successfully")));
    }
}


int main() {

    print(FBLU(BOLD("Testing multi-dimensional tensor views: single precision")));
    run<float>();
    print(FBLU(BOLD("Testing multi-dimensional tensor views: double precision")));
    run<double>();

    return 0;
}
