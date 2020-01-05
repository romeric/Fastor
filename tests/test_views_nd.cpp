#include <Fastor.h>
using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5

template<typename T>
void run() {
    using std::abs;

    {
        Tensor<T,3,4,2,4,5> r1; r1.iota(0);

        // Scalar indexing
        FASTOR_EXIT_ASSERT(abs(r1(0,2,1,3,4) - 119) < Tol);

        // Views
        // Assignment to views
        r1(all,0,all,all,3) = -1234;
        FASTOR_EXIT_ASSERT(abs(norm(r1) - 8491.192377) < BigTol);

        r1.iota();
        r1(seq(0,3),0,seq(0,-1),seq(0,4,1),3) = -1234;
        FASTOR_EXIT_ASSERT(abs(norm(r1) - 8491.192377) < BigTol);

        // other operators
        r1(seq(0,3,2),all,1,seq(1,4,3),4) += 1234;
        FASTOR_EXIT_ASSERT(abs(norm(r1) - 9444.503) < 1e-3);
        r1(seq(0,3,2),all,1,seq(1,4,3),4) -= 1234;
        FASTOR_EXIT_ASSERT(abs(norm(r1) - 8491.192377) < BigTol);
        r1(seq(0,3,2),0,1,seq(1,4,3),4) *= 12;
        FASTOR_EXIT_ASSERT(abs(norm(r1) - 9467.742) < 1e-3);
        r1(seq(0,3,2),0,1,seq(1,4,3),4) /= 12;
        FASTOR_EXIT_ASSERT(abs(norm(r1) - 8491.192377) < BigTol);


        // Check construction from views
        Tensor<T,3,3,5> r2; r2.iota(100);
        decltype(r2) r3 = r2(all,all,all);
        FASTOR_EXIT_ASSERT(abs(norm(r2) - norm(r3)) < Tol);
        r3 = 2*r2(all,seq(0,-1),seq(0,5))/12 + 5;
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 170.560351) < 1e-3);
        r3 += -3*r3(all,all,all) - 15;
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 441.462) < 1e-3);
        r3 -= -3*r3(all,all,all) - 15;
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 1665.45439) < 1e-3);
        r3 *= r2/5 - 6*r3(all,all,all) + r2(all,all,all);
        FASTOR_EXIT_ASSERT(abs(norm(r3) -  2753466.75326) < 0.5);
        r3 /= r2/5 - 6*r3(all,all,all) / r2(all,all,all);
        FASTOR_EXIT_ASSERT(abs(norm(r3) -  137.001) < 1e-2);

        // Assigning to a view from numbers/tensors/views
        r3(seq(0,-1,1),all,seq(0,5)) = 2;
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 13.416407) < 1e-3);
        r3(all,all,all) = r3(all,all,all);
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 13.416407) < 1e-3);
        r3(0,seq(0,2),seq(1,-1,2)) += 5;
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 18.9736659) < 1e-3);
        r3(0,seq(0,2),seq(1,-1,2)) -= 5;
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 13.416407) < 1e-3);
        r3(0,seq(0,2),seq(1,-1,2)) *= 5;
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 23.748684) < 1e-3);
        r3(0,seq(0,2),seq(1,-1,2)) /= 5;
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 13.416407) < 1e-3);

        r3(all,all,all) = r2;
        FASTOR_EXIT_ASSERT(abs(norm(r2) - norm(r3)) < Tol);
        r3(all,all,all) += r2;
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 1646.049816) < 1e-3);
        r3(all,all,all) -= r2;
        FASTOR_EXIT_ASSERT(abs(norm(r2) - norm(r3)) < BigTol);
        r3(all,all,all) *= r2;
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 103194.60105) < 0.5);
        r3(all,all,all) /= r2;
        FASTOR_EXIT_ASSERT(abs(norm(r2) - norm(r3)) < BigTol);


        r3(all,all,all) = r2(all,seq(0,-1),seq(0,-1,1));
        FASTOR_EXIT_ASSERT(abs(norm(r2) - norm(r3)) < Tol);
        r3(all,all,all) += r2(seq(0,-1,1),seq(0,3),seq(0,5));
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 1646.049816) < 1e-3);
        r3(all,all,all) -= r2(all,all,all);
        FASTOR_EXIT_ASSERT(abs(norm(r2) - norm(r3)) < BigTol);
        r3(all,all,all) *= r2(seq(0,3),seq(0,3),seq(0,5));
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 103194.60105) < 0.5);
        r3(all,all,all) /= r2(all,all,seq(0,-1));
        FASTOR_EXIT_ASSERT(abs(norm(r2) - norm(r3)) < BigTol);



        // Check overlap
        r3 = r2;
        FASTOR_EXIT_ASSERT(abs(norm(r2) - norm(r3)) < BigTol);
        r3(all,all,seq(1,3)) = 2*r3(all,all,seq(1,3)); // Perfect overlapping does not require noalias()
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 1218.00862) < 1e-3);
        r3(all,all,seq(1,3)).noalias() = r3(all,all,seq(1,3)); // Check with noalias() nevertheless
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 1218.00862) < 1e-3);
        r3(all,all,seq(1,3)).noalias() = r3(all,all,seq(0,2));
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 1035.77072) < 1e-3);
        r3(all,all,seq(1,3)).noalias() += r3(all,all,seq(0,2));
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 1458.0606) < 1e-3);
        r3(all,all,seq(1,3)).noalias() -= r3(all,all,seq(0,2));
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 821.7061518) < 1e-3);
        r3(seq(0,2),all,seq(1,3)).noalias() *= r3(seq(1,3),all,seq(0,2));
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 50891.812966) < 1e-2);
        r3(0,all,all).noalias() /= r3(0,all,all); // perfect overlap noalias not required
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 40141.5219) < 1e-2);
        r3(all,seq(0,2),seq(first,last)).noalias() /= r3(all,seq(1,3),all);
        FASTOR_EXIT_ASSERT(abs(norm(r3) - 24950.50328) < 1e-2);


        // Check scanning
        Tensor<T,2,3,4,1> r4; r4.iota(2);
        Tensor<T,2,3,4,1> r5 = r4(seq(first,last),all,all,last);
        FASTOR_EXIT_ASSERT(abs(norm(r5) - norm(r4)) < Tol);
        Tensor<T,1,3,4,1> r6 = r4(0,all,all,last);
        FASTOR_EXIT_ASSERT(abs(norm(r6) - 28.600699) < 1e-3);
        Tensor<T,1,1,4,1> r7 = r4(0,last,all,last);
        FASTOR_EXIT_ASSERT(abs(r7.sum() - 46) < BigTol);
        Tensor<T,1,1,1,1> r8 = r4(first,2,first,0);
        FASTOR_EXIT_ASSERT(abs(r8.toscalar() - 10) < BigTol);
        Tensor<T,1,1,1,1> r9 = r4(last,last,last,last);
        FASTOR_EXIT_ASSERT(abs(r9.toscalar() - 25) < BigTol);

        r4.iota();
        int counter = 0;
        for (int i=0; i<2; ++i) {
            for (int j=0; j<3; ++j) {
                for (int k=0; k<4; ++k) {
                    for (int l=0; l<1; ++l) {
                        FASTOR_EXIT_ASSERT(abs(r4(i,j,k,l) - counter) < Tol);
                        FASTOR_EXIT_ASSERT(abs(r4(last-i,last-j,last-k,last-l) - (23-counter) ) < Tol);
                        counter++;
                    }
                }
            }
        }

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