#include <Fastor/Fastor.h>

using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5



template<typename T>
void run() {

    // the test assumes that matmul is tested already elsewhere
    {
        Tensor<T,3,4> a;
        Tensor<T,4,5> b;
        Tensor<T,3,5> ab(2);
        Tensor<T,3> c(0);
        Tensor<T,5> d(1);
        a.iota(1);
        b.iota(2);

        Tensor<T,3,5> e0 = matmul(a,b);
        Tensor<T,3,5> e1 = a%b;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        // test matmul expression assigns
        e0 += matmul(a,b);
        e1 += a%b;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 -= matmul(a,b);
        e1 -= a%b;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b);
        e1 *= a%b;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 /= matmul(a,b);
        e1 /= a%b;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);


        // test binary_add expression assigns when matmul is present
        e0 = matmul(a,b) + 2;
        e1 = a%b + 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 += matmul(a,b) + 2;
        e1 += a%b + 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 -= matmul(a,b) + 2;
        e1 -= a%b + 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) + 2;
        e1 *= a%b + 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 /= matmul(a,b) + 2;
        e1 /= a%b + 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);


        // test binary_sub expression assigns when matmul is present
        e0 = matmul(a,b) - 2;
        e1 = a%b - 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 += matmul(a,b) - 2;
        e1 += a%b - 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 -= matmul(a,b) - 2;
        e1 -= a%b - 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) - 2;
        e1 *= a%b - 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 /= matmul(a,b) - 2;
        e1 /= a%b - 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);


        // test binary_mul expression assigns when matmul is present
        e0 = matmul(a,b) * 2;
        e1 = a%b * 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 += matmul(a,b) * 2;
        e1 += a%b * 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 -= matmul(a,b) * 2;
        e1 -= a%b * 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) * 2;
        e1 *= a%b * 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 /= matmul(a,b) * 2;
        e1 /= a%b * 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);


        // test binary_div expression assigns when matmul is present
        e0 = matmul(a,b) / 2;
        e1 = a%b / 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 += matmul(a,b) / 2;
        e1 += a%b / 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 -= matmul(a,b) / 2;
        e1 -= a%b / 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) / 2;
        e1 *= a%b / 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 /= matmul(a,b) / 2;
        e1 /= a%b / 2;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);


       // test binary_add expression assigns when matmul and aliasing is present
        e0 = matmul(a,b) + e0;
        e1 = a%b + e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 += matmul(a,b) + e0;
        e1 += a%b + e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 -= e0 * matmul(a,b) + e0;
        e1 -= e1 * (a%b) + e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= e0 / matmul(a,b) + e0;
        e1 *= e1 / (a%b) + e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 /= e0 * matmul(a,b) + e0;
        e1 /= e1 * (a%b) + e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);


       // test binary_sub expression assigns when matmul and aliasing is present
        e0 = e0 - matmul(a,b) - e0;
        e1 = e1 - a%b - e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 += e0 - matmul(a,b) - e0;
        e1 += e1 - a%b - e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 -= e0 * matmul(a,b) - e0;
        e1 -= e1 * (a%b) - e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= e0 / matmul(a,b) - e0;
        e1 *= e1 / (a%b) - e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 /= matmul(a,b) - e0;
        e1 /= (a%b) - e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);


        // test binary_mul expression assigns when matmul and aliasing is present
        e0 = matmul(a,b) * e0;
        e1 = (a%b) * e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 += matmul(a,b) * e0;
        e1 += (a%b) * e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) * e0;
        e1 *= (a%b) * e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) * e0;
        e1 *= (a%b) * e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) * e0;
        e1 *= (a%b) * e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);


        // test binary_div expression assigns when matmul and aliasing is present
        e0 = matmul(a,b) / e0;
        e1 = (a%b) / e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 += matmul(a,b) / e0;
        e1 += (a%b) / e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) / e0;
        e1 *= (a%b) / e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) / e0;
        e1 *= (a%b) / e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);

        e0 *= matmul(a,b) / e0;
        e1 *= (a%b) / e1;
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < Tol);


        // test unary ops
        e0 = sqrt(matmul(a,b));
        e1 = sqrt(a%b);
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < BigTol);

        e0 += sqrt(matmul(a,b));
        e1 += sqrt(a%b);
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < BigTol);

        e0 -= sqrt(matmul(a,b));
        e1 -= sqrt(a%b);
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < BigTol);

        e0 *= sqrt(matmul(a,b));
        e1 *= sqrt(a%b);
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < BigTol);

        e0 /= sqrt(matmul(a,b));
        e1 /= sqrt(a%b);
        FASTOR_DOES_CHECK_PASS(std::abs(e0.sum() - e1.sum()) < BigTol);


        // double matmul
        Tensor<T,3> e2 = matmul(matmul(a,b),d);
        Tensor<T,3> e3 = a % b % d;
        FASTOR_DOES_CHECK_PASS(std::abs(e3.sum() - e2.sum()) < BigTol);

        e2 += matmul(matmul(a,b),d);
        e3 += a % b % d;
        FASTOR_DOES_CHECK_PASS(std::abs(e3.sum() - e2.sum()) < BigTol);

        e2 -= matmul(matmul(a,b),d);
        e3 -= a % b % d;
        FASTOR_DOES_CHECK_PASS(std::abs(e3.sum() - e2.sum()) < BigTol);

        e2 *= matmul(matmul(a,b),d);
        e3 *= a % b % d;
        FASTOR_DOES_CHECK_PASS(std::abs(e3.sum() - e2.sum()) < BigTol);

        e2 /= matmul(matmul(a,b),d);
        e3 /= a % b % d;
        FASTOR_DOES_CHECK_PASS(std::abs(e3.sum() - e2.sum()) < BigTol);

        // double matmul expr
        e2 = 1 + matmul(matmul(a,b),d);
        e3 = 1 + a % b % d;
        FASTOR_DOES_CHECK_PASS(std::abs(e3.sum() - e2.sum()) < BigTol);

        e2 += matmul(matmul(a,b),d) - 3;
        e3 += a % b % d - 3;
        FASTOR_DOES_CHECK_PASS(std::abs(e3.sum() - e2.sum()) < BigTol);

        e2 -= matmul(matmul(a,b),d) * e2;
        e3 -= (a % b % d) * e3;
        FASTOR_DOES_CHECK_PASS(std::abs(e3.sum() - e2.sum()) < BigTol);

        e2 *= matmul(matmul(a,b),d) / e2;
        e3 *= (a % b % d) / e3;
        FASTOR_DOES_CHECK_PASS(std::abs(e3.sum() - e2.sum()) < BigTol);

        e2 /= 2*matmul(matmul(a,b),d) + 2*e2 + e2 + 1;
        e3 /= 2*(a % b % d) + 2*e3 + e3 + 1;
        FASTOR_DOES_CHECK_PASS(std::abs(e3.sum() - e2.sum()) < BigTol);
    }


    print(FGRN(BOLD("All tests passed successfully")));

}

int main() {

    print(FBLU(BOLD("Testing matmul expression with double precision")));
    run<double>();

    return 0;
}
