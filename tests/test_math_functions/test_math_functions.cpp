#include <Fastor/Fastor.h>

using namespace Fastor;


#define Tol    1e-08
#define BigTol 1e-04


template<typename T>
void test_math_functions() {

    {
        Tensor<T,2,2,2,2> t11; t11.fill(16);
        FASTOR_EXIT_ASSERT(std::abs(norm(sqrt(t11)) - 16       ) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(cbrt(t11)) - 10.079368399158986    ) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(log(t11))  - 11.090354888959125) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(log10(t11)) - 4.816479930623699) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(log2(t11))  - 16) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(exp(t11/100.))  - 4.694043483967241) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(sin(t11))  - 1.1516132) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(cos(t11))  - 3.8306379) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(tan(t11))  - 1.2025289) < BigTol);

        Tensor<T,2,2,2,2> t12 = sqrt(t11 + t11 - 2*t11 + t11/2 + 16/t11);
        FASTOR_EXIT_ASSERT(std::abs(norm(t12)- 12) < BigTol);

        Tensor<T,10> t13;
        t13.iota(1);
        FASTOR_EXIT_ASSERT(std::abs(t13.product() - 3628800) < Tol);
        t13.iota(0);
        FASTOR_EXIT_ASSERT(std::abs(t13.product()) < Tol);
    }

    {
        Tensor<T,5,5> t; t.arange();
        FASTOR_EXIT_ASSERT(std::abs(sum(sinh(t/T(100))) - 3.015030007092837 ) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(cosh(t/T(100))) - 25.2457356407902  ) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(tanh(t/T(100))) - 2.9704710425517478) < BigTol);

        FASTOR_EXIT_ASSERT(std::abs(sum(asinh(t/T(100))) - 2.9852627858720218) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(acosh(t + T(1))) - 74.478917883212   ) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(atanh(t/T(100))) - 3.0307433885849693) < BigTol);
    }

    {
        Tensor<T,20> t; t.iota(T(1.3));
        FASTOR_EXIT_ASSERT(std::abs(sum(floor(t)) - 210) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(ceil (t)) - 230) < BigTol);
        FASTOR_EXIT_ASSERT(std::abs(sum(round(t)) - 210) < BigTol);
    }

    print(FGRN(BOLD("All tests passed successfully")));

}

int main() {

    print(FBLU(BOLD("Testing tensor math functions: single precision")));
    test_math_functions<float>();
    print(FBLU(BOLD("Testing tensor math functions: double precision")));
    test_math_functions<double>();

    return 0;
}

