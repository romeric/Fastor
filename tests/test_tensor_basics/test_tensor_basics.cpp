#include <Fastor/Fastor.h>

using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5


template<typename T>
void test_basics() {

    {
        // Check initialiser list constructors
        Tensor<T,3> a0 = {1,2,3};
        Tensor<T,3,1> a1 = {{1},{2},{3}};
        Tensor<T,1,3> a2 = {{1,2,3}};
        Tensor<T,1,2,3> a3 = {{{1,2,3},{4,5,6}}};

        // Basic scalar indexing
        FASTOR_EXIT_ASSERT(std::abs(a0(0)-1)<Tol);
        FASTOR_EXIT_ASSERT(std::abs(a0(1)-2)<Tol);
        FASTOR_EXIT_ASSERT(std::abs(a0(2)-3)<Tol);

        FASTOR_EXIT_ASSERT(std::abs(a1(0,0)-1)<Tol);
        FASTOR_EXIT_ASSERT(std::abs(a1(1,0)-2)<Tol);
        FASTOR_EXIT_ASSERT(std::abs(a1(2,0)-3)<Tol);

        FASTOR_EXIT_ASSERT(std::abs(a2(0,0)-1)<Tol);
        FASTOR_EXIT_ASSERT(std::abs(a2(0,1)-2)<Tol);
        FASTOR_EXIT_ASSERT(std::abs(a2(0,2)-3)<Tol);

        FASTOR_EXIT_ASSERT(std::abs(a3(0,0,0)-1)<Tol);
        FASTOR_EXIT_ASSERT(std::abs(a3(0,0,1)-2)<Tol);
        FASTOR_EXIT_ASSERT(std::abs(a3(0,0,2)-3)<Tol);

        FASTOR_EXIT_ASSERT(std::abs(a3(0,1,0)-4)<Tol);
        FASTOR_EXIT_ASSERT(std::abs(a3(0,1,1)-5)<Tol);
        FASTOR_EXIT_ASSERT(std::abs(a3(0,1,2)-6)<Tol);

        FASTOR_EXIT_ASSERT(std::abs(a3.sum()     - 21  ) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(a3.product() - 720 ) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(sum(a3)      - 21  ) < Tol);
        FASTOR_EXIT_ASSERT(std::abs(product(a3)  - 720 ) < Tol);
    }

    // Scalar indexing
    {
        Tensor<T,2,3,4,5,6> a; a.iota(0);
        size_t counter = 0;
        for (size_t i=0; i<2; ++i) {
            for (size_t j=0; j<3; ++j) {
                for (size_t k=0; k<4; ++k) {
                    for (size_t l=0; l<5; ++l) {
                        for (size_t m=0; m<6; ++m) {
                            FASTOR_EXIT_ASSERT(std::abs(a(i,j,k,l,m)-counter)<Tol);
                            counter++;
                        }
                    }
                }
            }
        }
    }

    {
        T number = T(12.67);
        Tensor<T> a0 = T(12.67);
        FASTOR_EXIT_ASSERT(std::fabs(norm(a0)-number)<Tol);

        Tensor<T,4> a1 = 1;
        FASTOR_EXIT_ASSERT(std::fabs(norm(a1)-2)<Tol);

        Tensor<T,4,4> a2 = 1;
        FASTOR_EXIT_ASSERT(std::fabs(norm(a2)-4)<Tol);

        Tensor<T,2,2,4> a3 = 1;
        FASTOR_EXIT_ASSERT(std::fabs(norm(a3)-4)<Tol);

        Tensor<T,2,2,4,4> a4 = 1;
        FASTOR_EXIT_ASSERT(std::fabs(norm(a4)-8)<Tol);

        Tensor<T,2,2,4,4,4> a5 = 2;
        FASTOR_EXIT_ASSERT(std::fabs(norm(a5)-32)<Tol);

        Tensor<T,2,2,4,4,4,4> a6 = 2;
        FASTOR_EXIT_ASSERT(std::fabs(norm(a6)-64)<Tol);


        Tensor<T,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2> t1;
        FASTOR_EXIT_ASSERT(t1.rank()==17);
        FASTOR_EXIT_ASSERT(t1.size()==131072);
        FASTOR_EXIT_ASSERT(t1.dimension(10)==t1.dimension(0));


        Tensor<T,3,4,5> t2, t3;
        t2.fill(3); t3.fill(3);
        Tensor<T,3,4,5> t4 = t2+t3;
        Tensor<T,3,4,5> t5 = t2+t3 - (t2+t3);
        Tensor<T,3,4,5> t6 = t2*t3;
        Tensor<T,3,4,5> t7 = t2/t3;
        Tensor<T,3,4,5> t8 = t2+3;
        Tensor<T,3,4,5> t9 = 1.5+t2+1.5;
        Tensor<T,3,4,5> t10 = (t2*t3)/9;

        FASTOR_EXIT_ASSERT(std::abs(norm(t4) - 46.4758001)<BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(t5))<Tol);
        FASTOR_EXIT_ASSERT(std::abs(norm(t6) - 69.713700)<BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(t7) - 7.7459667)<BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(t8) - 46.4758001)<BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(t9) - 46.4758001)<BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(t10) - 7.7459667)<BigTol);


        Tensor<T,2,2,2,2> t11; t11.fill(16);
        FASTOR_EXIT_ASSERT(std::abs(norm(static_cast<decltype(t11)>(sqrt(t11)))- 16)<BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(static_cast<decltype(t11)>(log(t11)))- 11.090354)<BigTol);
        // FASTOR_EXIT_ASSERT(std::abs(norm(static_cast<decltype(t11)>(exp(t11)))- 3.55444208203e+07)<1e-05);
        FASTOR_EXIT_ASSERT(std::abs(norm(static_cast<decltype(t11)>(sin(t11)))- 1.1516132)<BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(static_cast<decltype(t11)>(cos(t11)))- 3.8306379)<BigTol);
        FASTOR_EXIT_ASSERT(std::abs(norm(static_cast<decltype(t11)>(tan(t11)))- 1.2025289)<BigTol);

        Tensor<T,2,2,2,2> t12 = sqrt(t11 + t11 - 2*t11 + t11/2 + 16/t11);
        FASTOR_EXIT_ASSERT(std::abs(norm(t12)- 12)<BigTol);

        Tensor<T,10> t13;
        t13.iota(1);
        FASTOR_EXIT_ASSERT(std::abs(t13.product() - 3628800) < Tol);
        t13.iota(0);
        FASTOR_EXIT_ASSERT(std::abs(t13.product()) < Tol);

    }

    // check numbers assignment
    {
        Tensor<T,3,3> a0 = 2;
        FASTOR_EXIT_ASSERT(std::abs(a0.sum()-18) < Tol);
        a0 += 2;
        FASTOR_EXIT_ASSERT(std::abs(a0.sum()-36) < Tol);
        a0 -= 2;
        FASTOR_EXIT_ASSERT(std::abs(a0.sum()-18) < Tol);
        a0 *= 2;
        FASTOR_EXIT_ASSERT(std::abs(a0.sum()-36) < Tol);
        a0 /= 2;
        FASTOR_EXIT_ASSERT(std::abs(a0.sum()-18) < Tol);
    }

    print(FGRN(BOLD("All tests passed successfully")));

}

int main() {

    print(FBLU(BOLD("Testing basic tensor construction routines with single precision")));
    test_basics<float>();
    print(FBLU(BOLD("Testing basic tensor construction routines with double precision")));
    test_basics<double>();

    return 0;
}

