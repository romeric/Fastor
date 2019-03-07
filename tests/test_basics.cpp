#include <Fastor.h>

using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5


template<typename T>
void test_basics() {

    T number = 12.67;
    Tensor<T> a0 = 12.67;
    assert(std::fabs(norm(a0)-number)<Tol);

    Tensor<T,4> a1 = 1;
    assert(std::fabs(norm(a1)-2)<Tol);

    Tensor<T,4,4> a2 = 1;
    assert(std::fabs(norm(a2)-4)<Tol);

    Tensor<T,2,2,4> a3 = 1;
    assert(std::fabs(norm(a3)-4)<Tol);

    Tensor<T,2,2,4,4> a4 = 1;
    assert(std::fabs(norm(a4)-8)<Tol);

    Tensor<T,2,2,4,4,4> a5 = 2;
    assert(std::fabs(norm(a5)-32)<Tol);

    Tensor<T,2,2,4,4,4,4> a6 = 2;
    assert(std::fabs(norm(a6)-64)<Tol);


    Tensor<T,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2> t1;
    assert(t1.rank()==17);
    assert(t1.size()==131072);
    assert(t1.dimension(10)==t1.dimension(0));


    Tensor<T,3,4,5> t2, t3;
    t2.fill(3); t3.fill(3);
    Tensor<T,3,4,5> t4 = t2+t3;
    Tensor<T,3,4,5> t5 = t2+t3 - (t2+t3);
    Tensor<T,3,4,5> t6 = t2*t3;
    Tensor<T,3,4,5> t7 = t2/t3;
    Tensor<T,3,4,5> t8 = t2+3;
    Tensor<T,3,4,5> t9 = 1.5+t2+1.5;
    Tensor<T,3,4,5> t10 = (t2*t3)/9;

    assert(std::abs(norm(t4) - 46.4758001)<BigTol);
    assert(std::abs(norm(t5))<Tol);
    assert(std::abs(norm(t6) - 69.713700)<BigTol);
    assert(std::abs(norm(t7) - 7.7459667)<BigTol);
    assert(std::abs(norm(t8) - 46.4758001)<BigTol);
    assert(std::abs(norm(t9) - 46.4758001)<BigTol);
    assert(std::abs(norm(t10) - 7.7459667)<BigTol);


    Tensor<T,2,2,2,2> t11; t11.fill(16);
    assert(std::abs(norm(static_cast<decltype(t11)>(sqrt(t11)))- 16)<BigTol);
    assert(std::abs(norm(static_cast<decltype(t11)>(log(t11)))- 11.090354)<BigTol);
    // assert(std::abs(norm(static_cast<decltype(t11)>(exp(t11)))- 3.55444208203e+07)<1e-05);
    assert(std::abs(norm(static_cast<decltype(t11)>(sin(t11)))- 1.1516132)<BigTol);
    assert(std::abs(norm(static_cast<decltype(t11)>(cos(t11)))- 3.8306379)<BigTol);
    assert(std::abs(norm(static_cast<decltype(t11)>(tan(t11)))- 1.2025289)<BigTol);

    Tensor<T,2,2,2,2> t12 = sqrt(t11 + t11 - 2*t11 + t11/2 + 16/t11);
    assert(std::abs(norm(t12)- 12)<BigTol);

    Tensor<T,10> t13;
    t13.iota(1);
    assert(std::abs(t13.product() - 3628800) < Tol);
    t13.iota(0);
    assert(std::abs(t13.product()) < Tol);

    print(FGRN(BOLD("All tests passed successfully")));

}

int main() {

    print(FBLU(BOLD("Testing basic tensor construction routines with single precision")));
    test_basics<float>();
    print(FBLU(BOLD("Testing basic tensor construction routines with double precision")));
    test_basics<double>();

    return 0;
}

