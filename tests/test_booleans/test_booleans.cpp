#include <Fastor/Fastor.h>

using namespace Fastor;


#define Tol 1e-12
#define BigTol 1e-5


template<typename T>
void test_boolean() {

    // The test assumes that trans function is already
    // tested elsewhere

    // is square/uniform
    {
        Tensor<T,3,3> a;
        Tensor<T,3,4> b;
        Tensor<T,4,4,4> c;
        Tensor<T,4,3,4> d;

        FASTOR_EXIT_ASSERT(issquare(a)        == true);
        FASTOR_EXIT_ASSERT(issquare(a+0)      == true);
        FASTOR_EXIT_ASSERT(issquare(trans(a)) == true);
        FASTOR_EXIT_ASSERT(issquare(b)        == false);
        FASTOR_EXIT_ASSERT(issquare(b+0)      == false);

        FASTOR_EXIT_ASSERT(isuniform(a)       == true);
        FASTOR_EXIT_ASSERT(isuniform(c)       == true);
        FASTOR_EXIT_ASSERT(isuniform(c+0)     == true);
        FASTOR_EXIT_ASSERT(isuniform(d)       == false);
        FASTOR_EXIT_ASSERT(isuniform(d+0)     == false);
    }

    // issymmetric
    {
        Tensor<T,3,3> a;
        a.iota();

        FASTOR_EXIT_ASSERT(issymmetric(a)        == false);

        a += transpose(a);
        FASTOR_EXIT_ASSERT(issymmetric(a)        == true);
        FASTOR_EXIT_ASSERT(issymmetric(a+1)      == true);
        FASTOR_EXIT_ASSERT(issymmetric(trans(a)) == true);
    }

    // Orthogonal SL3/SO3
    {
        Tensor<T,3,3> a; a.eye2();
        Tensor<T,3,3> b; b.iota();

        FASTOR_EXIT_ASSERT(isorthogonal(a)       == true);
        FASTOR_EXIT_ASSERT(isorthogonal(a+0)     == true);
        FASTOR_EXIT_ASSERT(isorthogonal(trans(a))== true);
        FASTOR_EXIT_ASSERT(isorthogonal(b)       == false);

        FASTOR_EXIT_ASSERT(doesbelongtoSL3(a)   == true);
        FASTOR_EXIT_ASSERT(doesbelongtoSL3(a*1) == true);
        FASTOR_EXIT_ASSERT(doesbelongtoSL3(b)   == false);

        FASTOR_EXIT_ASSERT(doesbelongtoSO3(a)   == true);
        FASTOR_EXIT_ASSERT(doesbelongtoSO3(a*1) == true);
        FASTOR_EXIT_ASSERT(doesbelongtoSO3(b)   == false);
    }

    // isequal
    {
        Tensor<T,3,3> a; a.eye2();
        Tensor<T,3,3> b; b.iota();

        FASTOR_EXIT_ASSERT(isequal(a,a)                     == true);
        FASTOR_EXIT_ASSERT(isequal(a,a+0)                   == true);
        FASTOR_EXIT_ASSERT(isequal(a+0,a)                   == true);
        FASTOR_EXIT_ASSERT(isequal(a+0,a+0)                 == true);
        FASTOR_EXIT_ASSERT(isequal(a+0,trans(a))            == true);
        FASTOR_EXIT_ASSERT(isequal(a,b)                     == false);
        FASTOR_EXIT_ASSERT(isequal(b,b)                     == true);
        FASTOR_EXIT_ASSERT(isequal(b,b+1)                   == false);
        FASTOR_EXIT_ASSERT(isequal(trans(b),trans(b))       == true);
    }


    print(FGRN(BOLD("All tests passed successfully")));

}


int main() {

    print(FBLU(BOLD("Testing boolean tensor routines with single precision")));
    test_boolean<float>();
    print(FBLU(BOLD("Testing boolean tensor routines with double precision")));
    test_boolean<double>();

    return 0;
}

