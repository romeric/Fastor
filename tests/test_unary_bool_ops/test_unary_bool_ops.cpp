#include <Fastor/Fastor.h>

using namespace Fastor;
namespace ft = Fastor;


template<typename T, FASTOR_INDEX mm, FASTOR_INDEX nn>
void run_fixed_size() {

    constexpr seq sall = seq(0,-1,1);

    {
        Tensor<T,mm,nn> aa; aa.iota(2);

        Tensor<bool,mm,nn> ba0 = ft::isnan(aa);
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_DOES_CHECK_PASS(ba0(i,j) == false, "TEST FAILED");
            }
        }

        FASTOR_DOES_CHECK_PASS(all_of (ba0 == false) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(any_of (ba0 == false) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(none_of(ba0 == false) == true, "TEST FAILED");

        Tensor<bool,mm,nn> ba1 = ft::isinf(aa);
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_DOES_CHECK_PASS(ba1(i,j) == false, "TEST FAILED");
            }
        }

        FASTOR_DOES_CHECK_PASS(all_of (ba1 == false) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(any_of (ba1 == false) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(none_of(ba1 == false) == true, "TEST FAILED");

        Tensor<bool,mm,nn> ba2 = ft::isfinite(aa);
        for (FASTOR_INDEX i=0; i<mm; ++i) {
            for (FASTOR_INDEX j=0; j<nn; ++j) {
                FASTOR_DOES_CHECK_PASS(ba2(i,j) == true, "TEST FAILED");
            }
        }

        FASTOR_DOES_CHECK_PASS(all_of (ba2 == true) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(any_of (ba2 == true) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(none_of(ba2 == true) == true, "TEST FAILED");

        // requires evaluation
        FASTOR_DOES_CHECK_PASS(all_of (ft::isinf(trans(aa)) == false ) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(any_of (ft::isinf(trans(aa)) == false ) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(none_of(ft::isinf(trans(aa)) == false ) == true, "TEST FAILED");

        FASTOR_DOES_CHECK_PASS(all_of (ft::isnan(trans(aa)) == false ) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(any_of (ft::isnan(trans(aa)) == false ) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(none_of(ft::isnan(trans(aa)) == false ) == true, "TEST FAILED");

        FASTOR_DOES_CHECK_PASS(all_of (ft::isfinite(trans(aa)) == true ) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(any_of (ft::isfinite(trans(aa)) == true ) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(none_of(ft::isfinite(trans(aa)) == true ) == true, "TEST FAILED");

        // check nested expressions
        FASTOR_DOES_CHECK_PASS(all_of (!ft::isnan(trans(aa)) == true     ) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(any_of (!ft::isinf(trans(aa)) == true     ) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(none_of(!ft::isfinite(trans(aa)) == false ) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(all_of (!(aa!=aa) == true     ) == true, "TEST FAILED");
    }

    {
        Tensor<T,mm,nn> aa; aa.iota(2);
        aa(0,1) = NAN;
        aa(1,1) = INFINITY;

        Tensor<bool,mm,nn> ba0 = ft::isnan(aa);
        FASTOR_DOES_CHECK_PASS(ba0(0,0) == false, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(ba0(0,1) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(ba0(1,1) == false, "TEST FAILED");

        FASTOR_DOES_CHECK_PASS(all_of (ba0) == false, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(any_of (ba0) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(none_of(ba0) == true, "TEST FAILED");

        Tensor<bool,mm,nn> ba1 = ft::isinf(aa);
        FASTOR_DOES_CHECK_PASS(ba1(0,0) == false, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(ba1(0,1) == false, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(ba1(1,1) == true, "TEST FAILED");

        FASTOR_DOES_CHECK_PASS(all_of (ba1) == false, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(any_of (ba1) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(none_of(ba1) == true, "TEST FAILED");

        Tensor<bool,mm,nn> ba2 = ft::isfinite(aa);
        FASTOR_DOES_CHECK_PASS(ba2(0,0) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(ba2(0,1) == false, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(ba2(1,1) == false, "TEST FAILED");

        FASTOR_DOES_CHECK_PASS(all_of (ba2) == false, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(any_of (ba2) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(none_of(ba2) == true, "TEST FAILED");
    }

    // views
    {
        Tensor<T,mm,nn> aa; aa.iota(2);
        aa(0,1) = NAN;
        aa(1,1) = INFINITY;

        Tensor<bool,mm,nn> ba0 = ft::isnan(aa(sall,sall));
        FASTOR_DOES_CHECK_PASS(ba0(0,0) == false, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(ba0(0,1) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(ba0(1,1) == false, "TEST FAILED");

        FASTOR_DOES_CHECK_PASS(all_of (ba0) == false, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(any_of (ba0) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(none_of(ba0) == true, "TEST FAILED");

        Tensor<bool,mm,nn> ba1 = ft::isinf(aa(sall,sall));
        FASTOR_DOES_CHECK_PASS(ba1(0,0) == false, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(ba1(0,1) == false, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(ba1(1,1) == true, "TEST FAILED");

        FASTOR_DOES_CHECK_PASS(all_of (ba1) == false, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(any_of (ba1) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(none_of(ba1) == true, "TEST FAILED");

        Tensor<bool,mm,nn> ba2 = ft::isfinite(aa(sall,sall));
        FASTOR_DOES_CHECK_PASS(ba2(0,0) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(ba2(0,1) == false, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(ba2(1,1) == false, "TEST FAILED");

        FASTOR_DOES_CHECK_PASS(all_of (ba2) == false, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(any_of (ba2) == true, "TEST FAILED");
        FASTOR_DOES_CHECK_PASS(none_of(ba2) == true, "TEST FAILED");
    }


}


template<typename T>
void run() {
    run_fixed_size<T,2,2>();
    print(FGRN(BOLD("All tests passed successfully")));
    run_fixed_size<T,4,4>();
    print(FGRN(BOLD("All tests passed successfully")));
    run_fixed_size<T,2,8>();
    print(FGRN(BOLD("All tests passed successfully")));
    run_fixed_size<T,3,13>();
    print(FGRN(BOLD("All tests passed successfully")));
}


int main() {
    print(FBLU(BOLD("Testing binary comparison operators: single precision")));
    run<float>();
    print(FBLU(BOLD("Testing binary comparison operators: double precision")));
    run<double>();
}

