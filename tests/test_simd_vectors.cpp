#include "Fastor.h"

using namespace Fastor;

#define Tol 1e-12


template<typename T, int ABI>
void test_simd_vectors() {

    SIMDVector<T,ABI> t1, t2;
    t1.set_sequential(1); t2.set_sequential(100);
    auto t3 = t1+t2; auto tester = std::pow(t1.Size+50,2) - 2500;
    FASTOR_ASSERT((t3.sum() - tester)< Tol, "TEST FAILED");

    auto t4 = T(1)*t2-t1+(T)1-(T)0;
    FASTOR_ASSERT((t4.sum() - 100*t4.Size)< Tol, "TEST FAILED");

    auto n = t1.Size;
    FASTOR_ASSERT((t1.dot(t1) - n*(2*n*n+3*n+1)/6)< Tol, "TEST FAILED");

    print(FGRN(BOLD("All tests passed successfully")));

}

int main() {

    print(FBLU(BOLD("Testing SIMDVector of single precision - 32")));
    test_simd_vectors<float,32>();
    print(FBLU(BOLD("Testing SIMDVector of single precision - 64")));
    test_simd_vectors<float,64>();
    print(FBLU(BOLD("Testing SIMDVector of single precision - 128")));
    test_simd_vectors<float,128>();
    print(FBLU(BOLD("Testing SIMDVector of single precision - 256")));
    test_simd_vectors<float,256>();
    print(FBLU(BOLD("Testing SIMDVector of single precision - 512")));
    test_simd_vectors<float,512>();

    print(FBLU(BOLD("Testing SIMDVector of double precision - 64")));
    test_simd_vectors<double,64>();
    print(FBLU(BOLD("Testing SIMDVector of double precision - 128")));
    test_simd_vectors<double,128>();
    print(FBLU(BOLD("Testing SIMDVector of double precision - 256")));
    test_simd_vectors<double,256>();
    print(FBLU(BOLD("Testing SIMDVector of double precision - 512")));
    test_simd_vectors<double,512>();

    print(FBLU(BOLD("Testing SIMDVector of int - 32")));
    test_simd_vectors<int,32>();
    print(FBLU(BOLD("Testing SIMDVector of int - 64")));
    test_simd_vectors<int,64>();
    print(FBLU(BOLD("Testing SIMDVector of int - 128")));
    test_simd_vectors<int,128>();
    print(FBLU(BOLD("Testing SIMDVector of int - 256")));
    test_simd_vectors<int,256>();
    print(FBLU(BOLD("Testing SIMDVector of int - 512")));
    test_simd_vectors<int,512>();

    print(FBLU(BOLD("Testing SIMDVector of long long - 64")));
    test_simd_vectors<Int64,64>();
    print(FBLU(BOLD("Testing SIMDVector of long long - 128")));
    test_simd_vectors<Int64,128>();
    print(FBLU(BOLD("Testing SIMDVector of long long - 256")));
    test_simd_vectors<Int64,256>();
    print(FBLU(BOLD("Testing SIMDVector of long long - 512")));
    test_simd_vectors<Int64,512>();

    return 0;
}