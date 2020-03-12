#include <Fastor/Fastor.h>

using namespace Fastor;

#define Tol 1e-12


template<typename T, int ABI>
void test_intergers_divs();

template<>
void test_intergers_divs<int,256>() {

    using TT = int;
    constexpr int ABI = 256;
        std::array<TT,8> arr = {70,3,6,1,5,9,14,20};
    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= 2;
        FASTOR_EXIT_ASSERT((a.sum() - 62)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= a;
        FASTOR_EXIT_ASSERT((a.sum() - 8)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= a.value;
        FASTOR_EXIT_ASSERT((a.sum() - 8)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        SIMDVector<TT,ABI> b(arr.data(),false);
        SIMDVector<TT,ABI> aa = a / b;
        FASTOR_EXIT_ASSERT((aa.sum() - 8)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        SIMDVector<TT,ABI> aa = a / (TT)2;
        FASTOR_EXIT_ASSERT((aa.sum() - 62)< Tol, "TEST FAILED");
        SIMDVector<TT,ABI> bb = (TT)100 / a;
        FASTOR_EXIT_ASSERT((bb.sum() - 193)< Tol, "TEST FAILED");
    }

    print(FGRN(BOLD("All tests passed successfully")));
}

template<>
void test_intergers_divs<int,128>() {

    using TT = int;
    constexpr int ABI = 128;
        std::array<TT,4> arr = {70,3,6,1};
    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= 2;
        FASTOR_EXIT_ASSERT((a.sum() - 39)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= a;
        FASTOR_EXIT_ASSERT((a.sum() - 4)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= a.value;
        FASTOR_EXIT_ASSERT((a.sum() - 4)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        SIMDVector<TT,ABI> b(arr.data(),false);
        SIMDVector<TT,ABI> aa = a / b;
        FASTOR_EXIT_ASSERT((aa.sum() - 4)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        SIMDVector<TT,ABI> aa = a / (TT)2;
        FASTOR_EXIT_ASSERT((aa.sum() - 39)< Tol, "TEST FAILED");
        SIMDVector<TT,ABI> bb = (TT)100 / a;
        FASTOR_EXIT_ASSERT((bb.sum() - 150)< Tol, "TEST FAILED");
    }

    print(FGRN(BOLD("All tests passed successfully")));
}

template<>
void test_intergers_divs<Int64,256>() {

    using TT = Int64;
    constexpr int ABI = 256;
    std::array<TT,4> arr = {5,9,14,20};
    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= 2;
        FASTOR_EXIT_ASSERT((a.sum() - 23)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= a;
        FASTOR_EXIT_ASSERT((a.sum() - 4)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= a.value;
        FASTOR_EXIT_ASSERT((a.sum() - 4)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        SIMDVector<TT,ABI> b(arr.data(),false);
        SIMDVector<TT,ABI> aa = a / b;
        FASTOR_EXIT_ASSERT((aa.sum() - 4)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        SIMDVector<TT,ABI> aa = a / (TT)2;
        FASTOR_EXIT_ASSERT((aa.sum() - 23)< Tol, "TEST FAILED");
        SIMDVector<TT,ABI> bb = (TT)100 / a;
        FASTOR_EXIT_ASSERT((bb.sum() - 43)< Tol, "TEST FAILED");
    }

    print(FGRN(BOLD("All tests passed successfully")));
}

template<>
void test_intergers_divs<Int64,128>() {

    using TT = Int64;
    constexpr int ABI = 128;
    std::array<TT,2> arr = {14,20};
    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= 2;
        FASTOR_EXIT_ASSERT((a.sum() - 17)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= a;
        FASTOR_EXIT_ASSERT((a.sum() - 2)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= a.value;
        FASTOR_EXIT_ASSERT((a.sum() - 2)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        SIMDVector<TT,ABI> b(arr.data(),false);
        SIMDVector<TT,ABI> aa = a / b;
        FASTOR_EXIT_ASSERT((aa.sum() - 2)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        SIMDVector<TT,ABI> aa = a / (TT)2;
        FASTOR_EXIT_ASSERT((aa.sum() - 17)< Tol, "TEST FAILED");
        SIMDVector<TT,ABI> bb = (TT)100 / a;
        FASTOR_EXIT_ASSERT((bb.sum() - 12)< Tol, "TEST FAILED");
    }

    print(FGRN(BOLD("All tests passed successfully")));
}




template<typename T, int ABI>
void test_simd_vectors() {

    SIMDVector<T,ABI> t1, t2;
    t1.set_sequential(1); t2.set_sequential(100);
    auto t3 = t1+t2; auto tester = std::pow(t1.Size+50,2) - 2500;
    FASTOR_EXIT_ASSERT((t3.sum() - tester)< Tol, "TEST FAILED");

    auto t4 = T(1)*t2-t1+(T)1-(T)0;
    FASTOR_EXIT_ASSERT((t4.sum() - 100*t4.Size)< Tol, "TEST FAILED");

    auto n = t1.Size;
    FASTOR_EXIT_ASSERT((t1.dot(t1) - n*(2*n*n+3*n+1)/6)< Tol, "TEST FAILED");

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


    print(FBLU(BOLD("Testing SIMDVector of int for division - 128")));
    test_intergers_divs<int,128>();
    print(FBLU(BOLD("Testing SIMDVector of int for division - 256")));
    test_intergers_divs<int,256>();

    print(FBLU(BOLD("Testing SIMDVector of long long for division - 128")));
    test_intergers_divs<Int64,128>();
    print(FBLU(BOLD("Testing SIMDVector of long long for division - 256")));
    test_intergers_divs<Int64,256>();

    return 0;
}