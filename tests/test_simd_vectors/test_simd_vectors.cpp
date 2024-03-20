#include <Fastor/Fastor.h>
#include <complex>

using namespace Fastor;

#define Tol 1e-12


template<typename T, typename ABI>
void test_intergers_divs();

template<>
void test_intergers_divs<int,simd_abi::avx>() {

    using TT = int;
    using ABI = simd_abi::avx;
    std::array<TT,8> arr = {70,3,6,1,5,9,14,20};
    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= 2;
        FASTOR_DOES_CHECK_PASS((a.sum() - 62)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= a;
        FASTOR_DOES_CHECK_PASS((a.sum() - 8)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= a.value;
        FASTOR_DOES_CHECK_PASS((a.sum() - 8)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        SIMDVector<TT,ABI> b(arr.data(),false);
        SIMDVector<TT,ABI> aa = a / b;
        FASTOR_DOES_CHECK_PASS((aa.sum() - 8)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        SIMDVector<TT,ABI> aa = a / (TT)2;
        FASTOR_DOES_CHECK_PASS((aa.sum() - 62)< Tol, "TEST FAILED");
        SIMDVector<TT,ABI> bb = (TT)100 / a;
        FASTOR_DOES_CHECK_PASS((bb.sum() - 193)< Tol, "TEST FAILED");
    }

    print(FGRN(BOLD("All tests passed successfully")));
}

template<>
void test_intergers_divs<int,simd_abi::sse>() {

    using TT = int;
    using ABI = simd_abi::sse;
        std::array<TT,4> arr = {70,3,6,1};
    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= 2;
        FASTOR_DOES_CHECK_PASS((a.sum() - 39)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= a;
        FASTOR_DOES_CHECK_PASS((a.sum() - 4)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= a.value;
        FASTOR_DOES_CHECK_PASS((a.sum() - 4)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        SIMDVector<TT,ABI> b(arr.data(),false);
        SIMDVector<TT,ABI> aa = a / b;
        FASTOR_DOES_CHECK_PASS((aa.sum() - 4)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        SIMDVector<TT,ABI> aa = a / (TT)2;
        FASTOR_DOES_CHECK_PASS((aa.sum() - 39)< Tol, "TEST FAILED");
        SIMDVector<TT,ABI> bb = (TT)100 / a;
        FASTOR_DOES_CHECK_PASS((bb.sum() - 150)< Tol, "TEST FAILED");
    }

    print(FGRN(BOLD("All tests passed successfully")));
}

template<>
void test_intergers_divs<Int64,simd_abi::avx>() {

    using TT = Int64;
    using ABI = simd_abi::avx;
    std::array<TT,4> arr = {5,9,14,20};
    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= 2;
        FASTOR_DOES_CHECK_PASS((a.sum() - 23)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= a;
        FASTOR_DOES_CHECK_PASS((a.sum() - 4)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= a.value;
        FASTOR_DOES_CHECK_PASS((a.sum() - 4)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        SIMDVector<TT,ABI> b(arr.data(),false);
        SIMDVector<TT,ABI> aa = a / b;
        FASTOR_DOES_CHECK_PASS((aa.sum() - 4)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        SIMDVector<TT,ABI> aa = a / (TT)2;
        FASTOR_DOES_CHECK_PASS((aa.sum() - 23)< Tol, "TEST FAILED");
        SIMDVector<TT,ABI> bb = (TT)100 / a;
        FASTOR_DOES_CHECK_PASS((bb.sum() - 43)< Tol, "TEST FAILED");
    }

    print(FGRN(BOLD("All tests passed successfully")));
}

template<>
void test_intergers_divs<Int64,simd_abi::sse>() {

    using TT = Int64;
    using ABI = simd_abi::sse;
    std::array<TT,2> arr = {14,20};
    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= 2;
        FASTOR_DOES_CHECK_PASS((a.sum() - 17)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= a;
        FASTOR_DOES_CHECK_PASS((a.sum() - 2)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        a /= a.value;
        FASTOR_DOES_CHECK_PASS((a.sum() - 2)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        SIMDVector<TT,ABI> b(arr.data(),false);
        SIMDVector<TT,ABI> aa = a / b;
        FASTOR_DOES_CHECK_PASS((aa.sum() - 2)< Tol, "TEST FAILED");
    }

    {
        SIMDVector<TT,ABI> a(arr.data(),false);
        SIMDVector<TT,ABI> aa = a / (TT)2;
        FASTOR_DOES_CHECK_PASS((aa.sum() - 17)< Tol, "TEST FAILED");
        SIMDVector<TT,ABI> bb = (TT)100 / a;
        FASTOR_DOES_CHECK_PASS((bb.sum() - 12)< Tol, "TEST FAILED");
    }

    print(FGRN(BOLD("All tests passed successfully")));
}




template<typename T, typename ABI>
void test_simd_vectors() {

    SIMDVector<T,ABI> t1, t2;
    t1.set_sequential(1); t2.set_sequential(100);
    auto t3 = t1+t2; auto tester = std::pow(t1.Size+50,2) - 2500;
    FASTOR_DOES_CHECK_PASS((t3.sum() - tester)< Tol, "TEST FAILED");

    auto t4 = T(1)*t2-t1+(T)1-(T)0;
    FASTOR_DOES_CHECK_PASS((t4.sum() - 100*t4.Size)< Tol, "TEST FAILED");

    auto n = t1.Size;
    FASTOR_DOES_CHECK_PASS((t1.dot(t1) - n*(2*n*n+3*n+1)/6)< Tol, "TEST FAILED");

#if defined(FASTOR_AVX_IMPL) && !defined(FASTOR_AVX512_IMPL)
    FASTOR_DOES_CHECK_PASS(SIMDVector<double>::size()==4);
    FASTOR_DOES_CHECK_PASS(SIMDVector<double,simd_abi::avx>::size()==4);
    FASTOR_DOES_CHECK_PASS(SIMDVector<double,simd_abi::sse>::size()==2);
    FASTOR_DOES_CHECK_PASS(SIMDVector<double,simd_abi::scalar>::size()==1);

    FASTOR_DOES_CHECK_PASS(SIMDVector<float>::size()==8);
    FASTOR_DOES_CHECK_PASS(SIMDVector<float,simd_abi::avx>::size()==8);
    FASTOR_DOES_CHECK_PASS(SIMDVector<float,simd_abi::sse>::size()==4);
    FASTOR_DOES_CHECK_PASS(SIMDVector<float,simd_abi::scalar>::size()==1);

    FASTOR_DOES_CHECK_PASS(SIMDVector<int>::size()==8);
    FASTOR_DOES_CHECK_PASS(SIMDVector<int,simd_abi::avx>::size()==8);
    FASTOR_DOES_CHECK_PASS(SIMDVector<int,simd_abi::sse>::size()==4);
    FASTOR_DOES_CHECK_PASS(SIMDVector<int,simd_abi::scalar>::size()==1);

    FASTOR_DOES_CHECK_PASS(SIMDVector<Int64>::size()==4);
    FASTOR_DOES_CHECK_PASS(SIMDVector<Int64,simd_abi::avx>::size()==4);
    FASTOR_DOES_CHECK_PASS(SIMDVector<Int64,simd_abi::sse>::size()==2);
    FASTOR_DOES_CHECK_PASS(SIMDVector<Int64,simd_abi::scalar>::size()==1);

    // FASTOR_DOES_CHECK_PASS(SIMDVector<std::complex<double>>::size()==2);
    // FASTOR_DOES_CHECK_PASS(SIMDVector<std::complex<double>,simd_abi::avx>::size()==2);
    // FASTOR_DOES_CHECK_PASS(SIMDVector<std::complex<double>,simd_abi::sse>::size()==1);
    // FASTOR_DOES_CHECK_PASS(SIMDVector<std::complex<double>,simd_abi::scalar>::size()==1);

    // FASTOR_DOES_CHECK_PASS(SIMDVector<std::complex<float>>::size()==4);
    // FASTOR_DOES_CHECK_PASS(SIMDVector<std::complex<float>,simd_abi::avx>::size()==4);
    // FASTOR_DOES_CHECK_PASS(SIMDVector<std::complex<float>,simd_abi::sse>::size()==2);
    // FASTOR_DOES_CHECK_PASS(SIMDVector<std::complex<float>,simd_abi::scalar>::size()==1);
#endif

    print(FGRN(BOLD("All tests passed successfully")));

}

int main() {

    print(FBLU(BOLD("Testing SIMDVector of single precision - 32")));
    test_simd_vectors<float,simd_abi::scalar>();
    print(FBLU(BOLD("Testing SIMDVector of single precision - 64")));
    test_simd_vectors<float,simd_abi::fixed_size<2>>();
    print(FBLU(BOLD("Testing SIMDVector of single precision - 128")));
    test_simd_vectors<float,simd_abi::sse>();
    print(FBLU(BOLD("Testing SIMDVector of single precision - 256")));
    test_simd_vectors<float,simd_abi::avx>();
    print(FBLU(BOLD("Testing SIMDVector of single precision - 512")));
    test_simd_vectors<float,simd_abi::fixed_size<16>>();

    print(FBLU(BOLD("Testing SIMDVector of double precision - 64")));
    test_simd_vectors<double,simd_abi::scalar>();
    print(FBLU(BOLD("Testing SIMDVector of double precision - 128")));
    test_simd_vectors<double,simd_abi::sse>();
    print(FBLU(BOLD("Testing SIMDVector of double precision - 256")));
    test_simd_vectors<double,simd_abi::avx>();
    print(FBLU(BOLD("Testing SIMDVector of double precision - 512")));
    test_simd_vectors<double,simd_abi::fixed_size<8>>();

    print(FBLU(BOLD("Testing SIMDVector of int - 32")));
    test_simd_vectors<int,simd_abi::scalar>();
    print(FBLU(BOLD("Testing SIMDVector of int - 64")));
    test_simd_vectors<int,simd_abi::fixed_size<2>>();
    print(FBLU(BOLD("Testing SIMDVector of int - 128")));
    test_simd_vectors<int,simd_abi::sse>();
    print(FBLU(BOLD("Testing SIMDVector of int - 256")));
    test_simd_vectors<int,simd_abi::avx>();
    print(FBLU(BOLD("Testing SIMDVector of int - 512")));
#ifdef FASTOR_AVX512F_IMPL
    test_simd_vectors<int,simd_abi::avx512>();
#else
    test_simd_vectors<int,simd_abi::fixed_size<16>>();
#endif

    print(FBLU(BOLD("Testing SIMDVector of long long - 64")));
    test_simd_vectors<Int64,simd_abi::scalar>();
    print(FBLU(BOLD("Testing SIMDVector of long long - 128")));
    test_simd_vectors<Int64,simd_abi::sse>();
    print(FBLU(BOLD("Testing SIMDVector of long long - 256")));
    test_simd_vectors<Int64,simd_abi::avx>();
    print(FBLU(BOLD("Testing SIMDVector of long long - 512")));
#ifdef FASTOR_AVX512F_IMPL
    test_simd_vectors<Int64,simd_abi::avx512>();
#else
    test_simd_vectors<Int64,simd_abi::fixed_size<8>>();
#endif


    print(FBLU(BOLD("Testing SIMDVector of int for division - 128")));
    test_intergers_divs<int,simd_abi::sse>();
    print(FBLU(BOLD("Testing SIMDVector of int for division - 256")));
    test_intergers_divs<int,simd_abi::avx>();

    print(FBLU(BOLD("Testing SIMDVector of long long for division - 128")));
    test_intergers_divs<Int64,simd_abi::sse>();
    print(FBLU(BOLD("Testing SIMDVector of long long for division - 256")));
    test_intergers_divs<Int64,simd_abi::avx>();

    return 0;
}
