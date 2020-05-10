#include <Fastor/Fastor.h>
#include <complex>

using namespace Fastor;

#define Tol 1e-12
#define BigTol 1e-5


template<typename T, size_t N>
std::array<std::complex<T>,N>
zadd(const std::array<std::complex<T>,N> &a, const std::array<std::complex<T>,N> &b) {
    std::array<std::complex<T>,N> out;
    for (size_t i=0; i< N; ++i) out[i] = a[i]+b[i];
    return out;
}
template<typename T, size_t N>
std::array<std::complex<T>,N>
zsub(const std::array<std::complex<T>,N> &a, const std::array<std::complex<T>,N> &b) {
    std::array<std::complex<T>,N> out;
    for (size_t i=0; i< N; ++i) out[i] = a[i]-b[i];
    return out;
}
template<typename T, size_t N>
std::array<std::complex<T>,N>
zmul(const std::array<std::complex<T>,N> &a, const std::array<std::complex<T>,N> &b) {
    std::array<std::complex<T>,N> out;
    for (size_t i=0; i< N; ++i) out[i] = a[i]*b[i];
    return out;
}
template<typename T, size_t N>
std::array<std::complex<T>,N>
zdiv(const std::array<std::complex<T>,N> &a, const std::array<std::complex<T>,N> &b) {
    std::array<std::complex<T>,N> out;
    for (size_t i=0; i< N; ++i) out[i] = a[i]/b[i];
    return out;
}
// horizontal sum
template<typename T, size_t N>
std::complex<T>
zsum(const std::array<std::complex<T>,N> &a) {
    std::complex<T> out(a[0]);
    for (size_t i=1; i< N; ++i) out += a[i];
    return out;
}
template<typename T, size_t N>
std::complex<T>
zproduct(const std::array<std::complex<T>,N> &a) {
    std::complex<T> out(a[0]);
    for (size_t i=1; i< N; ++i) out *= a[i];
    return out;
}
template<typename T, size_t N>
std::array<T,N>
znorm(const std::array<std::complex<T>,N> &a) {
    std::array<T,N> out;
    for (size_t i=0; i< N; ++i) out[i] = std::norm(a[i]);
    return out;
}
template<typename T, size_t N>
std::array<T,N>
zabs(const std::array<std::complex<T>,N> &a) {
    std::array<T,N> out;
    for (size_t i=0; i< N; ++i) out[i] = std::abs(a[i]);
    return out;
}
template<typename T, size_t N>
std::array<std::complex<T>,N>
zconj(const std::array<std::complex<T>,N> &a) {
    std::array<std::complex<T>,N> out;
    for (size_t i=0; i< N; ++i) out[i] = std::conj(a[i]);
    return out;
}
template<typename T, size_t N>
std::array<T,N>
zarg(const std::array<std::complex<T>,N> &a) {
    std::array<T,N> out;
    for (size_t i=0; i< N; ++i) out[i] = std::arg(a[i]);
    return out;
}


template<typename T, typename ABI>
void test_complex_double_sse();

template<>
void test_complex_double_sse<double,simd_abi::sse>() {

    using TT = std::complex<double>;
    using ABI = simd_abi::sse;
    // These arrays are used to mimick complex SIMDVector
    std::array<TT,2> arr1 = {TT(3,-4),TT(7,12)};
    std::array<TT,2> arr2 = {TT(6,14),TT(-5,1.2)};

    std::array<TT,2> arr3 = {arr1[0],arr1[0]};
    // std::array<TT,2> arr4 = {arr2[0],TT(0,0)};
    // std::array<TT,2> out;
    TT diff;
    {
        // Size
        {
            using V = SIMDVector<TT,ABI>;
            FASTOR_EXIT_ASSERT( std::abs(int(V::Size)   - 2) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(int(V::size()) - 2) < Tol, "TEST FAILED");
        }
        // Load/store
        {
            SIMDVector<TT,ABI> a(arr1.data(),false);
            SIMDVector<TT,ABI> b(arr2.data(),false);

            diff = a.sum() - zsum(arr1);
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            diff = b.sum() - zsum(arr2);
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            a.load(arr1.data(),false);
            diff = a.sum() - zsum(arr1);
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            b.load(arr2.data(),false);
            diff = b.sum() - zsum(arr2);
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            a.store(arr1.data(),false);
            diff = a.sum() - zsum(arr1);
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            b.store(arr2.data(),false);
            diff = b.sum() - zsum(arr2);
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            diff = a.product() - zproduct(arr1);
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            diff = b.product() - zproduct(arr2);
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");
        }


        SIMDVector<TT,ABI> a(arr1.data(),false);
        SIMDVector<TT,ABI> b(arr2.data(),false);

        // vector-vector
        // addition
        diff = (a+b).sum() - zsum(zadd(arr1,arr2));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

        // subtraction
        diff = (a-b).sum() - zsum(zsub(arr1,arr2));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

        // multiplication
        diff = (a*b).sum() - zsum(zmul(arr1,arr2));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

        // division
        diff = (a/b).sum() - zsum(zdiv(arr1,arr2));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

        // vector-scalar
        // addition
        diff = (a+arr1[0]).sum() - zsum(zadd(arr1,arr3));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

        // subtraction
        diff = (a-arr1[0]).sum() - zsum(zsub(arr1,arr3));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

        // multiplication
        diff = (a*arr1[0]).sum() - zsum(zmul(arr1,arr3));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

        // division
        diff = (a/arr1[0]).sum() - zsum(zdiv(arr1,arr3));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");


        // scalar-vector
        // addition
        diff = (arr1[0]+a).sum() - zsum(zadd(arr3,arr1));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

        // subtraction
        diff = (arr1[0]-a).sum() - zsum(zsub(arr3,arr1));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

        // multiplication
        diff = (arr1[0]*a).sum() - zsum(zmul(arr3,arr1));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

        // division
        diff = (arr1[0]/a).sum() - zsum(zdiv(arr3,arr1));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");


        // In-place
        // with another vector
        a += b;
        diff = a.sum() - zsum(zadd(arr1,arr2));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

        a.load(arr1.data(),false);
        a -= b;
        diff = a.sum() - zsum(zsub(arr1,arr2));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

        a.load(arr1.data(),false);
        a *= b;
        diff = a.sum() - zsum(zmul(arr1,arr2));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

        a.load(arr1.data(),false);
        a /= b;
        diff = a.sum() - zsum(zdiv(arr1,arr2));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

        // with scalar
        a.load(arr1.data(),false);
        a += arr1[0];
        diff = a.sum() - zsum(zadd(arr1,arr3));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

        a.load(arr1.data(),false);
        a -= arr1[0];
        diff = a.sum() - zsum(zsub(arr1,arr3));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

        a.load(arr1.data(),false);
        a *= arr1[0];
        diff = a.sum() - zsum(zmul(arr1,arr3));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

        a.load(arr1.data(),false);
        a /= arr1[0];
        diff = a.sum() - zsum(zdiv(arr1,arr3));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");


        // norm
        a.load(arr1.data(),false);
        FASTOR_EXIT_ASSERT( std::abs( a.norm().sum() - znorm(arr1)[0] - znorm(arr1)[1] ) < Tol, "TEST FAILED");

        // abs
        a.load(arr1.data(),false);
        FASTOR_EXIT_ASSERT( std::abs( abs(a).real().sum() - zabs(arr1)[0] - zabs(arr1)[1] ) < Tol, "TEST FAILED");

        // conj
        a.load(arr1.data(),false);
        diff = conj(a).sum() - zsum(zconj(arr1));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

        // conj
        a.load(arr1.data(),false);
        FASTOR_EXIT_ASSERT( std::abs( arg(a).real().sum() - zarg(arr1)[0] - zarg(arr1)[1] ) < BigTol, "TEST FAILED");

    }

    print(FGRN(BOLD("All tests passed successfully")));
}

int main() {

    print(FBLU(BOLD("Testing SIMDVector of complex double precision")));
    test_complex_double_sse<double,simd_abi::sse>();

    return 0;
}