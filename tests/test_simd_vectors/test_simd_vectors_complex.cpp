#include <Fastor/Fastor.h>
#include <complex>

using namespace Fastor;


template<typename T, size_t N>
std::array<std::complex<T>,N>
zadd(const std::array<std::complex<T>,N> &a, const std::array<std::complex<T>,N> &b) {
    std::array<std::complex<T>,N> out;
    for (size_t i=0; i< N; ++i) out[i] = a[i]+b[i];
    return out;
}
template<typename T, size_t N>
std::array<std::complex<T>,N>
zadd(const std::array<std::complex<T>,N> &a, const T &b) {
    std::array<std::complex<T>,N> out;
    for (size_t i=0; i< N; ++i) out[i] = a[i] + std::complex<T>(b,0);
    return out;
}
template<typename T, size_t N>
std::array<std::complex<T>,N>
zadd(const T &b, const std::array<std::complex<T>,N> &a) {
    std::array<std::complex<T>,N> out;
    for (size_t i=0; i< N; ++i) out[i] = std::complex<T>(b,0) + a[i];
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
zsub(const std::array<std::complex<T>,N> &a, const T &b) {
    std::array<std::complex<T>,N> out;
    for (size_t i=0; i< N; ++i) out[i] = a[i] - std::complex<T>(b,0);
    return out;
}
template<typename T, size_t N>
std::array<std::complex<T>,N>
zsub(const T &b, const std::array<std::complex<T>,N> &a) {
    std::array<std::complex<T>,N> out;
    for (size_t i=0; i< N; ++i) out[i] = std::complex<T>(b,0) - a[i];
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
zmul(const std::array<std::complex<T>,N> &a, const T &b) {
    std::array<std::complex<T>,N> out;
    for (size_t i=0; i< N; ++i) {
        out[i] = a[i] * std::complex<T>(b,0);
    }
    return out;
}
template<typename T, size_t N>
std::array<std::complex<T>,N>
zmul(const T &b, const std::array<std::complex<T>,N> &a) {
    std::array<std::complex<T>,N> out;
    for (size_t i=0; i< N; ++i) {
        out[i] = std::complex<T>(b,0) * a[i];
    }
    return out;
}
template<typename T, size_t N>
std::array<std::complex<T>,N>
zdiv(const std::array<std::complex<T>,N> &a, const std::array<std::complex<T>,N> &b) {
    std::array<std::complex<T>,N> out;
    for (size_t i=0; i< N; ++i) out[i] = a[i]/b[i];
    return out;
}
template<typename T, size_t N>
std::array<std::complex<T>,N>
zdiv(const std::array<std::complex<T>,N> &a, const T &b) {
    std::array<std::complex<T>,N> out;
    for (size_t i=0; i< N; ++i) {
        out[i] = a[i] / std::complex<T>(b,0);
    }
    return out;
}
template<typename T, size_t N>
std::array<std::complex<T>,N>
zdiv(const T &b, const std::array<std::complex<T>,N> &a) {
    std::array<std::complex<T>,N> out;
    for (size_t i=0; i< N; ++i) {
        out[i] = std::complex<T>(b,0) / a[i];
    }
    return out;
}

template<typename T, size_t N>
T
zsum(const std::array<T,N> &a) {
    T out(a[0]);
    for (size_t i=1; i< N; ++i) out += a[i];
    return out;
}
template<typename T, size_t N>
T
zproduct(const std::array<T,N> &a) {
    T out(a[0]);
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




template<typename ABI, typename TT, size_t N>
void test_simd_complex_impl(std::array<TT,N> & arr1, std::array<TT,N> & arr2, std::array<TT,N> & arr3) {


    TT diff;
    {
        using V = SIMDVector<TT,ABI>;
        using T = typename TT::value_type;

        T Tol    = is_same_v_<T,double> ? 1e-9 : (T)1e-04;
        T BigTol = is_same_v_<T,double> ? 1e-4 : (T)1e-03;

        // Size
        {
            FASTOR_EXIT_ASSERT( std::abs(int(V::Size)   - (int)N) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(int(V::size()) - (int)N) < Tol, "TEST FAILED");
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
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < BigTol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < BigTol, "TEST FAILED");

            diff = b.product() - zproduct(arr2);
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < BigTol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < BigTol, "TEST FAILED");
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
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < BigTol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < BigTol, "TEST FAILED");

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
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < BigTol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < BigTol, "TEST FAILED");


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
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < BigTol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < BigTol, "TEST FAILED");


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
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < BigTol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < BigTol, "TEST FAILED");

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
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < BigTol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < BigTol, "TEST FAILED");

#if defined(FASTOR_SSE2_IMPL) && defined(FASTOR_AVX_IMPL)
        // scaling with real numbers
        {
            // in-place
            a.load(arr1.data(),false);
            a += 5;
            diff = (a).sum() - zsum(zadd(arr1,T(5)));
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            a.load(arr1.data(),false);
            a -= 5;
            diff = (a).sum() - zsum(zsub(arr1,T(5)));
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            a.load(arr1.data(),false);
            a *= 5;
            diff = (a).sum() - zsum(zmul(arr1,T(5)));
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            a.load(arr1.data(),false);
            a /= 5;
            diff = (a).sum() - zsum(zdiv(arr1,T(5)));
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < BigTol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < BigTol, "TEST FAILED");

            // binary
            a.load(arr1.data(),false);
            diff = (a+5).sum() - zsum(zadd(arr1,T(5)));
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            diff = (5+a).sum() - zsum(zadd(T(5),arr1));
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            diff = (a-5).sum() - zsum(zsub(arr1,T(5)));
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            diff = (5-a).sum() - zsum(zsub(T(5),arr1));
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            diff = (a*5).sum() - zsum(zmul(arr1,T(5)));
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            diff = (5*a).sum() - zsum(zmul(T(5),arr1));
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            diff = (a/5).sum() - zsum(zdiv(arr1,T(5)));
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < BigTol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < BigTol, "TEST FAILED");

            diff = (5/a).sum() - zsum(zdiv(T(5),arr1));
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < BigTol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < BigTol, "TEST FAILED");
        }
#endif

        // FMAs
        {
            a.load(arr1.data(),false);
            diff = (a*b+a).sum() - zsum(zadd(zmul(arr1,arr2),arr1));
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            diff = fmadd(a,b,a).sum() - zsum(zadd(zmul(arr1,arr2),arr1));
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            diff = (a*b-a).sum() - zsum(zsub(zmul(arr1,arr2),arr1));
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            diff = fmsub(a,b,a).sum() - zsum(zsub(zmul(arr1,arr2),arr1));
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            diff = (a-a*b).sum() - zsum(zsub(arr1,zmul(arr1,arr2)));
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");

            diff = fnmadd(a,b,a).sum() - zsum(zsub(arr1,zmul(arr1,arr2)));
            FASTOR_EXIT_ASSERT( std::abs(diff.real()) < Tol, "TEST FAILED");
            FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < Tol, "TEST FAILED");
        }


        // norm
        a.load(arr1.data(),false);
        FASTOR_EXIT_ASSERT( std::abs( a.norm().sum() - zsum(znorm(arr1)) ) < BigTol, "TEST FAILED");

        // abs
        a.load(arr1.data(),false);
        FASTOR_EXIT_ASSERT( std::abs( abs(a).real().sum() - zsum(zabs(arr1)) ) < BigTol, "TEST FAILED");

        // conj
        a.load(arr1.data(),false);
        diff = conj(a).sum() - zsum(zconj(arr1));
        FASTOR_EXIT_ASSERT( std::abs(diff.real()) < BigTol, "TEST FAILED");
        FASTOR_EXIT_ASSERT( std::abs(diff.imag()) < BigTol, "TEST FAILED");

        // arg
        a.load(arr1.data(),false);
        FASTOR_EXIT_ASSERT( std::abs( arg(a).real().sum() - zsum(zarg(arr1)) ) < BigTol, "TEST FAILED");

    }

    print(FGRN(BOLD("All tests passed successfully")));
}


template<typename T, typename ABI>
void test_simd_complex();
template<>
void test_simd_complex<float,simd_abi::scalar>() {
    using TT = std::complex<float>;
    // These arrays are used to mimick complex SIMDVector
    std::array<TT,1> arr1 = {TT(3,-4)};
    std::array<TT,1> arr2 = {TT(6,14)};
    std::array<TT,1> arr3 = {arr1[0]};
    test_simd_complex_impl<simd_abi::scalar>(arr1,arr2,arr3);
}
#ifdef FASTOR_SSE2_IMPL
template<>
void test_simd_complex<float,simd_abi::sse>() {
    using TT = std::complex<float>;
    // These arrays are used to mimick complex SIMDVector
    std::array<TT,4> arr1 = {TT(3,-4),TT(7,12),TT(5,-2),TT(11,9)};
    std::array<TT,4> arr2 = {TT(6,14),TT(-5,1.2f),TT(0,8),TT(1,15)};
    std::array<TT,4> arr3 = {arr1[0],arr1[0],arr1[0],arr1[0]};
    test_simd_complex_impl<simd_abi::sse>(arr1,arr2,arr3);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
void test_simd_complex<float,simd_abi::avx>() {
    using TT = std::complex<float>;
    // These arrays are used to mimick complex SIMDVector
    std::array<TT,8> arr1 = {TT(3,-4),TT(7,12),TT(5,-2),TT(11,9),TT(13,-4),TT(7,-12),TT(15,-2),TT(11,19)};
    std::array<TT,8> arr2 = {TT(6,14),TT(-5,1.2f),TT(0,8),TT(1,15),TT(-6,14),TT(5,2.2),TT(0,-8),TT(-1,15)};
    std::array<TT,8> arr3 = {arr1[0],arr1[0],arr1[0],arr1[0],arr1[0],arr1[0],arr1[0],arr1[0]};
    test_simd_complex_impl<simd_abi::avx>(arr1,arr2,arr3);
}
#endif
#ifdef FASTOR_AVX512F_IMPL
template<>
void test_simd_complex<float,simd_abi::avx512>() {
    using TT = std::complex<float>;
    // These arrays are used to mimick complex SIMDVector
    std::array<TT,16> arr1 = {TT(3,-4),TT(7,12),TT(5,-2),TT(11,9),TT(13,-4),TT(7,-12),TT(15,-2),TT(11,19),
        TT(3,-4),TT(7,12),TT(5,-2),TT(11,9),TT(13,-4),TT(7,-12),TT(15,-2),TT(11,21)};
    std::array<TT,16> arr2 = {TT(6,14),TT(-5,1.2f),TT(0,8),TT(1,15),TT(-6,14),TT(5,2.2),TT(0,-8),TT(-1,15),
        TT(6,14),TT(-5,1.2),TT(0,8),TT(1,15),TT(-6,14),TT(5,2.2),TT(0,-8),TT(-1,-15)};
    std::array<TT,16> arr3 = {arr1[0],arr1[0],arr1[0],arr1[0],arr1[0],arr1[0],arr1[0],arr1[0],
        arr1[0],arr1[0],arr1[0],arr1[0],arr1[0],arr1[0],arr1[0],arr1[0]};
    test_simd_complex_impl<simd_abi::avx512>(arr1,arr2,arr3);
}
#endif

template<>
void test_simd_complex<double,simd_abi::scalar>() {
    using TT = std::complex<double>;
    // These arrays are used to mimick complex SIMDVector
    std::array<TT,1> arr1 = {TT(3,-4)};
    std::array<TT,1> arr2 = {TT(6,14)};
    std::array<TT,1> arr3 = {arr1[0]};
    test_simd_complex_impl<simd_abi::scalar>(arr1,arr2,arr3);
}
#ifdef FASTOR_SSE2_IMPL
template<>
void test_simd_complex<double,simd_abi::sse>() {
    using TT = std::complex<double>;
    // These arrays are used to mimick complex SIMDVector
    std::array<TT,2> arr1 = {TT(3,-4),TT(7,12)};
    std::array<TT,2> arr2 = {TT(6,14),TT(-5,1.2)};
    std::array<TT,2> arr3 = {arr1[0],arr1[0]};
    test_simd_complex_impl<simd_abi::sse>(arr1,arr2,arr3);
}
#endif
#ifdef FASTOR_AVX_IMPL
template<>
void test_simd_complex<double,simd_abi::avx>() {
    using TT = std::complex<double>;
    // These arrays are used to mimick complex SIMDVector
    std::array<TT,4> arr1 = {TT(3,-4),TT(7,12),TT(5,-2),TT(11,9)};
    std::array<TT,4> arr2 = {TT(6,14),TT(-5,1.2),TT(0,8),TT(1,15)};
    std::array<TT,4> arr3 = {arr1[0],arr1[0],arr1[0],arr1[0]};
    test_simd_complex_impl<simd_abi::avx>(arr1,arr2,arr3);
}
#endif
#ifdef FASTOR_AVX512F_IMPL
template<>
void test_simd_complex<double,simd_abi::avx512>() {
    using TT = std::complex<double>;
    // These arrays are used to mimick complex SIMDVector
    std::array<TT,8> arr1 = {TT(3,-4),TT(7,12),TT(5,-2),TT(11,9),TT(13,-4),TT(7,-12),TT(15,-2),TT(11,19)};
    std::array<TT,8> arr2 = {TT(6,14),TT(-5,1.2),TT(0,8),TT(1,15),TT(-6,14),TT(5,2.2),TT(0,-8),TT(-1,15)};
    std::array<TT,8> arr3 = {arr1[0],arr1[0],arr1[0],arr1[0],arr1[0],arr1[0],arr1[0],arr1[0]};
    test_simd_complex_impl<simd_abi::avx512>(arr1,arr2,arr3);
}
#endif


#ifdef FASTOR_AVX2_IMPL
void test_mask_loading_single() {

    using TT = std::complex<float>;
    float Tol = (float)1e-06;

    // SSE complex float
    {
        using V = SIMDVector<TT,simd_abi::sse>;

        std::array<TT,4> carr1 = {TT(3,-4),TT(7,12),TT(5,-2),TT(11,9)};
        std::array<TT,4> carr2 = {TT(0,0),TT(0,0),TT(0,0),TT(0,0)};
        V a;

        {
            int maska[4] = {-1,-1,-1,-1};

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() - 26 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() - 15 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() - 26 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() - 15 ) < Tol );
        }
        {
            int maska[4] = {0,-1,-1,-1};
            std::fill(carr2.begin(),carr2.end(),0);

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() - 15 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() -  6 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() - 15 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() -  6 ) < Tol );
        }
        {
            int maska[4] = {0,0,-1,-1};
            std::fill(carr2.begin(),carr2.end(),0);

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() - 10 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() -  8 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() - 10 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() -  8 ) < Tol );
        }
        {
            int maska[4] = {0,0,0,-1};
            std::fill(carr2.begin(),carr2.end(),0);

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() -  3 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() +  4 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() -  3 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() +  4 ) < Tol );
        }
        {
            int maska[4] = {0,0,0,0};
            std::fill(carr2.begin(),carr2.end(),0);

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() -  0 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() -  0 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() -  0 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() -  0 ) < Tol );
        }
    }

    // AVX complex float
    {
        using V = SIMDVector<TT,simd_abi::avx>;

        std::array<TT,8> carr1 = {TT(3,-4),TT(7,12),TT(5,-2),TT(11,9),TT(13,-4),TT(7,-12),TT(15,-2),TT(11,19)};
        std::array<TT,8> carr2 = {TT(0,0),TT(0,0),TT(0,0),TT(0,0),TT(0,0),TT(0,0),TT(0,0),TT(0,0)};
        V a;

        {
            int maska[8] = {-1,-1,-1,-1,-1,-1,-1,-1};

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() - 72 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() - 16 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() - 72 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() - 16 ) < Tol );
        }
        {
            int maska[8] = {0,-1,-1,-1,-1,-1,-1,-1};
            std::fill(carr2.begin(),carr2.end(),0);

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() - 61 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() +  3 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() - 61 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() +  3 ) < Tol );
        }
        {
            int maska[8] = {0,0,-1,-1,-1,-1,-1,-1};
            std::fill(carr2.begin(),carr2.end(),0);

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() - 46 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() +  1 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() - 46 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() +  1 ) < Tol );
        }
        {
            int maska[8] = {0,0,0,-1,-1,-1,-1,-1};
            std::fill(carr2.begin(),carr2.end(),0);

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() - 39 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() - 11 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() - 39 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() - 11 ) < Tol );
        }
        {
            int maska[8] = {0,0,0,0,-1,-1,-1,-1};
            std::fill(carr2.begin(),carr2.end(),0);

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() - 26 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() - 15 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() - 26 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() - 15 ) < Tol );
        }
        {
            int maska[8] = {0,0,0,0,0,-1,-1,-1};
            std::fill(carr2.begin(),carr2.end(),0);

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() - 15 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() -  6 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() - 15 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() -  6 ) < Tol );
        }
        {
            int maska[8] = {0,0,0,0,0,0,-1,-1};
            std::fill(carr2.begin(),carr2.end(),0);

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() - 10 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() -  8 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() - 10 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() -  8 ) < Tol );
        }
        {
            int maska[8] = {0,0,0,0,0,0,0,-1};
            std::fill(carr2.begin(),carr2.end(),0);

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() -  3 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() +  4 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() -  3 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() +  4 ) < Tol );
        }
        {
            int maska[8] = {0,0,0,0,0,0,0,0};
            std::fill(carr2.begin(),carr2.end(),0);

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() -  0 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() -  0 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() -  0 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() -  0 ) < Tol );
        }
    }

    print(FGRN(BOLD("All tests passed successfully")));
}


void test_mask_loading_double() {

    using TT = std::complex<double>;
    double Tol = 1e-09;

    // SSE complex double
    {
        using V = SIMDVector<TT,simd_abi::sse>;

        std::array<TT,2> carr1 = {TT(3,-4),TT(7,12)};
        std::array<TT,2> carr2 = {TT(0,0),TT(0,0)};
        V a;

        {
            int maska[2] = {-1,-1};

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() - 10 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() -  8 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() - 10 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() -  8 ) < Tol );
        }
        {
            int maska[2] = {0,-1};
            std::fill(carr2.begin(),carr2.end(),0);

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() -  3 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() +  4 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() -  3 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() +  4 ) < Tol );
        }
        {
            int maska[2] = {0,0};
            std::fill(carr2.begin(),carr2.end(),0);

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() -  0 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() -  0 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() -  0 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() -  0 ) < Tol );
        }
    }

    // AVX complex double
    {
        using V = SIMDVector<TT,simd_abi::avx>;

        std::array<TT,4> carr1 = {TT(3,-4),TT(7,12),TT(5,-2),TT(11,9)};
        std::array<TT,4> carr2 = {TT(0,0),TT(0,0),TT(0,0),TT(0,0)};
        V a;

        {
            int maska[4] = {-1,-1,-1,-1};

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() - 26 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() - 15 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() - 26 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() - 15 ) < Tol );
        }
        {
            int maska[4] = {0,-1,-1,-1};
            std::fill(carr2.begin(),carr2.end(),0);

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() - 15 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() -  6 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() - 15 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() -  6 ) < Tol );
        }
        {
            int maska[4] = {0,0,-1,-1};
            std::fill(carr2.begin(),carr2.end(),0);

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() - 10 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() -  8 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() - 10 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() -  8 ) < Tol );
        }
        {
            int maska[4] = {0,0,0,-1};
            std::fill(carr2.begin(),carr2.end(),0);

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() -  3 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() +  4 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() -  3 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() +  4 ) < Tol );
        }
        {
            int maska[4] = {0,0,0,0};
            std::fill(carr2.begin(),carr2.end(),0);

            a = maskload<V>(carr1.data(), maska);
            FASTOR_EXIT_ASSERT( std::abs( a.real().sum() -  0 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( a.imag().sum() -  0 ) < Tol );

            maskstore(carr2.data(),maska,a);
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).real() -  0 ) < Tol );
            FASTOR_EXIT_ASSERT( std::abs( zsum(carr2).imag() -  0 ) < Tol );
        }
    }

    print(FGRN(BOLD("All tests passed successfully")));
}
#endif



int main() {

    print(FBLU(BOLD("Testing SIMDVector of complex single precision")));
    test_simd_complex<float,simd_abi::scalar>();
#ifdef FASTOR_SSE2_IMPL
    test_simd_complex<float,simd_abi::sse>();
#endif
#ifdef FASTOR_AVX_IMPL
    test_simd_complex<float,simd_abi::avx>();
#endif
#ifdef FASTOR_AVX512F_IMPL
    test_simd_complex<float,simd_abi::avx512>();
#endif
    print(FBLU(BOLD("Testing SIMDVector of complex double precision")));
    test_simd_complex<double,simd_abi::scalar>();
#ifdef FASTOR_SSE2_IMPL
    test_simd_complex<double,simd_abi::sse>();
#endif
#ifdef FASTOR_AVX_IMPL
    test_simd_complex<double,simd_abi::avx>();
#endif
#ifdef FASTOR_AVX512F_IMPL
    test_simd_complex<double,simd_abi::avx512>();
#endif

#ifdef FASTOR_AVX2_IMPL
    print(FBLU(BOLD("Testing SIMDVector AVX2 mask operations - single precision")));
    test_mask_loading_single();
    print(FBLU(BOLD("Testing SIMDVector AVX2 mask operations - double precision")));
    test_mask_loading_double();
#endif

    return 0;
}
