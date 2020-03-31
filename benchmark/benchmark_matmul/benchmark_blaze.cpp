#include "helper.h"
using benchmarks_general::println;
using benchmarks_general::rtimeit;
using benchmarks_general::unused;

#define BLAZE_DEFAULT_ALIGNMENT_FLAG blaze::unaligned
#define BLAZE_DEFAULT_PADDING_FLAG blaze::unpadded
#include <blaze/math/StaticVector.h>
#include <blaze/math/StaticMatrix.h>

template<typename T, size_t M, size_t K, size_t N>
T single_test() {
    using namespace blaze;
    StaticMatrix<T,M,K,rowMajor> a(3);
    StaticMatrix<T,K,N,rowMajor> b(4);
    StaticMatrix<T,M,N,rowMajor> c = a*b;

    StaticMatrix<T,M,N,rowMajor> tmp(0);
    benchmarks_general::matmul_ref<M,K,N>(a,b,tmp);
    return std::abs(sum(tmp) - sum(c));
}

template<typename T, size_t M, size_t K, size_t N>
void run_single_test() {
    T value = single_test<T,M,K,N>();
    benchmarks_general::EXIT_ASSERT(value < 1e-10, "TEST FAILED");
}

template<typename T>
void run_tests() {
    TEST_RUN_BENCHMARK(run_single_test, T)
}



template<typename T, size_t M, size_t K, size_t N>
void single_benchmark() {
    using namespace blaze;
    StaticMatrix<T,M,K,rowMajor> a(3);
    StaticMatrix<T,K,N,rowMajor> b(4);
    StaticMatrix<T,M,N,rowMajor> c = a*b;
    unused(c);
}

template<typename T, size_t M, size_t K, size_t N>
void run_single_benchmark() {

    double elapsed_time = rtimeit(static_cast<void (*)()>(&single_benchmark<T,M,K,N>));
    double max_gflops = 2.0 * M * N * K / (elapsed_time * 1.0e9);
    println("(M, N, K):", M, N, K, "GFLOPS:", max_gflops, "minimum runtime:", elapsed_time,'\n');
}


template<typename T>
void run_benchmarks() {
    TEST_RUN_BENCHMARK(run_single_benchmark, T)
}



int main() {

#ifdef RUN_SINGLE
    println("Running blaze benchmark: single precision\n");
    run_tests<float>();
    run_benchmarks<float>();
#else
    println("Running blaze benchmark: double precision\n");
    run_tests<double>();
    run_benchmarks<double>();
#endif

    return 0;
}
