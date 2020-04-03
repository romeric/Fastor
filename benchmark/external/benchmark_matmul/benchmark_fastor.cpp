#include "benchmarks_general.h"
using benchmarks_general::println;
using benchmarks_general::rtimeit;

#include <Fastor/Fastor.h>

template<typename T, size_t M, size_t K, size_t N>
T single_test() {
    using namespace Fastor;
    Tensor<T,M,K> a(3);
    Tensor<T,K,N> b(4);
    Tensor<T,M,N> c = matmul(a,b);

    Tensor<T,M,N> tmp(0);
    benchmarks_general::matmul_ref<M,K,N>(a,b,tmp);
    return std::abs(tmp.sum() - c.sum());
}

template<typename T, size_t M, size_t K, size_t N>
void run_single_test() {
    T value = single_test<T,M,K,N>();
    benchmarks_general::EXIT_ASSERT(value < 1e-10, "TEST FAILED");
}

template<typename T>
void run_tests() {
    TEST_RUN_MATMUL_BENCHMARK(run_single_test, T)
}


template<typename T, size_t M, size_t K, size_t N>
void single_benchmark() {
    using namespace Fastor;
    Tensor<T,M,K> a(3);
    Tensor<T,K,N> b(4);
    Tensor<T,M,N> c = matmul(a,b);
    benchmarks_general::unused(c);
}

template<typename T, size_t M, size_t K, size_t N>
void run_single_benchmark() {

    double elapsed_time = rtimeit(static_cast<void (*)()>(&single_benchmark<T,M,K,N>));
    double max_gflops = 2.0 * M * N * K / (elapsed_time * 1.0e9);
    println("(M, N, K):", M, N, K, "GFLOPS:", max_gflops, "minimum runtime:", elapsed_time,'\n');
}


template<typename T>
void run_benchmarks() {
    TEST_RUN_MATMUL_BENCHMARK(run_single_benchmark, T)
}



int main() {

#ifdef RUN_SINGLE
    println("Running fastor benchmark: single precision\n");
    run_tests<float>();
    run_benchmarks<float>();
#else
    println("Running fastor benchmark: double precision\n");
    run_tests<double>();
    run_benchmarks<double>();
#endif

    return 0;
}
