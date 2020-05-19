#include "../benchmarks_general.h"

#include <Fastor/Fastor.h>
#include <Eigen/Core>
#include <Eigen/Dense>

template<typename T, size_t M, size_t N>
void benchmark_inverse_eigen() {
    using namespace Eigen;
    Matrix<T,M,N,RowMajor> a;
    a.setConstant(3);
    Eigen::Matrix<T,N,M,RowMajor> c = a.inverse();
    benchmarks_general::unused(c);
}

template<typename T, size_t M, size_t N>
void benchmark_inverse_fastor() {
    using namespace Fastor;
    Tensor<T,M,N> a(3);
    Tensor<T,N,M> c = inverse(a);
    benchmarks_general::unused(c);
}



template<typename T, size_t M>
void benchmark_run() {

    using benchmarks_general::println;
    using benchmarks_general::rtimeit;

    println("Testing size (M, N)", M, M,'\n');

    double etime = rtimeit(static_cast<void (*)()>(&benchmark_inverse_eigen<T,M,M>));
    double ftime = rtimeit(static_cast<void (*)()>(&benchmark_inverse_fastor<T,M,M>));

    println("Elapsed time -> Eigen, Fastor\n", etime, ftime,'\n');
}

template<size_t step, size_t from, size_t to>
struct benchmark_generate {
    template<typename T>
    static inline void generate() {
        benchmark_run<T,from>();
        benchmark_generate<step,from+step,to>::template generate<T>();
    }
};
template<size_t step, size_t from>
struct benchmark_generate<step,from,from> {
    template<typename T>
    static inline void generate() {
        benchmark_run<T,from>();
    }
};


int main () {

#ifdef RUN_SINGLE
    benchmark_generate<8,8,128>::template generate<float>();
#else
    benchmark_generate<8,8,128>::template generate<double>();
#endif

    return 0;
}
