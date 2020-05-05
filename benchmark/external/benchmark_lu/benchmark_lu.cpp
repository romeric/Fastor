#include "../benchmarks_general.h"

#include <Fastor/Fastor.h>
#include <Eigen/Core>
#include <Eigen/Dense>

template<typename T, size_t M, size_t N>
void benchmark_lu_eigen() {
    using namespace Eigen;
    Matrix<T,M,N,RowMajor> a;
    std::iota(a.data(),a.data()+M*N,0);
    for (size_t i=0; i<M; ++i) a(i,i) = 100 + i;
    auto lu = a.lu();
    Eigen::Matrix<T,N,M,RowMajor> lum = lu.matrixLU();
    Eigen::Matrix<T,N,M,RowMajor> pm  = lu.permutationP();
    benchmarks_general::unused(lum);
    benchmarks_general::unused(pm);
}

template<typename T, size_t M, size_t N>
void benchmark_lu_fastor() {
    using namespace Fastor;
    Tensor<T,M,N> a;
    std::iota(a.data(),a.data()+M*N,0);
    for (size_t i=0; i<M; ++i) a(i,i) = 100 + i;
    Tensor<T,M,N> l, u;
    Tensor<size_t,M> p;
    lu<LUCompType::BlockLUPiv>(a, l, u, p);
    benchmarks_general::unused(l);
    benchmarks_general::unused(u);
    benchmarks_general::unused(p);
}



template<typename T, size_t M>
void benchmark_run() {

    using benchmarks_general::println;
    using benchmarks_general::rtimeit;

    println("Testing size (M, N)", M, M,'\n');

    double etime = rtimeit(static_cast<void (*)()>(&benchmark_lu_eigen<T,M,M>));
    double ftime = rtimeit(static_cast<void (*)()>(&benchmark_lu_fastor<T,M,M>));

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
    benchmark_generate<2,2,32>::template generate<float>();
#else
    benchmark_generate<2,2,32>::template generate<double>();
#endif

    return 0;
}