#include <iostream>
#include <chrono>
#include <limits>


#define TEST_RUN_BENCHMARK(FUNC, T) \
    FUNC<T,3,18,3>();\
    FUNC<T,9,15,4>();\
    FUNC<T,5,13,5>();\
    FUNC<T,8,8,8>();\
    FUNC<T,18,18,18>();\
    FUNC<T,21,21,21>();\
    FUNC<T,19,17,7>();\
    FUNC<T,15,20,25>();\
    FUNC<T,33,33,33>();\
    FUNC<T,36,32,36>();\
    FUNC<T,45,45,47>();\
    FUNC<T,48,48,48>();\
    FUNC<T,64,64,64>();\
    FUNC<T,96,96,96>();\
    FUNC<T,128,128,128>();\


namespace benchmarks_general {

template<typename T>
inline void println(const T &a) {
    std::cout << a;
}
template<typename T, typename ... Rest>
inline void println(const T &first, const Rest& ... rest) {
    println(first);
    std::cout << ' ';
    println(rest...);
}


#define BENCH_RUNTIME 1.0

template<typename T, typename ... Params, typename ... Args>
inline double rtimeit(T (*func)(Params...), Args...args)
{
    double counter = 1.0;
    double mean_time = 0.0;
    double best_time = 1.0e20;

    for (auto iter=0; iter<1e09; ++iter)
    {
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();

        // Run the function
        func(std::forward<Params>(args)...);

        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;

        mean_time += elapsed_seconds.count();
        if (elapsed_seconds.count() < best_time) {
            best_time = elapsed_seconds.count();
        };
        counter++;

        if (mean_time > BENCH_RUNTIME)
        {
            mean_time /= counter;
            break;
        }
    }

    return mean_time;
}

//clobber
template <typename T> void unused(T &&x) {
#ifndef _WIN32
    asm("" ::"m"(x));
#endif
}
template <typename T, typename ... U> void unused(T&& x, U&& ...y) { unused(x); unused(y...); }


inline void EXIT_ASSERT(bool cond, const std::string &x="") {
    if (!cond) {
        std::cout << x << '\n';
        exit(EXIT_FAILURE);
    }
}


template<size_t M, size_t K, size_t N, class MT1, class MT2, class MT3>
void matmul_ref(const MT1 &a, const MT2 &b, MT3 &c) {
    for (size_t i=0; i<M; ++i) {
        for (size_t j=0; j<K; ++j) {
            for (size_t k=0; k<N; ++k) {
                c(i,k) += a(i,j)*b(j,k);
            }
        }
    }
}


}

