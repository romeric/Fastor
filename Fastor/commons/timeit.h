#ifndef TIMEIT_H
#define TIMEIT_H

#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <utility>
#include <tuple>

#ifdef _WIN32
#include <intrin.h>
#endif


#ifndef FASTOR_NO_COLOUR_PRINT

/* FOREGROUND */
#define RST  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#define FRED(x) KRED x RST
#define FGRN(x) KGRN x RST
#define FYEL(x) KYEL x RST
#define FBLU(x) KBLU x RST
#define FMAG(x) KMAG x RST
#define FCYN(x) KCYN x RST
#define FWHT(x) KWHT x RST

#define BOLD(x) "\x1B[1m" x RST
#define UNDL(x) "\x1B[4m" x RST

#else

#define RST
#define KRED
#define KGRN
#define KYEL
#define KBLU
#define KMAG
#define KCYN
#define KWHT

#define FRED(x) x
#define FGRN(x) x
#define FYEL(x) x
#define FBLU(x) x
#define FMAG(x) x
#define FCYN(x) x
#define FWHT(x) x

#define BOLD(x) x
#define UNDL(x) x


#endif // FASTOR_NO_COLOUR_PRINT

#ifndef NO_CYCLES
#define CYCLES
#endif

namespace Fastor {

// Get cpu cycle count

//  Windows
#ifdef _WIN32
inline uint64_t rdtsc(){
    return __rdtsc();
}
//  Linux/GCC
#else
inline uint64_t rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}
#endif


// #define USE_SYSTEM_CLOCK

//#if defined(INCREASE_BENCH_TIMER_0)
//#define RUNTIME 2.0
//#elif defined(INCREASE_BENCH_TIMER_1)
//#define RUNTIME 3.0
//#elif defined(INCREASE_BENCH_TIMER_2)
//#define RUNTIME 5.0
//#elif defined(INCREASE_BENCH_TIMER_3)
//#define RUNTIME 10.0
//#else
//#define RUNTIME 1.0
//#endif
#define RUNTIME 1.0

template<typename T, typename ... Params, typename ... Args>
inline double timeit(T (*func)(Params...), Args...args)
{
    double counter = 1.0;
    double mean_time = 0.0;
    double best_time = 1.0e20;
    uint64_t cycles = 0;
#ifndef CYCLES
    uint64_t cycle=0;
#endif

    for (auto iter=0; iter<1e09; ++iter)
    {
#ifdef USE_SYSTEM_CLOCK
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
#else
        std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
        start = std::chrono::high_resolution_clock::now();
#endif
#ifdef CYCLES
        auto cycle = rdtsc();
#endif

        // Run the function
        func(std::forward<Params>(args)...);

#ifdef CYCLES
        cycle = rdtsc() - cycle;
#endif
#ifdef USE_SYSTEM_CLOCK
        end = std::chrono::system_clock::now();
#else
        end = std::chrono::high_resolution_clock::now();
#endif
        std::chrono::duration<double> elapsed_seconds = end-start;

        mean_time += elapsed_seconds.count();
#ifdef CYCLES
        cycles += cycle;
#endif

        if (elapsed_seconds.count() < best_time) {
            best_time = elapsed_seconds.count();
        };

        counter++;

        if (mean_time > RUNTIME)
        {
#ifndef FASTOR_USE_BEST_TIME
            mean_time /= counter;
#else
            mean_time = best_time;
#endif
            if (mean_time >= 1.0e-3 && mean_time < 1.)
                std::cout << static_cast<long int>(counter)
#ifndef FASTOR_USE_BEST_TIME
                          << FGRN(BOLD(" runs, average elapsed time is "))
#else
                          << FGRN(BOLD(" runs, best elapsed time is "))
#endif
                          << mean_time/1.0e-03 << " ms" << ". " << FGRN(BOLD("No of CPU cycles "))
                          << uint64_t(cycles/(1.0*counter)) << std::endl;
            else if (mean_time >= 1.0e-6 && mean_time < 1.0e-3)
                std::cout << static_cast<long int>(counter)
#ifndef FASTOR_USE_BEST_TIME
                          << FGRN(BOLD(" runs, average elapsed time is "))
#else
                          << FGRN(BOLD(" runs, best elapsed time is "))
#endif
                          << mean_time/1.0e-06 << " \xC2\xB5s" << ". " << FGRN(BOLD("No of CPU cycles "))
                          << uint64_t(cycles/(1.0*counter)) << std::endl; //\xE6
            else if (mean_time < 1.0e-6)
                std::cout << static_cast<long int>(counter)
#ifndef FASTOR_USE_BEST_TIME
                          << FGRN(BOLD(" runs, average elapsed time is "))
#else
                          << FGRN(BOLD(" runs, best elapsed time is "))
#endif
                          << mean_time/1.0e-09 << " ns" << ". " << FGRN(BOLD("No of CPU cycles "))
                          << uint64_t(cycles/(1.0*counter)) << std::endl;
            else
                std::cout << static_cast<long int>(counter)
#ifndef FASTOR_USE_BEST_TIME
                          << FGRN(BOLD(" runs, average elapsed time is "))
#else
                          << FGRN(BOLD(" runs, best elapsed time is "))
#endif
                          << mean_time << " s" << ". " << FGRN(BOLD("No of CPU cycles "))
                          << uint64_t(cycles/(1.0*counter)) << std::endl;

            break;
        }
    }

    return mean_time;
}



// timeit with return values
template<typename T, typename ... Params, typename ... Args>
inline std::tuple<double,uint64_t> rtimeit(T (*func)(Params...), Args...args)
{
    double counter = 1.0;
    double mean_time = 0.0;
    double best_time = 1.0e20;
    uint64_t cycles = 0;
#ifndef CYCLES
    uint64_t cycle=0;
#endif

    for (auto iter=0; iter<1e09; ++iter)
    {
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
#ifdef CYCLES
        auto cycle = rdtsc();
#endif

        // Run the function
        func(std::forward<Params>(args)...);

#ifdef CYCLES
        cycle = rdtsc() - cycle;
#endif
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;

        mean_time += elapsed_seconds.count();
#ifdef CYCLES
        cycles += cycle;
#endif

        if (elapsed_seconds.count() < best_time) {
            best_time = elapsed_seconds.count();
        };

        counter++;

        if (mean_time > RUNTIME)
        {
#ifndef FASTOR_USE_BEST_TIME
            mean_time /= counter;
#else
            mean_time = best_time;
#endif
            break;
        }
    }

    return std::make_tuple(mean_time,uint64_t(cycles/(1.0*counter)));
}



// tic toc
template<typename T=double>
struct timer
{
    inline void tic() {t0 = std::chrono::high_resolution_clock::now();}

    inline T toc(const std::string &msg="") {
        using namespace std::chrono;
        elapsed = high_resolution_clock::now() - t0;
        T elapsed_seconds = duration<T,seconds::period>(elapsed).count();
        if (msg.empty()) std::cout << FGRN(BOLD("Elapsed time is: ")) <<
            elapsed_seconds << FGRN(BOLD(" seconds \n"));
        else std::cout  << std::string("\x1B[32m ")+std::string("\x1B[1m")+msg+std::string("\x1B[0m")+" "
                        << elapsed_seconds << FGRN(BOLD(" seconds \n"));
        return elapsed_seconds;
    }

    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::duration elapsed;
};


// Define a no operation function
inline void no_op(){}

//clobber
template <typename T> void unused(T &&x) {
#ifndef _WIN32
    asm("" ::"m"(x));
#endif
}
template <typename T, typename ... U> void unused(T&& x, U&& ...y) { unused(x); unused(y...); }


} // end of namespace


#endif