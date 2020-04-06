#ifndef TIMEIT_H
#define TIMEIT_H

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <limits>
#include <tuple>
#include <utility>

#ifdef _WIN32
#include <intrin.h>
#endif

#include "Fastor/commons/cpuid.h"


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

namespace Fastor {


#ifndef FASTOR_USE_RDTSC
#define FASTOR_USE_RDTSC
#endif

#ifndef FASTOR_SIMPLE_RDTSC
#define FASTOR_SIMPLE_RDTSC
#endif

#ifndef FASTOR_BENCH_RUNTIME
#define FASTOR_BENCH_RUNTIME 1.0
#endif


// Get cpu cycle count
#ifdef _WIN32
inline uint64_t rdtsc() {
    return __rdtsc();
}
//  Linux/GCC
#else
inline uint64_t rdtsc() {
    unsigned int lo, hi;
    // This does not clobber the register so rdtsc overwrites
    // the register, see
    // How to Benchmark Code Execution Times on Intel® IA-32
    // and IA-64 Instruction Set Architectures pp-9
    // https://intel.ly/3dXFfQN
#ifdef FASTOR_SIMPLE_RDTSC
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
#else
    // Use this instead
    __asm__ __volatile__ ("RDTSC\n\t"
                         "mov %%edx, %0\n\t"
                         "mov %%eax, %1\n\t": "=r" (hi), "=r" (lo) ::
                         // we need to clobber
                         // "%eax", "%edx" // IA-32
                         "%rax", "%rdx"    // IA-64
                     );
#endif
    return ((uint64_t)hi << 32) | lo;
}

#ifndef FASTOR_SIMPLE_RDTSC
// There is still one problem with the function above [rdtsc()]
// and that is it does not take care of cpu's out-of-order
// executation. In order to serialise we make a call to cpuid
// just before rdtsc. While this does not effect timing of the
// function itself, it introduces a lot of overhead when called
// multiple times within a loop
// https://intel.ly/3dXFfQN
inline uint64_t rdtsc_begin() {
    unsigned int lo, hi;
    __asm__ __volatile__ ("CPUID\n\t"
                         "RDTSC\n\t"
                         "mov %%edx, %0\n\t"
                         "mov %%eax, %1\n\t": "=r" (hi), "=r" (lo) ::
                         "%rax", "%rbx", "%rcx", "%rdx" // clobber memory
                         );
    return ((uint64_t)hi << 32) | lo;
}
// and then a call to cpuid immediately after rdtsc
inline uint64_t rdtsc_end() {
    unsigned int lo, hi;
    __asm__ __volatile__("RDTSCP\n\t"
                         "mov %%edx, %0\n\t"
                         "mov %%eax, %1\n\t"
                         "CPUID\n\t": "=r" (hi), "=r" (lo)::
                        "%rax", "%rbx", "%rcx", "%rdx" // clobber memory
                        );
    return ((uint64_t)hi << 32) | lo;
}
#else
inline uint64_t rdtsc_begin() { return rdtsc();}
inline uint64_t rdtsc_end() { return rdtsc();}
#endif
#endif



namespace useless {
inline
double format_time(double _time) {
    if (_time >= 1.0e-3 && _time < 1.) return _time / 1e-3;
    else if (_time >= 1.0e-6 && _time < 1.0e-3) return _time / 1e-6;
    else if (_time < 1.0e-6) return _time / 1e-9;
    else return _time;
}

inline
std::string format_time_string(double _time) {
    if (_time >= 1.0e-3 && _time < 1.) return " ms";
    else if (_time >= 1.0e-6 && _time < 1.0e-3) return " \xC2\xB5s";
    else if (_time < 1.0e-6) return " ns";
    else return " s";
}
}

#define FASTOR_FORMAT_BENCH_TIME_DISPLAY_1()\
    std::cout << counter\
    << FGRN(BOLD(" runs, min time: "))\
    << std::setprecision(6) << useless::format_time(best_time) << useless::format_time_string(best_time) << ". " \
    << FGRN(BOLD("mean time: "))\
    << useless::format_time(mean_time) << useless::format_time_string(mean_time) << ". "\
    << FGRN(BOLD("max time: "))\
    << useless::format_time(worst_time) << useless::format_time_string(worst_time) << ". "\
    << FGRN(BOLD("No of RDTSC CPU cycles "))\
    << uint64_t(cycles/(1.0*counter)) << std::endl;\

#define FASTOR_FORMAT_BENCH_TIME_DISPLAY_2()\
    std::cout << counter\
    << FGRN(BOLD(" runs, min time: "))\
    << std::setprecision(6) << useless::format_time(best_time) << useless::format_time_string(best_time) << ". " \
    << FGRN(BOLD("mean time: "))\
    << useless::format_time(mean_time) << useless::format_time_string(mean_time) << ". "\
    << FGRN(BOLD("max time: "))\
    << useless::format_time(worst_time) << useless::format_time_string(worst_time) << std::endl;\





template<typename T, typename ... Params, typename ... Args>
inline
std::tuple<double,double,double> // min, mean, max times
timeit(T (*func)(Params...), Args...args)
{
    uint64_t counter = 1;
    double mean_time = 0.0;
    double best_time = std::numeric_limits<double>::max();
    double worst_time = 0.0;
#if defined(FASTOR_USE_RDTSC)
    uint64_t cycles = 0;
    CPUID cpuID(0);
    // A cycle is 1 second per max cpu frequency assuming constant_tsc
    // Caution: in theory, there is no guarantee that rdtsc would have
    // strong relation to CPU cycles
    // https://stackoverflow.com/questions/36663379/seconds-calculation-using-rdtsc
    double tsc_to_time = 1.0/cpuID.EBX();

// #ifndef _WIN32
//     rdtsc_begin();
//     rdtsc_end();
// #endif
#endif
            // std::cout << () * (cycles/(1.0*counter)) << "\n";

    for (auto iter=0; iter<1e09; ++iter)
    {
#if defined(FASTOR_USE_SYSTEM_CLOCK)
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
#elif defined(FASTOR_USE_RDTSC)
#ifdef _WIN32
        auto cycle = rdtsc();
#else
        auto cycle = rdtsc_begin();
#endif
#else
        std::chrono::time_point<std::chrono::steady_clock> start, end;
        start = std::chrono::steady_clock::now();
#endif

        // Run the function
        func(std::forward<Params>(args)...);
        // Ignore the few first runs for cache hot measurements
        if (iter < 1) continue;

#if defined(FASTOR_USE_RDTSC)
#ifdef _WIN32
        cycle = rdtsc() - cycle;
#else
        cycle = rdtsc_end() - cycle;
#endif
        cycles += cycle;

        double elapsed_t = tsc_to_time*cycle;
        if (elapsed_t < best_time && elapsed_t != 0.0) {
            best_time = elapsed_t;
        };
        if (elapsed_t > worst_time) {
            worst_time = elapsed_t;
        };
        mean_time += elapsed_t;
#else
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;

        if (elapsed_seconds.count() < best_time && elapsed_seconds.count() != 0.0) {
            best_time = elapsed_seconds.count();
        };
        if (elapsed_seconds.count() > worst_time) {
            worst_time = elapsed_seconds.count();
        };
        mean_time += elapsed_seconds.count();
#endif

        counter++;

        if (mean_time > FASTOR_BENCH_RUNTIME)
        {
            mean_time /= (double)counter;
#if defined(FASTOR_USE_RDTSC)
            FASTOR_FORMAT_BENCH_TIME_DISPLAY_1()
#else
            FASTOR_FORMAT_BENCH_TIME_DISPLAY_2()
#endif
            break;
        }
    }

    return std::make_tuple(mean_time, best_time, worst_time);
}



// timeit with return values
template<typename T, typename ... Params, typename ... Args>
inline std::tuple<double,uint64_t> rtimeit(T (*func)(Params...), Args...args)
{
    uint64_t counter = 1;
    double mean_time = 0.0;
    double best_time = std::numeric_limits<double>::max();
    double worst_time = 0;
    uint64_t cycles = 0;

    for (auto iter=0; iter<1e09; ++iter)
    {
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        auto cycle = rdtsc();

        // Run the function
        func(std::forward<Params>(args)...);
        // Ignore the few first runs for cache hot measurements
        if (iter < 1) continue;

        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        cycle = rdtsc() - cycle;
        cycles += cycle;

        mean_time += elapsed_seconds.count();

        if (elapsed_seconds.count() < best_time && elapsed_seconds.count() != 0) {
            best_time = elapsed_seconds.count();
        };
        if (elapsed_seconds.count() > worst_time) {
            worst_time = elapsed_seconds.count();
        };

        counter++;

        if (mean_time > FASTOR_BENCH_RUNTIME)
        {
            mean_time /= counter;
            break;
        }
    }

    return std::make_tuple(mean_time,uint64_t(cycles/(1.0*counter)));
}



// tic toc
template<typename T=double>
struct timer
{
    inline void tic() {t0 = std::chrono::steady_clock::now();}

    inline T toc(const std::string &msg="") {
        using namespace std::chrono;
        elapsed = steady_clock::now() - t0;
        T elapsed_seconds = duration<T,seconds::period>(elapsed).count();
        if (msg.empty()) std::cout << FGRN(BOLD("Elapsed time is: ")) <<
            elapsed_seconds << FGRN(BOLD(" seconds \n"));
        else std::cout  << std::string("\x1B[32m ")+std::string("\x1B[1m")+msg+std::string("\x1B[0m")+" "
                        << elapsed_seconds << FGRN(BOLD(" seconds \n"));
        return elapsed_seconds;
    }

    std::chrono::steady_clock::time_point t0;
    std::chrono::steady_clock::duration elapsed;
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
