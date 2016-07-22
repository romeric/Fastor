#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdint>

#ifndef _COLORS_
#define _COLORS_

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

#endif  
/* _COLORS_ */

// Get cpu cycle count

//  Windows
#ifdef _WIN32

#include <intrin.h>
uint64_t rdtsc(){
    return __rdtsc();
}
//  Linux/GCC
#else
uint64_t rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}
#endif

#ifndef CYCLES
#define CYCLES
#endif


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
            mean_time /= counter;
            if (mean_time >= 1.0e-3 && mean_time < 1.)
                std::cout << static_cast<long int>(counter)
                          << FGRN(BOLD(" runs, average elapsed time is "))
                          << mean_time/1.0e-03 << " ms" << ". " << FGRN(BOLD("No of CPU cycles ")) 
                          << uint64_t(cycles/(1.0*counter)) << std::endl;
            else if (mean_time >= 1.0e-6 && mean_time < 1.0e-3)
                std::cout << static_cast<long int>(counter)
                          << FGRN(BOLD(" runs, average elapsed time is "))
                          << mean_time/1.0e-06 << " \xC2\xB5s" << ". " << FGRN(BOLD("No of CPU cycles ")) 
                          << uint64_t(cycles/(1.0*counter)) << std::endl; //\xE6
            else if (mean_time < 1.0e-6)
                std::cout << static_cast<long int>(counter)
                          << FGRN(BOLD(" runs, average elapsed time is "))
                          << mean_time/1.0e-09 << " ns" << ". " << FGRN(BOLD("No of CPU cycles ")) 
                          << uint64_t(cycles/(1.0*counter)) << std::endl;
            else
                std::cout << static_cast<long int>(counter)
                          << FGRN(BOLD(" runs, average elapsed time is ")) 
                          << mean_time << " s" << ". " << FGRN(BOLD("No of CPU cycles ")) 
                          << uint64_t(cycles/(1.0*counter)) << std::endl;

            break;
        }
    }

    return best_time;
}



// timeit with return values
#include <tuple>
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
            mean_time /= counter;
            // if (mean_time >= 1.0e-3 && mean_time < 1.)
            //     mean_time /= 1.0e-03;
            // else if (mean_time >= 1.0e-6 && mean_time < 1.0e-3)
            //     mean_time /= 1.0e-06;
            // else if (mean_time < 1.0e-6)
            //     mean_time /= 1.0e-09;
            // else
            //     mean_time = mean_time;

            break;
        }
    }

    return std::make_tuple(mean_time,uint64_t(cycles/(1.0*counter)));
}


// Define a no operation function
void no_op(){}

//clobber
template <typename T> void unused(T &&x) { asm("" ::"m"(x)); }
