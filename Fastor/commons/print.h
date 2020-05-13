#ifndef PRINT_H
#define PRINT_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <string>
#include <sstream>

#ifdef FASTOR_SSE2_IMPL
#include <emmintrin.h>
#endif
#ifdef FASTOR_AVX_IMPL
#include <immintrin.h>
#endif

namespace Fastor {


//! IOFormat class for tensors
struct IOFormat {
    inline  IOFormat(
                int precision,
                const std::string& colsep,
                const std::string& rowsep,
                const std::string& rowprefix,
                const std::string& rowsuffix,
                bool print_dimensions
                ) {
        _precision = precision;
        _colsep = colsep;
        _rowsep = rowsep;
        _rowprefix = rowprefix;
        _rowsuffix = rowsuffix;
        _print_dimensions = print_dimensions;
    }

    int _precision;
    std::string _colsep;
    std::string _rowsep;
    std::string _rowprefix;
    std::string _rowsuffix;
    bool _print_dimensions;
};


#ifndef FASTOR_DEFINE_IO_FORMAT
#define FASTOR_DEFINE_IO_FORMAT {std::numeric_limits<double>::digits10,", ","\n","[","]",true};
// #define FASTOR_DEFINE_IO_FORMAT IOFormat(9,",","\n","[","]",true);
#endif




#ifdef FASTOR_SSE2_IMPL
inline void print(__m128 b) {
    const float* a = (const float*)&b;
    std::cout << a[0] << " " << a[1] << " " << a[2] << " " << a[3] << '\n';
}
inline void print(__m128d b) {
    const double* a = (const double*)&b;
    std::cout << a[0] << " " << a[1] << '\n';
}
#endif
#ifdef FASTOR_AVX_IMPL
inline void print(__m256 b) {
    const float* a = (const float*)&b;
    std::cout << a[0] << " " << a[1] << " " << a[2] << " " << a[3] <<  " " <<
    a[4] << " " << a[5] << " " << a[6] << " " << a[7] << '\n';
}
inline void print(__m256d b) {
    const double* a = (const double*)&b;
    std::cout << a[0] << " " << a[1] << " " << a[2] << " " << a[3] << '\n';
}
#endif

template<typename T>
inline void print(const std::vector<T> &v) {
    for (auto &k: v) {
        std::cout << k << '\n';
    }
    std::cout << std::endl;
}

template<typename T, std::size_t N>
inline void print(const std::array<T,N> &arr) {
    for (std::size_t i=0; i<N; i++) {
        std::cout << arr[i] << '\n';
    }
    std::cout << std::endl;
}

template<typename T>
inline void print(const std::vector<std::vector<T>> &arr) {
    for (std::size_t i=0; i<arr.size(); i++) {
        for (std::size_t j=0; j<arr[i].size(); j++) {
            std::cout << arr[i][j] << " ";
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}

template<typename T, std::size_t M>
inline void print(const std::vector<std::array<T,M>> &arr) {
    for (std::size_t i=0; i<arr.size(); i++) {
        for (std::size_t j=0; j<M; j++) {
            std::cout << arr[i][j] << " ";
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}

template<typename T, std::size_t M, std::size_t N>
inline void print(const std::array<std::array<T,M>,N> &arr) {
    for (std::size_t i=0; i<N; i++) {
        for (std::size_t j=0; j<M; j++) {
            std::cout << arr[i][j] << " ";
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}

template<typename T,std::size_t N>
inline void print(const T *arr) {
    for (std::size_t i=0; i<N; i++) {
        std::cout << arr[i] << '\n';
    }
    std::cout << std::endl;
}


template<typename T>
inline void print(const T &a) {
    std::cout << a << '\n';
}

template<typename T, typename ... Rest>
inline void print(const T &first, const Rest& ... rest) {
    print(first);
    print(rest...);
}

inline void print() {
    std::cout << '\n';
}
/*--------------------------------------*/



// Print horizontally
/*--------------------------------------*/
template<typename T>
inline void println(const std::vector<T> &v) {
    for (auto &k: v) {
        std::cout << k << ' ';
    }
    std::cout << '\n';
}

template<typename T, std::size_t N>
inline void println(const std::array<T,N> &arr) {
    for (std::size_t i=0; i<N; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << '\n';
}

template<typename T,std::size_t N>
inline void println(const T *arr) {
    for (std::size_t i=0; i<N; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << '\n';
}


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

inline void println() {
    std::cout << ' ';
}
/*--------------------------------------*/






// Warn
/*--------------------------------------*/
template<typename T>
inline void warn(const T &a) {
    std::cerr << a << '\n';
}

template<typename T, typename ... Rest>
inline void warn(const T &first, const Rest& ... rest) {
    warn(first);
    warn(rest...);
}


} // end of namespace

#endif
