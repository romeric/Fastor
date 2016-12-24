#include <iostream>
#include <iomanip>
#include <vector>
#include <array>


#ifdef __SSE2__
#include <emmintrin.h>
void print(__m128 a) {
    std::cout << a[0] << " " << a[1] << " " << a[2] << " " << a[3] << "\n";
}
void print(__m128d a) {
    std::cout << a[0] << " " << a[1] << "\n";
}
#endif
#ifdef __AVX__
#include <immintrin.h>
void print(__m256 a) {
    std::cout << a[0] << " " << a[1] << " " << a[2] << " " << a[3] <<  " " << 
    a[4] << " " << a[5] << " " << a[6] << " " << a[7] << "\n";
}
void print(__m256d a) {
    std::cout << a[0] << " " << a[1] << " " << a[2] << " " << a[3] << "\n";
}
#endif

template<typename T>
void print(const std::vector<T> &v) {
    for (auto &k: v) {
        std::cout << k << "\n";
    }
    std::cout << std::endl;
}

template<typename T, std::size_t N>
void print(const std::array<T,N> &arr) {
    for (std::size_t i=0; i<N; i++) {
        std::cout << arr[i] << "\n";
    }
    std::cout << std::endl;
}

template<typename T>
void print(const std::vector<std::vector<T>> &arr) {
    for (std::size_t i=0; i<arr.size(); i++) {
        for (std::size_t j=0; j<arr[i].size(); j++) {
            std::cout << arr[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

template<typename T, std::size_t M, std::size_t N>
void print(const std::array<std::array<T,M>,N> &arr) {
    for (std::size_t i=0; i<N; i++) {
        for (std::size_t j=0; j<M; j++) {
            std::cout << arr[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

template<typename T,std::size_t N>
void print(const T *arr) {
    for (std::size_t i=0; i<N; i++) {
        std::cout << arr[i] << "\n";
    }
    std::cout << std::endl;
}


template<typename T>
void print(const T &a) {
    std::cout << a << "\n";
}

template<typename T, typename ... Rest>
void print(const T &first, const Rest& ... rest) {
    print(first);
    print(rest...);
}

void print() {
    std::cout << "\n";
}
/*--------------------------------------*/




// Print horizontally
/*--------------------------------------*/
template<typename T>
void println(const std::vector<T> &v) {
    for (auto &k: v) {
        std::cout << k << " ";
    }
    std::cout << std::endl;
}

template<typename T, std::size_t N>
void println(const std::array<T,N> &arr) {
    for (std::size_t i=0; i<N; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

template<typename T,std::size_t N>
void println(const T *arr) {
    for (std::size_t i=0; i<N; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}


template<typename T>
void println(const T &a) {
    std::cout << a << " ";
}

template<typename T, typename ... Rest>
void println(const T &first, const Rest& ... rest) {
    println(first);
    println(rest...);
}

void println() {
    std::cout << " ";
}
/*--------------------------------------*/






// Warn
/*--------------------------------------*/
template<typename T>
void warn(const T &a) {
    std::cerr << a << "\n";
}

template<typename T, typename ... Rest>
void warn(const T &first, const Rest& ... rest) {
    warn(first);
    warn(rest...);
}
