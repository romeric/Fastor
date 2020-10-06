#ifndef EXTENDED_ALGORITHMS_H
#define EXTENDED_ALGORITHMS_H

#include <algorithm>
#include <functional>
#include <utility>
#include <type_traits>
#include <array>
#include <vector>
#include <numeric>
#include <cstring>

#if defined(FASTOR_UNIX_OS)
#include <sys/resource.h>
#endif

namespace Fastor {

// Implementation of STL iota to work on other types such as
// std::complex. For std::complex iota_impl increments the real
// part only
template<class ForwardIt, class T>
inline void iota_impl(ForwardIt first, ForwardIt last, T value)
{
    while(first != last) {
        *first++ = value;
        value += T(1);
    }
}

template <typename T, size_t N>
inline std::array<int,N> argsort(const std::array<T,N> &v) {
  std::array<int,N> idx;
  std::iota(idx.begin(),idx.end(),0);
  std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] < v[i2];});
  return idx;
}


template<typename T, size_t N, typename std::enable_if<std::is_arithmetic<T>::value,bool>::type = 0>
inline std::string itoa(const std::array<T,N>& arr) {
    std::string out = std::to_string(arr[0]);
    for (size_t i=1; i<N; ++i)
        out += ","+std::to_string(arr[i]);
    return out;
}


#if defined(FASTOR_UNIX_OS)
inline size_t set_stack_size(size_t size) {
    // If the function does not work, copy-paste it within
    // the body of the main

    // size is stack size in MB (for instance provide 80 for 80MB)
    // returns old stack size in MB
    const rlim_t stacksize = size*1024*1024;
    struct rlimit rl;
    int result;
    result = getrlimit(RLIMIT_STACK, &rl);
    rlim_t old = rl.rlim_cur = stacksize;
    if (result==0) {
        if (rl.rlim_cur < stacksize) {
            rl.rlim_cur = stacksize;
            result = setrlimit(RLIMIT_STACK,&rl);
            FASTOR_ASSERT(result !=0, "CHANGING STACK SIZE FAILED");
        }
    }

    return old;
}
#endif



// Get sign of a number
template <typename T>
inline constexpr int signum(T x, [[gnu::unused]] std::false_type is_signed) {
    return T(0) < x;
}
template <typename T>
inline constexpr int signum(T x, [[gnu::unused]] std::true_type is_signed) {
    return (T(0) < x) - (x < T(0));
}
template <typename T>
inline constexpr int signum(T x) {
    return signum(x, std::is_signed<T>());
}

// Get a string +/- based on sign
template <typename T> std::string signum_string(T val) {
    return signum(val) == 1 ? "+" : "-";
}


} // end of namespace Fastor


#endif // EXTENDED_ALGORITHMS_H

