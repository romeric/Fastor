#ifndef EXTENDED_ALGORITHMS_H
#define EXTENDED_ALGORITHMS_H

#include <algorithm>
#include <functional>
#include <utility>
#include <type_traits>
#include <array>
#include <vector>

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

#endif // EXTENDED_ALGORITHMS_H

