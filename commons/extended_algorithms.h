#ifndef EXTENDED_ALGORITHMS_H
#define EXTENDED_ALGORITHMS_H

#include <algorithm>
#include <functional>
#include <utility>
#include <type_traits>
#include <array>
#include <vector>

template <typename T, size_t N>
std::array<int,N> argsort(const std::array<T,N> &v) {
  std::array<int,N> idx;
  std::iota(idx.begin(),idx.end(),0);
  std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] < v[i2];});
  return idx;
}

#endif // EXTENDED_ALGORITHMS_H

