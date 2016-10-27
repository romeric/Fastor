#ifndef RANGE_H
#define RANGE_H

#include "commons/commons.h"

namespace Fastor {

template<size_t F, size_t L, size_t S=1>
struct Range {
    static constexpr size_t first = F;
    static constexpr size_t last= L;
    static constexpr size_t step = S;
};

}

#endif // RANGE_H

