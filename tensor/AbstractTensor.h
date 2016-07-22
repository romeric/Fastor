#ifndef TENSORBASE_H
#define TENSORBASE_H

#include "commons/commons.h"
#include "meta/tensor_meta.h"

namespace Fastor {


template<class Derived, size_t Rank>
class AbstractTensor {
public:
    AbstractTensor() = default;
    FASTOR_INLINE const Derived& self() const {return *static_cast<const Derived*>(this);}

    static constexpr FASTOR_INDEX Dimension = Rank;
    constexpr FASTOR_INDEX size() const {return Derived::Size;}
};

}

#endif // TENSORBASE_H
