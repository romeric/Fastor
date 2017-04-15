#ifndef TENSORBASE_H
#define TENSORBASE_H

#include "commons/commons.h"
#include "meta/tensor_meta.h"


namespace Fastor {

template<typename T, size_t ... Rest>
class Tensor;


template<class Derived, size_t Rank>
class AbstractTensor {
public:
    constexpr FASTOR_INLINE AbstractTensor() = default;
    FASTOR_INLINE const Derived& self() const {return *static_cast<const Derived*>(this);}
    FASTOR_INLINE Derived& self() {return *static_cast<Derived*>(this);}

    static constexpr FASTOR_INDEX Dimension = Rank;
    static constexpr FASTOR_INDEX size() {return Derived::Size;}
};

}

#endif // TENSORBASE_H
