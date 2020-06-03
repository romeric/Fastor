#ifndef TENSORBASE_H
#define TENSORBASE_H

#include "Fastor/config/config.h"
#include "Fastor/meta/tensor_meta.h"


namespace Fastor {

template<typename T, size_t ... Rest>
class Tensor;
template<typename T, size_t ... Rest>
class TensorMap;


template<class Derived, FASTOR_INDEX Rank>
class AbstractTensor {
public:
    constexpr FASTOR_INLINE AbstractTensor() = default;
    FASTOR_INLINE const Derived& self() const {return *static_cast<const Derived*>(this);}
    FASTOR_INLINE Derived& self() {return *static_cast<Derived*>(this);}

    static constexpr FASTOR_INDEX Dimension = Rank;
#ifndef FASTOR_DYNAMIC_MODE
    static constexpr FASTOR_INDEX size() {return Derived::Size;}
#else
    FASTOR_INDEX size() const {return (*static_cast<const Derived*>(this)).size();}
#endif
};

}

#endif // TENSORBASE_H
