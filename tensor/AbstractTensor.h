#ifndef TENSORBASE_H
#define TENSORBASE_H

#include "commons/commons.h"
#include "meta/tensor_meta.h"

namespace Fastor {

#define Symmetric -1
#define AntiSymmetric -2
#define Identity -3
#define Voigt -4

}

namespace Fastor {


template<class Derived, size_t DIM>
class AbstractTensor {
public:
    AbstractTensor() = default;
    static constexpr FASTOR_INDEX Dimension = DIM;
//    static const FASTOR_INDEX Dimension = DIM;
//    static const FASTOR_INDEX Size = Derived::Size;
//    static constexpr FASTOR_INDEX Size = Derived::Size;
    FASTOR_INLINE const Derived& self() const {return *static_cast<const Derived*>(this);}
//    FASTOR_INLINE Derived self() const {return static_cast<Derived&>(*this);}
    constexpr FASTOR_INDEX size() const {return Derived::Size;}
//    constexpr FASTOR_INDEX size() const {return Derived::size();}

//    FASTOR_INLINE Derived const& self() const {return static_cast<const Derived&>(*this);}
//    Derived& operator()() const {return self()();}
};

}

#endif // TENSORBASE_H
