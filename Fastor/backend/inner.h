#ifndef INNER_H_
#define INNER_H_

#include "Fastor/meta/meta.h"
#include "Fastor/backend/doublecontract.h"

namespace Fastor {

/* The dependency on doublecontract here is on purpose
    as it creates a necessary layer of indirection to avoid
    the case where M == 0
*/
template<typename T, size_t M,
    enable_if_t_<is_greater_v_<M,0>, bool> = false>
FASTOR_INLINE T _inner(const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b) {
    return _doublecontract<T,M,1>(a,b);
}

template<typename T, size_t M,
    enable_if_t_<M==0, bool> = false>
FASTOR_INLINE T _inner(const T* FASTOR_RESTRICT a, const T* FASTOR_RESTRICT b) {
    return (*a)*(*b);
}

} // end of namespace Fastor

#endif // INNER_H_
