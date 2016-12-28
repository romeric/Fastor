#ifndef EINSUM_H
#define EINSUM_H

#include "backend/backend.h"
#include "tensor/Tensor.h"
#include "meta/einsum_meta.h"
#include "indicial.h"

#include "reshape.h"
#include "permutation.h"
#include "reduction.h"
#include "summation.h"
#include "outerproduct.h"
#include "contraction.h"
#include "strided_contraction.h"
#include "network_contraction.h"
#include "network_contraction_no_opmin.h"

namespace Fastor {


//template<class Index_I, class Index_J,
//         typename T, size_t ... Rest0, size_t ... Rest1>
//auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b)
//-> decltype(extractor_contract_2<Index_I,Index_J>::contract_impl(a,b)) {
//        return extractor_contract_2<Index_I,Index_J>::contract_impl(a,b);
//}

//constexpr bool is_scalar_reduction = is_reduction<Index_I,Index_J>::value;


template<class Index_I, class Index_J,
         typename T, size_t ... Rest0, size_t ... Rest1,
         typename std::enable_if<is_reduction<Index_I,Index_J>::value,bool>::type=0>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b)
-> decltype(extractor_contract_2<Index_I,Index_J>::contract_impl(a,b)) {
    // Dispatch to the right routine
    return inner(a,b);
}


template<class Index_I, class Index_J,
         typename T, size_t ... Rest0, size_t ... Rest1,
         typename std::enable_if<!is_reduction<Index_I,Index_J>::value,bool>::type=0>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b)
-> decltype(extractor_contract_2<Index_I,Index_J>::contract_impl(a,b)) {

    // Dispatch to the right routine
    using vectorisability = is_vectorisable<Index_I,Index_J,Tensor<T,Rest1...>>;
    constexpr bool is_reducible = vectorisability::last_index_contracted;
    if (is_reducible) {
        return extractor_reducible_contract<Index_I,Index_J>::contract_impl(a,b);
    }
    else {
        return extractor_contract_2<Index_I,Index_J>::contract_impl(a,b);
    }
}


template<class Index_I, class Index_J, class Index_K,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c)
-> decltype(extractor_contract<Index_I,Index_J,Index_K>::contract_impl(a,b,c)) {
    return extractor_contract<Index_I,Index_J,Index_K>::contract_impl(a,b,c);
}


template<class Index_I, class Index_J, class Index_K, class Index_L,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d)
-> decltype(extractor_contract_4<Index_I,Index_J,Index_K,Index_L>::contract_impl(a,b,c,d)) {
    return extractor_contract_4<Index_I,Index_J,Index_K,Index_L>::contract_impl(a,b,c,d);
}


template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                 const Tensor<T,Rest4...> &e)
-> decltype(extractor_contract_5<Index_I,Index_J,Index_K,Index_L,Index_M>::contract_impl(a,b,c,d,e)) {
    return extractor_contract_5<Index_I,Index_J,Index_K,Index_L,Index_M>::contract_impl(a,b,c,d,e);
}



template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, class Index_N,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4, size_t ... Rest5>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                 const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f)
-> decltype(extractor_contract_6<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>::contract_impl(a,b,c,d,e,f)) {
    return extractor_contract_6<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>::contract_impl(a,b,c,d,e,f);
}




} // end of namespace

#endif // EINSUM_H

