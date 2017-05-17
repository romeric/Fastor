#ifndef EINSUM_H
#define EINSUM_H

#include "backend/backend.h"
#include "tensor/Tensor.h"
#include "meta/einsum_meta.h"
#include "indicial.h"
#include "backend/voigt.h"

#include "reshape.h"
#include "permutation.h"
#include "reduction.h"
#include "summation.h"
#include "outerproduct.h"
#include "contraction.h"
#include "contraction_single.h"
#include "strided_contraction.h"
#include "network_contraction.h"
#include "strided_network_contraction.h"
#include "network_contraction_no_opmin.h"

namespace Fastor {


// Single tensor
// This does not make sense
//template<class Index_I, typename T, size_t ... Rest0,
//         typename std::enable_if<is_single_reduction<Index_I>::value,bool>::type=0>
//auto einsum(const Tensor<T,Rest0...> &a)
//-> decltype(extractor_contract_1<Index_I>::contract_impl(a)) {
//    return inner(a);
//}


template<class Index_I, typename T, size_t ... Rest0,
         typename std::enable_if<!is_single_reduction<Index_I>::value,bool>::type=0>
auto einsum(const Tensor<T,Rest0...> &a)
-> decltype(extractor_contract_1<Index_I>::contract_impl(a)) {
    static_assert(einsum_index_checker<Index_I>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");
    return extractor_contract_1<Index_I>::contract_impl(a);
}


// Two tensor

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

    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");

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




// Networks
//-----------------------------------------------------------------------------------------
#ifndef FASTOR_DONT_PERFORM_OP_MIN


// 3
template<class Index_I, class Index_J, class Index_K,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c)
-> decltype(extractor_contract<Index_I,Index_J,Index_K>::contract_impl(a,b,c)) {

    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J,Index_K>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");

    // Dispatch to the right routine
    using Index0 = typename concat_<Index_I,Index_J>::type;
    using vectorisability = is_vectorisable<Index0,Index_K,Tensor<T,Rest2...>>;
    constexpr bool is_reducible = vectorisability::last_index_contracted;
    if (is_reducible) {
        return extractor_strided_contract<Index_I,Index_J,Index_K>::contract_impl(a,b,c);
    }
    else {
        return extractor_contract<Index_I,Index_J,Index_K>::contract_impl(a,b,c);
    }
}



// 4
template<class Index_I, class Index_J, class Index_K, class Index_L,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d)
-> decltype(extractor_contract_4<Index_I,Index_J,Index_K,Index_L>::contract_impl(a,b,c,d)) {

    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J,Index_K,Index_L>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");

    // Dispatch to the right routine
    using Index0 = typename concat_<Index_I,Index_J,Index_K>::type;
    using vectorisability = is_vectorisable<Index0,Index_L,Tensor<T,Rest3...>>;
    constexpr bool is_reducible = vectorisability::last_index_contracted;
    if (is_reducible) {
        return extractor_strided_contract_4<Index_I,Index_J,Index_K,Index_L>::contract_impl(a,b,c,d);
    }
    else {
        return extractor_contract_4<Index_I,Index_J,Index_K,Index_L>::contract_impl(a,b,c,d);
    }
}



// 5
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                 const Tensor<T,Rest4...> &e)
-> decltype(extractor_contract_5<Index_I,Index_J,Index_K,Index_L,Index_M>::contract_impl(a,b,c,d,e)) {

    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J,Index_K,Index_L,Index_M>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");

    // Dispatch to the right routine
    using Index0 = typename concat_<Index_I,Index_J,Index_K,Index_L>::type;
    using vectorisability = is_vectorisable<Index0,Index_M,Tensor<T,Rest4...>>;
    constexpr bool is_reducible = vectorisability::last_index_contracted;
    if (is_reducible) {
        return extractor_strided_contract_5<Index_I,Index_J,Index_K,Index_L,Index_M>::contract_impl(a,b,c,d,e);
    }
    else {
        return extractor_contract_5<Index_I,Index_J,Index_K,Index_L,Index_M>::contract_impl(a,b,c,d,e);
    }
}


// 6
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, class Index_N,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4, size_t ... Rest5>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                 const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f)
-> decltype(extractor_contract_6<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>::contract_impl(a,b,c,d,e,f)) {

    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");

    // Dispatch to the right routine
    using Index0 = typename concat_<Index_I,Index_J,Index_K,Index_L,Index_M>::type;
    using vectorisability = is_vectorisable<Index0,Index_N,Tensor<T,Rest5...>>;
    constexpr bool is_reducible = vectorisability::last_index_contracted;
    if (is_reducible) {
        return extractor_strided_contract_6<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>::contract_impl(a,b,c,d,e,f);
    }
    else {
        return extractor_contract_6<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>::contract_impl(a,b,c,d,e,f);
    }
}


// 7
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, class Index_N, class Index_O,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2,
            size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
            const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
            const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
            const Tensor<T,Rest6...> &g)
-> decltype(extractor_contract_7<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O>::contract_impl(a,b,c,d,e,f,g)) {

    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");

    // Dispatch to the right routine
    using Index0 = typename concat_<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>::type;
    using vectorisability = is_vectorisable<Index0,Index_O,Tensor<T,Rest6...>>;
    constexpr bool is_reducible = vectorisability::last_index_contracted;
    if (is_reducible) {
        return extractor_strided_contract_7<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O>::contract_impl(a,b,c,d,e,f,g);
    }
    else {
        return extractor_contract_7<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O>::contract_impl(a,b,c,d,e,f,g);
    }
}



// 8
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, class Index_N, class Index_O, class Index_P,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2,
            size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6, size_t ... Rest7>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
            const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
            const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
            const Tensor<T,Rest6...> &g, const Tensor<T,Rest7...> &h)
-> decltype(extractor_contract_8<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O,Index_P>::contract_impl(a,b,c,d,e,f,g,h)) {

    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O,Index_P>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");

    // Dispatch to the right routine
    using Index0 = typename concat_<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O>::type;
    using vectorisability = is_vectorisable<Index0,Index_P,Tensor<T,Rest7...>>;
    constexpr bool is_reducible = vectorisability::last_index_contracted;
    if (is_reducible) {
        return extractor_strided_contract_8<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O,Index_P>::contract_impl(a,b,c,d,e,f,g,h);
    }
    else {
        return extractor_contract_8<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O,Index_P>::contract_impl(a,b,c,d,e,f,g,h);
    }
}





#else


// No operation minimisation
//------------------------------------------------------------------------------------------------------------------//

// 3
template<class Index_I, class Index_J, class Index_K,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c)
-> decltype(extractor_contract_3_no_opt<Index_I,Index_J,Index_K>::contract_impl(a,b,c)) {
    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J,Index_K>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");
    return extractor_contract_3_no_opt<Index_I,Index_J,Index_K>::contract_impl(a,b,c);
}

// 4
template<class Index_I, class Index_J, class Index_K, class Index_L,
         typename T, size_t ... Rest0, size_t ... Rest1,
         size_t ... Rest2, size_t ... Rest3>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
            const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d)
-> decltype(extractor_contract_4_no_opt<Index_I,Index_J,Index_K,Index_L>::contract_impl(a,b,c,d)) {
    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J,Index_K,Index_L>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");
    return extractor_contract_4_no_opt<Index_I,Index_J,Index_K,Index_L>::contract_impl(a,b,c,d);
}

// 5
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M,
         typename T, size_t ... Rest0, size_t ... Rest1,
         size_t ... Rest2, size_t ... Rest3, size_t ... Rest4>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
            const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
            const Tensor<T,Rest4...> &e)
-> decltype(extractor_contract_5_no_opt<Index_I,Index_J,Index_K,Index_L,Index_M>::contract_impl(a,b,c,d,e)) {
    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J,Index_K,Index_L,Index_M>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");
    return extractor_contract_5_no_opt<Index_I,Index_J,Index_K,Index_L,Index_M>::contract_impl(a,b,c,d,e);
}

// 6
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, class Index_N,
         typename T, size_t ... Rest0, size_t ... Rest1,
         size_t ... Rest2, size_t ... Rest3, size_t ... Rest4, size_t ... Rest5>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
            const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
            const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f)
-> decltype(extractor_contract_6_no_opt<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>::contract_impl(a,b,c,d,e,f)) {
    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");
    return extractor_contract_6_no_opt<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>::contract_impl(a,b,c,d,e,f);
}

// 7
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, class Index_N, class Index_O,
         typename T, size_t ... Rest0, size_t ... Rest1,
         size_t ... Rest2, size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
            const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
            const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
            const Tensor<T,Rest6...> &g)
-> decltype(extractor_contract_7_no_opt<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O>::contract_impl(a,b,c,d,e,f,g)) {
    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");
    return extractor_contract_7_no_opt<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O>::contract_impl(a,b,c,d,e,f,g);
}


// 8
template<class Ind0, class Ind1, class Ind2, class Ind3, class Ind4, class Ind5, class Ind6, class Ind7,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3,
         size_t ... Rest4, size_t ... Rest5, size_t ... Rest6, size_t ... Rest7>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
            const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
            const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
            const Tensor<T,Rest6...> &g, const Tensor<T,Rest7...> &h)
-> decltype(extractor_contract_8_no_opt<Ind0,Ind1,Ind2,Ind3,Ind4,Ind5,Ind6,Ind7>::contract_impl(a,b,c,d,e,f,g,h)) {
    static_assert(einsum_index_checker<typename concat_<Ind0,Ind1,Ind2,Ind3,Ind4,Ind5,Ind6,Ind7>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");
    return extractor_contract_8_no_opt<Ind0,Ind1,Ind2,Ind3,Ind4,Ind5,Ind6,Ind7>::contract_impl(a,b,c,d,e,f,g,h);
}


// 9
template<class Ind0, class Ind1, class Ind2, class Ind3, class Ind4, class Ind5, class Ind6,
         class Ind7, class Ind8,
         typename T, size_t ... Rest0, size_t ... Rest1,
         size_t ... Rest2, size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6,
         size_t ... Rest7, size_t ... Rest8>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
            const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
            const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
            const Tensor<T,Rest6...> &g, const Tensor<T,Rest7...> &h,
            const Tensor<T,Rest8...> &h1)
-> decltype(extractor_contract_9_no_opt<Ind0,Ind1,Ind2,Ind3,Ind4,Ind5,Ind6,Ind7,Ind8>::contract_impl(a,b,c,d,e,f,g,h,h1)) {
    static_assert(einsum_index_checker<typename concat_<Ind0,Ind1,Ind2,Ind3,Ind4,Ind5,Ind6,Ind7,Ind8>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");
    return extractor_contract_9_no_opt<Ind0,Ind1,Ind2,Ind3,Ind4,Ind5,Ind6,Ind7,Ind8>::contract_impl(a,b,c,d,e,f,g,h,h1);
}



// 10
template<class Ind0, class Ind1, class Ind2, class Ind3, class Ind4, class Ind5, class Ind6,
         class Ind7, class Ind8, class Ind9,
         typename T, size_t ... Rest0, size_t ... Rest1,
         size_t ... Rest2, size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6,
         size_t ... Rest7, size_t ... Rest8, size_t ... Rest9>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
            const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
            const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
            const Tensor<T,Rest6...> &g, const Tensor<T,Rest7...> &h,
            const Tensor<T,Rest8...> &h1, const Tensor<T,Rest9...> &h2)
-> decltype(extractor_contract_10_no_opt<Ind0,Ind1,Ind2,Ind3,Ind4,Ind5,Ind6,Ind7,Ind8,Ind9>::contract_impl(a,b,c,d,e,f,g,h,h1,h2)) {
    static_assert(einsum_index_checker<typename concat_<Ind0,Ind1,Ind2,Ind3,Ind4,Ind5,Ind6,Ind7,Ind8,Ind9>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");
    return extractor_contract_10_no_opt<Ind0,Ind1,Ind2,Ind3,Ind4,Ind5,Ind6,Ind7,Ind8,Ind9>::contract_impl(a,b,c,d,e,f,g,h,h1,h2);
}



// 11
template<class Ind0, class Ind1, class Ind2, class Ind3, class Ind4, class Ind5, class Ind6,
         class Ind7, class Ind8, class Ind9, class Ind10,
         typename T, size_t ... Rest0, size_t ... Rest1,
         size_t ... Rest2, size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6,
         size_t ... Rest7, size_t ... Rest8, size_t ... Rest9, size_t ... Rest10>
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
            const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
            const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
            const Tensor<T,Rest6...> &g, const Tensor<T,Rest7...> &h,
            const Tensor<T,Rest8...> &h1, const Tensor<T,Rest9...> &h2,
            const Tensor<T,Rest10...> &h3)
-> decltype(extractor_contract_11_no_opt<Ind0,Ind1,Ind2,Ind3,Ind4,Ind5,Ind6,Ind7,Ind8,Ind9,Ind10>::contract_impl(a,b,c,d,e,f,g,h,h1,h2,h3)) {
    static_assert(einsum_index_checker<typename concat_<Ind0,Ind1,Ind2,Ind3,Ind4,Ind5,Ind6,Ind7,Ind8,Ind9,Ind10>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");
    return extractor_contract_11_no_opt<Ind0,Ind1,Ind2,Ind3,Ind4,Ind5,Ind6,Ind7,Ind8,Ind9,Ind10>::contract_impl(a,b,c,d,e,f,g,h,h1,h2,h3);
}


#endif


#ifdef __AVX__

// Specific overloads

// With Voigt conversion
template<class Ind0, class Ind1, int Convert,
         typename T, size_t I, size_t J, size_t K, size_t L,
         typename std::enable_if<(std::is_same<T,float>::value || std::is_same<T,double>::value) &&
                                 I==J && J==K && K==L && (I==2 || I==3) &&
                                 Ind0::NoIndices==2 && Ind1::NoIndices==2 && Convert==Voigt,bool>::type = 0>
FASTOR_INLINE typename VoigtType<T,I,J,K,L>::return_type
einsum(const Tensor<T,I,J> & a, const Tensor<T,K,L> &b) {

    using OutTensor = typename VoigtType<T,I,J,K,L>::return_type;
    OutTensor out;

    constexpr int i = static_cast<int>(Ind0::_IndexHolder[0]);
    constexpr int j = static_cast<int>(Ind0::_IndexHolder[1]);
    constexpr int k = static_cast<int>(Ind1::_IndexHolder[0]);
    constexpr int l = static_cast<int>(Ind1::_IndexHolder[1]);

    constexpr bool is_dyadic = i<j && j<k && k<l;
    constexpr bool is_cyclic = (i<j && i<k && i<l) && j>k && j<l;
    static_assert(is_dyadic || is_cyclic, "INCORRECT INPUT FOR EINSUM FUNCTION");

    if (is_dyadic) {
        _outer<T,I,J,K,L>(a.data(),b.data(),out.data());
    }

    if (is_cyclic) {
        _cyclic<T,I,J,K,L>(a.data(),b.data(),out.data());
    }

    return out;
}


template<class Ind0, class Ind1, int Convert,
         typename T, size_t I, size_t J, size_t K, size_t L,
         typename std::enable_if<(!std::is_same<T,float>::value && !std::is_same<T,double>::value) &&
                                 I==J && J==K && K==L && (I==2 || I==3) &&
                                 Ind0::NoIndices==2 && Ind1::NoIndices==2 && Convert==Voigt,bool>::type = 0>
FASTOR_INLINE typename VoigtType<T,I,J,K,L>::return_type
einsum(const Tensor<T,I,J> & a, const Tensor<T,K,L> &b) {

    constexpr int i = static_cast<int>(Ind0::_IndexHolder[0]);
    constexpr int j = static_cast<int>(Ind0::_IndexHolder[1]);
    constexpr int k = static_cast<int>(Ind1::_IndexHolder[0]);
    constexpr int l = static_cast<int>(Ind1::_IndexHolder[1]);

    constexpr bool is_dyadic = i<j && j<k && k<l;
    constexpr bool is_cyclic = (i<j && i<k && i<l) && j>k && j<l;
    static_assert(is_dyadic || is_cyclic, "INCORRECT INPUT FOR EINSUM FUNCTION");

    using Ind = typename concat_<Ind0,Ind1>::type;

    if (is_dyadic) {
        auto out = contraction<Ind0,Ind1>(a,b);
        return voigt(out);
    }

    if (is_cyclic) {
        auto out = permutation<Ind>(contraction<Ind0,Ind1>(a,b));
        return voigt(out);
    }
}


// matmul dispatcher for 2nd order tensors
template<class Ind0, class Ind1,
         typename T, size_t I, size_t J, size_t K,
         typename std::enable_if<Ind0::NoIndices==2 && Ind1::NoIndices==2 &&
                                 Ind0::_IndexHolder[1] == Ind1::_IndexHolder[0] && 
                                 Ind0::_IndexHolder[1] != Ind0::_IndexHolder[0] &&
                                 Ind0::_IndexHolder[1] != Ind1::_IndexHolder[1] &&
                                 Ind0::_IndexHolder[0] != Ind1::_IndexHolder[1],bool>::type = 0>
FASTOR_INLINE Tensor<T,I,K>
einsum(const Tensor<T,I,J> &a, const Tensor<T,J,K> &b) {

    Tensor<T,I,K> out;
    _matmul<T,I,J,K>(a.data(),b.data(),out.data());
    return out;
}


// The following two overloads are provided for an external use case
// A_ijk*B_kl
template<class Ind0, class Ind1,
         typename T, size_t I, size_t J, size_t K, size_t L,
         typename std::enable_if<Ind0::NoIndices==3 && Ind1::NoIndices==2 &&
                                Ind0::_IndexHolder[0] != Ind0::_IndexHolder[1] &&
                                Ind0::_IndexHolder[0] != Ind0::_IndexHolder[2] && 
                                Ind0::_IndexHolder[0] != Ind1::_IndexHolder[0] && 
                                Ind0::_IndexHolder[0] != Ind1::_IndexHolder[1] && 

                                Ind0::_IndexHolder[1] != Ind0::_IndexHolder[2] && 
                                Ind0::_IndexHolder[1] != Ind1::_IndexHolder[0] && 
                                Ind0::_IndexHolder[1] != Ind1::_IndexHolder[1] && 

                                Ind0::_IndexHolder[2] == Ind1::_IndexHolder[0] && 
                                Ind0::_IndexHolder[2] != Ind1::_IndexHolder[1] && 

                                Ind1::_IndexHolder[0] != Ind1::_IndexHolder[1],bool>::type = 0>
FASTOR_INLINE Tensor<T,I,J,L>
einsum(const Tensor<T,I,J,K> &a, const Tensor<T,K,L> &b) {

    Tensor<T,I,J,L> out;
    _matmul<T,I*J,K,L>(a.data(),b.data(),out.data());
    return out;
}

// A_ijk*B_klm
template<class Ind0, class Ind1,
         typename T, size_t I, size_t J, size_t K, size_t L, size_t M,
         typename std::enable_if<Ind0::NoIndices==3 && Ind1::NoIndices==3 &&
                                Ind0::_IndexHolder[0] != Ind0::_IndexHolder[1] &&
                                Ind0::_IndexHolder[0] != Ind0::_IndexHolder[2] && 
                                Ind0::_IndexHolder[0] != Ind1::_IndexHolder[0] && 
                                Ind0::_IndexHolder[0] != Ind1::_IndexHolder[1] && 
                                Ind0::_IndexHolder[0] != Ind1::_IndexHolder[2] && 

                                Ind0::_IndexHolder[1] != Ind0::_IndexHolder[2] && 
                                Ind0::_IndexHolder[1] != Ind1::_IndexHolder[0] && 
                                Ind0::_IndexHolder[1] != Ind1::_IndexHolder[1] &&
                                Ind0::_IndexHolder[1] != Ind1::_IndexHolder[2] && 

                                Ind0::_IndexHolder[2] == Ind1::_IndexHolder[0] && 
                                Ind0::_IndexHolder[2] != Ind1::_IndexHolder[1] &&
                                Ind0::_IndexHolder[2] != Ind1::_IndexHolder[2] && 

                                Ind1::_IndexHolder[0] != Ind1::_IndexHolder[1] &&
                                Ind1::_IndexHolder[0] != Ind1::_IndexHolder[2] &&
                                
                                Ind1::_IndexHolder[1] != Ind1::_IndexHolder[2],bool>::type = 0>
FASTOR_INLINE Tensor<T,I,J,L,M>
einsum(const Tensor<T,I,J,K> &a, const Tensor<T,K,L,M> &b) {

    Tensor<T,I,J,L,M> out;
    _matmul<T,I*J,K,L*M>(a.data(),b.data(),out.data());
    return out;
}

#endif


} // end of namespace

#include "abstract_contraction.h"

#endif // EINSUM_H

