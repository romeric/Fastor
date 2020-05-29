#ifndef EINSUM_NETWORK_H
#define EINSUM_NETWORK_H

#include "Fastor/tensor_algebra/network_contraction.h"
#include "Fastor/tensor_algebra/network_contraction_no_opmin.h"

namespace Fastor {

// Networks
//-----------------------------------------------------------------------------------------
#ifndef FASTOR_DONT_PERFORM_OP_MIN

// 3
template<class Index_I, class Index_J, class Index_K,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2>
FASTOR_INLINE
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c)
-> decltype(extractor_contract_3<Index_I,Index_J,Index_K>::contract_impl(a,b,c)) {
    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J,Index_K>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");
    return extractor_contract_3<Index_I,Index_J,Index_K>::contract_impl(a,b,c);
}


// 4
template<class Index_I, class Index_J, class Index_K, class Index_L,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3>
FASTOR_INLINE auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b, const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d)
-> decltype(extractor_contract_4<Index_I,Index_J,Index_K,Index_L>::contract_impl(a,b,c,d)) {

    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J,Index_K,Index_L>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");
    return extractor_contract_4<Index_I,Index_J,Index_K,Index_L>::contract_impl(a,b,c,d);
}


// 5
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4>
FASTOR_INLINE
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                 const Tensor<T,Rest4...> &e)
-> decltype(extractor_contract_5<Index_I,Index_J,Index_K,Index_L,Index_M>::contract_impl(a,b,c,d,e)) {

    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J,Index_K,Index_L,Index_M>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");
    return extractor_contract_5<Index_I,Index_J,Index_K,Index_L,Index_M>::contract_impl(a,b,c,d,e);
}


// 6
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, class Index_N,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2, size_t ... Rest3, size_t ... Rest4, size_t ... Rest5>
FASTOR_INLINE
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
                 const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
                 const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f)
-> decltype(extractor_contract_6<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>::contract_impl(a,b,c,d,e,f)) {

    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");
    return extractor_contract_6<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N>::contract_impl(a,b,c,d,e,f);
}


// 7
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, class Index_N, class Index_O,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2,
            size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6>
FASTOR_INLINE
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
            const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
            const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
            const Tensor<T,Rest6...> &g)
-> decltype(extractor_contract_7<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O>::contract_impl(a,b,c,d,e,f,g)) {

    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");
    return extractor_contract_7<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O>::contract_impl(a,b,c,d,e,f,g);
}


// 8
template<class Index_I, class Index_J, class Index_K, class Index_L, class Index_M, class Index_N, class Index_O, class Index_P,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2,
            size_t ... Rest3, size_t ... Rest4, size_t ... Rest5, size_t ... Rest6, size_t ... Rest7>
FASTOR_INLINE
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b,
            const Tensor<T,Rest2...> &c, const Tensor<T,Rest3...> &d,
            const Tensor<T,Rest4...> &e, const Tensor<T,Rest5...> &f,
            const Tensor<T,Rest6...> &g, const Tensor<T,Rest7...> &h)
-> decltype(extractor_contract_8<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O,Index_P>::contract_impl(a,b,c,d,e,f,g,h)) {

    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O,Index_P>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE CONTRACTION INSTEAD");
    return extractor_contract_8<Index_I,Index_J,Index_K,Index_L,Index_M,Index_N,Index_O,Index_P>::contract_impl(a,b,c,d,e,f,g,h);
}



#else



// No operation minimisation
//------------------------------------------------------------------------------------------------------------------//

// 3
template<class Index_I, class Index_J, class Index_K,
         typename T, size_t ... Rest0, size_t ... Rest1, size_t ... Rest2>
FASTOR_INLINE
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
FASTOR_INLINE
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
FASTOR_INLINE
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
FASTOR_INLINE
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
FASTOR_INLINE
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
FASTOR_INLINE
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
FASTOR_INLINE
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
FASTOR_INLINE
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

} // end of namespace Fastor


#endif
