#ifndef EINSUM_H
#define EINSUM_H

#include "Fastor/backend/backend.h"
#include "Fastor/tensor/Tensor.h"
#include "Fastor/meta/einsum_meta.h"
#include "Fastor/tensor_algebra/indicial.h"
#include "Fastor/backend/voigt.h"

#include "Fastor/tensor_algebra/permutation.h"
#include "Fastor/tensor_algebra/permute.h"
#include "Fastor/tensor_algebra/innerproduct.h"
#include "Fastor/tensor_algebra/outerproduct.h"
#include "Fastor/tensor_algebra/contraction.h"
#include "Fastor/tensor_algebra/contraction_single.h"
#include "Fastor/tensor_algebra/strided_contraction.h"

namespace Fastor {


// Single tensor
//-----------------------------------------------------------------------------------------------------------------------//
template<class Index_I, typename T, size_t ... Rest0>
FASTOR_INLINE
auto einsum(const Tensor<T,Rest0...> &a)
-> decltype(extractor_contract_1<Index_I>::contract_impl(a)) {
    static_assert(einsum_index_checker<Index_I>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE INNER INSTEAD");
    return extractor_contract_1<Index_I>::contract_impl(a);
}
//-----------------------------------------------------------------------------------------------------------------------//


// Two tensor (by-pair)
//-----------------------------------------------------------------------------------------------------------------------//
// Inner product case
//-----------------------------------------------------------------------------------------------------------------------//
template<class Index_I, class Index_J,
         typename T, size_t ... Rest0, size_t ... Rest1,
         enable_if_t_<is_pair_reduction_v<Index_I,Index_J>,bool> = false>
FASTOR_INLINE
Tensor<T> einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) {
    return inner(a,b);
}

// General by-pair product cases
//-----------------------------------------------------------------------------------------------------------------------//
template<class Index_I, class Index_J,
         typename T, size_t ... Rest0, size_t ... Rest1,
         typename std::enable_if<!is_pair_reduction<Index_I,Index_J>::value &&
         !internal::is_generalised_matrix_vector<Index_I,Index_J>::value &&
         !internal::is_generalised_vector_matrix<Index_I,Index_J>::value &&
         !internal::is_generalised_matrix_matrix<Index_I,Index_J>::value
         ,bool>::type=0>
FASTOR_INLINE
auto einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b)
-> decltype(extractor_contract_2<Index_I,Index_J>::contract_impl(a,b)) {

    static_assert(einsum_index_checker<typename concat_<Index_I,Index_J>::type>::value,
                  "INDICES FOR EINSUM FUNCTION CANNOT APPEAR MORE THAN TWICE. USE INNER INSTEAD");

    // // Dispatch to the right routine
    // using vectorisability = is_vectorisable<Index_I,Index_J,Tensor<T,Rest1...>>;
    // // constexpr bool is_reducible = vectorisability::last_index_contracted;
    // constexpr bool is_reducible = vectorisability::is_reducible;
    // FASTOR_IF_CONSTEXPR (is_reducible) {
    //     return extractor_reducible_contract<Index_I,Index_J>::contract_impl(a,b);
    // }
    // else {
    //    return extractor_contract_2<Index_I,Index_J>::contract_impl(a,b);
    // }
    return extractor_contract_2<Index_I,Index_J>::contract_impl(a,b);
}


template<class Index_I, class Index_J,
         typename T, size_t ...Rest0, size_t ...Rest1,
         typename std::enable_if<
         internal::is_generalised_matrix_vector<Index_I,Index_J>::value,
         bool>::type = 0>
FASTOR_INLINE
auto
einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) //{
-> decltype(extractor_contract_2<Index_I,Index_J>::contract_impl(a,b)) {

    constexpr size_t which_one_is_vector = internal::is_generalised_matrix_vector<Index_I,Index_J>::which_one_is_vector;
    constexpr size_t matches_up_to = internal::is_generalised_matrix_vector<Index_I,Index_J>::matches_up_to;
    constexpr size_t rest0[sizeof...(Rest0)] = {Rest0...};
    constexpr size_t rest1[sizeof...(Rest1)] = {Rest1...};
    constexpr size_t product = which_one_is_vector == 1 ? partial_prod(rest0, matches_up_to) :  partial_prod(rest1, matches_up_to);
    constexpr size_t vec_product = which_one_is_vector == 1 ? pack_prod<Rest1...>::value : pack_prod<Rest0...>::value;
    decltype(extractor_contract_2<Index_I,Index_J>::contract_impl(a,b)) out;
    which_one_is_vector == 1 ? _matmul<T,product,vec_product,1>(a.data(),b.data(),out.data()) :\
      _matmul<T,product,vec_product,1>(b.data(),a.data(),out.data());
    return out;
}


template<class Index_I, class Index_J,
        typename T, size_t ...Rest0, size_t ...Rest1,
        typename std::enable_if<
        internal::is_generalised_vector_matrix<Index_I,Index_J>::value,
        bool>::type = 0>
FASTOR_INLINE
auto
einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) //{
-> decltype(extractor_contract_2<Index_I,Index_J>::contract_impl(a,b)) {

    constexpr size_t which_one_is_vector = internal::is_generalised_vector_matrix<Index_I,Index_J>::which_one_is_vector;
    constexpr size_t matches_up_to = internal::is_generalised_vector_matrix<Index_I,Index_J>::matches_up_to;
    constexpr size_t rest0[sizeof...(Rest0)] = {Rest0...};
    constexpr size_t rest1[sizeof...(Rest1)] = {Rest1...};
    constexpr size_t product = which_one_is_vector == 1 ? partial_prod_reverse(rest0, matches_up_to) :  partial_prod_reverse(rest1, matches_up_to);
    constexpr size_t vec_product = which_one_is_vector == 1 ? pack_prod<Rest1...>::value : pack_prod<Rest0...>::value;
    decltype(extractor_contract_2<Index_I,Index_J>::contract_impl(a,b)) out;
    which_one_is_vector == 1 ? _matmul<T,1,vec_product,product>(b.data(),a.data(),out.data()) :\
      _matmul<T,1,vec_product,product>(a.data(),b.data(),out.data());
    return out;
}

template<class Index_I, class Index_J,
        typename T, size_t ...Rest0, size_t ...Rest1,
        typename std::enable_if<
        internal::is_generalised_matrix_matrix<Index_I,Index_J>::value,
        bool>::type = 0>
FASTOR_INLINE
auto
einsum(const Tensor<T,Rest0...> &a, const Tensor<T,Rest1...> &b) //{
-> decltype(extractor_contract_2<Index_I,Index_J>::contract_impl(a,b)) {

    constexpr size_t matches_up_to = internal::is_generalised_matrix_matrix<Index_I,Index_J>::ncontracted;
    constexpr size_t rest0[sizeof...(Rest0)] = {Rest0...};
    constexpr size_t rest1[sizeof...(Rest1)] = {Rest1...};
    constexpr size_t K_product = partial_prod(rest1, matches_up_to - 1);
    constexpr size_t M = partial_prod(rest0, sizeof...(Rest0) - matches_up_to - 1);
    constexpr size_t N = partial_prod(rest1, sizeof...(Rest1) - 1, matches_up_to);

    decltype(extractor_contract_2<Index_I,Index_J>::contract_impl(a,b)) out;
    _matmul<T,M,K_product,N>(a.data(),b.data(),out.data());
    return out;
}
//-----------------------------------------------------------------------------------------------------------------------//


// matmul dispatcher for 2nd order tensors (matrix-matrix)
// also includes matrix-vector and vector-matrix when vector is of size
// nx1 or 1xn
template<class Ind0, class Ind1,
         typename T, size_t I, size_t J, size_t K,
         typename std::enable_if<Ind0::Size==2 && Ind1::Size==2 &&
                                 Ind0::values[1] == Ind1::values[0] &&
                                 Ind0::values[1] != Ind0::values[0] &&
                                 Ind0::values[1] != Ind1::values[1] &&
                                 Ind0::values[0] != Ind1::values[1]
                                 ,bool>::type = 0>
FASTOR_INLINE Tensor<T,I,K>
einsum(const Tensor<T,I,J> &a, const Tensor<T,J,K> &b) {
    Tensor<T,I,K> out;
    _matmul<T,I,J,K>(a.data(),b.data(),out.data());
    return out;
}


// matmul dispatcher for matrix-vector
template<class Ind0, class Ind1,
         typename T, size_t I, size_t J,
         typename std::enable_if<Ind0::Size==2 && Ind1::Size==1 &&
                                 Ind0::values[1] == Ind1::values[0] &&
                                 Ind0::values[0] != Ind1::values[0]
                                 ,bool>::type = 0>
FASTOR_INLINE Tensor<T,I>
einsum(const Tensor<T,I,J> &a, const Tensor<T,J> &b) {
    Tensor<T,I> out;
    _matmul<T,I,J,1>(a.data(),b.data(),out.data());
    return out;
}

// matmul dispatcher for matrix-vector
template<class Ind0, class Ind1,
         typename T, size_t I, size_t J,
         typename std::enable_if<Ind0::Size==2 && Ind1::Size==1 &&
                                 Ind0::values[0] == Ind1::values[0] &&
                                 Ind0::values[1] != Ind1::values[0],bool>::type = 0>
FASTOR_INLINE Tensor<T,J>
einsum(const Tensor<T,I,J> &a, const Tensor<T,I> &b) {
    Tensor<T,J> out;
     _matmul<T,1,I,J>(b.data(),a.data(),out.data());
    return out;
}


// matmul dispatcher for vector-matrix
template<class Ind0, class Ind1,
         typename T, size_t I, size_t J,
         typename std::enable_if<Ind1::Size==2 && Ind0::Size==1 &&
                                 Ind1::values[0] == Ind0::values[0] &&
                                 Ind1::values[1] != Ind0::values[0],bool>::type = 0>
FASTOR_INLINE Tensor<T,J>
einsum(const Tensor<T,I> &a, const Tensor<T,I,J> &b) {
    Tensor<T,J> out;
    _matmul<T,1,I,J>(a.data(),b.data(),out.data());
    return out;
}


// matmul dispatcher for vector-matrix
template<class Ind0, class Ind1,
         typename T, size_t I, size_t J,
         typename std::enable_if<Ind1::Size==2 && Ind0::Size==1 &&
                                 Ind1::values[1] == Ind0::values[0] &&
                                 Ind1::values[0] != Ind0::values[0],bool>::type = 0>
FASTOR_INLINE Tensor<T,I>
einsum(const Tensor<T,J> &a, const Tensor<T,I,J> &b) {
    Tensor<T,I> out;
    _matmul<T,I,J,1>(b.data(),a.data(),out.data());
    return out;
}



#ifdef FASTOR_AVX_IMPL

// Specific overloads

// With Voigt conversion
template<class Ind0, class Ind1, int Convert,
         typename T, size_t I, size_t J, size_t K, size_t L,
         typename std::enable_if<(std::is_same<T,float>::value || std::is_same<T,double>::value) &&
                                 I==J && J==K && K==L && (I==2 || I==3) &&
                                 Ind0::NoIndices==2 && Ind1::NoIndices==2 && Convert==FASTOR_Voigt,bool>::type = 0>
FASTOR_INLINE typename VoigtType<T,I,J,K,L>::return_type
einsum(const Tensor<T,I,J> & a, const Tensor<T,K,L> &b) {

    using OutTensor = typename VoigtType<T,I,J,K,L>::return_type;
    OutTensor out;

    constexpr int i = static_cast<int>(Ind0::values[0]);
    constexpr int j = static_cast<int>(Ind0::values[1]);
    constexpr int k = static_cast<int>(Ind1::values[0]);
    constexpr int l = static_cast<int>(Ind1::values[1]);

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
                                 Ind0::Size==2 && Ind1::Size==2 && Convert==FASTOR_Voigt,bool>::type = 0>
FASTOR_INLINE typename VoigtType<T,I,J,K,L>::return_type
einsum(const Tensor<T,I,J> & a, const Tensor<T,K,L> &b) {

    constexpr int i = static_cast<int>(Ind0::values[0]);
    constexpr int j = static_cast<int>(Ind0::values[1]);
    constexpr int k = static_cast<int>(Ind1::values[0]);
    constexpr int l = static_cast<int>(Ind1::values[1]);

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

#endif

} // end of namespace

#endif // EINSUM_H
