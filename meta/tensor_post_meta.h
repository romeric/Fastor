#ifndef TENSOR_POST_META_H
#define TENSOR_POST_META_H

#include <tensor/Tensor.h>

namespace Fastor {

template<typename TLhs, typename TRhs, size_t DIMS>
struct BinaryAddOp;

template<typename TLhs, typename TRhs, size_t DIMS>
struct BinarySubOp;

template<typename TLhs, typename TRhs, size_t DIMS>
struct BinaryMulOp;

template<typename TLhs, typename TRhs, size_t DIMS>
struct BinaryDivOp;

template<typename Expr, size_t DIMS>
struct UnaryAddOp;

template<typename Expr, size_t DIMS>
struct UnarySubOp;

template<typename Expr, size_t DIMS>
struct UnarySqrtOp;

template<typename Expr, size_t DIMS>
struct UnaryExpOp;

template<typename Expr, size_t DIMS>
struct UnaryLogOp;

template<typename Expr, size_t DIMS>
struct UnarySinOp;

template<typename Expr, size_t DIMS>
struct UnaryCosOp;

template<typename Expr, size_t DIMS>
struct UnaryTanOp;

template<typename Expr, size_t DIMS>
struct UnaryAsinOp;

template<typename Expr, size_t DIMS>
struct UnaryAcosOp;

template<typename Expr, size_t DIMS>
struct UnaryAtanOp;

template<typename Expr, size_t DIMS>
struct UnarySinhOp;

template<typename Expr, size_t DIMS>
struct UnaryCoshOp;

template<typename Expr, size_t DIMS>
struct UnaryTanhOp;




template<class T>
struct is_tensor {
    static constexpr bool value = false;
};

template<class T, size_t ...Rest>
struct is_tensor<Tensor<T,Rest...>> {
    static constexpr bool value = true;
};

template<class T>
struct is_abstracttensor {
    static constexpr bool value = false;
};

template<class T, size_t DIMS>
struct is_abstracttensor<AbstractTensor<T,DIMS>> {
    static constexpr bool value = true;
};

template<class T, size_t ...Rest>
struct is_abstracttensor<Tensor<T,Rest...>> {
    static constexpr bool value = true;
};

template<class TLhs, class TRhs, size_t DIMS>
struct is_abstracttensor<BinaryAddOp<TLhs,TRhs,DIMS>> {
    static constexpr bool value = true;
};

template<class TLhs, class TRhs, size_t DIMS>
struct is_abstracttensor<BinarySubOp<TLhs,TRhs,DIMS>> {
    static constexpr bool value = true;
};

template<class TLhs, class TRhs, size_t DIMS>
struct is_abstracttensor<BinaryMulOp<TLhs,TRhs,DIMS>> {
    static constexpr bool value = true;
};

template<class TLhs, class TRhs, size_t DIMS>
struct is_abstracttensor<BinaryDivOp<TLhs,TRhs,DIMS>> {
    static constexpr bool value = true;
};

template<class Expr, size_t DIMS>
struct is_abstracttensor<UnaryAddOp<Expr,DIMS>> {
    static constexpr bool value = true;
};

template<class Expr, size_t DIMS>
struct is_abstracttensor<UnarySubOp<Expr,DIMS>> {
    static constexpr bool value = true;
};

template<class Expr, size_t DIMS>
struct is_abstracttensor<UnarySqrtOp<Expr,DIMS>> {
    static constexpr bool value = true;
};

template<class Expr, size_t DIMS>
struct is_abstracttensor<UnaryExpOp<Expr,DIMS>> {
    static constexpr bool value = true;
};

template<class Expr, size_t DIMS>
struct is_abstracttensor<UnaryLogOp<Expr,DIMS>> {
    static constexpr bool value = true;
};

template<class Expr, size_t DIMS>
struct is_abstracttensor<UnarySinOp<Expr,DIMS>> {
    static constexpr bool value = true;
};

template<class Expr, size_t DIMS>
struct is_abstracttensor<UnaryCosOp<Expr,DIMS>> {
    static constexpr bool value = true;
};

template<class Expr, size_t DIMS>
struct is_abstracttensor<UnaryTanOp<Expr,DIMS>> {
    static constexpr bool value = true;
};

template<class Expr, size_t DIMS>
struct is_abstracttensor<UnaryAsinOp<Expr,DIMS>> {
    static constexpr bool value = true;
};

template<class Expr, size_t DIMS>
struct is_abstracttensor<UnaryAcosOp<Expr,DIMS>> {
    static constexpr bool value = true;
};

template<class Expr, size_t DIMS>
struct is_abstracttensor<UnaryAtanOp<Expr,DIMS>> {
    static constexpr bool value = true;
};

template<class Expr, size_t DIMS>
struct is_abstracttensor<UnarySinhOp<Expr,DIMS>> {
    static constexpr bool value = true;
};

template<class Expr, size_t DIMS>
struct is_abstracttensor<UnaryCoshOp<Expr,DIMS>> {
    static constexpr bool value = true;
};

template<class Expr, size_t DIMS>
struct is_abstracttensor<UnaryTanhOp<Expr,DIMS>> {
    static constexpr bool value = true;
};


//
template<class T>
struct type_extractor {
    using type = T;
};

template<class T, size_t ...Rest>
struct type_extractor<Tensor<T,Rest...>> {
    using type = T;
};

template <class TLhs, class TRhs>
struct scalar_type_finder {
    using l_scalar_type = typename type_extractor<TLhs>::type;
    using r_scalar_type = typename type_extractor<TRhs>::type;
    using type = typename std::conditional<is_abstracttensor<TLhs>::value, l_scalar_type, r_scalar_type>::type;
};

template <class TLhs, class TRhs>
struct tensor_type_finder {
    using type = typename std::conditional<is_abstracttensor<TLhs>::value, TLhs, TRhs>::type;
};

}

#endif // TENSOR_POST_META_H
