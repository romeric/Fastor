#ifndef INNERPRODUCT_H
#define INNERPRODUCT_H

#include "Fastor/backend/doublecontract.h"
#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/TensorTraits.h"

namespace Fastor {


// Inner products - reduction to a scalar
//----------------------------------------------------------------------------------------------
template<typename T, size_t ... Rest>
T inner(const Tensor<T,Rest...> &a) {
    //! Reduces a multi-dimensional tensor to a scalar
    //!
    //! If a is scalar/Tensor<T> returns the value itself
    //! If a is a vector Tensor<T,N> returns the sum of values
    //! If a is a second order tensor Tensor<T,N,N> returns the trace
    //! If a is a third order tensor Tensor<T,N,N,N> returns a_iii
    //! ...
    //!
    //! The size of the tensor in all dimensions should be equal (uniform)

    static_assert(no_of_unique<Rest...>::value<=1, "REDUCTION IS ONLY POSSIBLE ON UNIFORM TENSORS");
    constexpr int ndim = sizeof...(Rest);

    T *a_data = a.data();
    if (ndim==0) {
        return a_data[0];
    }
    else if (ndim==1) {
        return a.sum();
    }
    else {
        constexpr std::array<size_t,ndim> products = nprods<Index<Rest...>,
                typename std_ext::make_index_sequence<ndim>::type>::values;

        T reductor = static_cast<T>(0);
        for (size_t i=0; i<a.dimension(0); ++i) {
            size_t index_a = i;
            for(size_t it = 0; it< ndim; it++) {
                index_a += products[it]*i;
            }
            reductor += a_data[index_a];
        }
        return reductor;
    }
}

template<typename T, size_t ... Rest>
FASTOR_INLINE T inner(const Tensor<T,Rest...> &a, const Tensor<T,Rest...> &b) {
    //! Reduction of a tensor pair to a scalar, for instance A_ijklm * B_ijklm
    //! If a and b are scalars/vectors, returns dot product
    //! If a and b are matrices, returns double contraction
    //! For third order tensors returns a_ijk*b_ijk
    //! ...

    const T *a_data = a.data();
    const T *b_data = b.data();

    constexpr size_t ndim = sizeof...(Rest);
    FASTOR_IF_CONSTEXPR (ndim>0) {
        return _doublecontract<T,pack_prod<Rest...>::value,1>(a_data,b_data);
    }
    else {
        return (*a_data)*(*b_data);
    }
}


// Expressions
//---------------------------------------------------------------------------------------------------
template<typename Derived0, size_t DIM0>
FASTOR_INLINE
typename Derived0::scalar_type
inner(const AbstractTensor<Derived0,DIM0> &a) {
    using result_type = typename Derived0::result_type;
    return inner(result_type(a));
}

template<typename Derived0, size_t DIM0, typename Derived1, size_t DIM1,
    enable_if_t_<!is_tensor_v<Derived0> && !is_tensor_v<Derived1>,bool> = false >
FASTOR_INLINE
typename Derived0::scalar_type
inner(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b) {
    using lhs_type = typename Derived0::result_type;
    using rhs_type = typename Derived1::result_type;
    return inner(lhs_type(a),rhs_type(b));
}
template<typename Derived0, size_t DIM0, typename Derived1, size_t DIM1,
    enable_if_t_<!is_tensor_v<Derived0> && is_tensor_v<Derived1>,bool> = false >
FASTOR_INLINE
typename Derived0::scalar_type
inner(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b) {
    using lhs_type = typename Derived0::result_type;
    return inner(lhs_type(a),b.self());
}
template<typename Derived0, size_t DIM0, typename Derived1, size_t DIM1,
    enable_if_t_<is_tensor_v<Derived0> && !is_tensor_v<Derived1>,bool> = false >
FASTOR_INLINE
typename Derived0::scalar_type
inner(const AbstractTensor<Derived0,DIM0> &a, const AbstractTensor<Derived1,DIM1> &b) {
    using rhs_type = typename Derived1::result_type;
    return inner(a.self(),rhs_type(b));
}



// multiple chained expressions - network innerproduct
//---------------------------------------------------------------------------------------------------
#if FASTOR_CXX_VERSION >= 2014
namespace internal {
template<typename AbstractTensorType0>
FASTOR_INLINE
auto
innerproduct_chain_expression(const AbstractTensorType0& a)
-> decltype(a.self())
{
    return a.self();
}
template<typename AbstractTensorType0, typename AbstractTensorType1, typename ... AbstractTensorTypes>
FASTOR_INLINE
auto
innerproduct_chain_expression(const AbstractTensorType0& a, const AbstractTensorType1& b, const AbstractTensorTypes& ... rest)
// -> decltype(innerproduct_chain_expression(evaluate(a*b),rest...))
{
    const auto src = evaluate(a*b);
    return innerproduct_chain_expression(src,rest...);
}
} // internal

template<typename AbstractTensorType0, typename AbstractTensorType1, typename ... AbstractTensorTypes,
    enable_if_t_<sizeof...(AbstractTensorTypes) >= 1,bool> = false>
FASTOR_INLINE
auto
inner(const AbstractTensorType0& a, const AbstractTensorType1& b, const AbstractTensorTypes& ... rest)
-> decltype(internal::innerproduct_chain_expression(a,b,rest...).sum())
{
    return internal::innerproduct_chain_expression(a,b,rest...).sum();
}
#endif
//---------------------------------------------------------------------------------------------------


}

#endif // INNERPRODUCT_H

