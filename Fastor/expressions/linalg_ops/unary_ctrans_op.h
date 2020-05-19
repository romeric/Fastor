#ifndef UNARY_CTRANS_OP_H
#define UNARY_CTRANS_OP_H

#include "Fastor/meta/meta.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/Aliasing.h"
#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/Ranges.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/expressions/expression_traits.h"


namespace Fastor {

template<typename Expr, size_t DIM0>
struct UnaryCTransOp: public AbstractTensor<UnaryCTransOp<Expr, DIM0>,DIM0> {
    using expr_type = expression_t<Expr>;
    static constexpr FASTOR_INDEX M = get_tensor_dimension_v<0,typename Expr::result_type>;
    static constexpr FASTOR_INDEX N = get_tensor_dimension_v<1,typename Expr::result_type>;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    using scalar_type = typename scalar_type_finder<UnaryCTransOp<Expr, DIM0>>::type;
    using simd_vector_type = typename Expr::simd_vector_type;
    using simd_abi_type = typename simd_vector_type::abi_type;
    using result_type = Tensor<scalar_type,N,M>;

    FASTOR_INLINE UnaryCTransOp(expr_type inexpr) : _expr(inexpr) {
    }

    FASTOR_INLINE FASTOR_INDEX size() const {return M*N;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return i == 0 ? N : M;}

    constexpr FASTOR_INLINE expr_type expr() const {return _expr;}

private:
    expr_type _expr;
};

/* Tensor conjugate transpose returning a tensor expression */
template<typename Expr, size_t DIM0>
FASTOR_INLINE UnaryCTransOp<Expr, DIM0>
ctrans(const AbstractTensor<Expr,DIM0> &src) {
    return UnaryCTransOp<Expr, DIM0>(src.self());
}

/* Backend implementation */
template<typename T, size_t M, size_t N>
FASTOR_INLINE void _ctranspose(const T * FASTOR_RESTRICT a, T * FASTOR_RESTRICT out) {
    for (size_t j=0; j<N; ++j)
        for (size_t i=0; i< M; ++i)
            out[j*M+i] = conj(a[i*N+j]);
}

/* Tensor conjugate transpose immediately returning a tensor */
template<typename T, size_t I, size_t J>
FASTOR_INLINE Tensor<T,J,I> ctranspose(const Tensor<T,I,J> &a) {
    Tensor<T,J,I> out;
    _ctranspose<T,I,J>(a.data(),out.data());
    return out;
}

/* Tensor conjugate transpose for higher order tensors immediately returning a tensor */
template<typename T, size_t ... Rest, typename std::enable_if<sizeof...(Rest)>=3,bool>::type=0>
FASTOR_INLINE Tensor<T,Rest...>
ctranspose(const Tensor<T,Rest...> &a) {

    constexpr size_t remaining_product = last_matrix_extracter<Tensor<T,Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Rest)-2>::type>::remaining_product;

    constexpr size_t I = get_value<sizeof...(Rest)-1,Rest...>::value;
    constexpr size_t J = get_value<sizeof...(Rest),Rest...>::value;
    static_assert(I==J,"THE LAST TWO DIMENSIONS OF TENSOR MUST BE THE SAME");

    Tensor<T,Rest...> out;
    T *a_data = a.data();
    T *out_data = out.data();

    for (size_t i=0; i<remaining_product; ++i) {
        _ctranspose<T,J,J>(a_data+i*J*J,out_data+i*J*J);
    }

    return out;
}

// Conjagate transpose for generic expressions
template<typename Expr, size_t DIM0>
FASTOR_INLINE
Tensor<
    typename scalar_type_finder<Expr>::type,
    get_tensor_dimension_v<1,typename Expr::result_type>,
    get_tensor_dimension_v<0,typename Expr::result_type>>
ctranspose(const AbstractTensor<Expr,DIM0> &src) {
    // If we are here Expr is already an expression
    using result_type = typename Expr::result_type;
    const result_type tmp(src.self());
    return ctranspose(tmp);
}




// assignments
template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, const UnaryCTransOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    const result_type& tmp = evaluate(src.expr().self());
    using T = typename UnaryCTransOp<Expr, OtherDIM>::scalar_type;
    static constexpr size_t M = UnaryCTransOp<Expr, OtherDIM>::M;
    static constexpr size_t N = UnaryCTransOp<Expr, OtherDIM>::N;
    _ctranspose<T,M,N>(tmp.data(),dst.self().data());
}

template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, const UnaryCTransOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    // no copies if expr is a tensor
    const result_type& tmp = evaluate(src.expr().self());
    // one copy for the inverse
    using T = typename UnaryCTransOp<Expr, OtherDIM>::scalar_type;
    using result_t = typename UnaryCTransOp<Expr, OtherDIM>::result_type;
    result_t tmp_trans;
    static constexpr size_t M = UnaryCTransOp<Expr, OtherDIM>::M;
    static constexpr size_t N = UnaryCTransOp<Expr, OtherDIM>::N;
    _ctranspose<T,M,N>(tmp.data(),tmp_trans.data());
    trivial_assign_add(dst.self(),tmp_trans);
}

template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, const UnaryCTransOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    // no copies if expr is a tensor
    const result_type& tmp = evaluate(src.expr().self());
    // one copy for the inverse
    using T = typename UnaryCTransOp<Expr, OtherDIM>::scalar_type;
    using result_t = typename UnaryCTransOp<Expr, OtherDIM>::result_type;
    result_t tmp_trans;
    static constexpr size_t M = UnaryCTransOp<Expr, OtherDIM>::M;
    static constexpr size_t N = UnaryCTransOp<Expr, OtherDIM>::N;
    _ctranspose<T,M,N>(tmp.data(),tmp_trans.data());
    trivial_assign_sub(dst.self(),tmp_trans);
}

template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign_mul(AbstractTensor<Derived,DIM> &dst, const UnaryCTransOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    // no copies if expr is a tensor
    const result_type& tmp = evaluate(src.expr().self());
    // one copy for the inverse
    using T = typename UnaryCTransOp<Expr, OtherDIM>::scalar_type;
    using result_t = typename UnaryCTransOp<Expr, OtherDIM>::result_type;
    result_t tmp_trans;
    static constexpr size_t M = UnaryCTransOp<Expr, OtherDIM>::M;
    static constexpr size_t N = UnaryCTransOp<Expr, OtherDIM>::N;
    _ctranspose<T,M,N>(tmp.data(),tmp_trans.data());
    trivial_assign_mul(dst.self(),tmp_trans);
}

template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign_div(AbstractTensor<Derived,DIM> &dst, const UnaryCTransOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    // no copies if expr is a tensor
    const result_type& tmp = evaluate(src.expr().self());
    // one copy for the inverse
    using T = typename UnaryCTransOp<Expr, OtherDIM>::scalar_type;
    using result_t = typename UnaryCTransOp<Expr, OtherDIM>::result_type;
    result_t tmp_trans;
    static constexpr size_t M = UnaryCTransOp<Expr, OtherDIM>::M;
    static constexpr size_t N = UnaryCTransOp<Expr, OtherDIM>::N;
    _ctranspose<T,M,N>(tmp.data(),tmp_trans.data());
    trivial_assign_div(dst.self(),tmp_trans);
}


} // end of namespace Fastor


#endif // UNARY_CTRANS_OP_H
