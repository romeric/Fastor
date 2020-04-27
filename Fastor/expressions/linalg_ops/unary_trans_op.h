#ifndef UNARY_TRANS_OP_H
#define UNARY_TRANS_OP_H

#include "Fastor/backend/transpose/transpose.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/Aliasing.h"
#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/Ranges.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/expressions/expression_traits.h"


namespace Fastor {

template<typename Expr, size_t DIM0>
struct UnaryTransOp: public AbstractTensor<UnaryTransOp<Expr, DIM0>,DIM0> {
    using expr_type = expression_t<Expr>;
    static constexpr FASTOR_INDEX M = get_tensor_dimensions<typename Expr::result_type>::dims[0];
    static constexpr FASTOR_INDEX N = get_tensor_dimensions<typename Expr::result_type>::dims[1];
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    using scalar_type = typename scalar_type_finder<UnaryTransOp<Expr, DIM0>>::type;
    using result_type = Tensor<scalar_type,N,M>;

    FASTOR_INLINE UnaryTransOp(expr_type inexpr) : _expr(inexpr) {
    }

    FASTOR_INLINE FASTOR_INDEX size() const {return M*N;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return i == 0 ? N : M;}

    constexpr FASTOR_INLINE expr_type expr() const {return _expr;}

private:
    expr_type _expr;
};

template<typename Expr, size_t DIM0>
FASTOR_INLINE UnaryTransOp<Expr, DIM0>
trans(const AbstractTensor<Expr,DIM0> &src) {
  return UnaryTransOp<Expr, DIM0>(src.self());
}


// assignments
template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, const UnaryTransOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    const result_type& tmp = evaluate(src.expr().self());
    using T = typename UnaryTransOp<Expr, OtherDIM>::scalar_type;
    static constexpr size_t M = UnaryTransOp<Expr, OtherDIM>::M;
    static constexpr size_t N = UnaryTransOp<Expr, OtherDIM>::N;
    _transpose<T,M,N>(tmp.data(),dst.self().data());
}

template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, const UnaryTransOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    // no copies if expr is a tensor
    const result_type& tmp = evaluate(src.expr().self());
    // one copy for the inverse
    using T = typename UnaryTransOp<Expr, OtherDIM>::scalar_type;
    using result_t = typename UnaryTransOp<Expr, OtherDIM>::result_type;
    result_t tmp_trans;
    static constexpr size_t M = UnaryTransOp<Expr, OtherDIM>::M;
    static constexpr size_t N = UnaryTransOp<Expr, OtherDIM>::N;
    _transpose<T,M,N>(tmp.data(),tmp_trans.data());
    trivial_assign_add(dst.self(),tmp_trans);
}

template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, const UnaryTransOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    // no copies if expr is a tensor
    const result_type& tmp = evaluate(src.expr().self());
    // one copy for the inverse
    using T = typename UnaryTransOp<Expr, OtherDIM>::scalar_type;
    using result_t = typename UnaryTransOp<Expr, OtherDIM>::result_type;
    result_t tmp_trans;
    static constexpr size_t M = UnaryTransOp<Expr, OtherDIM>::M;
    static constexpr size_t N = UnaryTransOp<Expr, OtherDIM>::N;
    _transpose<T,M,N>(tmp.data(),tmp_trans.data());
    trivial_assign_sub(dst.self(),tmp_trans);
}

template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign_mul(AbstractTensor<Derived,DIM> &dst, const UnaryTransOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    // no copies if expr is a tensor
    const result_type& tmp = evaluate(src.expr().self());
    // one copy for the inverse
    using T = typename UnaryTransOp<Expr, OtherDIM>::scalar_type;
    using result_t = typename UnaryTransOp<Expr, OtherDIM>::result_type;
    result_t tmp_trans;
    static constexpr size_t M = UnaryTransOp<Expr, OtherDIM>::M;
    static constexpr size_t N = UnaryTransOp<Expr, OtherDIM>::N;
    _transpose<T,M,N>(tmp.data(),tmp_trans.data());
    trivial_assign_mul(dst.self(),tmp_trans);
}

template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign_div(AbstractTensor<Derived,DIM> &dst, const UnaryTransOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    // no copies if expr is a tensor
    const result_type& tmp = evaluate(src.expr().self());
    // one copy for the inverse
    using T = typename UnaryTransOp<Expr, OtherDIM>::scalar_type;
    using result_t = typename UnaryTransOp<Expr, OtherDIM>::result_type;
    result_t tmp_trans;
    static constexpr size_t M = UnaryTransOp<Expr, OtherDIM>::M;
    static constexpr size_t N = UnaryTransOp<Expr, OtherDIM>::N;
    _transpose<T,M,N>(tmp.data(),tmp_trans.data());
    trivial_assign_div(dst.self(),tmp_trans);
}


} // end of namespace Fastor


#endif // UNARY_TRANS_OP_H