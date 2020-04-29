#ifndef UNARY_COF_OP_H
#define UNARY_COF_OP_H

#include "Fastor/backend/cofactor.h"
#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/Aliasing.h"
#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/Ranges.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/expressions/expression_traits.h"


namespace Fastor {

template<typename Expr, size_t DIM0>
struct UnaryCofOp: public AbstractTensor<UnaryCofOp<Expr, DIM0>,DIM0> {
    using expr_type = expression_t<Expr>;
    using result_type = typename Expr::result_type;
    static constexpr FASTOR_INDEX M = get_tensor_dimension_v<0,result_type>;
    static constexpr FASTOR_INDEX N = get_tensor_dimension_v<1,result_type>;
    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    using scalar_type = typename scalar_type_finder<UnaryCofOp<Expr, DIM0>>::type;

    FASTOR_INLINE UnaryCofOp(expr_type inexpr) : _expr(inexpr) {
        static_assert(M==N, "MATRIX MUST BE SQUARE");
    }

    FASTOR_INLINE FASTOR_INDEX size() const {return M*N;}
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX ) const {return M;}

    constexpr FASTOR_INLINE expr_type expr() const {return _expr;}

private:
    expr_type _expr;
};

template<typename Expr, size_t DIM0>
FASTOR_INLINE UnaryCofOp<Expr, DIM0>
cof(const AbstractTensor<Expr,DIM0> &src) {
  return UnaryCofOp<Expr, DIM0>(src.self());
}


namespace internal {
template<typename T, size_t M, enable_if_t_<is_greater_v_<M,0> && is_less_equal_v_<M,4>,bool> = false>
FASTOR_INLINE void cofactor_dispatcher(const Tensor<T,M,M> &in, Tensor<T,M,M>& out) {
    _cofactor<T,M>(in.data(),out.data());
}
} // internal

// Cofactor for generic expressions is provided here
template<typename Derived, size_t DIM>
FASTOR_INLINE
typename Derived::result_type
cofactor(const AbstractTensor<Derived,DIM> &src) {
    // If we are here Derived is already an expression
    using result_type = typename Derived::result_type;
    const result_type tmp(src.self());
    return cofactor(tmp);
}






// assignments
template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign(AbstractTensor<Derived,DIM> &dst, const UnaryCofOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    const result_type& tmp = evaluate(src.expr().self());
    internal::cofactor_dispatcher(tmp,dst.self());
}

template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign_add(AbstractTensor<Derived,DIM> &dst, const UnaryCofOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    // no copies if expr is a tensor
    const result_type& tmp = evaluate(src.expr().self());
    // one copy for the inverse
    result_type tmp_inv;
    internal::cofactor_dispatcher(tmp,tmp_inv);
    trivial_assign_add(dst.self(),tmp_inv);
}

template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign_sub(AbstractTensor<Derived,DIM> &dst, const UnaryCofOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    // no copies if expr is a tensor
    const result_type& tmp = evaluate(src.expr().self());
    // one copy for the inverse
    result_type tmp_inv;
    internal::cofactor_dispatcher(tmp,tmp_inv);
    trivial_assign_sub(dst.self(),tmp_inv);
}

template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign_mul(AbstractTensor<Derived,DIM> &dst, const UnaryCofOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    // no copies if expr is a tensor
    const result_type& tmp = evaluate(src.expr().self());
    // one copy for the inverse
    result_type tmp_inv;
    internal::cofactor_dispatcher(tmp,tmp_inv);
    trivial_assign_mul(dst.self(),tmp_inv);
}

template<typename Derived, size_t DIM, typename Expr, size_t OtherDIM>
FASTOR_INLINE void assign_div(AbstractTensor<Derived,DIM> &dst, const UnaryCofOp<Expr, OtherDIM> &src) {
    using result_type = typename Expr::result_type;
    // no copies if expr is a tensor
    const result_type& tmp = evaluate(src.expr().self());
    // one copy for the inverse
    result_type tmp_inv;
    internal::cofactor_dispatcher(tmp,tmp_inv);
    trivial_assign_div(dst.self(),tmp_inv);
}


} // end of namespace Fastor


#endif // UNARY_COF_OP_H