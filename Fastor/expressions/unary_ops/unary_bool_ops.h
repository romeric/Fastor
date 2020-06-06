#ifndef UNARY_BOOL_OP_H
#define UNARY_BOOL_OP_H


#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/simd_math/simd_math.h"
#include "Fastor/tensor/Tensor.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"
#include "Fastor/expressions/expression_traits.h"

namespace Fastor {

// All unary bool ops
#define FASTOR_MAKE_UNARY_BOOL_OPS(OP_NAME, SIMD_OP, SCALAR_OP, STRUCT_NAME, EVAL_TYPE)\
template<typename Expr, size_t DIM0>\
struct Unary ##STRUCT_NAME ## Op: public AbstractTensor<Unary ##STRUCT_NAME ## Op<Expr, DIM0>,DIM0> {\
private:\
    expression_t<Expr> _expr;\
public:\
    using scalar_type = typename Expr::scalar_type;\
    using simd_vector_type = typename Expr::simd_vector_type;\
    using simd_abi_type = typename simd_vector_type::abi_type;\
    using result_type = to_bool_tensor_t<typename Expr::result_type>;\
    using result_scalar_type = typename result_type::scalar_type;\
    using result_simd_abi_type = typename result_type::simd_abi_type;\
    using result_simd_vector_type = typename result_type::simd_vector_type;\
    using UU = bool /*this needs to change to U once masks are implemented*/;\
    using ABI = simd_abi::fixed_size<SIMDVector<EVAL_TYPE,simd_abi_type>::Size>;\
    static constexpr FASTOR_INDEX Dimension = DIM0;\
    static constexpr FASTOR_INDEX rank() {return DIM0;}\
    FASTOR_INLINE FASTOR_INDEX size() const {return _expr.size();}\
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return _expr.dimension(i);}\
    Unary ##STRUCT_NAME ## Op(expression_t<Expr> inexpr) : _expr(inexpr) {}\
    FASTOR_INLINE expression_t<Expr> expr() const {return _expr;}\
    template<typename U=scalar_type>\
    FASTOR_INLINE SIMDVector<UU,ABI> eval(FASTOR_INDEX i) const {\
        return SIMD_OP(_expr.template eval<EVAL_TYPE>(i));\
    }\
    template<typename U=scalar_type>\
    FASTOR_INLINE UU eval_s(FASTOR_INDEX i) const {\
        return SCALAR_OP(_expr.template eval_s<EVAL_TYPE>(i));\
    }\
    template<typename U=scalar_type>\
    FASTOR_INLINE SIMDVector<UU,ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return SIMD_OP(_expr.template eval<EVAL_TYPE>(i,j));\
    }\
    template<typename U=scalar_type>\
    FASTOR_INLINE UU eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return SCALAR_OP(_expr.template eval_s<EVAL_TYPE>(i,j));\
    }\
    template<typename U>\
    FASTOR_INLINE SIMDVector<UU,ABI> teval(const std::array<int,DIM0> &as) const {\
        return SIMD_OP(_expr.template teval<EVAL_TYPE>(as));\
    }\
    template<typename U>\
    FASTOR_INLINE UU teval_s(const std::array<int,DIM0> &as) const {\
        return SCALAR_OP(_expr.template teval_s<EVAL_TYPE>(as));\
    }\
};\
template<typename Expr, size_t DIM0,\
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >\
FASTOR_INLINE Unary ##STRUCT_NAME ## Op<Expr, DIM0> OP_NAME(const AbstractTensor<Expr,DIM0> &_expr) {\
  return Unary ##STRUCT_NAME ## Op<Expr, DIM0>(_expr.self());\
}\

FASTOR_MAKE_UNARY_BOOL_OPS(operator!, !      , !            , Not     , scalar_type)
FASTOR_MAKE_UNARY_BOOL_OPS(isinf   , isinf   , std::isinf   , Isinf   , scalar_type)
FASTOR_MAKE_UNARY_BOOL_OPS(isnan   , isnan   , std::isnan   , Isnan   , scalar_type)
FASTOR_MAKE_UNARY_BOOL_OPS(isfinite, isfinite, std::isfinite, Isfinite, scalar_type)




#define FASTOR_MAKE_UNARY_BOOL_OP_ASSIGNMENT(OP, NAME, ASSIGN_TYPE)\
template<typename Derived, size_t DIM, typename OtherDerived, size_t OtherDIM,\
    typename std::enable_if<requires_evaluation_v<OtherDerived>,bool>::type = false>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, const Unary ##NAME ## Op<OtherDerived,OtherDIM> &src) {\
    using result_type = typename OtherDerived::result_type;\
    const result_type tmp(src.expr().self());\
    trivial_assign ##ASSIGN_TYPE (dst.self(), OP(tmp));\
}\
template<typename Derived, size_t DIM, typename OtherDerived, size_t OtherDIM,\
    typename std::enable_if<!requires_evaluation_v<OtherDerived>,bool>::type = false>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, const Unary ##NAME ## Op<OtherDerived,OtherDIM> &src) {\
    trivial_assign ##ASSIGN_TYPE (dst.self(), src.self());\
}\

// only assignment
FASTOR_MAKE_UNARY_BOOL_OP_ASSIGNMENT(!       ,  Not     ,  )
FASTOR_MAKE_UNARY_BOOL_OP_ASSIGNMENT(isinf   ,  Isinf   ,  )
FASTOR_MAKE_UNARY_BOOL_OP_ASSIGNMENT(isnan   ,  Isnan   ,  )
FASTOR_MAKE_UNARY_BOOL_OP_ASSIGNMENT(isfinite,  Isfinite,  )




} // end of namespace Fastor


#endif // UNARY_BOOL_OP_H

