#ifndef UNARY_MATH_OP_H
#define UNARY_MATH_OP_H


#include "Fastor/tensor/Tensor.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"
#include "Fastor/expressions/expression_traits.h"

namespace Fastor {

// All unary math ops
#define FASTOR_MAKE_UNARY_MATH_OPS(OP_NAME, SIMD_OP, SCALAR_OP, STRUCT_NAME, EVAL_TYPE)\
template<typename Expr, size_t DIM0>\
struct Unary ##STRUCT_NAME ## Op: public AbstractTensor<Unary ##STRUCT_NAME ## Op<Expr, DIM0>,DIM0> {\
private:\
    expression_t<Expr> _expr;\
public:\
    using scalar_type = typename scalar_type_finder<Expr>::type;\
    using result_type = typename Expr::result_type;\
    static constexpr FASTOR_INDEX Dimension = DIM0;\
    static constexpr FASTOR_INDEX rank() {return DIM0;}\
    FASTOR_INLINE FASTOR_INDEX size() const {return _expr.size();}\
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return _expr.dimension(i);}\
    Unary ##STRUCT_NAME ## Op(expression_t<Expr> inexpr) : _expr(inexpr) {}\
    FASTOR_INLINE expression_t<Expr> expr() const {return _expr;}\
    template<typename U=scalar_type>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,DEFAULT_ABI> eval(FASTOR_INDEX i) const {\
        return SIMD_OP(_expr.template eval<EVAL_TYPE>(i));\
    }\
    template<typename U=scalar_type>\
    FASTOR_INLINE EVAL_TYPE eval_s(FASTOR_INDEX i) const {\
        return SCALAR_OP(_expr.template eval_s<EVAL_TYPE>(i));\
    }\
    template<typename U=scalar_type>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return SIMD_OP(_expr.template eval<EVAL_TYPE>(i,j));\
    }\
    template<typename U=scalar_type>\
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return SCALAR_OP(_expr.template eval_s<EVAL_TYPE>(i,j));\
    }\
    template<typename U>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,DEFAULT_ABI> teval(const std::array<int,DIM0> &as) const {\
        return SIMD_OP(_expr.template teval<EVAL_TYPE>(as));\
    }\
    template<typename U>\
    FASTOR_INLINE EVAL_TYPE teval_s(const std::array<int,DIM0> &as) const {\
        return SCALAR_OP(_expr.template teval_s<EVAL_TYPE>(as));\
    }\
};\
template<typename Expr, size_t DIM0,\
         typename std::enable_if<!std::is_arithmetic<Expr>::value,bool>::type = 0 >\
FASTOR_INLINE Unary ##STRUCT_NAME ## Op<Expr, DIM0> OP_NAME(const AbstractTensor<Expr,DIM0> &_expr) {\
  return Unary ##STRUCT_NAME ## Op<Expr, DIM0>(_expr.self());\
}\


FASTOR_MAKE_UNARY_MATH_OPS(operator+, , , Add, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(operator-, -, -, Sub, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(abs, abs, std::abs, Abs, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(sqrt, sqrt, sqrts, Sqrt, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(exp, exp, std::exp, Exp, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(log, log, std::log, Log, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(sin, sin, std::sin, Sin, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(cos, cos, std::cos, Cos, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(tan, tan, std::tan, Tan, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(asin, asin, std::asin, Asin, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(acos, acos, std::acos, Acos, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(atan, atan, std::atan, Atan, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(sinh, sinh, std::sinh, Sinh, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(cosh, cosh, std::cosh, Cosh, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(tanh, tanh, std::tanh, Tanh, scalar_type)





#define FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(OP, NAME, ASSIGN_TYPE)\
template<typename Derived, size_t DIM, typename OtherDerived, size_t OtherDIM,\
    typename std::enable_if<requires_evaluation_v<OtherDerived>,bool>::type = false>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (const AbstractTensor<Derived,DIM> &dst, const Unary ##NAME ## Op<OtherDerived,OtherDIM> &src) {\
    assign ##ASSIGN_TYPE (dst.self(), src.expr().self());\
    trivial_assign ##ASSIGN_TYPE (dst.self(), OP(dst.self()));\
}\
template<typename Derived, size_t DIM, typename OtherDerived, size_t OtherDIM,\
    typename std::enable_if<!requires_evaluation_v<OtherDerived>,bool>::type = false>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (const AbstractTensor<Derived,DIM> &dst, const Unary ##NAME ## Op<OtherDerived,OtherDIM> &src) {\
    trivial_assign ##ASSIGN_TYPE (dst.self(), OP(src.expr().self()));\
}\

#define FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENTS(OP, NAME, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT( ,    Add,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(-,    Sub,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(abs,  Abs,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(sqrt, Sqrt, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(exp,  Exp,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(log,  Log,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(sin,  Sin,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(cos,  Cos,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(tan,  Tan,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(asin, Asin, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(acos, Acos, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(atan, Atan, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(sinh, Sinh, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(cosh, Cosh, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(tanh, Tanh, ASSIGN_TYPE)\

FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENTS(OP, NAME, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENTS(OP, NAME, _add)
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENTS(OP, NAME, _sub)
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENTS(OP, NAME, _mul)
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENTS(OP, NAME, _div)


}


#endif // UNARY_MATH_OP_H

