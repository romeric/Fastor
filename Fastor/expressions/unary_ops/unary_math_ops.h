#ifndef UNARY_MATH_OP_H
#define UNARY_MATH_OP_H


#include "Fastor/tensor/Tensor.h"
#include "Fastor/meta/tensor_post_meta.h"

namespace Fastor {

// All unary math ops
#define FASTOR_MAKE_UNARY_MATH_OPS(OP_NAME, SIMD_OP, SCALAR_OP, STRUCT_NAME, EVAL_TYPE)\
template<typename Expr, size_t DIM0>\
struct Unary ##STRUCT_NAME ## Op: public AbstractTensor<Unary ##STRUCT_NAME ## Op<Expr, DIM0>,DIM0> {\
private:\
    expression_t<Expr> _expr;\
public:\
    using scalar_type = typename scalar_type_finder<Expr>::type;\
    static constexpr FASTOR_INDEX Dimension = DIM0;\
    static constexpr FASTOR_INDEX rank() {return DIM0;}\
    FASTOR_INLINE FASTOR_INDEX size() const {return _expr.size();}\
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return _expr.dimension(i);}\
    Unary ##STRUCT_NAME ## Op(expression_t<Expr> inexpr) : _expr(inexpr) {}\
    FASTOR_INLINE FASTOR_INDEX expr() const {return _expr;}\
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

}


#endif // UNARY_MATH_OP_H

