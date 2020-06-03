#ifndef UNARY_MATH_OP_H
#define UNARY_MATH_OP_H


#include "Fastor/simd_vector/SIMDVector.h"
#include "Fastor/simd_math/simd_math.h"
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
    using scalar_type = typename Expr::scalar_type;\
    using simd_vector_type = typename Expr::simd_vector_type;\
    using simd_abi_type = typename simd_vector_type::abi_type;\
    using result_type = typename Expr::result_type;\
    static constexpr FASTOR_INDEX Dimension = DIM0;\
    static constexpr FASTOR_INDEX rank() {return DIM0;}\
    FASTOR_INLINE FASTOR_INDEX size() const {return _expr.size();}\
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return _expr.dimension(i);}\
    Unary ##STRUCT_NAME ## Op(expression_t<Expr> inexpr) : _expr(inexpr) {}\
    FASTOR_INLINE expression_t<Expr> expr() const {return _expr;}\
    template<typename U=scalar_type>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> eval(FASTOR_INDEX i) const {\
        return SIMD_OP(_expr.template eval<EVAL_TYPE>(i));\
    }\
    template<typename U=scalar_type>\
    FASTOR_INLINE EVAL_TYPE eval_s(FASTOR_INDEX i) const {\
        return SCALAR_OP(_expr.template eval_s<EVAL_TYPE>(i));\
    }\
    template<typename U=scalar_type>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return SIMD_OP(_expr.template eval<EVAL_TYPE>(i,j));\
    }\
    template<typename U=scalar_type>\
    FASTOR_INLINE EVAL_TYPE eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return SCALAR_OP(_expr.template eval_s<EVAL_TYPE>(i,j));\
    }\
    template<typename U>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> teval(const std::array<int,DIM0> &as) const {\
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
FASTOR_MAKE_UNARY_MATH_OPS(cbrt, cbrt, std::cbrt, Cbrt, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(exp, exp, std::exp, Exp, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(exp2, exp2, std::exp2, Exp2, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(expm1, expm1, std::expm1, Expm1, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(log, log, std::log, Log, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(log10, log10, std::log10, Log10, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(log2, log2, std::log2, Log2, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(log1p, log1p, std::log1p, Log1p, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(sin, sin, std::sin, Sin, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(cos, cos, std::cos, Cos, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(tan, tan, std::tan, Tan, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(asin, asin, std::asin, Asin, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(acos, acos, std::acos, Acos, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(atan, atan, std::atan, Atan, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(sinh, sinh, std::sinh, Sinh, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(cosh, cosh, std::cosh, Cosh, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(tanh, tanh, std::tanh, Tanh, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(asinh, asinh, std::asinh, Asinh, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(acosh, acosh, std::acosh, Acosh, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(atanh, atanh, std::atanh, Atanh, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(erf, erf, std::erf, Erf, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(tgamma, tgamma, std::tgamma, Tgamma, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(lgamma, lgamma, std::lgamma, Lgamma, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(ceil, ceil, std::ceil, Ceil, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(round, round, std::round, Round, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(floor, floor, std::floor, Floor, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(trunc, trunc, std::trunc, Trunc, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(conj, conj, std::conj, Conj, scalar_type)
FASTOR_MAKE_UNARY_MATH_OPS(arg , arg , std::arg , Arg , scalar_type)




#define FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(OP, NAME, ASSIGN_TYPE)\
template<typename Derived, size_t DIM, typename OtherDerived, size_t OtherDIM,\
    typename std::enable_if<requires_evaluation_v<OtherDerived>,bool>::type = false>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, const Unary ##NAME ## Op<OtherDerived,OtherDIM> &src) {\
    assign ##ASSIGN_TYPE (dst.self(), src.expr().self());\
    trivial_assign(dst.self(), OP(dst.self()));\
}\
template<typename Derived, size_t DIM, typename OtherDerived, size_t OtherDIM,\
    typename std::enable_if<!requires_evaluation_v<OtherDerived>,bool>::type = false>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, const Unary ##NAME ## Op<OtherDerived,OtherDIM> &src) {\
    trivial_assign ##ASSIGN_TYPE (dst.self(), src.self());\
}\

// only assignment
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT( ,    Add,  )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(-,    Sub,  )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(abs,  Abs,  )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(sqrt, Sqrt, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(cbrt, Cbrt, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(exp,  Exp,  )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(exp2,  Exp2,  )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(expm1,  Expm1,  )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(log,  Log,  )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(log10,  Log10,  )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(log2,  Log2,  )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(log1p,  Log1p,  )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(sin,  Sin,  )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(cos,  Cos,  )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(tan,  Tan,  )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(asin, Asin, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(acos, Acos, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(atan, Atan, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(sinh, Sinh, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(cosh, Cosh, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(tanh, Tanh, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(asinh, Asinh, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(acosh, Acosh, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(atanh, Atanh, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(erf, Erf, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(tgamma, Tgamma, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(lgamma, Lgamma, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(ceil, Ceil, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(round, Round, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(floor, Floor, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(trunc, Trunc, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(conj, Conj, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENT(arg , Arg , )



// arithmetic assignments
#define FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(OP, NAME, ASSIGN_TYPE)\
template<typename Derived, size_t DIM, typename OtherDerived, size_t OtherDIM,\
    typename std::enable_if<requires_evaluation_v<OtherDerived>,bool>::type = false>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, const Unary ##NAME ## Op<OtherDerived,OtherDIM> &src) {\
    using result_type = typename Unary ##NAME ## Op<OtherDerived,OtherDIM>::result_type;\
    const result_type tmp(src.expr().self());\
    trivial_assign ##ASSIGN_TYPE (dst.self(), OP(tmp));\
}\
template<typename Derived, size_t DIM, typename OtherDerived, size_t OtherDIM,\
    typename std::enable_if<!requires_evaluation_v<OtherDerived>,bool>::type = false>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, const Unary ##NAME ## Op<OtherDerived,OtherDIM> &src) {\
    trivial_assign ##ASSIGN_TYPE (dst.self(), src.self());\
}\

#define FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENTS(OP, NAME, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT( ,    Add,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(-,    Sub,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(abs,  Abs,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(sqrt, Sqrt, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(cbrt, Cbrt, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(exp,  Exp,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(exp2,  Exp2,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(expm1,  Expm1,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(log,  Log,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(log10,  Log10,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(log2,  Log2,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(log1p,  Log1p,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(sin,  Sin,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(cos,  Cos,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(tan,  Tan,  ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(asin, Asin, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(acos, Acos, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(atan, Atan, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(sinh, Sinh, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(cosh, Cosh, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(tanh, Tanh, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(asinh, Asinh, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(acosh, Acosh, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(atanh, Atanh, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(erf, Erf, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(tgamma, Tgamma, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(lgamma, Lgamma, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(ceil, Ceil, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(round, Round, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(floor, Floor, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(trunc, Trunc, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(conj, Conj, ASSIGN_TYPE)\
FASTOR_MAKE_UNARY_MATH_OP_ARITHMETIC_ASSIGNMENT(arg , Arg , ASSIGN_TYPE)\

// FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENTS(OP, NAME, )
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENTS(OP, NAME, _add)
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENTS(OP, NAME, _sub)
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENTS(OP, NAME, _mul)
FASTOR_MAKE_UNARY_MATH_OP_ASSIGNMENTS(OP, NAME, _div)

} // end of namespace Fastor


#endif // UNARY_MATH_OP_H

