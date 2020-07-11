#ifndef BINARY_MATH_OP_H
#define BINARY_MATH_OP_H

#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/expressions/expression_traits.h"


namespace Fastor {


#define FASTOR_MAKE_BINARY_MATH_OPS(OP_NAME, SIMD_OP, OP, NAME, EVAL_TYPE) \
template<typename TLhs, typename TRhs, size_t DIM0>\
struct Binary ##NAME ## Op: public AbstractTensor<Binary ##NAME ## Op<TLhs, TRhs, DIM0>,DIM0> {\
    expression_t<TLhs> _lhs;\
    expression_t<TRhs> _rhs;\
public:\
    static constexpr FASTOR_INDEX Dimension = DIM0;\
    static constexpr FASTOR_INDEX rank() {return DIM0;}\
    using scalar_type = typename scalar_type_finder<Binary ##NAME ## Op<TLhs, TRhs, DIM0>>::type;\
    using simd_vector_type = binary_op_simd_vector_t< Binary ##NAME ## Op<TLhs, TRhs, DIM0> >;\
    using simd_abi_type = typename simd_vector_type::abi_type;\
    using result_type = binary_arithmetic_result_t< Binary ##NAME ## Op<TLhs, TRhs, DIM0> >;\
    FASTOR_INLINE Binary ##NAME ## Op(expression_t<TLhs> inlhs, expression_t<TRhs> inrhs) : _lhs(inlhs), _rhs(inrhs) {}\
    FASTOR_INLINE FASTOR_INDEX size() const {return helper_size<TLhs,TRhs>();}\
    template<class LExpr, class RExpr,\
             typename std::enable_if<is_primitive_v_<LExpr>,bool>::type =0 >\
    FASTOR_INLINE FASTOR_INDEX helper_size() const {return _rhs.size();}\
    template<class LExpr, class RExpr,\
             typename std::enable_if<is_primitive_v_<RExpr>,bool>::type =0 >\
    FASTOR_INLINE FASTOR_INDEX helper_size() const {return _lhs.size();}\
    template<class LExpr, class RExpr,\
             typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                     !is_primitive_v_<RExpr>,bool>::type =0 >\
    FASTOR_INLINE FASTOR_INDEX helper_size() const {\
        FASTOR_ASSERT(_rhs.size()==_lhs.size(),"EXPRESSION SIZE MISMATCH");\
        return _rhs.size();\
    }\
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return helper_dimension<TLhs,TRhs>(i);}\
    template<class LExpr, class RExpr,\
             typename std::enable_if<is_primitive_v_<LExpr>,bool>::type =0 >\
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {return _rhs.dimension(i);}\
    template<class LExpr, class RExpr,\
             typename std::enable_if<is_primitive_v_<RExpr>,bool>::type =0 >\
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {return _lhs.dimension(i);}\
    template<class LExpr, class RExpr,\
             typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                     !is_primitive_v_<RExpr>,bool>::type =0 >\
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {\
        FASTOR_ASSERT(_rhs.dimension(i)==_lhs.dimension(i),"EXPRESSION SHAPE MISMATCH");\
        return _rhs.dimension(i);\
    }\
    FASTOR_INLINE expression_t<TLhs> lhs() const {return _lhs;}\
    FASTOR_INLINE expression_t<TRhs> rhs() const {return _rhs;}\
    template<typename U>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> eval(FASTOR_INDEX i) const {\
        return helper<TLhs,TRhs,U>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> helper(FASTOR_INDEX i) const {\
        return SIMD_OP(_lhs.template eval<EVAL_TYPE>(i), _rhs.template eval<EVAL_TYPE>(i));\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> helper(FASTOR_INDEX i) const {\
        return SIMD_OP((EVAL_TYPE)_lhs, _rhs.template eval<EVAL_TYPE>(i));\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> helper(FASTOR_INDEX i) const {\
        return SIMD_OP(_lhs.template eval<EVAL_TYPE>(i), (EVAL_TYPE)_rhs);\
    }\
    template<typename U>\
    FASTOR_INLINE EVAL_TYPE eval_s(FASTOR_INDEX i) const {\
        return helper_s<TLhs,TRhs,U>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE helper_s(FASTOR_INDEX i) const {\
        return OP(_lhs.template eval_s<EVAL_TYPE>(i), _rhs.template eval_s<EVAL_TYPE>(i));\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE helper_s(FASTOR_INDEX i) const {\
        return OP((EVAL_TYPE)_lhs, _rhs.template eval_s<EVAL_TYPE>(i));\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE helper_s(FASTOR_INDEX i) const {\
        return OP(_lhs.template eval_s<EVAL_TYPE>(i), (EVAL_TYPE)_rhs);\
    }\
    template<typename U>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return helper<TLhs,TRhs,U>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return SIMD_OP(_lhs.template eval<EVAL_TYPE>(i,j), _rhs.template eval<EVAL_TYPE>(i,j));\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return SIMD_OP((EVAL_TYPE)_lhs, _rhs.template eval<EVAL_TYPE>(i,j));\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return SIMD_OP(_lhs.template eval<EVAL_TYPE>(i,j), (EVAL_TYPE)_rhs);\
    }\
    template<typename U>\
    FASTOR_INLINE EVAL_TYPE eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return helper_s<TLhs,TRhs,U>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return OP(_lhs.template eval_s<EVAL_TYPE>(i,j), _rhs.template eval_s<EVAL_TYPE>(i,j));\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return OP((EVAL_TYPE)_lhs, _rhs.template eval_s<EVAL_TYPE>(i,j));\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return OP(_lhs.template eval_s<EVAL_TYPE>(i,j), (EVAL_TYPE)_rhs);\
    }\
    template<typename U>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> teval(const std::array<int,DIM0> &as) const {\
        return thelper<TLhs,TRhs,U>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> thelper(const std::array<int,DIM0> &as) const {\
        return SIMD_OP(_lhs.template teval<EVAL_TYPE>(as), _rhs.template teval<EVAL_TYPE>(as));\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> thelper(const std::array<int,DIM0> &as) const {\
        return SIMD_OP((EVAL_TYPE)_lhs, _rhs.template teval<EVAL_TYPE>(as));\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> thelper(const std::array<int,DIM0> &as) const {\
        return SIMD_OP(_lhs.template teval<EVAL_TYPE>(as), (EVAL_TYPE)_rhs);\
    }\
    template<typename U>\
    FASTOR_INLINE EVAL_TYPE teval_s(const std::array<int,DIM0> &as) const {\
        return thelper_s<TLhs,TRhs,U>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE thelper_s(const std::array<int,DIM0> &as) const {\
        return OP(_lhs.template teval_s<EVAL_TYPE>(as), _rhs.template teval_s<EVAL_TYPE>(as));\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE thelper_s(const std::array<int,DIM0> &as) const {\
        return OP((EVAL_TYPE)_lhs, _rhs.template teval_s<U>(as));\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE thelper_s(const std::array<int,DIM0> &as) const {\
        return OP(_lhs.template teval_s<EVAL_TYPE>(as), (EVAL_TYPE)_rhs);\
    }\
};\
template<typename TLhs, typename TRhs, size_t DIM0,\
         typename std::enable_if<!is_primitive_v_<TLhs> &&\
                                 !is_primitive_v_<TRhs>,bool>::type = 0 >\
FASTOR_INLINE Binary ##NAME ## Op<TLhs, TRhs, DIM0> OP_NAME(const AbstractTensor<TLhs,DIM0> &_lhs, const AbstractTensor<TRhs,DIM0> &_rhs) {\
  return Binary ##NAME ## Op<TLhs, TRhs, DIM0>(_lhs.self(), _rhs.self());\
}\
template<typename TLhs, typename TRhs, size_t DIM0,\
         typename std::enable_if<!is_primitive_v_<TLhs> &&\
                                 is_primitive_v_<TRhs>,bool>::type = 0 >\
FASTOR_INLINE Binary ##NAME ## Op<TLhs, TRhs, DIM0> OP_NAME(const AbstractTensor<TLhs,DIM0> &_lhs, TRhs bb) {\
  return Binary ##NAME ## Op<TLhs, TRhs, DIM0>(_lhs.self(), bb);\
}\
template<typename TLhs, typename TRhs, size_t DIM0,\
         typename std::enable_if<is_primitive_v_<TLhs> &&\
                                 !is_primitive_v_<TRhs>,bool>::type = 0 >\
FASTOR_INLINE Binary ##NAME ## Op<TLhs, TRhs, DIM0> OP_NAME(TLhs bb, const AbstractTensor<TRhs,DIM0> &_rhs) {\
  return Binary ##NAME ## Op<TLhs, TRhs, DIM0>(bb,_rhs.self());\
}\
template<typename TLhs, typename TRhs, size_t DIM0, size_t DIM1,\
         typename std::enable_if<!is_primitive_v_<TLhs> &&\
                                 !is_primitive_v_<TRhs> &&\
                                 DIM0!=DIM1,bool>::type = 0 >\
FASTOR_INLINE Binary ##NAME ## Op<TLhs, TRhs, meta_min<DIM0,DIM1>::value>\
OP_NAME(const AbstractTensor<TLhs,DIM0> &_lhs, const AbstractTensor<TRhs,DIM1> &_rhs) {\
  return Binary ##NAME ## Op<TLhs, TRhs, meta_min<DIM0,DIM1>::value>(_lhs.self(), _rhs.self());\
}\

// Dispatch based on type of expressions not the tensor
FASTOR_MAKE_BINARY_MATH_OPS(min, min, std::min, Min, scalar_type)
FASTOR_MAKE_BINARY_MATH_OPS(max, max, std::max, Max, scalar_type)
FASTOR_MAKE_BINARY_MATH_OPS(pow, pow, std::pow, Pow, scalar_type)
FASTOR_MAKE_BINARY_MATH_OPS(atan2, atan2, std::atan2, Atan2, scalar_type)
FASTOR_MAKE_BINARY_MATH_OPS(hypot, hypot, std::hypot, Hypot, scalar_type)

// Dispatch based on the type of tensor and not the expression
// FASTOR_MAKE_BINARY_MATH_OPS(min, min, std::min, Min, U)
// FASTOR_MAKE_BINARY_MATH_OPS(max, max, std::max, Max, U)
// FASTOR_MAKE_BINARY_MATH_OPS(pow, pow, std::pow, Pow, U)
// FASTOR_MAKE_BINARY_MATH_OPS(atan2, atan2, std::atan2, Atan2, U)
// FASTOR_MAKE_BINARY_MATH_OPS(hypot, hypot, std::hypot, Hypot, U)


// Create assignment for all binary math_ops
#define FASTOR_MAKE_BINARY_MATH_ASSIGNMENT(NAME, ASSIGN_TYPE)\
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,\
    enable_if_t_<!(requires_evaluation_v<TLhs> || requires_evaluation_v<TRhs>),bool> = false>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, const Binary ##NAME ## Op<TLhs, TRhs, OtherDIM> &src) {\
    trivial_assign ##ASSIGN_TYPE (dst.self(), src.self());\
}\
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,\
    enable_if_t_<(requires_evaluation_v<TLhs> || requires_evaluation_v<TRhs>),bool> = false>\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, const Binary ##NAME ## Op<TLhs, TRhs, OtherDIM> &src) {\
    using result_type = typename Binary ##NAME ## Op<TLhs, TRhs, OtherDIM>::result_type;\
    const result_type a(src.self());\
    trivial_assign ##ASSIGN_TYPE (dst.self(), a);\
}\


#define FASTOR_MAKE_BINARY_MATH_ASSIGNMENTS(NAME)\
FASTOR_MAKE_BINARY_MATH_ASSIGNMENT(NAME,     )\
FASTOR_MAKE_BINARY_MATH_ASSIGNMENT(NAME, _add)\
FASTOR_MAKE_BINARY_MATH_ASSIGNMENT(NAME, _sub)\
FASTOR_MAKE_BINARY_MATH_ASSIGNMENT(NAME, _mul)\
FASTOR_MAKE_BINARY_MATH_ASSIGNMENT(NAME, _div)\

FASTOR_MAKE_BINARY_MATH_ASSIGNMENTS(Min)
FASTOR_MAKE_BINARY_MATH_ASSIGNMENTS(Max)
FASTOR_MAKE_BINARY_MATH_ASSIGNMENTS(Pow)
FASTOR_MAKE_BINARY_MATH_ASSIGNMENTS(Atan2)
FASTOR_MAKE_BINARY_MATH_ASSIGNMENTS(Hypot)

}


#endif // BINARY_MATH_OP_H
