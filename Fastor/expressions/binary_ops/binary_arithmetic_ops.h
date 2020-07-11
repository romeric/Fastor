#ifndef BINARY_ARITHMETIC_OP_H
#define BINARY_ARITHMETIC_OP_H

#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/expressions/expression_traits.h"


namespace Fastor {


#define FASTOR_MAKE_BINARY_ARITHMETIC_OPS(OP, NAME, EVAL_TYPE) \
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
        return _lhs.template eval<EVAL_TYPE>(i) OP _rhs.template eval<EVAL_TYPE>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> helper(FASTOR_INDEX i) const {\
        return (EVAL_TYPE)_lhs OP _rhs.template eval<EVAL_TYPE>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> helper(FASTOR_INDEX i) const {\
        return _lhs.template eval<EVAL_TYPE>(i) OP (EVAL_TYPE)_rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE EVAL_TYPE eval_s(FASTOR_INDEX i) const {\
        return helper_s<TLhs,TRhs,U>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE helper_s(FASTOR_INDEX i) const {\
        return _lhs.template eval_s<EVAL_TYPE>(i) OP _rhs.template eval_s<EVAL_TYPE>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE helper_s(FASTOR_INDEX i) const {\
        return (EVAL_TYPE)_lhs OP _rhs.template eval_s<EVAL_TYPE>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE helper_s(FASTOR_INDEX i) const {\
        return _lhs.template eval_s<EVAL_TYPE>(i) OP (EVAL_TYPE)_rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return helper<TLhs,TRhs,U>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return _lhs.template eval<EVAL_TYPE>(i,j) OP _rhs.template eval<EVAL_TYPE>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return (EVAL_TYPE)_lhs OP _rhs.template eval<EVAL_TYPE>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return _lhs.template eval<EVAL_TYPE>(i,j) OP (EVAL_TYPE)_rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE EVAL_TYPE eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return helper_s<TLhs,TRhs,U>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return _lhs.template eval_s<EVAL_TYPE>(i,j) OP _rhs.template eval_s<EVAL_TYPE>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return (EVAL_TYPE)_lhs OP _rhs.template eval_s<EVAL_TYPE>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return _lhs.template eval_s<EVAL_TYPE>(i,j) OP (EVAL_TYPE)_rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> teval(const std::array<int,DIM0> &as) const {\
        return thelper<TLhs,TRhs,U>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> thelper(const std::array<int,DIM0> &as) const {\
        return _lhs.template teval<EVAL_TYPE>(as) OP _rhs.template teval<EVAL_TYPE>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> thelper(const std::array<int,DIM0> &as) const {\
        return (EVAL_TYPE)_lhs OP _rhs.template teval<EVAL_TYPE>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,simd_abi_type> thelper(const std::array<int,DIM0> &as) const {\
        return _lhs.template teval<EVAL_TYPE>(as) OP (EVAL_TYPE)_rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE EVAL_TYPE teval_s(const std::array<int,DIM0> &as) const {\
        return thelper_s<TLhs,TRhs,U>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE thelper_s(const std::array<int,DIM0> &as) const {\
        return _lhs.template teval_s<EVAL_TYPE>(as) OP _rhs.template teval_s<EVAL_TYPE>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<is_primitive_v_<LExpr> &&\
                                   !is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE thelper_s(const std::array<int,DIM0> &as) const {\
        return (EVAL_TYPE)_lhs OP _rhs.template teval_s<U>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!is_primitive_v_<LExpr> &&\
                                   is_primitive_v_<RExpr>,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE thelper_s(const std::array<int,DIM0> &as) const {\
        return _lhs.template teval_s<EVAL_TYPE>(as) OP (EVAL_TYPE)_rhs;\
    }\
};\
template<typename TLhs, typename TRhs, size_t DIM0,\
         typename std::enable_if<!is_primitive_v_<TLhs> &&\
                                 !is_primitive_v_<TRhs>,bool>::type = 0 >\
FASTOR_INLINE Binary ##NAME ## Op<TLhs, TRhs, DIM0> operator OP(const AbstractTensor<TLhs,DIM0> &_lhs, const AbstractTensor<TRhs,DIM0> &_rhs) {\
  return Binary ##NAME ## Op<TLhs, TRhs, DIM0>(_lhs.self(), _rhs.self());\
}\
template<typename TLhs, typename TRhs, size_t DIM0,\
         typename std::enable_if<!is_primitive_v_<TLhs> &&\
                                 is_primitive_v_<TRhs>,bool>::type = 0 >\
FASTOR_INLINE Binary ##NAME ## Op<TLhs, TRhs, DIM0> operator OP(const AbstractTensor<TLhs,DIM0> &_lhs, TRhs bb) {\
  return Binary ##NAME ## Op<TLhs, TRhs, DIM0>(_lhs.self(), bb);\
}\
template<typename TLhs, typename TRhs, size_t DIM0,\
         typename std::enable_if<is_primitive_v_<TLhs> &&\
                                 !is_primitive_v_<TRhs>,bool>::type = 0 >\
FASTOR_INLINE Binary ##NAME ## Op<TLhs, TRhs, DIM0> operator OP(TLhs bb, const AbstractTensor<TRhs,DIM0> &_rhs) {\
  return Binary ##NAME ## Op<TLhs, TRhs, DIM0>(bb,_rhs.self());\
}\
template<typename TLhs, typename TRhs, size_t DIM0, size_t DIM1,\
         typename std::enable_if<!is_primitive_v_<TLhs> &&\
                                 !is_primitive_v_<TRhs> &&\
                                 DIM0!=DIM1,bool>::type = 0 >\
FASTOR_INLINE Binary ##NAME ## Op<TLhs, TRhs, meta_min<DIM0,DIM1>::value>\
operator OP(const AbstractTensor<TLhs,DIM0> &_lhs, const AbstractTensor<TRhs,DIM1> &_rhs) {\
  return Binary ##NAME ## Op<TLhs, TRhs, meta_min<DIM0,DIM1>::value>(_lhs.self(), _rhs.self());\
}\

// Dispatch based on type of expressions not the tensor
FASTOR_MAKE_BINARY_ARITHMETIC_OPS(+, Add, scalar_type)
FASTOR_MAKE_BINARY_ARITHMETIC_OPS(-, Sub, scalar_type)
FASTOR_MAKE_BINARY_ARITHMETIC_OPS(*, Mul, scalar_type)
// FASTOR_MAKE_BINARY_ARITHMETIC_OPS(/, Div, scalar_type) // Dont create div as it is a special case

// Dispatch based on the type of tensor and not the expression
// FASTOR_MAKE_BINARY_ARITHMETIC_OPS(+, Add, U)
// FASTOR_MAKE_BINARY_ARITHMETIC_OPS(-, Sub, U)
// FASTOR_MAKE_BINARY_ARITHMETIC_OPS(*, Mul, U)
// FASTOR_MAKE_BINARY_ARITHMETIC_OPS(/, Div, U)

}


#endif // BINARY_ARITHMETIC_OP_H
