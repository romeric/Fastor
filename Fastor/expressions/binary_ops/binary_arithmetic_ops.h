#ifndef BINARY_ARITHMETIC_OP_H
#define BINARY_ARITHMETIC_OP_H

#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/meta/tensor_post_meta.h"



namespace Fastor {

#define FASTOR_MAKE_BINARY_ARITHMETIC_OPS(OP, NAME, EVAL_TYPE) \
template<typename TLhs, typename TRhs, size_t DIM0>\
struct Binary ##NAME ## Op: public AbstractTensor<Binary ##NAME ## Op<TLhs, TRhs, DIM0>,DIM0> {\
    typename ExprBinderType<TLhs>::type lhs;\
    typename ExprBinderType<TRhs>::type rhs;\
public:\
    static constexpr FASTOR_INDEX Dimension = DIM0;\
    static constexpr FASTOR_INDEX rank() {return DIM0;}\
    using scalar_type = typename scalar_type_finder<Binary ##NAME ## Op<TLhs, TRhs, DIM0>>::type;\
    FASTOR_INLINE Binary ##NAME ## Op(typename ExprBinderType<TLhs>::type lhs, typename ExprBinderType<TRhs>::type rhs) : lhs(lhs), rhs(rhs) {}\
    FASTOR_INLINE FASTOR_INDEX size() const {return helper_size<TLhs,TRhs>();}\
    template<class LExpr, class RExpr,\
             typename std::enable_if<std::is_arithmetic<LExpr>::value,bool>::type =0 >\
    FASTOR_INLINE FASTOR_INDEX helper_size() const {return rhs.size();}\
    template<class LExpr, class RExpr,\
             typename std::enable_if<std::is_arithmetic<RExpr>::value,bool>::type =0 >\
    FASTOR_INLINE FASTOR_INDEX helper_size() const {return lhs.size();}\
    template<class LExpr, class RExpr,\
             typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                     !std::is_arithmetic<RExpr>::value,bool>::type =0 >\
    FASTOR_INLINE FASTOR_INDEX helper_size() const {\
        FASTOR_ASSERT(rhs.size()==lhs.size(),"EXPRESSION SIZE MISMATCH");\
        return rhs.size();\
    }\
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return helper_dimension<TLhs,TRhs>(i);}\
    template<class LExpr, class RExpr,\
             typename std::enable_if<std::is_arithmetic<LExpr>::value,bool>::type =0 >\
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {return rhs.dimension(i);}\
    template<class LExpr, class RExpr,\
             typename std::enable_if<std::is_arithmetic<RExpr>::value,bool>::type =0 >\
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {return lhs.dimension(i);}\
    template<class LExpr, class RExpr,\
             typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                     !std::is_arithmetic<RExpr>::value,bool>::type =0 >\
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {\
        FASTOR_ASSERT(rhs.dimension(i)==lhs.dimension(i),"EXPRESSION SHAPE MISMATCH");\
        return rhs.dimension(i);\
    }\
    template<typename U>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,DEFAULT_ABI> eval(FASTOR_INDEX i) const {\
        return helper<TLhs,TRhs,U>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,DEFAULT_ABI> helper(FASTOR_INDEX i) const {\
        return lhs.template eval<EVAL_TYPE>(i) OP rhs.template eval<EVAL_TYPE>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,DEFAULT_ABI> helper(FASTOR_INDEX i) const {\
        return (EVAL_TYPE)lhs OP rhs.template eval<EVAL_TYPE>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,DEFAULT_ABI> helper(FASTOR_INDEX i) const {\
        return lhs.template eval<EVAL_TYPE>(i) OP (EVAL_TYPE)rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE EVAL_TYPE eval_s(FASTOR_INDEX i) const {\
        return helper_s<TLhs,TRhs,U>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE helper_s(FASTOR_INDEX i) const {\
        return lhs.template eval_s<EVAL_TYPE>(i) OP rhs.template eval_s<EVAL_TYPE>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE helper_s(FASTOR_INDEX i) const {\
        return (EVAL_TYPE)lhs OP rhs.template eval_s<EVAL_TYPE>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE helper_s(FASTOR_INDEX i) const {\
        return lhs.template eval_s<EVAL_TYPE>(i) OP (EVAL_TYPE)rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return helper<TLhs,TRhs,U>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,DEFAULT_ABI> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return lhs.template eval<EVAL_TYPE>(i,j) OP rhs.template eval<EVAL_TYPE>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,DEFAULT_ABI> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return (EVAL_TYPE)lhs OP rhs.template eval<EVAL_TYPE>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,DEFAULT_ABI> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return lhs.template eval<EVAL_TYPE>(i,j) OP (EVAL_TYPE)rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE EVAL_TYPE eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return helper_s<TLhs,TRhs,U>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return lhs.template eval_s<EVAL_TYPE>(i,j) OP rhs.template eval_s<EVAL_TYPE>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return (EVAL_TYPE)lhs OP rhs.template eval_s<EVAL_TYPE>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return lhs.template eval_s<EVAL_TYPE>(i,j) OP (EVAL_TYPE)rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,DEFAULT_ABI> teval(const std::array<int,DIM0> &as) const {\
        return thelper<TLhs,TRhs,U>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,DEFAULT_ABI> thelper(const std::array<int,DIM0> &as) const {\
        return lhs.template teval<EVAL_TYPE>(as) OP rhs.template teval<EVAL_TYPE>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,DEFAULT_ABI> thelper(const std::array<int,DIM0> &as) const {\
        return (EVAL_TYPE)lhs OP rhs.template teval<EVAL_TYPE>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<EVAL_TYPE,DEFAULT_ABI> thelper(const std::array<int,DIM0> &as) const {\
        return lhs.template teval<EVAL_TYPE>(as) OP (EVAL_TYPE)rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE EVAL_TYPE teval_s(const std::array<int,DIM0> &as) const {\
        return thelper_s<TLhs,TRhs,U>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE thelper_s(const std::array<int,DIM0> &as) const {\
        return lhs.template teval_s<EVAL_TYPE>(as) OP rhs.template teval_s<EVAL_TYPE>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE thelper_s(const std::array<int,DIM0> &as) const {\
        return (EVAL_TYPE)lhs OP rhs.template teval_s<U>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE EVAL_TYPE thelper_s(const std::array<int,DIM0> &as) const {\
        return lhs.template teval_s<EVAL_TYPE>(as) OP (EVAL_TYPE)rhs;\
    }\
};\
template<typename TLhs, typename TRhs, size_t DIM0,\
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&\
                                 !std::is_arithmetic<TRhs>::value,bool>::type = 0 >\
FASTOR_INLINE Binary ##NAME ## Op<TLhs, TRhs, DIM0> operator OP(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM0> &rhs) {\
  return Binary ##NAME ## Op<TLhs, TRhs, DIM0>(lhs.self(), rhs.self());\
}\
template<typename TLhs, typename TRhs, size_t DIM0,\
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&\
                                 std::is_arithmetic<TRhs>::value,bool>::type = 0 >\
FASTOR_INLINE Binary ##NAME ## Op<TLhs, TRhs, DIM0> operator OP(const AbstractTensor<TLhs,DIM0> &lhs, TRhs bb) {\
  return Binary ##NAME ## Op<TLhs, TRhs, DIM0>(lhs.self(), bb);\
}\
template<typename TLhs, typename TRhs, size_t DIM0,\
         typename std::enable_if<std::is_arithmetic<TLhs>::value &&\
                                 !std::is_arithmetic<TRhs>::value,bool>::type = 0 >\
FASTOR_INLINE Binary ##NAME ## Op<TLhs, TRhs, DIM0> operator OP(TLhs bb, const AbstractTensor<TRhs,DIM0> &rhs) {\
  return Binary ##NAME ## Op<TLhs, TRhs, DIM0>(bb,rhs.self());\
}\
template<typename TLhs, typename TRhs, size_t DIM0, size_t DIM1,\
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&\
                                 !std::is_arithmetic<TRhs>::value &&\
                                 DIM0!=DIM1,bool>::type = 0 >\
FASTOR_INLINE Binary ##NAME ## Op<TLhs, TRhs, meta_min<DIM0,DIM1>::value>\
operator OP(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM1> &rhs) {\
  return Binary ##NAME ## Op<TLhs, TRhs, meta_min<DIM0,DIM1>::value>(lhs.self(), rhs.self());\
}\

// Dispatch based on type of expressions not the tensor
FASTOR_MAKE_BINARY_ARITHMETIC_OPS(+, Add, scalar_type)
FASTOR_MAKE_BINARY_ARITHMETIC_OPS(-, Sub, scalar_type)
FASTOR_MAKE_BINARY_ARITHMETIC_OPS(*, Mul, scalar_type)
FASTOR_MAKE_BINARY_ARITHMETIC_OPS(/, Div, scalar_type)

// // Dispatch based on the type of tensor and not the expression
// FASTOR_MAKE_BINARY_ARITHMETIC_OPS(+, Add, U)
// FASTOR_MAKE_BINARY_ARITHMETIC_OPS(-, Sub, U)
// FASTOR_MAKE_BINARY_ARITHMETIC_OPS(*, Mul, U)
// FASTOR_MAKE_BINARY_ARITHMETIC_OPS(/, Div, U)

}


#endif // BINARY_ARITHMETIC_OP_H