#ifndef BINARY_CMP_OPS
#define BINARY_CMP_OPS

#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/meta/tensor_post_meta.h"



namespace Fastor {

#define FASTOR_MAKE_BINARY_CMP_TENSOR_OPS_(OP, NAME, EVAL_TYPE) \
template<typename TLhs, typename TRhs, size_t DIM0>\
struct BinaryCmpOp##NAME: public AbstractTensor<BinaryCmpOp##NAME<TLhs, TRhs, DIM0>,DIM0> {\
    typename ExprBinderType<TLhs>::type lhs;\
    typename ExprBinderType<TRhs>::type rhs;\
    static constexpr FASTOR_INDEX Dimension = DIM0;\
    static constexpr FASTOR_INDEX rank() {return DIM0;}\
    using scalar_type = typename scalar_type_finder<BinaryCmpOp##NAME<TLhs, TRhs, DIM0>>::type;\
    static constexpr int ABI = stride_finder<scalar_type>::value*8*sizeof(bool);\
    FASTOR_INLINE BinaryCmpOp##NAME(typename ExprBinderType<TLhs>::type lhs, typename ExprBinderType<TRhs>::type rhs) : lhs(lhs), rhs(rhs) {}\
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
        return rhs.dimension(i);\
    }\
    template<typename U>\
    FASTOR_INLINE SIMDVector<U,ABI> eval(FASTOR_INDEX i) const {\
        return helper<TLhs,TRhs,U>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<U,ABI> helper(FASTOR_INDEX i) const {\
        return lhs.template eval<EVAL_TYPE>(i) OP rhs.template eval<EVAL_TYPE>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<U,ABI> helper(FASTOR_INDEX i) const {\
        return (EVAL_TYPE)lhs OP rhs.template eval<EVAL_TYPE>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<U,ABI> helper(FASTOR_INDEX i) const {\
        return lhs.template eval<EVAL_TYPE>(i) OP (EVAL_TYPE)rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {\
        return helper_s<TLhs,TRhs,U>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE U helper_s(FASTOR_INDEX i) const {\
        return lhs.template eval_s<EVAL_TYPE>(i) OP rhs.template eval_s<EVAL_TYPE>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE U helper_s(FASTOR_INDEX i) const {\
        return (EVAL_TYPE)lhs OP rhs.template eval_s<EVAL_TYPE>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE U helper_s(FASTOR_INDEX i) const {\
        return lhs.template eval_s<EVAL_TYPE>(i) OP (EVAL_TYPE)rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE SIMDVector<U,ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return helper<TLhs,TRhs,U>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<U,ABI> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return lhs.template eval<EVAL_TYPE>(i,j) OP rhs.template eval<EVAL_TYPE>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<U,ABI> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return (EVAL_TYPE)lhs OP rhs.template eval<EVAL_TYPE>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<U,ABI> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return lhs.template eval<EVAL_TYPE>(i,j) OP (EVAL_TYPE)rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return helper_s<TLhs,TRhs,U>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE U helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return lhs.template eval_s<EVAL_TYPE>(i,j) OP rhs.template eval_s<EVAL_TYPE>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE U helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return (EVAL_TYPE)lhs OP rhs.template eval_s<EVAL_TYPE>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE U helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return lhs.template eval_s<EVAL_TYPE>(i,j) OP (EVAL_TYPE)rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> teval(const std::array<int,DIM0> &as) const {\
        return thelper<TLhs,TRhs,U>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> thelper(const std::array<int,DIM0> &as) const {\
        return lhs.template teval<EVAL_TYPE>(as) OP rhs.template teval<EVAL_TYPE>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> thelper(const std::array<int,DIM0> &as) const {\
        return (EVAL_TYPE)lhs OP rhs.template teval<EVAL_TYPE>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> thelper(const std::array<int,DIM0> &as) const {\
        return lhs.template teval<EVAL_TYPE>(as) OP (EVAL_TYPE)rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE U teval_s(const std::array<int,DIM0> &as) const {\
        return thelper_s<TLhs,TRhs,U>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE U thelper_s(const std::array<int,DIM0> &as) const {\
        return lhs.template teval_s<EVAL_TYPE>(as) OP rhs.template teval_s<EVAL_TYPE>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE U thelper_s(const std::array<int,DIM0> &as) const {\
        return (EVAL_TYPE)lhs OP rhs.template teval_s<EVAL_TYPE>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE U thelper_s(const std::array<int,DIM0> &as) const {\
        return lhs.template teval_s<EVAL_TYPE>(as) OP (EVAL_TYPE)rhs;\
    }\
};\
template<typename TLhs, typename TRhs, size_t DIM0,\
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&\
                                 !std::is_arithmetic<TRhs>::value,bool>::type = 0 >\
FASTOR_INLINE BinaryCmpOp##NAME<TLhs, TRhs, DIM0> operator OP(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM0> &rhs) {\
  return BinaryCmpOp##NAME<TLhs, TRhs, DIM0>(lhs.self(), rhs.self());\
}\
template<typename TLhs, typename TRhs, size_t DIM0,\
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&\
                                 std::is_arithmetic<TRhs>::value,bool>::type = 0 >\
FASTOR_INLINE BinaryCmpOp##NAME<TLhs, TRhs, DIM0> operator OP(const AbstractTensor<TLhs,DIM0> &lhs, TRhs bb) {\
  return BinaryCmpOp##NAME<TLhs, TRhs, DIM0>(lhs.self(), bb);\
}\
template<typename TLhs, typename TRhs, size_t DIM0,\
         typename std::enable_if<std::is_arithmetic<TLhs>::value &&\
                                 !std::is_arithmetic<TRhs>::value,bool>::type = 0 >\
FASTOR_INLINE BinaryCmpOp##NAME<TLhs, TRhs, DIM0> operator OP(TLhs bb, const AbstractTensor<TRhs,DIM0> &rhs) {\
  return BinaryCmpOp##NAME<TLhs, TRhs, DIM0>(bb,rhs.self());\
}\
template<typename TLhs, typename TRhs, size_t DIM0, size_t DIM1,\
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&\
                                 !std::is_arithmetic<TRhs>::value &&\
                                 DIM0!=DIM1,bool>::type = 0 >\
FASTOR_INLINE BinaryCmpOp##NAME<TLhs, TRhs, meta_min<DIM0,DIM1>::value>\
operator OP(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM1> &rhs) {\
  return BinaryCmpOp##NAME<TLhs, TRhs, meta_min<DIM0,DIM1>::value>(lhs.self(), rhs.self());\
}\


FASTOR_MAKE_BINARY_CMP_TENSOR_OPS_(== ,EQ, scalar_type);
FASTOR_MAKE_BINARY_CMP_TENSOR_OPS_(<  ,LT, scalar_type);
FASTOR_MAKE_BINARY_CMP_TENSOR_OPS_(>  ,GT, scalar_type);
FASTOR_MAKE_BINARY_CMP_TENSOR_OPS_(<= ,LE, scalar_type);
FASTOR_MAKE_BINARY_CMP_TENSOR_OPS_(>= ,GE, scalar_type);
FASTOR_MAKE_BINARY_CMP_TENSOR_OPS_(&& ,AND, scalar_type);
FASTOR_MAKE_BINARY_CMP_TENSOR_OPS_(|| ,OR, scalar_type);

}


#endif // BINARY_CMP_OPS