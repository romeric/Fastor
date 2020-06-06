#ifndef BINARY_CMP_OPS
#define BINARY_CMP_OPS

#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/tensor/TensorTraits.h"
#include "Fastor/expressions/expression_traits.h"

namespace Fastor {

#define FASTOR_MAKE_BINARY_CMP_TENSOR_OPS_(OP, NAME, EVAL_TYPE) \
template<typename TLhs, typename TRhs, size_t DIM0>\
struct BinaryCmpOp##NAME: public AbstractTensor<BinaryCmpOp##NAME<TLhs, TRhs, DIM0>,DIM0> {\
    expression_t<TLhs> _lhs;\
    expression_t<TRhs> _rhs;\
    static constexpr FASTOR_INDEX Dimension = DIM0;\
    static constexpr FASTOR_INDEX rank() {return DIM0;}\
    using scalar_type = typename scalar_type_finder<BinaryCmpOp##NAME<TLhs, TRhs, DIM0>>::type;\
    using result_type = to_bool_tensor_t<binary_arithmetic_result_t<BinaryCmpOp##NAME<TLhs, TRhs, DIM0>>>;\
    using simd_vector_type = binary_op_simd_vector_t<BinaryCmpOp##NAME<TLhs, TRhs, DIM0> >;\
    using simd_abi_type = typename simd_vector_type::abi_type;\
    using ABI = simd_abi::fixed_size<SIMDVector<scalar_type,simd_abi_type>::Size>;\
    using UU = bool /*this needs to change to U once masks are implemented*/;\
    FASTOR_INLINE BinaryCmpOp##NAME(expression_t<TLhs> inlhs, expression_t<TRhs> inrhs) : _lhs(inlhs), _rhs(inrhs) {}\
    FASTOR_INLINE FASTOR_INDEX size() const {return helper_size<TLhs,TRhs>();}\
    template<class LExpr, class RExpr,\
             typename std::enable_if<std::is_arithmetic<LExpr>::value,bool>::type =0 >\
    FASTOR_INLINE FASTOR_INDEX helper_size() const {return _rhs.size();}\
    template<class LExpr, class RExpr,\
             typename std::enable_if<std::is_arithmetic<RExpr>::value,bool>::type =0 >\
    FASTOR_INLINE FASTOR_INDEX helper_size() const {return _lhs.size();}\
    template<class LExpr, class RExpr,\
             typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                     !std::is_arithmetic<RExpr>::value,bool>::type =0 >\
    FASTOR_INLINE FASTOR_INDEX helper_size() const {\
        return _rhs.size();\
    }\
    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return helper_dimension<TLhs,TRhs>(i);}\
    template<class LExpr, class RExpr,\
             typename std::enable_if<std::is_arithmetic<LExpr>::value,bool>::type =0 >\
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {return _rhs.dimension(i);}\
    template<class LExpr, class RExpr,\
             typename std::enable_if<std::is_arithmetic<RExpr>::value,bool>::type =0 >\
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {return _lhs.dimension(i);}\
    template<class LExpr, class RExpr,\
             typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                     !std::is_arithmetic<RExpr>::value,bool>::type =0 >\
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {\
        return _rhs.dimension(i);\
    }\
    FASTOR_INLINE expression_t<TLhs> lhs() const {return _lhs;}\
    FASTOR_INLINE expression_t<TRhs> rhs() const {return _rhs;}\
    template<typename U>\
    FASTOR_INLINE SIMDVector<UU,ABI> eval(FASTOR_INDEX i) const {\
        return helper<TLhs,TRhs,U>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<UU,ABI> helper(FASTOR_INDEX i) const {\
        return _lhs.template eval<EVAL_TYPE>(i) OP _rhs.template eval<EVAL_TYPE>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<UU,ABI> helper(FASTOR_INDEX i) const {\
        return (EVAL_TYPE)_lhs OP _rhs.template eval<EVAL_TYPE>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<UU,ABI> helper(FASTOR_INDEX i) const {\
        return _lhs.template eval<EVAL_TYPE>(i) OP (EVAL_TYPE)_rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE UU eval_s(FASTOR_INDEX i) const {\
        return helper_s<TLhs,TRhs,U>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE UU helper_s(FASTOR_INDEX i) const {\
        return _lhs.template eval_s<EVAL_TYPE>(i) OP _rhs.template eval_s<EVAL_TYPE>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE UU helper_s(FASTOR_INDEX i) const {\
        return (EVAL_TYPE)_lhs OP _rhs.template eval_s<EVAL_TYPE>(i);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE UU helper_s(FASTOR_INDEX i) const {\
        return _lhs.template eval_s<EVAL_TYPE>(i) OP (EVAL_TYPE)_rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE SIMDVector<UU,ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return helper<TLhs,TRhs,U>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<UU,ABI> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return _lhs.template eval<EVAL_TYPE>(i,j) OP _rhs.template eval<EVAL_TYPE>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<UU,ABI> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return (EVAL_TYPE)_lhs OP _rhs.template eval<EVAL_TYPE>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<UU,ABI> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return _lhs.template eval<EVAL_TYPE>(i,j) OP (EVAL_TYPE)_rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE UU eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return helper_s<TLhs,TRhs,U>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE UU helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return _lhs.template eval_s<EVAL_TYPE>(i,j) OP _rhs.template eval_s<EVAL_TYPE>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE UU helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return (EVAL_TYPE)_lhs OP _rhs.template eval_s<EVAL_TYPE>(i,j);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE UU helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {\
        return _lhs.template eval_s<EVAL_TYPE>(i,j) OP (EVAL_TYPE)_rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE SIMDVector<UU,simd_abi_type> teval(const std::array<int,DIM0> &as) const {\
        return thelper<TLhs,TRhs,U>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<UU,simd_abi_type> thelper(const std::array<int,DIM0> &as) const {\
        return _lhs.template teval<EVAL_TYPE>(as) OP _rhs.template teval<EVAL_TYPE>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<UU,simd_abi_type> thelper(const std::array<int,DIM0> &as) const {\
        return (EVAL_TYPE)_lhs OP _rhs.template teval<EVAL_TYPE>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE SIMDVector<UU,simd_abi_type> thelper(const std::array<int,DIM0> &as) const {\
        return _lhs.template teval<EVAL_TYPE>(as) OP (EVAL_TYPE)_rhs;\
    }\
    template<typename U>\
    FASTOR_INLINE UU teval_s(const std::array<int,DIM0> &as) const {\
        return thelper_s<TLhs,TRhs,U>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE UU thelper_s(const std::array<int,DIM0> &as) const {\
        return _lhs.template teval_s<EVAL_TYPE>(as) OP _rhs.template teval_s<EVAL_TYPE>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&\
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE UU thelper_s(const std::array<int,DIM0> &as) const {\
        return (EVAL_TYPE)_lhs OP _rhs.template teval_s<EVAL_TYPE>(as);\
    }\
    template<typename LExpr, typename RExpr, typename U,\
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&\
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>\
    FASTOR_INLINE UU thelper_s(const std::array<int,DIM0> &as) const {\
        return _lhs.template teval_s<EVAL_TYPE>(as) OP (EVAL_TYPE)_rhs;\
    }\
};\
template<typename TLhs, typename TRhs, size_t DIM0,\
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&\
                                 !std::is_arithmetic<TRhs>::value,bool>::type = 0 >\
FASTOR_INLINE BinaryCmpOp##NAME<TLhs, TRhs, DIM0> operator OP(const AbstractTensor<TLhs,DIM0> &_lhs, const AbstractTensor<TRhs,DIM0> &_rhs) {\
  return BinaryCmpOp##NAME<TLhs, TRhs, DIM0>(_lhs.self(), _rhs.self());\
}\
template<typename TLhs, typename TRhs, size_t DIM0,\
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&\
                                 std::is_arithmetic<TRhs>::value,bool>::type = 0 >\
FASTOR_INLINE BinaryCmpOp##NAME<TLhs, TRhs, DIM0> operator OP(const AbstractTensor<TLhs,DIM0> &_lhs, TRhs bb) {\
  return BinaryCmpOp##NAME<TLhs, TRhs, DIM0>(_lhs.self(), bb);\
}\
template<typename TLhs, typename TRhs, size_t DIM0,\
         typename std::enable_if<std::is_arithmetic<TLhs>::value &&\
                                 !std::is_arithmetic<TRhs>::value,bool>::type = 0 >\
FASTOR_INLINE BinaryCmpOp##NAME<TLhs, TRhs, DIM0> operator OP(TLhs bb, const AbstractTensor<TRhs,DIM0> &_rhs) {\
  return BinaryCmpOp##NAME<TLhs, TRhs, DIM0>(bb,_rhs.self());\
}\
template<typename TLhs, typename TRhs, size_t DIM0, size_t DIM1,\
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&\
                                 !std::is_arithmetic<TRhs>::value &&\
                                 DIM0!=DIM1,bool>::type = 0 >\
FASTOR_INLINE BinaryCmpOp##NAME<TLhs, TRhs, meta_min<DIM0,DIM1>::value>\
operator OP(const AbstractTensor<TLhs,DIM0> &_lhs, const AbstractTensor<TRhs,DIM1> &_rhs) {\
  return BinaryCmpOp##NAME<TLhs, TRhs, meta_min<DIM0,DIM1>::value>(_lhs.self(), _rhs.self());\
}\


FASTOR_MAKE_BINARY_CMP_TENSOR_OPS_(== ,EQ, scalar_type)
FASTOR_MAKE_BINARY_CMP_TENSOR_OPS_(!= ,NEQ,scalar_type)
FASTOR_MAKE_BINARY_CMP_TENSOR_OPS_(<  ,LT, scalar_type)
FASTOR_MAKE_BINARY_CMP_TENSOR_OPS_(>  ,GT, scalar_type)
FASTOR_MAKE_BINARY_CMP_TENSOR_OPS_(<= ,LE, scalar_type)
FASTOR_MAKE_BINARY_CMP_TENSOR_OPS_(>= ,GE, scalar_type)
FASTOR_MAKE_BINARY_CMP_TENSOR_OPS_(&& ,AND,scalar_type)
FASTOR_MAKE_BINARY_CMP_TENSOR_OPS_(|| ,OR, scalar_type)





#define FASTOR_MAKE_BINARY_CMP_ASSIGNMENT(NAME, ASSIGN_TYPE)\
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,\
  enable_if_t_<!(requires_evaluation_v<TLhs> || requires_evaluation_v<TRhs>),bool> = false >\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, const BinaryCmpOp ##NAME <TLhs, TRhs, OtherDIM> &src) {\
    trivial_assign ##ASSIGN_TYPE (dst.self(), src.self());\
}\
template<typename Derived, size_t DIM, typename TLhs, typename TRhs, size_t OtherDIM,\
  enable_if_t_<(requires_evaluation_v<TLhs> || requires_evaluation_v<TRhs>),bool> = false >\
FASTOR_INLINE void assign ##ASSIGN_TYPE (AbstractTensor<Derived,DIM> &dst, const BinaryCmpOp ##NAME <TLhs, TRhs, OtherDIM> &src) {\
    using lhs_type = typename get_binary_arithmetic_result_type<TLhs>::type;\
    using rhs_type = typename get_binary_arithmetic_result_type<TRhs>::type;\
    const lhs_type a(src.lhs());\
    const rhs_type b(src.rhs());\
    trivial_assign ##ASSIGN_TYPE (dst.self(), BinaryCmpOp ##NAME <lhs_type, rhs_type, OtherDIM>(a,b));\
}\

#define FASTOR_MAKE_BINARY_CMP_ASSIGNMENTS(ASSIGN_TYPE)\
FASTOR_MAKE_BINARY_CMP_ASSIGNMENT(EQ,  ASSIGN_TYPE)\
FASTOR_MAKE_BINARY_CMP_ASSIGNMENT(NEQ, ASSIGN_TYPE)\
FASTOR_MAKE_BINARY_CMP_ASSIGNMENT(LT,  ASSIGN_TYPE)\
FASTOR_MAKE_BINARY_CMP_ASSIGNMENT(GT,  ASSIGN_TYPE)\
FASTOR_MAKE_BINARY_CMP_ASSIGNMENT(LE,  ASSIGN_TYPE)\
FASTOR_MAKE_BINARY_CMP_ASSIGNMENT(GE,  ASSIGN_TYPE)\
FASTOR_MAKE_BINARY_CMP_ASSIGNMENT(AND, ASSIGN_TYPE)\
FASTOR_MAKE_BINARY_CMP_ASSIGNMENT(OR,  ASSIGN_TYPE)\

FASTOR_MAKE_BINARY_CMP_ASSIGNMENTS(     )
FASTOR_MAKE_BINARY_CMP_ASSIGNMENTS( _add)
FASTOR_MAKE_BINARY_CMP_ASSIGNMENTS( _sub)
FASTOR_MAKE_BINARY_CMP_ASSIGNMENTS( _mul)
FASTOR_MAKE_BINARY_CMP_ASSIGNMENTS( _div)


} // end of namespace Fastor


#endif // BINARY_CMP_OPS
