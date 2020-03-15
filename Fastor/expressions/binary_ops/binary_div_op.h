#ifndef BINARY_DIV_OP_H
#define BINARY_DIV_OP_H

#include "Fastor/tensor/AbstractTensor.h"
#include "Fastor/meta/tensor_post_meta.h"

namespace Fastor {

// Dispatch based on type of expressions not the tensor
#define FASTOR_BD_OP_EVAL_TYPE scalar_type

// Dispatch based on the type of tensor and not the expression
// #define FASTOR_BD_OP_EVAL_TYPE U


template<typename TLhs, typename TRhs, size_t DIM0>
struct BinaryDivOp: public AbstractTensor<BinaryDivOp<TLhs, TRhs, DIM0>,DIM0> {
    typename ExprBinderType<TLhs>::type lhs;
    typename ExprBinderType<TRhs>::type rhs;
public:

    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    using scalar_type = typename scalar_type_finder<BinaryDivOp<TLhs, TRhs, DIM0>>::type;

    FASTOR_INLINE BinaryDivOp(typename ExprBinderType<TLhs>::type lhs, typename ExprBinderType<TRhs>::type rhs) : lhs(lhs), rhs(rhs) {}

    FASTOR_INLINE FASTOR_INDEX size() const {return helper_size<TLhs,TRhs>();}
    template<class LExpr, class RExpr,
             typename std::enable_if<std::is_arithmetic<LExpr>::value,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_size() const {return rhs.size();}
    template<class LExpr, class RExpr,
             typename std::enable_if<std::is_arithmetic<RExpr>::value,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_size() const {return lhs.size();}
    template<class LExpr, class RExpr,
             typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                     !std::is_arithmetic<RExpr>::value,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_size() const {
#ifndef NDEBUG
        FASTOR_ASSERT(rhs.size()==lhs.size(),"EXPRESSION SIZE MISMATCH");
#endif
        return rhs.size();
    }

    FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) const {return helper_dimension<TLhs,TRhs>(i);}
    template<class LExpr, class RExpr,
             typename std::enable_if<std::is_arithmetic<LExpr>::value,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {return rhs.dimension(i);}
    template<class LExpr, class RExpr,
             typename std::enable_if<std::is_arithmetic<RExpr>::value,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {return lhs.dimension(i);}
    template<class LExpr, class RExpr,
             typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                     !std::is_arithmetic<RExpr>::value,bool>::type =0 >
    FASTOR_INLINE FASTOR_INDEX helper_dimension(FASTOR_INDEX i) const {
#ifndef NDEBUG
        FASTOR_ASSERT(rhs.dimension(i)==lhs.dimension(i),"EXPRESSION SHAPE MISMATCH");
#endif
        return rhs.dimension(i);
    }

    template<typename U>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        return helper<TLhs,TRhs,U>(i);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,DEFAULT_ABI> helper(FASTOR_INDEX i) const {
#ifndef FASTOR_UNSAFE_MATH
        return lhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i) / rhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i);
#else
        return lhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i) * rcp(rhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i));
#endif
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,DEFAULT_ABI> helper(FASTOR_INDEX i) const {
#ifndef FASTOR_UNSAFE_MATH
        return (FASTOR_BD_OP_EVAL_TYPE)lhs / rhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i);
#else
        return (FASTOR_BD_OP_EVAL_TYPE)lhs * rcp(rhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i));
#endif
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,DEFAULT_ABI> helper(FASTOR_INDEX i) const {
#ifndef FASTOR_UNSAFE_MATH
        return lhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i) / (FASTOR_BD_OP_EVAL_TYPE)rhs;
      // return lhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i) *(1./(FASTOR_BD_OP_EVAL_TYPE)rhs);
#else
        return lhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i) * rcp(SIMDVector<FASTOR_BD_OP_EVAL_TYPE,DEFAULT_ABI>(rhs));
#endif
    }

    // scalar based
    template<typename U>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE eval_s(FASTOR_INDEX i) const {
        return helper_s<TLhs,TRhs,U>(i);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE helper_s(FASTOR_INDEX i) const {
        return lhs.template eval_s<FASTOR_BD_OP_EVAL_TYPE>(i) / rhs.template eval_s<FASTOR_BD_OP_EVAL_TYPE>(i);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE helper_s(FASTOR_INDEX i) const {
        return (FASTOR_BD_OP_EVAL_TYPE)lhs / rhs.template eval_s<FASTOR_BD_OP_EVAL_TYPE>(i);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE helper_s(FASTOR_INDEX i) const {
        return lhs.template eval_s<FASTOR_BD_OP_EVAL_TYPE>(i) / (FASTOR_BD_OP_EVAL_TYPE)rhs;
    }

    // for 2D tensors
    template<typename U>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return helper<TLhs,TRhs,U>(i,j);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,DEFAULT_ABI> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {
#ifndef FASTOR_UNSAFE_MATH
        return lhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i,j) / rhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i,j);
#else
        return lhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i,j) * rcp(rhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i,j));
#endif
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,DEFAULT_ABI> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {
#ifndef FASTOR_UNSAFE_MATH
        return (FASTOR_BD_OP_EVAL_TYPE)lhs / rhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i,j);
#else
        return (FASTOR_BD_OP_EVAL_TYPE)lhs * rcp(rhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i,j));
#endif
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,DEFAULT_ABI> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {
#ifndef FASTOR_UNSAFE_MATH
        return lhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i,j) / rhs;
        // return lhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i,j) * (1./(FASTOR_BD_OP_EVAL_TYPE)rhs);
#else
        return lhs.template eval<FASTOR_BD_OP_EVAL_TYPE>(i,j) * rcp(SIMDVector<FASTOR_BD_OP_EVAL_TYPE,DEFAULT_ABI>(rhs));
#endif
    }

    // scalar based (for 2D tensors)
    template<typename U>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return helper_s<TLhs,TRhs,U>(i,j);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return lhs.template eval_s<FASTOR_BD_OP_EVAL_TYPE>(i,j) / rhs.template eval_s<FASTOR_BD_OP_EVAL_TYPE>(i,j);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return (FASTOR_BD_OP_EVAL_TYPE)lhs / rhs.template eval_s<FASTOR_BD_OP_EVAL_TYPE>(i,j);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return lhs.template eval_s<FASTOR_BD_OP_EVAL_TYPE>(i,j) / (FASTOR_BD_OP_EVAL_TYPE)rhs;
    }

    // for nD tensors
    template<typename U>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> teval(const std::array<int,DIM0> &as) const {
        return thelper<TLhs,TRhs,U>(as);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,DEFAULT_ABI> thelper(const std::array<int,DIM0> &as) const {
        return lhs.template teval<FASTOR_BD_OP_EVAL_TYPE>(as) / rhs.template teval<FASTOR_BD_OP_EVAL_TYPE>(as);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,DEFAULT_ABI> thelper(const std::array<int,DIM0> &as) const {
        return (FASTOR_BD_OP_EVAL_TYPE)lhs / rhs.template teval<FASTOR_BD_OP_EVAL_TYPE>(as);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<FASTOR_BD_OP_EVAL_TYPE,DEFAULT_ABI> thelper(const std::array<int,DIM0> &as) const {
        return lhs.template teval<FASTOR_BD_OP_EVAL_TYPE>(as) / (FASTOR_BD_OP_EVAL_TYPE)rhs;
    }

    // scalar based (for nD tensors)
    template<typename U>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE teval_s(const std::array<int,DIM0> &as) const {
        return thelper_s<TLhs,TRhs,U>(as);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE thelper_s(const std::array<int,DIM0> &as) const {
        return lhs.template teval_s<FASTOR_BD_OP_EVAL_TYPE>(as) / rhs.template teval_s<FASTOR_BD_OP_EVAL_TYPE>(as);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE thelper_s(const std::array<int,DIM0> &as) const {
        return (FASTOR_BD_OP_EVAL_TYPE)lhs / rhs.template teval_s<FASTOR_BD_OP_EVAL_TYPE>(as);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE FASTOR_BD_OP_EVAL_TYPE thelper_s(const std::array<int,DIM0> &as) const {
        return lhs.template teval_s<FASTOR_BD_OP_EVAL_TYPE>(as) / (FASTOR_BD_OP_EVAL_TYPE)rhs;
    }
};

template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&
                                 !std::is_arithmetic<TRhs>::value,bool>::type = 0 >
FASTOR_INLINE BinaryDivOp<TLhs, TRhs, DIM0> operator/(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM0> &rhs) {
  return BinaryDivOp<TLhs, TRhs, DIM0>(lhs.self(), rhs.self());
}

#ifndef FASTOR_DISPATCH_DIV_TO_MUL_EXPR
template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&
                                 std::is_arithmetic<TRhs>::value,bool>::type = 0 >
FASTOR_INLINE BinaryDivOp<TLhs, TRhs, DIM0> operator/(const AbstractTensor<TLhs,DIM0> &lhs, TRhs bb) {
  return BinaryDivOp<TLhs, TRhs, DIM0>(lhs.self(), bb);
}
template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<std::is_arithmetic<TLhs>::value &&
                                 !std::is_arithmetic<TRhs>::value,bool>::type = 0 >
FASTOR_INLINE BinaryDivOp<TLhs, TRhs, DIM0> operator/(TLhs bb, const AbstractTensor<TRhs,DIM0> &rhs) {
  return BinaryDivOp<TLhs, TRhs, DIM0>(bb,rhs.self());
}
#else
template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&
                                 std::is_arithmetic<TRhs>::value,bool>::type = 0 >
FASTOR_INLINE BinaryMulOp<TLhs, TRhs, DIM0> operator/(const AbstractTensor<TLhs,DIM0> &lhs, TRhs bb) {
  return BinaryMulOp<TLhs, TRhs, DIM0>(
    lhs.self(),
    static_cast<typename scalar_type_finder<TLhs>::type>(1)/static_cast<typename scalar_type_finder<TLhs>::type>(bb));
}
template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<std::is_arithmetic<TLhs>::value &&
                                 !std::is_arithmetic<TRhs>::value,bool>::type = 0 >
FASTOR_INLINE BinaryMulOp<TLhs, TRhs, DIM0> operator/(TLhs bb, const AbstractTensor<TRhs,DIM0> &rhs) {
  return BinaryMulOp<TLhs, TRhs, DIM0>(
    static_cast<typename scalar_type_finder<TLhs>::type>(1)/static_cast<typename scalar_type_finder<TLhs>::type>(bb),
    rhs.self());
}
#endif

template<typename TLhs, typename TRhs, size_t DIM0, size_t DIM1,
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&
                                 !std::is_arithmetic<TRhs>::value &&
                                 DIM0!=DIM1,bool>::type = 0 >
FASTOR_INLINE BinaryDivOp<TLhs, TRhs, meta_min<DIM0,DIM1>::value>
operator/(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM1> &rhs) {
  return BinaryDivOp<TLhs, TRhs, meta_min<DIM0,DIM1>::value>(lhs.self(), rhs.self());
}

}

#endif // BINARY_DIV_OP_H

