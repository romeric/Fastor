#ifndef BINARY_ADD_OP_H
#define BINARY_ADD_OP_H

#include "tensor/AbstractTensor.h"
#include "meta/tensor_post_meta.h"


namespace Fastor {

template<typename T, typename std::enable_if<std::is_arithmetic<T>::value,bool>::type=0>
T forward_evaluate(const T &a) {
    return a;
}
template<typename T, typename std::enable_if<!std::is_arithmetic<T>::value,bool>::type=0>
auto forward_evaluate(const T &a) -> decltype(a.evaluate()) {
    return a.evaluate();
}

}



namespace Fastor {

template<typename TLhs, typename TRhs, size_t DIM0>
struct BinaryAddOp: public AbstractTensor<BinaryAddOp<TLhs, TRhs, DIM0>,DIM0> {

    BinaryAddOp(const TLhs& lhs, const TRhs& rhs) : lhs(lhs), rhs(rhs) {
    }

    static constexpr FASTOR_INDEX Dimension = DIM0;
    static constexpr FASTOR_INDEX rank() {return DIM0;}
    using scalar_type = typename scalar_type_finder<BinaryAddOp<TLhs, TRhs, DIM0>>::type;

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

    // Generic version of eval
    // The eval function evaluates the expression at position i
    template<typename U>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i) const {
        // Delay evaluation using a helper function to fully inform BinaryOp about lhs and rhs
        return helper<TLhs,TRhs,U>(i);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> helper(FASTOR_INDEX i) const {
        return lhs.template eval<U>(i) + rhs.template eval<U>(i);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> helper(FASTOR_INDEX i) const {
        return (U)lhs + rhs.template eval<U>(i);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> helper(FASTOR_INDEX i) const {
        return lhs.template eval<U>(i) + (U)rhs;
    }


    // scalar based
    template<typename U>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i) const {
        return helper_s<TLhs,TRhs,U>(i);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE U helper_s(FASTOR_INDEX i) const {
        return lhs.template eval_s<U>(i) + rhs.template eval_s<U>(i);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE U helper_s(FASTOR_INDEX i) const {
        return (U)lhs + rhs.template eval_s<U>(i);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE U helper_s(FASTOR_INDEX i) const {
        return lhs.template eval_s<U>(i) + (U)rhs;
    }


    // for 2D tensors
    template<typename U>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return helper<TLhs,TRhs,U>(i,j);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return lhs.template eval<U>(i,j) + rhs.template eval<U>(i,j);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return (U)lhs + rhs.template eval<U>(i,j);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> helper(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return lhs.template eval<U>(i,j) + (U)rhs;
    }

    // scalar based (for 2D tensors)
    template<typename U>
    FASTOR_INLINE U eval_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return helper_s<TLhs,TRhs,U>(i,j);
    }

    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE U helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return lhs.template eval_s<U>(i,j) + rhs.template eval_s<U>(i,j);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<std::is_arithmetic<LExpr>::value &&
                                   !std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE U helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return (U)lhs + rhs.template eval_s<U>(i,j);
    }
    template<typename LExpr, typename RExpr, typename U,
           typename std::enable_if<!std::is_arithmetic<LExpr>::value &&
                                   std::is_arithmetic<RExpr>::value,bool>::type = 0>
    FASTOR_INLINE U helper_s(FASTOR_INDEX i, FASTOR_INDEX j) const {
        return lhs.template eval_s<U>(i,j) + (U)rhs;
    }

    // constexpr FASTOR_INLINE TLhs evaluate() const {
    //     return forward_evaluate(lhs) + forward_evaluate(rhs);
    // }

// private:
    // const TLhs &lhs;
    // const TRhs &rhs;
    typename ExprBinderType<TLhs>::type lhs;
    typename ExprBinderType<TRhs>::type rhs;
};

template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&
                                 !std::is_arithmetic<TRhs>::value,bool>::type = 0 >
FASTOR_INLINE BinaryAddOp<TLhs, TRhs, DIM0> operator+(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM0> &rhs) {
  return BinaryAddOp<TLhs, TRhs, DIM0>(lhs.self(), rhs.self());
}
template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&
                                 std::is_arithmetic<TRhs>::value,bool>::type = 0 >
FASTOR_INLINE BinaryAddOp<TLhs, TRhs, DIM0> operator+(const AbstractTensor<TLhs,DIM0> &lhs, const TRhs &bb) {
  return BinaryAddOp<TLhs, TRhs, DIM0>(lhs.self(), bb);
}
template<typename TLhs, typename TRhs, size_t DIM0,
         typename std::enable_if<std::is_arithmetic<TLhs>::value &&
                                 !std::is_arithmetic<TRhs>::value,bool>::type = 0 >
FASTOR_INLINE BinaryAddOp<TLhs, TRhs, DIM0> operator+(const TLhs &bb, const AbstractTensor<TRhs,DIM0> &rhs) {
  return BinaryAddOp<TLhs, TRhs, DIM0>(bb,rhs.self());
}


template<typename TLhs, typename TRhs, size_t DIM0, size_t DIM1,
         typename std::enable_if<!std::is_arithmetic<TLhs>::value &&
                                 !std::is_arithmetic<TRhs>::value &&
                                 DIM0!=DIM1,bool>::type = 0 >
FASTOR_INLINE BinaryAddOp<TLhs, TRhs, meta_min<DIM0,DIM1>::value> 
operator+(const AbstractTensor<TLhs,DIM0> &lhs, const AbstractTensor<TRhs,DIM1> &rhs) {
  return BinaryAddOp<TLhs, TRhs, meta_min<DIM0,DIM1>::value>(lhs.self(), rhs.self());
}

}


#endif // BINARY_ADD_OP_H
